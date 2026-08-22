from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import championship as _legacy
from .checkpoint import load_checkpoint
from .history import HistoricalCheckpoint, HistoricalPolicyEnsemble


_ORIGINAL_READ_YAML = _legacy._read_yaml
_ORIGINAL_RUN_CHAMPIONSHIP = _legacy.run_championship
_SYNC_INTERVAL = 8
_VRAM_LIMIT_BYTES = 0
_VRAM_CHECK_EVERY = 32


def _normalized_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out.pop("compile", None)
    out.pop("compile_mode", None)
    return out


class AdaptiveBatchResize(RuntimeError):
    """Sygnał: aktualny rozmiar host blocku jest zbyt duży dla GPU."""


@dataclass(slots=True)
class PreparedLayout:
    batch_size: int
    max_tables: int
    gather_indices: torch.Tensor
    valid_sources: torch.Tensor
    target_indices: torch.Tensor
    identity: bool = False


class CNNCheckpointStore:
    """Lekki cache checkpointów CNN używany przez championship."""

    def __init__(self, cpu_cache_models: int = 16) -> None:
        self.capacity = max(0, int(cpu_cache_models))
        self.cache: OrderedDict[str, _legacy.LeanCheckpoint] = OrderedDict()

    def get(self, ref: _legacy.CheckpointRef) -> _legacy.LeanCheckpoint:
        key = str(ref.path)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached

        payload = load_checkpoint(ref.path, map_location="cpu")
        model_cfg = dict(payload["model_config"])
        game_cfg = dict(payload["game_config"])

        if int(model_cfg.get("architecture_version", 0)) != 4:
            raise RuntimeError(
                f"Checkpoint {ref.name} ma architecture_version="
                f"{model_cfg.get('architecture_version')}; championship CNN wymaga wersji 4."
            )

        lean = _legacy.LeanCheckpoint(
            ref=ref,
            model_state=payload["model_state"],
            model_config=model_cfg,
            game_config=game_cfg,
        )
        del payload

        if self.capacity > 0:
            self.cache[key] = lean
            self.cache.move_to_end(key)
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)

        return lean


class CNNBatchedPolicyEnsemble:
    """Wiele różnych CNN liczone równolegle jednym grouped-conv na warstwę."""

    def __init__(
        self,
        checkpoints: list[_legacy.LeanCheckpoint],
        device: torch.device,
    ) -> None:
        if not checkpoints:
            raise ValueError("CNNBatchedPolicyEnsemble wymaga co najmniej 1 checkpointu.")

        self.device = device
        self.num_models = len(checkpoints)
        self.refs = [cp.ref for cp in checkpoints]
        self._forward_calls = 0

        first = checkpoints[0]
        self.model_config = _normalized_model_cfg(first.model_config)
        self.board_size = int(first.game_config["board_size"])
        self.win_length = int(first.game_config.get("win_length", 6))

        if int(self.model_config.get("architecture_version", 0)) != 4:
            raise RuntimeError("Championship CNN obsługuje architecture_version=4.")

        for cp in checkpoints[1:]:
            if int(cp.game_config["board_size"]) != self.board_size or int(
                cp.game_config.get("win_length", 6)
            ) != self.win_length:
                raise ValueError("Checkpointy w jednym batchu używają różnych zasad gry.")
            if _normalized_model_cfg(cp.model_config) != self.model_config:
                raise ValueError(
                    "Checkpointy w jednym równoległym batchu mają różne architektury CNN. "
                    "Jeden championship powinien zawierać modele tej samej architektury."
                )

        historical = [
            HistoricalCheckpoint(
                path=cp.ref.path,
                update=cp.ref.update,
                model_state=cp.model_state,
                model_config=cp.model_config,
                game_config=cp.game_config,
            )
            for cp in checkpoints
        ]
        try:
            self._ensemble = HistoricalPolicyEnsemble(
                historical,
                device=device,
                dtype=torch.float32,
            )
        except torch.cuda.OutOfMemoryError as exc:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise AdaptiveBatchResize(
                f"OOM podczas ładowania {self.num_models} modeli CNN"
            ) from exc

        self.action_size = self.board_size * self.board_size
        self._guard_memory(force=True)

    def _guard_memory(self, *, force: bool = False) -> None:
        global _VRAM_LIMIT_BYTES
        if self.device.type != "cuda" or _VRAM_LIMIT_BYTES <= 0:
            return
        self._forward_calls += 1
        if not force and self._forward_calls % max(1, _VRAM_CHECK_EVERY) != 0:
            return

        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        if max(allocated, reserved) <= _VRAM_LIMIT_BYTES:
            return

        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        if max(allocated, reserved) > _VRAM_LIMIT_BYTES:
            raise AdaptiveBatchResize(
                "Limit VRAM przekroczony po czyszczeniu cache: "
                f"alloc={allocated / 2**30:.2f} GB, reserved={reserved / 2**30:.2f} GB"
            )

    def prepare_layout(self, model_indices: torch.Tensor) -> PreparedLayout:
        """Buduje mapowanie stół->model RAZ, poza hot path ruchów."""
        ids = [int(v) for v in model_indices.detach().cpu().tolist()]
        if not ids:
            raise ValueError("Nie można przygotować pustego layoutu.")
        if min(ids) < 0 or max(ids) >= self.num_models:
            raise IndexError("model_indices zawiera indeks spoza ensemble.")

        identity = len(ids) == self.num_models and all(i == value for i, value in enumerate(ids))
        if identity:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return PreparedLayout(
                batch_size=len(ids),
                max_tables=1,
                gather_indices=empty,
                valid_sources=empty,
                target_indices=empty,
                identity=True,
            )

        groups: list[list[int]] = [[] for _ in range(self.num_models)]
        for table_idx, model_idx in enumerate(ids):
            groups[model_idx].append(table_idx)
        max_tables = max(1, max(len(group) for group in groups))

        gather = torch.zeros(
            (self.num_models, max_tables), dtype=torch.long, device=self.device
        )
        valid_sources: list[int] = []
        targets: list[int] = []
        for model_idx, group in enumerate(groups):
            if not group:
                continue
            gather[model_idx, : len(group)] = torch.tensor(
                group, dtype=torch.long, device=self.device
            )
            base = model_idx * max_tables
            valid_sources.extend(base + j for j in range(len(group)))
            targets.extend(group)

        return PreparedLayout(
            batch_size=len(ids),
            max_tables=max_tables,
            gather_indices=gather,
            valid_sources=torch.tensor(valid_sources, dtype=torch.long, device=self.device),
            target_indices=torch.tensor(targets, dtype=torch.long, device=self.device),
            identity=False,
        )

    def forward_prepared(self, x: torch.Tensor, layout: PreparedLayout) -> torch.Tensor:
        if x.ndim != 4 or x.shape[0] != layout.batch_size:
            raise ValueError(
                f"Niepoprawny batch {tuple(x.shape)} dla layoutu {layout.batch_size}."
            )
        try:
            if layout.identity:
                logits = self._ensemble.forward_grouped(x.unsqueeze(1)).squeeze(1)
            else:
                gathered = x.index_select(0, layout.gather_indices.reshape(-1))
                padded = gathered.reshape(
                    self.num_models,
                    layout.max_tables,
                    x.shape[1],
                    x.shape[2],
                    x.shape[3],
                )
                grouped_logits = self._ensemble.forward_grouped(padded)
                flat_logits = grouped_logits.reshape(-1, self.action_size)
                selected = flat_logits.index_select(0, layout.valid_sources)
                logits = torch.empty(
                    (layout.batch_size, self.action_size),
                    device=self.device,
                    dtype=selected.dtype,
                )
                logits.index_copy_(0, layout.target_indices, selected)
            self._guard_memory()
            return logits
        except torch.cuda.OutOfMemoryError as exc:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise AdaptiveBatchResize("OOM podczas grouped-conv championship") from exc

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(self.num_models, device=self.device, dtype=torch.long)
        return self.forward_prepared(x, self.prepare_layout(indices))

    def forward_indexed(
        self,
        x: torch.Tensor,
        model_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Ścieżka awaryjna. Szybkie game-loopy używają forward_prepared().
        return self.forward_prepared(x, self.prepare_layout(model_indices))

    def forward_single_model(
        self,
        x: torch.Tensor,
        model_index: int = 0,
    ) -> torch.Tensor:
        idx = int(model_index)
        if idx < 0 or idx >= self.num_models:
            raise IndexError(f"model_index poza zakresem: {idx}")

        ensemble = self._ensemble
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)

        try:
            for kernel, weight, bias in zip(
                ensemble.kernels,
                ensemble.conv_weights,
                ensemble.conv_biases,
            ):
                w = weight[idx]
                b = None if bias is None else bias[idx]
                x = F.conv2d(x, w, b, stride=1, padding=kernel // 2)
                x = F.silu(x)

            policy_weight = ensemble.policy_weight[idx]
            policy_bias = None if ensemble.policy_bias is None else ensemble.policy_bias[idx]
            logits = F.conv2d(x, policy_weight, policy_bias).squeeze(1).flatten(1)
            self._guard_memory()
            return logits
        except torch.cuda.OutOfMemoryError as exc:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise AdaptiveBatchResize("OOM podczas inference challengera") from exc

    def release(self) -> None:
        ensemble = getattr(self, "_ensemble", None)
        if ensemble is None:
            return
        ensemble.conv_weights.clear()
        ensemble.conv_biases.clear()
        del ensemble.policy_weight
        if hasattr(ensemble, "policy_bias"):
            del ensemble.policy_bias
        del self._ensemble


def _black_to_move(move_index: int) -> bool:
    if move_index == 0:
        return True
    return ((move_index - 1) // 2) % 2 == 1


def _masked_step(
    env: Any,
    actions: torch.Tensor,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VectorConnect6.step bez nonzero/reset; nieaktywne stoły pozostają zamrożone."""
    actions = actions.to(device=env.device, dtype=torch.long, non_blocking=True)
    rows = torch.div(actions, env.board_size, rounding_mode="floor")
    cols = actions.remainder(env.board_size)
    batch = env._batch
    actor = env.current_player.clone()

    old = env.boards[batch, rows, cols]
    env.boards[batch, rows, cols] = torch.where(active, actor, old)
    active_i16 = active.to(torch.int16)
    env.empty_count.sub_(active_i16)
    env.move_count.add_(active_i16)

    won = env._check_win_local(rows, cols, actor) & active
    draw = env.empty_count.eq(0) & ~won & active
    done = won | draw
    winner = torch.where(won, actor, torch.zeros_like(actor))

    remaining = env.stones_left - 1
    live = active & ~done
    stay = live & remaining.gt(0)
    switch = live & ~remaining.gt(0)
    env.stones_left[stay] = remaining[stay]
    env.current_player[switch] = -env.current_player[switch]
    env.stones_left[switch] = 2
    return done, winner


def _record_game_winners(
    counters: list[dict[str, int]],
    winners: torch.Tensor,
    *,
    a_is_black: bool,
) -> None:
    # Jedna synchronizacja GPU->CPU na całą grę, zamiast przy każdym zakończeniu stołu.
    values = winners.detach().cpu().tolist()
    for env_idx, winner in enumerate(values):
        c = counters[env_idx]
        if winner == 0:
            c["draws"] += 1
            continue
        winner_is_a = (winner == 1 and a_is_black) or (winner == -1 and not a_is_black)
        if winner_is_a:
            c["a_game_wins"] += 1
            c["a_wins_as_black" if a_is_black else "a_wins_as_white"] += 1
        else:
            c["b_game_wins"] += 1
            c["b_wins_as_white" if a_is_black else "b_wins_as_black"] += 1


@torch.inference_mode()
def play_internal_parallel_fast(
    pairings: list[tuple[Any, Any, int, int]],
    hosts: CNNBatchedPolicyEnsemble,
    *,
    games_per_pair: int,
    temperature: float,
    amp: bool,
    amp_dtype: str,
    seed: int,
) -> list[dict[str, Any]]:
    if not pairings:
        return []

    device = hosts.device
    env = _legacy.VectorConnect6(
        len(pairings), hosts.board_size, hosts.win_length, device=device, debug_checks=False
    )
    pair_refs = [(a, b) for a, b, _, _ in pairings]
    a_slots = torch.tensor([sa for _, _, sa, _ in pairings], device=device, dtype=torch.long)
    b_slots = torch.tensor([sb for _, _, _, sb in pairings], device=device, dtype=torch.long)
    a_layout = hosts.prepare_layout(a_slots)
    b_layout = hosts.prepare_layout(b_slots)
    counters = [_legacy._empty_counter() for _ in pairings]
    generator = _legacy._make_generator(device, seed)

    for game_index in range(games_per_pair):
        a_is_black = game_index % 2 == 0
        black_layout = a_layout if a_is_black else b_layout
        white_layout = b_layout if a_is_black else a_layout
        env.reset()
        active = torch.ones(len(pairings), dtype=torch.bool, device=device)
        game_winners = torch.zeros(len(pairings), dtype=torch.int8, device=device)

        for move_index in range(env.action_size):
            x = env.network_input()
            legal = env.legal_mask()
            layout = black_layout if _black_to_move(move_index) else white_layout
            with _legacy._autocast_context(device, amp, amp_dtype):
                logits = hosts.forward_prepared(x, layout)
            actions = _legacy._choose_actions(logits, legal, temperature, generator)

            done, winner = _masked_step(env, actions, active)
            newly_done = active & done
            game_winners = torch.where(newly_done, winner, game_winners)
            active &= ~done

            if (move_index + 1) % max(1, _SYNC_INTERVAL) == 0:
                if not bool(active.any()):
                    break

        _record_game_winners(counters, game_winners, a_is_black=a_is_black)

    return _legacy._finalize_pair_rows(pair_refs, counters, games_per_pair)


@torch.inference_mode()
def play_cross_parallel_fast(
    pairings: list[tuple[Any, Any, int]],
    hosts: CNNBatchedPolicyEnsemble,
    challenger: CNNBatchedPolicyEnsemble,
    *,
    games_per_pair: int,
    temperature: float,
    amp: bool,
    amp_dtype: str,
    seed: int,
) -> list[dict[str, Any]]:
    if not pairings:
        return []
    if challenger.num_models != 1:
        raise ValueError("Cross batch oczekuje dokładnie jednego challengera.")
    if challenger.board_size != hosts.board_size or challenger.win_length != hosts.win_length:
        raise ValueError("Challenger ma inne zasady gry niż hosty.")
    if challenger.model_config != hosts.model_config:
        raise ValueError("Challenger ma inną architekturę CNN niż hosty.")

    device = hosts.device
    env = _legacy.VectorConnect6(
        len(pairings), hosts.board_size, hosts.win_length, device=device, debug_checks=False
    )
    pair_refs = [(a, b) for a, b, _ in pairings]
    host_slots = torch.tensor([slot for _, _, slot in pairings], device=device, dtype=torch.long)
    host_layout = hosts.prepare_layout(host_slots)
    counters = [_legacy._empty_counter() for _ in pairings]
    generator = _legacy._make_generator(device, seed)

    for game_index in range(games_per_pair):
        a_is_black = game_index % 2 == 0
        env.reset()
        active = torch.ones(len(pairings), dtype=torch.bool, device=device)
        game_winners = torch.zeros(len(pairings), dtype=torch.int8, device=device)

        for move_index in range(env.action_size):
            x = env.network_input()
            legal = env.legal_mask()
            black_turn = _black_to_move(move_index)
            host_turn = black_turn == a_is_black

            with _legacy._autocast_context(device, amp, amp_dtype):
                if host_turn:
                    logits = hosts.forward_prepared(x, host_layout)
                else:
                    logits = challenger.forward_single_model(x, 0)
            actions = _legacy._choose_actions(logits, legal, temperature, generator)

            done, winner = _masked_step(env, actions, active)
            newly_done = active & done
            game_winners = torch.where(newly_done, winner, game_winners)
            active &= ~done

            if (move_index + 1) % max(1, _SYNC_INTERVAL) == 0:
                if not bool(active.any()):
                    break

        _record_game_winners(counters, game_winners, a_is_black=a_is_black)

    return _legacy._finalize_pair_rows(pair_refs, counters, games_per_pair)


class AdaptiveTableController:
    def __init__(
        self,
        state_path: Path,
        *,
        gpu_name: str,
        limit_bytes: int,
        min_tables: int,
        max_tables: int,
    ) -> None:
        self.state_path = state_path
        self.gpu_name = gpu_name
        self.limit_bytes = int(limit_bytes)
        self.min_tables = max(2, int(min_tables))
        self.max_tables = max(self.min_tables, int(max_tables))
        self.unsafe_from: int | None = None
        self.benchmarks: dict[int, dict[str, float]] = {}
        self.last_selected: int | None = None
        self._load()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if data.get("gpu_name") != self.gpu_name:
            return
        old_limit = int(data.get("limit_bytes", 0))
        if old_limit != self.limit_bytes:
            return
        unsafe = data.get("unsafe_from")
        self.unsafe_from = int(unsafe) if unsafe is not None else None
        self.last_selected = (
            int(data["last_selected"]) if data.get("last_selected") is not None else None
        )

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "gpu_name": self.gpu_name,
            "limit_bytes": self.limit_bytes,
            "unsafe_from": self.unsafe_from,
            "last_selected": self.last_selected,
            "benchmarks": {str(k): v for k, v in sorted(self.benchmarks.items())},
        }
        self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def is_allowed(self, tables: int) -> bool:
        tables = int(tables)
        return tables >= self.min_tables and (
            self.unsafe_from is None or tables < self.unsafe_from
        )

    def mark_unsafe(self, tables: int) -> None:
        tables = max(self.min_tables, int(tables))
        self.unsafe_from = tables if self.unsafe_from is None else min(self.unsafe_from, tables)
        if self.last_selected is not None and self.last_selected >= self.unsafe_from:
            self.last_selected = None
        self.save()

    def next_lower(self, current: int, candidates: list[int]) -> int:
        lower = [v for v in candidates if v < current and self.is_allowed(v)]
        if lower:
            return max(lower)
        fallback = max(self.min_tables, current // 2)
        if fallback >= current:
            fallback = current - 1
        return max(self.min_tables, fallback)


def _candidate_tables(adaptive: dict[str, Any], checkpoint_count: int) -> list[int]:
    minimum = max(2, int(adaptive.get("min_tables", 16)))
    maximum = min(checkpoint_count, int(adaptive.get("max_tables", 2048)))
    configured = adaptive.get("candidates", [32, 64, 128, 256, 512, 1024, 2048])
    values = {minimum, maximum}
    values.update(int(v) for v in configured)
    return sorted(v for v in values if minimum <= v <= maximum)


def _benchmark_candidate(
    checkpoint: _legacy.LeanCheckpoint,
    tables: int,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    warmup: int,
    iterations: int,
    limit_bytes: int,
) -> dict[str, float]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    ensemble: CNNBatchedPolicyEnsemble | None = None
    try:
        ensemble = CNNBatchedPolicyEnsemble([checkpoint] * tables, device)
        ids = torch.arange(tables, device=device, dtype=torch.long)
        layout = ensemble.prepare_layout(ids)
        x = torch.zeros(
            (tables, 3, ensemble.board_size, ensemble.board_size),
            device=device,
            dtype=torch.float32,
        )

        with _legacy._autocast_context(device, amp, amp_dtype):
            for _ in range(max(1, warmup)):
                ensemble.forward_prepared(x, layout)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with _legacy._autocast_context(device, amp, amp_dtype):
            for _ in range(max(1, iterations)):
                ensemble.forward_prepared(x, layout)
        end.record()
        end.synchronize()

        elapsed_s = max(1e-9, start.elapsed_time(end) / 1000.0)
        peak = torch.cuda.max_memory_reserved(device)
        if peak > limit_bytes:
            raise AdaptiveBatchResize(
                f"benchmark {tables}: peak {peak / 2**30:.2f} GB > limit"
            )
        decisions = tables * max(1, iterations)
        return {
            "decisions_per_second": decisions / elapsed_s,
            "ms_per_forward": elapsed_s * 1000.0 / max(1, iterations),
            "peak_vram_gb": peak / 2**30,
        }
    finally:
        if ensemble is not None:
            ensemble.release()
        del ensemble
        torch.cuda.empty_cache()


def _autotune_tables(
    config_path: Path,
    cfg: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[int, AdaptiveTableController, list[int]]:
    ch = cfg.get("championship", cfg)
    project_root = (
        config_path.parent.parent
        if config_path.parent.name == "configs"
        else config_path.parent
    )
    checkpoint_dir = _legacy._resolve_path(
        project_root, ch.get("checkpoint_dir", "runs/connect6_cnn_01/checkpoints")
    )
    output_dir = _legacy._resolve_path(
        project_root, ch.get("output_dir", "runs/connect6_cnn_01/championship")
    )
    refs = _legacy.discover_checkpoints(checkpoint_dir)
    if len(refs) < 2:
        return int(ch.get("tables", 2)), AdaptiveTableController(
            output_dir / "adaptive_tables.json",
            gpu_name="unknown",
            limit_bytes=1,
            min_tables=2,
            max_tables=2,
        ), [2]

    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    if device.type != "cuda":
        fixed = min(len(refs), int(ch.get("tables", 4)))
        controller = AdaptiveTableController(
            output_dir / "adaptive_tables.json",
            gpu_name="CPU",
            limit_bytes=1,
            min_tables=2,
            max_tables=max(2, fixed),
        )
        return fixed, controller, [fixed]

    limit_gb = float(adaptive.get("vram_limit_gb", 11.0))
    limit_bytes = int(limit_gb * 2**30)
    gpu_name = torch.cuda.get_device_name(device)
    candidates = _candidate_tables(adaptive, len(refs))
    controller = AdaptiveTableController(
        output_dir / "adaptive_tables.json",
        gpu_name=gpu_name,
        limit_bytes=limit_bytes,
        min_tables=min(candidates),
        max_tables=max(candidates),
    )

    store = CNNCheckpointStore(cpu_cache_models=1)
    checkpoint = store.get(refs[0])
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    warmup = int(adaptive.get("benchmark_warmup", 1))
    iterations = int(adaptive.get("benchmark_iterations", 3))

    print("\n[ADAPTIVE GPU] krótki autotest liczby stołów")
    print(f"GPU: {gpu_name} | limit VRAM: {limit_gb:.2f} GB")
    best_tables: int | None = None
    best_rate = -1.0

    for tables in candidates:
        if not controller.is_allowed(tables):
            print(f"[AUTOTUNE] {tables:>4} stołów -> pomijam (wcześniej oznaczone unsafe)")
            continue
        try:
            result = _benchmark_candidate(
                checkpoint,
                tables,
                device=device,
                amp=amp,
                amp_dtype=amp_dtype,
                warmup=warmup,
                iterations=iterations,
                limit_bytes=limit_bytes,
            )
        except (AdaptiveBatchResize, torch.cuda.OutOfMemoryError) as exc:
            controller.mark_unsafe(tables)
            print(f"[AUTOTUNE] {tables:>4} stołów -> UNSAFE: {exc}")
            break

        controller.benchmarks[tables] = result
        rate = result["decisions_per_second"]
        print(
            f"[AUTOTUNE] {tables:>4} stołów | {rate:,.0f} decyzji/s | "
            f"{result['ms_per_forward']:.2f} ms/forward | "
            f"peak {result['peak_vram_gb']:.2f} GB"
        )
        if rate > best_rate:
            best_rate = rate
            best_tables = tables

    if best_tables is None:
        best_tables = min(candidates)
    controller.last_selected = best_tables
    controller.save()
    print(f"[AUTOTUNE] wybrano {best_tables} stołów ({best_rate:,.0f} decyzji/s)\n")
    return best_tables, controller, candidates


def _run_legacy_with_overrides(
    config_path: Path,
    *,
    tables: int,
) -> None:
    original_reader = _legacy._read_yaml

    def patched_reader(path: Path) -> dict[str, Any]:
        data = copy.deepcopy(_ORIGINAL_READ_YAML(path))
        section = data.get("championship", data)
        section["tables"] = int(tables)
        # Cache CUDA zostawiamy allocatorowi; kontroler czyści go tylko przy presji VRAM.
        section["empty_cache_after_unload"] = False
        return data

    _legacy._read_yaml = patched_reader
    try:
        _ORIGINAL_RUN_CHAMPIONSHIP(config_path)
    finally:
        _legacy._read_yaml = original_reader


def run_championship_adaptive(config_path: str | Path) -> None:
    global _SYNC_INTERVAL, _VRAM_LIMIT_BYTES, _VRAM_CHECK_EVERY

    config_path = Path(config_path).resolve()
    cfg = _ORIGINAL_READ_YAML(config_path)
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {})
    if not isinstance(adaptive, dict):
        adaptive = {}

    _SYNC_INTERVAL = max(1, int(adaptive.get("sync_interval_moves", 8)))
    _VRAM_CHECK_EVERY = max(1, int(adaptive.get("vram_check_every_forwards", 32)))
    enabled = bool(adaptive.get("enabled", True))

    if not enabled:
        _VRAM_LIMIT_BYTES = 0
        _ORIGINAL_RUN_CHAMPIONSHIP(config_path)
        return

    selected, controller, candidates = _autotune_tables(config_path, cfg, adaptive)
    _VRAM_LIMIT_BYTES = controller.limit_bytes if controller.gpu_name != "CPU" else 0

    while True:
        try:
            controller.last_selected = selected
            controller.save()
            _run_legacy_with_overrides(config_path, tables=selected)
            return
        except AdaptiveBatchResize as exc:
            controller.mark_unsafe(selected)
            new_tables = controller.next_lower(selected, candidates)
            if new_tables >= selected or selected <= controller.min_tables:
                raise RuntimeError(
                    f"Nie mogę zejść niżej z liczbą stołów po błędzie VRAM: {exc}"
                ) from exc
            print(
                f"\n[ADAPTIVE GPU] {selected} stołów okazało się zbyt ciężkie w realnym "
                f"turnieju ({exc}). Zapamiętuję ten próg i wznawiam od "
                f"{new_tables} stołów. Rozegrane mecze nie będą powtarzane.\n"
            )
            selected = new_tables
            torch.cuda.empty_cache()


# Podmieniamy tylko backend i hot path; ranking/CSV/HTML/resume zostają sprawdzone.
_legacy.CheckpointStore = CNNCheckpointStore
_legacy.BatchedPolicyEnsemble = CNNBatchedPolicyEnsemble
_legacy.play_internal_parallel = play_internal_parallel_fast
_legacy.play_cross_parallel = play_cross_parallel_fast
_legacy.run_championship = run_championship_adaptive


def main() -> None:
    parser = argparse.ArgumentParser(
        description="King of Connect6 — adaptacyjne równoległe mistrzostwa CNN"
    )
    parser.add_argument(
        "--config",
        default="configs/championship.yaml",
        help="Ścieżka do konfiguracji mistrzostw YAML",
    )
    args = parser.parse_args()
    run_championship_adaptive(args.config)


if __name__ == "__main__":
    main()
