from __future__ import annotations

import argparse
import csv
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import championship_cnn as base

_legacy = base._legacy


@dataclass(frozen=True, slots=True)
class GameJob:
    a: Any
    b: Any
    game_index: int
    a_slot: int
    b_slot: int

    @property
    def a_is_black(self) -> bool:
        return self.game_index % 2 == 0

    @property
    def black_slot(self) -> int:
        return self.a_slot if self.a_is_black else self.b_slot

    @property
    def white_slot(self) -> int:
        return self.b_slot if self.a_is_black else self.a_slot


class DirectIndexedEnsemble(base.CNNBatchedPolicyEnsemble):
    """CNN ensemble z dowolnym model_id dla każdej planszy, bez CPU grouping.

    Dla B aktywnych plansz wybieramy B kompletów wag przez index_select na GPU,
    a następnie liczymy jedną konwolucję groups=B. Dzięki temu wszystkie gry,
    nawet będące na różnych numerach ruchu, pozostają jednym GPU batchem.
    """

    @staticmethod
    def _indexed_conv(
        x: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor | None,
        model_ids: torch.Tensor,
        *,
        padding: int,
    ) -> torch.Tensor:
        bsz, in_channels, height, width = x.shape
        selected_w = weights.index_select(0, model_ids)
        out_channels = int(selected_w.shape[1])
        kernel = int(selected_w.shape[-1])
        selected_b = None if biases is None else biases.index_select(0, model_ids)

        grouped_x = x.reshape(1, bsz * in_channels, height, width)
        grouped_w = selected_w.reshape(bsz * out_channels, in_channels, kernel, kernel)
        grouped_b = None if selected_b is None else selected_b.reshape(bsz * out_channels)
        y = F.conv2d(
            grouped_x,
            grouped_w,
            grouped_b,
            stride=1,
            padding=padding,
            groups=bsz,
        )
        return y.reshape(bsz, out_channels, height, width)

    def forward_indexed_direct(
        self,
        x: torch.Tensor,
        model_ids: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Oczekiwano [B,C,H,W], otrzymano {tuple(x.shape)}")
        model_ids = model_ids.to(self.device, dtype=torch.long, non_blocking=True)
        if x.shape[0] != model_ids.numel():
            raise ValueError("Liczba plansz i model_ids musi być identyczna.")
        if x.shape[0] == 0:
            return torch.empty((0, self.action_size), device=self.device)

        ens = self._ensemble
        try:
            for kernel, weight, bias in zip(
                ens.kernels,
                ens.conv_weights,
                ens.conv_biases,
            ):
                x = self._indexed_conv(
                    x,
                    weight,
                    bias,
                    model_ids,
                    padding=kernel // 2,
                )
                x = F.silu(x)

            logits = self._indexed_conv(
                x,
                ens.policy_weight,
                ens.policy_bias,
                model_ids,
                padding=0,
            ).squeeze(1).flatten(1)
            self._guard_memory()
            return logits
        except torch.cuda.OutOfMemoryError as exc:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            raise base.AdaptiveBatchResize("OOM podczas direct indexed CNN") from exc


class GlobalTableScheduler:
    """Jeden VectorConnect6 i jeden GPU forward dla wszystkich aktywnych stołów."""

    def __init__(
        self,
        tables: int,
        ensemble: DirectIndexedEnsemble,
        *,
        temperature: float,
        amp: bool,
        amp_dtype: str,
        seed: int,
    ) -> None:
        self.tables = int(tables)
        self.ensemble = ensemble
        self.device = ensemble.device
        self.temperature = float(temperature)
        self.amp = bool(amp)
        self.amp_dtype = str(amp_dtype)
        self.generator = _legacy._make_generator(self.device, seed)
        self.env = _legacy.VectorConnect6(
            self.tables,
            ensemble.board_size,
            ensemble.win_length,
            device=self.device,
            debug_checks=False,
        )
        self.active = torch.zeros(self.tables, dtype=torch.bool, device=self.device)
        self.winners = torch.zeros(self.tables, dtype=torch.int8, device=self.device)
        self.black_model_ids = torch.zeros(self.tables, dtype=torch.long, device=self.device)
        self.white_model_ids = torch.zeros(self.tables, dtype=torch.long, device=self.device)
        self.active_indices = torch.empty(0, dtype=torch.long, device=self.device)
        self.jobs: list[GameJob | None] = [None] * self.tables
        self._moves_since_sync = 0

    @property
    def active_count(self) -> int:
        return int(self.active_indices.numel())

    def refill(self, slots: list[int], jobs: list[GameJob]) -> None:
        if not slots:
            return
        if len(slots) != len(jobs):
            raise ValueError("slots/jobs mismatch")
        idx = torch.tensor(slots, dtype=torch.long, device=self.device)
        self.env.reset(idx)
        self.active[idx] = True
        self.winners[idx] = 0
        self.black_model_ids[idx] = torch.tensor(
            [job.black_slot for job in jobs], dtype=torch.long, device=self.device
        )
        self.white_model_ids[idx] = torch.tensor(
            [job.white_slot for job in jobs], dtype=torch.long, device=self.device
        )
        for slot, job in zip(slots, jobs):
            self.jobs[slot] = job
        self._rebuild_active_indices()

    def _rebuild_active_indices(self, active_cpu: list[bool] | None = None) -> None:
        if active_cpu is None:
            active_cpu = self.active.detach().cpu().tolist()
        indices = [i for i, value in enumerate(active_cpu) if value]
        self.active_indices = torch.tensor(indices, dtype=torch.long, device=self.device)

    def sync(self) -> tuple[list[tuple[int, GameJob, int]], list[int]]:
        active_cpu = self.active.detach().cpu().tolist()
        winners_cpu = self.winners.detach().cpu().tolist()
        finished: list[tuple[int, GameJob, int]] = []
        free: list[int] = []
        for slot, is_active in enumerate(active_cpu):
            if is_active:
                continue
            free.append(slot)
            job = self.jobs[slot]
            if job is not None:
                finished.append((slot, job, int(winners_cpu[slot])))
                self.jobs[slot] = None
        self._rebuild_active_indices(active_cpu)
        self._moves_since_sync = 0
        return finished, free

    @torch.inference_mode()
    def step(self, sync_interval: int) -> tuple[list[tuple[int, GameJob, int]], list[int]]:
        if self.active_count == 0:
            return self.sync()

        x_all = self.env.network_input()
        legal_all = self.env.legal_mask()
        idx = self.active_indices
        x = x_all.index_select(0, idx)
        legal = legal_all.index_select(0, idx)

        players = self.env.current_player.index_select(0, idx)
        black_ids = self.black_model_ids.index_select(0, idx)
        white_ids = self.white_model_ids.index_select(0, idx)
        actor_ids = torch.where(players.eq(1), black_ids, white_ids)

        with _legacy._autocast_context(self.device, self.amp, self.amp_dtype):
            logits = self.ensemble.forward_indexed_direct(x, actor_ids)
        chosen = _legacy._choose_actions(logits, legal, self.temperature, self.generator)

        full_actions = legal_all.to(torch.int8).argmax(dim=1).to(torch.long)
        full_actions.index_copy_(0, idx, chosen)
        done, winner = base._masked_step(self.env, full_actions, self.active)
        newly_done = self.active & done
        self.winners = torch.where(newly_done, winner, self.winners)
        self.active &= ~done
        self._moves_since_sync += 1

        if self._moves_since_sync >= max(1, int(sync_interval)):
            return self.sync()
        return [], []


def _gpu_timed(device: torch.device, fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return max(1e-9, start.elapsed_time(end) / 1000.0)


def _make_bench_ensemble(cp: Any, tables: int, device: torch.device, pool_models: int) -> DirectIndexedEnsemble:
    model_count = max(2, min(int(pool_models), int(tables)))
    return DirectIndexedEnsemble([cp] * model_count, device)


def _bench_host_host(
    cp: Any,
    tables: int,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    iterations: int,
    pool_models: int,
) -> float:
    ensemble = _make_bench_ensemble(cp, tables, device, pool_models)
    try:
        x = torch.zeros((tables, 3, ensemble.board_size, ensemble.board_size), device=device)
        ids = torch.arange(tables, device=device, dtype=torch.long).remainder(ensemble.num_models)
        with _legacy._autocast_context(device, amp, amp_dtype):
            ensemble.forward_indexed_direct(x, ids)
        torch.cuda.synchronize(device)

        def work() -> None:
            with _legacy._autocast_context(device, amp, amp_dtype):
                for _ in range(iterations):
                    ensemble.forward_indexed_direct(x, ids)

        elapsed = _gpu_timed(device, work)
        return tables * iterations / elapsed
    finally:
        ensemble.release()


def _bench_cross(
    cp: Any,
    tables: int,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    iterations: int,
    pool_models: int,
) -> float:
    ensemble = _make_bench_ensemble(cp, tables, device, pool_models)
    try:
        x = torch.zeros((tables, 3, ensemble.board_size, ensemble.board_size), device=device)
        mixed = torch.arange(tables, device=device, dtype=torch.long).remainder(ensemble.num_models)
        same = torch.zeros(tables, device=device, dtype=torch.long)
        with _legacy._autocast_context(device, amp, amp_dtype):
            ensemble.forward_indexed_direct(x, mixed)
            ensemble.forward_single_model(x, 0)
        torch.cuda.synchronize(device)

        def work() -> None:
            with _legacy._autocast_context(device, amp, amp_dtype):
                for _ in range(iterations):
                    ensemble.forward_indexed_direct(x, mixed)
                    ensemble.forward_single_model(x, 0)

        elapsed = _gpu_timed(device, work)
        return 2 * tables * iterations / elapsed
    finally:
        ensemble.release()


def _bench_full_loop(
    cp: Any,
    tables: int,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    moves: int,
    pool_models: int,
) -> float:
    ensemble = _make_bench_ensemble(cp, tables, device, pool_models)
    try:
        env = _legacy.VectorConnect6(
            tables,
            ensemble.board_size,
            ensemble.win_length,
            device=device,
            debug_checks=False,
        )
        black = torch.arange(tables, device=device, dtype=torch.long).remainder(ensemble.num_models)
        white = torch.flip(black, dims=(0,))
        generator = _legacy._make_generator(device, 987654)

        x = env.network_input()
        with _legacy._autocast_context(device, amp, amp_dtype):
            ensemble.forward_indexed_direct(x, black)
        torch.cuda.synchronize(device)

        def work() -> None:
            for _ in range(moves):
                x = env.network_input()
                legal = env.legal_mask()
                actor_ids = torch.where(env.current_player.eq(1), black, white)
                with _legacy._autocast_context(device, amp, amp_dtype):
                    logits = ensemble.forward_indexed_direct(x, actor_ids)
                actions = _legacy._choose_actions(logits, legal, 0.0, generator)
                env.step(actions)

        elapsed = _gpu_timed(device, work)
        return tables * moves / elapsed
    finally:
        ensemble.release()


def _harmonic_score(hh: float, cross: float, loop: float) -> float:
    weights = ((0.20, hh), (0.25, cross), (0.55, loop))
    return 1.0 / sum(w / max(rate, 1e-9) for w, rate in weights)


def autotune_three_tests(
    config_path: Path,
    cfg: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[int, base.AdaptiveTableController, list[int]]:
    ch = cfg.get("championship", cfg)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(project_root, ch["checkpoint_dir"])
    output_dir = _legacy._resolve_path(project_root, ch["output_dir"])
    refs = _legacy.discover_checkpoints(checkpoint_dir)
    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    if device.type != "cuda":
        fixed = min(len(refs), int(ch.get("tables", 4)))
        controller = base.AdaptiveTableController(
            output_dir / "adaptive_tables.json", gpu_name="CPU", limit_bytes=1,
            min_tables=2, max_tables=max(2, fixed),
        )
        return fixed, controller, [fixed]

    limit_gb = float(adaptive.get("vram_limit_gb", 11.0))
    limit_bytes = int(limit_gb * 2**30)
    gpu_name = torch.cuda.get_device_name(device)
    candidates = base._candidate_tables(adaptive, len(refs))
    controller = base.AdaptiveTableController(
        output_dir / "adaptive_tables.json",
        gpu_name=gpu_name,
        limit_bytes=limit_bytes,
        min_tables=min(candidates),
        max_tables=max(candidates),
    )
    store = base.CNNCheckpointStore(cpu_cache_models=1)
    cp = store.get(refs[0])
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    iterations = max(2, int(adaptive.get("benchmark_iterations", 3)))
    loop_moves = max(4, int(adaptive.get("benchmark_loop_moves", 8)))
    pool_models = max(2, int(adaptive.get("benchmark_model_pool", 128)))

    base._VRAM_LIMIT_BYTES = limit_bytes
    print("\n[ADAPTIVE GPU] autotest 3-trybowy: HOST-HOST / CROSS / FULL-LOOP")
    print(f"GPU: {gpu_name} | limit VRAM: {limit_gb:.2f} GB | benchmark pool: {pool_models} modeli")
    best_tables: int | None = None
    best_score = -1.0

    for tables in candidates:
        if not controller.is_allowed(tables):
            print(f"[AUTOTUNE] {tables:>4} -> pomijam, znane UNSAFE")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            hh = _bench_host_host(cp, tables, device=device, amp=amp, amp_dtype=amp_dtype,
                                  iterations=iterations, pool_models=pool_models)
            cross = _bench_cross(cp, tables, device=device, amp=amp, amp_dtype=amp_dtype,
                                 iterations=iterations, pool_models=pool_models)
            loop = _bench_full_loop(cp, tables, device=device, amp=amp, amp_dtype=amp_dtype,
                                    moves=loop_moves, pool_models=pool_models)
            peak = torch.cuda.max_memory_reserved(device)
            if peak > limit_bytes:
                raise base.AdaptiveBatchResize(
                    f"peak {peak / 2**30:.2f} GB > {limit_gb:.2f} GB"
                )
        except (base.AdaptiveBatchResize, torch.cuda.OutOfMemoryError) as exc:
            controller.mark_unsafe(tables)
            torch.cuda.empty_cache()
            print(f"[AUTOTUNE] {tables:>4} -> UNSAFE: {exc}")
            break

        score = _harmonic_score(hh, cross, loop)
        controller.benchmarks[tables] = {
            "host_host_dps": hh, "cross_dps": cross, "full_loop_dps": loop,
            "score": score, "peak_vram_gb": peak / 2**30,
        }
        print(
            f"[AUTOTUNE] {tables:>4} | HH {hh:>9,.0f} d/s | CROSS {cross:>9,.0f} d/s | "
            f"LOOP {loop:>9,.0f} d/s | score {score:>9,.0f} | VRAM {peak / 2**30:.2f} GB"
        )
        if score > best_score:
            best_score = score
            best_tables = tables

    if best_tables is None:
        best_tables = min(candidates)
    controller.last_selected = best_tables
    controller.save()
    print(f"[AUTOTUNE] wybrano {best_tables} stołów | score {best_score:,.0f}\n")
    return best_tables, controller, candidates


def _update_counter(counter: dict[str, int], winner: int, *, a_is_black: bool) -> None:
    if winner == 0:
        counter["draws"] += 1
        return
    winner_is_a = (winner == 1 and a_is_black) or (winner == -1 and not a_is_black)
    if winner_is_a:
        counter["a_game_wins"] += 1
        counter["a_wins_as_black" if a_is_black else "a_wins_as_white"] += 1
    else:
        counter["b_game_wins"] += 1
        counter["b_wins_as_white" if a_is_black else "b_wins_as_black"] += 1


def _pair_jobs(
    left: list[Any], right: list[Any], *, same_block: bool,
    games_per_pair: int, completed: set[str], slot_of: dict[str, int],
) -> list[GameJob]:
    pairs = combinations(left, 2) if same_block else product(left, right)
    jobs: list[GameJob] = []
    for a, b in pairs:
        if _legacy._pair_id(a, b) in completed:
            continue
        for game_index in range(games_per_pair):
            jobs.append(GameJob(a, b, game_index, slot_of[a.name], slot_of[b.name]))
    return jobs


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_legacy.MATCH_FIELDS)
        if new_file:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in _legacy.MATCH_FIELDS})
        handle.flush()


def _write_progress(
    checkpoints: list[Any], match_rows: list[dict[str, Any]], completed_ids: set[str],
    total_pairs: int, ranking_path: Path, html_path: Path, games_per_pair: int,
    tables: int, temperature: float,
) -> list[dict[str, Any]]:
    ranking = _legacy.build_ranking(checkpoints, match_rows)
    _legacy.write_ranking_csv(ranking_path, ranking)
    _legacy.write_html(
        html_path, ranking, completed_pairs=len(completed_ids), total_pairs=total_pairs,
        checkpoint_count=len(checkpoints), games_per_pair=games_per_pair,
        tables=tables, temperature=temperature, running=len(completed_ids) < total_pairs,
    )
    return ranking


def _build_pool_specs(blocks: list[list[Any]]) -> list[tuple[int, int, list[Any], list[Any], bool]]:
    specs: list[tuple[int, int, list[Any], list[Any], bool]] = []
    for bi, left in enumerate(blocks):
        for bj in range(bi, len(blocks)):
            specs.append((bi, bj, left, blocks[bj], bi == bj))
    return specs


def _unique_refs(left: list[Any], right: list[Any], same: bool) -> list[Any]:
    refs = left if same else left + right
    out: list[Any] = []
    seen: set[str] = set()
    for ref in refs:
        if ref.name not in seen:
            seen.add(ref.name)
            out.append(ref)
    return out


def run_streaming_once(
    config_path: Path,
    cfg: dict[str, Any],
    *,
    tables: int,
    controller: base.AdaptiveTableController,
) -> None:
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(project_root, ch["checkpoint_dir"])
    output_dir = _legacy._resolve_path(project_root, ch["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _legacy.discover_checkpoints(checkpoint_dir)
    games_per_pair = int(ch.get("games_per_pair", 2))
    temperature = float(ch.get("temperature", 0.0))
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    cpu_cache_models = int(ch.get("cpu_cache_models", 16384))
    seed = int(ch.get("seed", 12345))
    sync_interval = max(1, int(adaptive.get("sync_interval_moves", 8)))
    refill_batch = max(1, min(tables, int(adaptive.get("refill_batch", 32))))
    model_block_size = max(2, int(adaptive.get("model_pool_block_size", 64)))
    model_block_size = min(model_block_size, max(2, len(checkpoints)))
    progress_seconds = max(1.0, float(adaptive.get("progress_every_seconds", 5.0)))
    progress_matches = max(1, int(adaptive.get("progress_every_matches", 500)))
    log_seconds = max(0.25, float(adaptive.get("log_every_seconds", 1.0)))

    matches_path = output_dir / "matches.csv"
    ranking_path = output_dir / "ranking.csv"
    html_path = output_dir / "championship.html"
    state_path = output_dir / "state.json"
    _legacy._validate_or_create_state(
        state_path, checkpoints=checkpoints,
        games_per_pair=games_per_pair, temperature=temperature,
    )
    completed_ids, match_rows = _legacy._load_completed_matches(matches_path)
    total_pairs = len(checkpoints) * (len(checkpoints) - 1) // 2
    ranking = _write_progress(
        checkpoints, match_rows, completed_ids, total_pairs, ranking_path, html_path,
        games_per_pair, tables, temperature,
    )

    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    base._VRAM_LIMIT_BYTES = controller.limit_bytes if device.type == "cuda" else 0
    store = base.CNNCheckpointStore(cpu_cache_models=cpu_cache_models)
    blocks = [checkpoints[i : i + model_block_size] for i in range(0, len(checkpoints), model_block_size)]
    specs = _build_pool_specs(blocks)

    print("=" * 78)
    print("KING OF CONNECT6 — GLOBAL PERSISTENT GPU STREAM")
    print("=" * 78)
    print(f"Stoły: {tables} | refill batch: {refill_batch} | model block: {model_block_size}")
    print(f"Jeden VectorConnect6 + jeden CNN forward dla wszystkich aktywnych stołów")
    print(f"Synchronizacja CPU scheduler-a: co {sync_interval} ruchów")
    print(f"Pary: {len(completed_ids):,}/{total_pairs:,} ukończone")

    pair_counters: dict[str, dict[str, int]] = {}
    pair_refs: dict[str, tuple[Any, Any]] = {}
    started_at = time.perf_counter()
    finished_games_total = 0
    last_progress = time.perf_counter()
    last_progress_matches = len(completed_ids)
    last_log = 0.0

    def load_unique(refs: list[Any]) -> list[Any]:
        return [store.get(ref) for ref in refs]

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="checkpoint-prefetch") as executor:
        prefetched: Future[list[Any]] | None = None
        prefetched_index = -1

        for spec_index, (bi, bj, left, right, same) in enumerate(specs):
            unique = _unique_refs(left, right, same)
            slot_of = {ref.name: i for i, ref in enumerate(unique)}
            jobs = _pair_jobs(
                left, right, same_block=same, games_per_pair=games_per_pair,
                completed=completed_ids, slot_of=slot_of,
            )
            if not jobs:
                continue

            if prefetched is not None and prefetched_index == spec_index:
                lean = prefetched.result()
                prefetched = None
            else:
                lean = load_unique(unique)

            # Prefetch następnej faktycznie istniejącej puli do RAM podczas pracy GPU.
            for ni in range(spec_index + 1, len(specs)):
                nbi, nbj, nleft, nright, nsame = specs[ni]
                nunique = _unique_refs(nleft, nright, nsame)
                nslot = {ref.name: i for i, ref in enumerate(nunique)}
                njobs = _pair_jobs(
                    nleft, nright, same_block=nsame, games_per_pair=games_per_pair,
                    completed=completed_ids, slot_of=nslot,
                )
                if njobs:
                    prefetched = executor.submit(load_unique, nunique)
                    prefetched_index = ni
                    break

            print(
                f"\n[MODEL POOL] bloki {bi + 1}/{bj + 1} | modele={len(unique)} | "
                f"gry={len(jobs)}"
            )
            ensemble = DirectIndexedEnsemble(lean, device)
            scheduler = GlobalTableScheduler(
                tables, ensemble, temperature=temperature, amp=amp,
                amp_dtype=amp_dtype, seed=seed + spec_index,
            )
            pending = 0

            initial = min(tables, len(jobs))
            scheduler.refill(list(range(initial)), jobs[:initial])
            pending = initial

            try:
                while scheduler.active_count > 0 or pending < len(jobs):
                    finished, free = scheduler.step(sync_interval)
                    completed_rows: list[dict[str, Any]] = []

                    for _slot, job, winner in finished:
                        finished_games_total += 1
                        pid = _legacy._pair_id(job.a, job.b)
                        counter = pair_counters.setdefault(pid, _legacy._empty_counter())
                        pair_refs[pid] = (job.a, job.b)
                        _update_counter(counter, winner, a_is_black=job.a_is_black)
                        games_done = counter["a_game_wins"] + counter["b_game_wins"] + counter["draws"]
                        if games_done == games_per_pair:
                            row = _legacy._finalize_pair_rows([pair_refs[pid]], [counter], games_per_pair)[0]
                            row["elapsed_seconds"] = 0.0
                            completed_rows.append(row)
                            completed_ids.add(pid)
                            match_rows.append(row)
                            del pair_counters[pid]
                            del pair_refs[pid]

                    _append_rows(matches_path, completed_rows)

                    remaining = len(jobs) - pending
                    if remaining > 0 and (len(free) >= refill_batch or scheduler.active_count == 0):
                        count = min(len(free), remaining)
                        if count > 0:
                            new_jobs = jobs[pending : pending + count]
                            scheduler.refill(free[:count], new_jobs)
                            pending += count

                    now = time.perf_counter()
                    if (
                        now - last_progress >= progress_seconds
                        or len(completed_ids) - last_progress_matches >= progress_matches
                    ):
                        ranking = _write_progress(
                            checkpoints, match_rows, completed_ids, total_pairs,
                            ranking_path, html_path, games_per_pair, tables, temperature,
                        )
                        last_progress = now
                        last_progress_matches = len(completed_ids)

                    if finished and now - last_log >= log_seconds:
                        elapsed = max(1e-9, now - started_at)
                        print(
                            f"[STREAM] aktywne {scheduler.active_count:>4}/{tables} | "
                            f"free={tables - scheduler.active_count:>3} | "
                            f"gry/s={finished_games_total / elapsed:.2f} | "
                            f"pary={len(completed_ids):,}/{total_pairs:,} | "
                            f"{_legacy._vram_label(device)}"
                        )
                        last_log = now
            finally:
                ensemble.release()
                del ensemble

    ranking = _write_progress(
        checkpoints, match_rows, completed_ids, total_pairs, ranking_path, html_path,
        games_per_pair, tables, temperature,
    )
    print("\n" + "=" * 78)
    print("MISTRZOSTWA ZAKOŃCZONE")
    if ranking:
        print(f"👑 KING OF CONNECT6: {ranking[0]['model']} | {ranking[0]['points']} pkt")
    print(f"Ranking HTML: {html_path}")
    print("=" * 78)


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = base._ORIGINAL_READ_YAML(config_path)
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}

    selected, controller, candidates = autotune_three_tests(config_path, cfg, adaptive)
    while True:
        try:
            controller.last_selected = selected
            controller.save()
            run_streaming_once(config_path, cfg, tables=selected, controller=controller)
            return
        except base.AdaptiveBatchResize as exc:
            controller.mark_unsafe(selected)
            new_tables = controller.next_lower(selected, candidates)
            if new_tables >= selected or selected <= controller.min_tables:
                raise
            print(
                f"\n[ADAPTIVE GPU] {selected} stołów przekroczyło budżet: {exc}. "
                f"Zapamiętuję i wznawiam z {new_tables}.\n"
            )
            selected = new_tables
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="King of Connect6 — global persistent streaming CNN championship"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
