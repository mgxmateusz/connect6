from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import re
import time
from collections import OrderedDict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
import yaml

from .checkpoint import load_checkpoint
from .model import mask_logits
from .vector_env import VectorConnect6

_UPDATE_RE = re.compile(r"model_update_(\d+)\.pt$")

MATCH_FIELDS = [
    "match_id",
    "model_a",
    "update_a",
    "model_b",
    "update_b",
    "games",
    "a_game_wins",
    "draws",
    "b_game_wins",
    "a_wins_as_black",
    "a_wins_as_white",
    "b_wins_as_black",
    "b_wins_as_white",
    "a_game_score",
    "b_game_score",
    "a_points",
    "b_points",
    "match_result",
    "elapsed_seconds",
]


@dataclass(frozen=True, slots=True)
class CheckpointRef:
    path: Path
    name: str
    update: int


@dataclass(slots=True)
class LeanCheckpoint:
    ref: CheckpointRef
    model_state: dict[str, torch.Tensor]
    model_config: dict[str, Any]
    game_config: dict[str, Any]


@dataclass(slots=True)
class BatchedLayer:
    weight: torch.Tensor
    bias: torch.Tensor | None
    norm: str
    activation: str
    norm_weight: torch.Tensor | None = None
    norm_bias: torch.Tensor | None = None
    running_mean: torch.Tensor | None = None
    running_var: torch.Tensor | None = None
    eps: float = 1e-5


# =============================================================================
# CONFIG / CHECKPOINT DISCOVERY
# =============================================================================

def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("championship.yaml musi zawierać mapę YAML.")
    return cfg


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def _checkpoint_ref(path: Path) -> CheckpointRef:
    match = _UPDATE_RE.search(path.name)
    if match is None:
        raise ValueError(f"Niepoprawna nazwa checkpointu: {path.name}")
    return CheckpointRef(path=path.resolve(), name=path.name, update=int(match.group(1)))


def discover_checkpoints(directory: Path) -> list[CheckpointRef]:
    refs = [
        _checkpoint_ref(path)
        for path in directory.glob("model_update_*.pt")
        if path.is_file()
    ]
    refs.sort(key=lambda x: (x.update, x.name))
    return refs


def _pair_id(a: CheckpointRef, b: CheckpointRef) -> str:
    left, right = sorted((a.name, b.name))
    return f"{left}__VS__{right}"


def _chunks(items: list[Any], size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# =============================================================================
# LIGHT CPU CACHE
# =============================================================================

class CheckpointStore:
    """Wczytuje checkpoint raz i trzyma tylko dane potrzebne do inference.

    Pełny checkpoint zawiera także optimizer. Po odczycie optimizer jest wyrzucany.
    Opcjonalny LRU cache ogranicza ponowne czytanie tych samych checkpointów z dysku.
    """

    def __init__(self, cpu_cache_models: int = 16) -> None:
        self.capacity = max(0, int(cpu_cache_models))
        self.cache: OrderedDict[str, LeanCheckpoint] = OrderedDict()

    def get(self, ref: CheckpointRef) -> LeanCheckpoint:
        key = str(ref.path)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache.move_to_end(key)
            return cached

        payload = load_checkpoint(ref.path, map_location="cpu")
        model_cfg = dict(payload["model_config"])
        game_cfg = dict(payload["game_config"])

        if int(model_cfg.get("architecture_version", 1)) != 3:
            raise RuntimeError(
                f"Checkpoint {ref.name} używa starej architektury. "
                "Championship parallel obsługuje aktualne MLP architecture_version=3."
            )

        lean = LeanCheckpoint(
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


# =============================================================================
# TRUE PARALLEL BATCHED MLP
# =============================================================================

class BatchedPolicyEnsemble:
    """Kilka RÓŻNYCH checkpointów liczone jednym bmm na każdą warstwę.

    Wagi Linear mają kształt:
        [MODELE, OUT, IN]

    Wejście dla forward_all:
        [MODELE, IN]

    Jeden torch.bmm liczy wszystkie modele równolegle:
        [MODELE, OUT, IN] @ [MODELE, IN, 1]
    """

    def __init__(self, checkpoints: list[LeanCheckpoint], device: torch.device) -> None:
        if not checkpoints:
            raise ValueError("BatchedPolicyEnsemble wymaga co najmniej 1 checkpointu.")

        self.device = device
        self.num_models = len(checkpoints)
        self.refs = [cp.ref for cp in checkpoints]

        first = checkpoints[0]
        self.model_config = self._normalized_model_cfg(first.model_config)
        self.board_size = int(first.game_config["board_size"])
        self.win_length = int(first.game_config.get("win_length", 6))

        for cp in checkpoints[1:]:
            if int(cp.game_config["board_size"]) != self.board_size or int(
                cp.game_config.get("win_length", 6)
            ) != self.win_length:
                raise ValueError("Checkpointy w jednym batchu używają różnych zasad gry.")
            if self._normalized_model_cfg(cp.model_config) != self.model_config:
                raise ValueError(
                    "Checkpointy w jednym równoległym batchu mają różne architektury MLP. "
                    "W jednym championship katalog powinien zawierać modele tej samej architektury."
                )

        states = [cp.model_state for cp in checkpoints]
        shared_cfg = list(self.model_config.get("layers") or [])
        policy_cfg = list(self.model_config.get("policy_layers") or [])

        self.shared_layers = self._build_layers(states, "layers", shared_cfg)
        self.policy_layers = self._build_layers(states, "policy_layers", policy_cfg)
        self.policy_weight = self._stack(states, "policy_output.weight")
        self.policy_bias = self._stack(states, "policy_output.bias")
        self.action_size = int(self.policy_weight.shape[1])

        first_weight = self.shared_layers[0].weight if self.shared_layers else self.policy_weight
        self.input_size = int(first_weight.shape[-1])

    @staticmethod
    def _normalized_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
        out = dict(cfg)
        out.pop("compile", None)
        out.pop("compile_mode", None)
        return out

    def _stack(self, states: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        try:
            data = torch.stack([state[key].detach().cpu() for state in states], dim=0)
        except KeyError as exc:
            raise KeyError(f"Brak parametru {key} w checkpointach.") from exc
        return data.to(self.device, non_blocking=True)

    def _stack_optional(
        self, states: list[dict[str, torch.Tensor]], key: str
    ) -> torch.Tensor | None:
        if key not in states[0]:
            return None
        return self._stack(states, key)

    def _build_layers(
        self,
        states: list[dict[str, torch.Tensor]],
        prefix: str,
        configs: list[dict[str, Any]],
    ) -> list[BatchedLayer]:
        result: list[BatchedLayer] = []
        for i, cfg in enumerate(configs):
            base = f"{prefix}.{i}.warstwa"
            norm = str(cfg.get("norm", "none")).lower()
            activation = str(cfg.get("activation", "silu")).lower()

            layer = BatchedLayer(
                weight=self._stack(states, f"{base}.0.weight"),
                bias=self._stack_optional(states, f"{base}.0.bias"),
                norm=norm,
                activation=activation,
            )

            if norm == "layer":
                layer.norm_weight = self._stack(states, f"{base}.1.weight")
                layer.norm_bias = self._stack(states, f"{base}.1.bias")
            elif norm == "batch":
                layer.norm_weight = self._stack(states, f"{base}.1.weight")
                layer.norm_bias = self._stack(states, f"{base}.1.bias")
                layer.running_mean = self._stack(states, f"{base}.1.running_mean")
                layer.running_var = self._stack(states, f"{base}.1.running_var")
            elif norm not in ("none", "identity", "off"):
                raise ValueError(f"Nieobsługiwana normalizacja w championship: {norm}")

            result.append(layer)
        return result

    @staticmethod
    def _activate(x: torch.Tensor, name: str) -> torch.Tensor:
        if name in ("none", "identity", "off"):
            return x
        if name == "silu":
            return F.silu(x)
        if name == "gelu":
            return F.gelu(x)
        if name == "relu":
            return F.relu(x)
        if name == "tanh":
            return torch.tanh(x)
        if name == "sigmoid":
            return torch.sigmoid(x)
        raise ValueError(f"Nieobsługiwana aktywacja w championship: {name}")

    @staticmethod
    def _select(tensor: torch.Tensor | None, indices: torch.Tensor) -> torch.Tensor | None:
        if tensor is None:
            return None
        return tensor.index_select(0, indices)

    def _apply_layer_with_params(
        self,
        x: torch.Tensor,
        layer: BatchedLayer,
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        if indices is None:
            w = layer.weight
            b = layer.bias
            nw = layer.norm_weight
            nb = layer.norm_bias
            rm = layer.running_mean
            rv = layer.running_var
        else:
            w = layer.weight.index_select(0, indices)
            b = self._select(layer.bias, indices)
            nw = self._select(layer.norm_weight, indices)
            nb = self._select(layer.norm_bias, indices)
            rm = self._select(layer.running_mean, indices)
            rv = self._select(layer.running_var, indices)

        # x [B, IN], w [B, OUT, IN] -> [B, OUT]
        x = torch.bmm(w, x.unsqueeze(-1)).squeeze(-1)
        if b is not None:
            x = x + b

        if layer.norm == "layer":
            mean = x.mean(dim=-1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            x = (x - mean) * torch.rsqrt(var + layer.eps)
            if nw is not None:
                x = x * nw
            if nb is not None:
                x = x + nb
        elif layer.norm == "batch":
            assert rm is not None and rv is not None
            x = (x - rm) * torch.rsqrt(rv + layer.eps)
            if nw is not None:
                x = x * nw
            if nb is not None:
                x = x + nb

        return self._activate(x, layer.activation)

    def _forward_core(
        self,
        x: torch.Tensor,
        indices: torch.Tensor | None,
    ) -> torch.Tensor:
        for layer in self.shared_layers:
            x = self._apply_layer_with_params(x, layer, indices)
        for layer in self.policy_layers:
            x = self._apply_layer_with_params(x, layer, indices)

        if indices is None:
            w = self.policy_weight
            b = self.policy_bias
        else:
            w = self.policy_weight.index_select(0, indices)
            b = self.policy_bias.index_select(0, indices)

        return torch.bmm(w, x.unsqueeze(-1)).squeeze(-1) + b

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Model i otrzymuje dokładnie pozycję i. Najszybsza ścieżka."""
        if x.shape[0] != self.num_models:
            raise ValueError(
                f"forward_all wymaga {self.num_models} pozycji, otrzymano {x.shape[0]}."
            )
        return self._forward_core(x, None)

    def forward_indexed(self, x: torch.Tensor, model_indices: torch.Tensor) -> torch.Tensor:
        """Każda pozycja może wskazać dowolny model z ensemble."""
        model_indices = model_indices.to(self.device, dtype=torch.long, non_blocking=True)
        if x.shape[0] != model_indices.numel():
            raise ValueError("Liczba pozycji i model_indices musi być taka sama.")
        return self._forward_core(x, model_indices)

    def forward_single_model(self, x: torch.Tensor, model_index: int = 0) -> torch.Tensor:
        """Jeden checkpoint liczy zwykły batch wielu pozycji.

        Ta ścieżka jest używana przez challengera. Używa F.linear, więc jeden model
        efektywnie liczy wszystkie stoły jako normalny batch GPU.
        """
        idx = int(model_index)
        for layer in self.shared_layers:
            w = layer.weight[idx]
            b = None if layer.bias is None else layer.bias[idx]
            x = F.linear(x, w, b)

            if layer.norm == "layer":
                nw = None if layer.norm_weight is None else layer.norm_weight[idx]
                nb = None if layer.norm_bias is None else layer.norm_bias[idx]
                x = F.layer_norm(x, (x.shape[-1],), nw, nb, layer.eps)
            elif layer.norm == "batch":
                assert layer.running_mean is not None and layer.running_var is not None
                nw = None if layer.norm_weight is None else layer.norm_weight[idx]
                nb = None if layer.norm_bias is None else layer.norm_bias[idx]
                x = F.batch_norm(
                    x,
                    layer.running_mean[idx],
                    layer.running_var[idx],
                    nw,
                    nb,
                    training=False,
                    momentum=0.0,
                    eps=layer.eps,
                )

            x = self._activate(x, layer.activation)

        for layer in self.policy_layers:
            w = layer.weight[idx]
            b = None if layer.bias is None else layer.bias[idx]
            x = F.linear(x, w, b)

            if layer.norm == "layer":
                nw = None if layer.norm_weight is None else layer.norm_weight[idx]
                nb = None if layer.norm_bias is None else layer.norm_bias[idx]
                x = F.layer_norm(x, (x.shape[-1],), nw, nb, layer.eps)
            elif layer.norm == "batch":
                assert layer.running_mean is not None and layer.running_var is not None
                nw = None if layer.norm_weight is None else layer.norm_weight[idx]
                nb = None if layer.norm_bias is None else layer.norm_bias[idx]
                x = F.batch_norm(
                    x,
                    layer.running_mean[idx],
                    layer.running_var[idx],
                    nw,
                    nb,
                    training=False,
                    momentum=0.0,
                    eps=layer.eps,
                )

            x = self._activate(x, layer.activation)

        return F.linear(x, self.policy_weight[idx], self.policy_bias[idx])

    def release(self) -> None:
        self.shared_layers.clear()
        self.policy_layers.clear()
        del self.policy_weight
        del self.policy_bias


# =============================================================================
# AMP / ACTIONS
# =============================================================================

def _torch_device(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        print("[championship] CUDA niedostępna -> przechodzę na CPU")
        return torch.device("cpu")
    return torch.device(value)


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if device.type != "cuda" or not enabled:
        return torch.autocast(device_type=device.type, enabled=False)
    dtype_name = dtype_name.lower()
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        raise ValueError("amp_dtype musi być 'bfloat16' albo 'float16'.")
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _choose_actions(
    logits: torch.Tensor,
    legal: torch.Tensor,
    temperature: float,
    generator: torch.Generator,
) -> torch.Tensor:
    logits = mask_logits(logits.float(), legal)
    if temperature <= 0:
        return logits.argmax(dim=1)
    probs = torch.softmax(logits / max(temperature, 1e-4), dim=1)
    return torch.multinomial(probs, 1, generator=generator).squeeze(1)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device="cuda") if device.type == "cuda" else torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def _dummy_actions(legal: torch.Tensor) -> torch.Tensor:
    return legal.to(torch.int8).argmax(dim=1).to(torch.long)


def _reset_finished_envs(env: VectorConnect6, done: torch.Tensor) -> None:
    idx = torch.nonzero(done, as_tuple=False).flatten()
    if idx.numel() > 0:
        env.reset(idx)


# =============================================================================
# MATCH RESULT HELPERS
# =============================================================================

def _empty_counter() -> dict[str, int]:
    return {
        "a_game_wins": 0,
        "b_game_wins": 0,
        "draws": 0,
        "a_wins_as_black": 0,
        "a_wins_as_white": 0,
        "b_wins_as_black": 0,
        "b_wins_as_white": 0,
    }


def _record_done(
    counters: list[dict[str, int]],
    done_indices: torch.Tensor,
    winners: torch.Tensor,
    *,
    a_is_black: bool,
) -> None:
    done_cpu = done_indices.detach().cpu().tolist()
    winners_cpu = winners.detach().cpu().tolist()
    for env_idx, winner in zip(done_cpu, winners_cpu):
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


def _finalize_pair_rows(
    pairings: list[tuple[CheckpointRef, CheckpointRef]],
    counters: list[dict[str, int]],
    games_per_pair: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (a, b), c in zip(pairings, counters):
        a_score = c["a_game_wins"] + 0.5 * c["draws"]
        b_score = c["b_game_wins"] + 0.5 * c["draws"]
        if a_score > b_score:
            a_points, b_points, result = 3, 0, "A"
        elif b_score > a_score:
            a_points, b_points, result = 0, 3, "B"
        else:
            a_points, b_points, result = 1, 1, "DRAW"

        rows.append(
            {
                "match_id": _pair_id(a, b),
                "model_a": a.name,
                "update_a": a.update,
                "model_b": b.name,
                "update_b": b.update,
                "games": games_per_pair,
                **c,
                "a_game_score": a_score,
                "b_game_score": b_score,
                "a_points": a_points,
                "b_points": b_points,
                "match_result": result,
            }
        )
    return rows


# =============================================================================
# TRUE PARALLEL GAME LOOPS
# =============================================================================

@torch.inference_mode()
def play_internal_parallel(
    pairings: list[tuple[CheckpointRef, CheckpointRef, int, int]],
    hosts: BatchedPolicyEnsemble,
    *,
    games_per_pair: int,
    temperature: float,
    amp: bool,
    amp_dtype: str,
    seed: int,
) -> list[dict[str, Any]]:
    """Host-vs-host. Wszystkie aktywne stoły idą przez jeden batched MLP."""
    if not pairings:
        return []

    device = hosts.device
    env = VectorConnect6(
        len(pairings), hosts.board_size, hosts.win_length, device=device, debug_checks=False
    )
    pair_refs = [(a, b) for a, b, _, _ in pairings]
    a_slots = torch.tensor([sa for _, _, sa, _ in pairings], device=device, dtype=torch.long)
    b_slots = torch.tensor([sb for _, _, _, sb in pairings], device=device, dtype=torch.long)
    counters = [_empty_counter() for _ in pairings]
    generator = _make_generator(device, seed)

    for game_index in range(games_per_pair):
        a_is_black = game_index % 2 == 0
        black_slots = a_slots if a_is_black else b_slots
        white_slots = b_slots if a_is_black else a_slots
        env.reset()
        active = torch.ones(len(pairings), dtype=torch.bool, device=device)

        while bool(active.any()):
            x = env.network_input()
            legal = env.legal_mask()
            actions = _dummy_actions(legal)
            idx = torch.nonzero(active, as_tuple=False).flatten()
            actor_slots = torch.where(
                env.current_player[idx].eq(1), black_slots[idx], white_slots[idx]
            )

            with _autocast_context(device, amp, amp_dtype):
                logits = hosts.forward_indexed(x[idx], actor_slots)
            actions[idx] = _choose_actions(logits, legal[idx], temperature, generator)

            step = env.step(actions)
            newly_done = active & step.done
            done_idx = torch.nonzero(newly_done, as_tuple=False).flatten()
            if done_idx.numel() > 0:
                _record_done(
                    counters,
                    done_idx,
                    step.winner[done_idx],
                    a_is_black=a_is_black,
                )
                active[done_idx] = False
            _reset_finished_envs(env, step.done)

    return _finalize_pair_rows(pair_refs, counters, games_per_pair)


@torch.inference_mode()
def play_cross_parallel(
    pairings: list[tuple[CheckpointRef, CheckpointRef, int]],
    hosts: BatchedPolicyEnsemble,
    challenger: BatchedPolicyEnsemble,
    *,
    games_per_pair: int,
    temperature: float,
    amp: bool,
    amp_dtype: str,
    seed: int,
) -> list[dict[str, Any]]:
    """N hostów kontra jeden challenger.

    Hosty: różne modele -> jeden bmm.
    Challenger: jeden model -> zwykły batch wszystkich stołów.
    """
    if not pairings:
        return []

    if challenger.num_models != 1:
        raise ValueError("Cross batch oczekuje dokładnie jednego challengera.")
    if challenger.board_size != hosts.board_size or challenger.win_length != hosts.win_length:
        raise ValueError("Challenger ma inne zasady gry niż hosty.")
    if challenger.model_config != hosts.model_config:
        raise ValueError(
            "Challenger ma inną architekturę MLP niż hosty. "
            "Ten szybki tryb bmm wymaga zgodnej architektury w jednym turnieju."
        )

    device = hosts.device
    env = VectorConnect6(
        len(pairings), hosts.board_size, hosts.win_length, device=device, debug_checks=False
    )
    pair_refs = [(a, b) for a, b, _ in pairings]
    host_slots = torch.tensor([slot for _, _, slot in pairings], device=device, dtype=torch.long)
    counters = [_empty_counter() for _ in pairings]
    generator = _make_generator(device, seed)

    for game_index in range(games_per_pair):
        a_is_black = game_index % 2 == 0  # A = host
        env.reset()
        active = torch.ones(len(pairings), dtype=torch.bool, device=device)

        while bool(active.any()):
            x = env.network_input()
            legal = env.legal_mask()
            actions = _dummy_actions(legal)
            idx = torch.nonzero(active, as_tuple=False).flatten()

            player = env.current_player[idx]
            host_actor = (player.eq(1) & a_is_black) | (player.eq(-1) & (not a_is_black))
            host_idx = idx[host_actor]
            challenger_idx = idx[~host_actor]

            if host_idx.numel() > 0:
                # Pozycja w pairings == pozycja w host_slots.
                selected_slots = host_slots[host_idx]
                with _autocast_context(device, amp, amp_dtype):
                    logits = hosts.forward_indexed(x[host_idx], selected_slots)
                actions[host_idx] = _choose_actions(
                    logits, legal[host_idx], temperature, generator
                )

            if challenger_idx.numel() > 0:
                with _autocast_context(device, amp, amp_dtype):
                    logits = challenger.forward_single_model(x[challenger_idx], 0)
                actions[challenger_idx] = _choose_actions(
                    logits, legal[challenger_idx], temperature, generator
                )

            step = env.step(actions)
            newly_done = active & step.done
            done_idx = torch.nonzero(newly_done, as_tuple=False).flatten()
            if done_idx.numel() > 0:
                _record_done(
                    counters,
                    done_idx,
                    step.winner[done_idx],
                    a_is_black=a_is_black,
                )
                active[done_idx] = False
            _reset_finished_envs(env, step.done)

    return _finalize_pair_rows(pair_refs, counters, games_per_pair)


# =============================================================================
# CSV / RESUME
# =============================================================================

def _load_completed_matches(path: Path) -> tuple[set[str], list[dict[str, str]]]:
    if not path.exists() or path.stat().st_size == 0:
        return set(), []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["match_id"] for row in rows if row.get("match_id")}, rows


def _append_match(path: Path, row: dict[str, Any]) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MATCH_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in MATCH_FIELDS})
        f.flush()


def _write_state(
    path: Path,
    *,
    checkpoints: list[CheckpointRef],
    games_per_pair: int,
    temperature: float,
) -> None:
    state = {
        "version": 2,
        "games_per_pair": games_per_pair,
        "temperature": temperature,
        "checkpoints": [ref.name for ref in checkpoints],
    }
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def _validate_or_create_state(
    path: Path,
    *,
    checkpoints: list[CheckpointRef],
    games_per_pair: int,
    temperature: float,
) -> None:
    if not path.exists():
        _write_state(
            path,
            checkpoints=checkpoints,
            games_per_pair=games_per_pair,
            temperature=temperature,
        )
        return

    old = json.loads(path.read_text(encoding="utf-8"))
    if int(old.get("games_per_pair", -1)) != games_per_pair:
        raise RuntimeError(
            "Istniejący turniej używa innego games_per_pair. "
            "Użyj nowego output_dir albo usuń stary katalog mistrzostw."
        )
    if not math.isclose(
        float(old.get("temperature", 999.0)), temperature, abs_tol=1e-12
    ):
        raise RuntimeError(
            "Istniejący turniej używa innej temperatury. "
            "Użyj nowego output_dir albo usuń stary katalog mistrzostw."
        )

    old_names = set(old.get("checkpoints", []))
    new_names = {ref.name for ref in checkpoints}
    if not old_names.issubset(new_names):
        missing = sorted(old_names - new_names)
        raise RuntimeError(
            "Z katalogu checkpointów zniknęły modele użyte w tym turnieju: "
            + ", ".join(missing[:10])
        )
    if old_names != new_names:
        _write_state(
            path,
            checkpoints=checkpoints,
            games_per_pair=games_per_pair,
            temperature=temperature,
        )


# =============================================================================
# RANKING
# =============================================================================

def build_ranking(
    checkpoints: list[CheckpointRef], match_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for ref in checkpoints:
        stats[ref.name] = {
            "model": ref.name,
            "update": ref.update,
            "points": 0,
            "match_wins": 0,
            "match_draws": 0,
            "match_losses": 0,
            "game_wins": 0,
            "game_draws": 0,
            "game_losses": 0,
            "wins_as_black": 0,
            "wins_as_white": 0,
            "games_as_black": 0,
            "games_as_white": 0,
            "opponents_played": 0,
        }

    def as_int(row: dict[str, Any], key: str) -> int:
        return int(float(row.get(key, 0) or 0))

    for row in match_rows:
        a_name, b_name = str(row["model_a"]), str(row["model_b"])
        if a_name not in stats or b_name not in stats:
            continue
        a, b = stats[a_name], stats[b_name]
        games = as_int(row, "games")
        half = games // 2
        draws = as_int(row, "draws")
        a_wins = as_int(row, "a_game_wins")
        b_wins = as_int(row, "b_game_wins")
        a_points = as_int(row, "a_points")
        b_points = as_int(row, "b_points")

        a["points"] += a_points
        b["points"] += b_points
        a["opponents_played"] += 1
        b["opponents_played"] += 1

        if a_points == 3:
            a["match_wins"] += 1
            b["match_losses"] += 1
        elif b_points == 3:
            b["match_wins"] += 1
            a["match_losses"] += 1
        else:
            a["match_draws"] += 1
            b["match_draws"] += 1

        a["game_wins"] += a_wins
        a["game_draws"] += draws
        a["game_losses"] += b_wins
        b["game_wins"] += b_wins
        b["game_draws"] += draws
        b["game_losses"] += a_wins

        a["wins_as_black"] += as_int(row, "a_wins_as_black")
        a["wins_as_white"] += as_int(row, "a_wins_as_white")
        b["wins_as_black"] += as_int(row, "b_wins_as_black")
        b["wins_as_white"] += as_int(row, "b_wins_as_white")
        a["games_as_black"] += half
        a["games_as_white"] += half
        b["games_as_black"] += half
        b["games_as_white"] += half

    ranking: list[dict[str, Any]] = []
    for row in stats.values():
        games = row["game_wins"] + row["game_draws"] + row["game_losses"]
        score = row["game_wins"] + 0.5 * row["game_draws"]
        row["games_played"] = games
        row["game_score"] = score
        row["game_score_pct"] = 100.0 * score / games if games else 0.0
        row["black_win_rate"] = (
            100.0 * row["wins_as_black"] / row["games_as_black"]
            if row["games_as_black"]
            else 0.0
        )
        row["white_win_rate"] = (
            100.0 * row["wins_as_white"] / row["games_as_white"]
            if row["games_as_white"]
            else 0.0
        )
        ranking.append(row)

    ranking.sort(
        key=lambda x: (
            x["points"],
            x["match_wins"],
            x["game_score"],
            x["game_wins"],
            x["update"],
        ),
        reverse=True,
    )
    for place, row in enumerate(ranking, start=1):
        row["place"] = place
    return ranking


def write_ranking_csv(path: Path, ranking: list[dict[str, Any]]) -> None:
    fields = [
        "place",
        "model",
        "update",
        "points",
        "match_wins",
        "match_draws",
        "match_losses",
        "game_wins",
        "game_draws",
        "game_losses",
        "games_played",
        "game_score_pct",
        "wins_as_black",
        "wins_as_white",
        "black_win_rate",
        "white_win_rate",
        "opponents_played",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ranking:
            writer.writerow({key: row.get(key, "") for key in fields})


# =============================================================================
# HTML
# =============================================================================

def write_html(
    path: Path,
    ranking: list[dict[str, Any]],
    *,
    completed_pairs: int,
    total_pairs: int,
    checkpoint_count: int,
    games_per_pair: int,
    tables: int,
    temperature: float,
    running: bool,
) -> None:
    champion = ranking[0]["model"] if ranking and completed_pairs else "—"
    progress = 100.0 * completed_pairs / total_pairs if total_pairs else 100.0

    rows_html: list[str] = []
    for row in ranking:
        crown = "👑 " if row["place"] == 1 and completed_pairs else ""
        rows_html.append(
            "<tr>"
            f"<td class='place'>{row['place']}</td>"
            f"<td class='model'>{crown}{html.escape(str(row['model']))}</td>"
            f"<td>{row['update']}</td>"
            f"<td class='points'>{row['points']}</td>"
            f"<td>{row['match_wins']} / {row['match_draws']} / {row['match_losses']}</td>"
            f"<td>{row['game_wins']} / {row['game_draws']} / {row['game_losses']}</td>"
            f"<td>{row['game_score_pct']:.2f}%</td>"
            f"<td>{row['wins_as_black']}</td>"
            f"<td>{row['wins_as_white']}</td>"
            f"<td>{row['black_win_rate']:.2f}%</td>"
            f"<td>{row['white_win_rate']:.2f}%</td>"
            f"<td>{row['opponents_played']}</td>"
            "</tr>"
        )

    refresh = "<meta http-equiv='refresh' content='10'>" if running else ""
    status = "TRWA" if running else "ZAKOŃCZONE"

    doc = """<!doctype html>
<html lang='pl'>
<head>
<meta charset='utf-8'>
__REFRESH__
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>King of Connect6 — AI Championship</title>
<style>
*{box-sizing:border-box}
body{margin:0;background:#f3f5f8;color:#1f2937;font-family:Segoe UI,Arial,sans-serif}
.container{max-width:1800px;margin:0 auto;padding:24px}
h1{margin:0 0 6px;font-size:30px}
.subtitle{color:#64748b;margin-bottom:18px}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}
.card,.panel{background:#fff;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.08)}
.card{padding:14px 16px}.label{font-size:13px;color:#64748b;margin-bottom:5px}.value{font-size:22px;font-weight:700}
.panel{overflow:auto}.progress{height:12px;background:#e5e7eb;border-radius:999px;overflow:hidden;margin:0 0 18px}.bar{height:100%;background:#334155;width:__PROGRESS__%}
table{width:100%;border-collapse:collapse;min-width:1250px}th,td{padding:10px 12px;border-bottom:1px solid #e5e7eb;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:#f8fafc;color:#475569;font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.model,th.model{text-align:left}.place{text-align:center}.points{font-weight:800}tbody tr:nth-child(1){background:#fff7cc}tbody tr:nth-child(2){background:#f8fafc}tbody tr:nth-child(3){background:#fff4e6}
.note{color:#64748b;font-size:13px;line-height:1.5;margin:14px 2px}.status{font-weight:800}
</style></head><body><div class='container'>
<h1>👑 King of Connect6 — AI Championship</h1>
<div class='subtitle'>Pełny round-robin. Stoły są liczone równolegle na GPU; różne hosty przechodzą przez batched MLP / torch.bmm.</div>
<div class='cards'>
<div class='card'><div class='label'>Status</div><div class='value status'>__STATUS__</div></div>
<div class='card'><div class='label'>Aktualny lider</div><div class='value'>__CHAMPION__</div></div>
<div class='card'><div class='label'>Pojedynki</div><div class='value'>__DONE__ / __TOTAL__</div></div>
<div class='card'><div class='label'>Postęp</div><div class='value'>__PROGRESS_TEXT__%</div></div>
<div class='card'><div class='label'>Modele</div><div class='value'>__MODELS__</div></div>
<div class='card'><div class='label'>Stoły GPU równolegle</div><div class='value'>__TABLES__</div></div>
<div class='card'><div class='label'>Gier na parę</div><div class='value'>__GAMES__</div></div>
<div class='card'><div class='label'>Temperatura</div><div class='value'>__TEMP__</div></div>
</div>
<div class='progress'><div class='bar'></div></div>
<div class='panel'><table><thead><tr>
<th>#</th><th class='model'>Model</th><th>Update</th><th>Pkt</th><th>Mecze W/R/P</th><th>Gry W/R/P</th><th>Game score</th><th>Wygr. czarne</th><th>Wygr. białe</th><th>WR czarne</th><th>WR białe</th><th>Rywali</th>
</tr></thead><tbody>__ROWS__</tbody></table></div>
<div class='note'>3 pkt za wygrany cały pojedynek/serię, 1 pkt za remis serii, 0 pkt za przegraną. „Mecze W/R/P” dotyczą całych serii z innymi modelami. „Gry W/R/P” dotyczą pojedynczych partii.</div>
</div></body></html>"""

    replacements = {
        "__REFRESH__": refresh,
        "__PROGRESS__": f"{progress:.4f}",
        "__PROGRESS_TEXT__": f"{progress:.2f}",
        "__STATUS__": status,
        "__CHAMPION__": html.escape(str(champion)),
        "__DONE__": f"{completed_pairs:,}".replace(",", " "),
        "__TOTAL__": f"{total_pairs:,}".replace(",", " "),
        "__MODELS__": str(checkpoint_count),
        "__TABLES__": str(tables),
        "__GAMES__": str(games_per_pair),
        "__TEMP__": str(temperature),
        "__ROWS__": "".join(rows_html),
    }
    for key, value in replacements.items():
        doc = doc.replace(key, value)
    path.write_text(doc, encoding="utf-8")


# =============================================================================
# VRAM INFO
# =============================================================================

def _vram_label(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    allocated = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    return f"VRAM alloc={allocated:.2f} GB reserved={reserved:.2f} GB"


def _cleanup_cuda(device: torch.device, empty_cache: bool) -> None:
    gc.collect()
    if device.type == "cuda" and empty_cache:
        torch.cuda.empty_cache()


# =============================================================================
# MAIN CHAMPIONSHIP
# =============================================================================

def run_championship(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = _read_yaml(config_path)
    ch = cfg.get("championship", cfg)
    if not isinstance(ch, dict):
        raise ValueError("Brak sekcji 'championship' w YAML.")

    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _resolve_path(
        project_root, ch.get("checkpoint_dir", "runs/connect6_mlp_01/checkpoints")
    )
    output_dir = _resolve_path(
        project_root, ch.get("output_dir", "runs/connect6_mlp_01/championship")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = int(ch.get("tables", 4))
    games_per_pair = int(ch.get("games_per_pair", 2))
    temperature = float(ch.get("temperature", 0.0))
    device = _torch_device(str(ch.get("device", "cuda")))
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    empty_cache = bool(ch.get("empty_cache_after_unload", True))
    cpu_cache_models = int(ch.get("cpu_cache_models", 16))
    seed = int(ch.get("seed", 12345))

    if tables <= 0:
        raise ValueError("tables musi być > 0.")
    if games_per_pair <= 0 or games_per_pair % 2 != 0:
        raise ValueError("games_per_pair musi być dodatnią liczbą PARZYSTĄ.")

    checkpoints = discover_checkpoints(checkpoint_dir)
    if len(checkpoints) < 2:
        raise RuntimeError(
            f"Potrzeba co najmniej 2 checkpointów model_update_*.pt w {checkpoint_dir}"
        )

    matches_path = output_dir / "matches.csv"
    ranking_path = output_dir / "ranking.csv"
    html_path = output_dir / "championship.html"
    state_path = output_dir / "state.json"

    _validate_or_create_state(
        state_path,
        checkpoints=checkpoints,
        games_per_pair=games_per_pair,
        temperature=temperature,
    )
    completed_ids, match_rows = _load_completed_matches(matches_path)
    total_pairs = len(checkpoints) * (len(checkpoints) - 1) // 2

    print("=" * 78)
    print("KING OF CONNECT6 — TRUE PARALLEL AI CHAMPIONSHIP")
    print("=" * 78)
    print(f"Checkpointy: {len(checkpoints)}")
    print(f"Pary łącznie: {total_pairs:,}")
    print(f"Już rozegrane: {len(completed_ids):,}")
    print(f"Stoły GPU równolegle: {tables}")
    print(f"Gier na parę: {games_per_pair} ({games_per_pair // 2} każdym kolorem)")
    print(f"Modeli wagowo w VRAM podczas cross-round: ~{min(tables, len(checkpoints)) + 1}")
    print(f"CPU lean-cache checkpointów: {cpu_cache_models}")
    print(f"Device: {device}")
    print(f"Wyniki: {output_dir}")
    if temperature <= 0 and games_per_pair > 2:
        print(
            "UWAGA: temperature=0 i games_per_pair>2 -> kolejne gry tych samych "
            "kolorów mogą być identyczne."
        )

    ranking = build_ranking(checkpoints, match_rows)
    write_ranking_csv(ranking_path, ranking)
    write_html(
        html_path,
        ranking,
        completed_pairs=len(completed_ids),
        total_pairs=total_pairs,
        checkpoint_count=len(checkpoints),
        games_per_pair=games_per_pair,
        tables=tables,
        temperature=temperature,
        running=len(completed_ids) < total_pairs,
    )

    store = CheckpointStore(cpu_cache_models=cpu_cache_models)

    def commit_rows(rows: list[dict[str, Any]], elapsed: float) -> None:
        nonlocal match_rows, ranking
        if not rows:
            return
        per_match_elapsed = elapsed / len(rows)
        for row in rows:
            row["elapsed_seconds"] = per_match_elapsed
            _append_match(matches_path, row)
            completed_ids.add(str(row["match_id"]))
            match_rows.append(row)
            print(
                f"[{len(completed_ids):>6}/{total_pairs}] "
                f"{row['model_a']} vs {row['model_b']} | "
                f"gry {row['a_game_wins']}-{row['draws']}-{row['b_game_wins']} | "
                f"pkt {row['a_points']}:{row['b_points']}"
            )

        ranking = build_ranking(checkpoints, match_rows)
        write_ranking_csv(ranking_path, ranking)
        write_html(
            html_path,
            ranking,
            completed_pairs=len(completed_ids),
            total_pairs=total_pairs,
            checkpoint_count=len(checkpoints),
            games_per_pair=games_per_pair,
            tables=tables,
            temperature=temperature,
            running=len(completed_ids) < total_pairs,
        )
        rate = len(rows) / elapsed if elapsed > 0 else 0.0
        print(
            f"[BATCH] {len(rows)} pojedynków równolegle | {elapsed:.2f}s | "
            f"{rate:.3f} pojedynków/s | {_vram_label(device)}"
        )

    try:
        # ---------------------------------------------------------------------
        # BLOK HOSTÓW = dokładnie maks. 'tables' różnych modeli.
        # Ich wagi są scalane w jeden BatchedPolicyEnsemble i pozostają w VRAM.
        # ---------------------------------------------------------------------
        for block_start in range(0, len(checkpoints), tables):
            host_refs = checkpoints[block_start : block_start + tables]
            challenger_start = block_start + len(host_refs)

            internal_pending = [
                (a, b)
                for a, b in combinations(host_refs, 2)
                if _pair_id(a, b) not in completed_ids
            ]
            has_cross_pending = any(
                _pair_id(host, challenger) not in completed_ids
                for challenger in checkpoints[challenger_start:]
                for host in host_refs
            )
            if not internal_pending and not has_cross_pending:
                continue

            print("\n" + "-" * 78)
            print(
                f"[HOST BLOCK] {block_start + 1}-{block_start + len(host_refs)} / "
                f"{len(checkpoints)} | buduję batched MLP dla {len(host_refs)} hostów"
            )
            host_checkpoints = [store.get(ref) for ref in host_refs]
            hosts = BatchedPolicyEnsemble(host_checkpoints, device)
            host_slot = {ref.name: i for i, ref in enumerate(host_refs)}
            print(f"[HOST BLOCK READY] {_vram_label(device)}")

            # -------------------------------------------------------------
            # Mecze wewnątrz host blocku. Nadal TRUE PARALLEL: jeden bmm
            # obsługuje wszystkie stoły w batchu.
            # -------------------------------------------------------------
            internal_with_slots = [
                (a, b, host_slot[a.name], host_slot[b.name])
                for a, b in internal_pending
            ]
            for batch in _chunks(internal_with_slots, tables):
                started = time.perf_counter()
                rows = play_internal_parallel(
                    batch,
                    hosts,
                    games_per_pair=games_per_pair,
                    temperature=temperature,
                    amp=amp,
                    amp_dtype=amp_dtype,
                    seed=seed + len(completed_ids),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                commit_rows(rows, time.perf_counter() - started)

            # -------------------------------------------------------------
            # Każdy challenger jest wczytywany JEDEN RAZ dla tego host blocku
            # i gra z maks. 'tables' hostami równolegle.
            # -------------------------------------------------------------
            for challenger_ref in checkpoints[challenger_start:]:
                cross = [
                    (host, challenger_ref, host_slot[host.name])
                    for host in host_refs
                    if _pair_id(host, challenger_ref) not in completed_ids
                ]
                if not cross:
                    continue

                challenger_cp = store.get(challenger_ref)
                challenger = BatchedPolicyEnsemble([challenger_cp], device)
                print(
                    f"[CHALLENGER] {challenger_ref.name} vs {len(cross)} hostów RÓWNOLEGLE | "
                    f"{_vram_label(device)}"
                )

                started = time.perf_counter()
                rows = play_cross_parallel(
                    cross,
                    hosts,
                    challenger,
                    games_per_pair=games_per_pair,
                    temperature=temperature,
                    amp=amp,
                    amp_dtype=amp_dtype,
                    seed=seed + len(completed_ids),
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                commit_rows(rows, time.perf_counter() - started)

                challenger.release()
                del challenger
                del challenger_cp
                _cleanup_cuda(device, empty_cache)

            hosts.release()
            del hosts
            del host_checkpoints
            _cleanup_cuda(device, empty_cache)

    except KeyboardInterrupt:
        print(
            "\n[championship] Przerwano. Wyniki są zapisane. "
            "Ponowne uruchomienie wznowi turniej."
        )
    finally:
        _cleanup_cuda(device, empty_cache)
        completed_ids, final_rows = _load_completed_matches(matches_path)
        ranking = build_ranking(checkpoints, final_rows)
        write_ranking_csv(ranking_path, ranking)
        write_html(
            html_path,
            ranking,
            completed_pairs=len(completed_ids),
            total_pairs=total_pairs,
            checkpoint_count=len(checkpoints),
            games_per_pair=games_per_pair,
            tables=tables,
            temperature=temperature,
            running=len(completed_ids) < total_pairs,
        )

    if len(completed_ids) >= total_pairs:
        print("\n" + "=" * 78)
        print("MISTRZOSTWA ZAKOŃCZONE")
        if ranking:
            print(f"👑 KING OF CONNECT6: {ranking[0]['model']} | {ranking[0]['points']} pkt")
        print(f"Ranking HTML: {html_path}")
        print("=" * 78)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="King of Connect6 — równoległe mistrzostwa checkpointów"
    )
    parser.add_argument(
        "--config",
        default="configs/championship.yaml",
        help="Ścieżka do konfiguracji mistrzostw YAML",
    )
    args = parser.parse_args()
    run_championship(args.config)


if __name__ == "__main__":
    main()