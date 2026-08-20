from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .checkpoint import list_versioned_checkpoints, load_checkpoint


_UPDATE_RE = re.compile(r"model_update_(\d+)\.pt$")


@dataclass(slots=True)
class HistoricalModel:
    """Legacy compatibility container used by older tests/tools."""

    path: Path
    update: int
    model: torch.nn.Module


@dataclass(slots=True)
class HistoricalCheckpoint:
    path: Path
    update: int
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


def checkpoint_update(path: str | Path) -> int | None:
    match = _UPDATE_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def _normalized_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out.pop("compile", None)
    out.pop("compile_mode", None)
    return out


def _lean_cache_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.parent / "_history_cache" / checkpoint_path.name


def _load_lean_checkpoint(path: Path) -> HistoricalCheckpoint:
    """Read a lightweight history payload, creating a sidecar cache if needed."""

    update = checkpoint_update(path)
    if update is None:
        raise ValueError(f"Niepoprawna nazwa checkpointu: {path.name}")

    cache_path = _lean_cache_path(path)
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    else:
        full = load_checkpoint(path, map_location="cpu")
        payload = {
            "update": update,
            "model_state": full["model_state"],
            "model_config": full["model_config"],
            "game_config": full["game_config"],
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
        try:
            torch.save(payload, tmp)
            tmp.replace(cache_path)
        except OSError:
            # Cache is an optimization only. Failure must not kill training.
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        del full

    return HistoricalCheckpoint(
        path=path,
        update=int(payload.get("update", update)),
        model_state=payload["model_state"],
        model_config=dict(payload["model_config"]),
        game_config=dict(payload["game_config"]),
    )


def load_random_historical_checkpoints(
    checkpoint_dir: str | Path,
    *,
    current_update: int,
    requested_count: int,
    required_model_config: dict[str, Any] | None = None,
    required_game_config: dict[str, Any] | None = None,
) -> list[HistoricalCheckpoint]:
    """Choose a random compatible pool of old checkpoints for one PPO update.

    `latest.pt` is ignored. Lean sidecars exclude optimizer state. One grouped bmm
    requires identical tensor shapes, therefore every returned checkpoint belongs
    to the same model/game family. If required_* is supplied, that exact current
    family is used; otherwise the first compatible random checkpoint defines it.
    """

    requested_count = max(0, int(requested_count))
    if requested_count == 0:
        return []

    required_model = (
        _normalized_model_cfg(required_model_config)
        if required_model_config is not None
        else None
    )
    required_board_size = (
        int(required_game_config["board_size"])
        if required_game_config is not None
        else None
    )
    required_win_length = (
        int(required_game_config.get("win_length", 6))
        if required_game_config is not None
        else None
    )

    candidates: list[Path] = []
    for path in list_versioned_checkpoints(checkpoint_dir):
        update = checkpoint_update(path)
        if update is not None and update < int(current_update):
            candidates.append(Path(path))
    random.shuffle(candidates)

    family_model = required_model
    family_board = required_board_size
    family_win = required_win_length

    loaded: list[HistoricalCheckpoint] = []
    for path in candidates:
        if len(loaded) >= requested_count:
            break

        try:
            checkpoint = _load_lean_checkpoint(path)
        except (RuntimeError, KeyError, ValueError, OSError) as exc:
            print(f"[history] pomijam {path.name}: {exc}")
            continue

        cfg = _normalized_model_cfg(checkpoint.model_config)
        board_size = int(checkpoint.game_config["board_size"])
        win_length = int(checkpoint.game_config.get("win_length", 6))
        if int(cfg.get("architecture_version", 1)) != 3:
            continue

        if family_model is None:
            family_model = cfg
            family_board = board_size
            family_win = win_length

        if cfg != family_model or board_size != family_board or win_length != family_win:
            continue

        loaded.append(checkpoint)

    return loaded


class HistoricalPolicyEnsemble:
    """Many frozen historical MLP policies evaluated as one batched GPU model.

    Weights are stacked as [MODELS, OUT, IN]. Inputs are grouped by permanently
    assigned tables and have shape [MODELS, TABLES_PER_MODEL, IN]. Each MLP layer
    is one torch.bmm across every historical model instead of a Python loop issuing
    tiny forwards. Only the shared trunk and policy head are materialized.
    """

    def __init__(
        self,
        checkpoints: list[HistoricalCheckpoint],
        device: torch.device,
    ) -> None:
        if not checkpoints:
            raise ValueError("HistoricalPolicyEnsemble wymaga co najmniej 1 checkpointu")

        self.device = device
        self.num_models = len(checkpoints)
        self.updates = [cp.update for cp in checkpoints]
        self.paths = [cp.path for cp in checkpoints]

        first_cfg = _normalized_model_cfg(checkpoints[0].model_config)
        first_game = checkpoints[0].game_config
        self.board_size = int(first_game["board_size"])

        for cp in checkpoints[1:]:
            if int(cp.game_config["board_size"]) != self.board_size:
                raise ValueError("Historyczne checkpointy używają różnych rozmiarów planszy")
            if _normalized_model_cfg(cp.model_config) != first_cfg:
                raise ValueError("Historyczne checkpointy mają różne architektury MLP")

        states = [cp.model_state for cp in checkpoints]
        shared_cfg = list(first_cfg.get("layers") or [])
        policy_cfg = list(first_cfg.get("policy_layers") or [])

        self.shared_layers = self._build_layers(states, "layers", shared_cfg)
        self.policy_layers = self._build_layers(states, "policy_layers", policy_cfg)
        self.policy_weight = self._stack(states, "policy_output.weight")
        self.policy_bias = self._stack_optional(states, "policy_output.bias")
        self.action_size = int(self.policy_weight.shape[1])

        first_weight = self.shared_layers[0].weight if self.shared_layers else self.policy_weight
        self.input_size = int(first_weight.shape[-1])

    def _stack(self, states: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        try:
            tensor = torch.stack([state[key].detach().cpu() for state in states], dim=0)
        except KeyError as exc:
            raise KeyError(f"Brak parametru {key} w historycznych checkpointach") from exc
        return tensor.to(self.device, non_blocking=True)

    def _stack_optional(
        self,
        states: list[dict[str, torch.Tensor]],
        key: str,
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
                raise ValueError(f"Nieobsługiwana normalizacja history ensemble: {norm}")

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
        raise ValueError(f"Nieobsługiwana aktywacja history ensemble: {name}")

    def _apply_layer(self, x: torch.Tensor, layer: BatchedLayer) -> torch.Tensor:
        # x [M, T, IN], w [M, OUT, IN]
        # bmm -> [M, OUT, T] -> [M, T, OUT]
        x = torch.bmm(layer.weight, x.transpose(1, 2)).transpose(1, 2)
        if layer.bias is not None:
            x = x + layer.bias.unsqueeze(1)

        if layer.norm == "layer":
            mean = x.mean(dim=-1, keepdim=True)
            var = (x - mean).pow(2).mean(dim=-1, keepdim=True)
            x = (x - mean) * torch.rsqrt(var + layer.eps)
            if layer.norm_weight is not None:
                x = x * layer.norm_weight.unsqueeze(1)
            if layer.norm_bias is not None:
                x = x + layer.norm_bias.unsqueeze(1)
        elif layer.norm == "batch":
            assert layer.running_mean is not None and layer.running_var is not None
            x = (x - layer.running_mean.unsqueeze(1)) * torch.rsqrt(
                layer.running_var.unsqueeze(1) + layer.eps
            )
            if layer.norm_weight is not None:
                x = x * layer.norm_weight.unsqueeze(1)
            if layer.norm_bias is not None:
                x = x + layer.norm_bias.unsqueeze(1)

        return self._activate(x, layer.activation)

    def forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        """Return policy logits for [models, fixed_tables_per_model, input]."""
        if x.ndim != 3:
            raise ValueError("History ensemble oczekuje wejścia [MODELE, STOŁY, INPUT]")
        if x.shape[0] != self.num_models or x.shape[2] != self.input_size:
            raise ValueError(
                f"Niepoprawny history batch {tuple(x.shape)}; oczekiwano "
                f"[{self.num_models}, T, {self.input_size}]"
            )

        for layer in self.shared_layers:
            x = self._apply_layer(x, layer)
        for layer in self.policy_layers:
            x = self._apply_layer(x, layer)

        logits = torch.bmm(self.policy_weight, x.transpose(1, 2)).transpose(1, 2)
        if self.policy_bias is not None:
            logits = logits + self.policy_bias.unsqueeze(1)
        return logits
