from __future__ import annotations

import random
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .checkpoint import list_versioned_checkpoints, load_checkpoint


_UPDATE_RE = re.compile(r"model_update_(\d+)\.pt$")


@dataclass(slots=True)
class HistoricalModel:
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


class HistoricalCheckpointCache:
    """Ograniczony LRU cache lekkich checkpointów historycznych w RAM."""

    def __init__(self, max_models: int = 0) -> None:
        self.max_models = max(0, int(max_models))
        self._items: OrderedDict[Path, HistoricalCheckpoint] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._items)

    def get(self, path: str | Path) -> HistoricalCheckpoint:
        key = Path(path)
        cached = self._items.pop(key, None)
        if cached is not None:
            self._items[key] = cached
            self.hits += 1
            return cached

        self.misses += 1
        checkpoint = _load_lean_checkpoint(key)
        if self.max_models > 0:
            self._items[key] = checkpoint
            while len(self._items) > self.max_models:
                self._items.popitem(last=False)
        return checkpoint


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
    ram_cache: HistoricalCheckpointCache | None = None,
) -> list[HistoricalCheckpoint]:
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
            checkpoint = ram_cache.get(path) if ram_cache is not None else _load_lean_checkpoint(path)
        except (RuntimeError, KeyError, ValueError, OSError) as exc:
            print(f"[history] pomijam {path.name}: {exc}")
            continue

        cfg = _normalized_model_cfg(checkpoint.model_config)
        board_size = int(checkpoint.game_config["board_size"])
        win_length = int(checkpoint.game_config.get("win_length", 6))
        if int(cfg.get("architecture_version", 0)) != 4:
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
    """Wiele zamrożonych polityk CNN liczonych jednym grouped-conv na warstwę.

    Wejście ma [MODELE, STOŁY, 3, H, W]. Dla każdej warstwy modele są składane
    w grupy kanałów i wykonywany jest jeden F.conv2d(groups=MODELE), dzięki czemu
    zachowujemy wcześniejszą zaletę: brak pętli małych forwardów model-po-modelu.
    Liczona jest wyłącznie policy; historyczne VALUE nie jest potrzebne.
    """

    def __init__(
        self,
        checkpoints: list[HistoricalCheckpoint],
        device: torch.device,
        *,
        dtype: torch.dtype | None = None,
    ) -> None:
        if not checkpoints:
            raise ValueError("HistoricalPolicyEnsemble wymaga co najmniej 1 checkpointu")

        self.device = device
        self.dtype = dtype or torch.float32
        self.num_models = len(checkpoints)
        self.updates = [cp.update for cp in checkpoints]
        self.paths = [cp.path for cp in checkpoints]

        first_cfg = _normalized_model_cfg(checkpoints[0].model_config)
        first_game = checkpoints[0].game_config
        self.board_size = int(first_game["board_size"])
        self.kernels = tuple(int(v) for v in first_cfg.get("kernels", (23, 3, 3, 3, 3, 3, 3, 3)))
        self.channels = tuple(int(v) for v in first_cfg.get("channels", (32, 32, 64, 64, 64, 96, 96, 96)))
        self.input_channels = 3

        for cp in checkpoints[1:]:
            if int(cp.game_config["board_size"]) != self.board_size:
                raise ValueError("Historyczne checkpointy używają różnych rozmiarów planszy")
            if _normalized_model_cfg(cp.model_config) != first_cfg:
                raise ValueError("Historyczne checkpointy mają różne architektury CNN")

        states = [cp.model_state for cp in checkpoints]
        self.conv_weights: list[torch.Tensor] = []
        self.conv_biases: list[torch.Tensor | None] = []
        for i in range(len(self.kernels)):
            self.conv_weights.append(self._stack(states, f"convs.{i}.weight"))
            self.conv_biases.append(self._stack_optional(states, f"convs.{i}.bias"))

        self.policy_weight = self._stack(states, "policy_output.weight")
        self.policy_bias = self._stack_optional(states, "policy_output.bias")
        self.action_size = self.board_size * self.board_size

    def _stack(self, states: list[dict[str, torch.Tensor]], key: str) -> torch.Tensor:
        try:
            tensor = torch.stack([state[key].detach().cpu() for state in states], dim=0)
        except KeyError as exc:
            raise KeyError(f"Brak parametru {key} w historycznych checkpointach") from exc
        return tensor.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def _stack_optional(
        self,
        states: list[dict[str, torch.Tensor]],
        key: str,
    ) -> torch.Tensor | None:
        if key not in states[0]:
            return None
        return self._stack(states, key)

    def _grouped_conv(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        padding: int,
    ) -> torch.Tensor:
        # x: [M,T,C,H,W], weight: [M,O,C,K,K]
        m, t, c, h, w = x.shape
        out_channels = int(weight.shape[1])
        kernel = int(weight.shape[-1])

        grouped_x = x.permute(1, 0, 2, 3, 4).reshape(t, m * c, h, w)
        grouped_w = weight.reshape(m * out_channels, c, kernel, kernel)
        grouped_b = bias.reshape(m * out_channels) if bias is not None else None
        y = F.conv2d(
            grouped_x,
            grouped_w,
            grouped_b,
            stride=1,
            padding=padding,
            groups=m,
        )
        return y.reshape(t, m, out_channels, h, w).permute(1, 0, 2, 3, 4)

    def forward_grouped(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "History ensemble oczekuje wejścia [MODELE, STOŁY, 3, H, W]"
            )
        expected_tail = (self.input_channels, self.board_size, self.board_size)
        if x.shape[0] != self.num_models or tuple(x.shape[2:]) != expected_tail:
            raise ValueError(
                f"Niepoprawny history batch {tuple(x.shape)}; oczekiwano "
                f"[{self.num_models}, T, {expected_tail[0]}, {expected_tail[1]}, {expected_tail[2]}]"
            )

        if x.device != self.device or x.dtype != self.dtype:
            x = x.to(device=self.device, dtype=self.dtype, non_blocking=True)

        for kernel, weight, bias in zip(
            self.kernels, self.conv_weights, self.conv_biases
        ):
            x = self._grouped_conv(x, weight, bias, kernel // 2)
            x = F.silu(x)

        logits = self._grouped_conv(
            x,
            self.policy_weight,
            self.policy_bias,
            padding=0,
        )
        return logits.squeeze(2).flatten(2)
