from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import championship as _legacy
from .checkpoint import load_checkpoint
from .history import HistoricalCheckpoint, HistoricalPolicyEnsemble


def _normalized_model_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    out = dict(cfg)
    out.pop("compile", None)
    out.pop("compile_mode", None)
    return out


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
    """Równoległy ensemble checkpointów CNN dla turnieju.

    Wiele modeli jest liczone jednym grouped-conv na warstwę. Gdy liczba pozycji
    przypisana do poszczególnych modeli jest różna, batch jest dopełniany do
    wspólnej szerokości, a nieużywane wyniki są ignorowane.
    """

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
        self._ensemble = HistoricalPolicyEnsemble(
            historical,
            device=device,
            dtype=torch.float32,
        )
        self.action_size = self.board_size * self.board_size

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[0] != self.num_models:
            raise ValueError(
                f"forward_all wymaga {self.num_models} pozycji, otrzymano {x.shape[0]}."
            )
        indices = torch.arange(self.num_models, device=x.device, dtype=torch.long)
        return self.forward_indexed(x, indices)

    def forward_indexed(
        self,
        x: torch.Tensor,
        model_indices: torch.Tensor,
    ) -> torch.Tensor:
        model_indices = model_indices.to(self.device, dtype=torch.long, non_blocking=True)
        if x.shape[0] != model_indices.numel():
            raise ValueError("Liczba pozycji i model_indices musi być taka sama.")
        if x.ndim != 4:
            raise ValueError(
                "Championship CNN oczekuje wejścia [B, 3, H, W], "
                f"otrzymano {tuple(x.shape)}."
            )
        if x.shape[0] == 0:
            return torch.empty(
                (0, self.action_size),
                device=self.device,
                dtype=torch.float32,
            )
        if bool((model_indices < 0).any()) or bool((model_indices >= self.num_models).any()):
            raise IndexError("model_indices zawiera indeks spoza ensemble.")

        groups = [
            torch.nonzero(model_indices.eq(i), as_tuple=False).flatten()
            for i in range(self.num_models)
        ]
        max_tables = max(int(group.numel()) for group in groups)

        padded = torch.zeros(
            (
                self.num_models,
                max_tables,
                x.shape[1],
                x.shape[2],
                x.shape[3],
            ),
            device=self.device,
            dtype=x.dtype,
        )
        for model_id, group in enumerate(groups):
            if group.numel() > 0:
                padded[model_id, : group.numel()].copy_(x.index_select(0, group))

        grouped_logits = self._ensemble.forward_grouped(padded)
        result = torch.empty(
            (x.shape[0], self.action_size),
            device=self.device,
            dtype=grouped_logits.dtype,
        )
        for model_id, group in enumerate(groups):
            if group.numel() > 0:
                result.index_copy_(
                    0,
                    group,
                    grouped_logits[model_id, : group.numel()],
                )
        return result

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
        policy_bias = (
            None
            if ensemble.policy_bias is None
            else ensemble.policy_bias[idx]
        )
        return F.conv2d(x, policy_weight, policy_bias).squeeze(1).flatten(1)

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


# Zachowujemy sprawdzoną logikę parowania, zapisu CSV/HTML i wznawiania
# championship, podmieniając wyłącznie backend checkpointów i inference.
_legacy.CheckpointStore = CNNCheckpointStore
_legacy.BatchedPolicyEnsemble = CNNBatchedPolicyEnsemble


def main() -> None:
    _legacy.main()


if __name__ == "__main__":
    main()
