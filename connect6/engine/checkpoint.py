from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

from .model import build_model


_UPDATE_RE = re.compile(r"model_update_(\d+)\.pt$")


class CheckpointManager:
    def __init__(self, checkpoint_dir: str | Path):
        self.dir = Path(checkpoint_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.latest_path = self.dir / "latest.pt"

    def save(
        self,
        update: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        config: dict[str, Any],
        global_step: int,
        scaler_state: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> Path:
        raw_model = getattr(model, "_orig_mod", model)
        payload = {
            "update": int(update),
            "global_step": int(global_step),
            "model_state": raw_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler_state,
            "config": config,
            "model_config": config["model"],
            "game_config": config["game"],
            "extra": extra or {},
            "torch_rng_state": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            payload["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()

        versioned = self.dir / f"model_update_{update:08d}.pt"
        self._atomic_torch_save(payload, versioned)
        self._atomic_torch_save(payload, self.latest_path)
        return versioned

    @staticmethod
    def _atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, target)

    def find_latest(self) -> Path | None:
        if self.latest_path.exists():
            return self.latest_path
        checkpoints = list_versioned_checkpoints(self.dir)
        return checkpoints[0] if checkpoints else None


def list_versioned_checkpoints(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    found: list[tuple[int, Path]] = []
    for path in directory.glob("model_update_*.pt"):
        m = _UPDATE_RE.search(path.name)
        if m:
            found.append((int(m.group(1)), path))
    found.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in found]


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def load_model_for_inference(
    path: str | Path, device: str | torch.device = "cuda"
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = load_checkpoint(path, map_location="cpu")
    game_cfg = payload["game_config"]
    model_cfg = payload["model_config"]
    if int(model_cfg.get("architecture_version", 0)) != 5:
        raise RuntimeError(
            f"Checkpoint {path} nie używa aktualnej architektury CNN v5."
        )
    model = build_model(model_cfg, int(game_cfg["board_size"]))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    model.eval()
    return model, payload
