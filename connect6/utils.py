from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def newest_checkpoint_under(runs_dir: str | Path) -> Path | None:
    candidates = list(Path(runs_dir).glob("*/checkpoints/latest.pt"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
