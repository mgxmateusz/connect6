from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from .checkpoint import list_versioned_checkpoints, load_model_for_inference


_UPDATE_RE = re.compile(r"model_update_(\d+)\.pt$")


@dataclass(slots=True)
class HistoricalModel:
    path: Path
    update: int
    model: torch.nn.Module


def checkpoint_update(path: str | Path) -> int | None:
    match = _UPDATE_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


def load_random_historical_models(
    checkpoint_dir: str | Path,
    *,
    current_update: int,
    requested_count: int,
    device: torch.device,
) -> list[HistoricalModel]:
    """Losuje i ładuje zamrożone stare checkpointy do inference.

    Bierzemy wyłącznie versioned checkpointy starsze od bieżącego update'u.
    `latest.pt` nie bierze udziału w losowaniu. Jeśli kompatybilnych checkpointów
    jest mniej niż requested_count, zwracamy wszystkie dostępne kompatybilne.
    """

    requested_count = max(0, int(requested_count))
    if requested_count == 0:
        return []

    candidates: list[Path] = []
    for path in list_versioned_checkpoints(checkpoint_dir):
        update = checkpoint_update(path)
        if update is not None and update < int(current_update):
            candidates.append(Path(path))

    random.shuffle(candidates)

    loaded: list[HistoricalModel] = []
    for path in candidates:
        if len(loaded) >= requested_count:
            break

        update = checkpoint_update(path)
        if update is None:
            continue

        try:
            model, payload = load_model_for_inference(path, device=device)
        except (RuntimeError, KeyError, ValueError) as exc:
            print(f"[history] pomijam {path.name}: {exc}")
            continue

        # Payload może zawierać duży optimizer_state. Nie trzymamy go w pamięci.
        del payload

        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        loaded.append(
            HistoricalModel(
                path=path,
                update=update,
                model=model,
            )
        )

    return loaded
