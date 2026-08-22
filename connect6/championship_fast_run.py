from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from . import championship_backend_tuner as backend_tuner
from . import championship_resident as resident
from . import championship_resident_only as resident_only
from . import championship_stream as stream
from .championship_fast_env import (
    FastChampionshipConnect6,
    assert_checkpoint_input_compatibility,
    assert_gameplay_compatibility,
)

base = stream.base
_legacy = stream._legacy


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = base._ORIGINAL_READ_YAML(config_path)

    # Backend benchmark pozostaje na referencyjnym środowisku, aby wybór
    # grouped-conv/BMM nie był mieszany z optymalizacją silnika gry.
    backend_cls, dtype_name = backend_tuner.choose_backend_and_dtype(config_path, cfg)
    stream.DirectIndexedEnsemble = backend_cls

    dtype = _dtype_from_name(dtype_name)
    FastChampionshipConnect6.network_dtype = dtype

    device = _legacy._torch_device(
        str(cfg.get("championship", cfg).get("device", "cuda"))
    )
    print("\n" + "=" * 78)
    print("FAST CHAMPIONSHIP ENGINE COMPATIBILITY")
    print("=" * 78)
    assert_checkpoint_input_compatibility(device, dtype=dtype)
    assert_gameplay_compatibility(device)
    print(
        "[FAST ENGINE] input checkpointów: OK | zasady/turn state: OK | "
        f"network dtype={dtype_name}"
    )

    # Od tego miejsca wszystkie schedulery championship tworzą już wyłącznie
    # wyspecjalizowany FastChampionshipConnect6. Treningowy VectorConnect6 nie jest
    # zmieniany w repo ani w procesie treningowym.
    _legacy.VectorConnect6 = FastChampionshipConnect6
    resident._benchmark_resident_scheduler = resident_only._deep_resident_autotune

    tuned_cfg = copy.deepcopy(cfg)
    ch = tuned_cfg.get("championship", tuned_cfg)
    ch["amp_dtype"] = dtype_name

    ok = resident.run_resident(
        config_path,
        tuned_cfg,
        backend_cls=backend_cls,
        dtype_name=dtype_name,
    )
    if not ok:
        raise RuntimeError(
            "All-resident fast championship jest wymagany, ale modele nie "
            "zmieściły się w skonfigurowanym limicie VRAM."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6 championship — fast checkpoint-compatible engine"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
