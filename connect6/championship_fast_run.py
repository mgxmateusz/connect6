from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch

from . import championship_backend_tuner as backend_tuner
from . import championship_resident as resident
from . import championship_stream as stream
from .championship_fast_env import (
    FastChampionshipConnect6,
    assert_checkpoint_input_compatibility,
    assert_gameplay_compatibility,
)

base = stream.base
_legacy = stream._legacy


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = base._ORIGINAL_READ_YAML(config_path)

    # Championship ma stały backend. Żadnego backend autotuningu przy starcie.
    backend_cls = backend_tuner.BMMIndexedEnsemble
    dtype_name = "float16"
    stream.DirectIndexedEnsemble = backend_cls
    FastChampionshipConnect6.network_dtype = torch.float16

    device = _legacy._torch_device(
        str(cfg.get("championship", cfg).get("device", "cuda"))
    )
    print("\n" + "=" * 78)
    print("FAST CHAMPIONSHIP ENGINE COMPATIBILITY")
    print("=" * 78)
    assert_checkpoint_input_compatibility(device, dtype=torch.float16)
    assert_gameplay_compatibility(device)
    print(
        "[FAST ENGINE] input checkpointów: OK | zasady/turn state: OK | "
        "backend=BMM_IM2COL | dtype=float16"
    )

    # Treningowy VectorConnect6 pozostaje nietknięty; tylko championship używa fast env.
    _legacy.VectorConnect6 = FastChampionshipConnect6

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
            "Fast all-resident championship nie może wystartować — modele nie "
            "zmieściły się w skonfigurowanym limicie VRAM lub CUDA jest niedostępna."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6 championship — fixed BMM FP16 fast engine"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
