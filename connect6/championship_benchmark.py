from __future__ import annotations

import argparse
from pathlib import Path

import torch

from . import championship_backend_tuner as backend_tuner
from . import championship_fast_tuner as fast_tuner
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
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}

    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("Benchmark championship wymaga CUDA")

    # Benchmarkujemy dokładnie finalną architekturę runtime: BMM + FP16 + fast env.
    backend_cls = backend_tuner.BMMIndexedEnsemble
    dtype_name = "float16"
    stream.DirectIndexedEnsemble = backend_cls
    FastChampionshipConnect6.network_dtype = torch.float16
    _legacy.VectorConnect6 = FastChampionshipConnect6

    print("\n" + "=" * 78)
    print("ONE-SHOT CHAMPIONSHIP BENCHMARK")
    print("=" * 78)
    print("Backend: BMM_IM2COL | dtype: float16 | engine: FastChampionshipConnect6")
    print("Ten benchmark NIE uruchamia turnieju i NIE zmienia championship.yaml.")

    assert_checkpoint_input_compatibility(device, dtype=torch.float16)
    assert_gameplay_compatibility(device)
    print("[COMPAT] checkpoint input: OK | gameplay: OK")

    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(root, ch["checkpoint_dir"])
    refs = _legacy.discover_checkpoints(checkpoint_dir)
    if len(refs) < 2:
        raise RuntimeError("Brak wystarczającej liczby checkpointów")

    limit_gb = float(adaptive.get("vram_limit_gb", 11.0))
    base._VRAM_LIMIT_BYTES = int(limit_gb * 2**30)

    print(f"[LOAD] ładuję {len(refs)} modeli do VRAM...")
    store = base.CNNCheckpointStore(cpu_cache_models=len(refs))
    lean = [store.get(ref) for ref in refs]
    torch.cuda.empty_cache()
    ensemble = backend_cls(lean, device)
    resident._cast_ensemble_storage(ensemble, dtype_name)
    torch.cuda.synchronize(device)

    alloc = torch.cuda.memory_allocated(device) / 2**30
    reserved = torch.cuda.memory_reserved(device) / 2**30
    print(f"[LOAD] alloc={alloc:.2f} GB | reserved={reserved:.2f} GB")

    try:
        tuned_ch = dict(ch)
        tuned_ch["amp_dtype"] = dtype_name
        tables, refill, sync = fast_tuner.deep_fixed_corpus_autotune(
            ensemble,
            refs,
            ch=tuned_ch,
            adaptive=adaptive,
        )

        print("\n" + "=" * 78)
        print("BENCHMARK COMPLETE — WPISZ TO DO configs/championship.yaml")
        print("=" * 78)
        print("championship:")
        print(f"  tables: {tables}")
        print("  adaptive_tables:")
        print(f"    sync_interval_moves: {sync}")
        print(f"    refill_batch: {refill}")
        print("\nBackend pozostaje na stałe: BMM_IM2COL + float16")
        print("Po wpisaniu tych 3 wartości używaj tylko: python run_championship.py")
    finally:
        ensemble.release()
        del ensemble
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jednorazowy benchmark konfiguracji championship"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
