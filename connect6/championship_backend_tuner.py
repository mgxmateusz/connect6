from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from . import championship_stream as stream

base = stream.base
_legacy = stream._legacy


class BMMIndexedEnsemble(stream.DirectIndexedEnsemble):
    """Per-plansza CNN przez im2col + batched GEMM.

    Każda aktywna plansza może używać innego modelu. Zamiast conv2d(groups=B)
    wybieramy B zestawów wag i wykonujemy B niezależnych mnożeń macierzy jednym
    torch.bmm. To jest workload bliższy Tensor Cores i na części GPU jest
    znacznie szybszy od bardzo wysokiego groups w cuDNN.
    """

    @staticmethod
    def _indexed_conv(
        x: torch.Tensor,
        weights: torch.Tensor,
        biases: torch.Tensor | None,
        model_ids: torch.Tensor,
        *,
        padding: int,
    ) -> torch.Tensor:
        bsz, in_channels, height, width = x.shape
        selected_w = weights.index_select(0, model_ids)
        out_channels = int(selected_w.shape[1])
        kernel = int(selected_w.shape[-1])

        if kernel == 1:
            patches = x.flatten(2)
        else:
            patches = F.unfold(
                x,
                kernel_size=(kernel, kernel),
                padding=padding,
                stride=1,
            )

        matrix_w = selected_w.reshape(
            bsz,
            out_channels,
            in_channels * kernel * kernel,
        )
        y = torch.bmm(matrix_w, patches)
        if biases is not None:
            selected_b = biases.index_select(0, model_ids)
            y = y + selected_b.unsqueeze(-1)
        return y.reshape(bsz, out_channels, height, width)


def _project(config_path: Path, cfg: dict[str, Any]) -> tuple[list[Any], Path]:
    ch = cfg.get("championship", cfg)
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(root, ch["checkpoint_dir"])
    output_dir = _legacy._resolve_path(root, ch["output_dir"])
    return _legacy.discover_checkpoints(checkpoint_dir), output_dir


def choose_backend_and_dtype(
    config_path: str | Path,
    cfg: dict[str, Any],
) -> tuple[type[stream.DirectIndexedEnsemble], str]:
    """Krótki realny benchmark backendu przed głębokim tuningiem schedulera."""
    from . import championship_super_tuner as super_tuner

    config_path = Path(config_path).resolve()
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}
    tune = adaptive.get("backend_autotune", {}) or {}
    if not bool(tune.get("enabled", True)):
        return stream.DirectIndexedEnsemble, str(ch.get("amp_dtype", "bfloat16"))

    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    if device.type != "cuda":
        return stream.DirectIndexedEnsemble, str(ch.get("amp_dtype", "bfloat16"))

    refs, _ = _project(config_path, cfg)
    if len(refs) < 2:
        return stream.DirectIndexedEnsemble, str(ch.get("amp_dtype", "bfloat16"))

    tables_list = [
        int(v)
        for v in tune.get("tables", [96, 112, 128, 144])
        if 16 <= int(v) <= len(refs)
    ]
    if not tables_list:
        tables_list = [min(128, len(refs))]
    block = max(2, int(tune.get("model_block", 32)))
    refill = max(1, int(tune.get("refill", 24)))
    sync = max(1, int(tune.get("sync", 1)))
    warmup_games = max(32, int(tune.get("warmup_games", 64)))
    measure_games = max(128, int(tune.get("measure_games", 384)))
    max_seconds = max(2.0, float(tune.get("max_seconds_per_test", 8.0)))
    dtypes = [str(v) for v in tune.get("amp_dtypes", ["bfloat16", "float16"])]

    limit_gb = float(adaptive.get("vram_limit_gb", 11.0))
    limit_bytes = int(limit_gb * 2**30)
    games_per_pair = int(ch.get("games_per_pair", 2))
    temperature = float(ch.get("temperature", 0.0))
    seed = int(ch.get("seed", 12345))

    model_count = min(len(refs), max(2, 2 * block))
    store = base.CNNCheckpointStore(cpu_cache_models=model_count)
    lean = [store.get(ref) for ref in refs[:model_count]]

    candidates: list[tuple[str, type[stream.DirectIndexedEnsemble]]] = [
        ("GROUPED_CONV", stream.DirectIndexedEnsemble),
        ("BMM_IM2COL", BMMIndexedEnsemble),
    ]
    original_cls = stream.DirectIndexedEnsemble
    results: list[tuple[float, str, str, type[stream.DirectIndexedEnsemble], int, float]] = []

    print("\n" + "=" * 78)
    print("CNN INFERENCE BACKEND AUTOTUNE")
    print("=" * 78)
    print(
        f"Porównuję grouped-conv vs im2col+BMM | block={block} | "
        f"refill={refill} | sync={sync}"
    )

    try:
        for backend_name, backend_cls in candidates:
            stream.DirectIndexedEnsemble = backend_cls
            for dtype_name in dtypes:
                if dtype_name not in ("bfloat16", "float16"):
                    continue
                local_results: list[tuple[float, int, float]] = []
                for tables in tables_list:
                    try:
                        r = super_tuner._bench_stream_config(
                            lean,
                            tables=tables,
                            refill=min(refill, tables),
                            sync=sync,
                            temperature=temperature,
                            amp=True,
                            amp_dtype=dtype_name,
                            seed=seed + tables * 101,
                            warmup_games=warmup_games,
                            measure_games=measure_games,
                            max_seconds=max_seconds,
                            vram_limit_bytes=limit_bytes,
                            real_block_size=block,
                            games_per_pair=games_per_pair,
                        )
                    except (base.AdaptiveBatchResize, torch.cuda.OutOfMemoryError) as exc:
                        print(
                            f"[BACKEND] {backend_name:>12} {dtype_name:>8} "
                            f"T={tables:>3} -> UNSAFE: {exc}"
                        )
                        torch.cuda.empty_cache()
                        continue
                    local_results.append(
                        (r.adjusted_games_per_second, tables, r.peak_vram_gb)
                    )
                    print(
                        f"[BACKEND] {backend_name:>12} {dtype_name:>8} "
                        f"T={tables:>3} | {r.adjusted_games_per_second:>7.1f} gry/s | "
                        f"VRAM {r.peak_vram_gb:.2f} GB"
                    )
                if local_results:
                    best_local, best_tables, best_peak = max(
                        local_results, key=lambda item: item[0]
                    )
                    results.append(
                        (
                            best_local,
                            backend_name,
                            dtype_name,
                            backend_cls,
                            best_tables,
                            best_peak,
                        )
                    )
    finally:
        stream.DirectIndexedEnsemble = original_cls

    if not results:
        print("[BACKEND] brak poprawnego alternatywnego wyniku -> zostaje grouped-conv")
        return original_cls, str(ch.get("amp_dtype", "bfloat16"))

    results.sort(key=lambda item: item[0], reverse=True)
    score, name, dtype_name, cls, local_tables, peak = results[0]
    print("-" * 78)
    print(
        f"[BACKEND WINNER] {name} + {dtype_name} | lokalnie T={local_tables} | "
        f"{score:.1f} gry/s | peak {peak:.2f} GB"
    )
    print("-" * 78 + "\n")
    return cls, dtype_name


def run(config_path: str | Path) -> None:
    from . import championship_super_tuner as super_tuner

    config_path = Path(config_path).resolve()
    cfg = base._ORIGINAL_READ_YAML(config_path)
    backend_cls, dtype_name = choose_backend_and_dtype(config_path, cfg)

    # Wszystkie dalsze benchmarki i właściwy turniej korzystają już z wybranego backendu.
    stream.DirectIndexedEnsemble = backend_cls

    # super_tuner.run czyta YAML ponownie, więc tymczasowo podmieniamy reader, aby
    # zwycięski dtype trafił także do jego benchmarków bez modyfikowania pliku.
    original_reader = base._ORIGINAL_READ_YAML

    def patched_reader(path: Path) -> dict[str, Any]:
        data = copy.deepcopy(original_reader(path))
        section = data.get("championship", data)
        section["amp_dtype"] = dtype_name
        return data

    base._ORIGINAL_READ_YAML = patched_reader
    try:
        super_tuner.run(config_path)
    finally:
        base._ORIGINAL_READ_YAML = original_reader


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Connect6 championship — backend + scheduler autotune"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
