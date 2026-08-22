from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import torch

from . import championship_stream as stream

base = stream.base
_legacy = stream._legacy


@dataclass(slots=True)
class TuneResult:
    tables: int
    model_block: int
    refill: int
    sync: int
    games_per_second: float
    peak_vram_gb: float
    load_seconds: float
    adjusted_games_per_second: float


def _project_paths(config_path: Path, cfg: dict[str, Any]) -> tuple[Path, Path, list[Any]]:
    ch = cfg.get("championship", cfg)
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(root, ch["checkpoint_dir"])
    output_dir = _legacy._resolve_path(root, ch["output_dir"])
    refs = _legacy.discover_checkpoints(checkpoint_dir)
    return checkpoint_dir, output_dir, refs


def _unique_ints(values: list[int], *, lo: int, hi: int) -> list[int]:
    return sorted({int(v) for v in values if lo <= int(v) <= hi})


def _make_jobs(refs: list[Any], count: int) -> list[stream.GameJob]:
    n = len(refs)
    if n < 2:
        raise ValueError("Benchmark schedulera wymaga co najmniej 2 modeli.")
    jobs: list[stream.GameJob] = []
    for i in range(count):
        a_slot = i % n
        b_slot = (i * 17 + 7) % n
        if b_slot == a_slot:
            b_slot = (b_slot + 1) % n
        jobs.append(
            stream.GameJob(
                a=refs[a_slot],
                b=refs[b_slot],
                game_index=i & 1,
                a_slot=a_slot,
                b_slot=b_slot,
            )
        )
    return jobs


def _bench_stream_config(
    lean: list[Any],
    *,
    tables: int,
    refill: int,
    sync: int,
    temperature: float,
    amp: bool,
    amp_dtype: str,
    seed: int,
    warmup_games: int,
    measure_games: int,
    max_seconds: float,
    vram_limit_bytes: int,
    real_block_size: int,
    games_per_pair: int,
) -> TuneResult:
    device = torch.device("cuda")
    base._VRAM_LIMIT_BYTES = int(vram_limit_bytes)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    load_started = time.perf_counter()
    ensemble: stream.DirectIndexedEnsemble | None = None
    try:
        ensemble = stream.DirectIndexedEnsemble(lean, device)
        torch.cuda.synchronize(device)
        load_seconds = time.perf_counter() - load_started

        scheduler = stream.GlobalTableScheduler(
            tables,
            ensemble,
            temperature=temperature,
            amp=amp,
            amp_dtype=amp_dtype,
            seed=seed,
        )
        # Zapas jobów większy niż benchmark, żeby refill nigdy nie zabrakł pracy.
        total_needed = tables + warmup_games + measure_games + tables * 4
        jobs = _make_jobs([cp.ref for cp in lean], total_needed)
        pending = min(tables, len(jobs))
        scheduler.refill(list(range(pending)), jobs[:pending])

        finished_total = 0
        measured = 0
        measure_started: float | None = None
        hard_started = time.perf_counter()

        while measured < measure_games:
            finished, free = scheduler.step(sync)
            if finished:
                finished_total += len(finished)
                if measure_started is None and finished_total >= warmup_games:
                    torch.cuda.synchronize(device)
                    measure_started = time.perf_counter()
                    measured = 0
                elif measure_started is not None:
                    measured += len(finished)

            remaining = len(jobs) - pending
            if remaining > 0 and (len(free) >= refill or scheduler.active_count == 0):
                count = min(len(free), remaining)
                if count:
                    scheduler.refill(free[:count], jobs[pending : pending + count])
                    pending += count

            if time.perf_counter() - hard_started >= max_seconds:
                break

        torch.cuda.synchronize(device)
        if measure_started is None:
            measured_seconds = max(1e-9, time.perf_counter() - hard_started)
            measured = max(1, finished_total)
        else:
            measured_seconds = max(1e-9, time.perf_counter() - measure_started)
            measured = max(1, measured)

        raw_gps = measured / measured_seconds
        peak = torch.cuda.max_memory_reserved(device)
        if peak > vram_limit_bytes:
            raise base.AdaptiveBatchResize(
                f"peak {peak / 2**30:.2f} GB przekracza limit {vram_limit_bytes / 2**30:.2f} GB"
            )

        # W realnym cross-poolu blok B daje B*B par. Koszt przeładowania modeli
        # amortyzujemy więc przez B^2 * games_per_pair gier.
        games_per_pool = max(1, real_block_size * real_block_size * games_per_pair)
        sec_per_game = 1.0 / max(raw_gps, 1e-9) + load_seconds / games_per_pool
        adjusted = 1.0 / sec_per_game
        return TuneResult(
            tables=tables,
            model_block=real_block_size,
            refill=refill,
            sync=sync,
            games_per_second=raw_gps,
            peak_vram_gb=peak / 2**30,
            load_seconds=load_seconds,
            adjusted_games_per_second=adjusted,
        )
    finally:
        if ensemble is not None:
            ensemble.release()
        del ensemble
        torch.cuda.empty_cache()


def _print_result(prefix: str, r: TuneResult) -> None:
    print(
        f"{prefix} T={r.tables:>3} block={r.model_block:>3} refill={r.refill:>3} "
        f"sync={r.sync:>2} | {r.games_per_second:>7.1f} gry/s | "
        f"adj {r.adjusted_games_per_second:>7.1f} | load {r.load_seconds:.3f}s | "
        f"VRAM {r.peak_vram_gb:.2f} GB"
    )


def autotune_scheduler(
    config_path: Path,
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], base.AdaptiveTableController, list[int]]:
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}
    tune = adaptive.get("scheduler_autotune", {}) or {}
    _, output_dir, refs = _project_paths(config_path, cfg)

    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    if device.type != "cuda" or not bool(tune.get("enabled", True)):
        tables, controller, candidates = stream.autotune_three_tests(config_path, cfg, adaptive)
        selected = {
            "tables": tables,
            "model_pool_block_size": int(adaptive.get("model_pool_block_size", 64)),
            "refill_batch": int(adaptive.get("refill_batch", 32)),
            "sync_interval_moves": int(adaptive.get("sync_interval_moves", 8)),
        }
        return selected, controller, candidates

    limit_gb = float(adaptive.get("vram_limit_gb", 11.0))
    limit_bytes = int(limit_gb * 2**30)
    gpu_name = torch.cuda.get_device_name(device)

    table_candidates = _unique_ints(
        list(tune.get("table_candidates", [64, 80, 96, 112, 128, 144, 160, 192, 224, 256])),
        lo=16,
        hi=min(2048, len(refs)),
    )
    block_candidates = _unique_ints(
        list(tune.get("model_block_candidates", [32, 48, 64, 96, 128, 160, 192, 256])),
        lo=2,
        hi=max(2, len(refs) // 2),
    )
    refill_candidates_raw = [int(v) for v in tune.get("refill_candidates", [8, 16, 24, 32, 48, 64])]
    sync_candidates = _unique_ints(
        list(tune.get("sync_candidates", [1, 2, 4, 6, 8, 12, 16])),
        lo=1,
        hi=64,
    )
    warmup_games = max(16, int(tune.get("warmup_games", 64)))
    measure_games = max(64, int(tune.get("measure_games", 384)))
    max_seconds = max(2.0, float(tune.get("max_seconds_per_test", 8.0)))
    validation_games = max(measure_games, int(tune.get("validation_games", 768)))
    games_per_pair = int(ch.get("games_per_pair", 2))
    temperature = float(ch.get("temperature", 0.0))
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    seed = int(ch.get("seed", 12345))

    controller = base.AdaptiveTableController(
        output_dir / "adaptive_tables.json",
        gpu_name=gpu_name,
        limit_bytes=limit_bytes,
        min_tables=min(table_candidates),
        max_tables=max(table_candidates),
    )
    table_candidates = [v for v in table_candidates if controller.is_allowed(v)]
    if not table_candidates:
        raise RuntimeError("Brak bezpiecznych kandydatów liczby stołów.")

    # Ładujemy do RAM największy potrzebny zestaw raz. Poszczególne testy tylko
    # składają z niego ensemble GPU; dysk nie zaburza porównania throughputu.
    max_models = min(len(refs), 2 * max(block_candidates))
    store = base.CNNCheckpointStore(cpu_cache_models=max_models)
    lean_all = [store.get(ref) for ref in refs[:max_models]]

    baseline_block = min(block_candidates, key=lambda v: abs(v - int(adaptive.get("model_pool_block_size", 64))))
    baseline_refill = int(adaptive.get("refill_batch", 32))
    baseline_sync = int(adaptive.get("sync_interval_moves", 8))

    all_results: list[TuneResult] = []
    unsafe_configs: list[dict[str, int]] = []

    def test(tables: int, block: int, refill: int, sync: int, *, games: int | None = None) -> TuneResult | None:
        refill = max(1, min(tables, refill))
        model_count = min(len(lean_all), max(2, 2 * block))
        lean = lean_all[:model_count]
        try:
            r = _bench_stream_config(
                lean,
                tables=tables,
                refill=refill,
                sync=sync,
                temperature=temperature,
                amp=amp,
                amp_dtype=amp_dtype,
                seed=seed + tables * 1009 + block * 31 + refill * 7 + sync,
                warmup_games=warmup_games,
                measure_games=games or measure_games,
                max_seconds=max_seconds if games is None else max_seconds * 2.0,
                vram_limit_bytes=limit_bytes,
                real_block_size=block,
                games_per_pair=games_per_pair,
            )
            all_results.append(r)
            _print_result("[TUNE]", r)
            return r
        except (base.AdaptiveBatchResize, torch.cuda.OutOfMemoryError) as exc:
            unsafe_configs.append({"tables": tables, "block": block, "refill": refill, "sync": sync})
            print(f"[TUNE] T={tables} block={block} refill={refill} sync={sync} -> UNSAFE: {exc}")
            torch.cuda.empty_cache()
            return None

    print("\n" + "=" * 78)
    print("DEEP GPU/SCHEDULER AUTOTUNE")
    print("=" * 78)
    print(
        f"GPU: {gpu_name} | VRAM limit {limit_gb:.2f} GB | "
        f"warmup {warmup_games} gier | pomiar {measure_games} gier/test"
    )

    # ETAP 1: gęsty sweep liczby stołów na realnym persistent schedulerze.
    print("\n[TUNE 1/5] liczba stołów")
    table_results: list[TuneResult] = []
    for tables in table_candidates:
        r = test(tables, baseline_block, min(baseline_refill, tables), baseline_sync)
        if r is not None:
            table_results.append(r)
    if not table_results:
        raise RuntimeError("Autotuner nie znalazł działającej liczby stołów.")
    table_results.sort(key=lambda r: r.adjusted_games_per_second, reverse=True)
    best_tables = table_results[0].tables

    # ETAP 2: pula modeli może być znacznie większa od liczby stołów.
    print("\n[TUNE 2/5] model_pool_block_size")
    block_results: list[TuneResult] = []
    for block in block_candidates:
        r = test(best_tables, block, min(baseline_refill, best_tables), baseline_sync)
        if r is not None:
            block_results.append(r)
    block_results.sort(key=lambda r: r.adjusted_games_per_second, reverse=True)
    best_block = block_results[0].model_block

    # ETAP 3: próg refill.
    print("\n[TUNE 3/5] refill_batch")
    refill_candidates = _unique_ints(refill_candidates_raw + [best_tables // 8, best_tables // 4, best_tables // 2], lo=1, hi=best_tables)
    refill_results: list[TuneResult] = []
    for refill in refill_candidates:
        r = test(best_tables, best_block, refill, baseline_sync)
        if r is not None:
            refill_results.append(r)
    refill_results.sort(key=lambda r: r.adjusted_games_per_second, reverse=True)
    best_refill = refill_results[0].refill

    # ETAP 4: częstotliwość synchronizacji CPU/GPU.
    print("\n[TUNE 4/5] sync_interval_moves")
    sync_results: list[TuneResult] = []
    for sync in sync_candidates:
        r = test(best_tables, best_block, best_refill, sync)
        if r is not None:
            sync_results.append(r)
    sync_results.sort(key=lambda r: r.adjusted_games_per_second, reverse=True)
    best_sync = sync_results[0].sync

    # ETAP 5: walidacja kombinacji z top-3 stołów i top-3 bloków. To łapie
    # interakcje, których coordinate descent mógł nie zauważyć.
    print("\n[TUNE 5/5] walidacja top konfiguracji")
    finalists: list[TuneResult] = []
    top_tables = [r.tables for r in table_results[:3]]
    top_blocks = [r.model_block for r in block_results[:3]]
    seen: set[tuple[int, int, int, int]] = set()
    for tables in top_tables:
        for block in top_blocks:
            refill = min(best_refill, tables)
            key = (tables, block, refill, best_sync)
            if key in seen:
                continue
            seen.add(key)
            r = test(tables, block, refill, best_sync, games=validation_games)
            if r is not None:
                finalists.append(r)

    pool = finalists if finalists else all_results
    best = max(pool, key=lambda r: r.adjusted_games_per_second)
    selected = {
        "tables": best.tables,
        "model_pool_block_size": best.model_block,
        "refill_batch": best.refill,
        "sync_interval_moves": best.sync,
    }

    state = {
        "version": 2,
        "gpu": gpu_name,
        "vram_limit_gb": limit_gb,
        "selected": selected,
        "best": asdict(best),
        "unsafe_configs": unsafe_configs,
        "results": [asdict(r) for r in all_results],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "scheduler_autotune.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    controller.last_selected = best.tables
    controller.save()
    print("\n" + "-" * 78)
    print(
        f"[TUNE WINNER] tables={best.tables} | block={best.model_block} | "
        f"refill={best.refill} | sync={best.sync} | "
        f"{best.adjusted_games_per_second:.1f} adjusted gry/s"
    )
    print("-" * 78 + "\n")
    return selected, controller, table_candidates


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = base._ORIGINAL_READ_YAML(config_path)
    selected, controller, table_candidates = autotune_scheduler(config_path, cfg)

    tuned_cfg = copy.deepcopy(cfg)
    ch = tuned_cfg.get("championship", tuned_cfg)
    adaptive = ch.setdefault("adaptive_tables", {})
    adaptive["model_pool_block_size"] = selected["model_pool_block_size"]
    adaptive["refill_batch"] = selected["refill_batch"]
    adaptive["sync_interval_moves"] = selected["sync_interval_moves"]

    tables = int(selected["tables"])
    while True:
        try:
            controller.last_selected = tables
            controller.save()
            stream.run_streaming_once(config_path, tuned_cfg, tables=tables, controller=controller)
            return
        except base.AdaptiveBatchResize as exc:
            controller.mark_unsafe(tables)
            lower = [v for v in table_candidates if v < tables and controller.is_allowed(v)]
            if not lower:
                raise
            new_tables = max(lower)
            print(
                f"\n[ADAPTIVE GPU] runtime odrzucił {tables} stołów: {exc}. "
                f"Wznawiam z {new_tables}.\n"
            )
            tables = new_tables
            torch.cuda.empty_cache()
