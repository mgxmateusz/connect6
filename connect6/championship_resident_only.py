from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Any

import torch

from . import championship_backend_tuner as backend_tuner
from . import championship_resident as resident
from . import championship_stream as stream

base = stream.base
_legacy = stream._legacy


def _deep_resident_autotune(
    ensemble: stream.DirectIndexedEnsemble,
    refs: list[Any],
    *,
    ch: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[int, int, int]:
    """Długi autotuner resident-only.

    Najpierw przeszukuje szeroki zakres liczby stołów, głównie wielokrotności 32.
    Następnie stroi refill i sync, a na końcu ponownie mierzy top-3 konfiguracje
    na znacznie dłuższym przebiegu. Długość pomiaru skaluje się z liczbą stołów,
    aby duże batch-e nie były oceniane na mniej niż kilku pełnych falach gier.
    """
    tune = adaptive.get("resident_autotune", {}) or {}
    table_candidates = [
        int(v)
        for v in tune.get(
            "table_candidates",
            [
                64, 96, 128, 160, 192, 224, 256,
                320, 384, 448, 512, 640, 768,
                896, 1024, 1280, 1536, 1792, 2048,
            ],
        )
        if 2 <= int(v) <= len(refs)
    ]
    table_candidates = sorted(set(table_candidates))
    refill_candidates_cfg = [
        int(v)
        for v in tune.get(
            "refill_candidates",
            [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        )
    ]
    sync_candidates = sorted(
        set(int(v) for v in tune.get("sync_candidates", [1, 2, 4, 6, 8]))
    )
    warmup_games_cfg = max(64, int(tune.get("warmup_games", 256)))
    measure_games_cfg = max(256, int(tune.get("measure_games", 2048)))
    waves = max(2, int(tune.get("minimum_measure_waves", 4)))
    validation_waves = max(waves, int(tune.get("validation_waves", 10)))
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    temperature = float(ch.get("temperature", 0.0))
    seed = int(ch.get("seed", 12345))

    def bench(
        tables: int,
        refill: int,
        sync: int,
        *,
        validation: bool = False,
    ) -> float:
        tables = min(max(2, int(tables)), len(refs))
        refill = min(max(1, int(refill)), tables)
        local_waves = validation_waves if validation else waves
        warmup_games = max(warmup_games_cfg, tables * 2)
        measure_games = max(measure_games_cfg, tables * local_waves)

        scheduler = stream.GlobalTableScheduler(
            tables,
            ensemble,
            temperature=temperature,
            amp=amp,
            amp_dtype=amp_dtype,
            seed=seed + tables * 1009 + refill * 31 + sync + (99991 if validation else 0),
        )

        # Deterministyczny, identyczny wzorzec par dla każdego testu.
        need = tables + warmup_games + measure_games + tables * 3
        jobs: list[stream.GameJob] = []
        n = len(refs)
        k = 0
        while len(jobs) < need:
            a_slot = k % n
            b_slot = (k * 17 + 7) % n
            if a_slot == b_slot:
                b_slot = (b_slot + 1) % n
            jobs.append(
                stream.GameJob(
                    a=refs[a_slot],
                    b=refs[b_slot],
                    game_index=k & 1,
                    a_slot=a_slot,
                    b_slot=b_slot,
                )
            )
            k += 1

        pending = min(tables, len(jobs))
        scheduler.refill(list(range(pending)), jobs[:pending])
        finished_total = 0
        measured = 0
        start_measure: float | None = None

        while measured < measure_games:
            finished, free = scheduler.step(sync)
            if finished:
                finished_total += len(finished)
                if start_measure is None and finished_total >= warmup_games:
                    torch.cuda.synchronize(ensemble.device)
                    start_measure = time.perf_counter()
                    measured = 0
                elif start_measure is not None:
                    measured += len(finished)

            remaining = len(jobs) - pending
            if remaining > 0 and (len(free) >= refill or scheduler.active_count == 0):
                count = min(len(free), remaining)
                if count:
                    scheduler.refill(free[:count], jobs[pending : pending + count])
                    pending += count

            if pending >= len(jobs) and scheduler.active_count == 0:
                break

        torch.cuda.synchronize(ensemble.device)
        if start_measure is None or measured <= 0:
            return 0.0
        elapsed = max(1e-9, time.perf_counter() - start_measure)
        return measured / elapsed

    baseline_refill = int(adaptive.get("refill_batch", 32))
    baseline_sync = int(adaptive.get("sync_interval_moves", 1))

    print("\n" + "=" * 78)
    print("ALL-RESIDENT DEEP SCHEDULER AUTOTUNE")
    print("=" * 78)
    print(
        "Szeroki sweep tables (wielokrotności 32) + pomiar skalowany liczbą stołów"
    )

    table_scores: list[tuple[float, int]] = []
    for tables in table_candidates:
        score = bench(tables, min(baseline_refill, tables), baseline_sync)
        print(f"[RESIDENT TUNE] tables={tables:>4} | {score:>8.1f} gry/s")
        table_scores.append((score, tables))
    table_scores.sort(reverse=True)
    best_tables = table_scores[0][1]

    # Strojenie refill przy wybranej liczbie stołów. Dodajemy też ułamki batcha,
    # dzięki czemu dla dużych T nie zostajemy przy progach dobranych pod T=128.
    refill_candidates = set(refill_candidates_cfg)
    refill_candidates.update(
        max(1, best_tables // d) for d in (16, 12, 8, 6, 4, 3, 2)
    )
    refill_candidates = sorted(v for v in refill_candidates if 1 <= v <= best_tables)

    refill_scores: list[tuple[float, int]] = []
    for refill in refill_candidates:
        score = bench(best_tables, refill, baseline_sync)
        print(f"[RESIDENT TUNE] refill={refill:>4} | {score:>8.1f} gry/s")
        refill_scores.append((score, refill))
    refill_scores.sort(reverse=True)
    best_refill = refill_scores[0][1]

    sync_scores: list[tuple[float, int]] = []
    for sync in sync_candidates:
        score = bench(best_tables, best_refill, sync)
        print(f"[RESIDENT TUNE] sync={sync:>4} | {score:>8.1f} gry/s")
        sync_scores.append((score, sync))
    sync_scores.sort(reverse=True)
    best_sync = sync_scores[0][1]

    # Długi retest top-3 tables. Refill jest skalowany proporcjonalnie do batcha,
    # a sync pozostaje najlepszy z poprzedniego etapu.
    print("\n[RESIDENT TUNE] długi retest top-3 tables")
    finalists: list[tuple[float, int, int, int]] = []
    for _coarse_score, tables in table_scores[:3]:
        scaled_refill = max(1, round(best_refill * tables / best_tables))
        scaled_refill = min(scaled_refill, tables)
        score = bench(
            tables,
            scaled_refill,
            best_sync,
            validation=True,
        )
        print(
            f"[RESIDENT VALIDATE] tables={tables:>4} | refill={scaled_refill:>4} | "
            f"sync={best_sync:>2} | {score:>8.1f} gry/s"
        )
        finalists.append((score, tables, scaled_refill, best_sync))

    finalists.sort(reverse=True)
    best_score, best_tables, best_refill, best_sync = finalists[0]
    print(
        f"[RESIDENT WINNER] tables={best_tables} | refill={best_refill} | "
        f"sync={best_sync} | {best_score:.1f} gry/s\n"
    )
    return best_tables, best_refill, best_sync


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = base._ORIGINAL_READ_YAML(config_path)

    # Backend/dtype nadal wybieramy automatycznie, ale sam championship ma tylko
    # jedną ścieżkę wykonania: wszystkie modele resident w VRAM.
    backend_cls, dtype_name = backend_tuner.choose_backend_and_dtype(config_path, cfg)
    stream.DirectIndexedEnsemble = backend_cls
    resident._benchmark_resident_scheduler = _deep_resident_autotune

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
            "All-resident championship jest wymagany, ale wszystkie modele nie "
            "zmieściły się w skonfigurowanym limicie VRAM. Pool fallback został "
            "celowo usunięty."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6 championship — all models resident only"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
