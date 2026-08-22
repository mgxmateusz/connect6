from __future__ import annotations

import time
from typing import Any

import torch

from . import championship_stream as stream


def _fixed_jobs(refs: list[Any], count: int, *, offset: int = 0) -> list[stream.GameJob]:
    """Identyczny deterministyczny corpus dla wszystkich konfiguracji."""
    n = len(refs)
    jobs: list[stream.GameJob] = []
    for q in range(count):
        k = q + offset
        a_slot = (k * 13 + 3) % n
        b_slot = (k * 37 + 11) % n
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
    return jobs


def _run_closed_corpus(
    ensemble: stream.DirectIndexedEnsemble,
    jobs: list[stream.GameJob],
    filler: list[stream.GameJob],
    *,
    tables: int,
    refill: int,
    sync: int,
    temperature: float,
    amp: bool,
    amp_dtype: str,
    seed: int,
) -> float:
    """Mierzy dokładnie jobs; filler zapobiega drainowi po końcu corpus."""
    scheduler = stream.GlobalTableScheduler(
        tables,
        ensemble,
        temperature=temperature,
        amp=amp,
        amp_dtype=amp_dtype,
        seed=seed,
    )
    target_ids = {id(job) for job in jobs}
    queue = jobs + filler
    pending = min(tables, len(queue))
    scheduler.refill(list(range(pending)), queue[:pending])

    torch.cuda.synchronize(ensemble.device)
    started = time.perf_counter()
    target_finished = 0

    while target_finished < len(jobs):
        finished, free = scheduler.step(sync)
        if finished:
            for _slot, job, _winner in finished:
                if id(job) in target_ids:
                    target_finished += 1

        remaining = len(queue) - pending
        if remaining > 0 and (len(free) >= refill or scheduler.active_count == 0):
            count = min(len(free), remaining)
            if count:
                scheduler.refill(free[:count], queue[pending : pending + count])
                pending += count

    torch.cuda.synchronize(ensemble.device)
    elapsed = max(1e-9, time.perf_counter() - started)
    return len(jobs) / elapsed


def deep_fixed_corpus_autotune(
    ensemble: stream.DirectIndexedEnsemble,
    refs: list[Any],
    *,
    ch: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[int, int, int]:
    tune = adaptive.get("resident_autotune", {}) or {}
    table_candidates = sorted(
        {
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
        }
    )
    refill_cfg = [
        int(v)
        for v in tune.get(
            "refill_candidates",
            [8, 16, 24, 32, 48, 64, 96, 128, 192, 256],
        )
    ]
    sync_candidates = sorted(
        {int(v) for v in tune.get("sync_candidates", [1, 2, 4, 6, 8])}
    )
    warmup_count = max(512, int(tune.get("fixed_warmup_games", 2048)))
    coarse_count = max(2048, int(tune.get("fixed_measure_games", 4096)))
    validation_count = max(
        coarse_count, int(tune.get("fixed_validation_games", 12288))
    )
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    temperature = float(ch.get("temperature", 0.0))
    seed = int(ch.get("seed", 12345))
    baseline_refill = int(adaptive.get("refill_batch", 32))
    baseline_sync = int(adaptive.get("sync_interval_moves", 1))

    warmup_jobs = _fixed_jobs(refs, warmup_count, offset=0)
    coarse_jobs = _fixed_jobs(refs, coarse_count, offset=100_000)
    validation_jobs = _fixed_jobs(refs, validation_count, offset=300_000)
    filler_count = max(table_candidates) * 3
    filler_jobs = _fixed_jobs(refs, filler_count, offset=900_000)

    def warmup(tables: int, refill: int, sync: int) -> None:
        _run_closed_corpus(
            ensemble,
            warmup_jobs,
            filler_jobs,
            tables=tables,
            refill=min(refill, tables),
            sync=sync,
            temperature=temperature,
            amp=amp,
            amp_dtype=amp_dtype,
            seed=seed + 7,
        )

    def bench(
        tables: int,
        refill: int,
        sync: int,
        *,
        validation: bool = False,
    ) -> float:
        refill = min(max(1, int(refill)), tables)
        warmup(tables, refill, sync)
        corpus = validation_jobs if validation else coarse_jobs
        return _run_closed_corpus(
            ensemble,
            corpus,
            filler_jobs,
            tables=tables,
            refill=refill,
            sync=sync,
            temperature=temperature,
            amp=amp,
            amp_dtype=amp_dtype,
            seed=seed + 17,
        )

    print("\n" + "=" * 78)
    print("ALL-RESIDENT FIXED-CORPUS AUTOTUNE")
    print("=" * 78)
    print(
        f"Każda konfiguracja: te same {coarse_count} gier | "
        f"walidacja top: te same {validation_count} gier"
    )

    table_scores: list[tuple[float, int]] = []
    for tables in table_candidates:
        score = bench(tables, min(baseline_refill, tables), baseline_sync)
        print(f"[FIXED TUNE] tables={tables:>4} | {score:>8.1f} gry/s")
        table_scores.append((score, tables))
    table_scores.sort(reverse=True)
    best_tables = table_scores[0][1]

    refill_candidates = set(refill_cfg)
    refill_candidates.update(
        max(1, best_tables // d) for d in (16, 12, 8, 6, 4, 3, 2)
    )
    refill_candidates = sorted(
        v for v in refill_candidates if 1 <= v <= best_tables
    )

    refill_scores: list[tuple[float, int]] = []
    for refill in refill_candidates:
        score = bench(best_tables, refill, baseline_sync)
        print(f"[FIXED TUNE] refill={refill:>4} | {score:>8.1f} gry/s")
        refill_scores.append((score, refill))
    refill_scores.sort(reverse=True)
    best_refill = refill_scores[0][1]

    sync_scores: list[tuple[float, int]] = []
    for sync in sync_candidates:
        score = bench(best_tables, best_refill, sync)
        print(f"[FIXED TUNE] sync={sync:>4} | {score:>8.1f} gry/s")
        sync_scores.append((score, sync))
    sync_scores.sort(reverse=True)
    best_sync = sync_scores[0][1]

    print("\n[FIXED TUNE] długi retest top-3 tables")
    finalists: list[tuple[float, int, int, int]] = []
    for _score, tables in table_scores[:3]:
        scaled_refill = min(
            tables,
            max(1, round(best_refill * tables / best_tables)),
        )
        score = bench(
            tables,
            scaled_refill,
            best_sync,
            validation=True,
        )
        print(
            f"[FIXED VALIDATE] tables={tables:>4} | refill={scaled_refill:>4} | "
            f"sync={best_sync:>2} | {score:>8.1f} gry/s"
        )
        finalists.append((score, tables, scaled_refill, best_sync))

    finalists.sort(reverse=True)
    best_score, best_tables, best_refill, best_sync = finalists[0]
    print(
        f"[FIXED WINNER] tables={best_tables} | refill={best_refill} | "
        f"sync={best_sync} | {best_score:.1f} gry/s\n"
    )
    return best_tables, best_refill, best_sync
