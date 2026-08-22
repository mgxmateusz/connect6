from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any

import torch

from . import championship_stream as stream

base = stream.base
_legacy = stream._legacy


class PairJobStream:
    """Leniwy generator wszystkich gier bez materializacji milionów obiektów."""

    def __init__(
        self,
        refs: list[Any],
        *,
        completed: set[str],
        games_per_pair: int,
    ) -> None:
        self.refs = refs
        self.completed = completed
        self.games_per_pair = int(games_per_pair)
        self.i = 0
        self.j = 1
        self.game_index = 0
        self._done = len(refs) < 2

    def _advance_pair(self) -> None:
        self.game_index = 0
        self.j += 1
        if self.j >= len(self.refs):
            self.i += 1
            self.j = self.i + 1
        if self.i >= len(self.refs) - 1:
            self._done = True

    def take(self, count: int) -> list[stream.GameJob]:
        out: list[stream.GameJob] = []
        count = max(0, int(count))
        while len(out) < count and not self._done:
            a = self.refs[self.i]
            b = self.refs[self.j]
            pid = _legacy._pair_id(a, b)
            if pid in self.completed:
                self._advance_pair()
                continue

            out.append(
                stream.GameJob(
                    a=a,
                    b=b,
                    game_index=self.game_index,
                    a_slot=self.i,
                    b_slot=self.j,
                )
            )
            self.game_index += 1
            if self.game_index >= self.games_per_pair:
                self._advance_pair()
        return out

    @property
    def exhausted(self) -> bool:
        return self._done


def _cast_ensemble_storage(
    ensemble: stream.DirectIndexedEnsemble,
    dtype_name: str,
) -> None:
    if dtype_name == "float16":
        dtype = torch.float16
    elif dtype_name == "bfloat16":
        dtype = torch.bfloat16
    else:
        return

    ens = ensemble._ensemble
    ens.conv_weights = [w.to(dtype=dtype) for w in ens.conv_weights]
    ens.conv_biases = [None if b is None else b.to(dtype=dtype) for b in ens.conv_biases]
    ens.policy_weight = ens.policy_weight.to(dtype=dtype)
    if ens.policy_bias is not None:
        ens.policy_bias = ens.policy_bias.to(dtype=dtype)
    ens.dtype = dtype


def _benchmark_resident_scheduler(
    ensemble: stream.DirectIndexedEnsemble,
    refs: list[Any],
    *,
    ch: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[int, int, int]:
    tune = adaptive.get("resident_autotune", {}) or {}
    table_candidates = [int(v) for v in tune.get("table_candidates", [96, 112, 120, 128, 136, 144])]
    refill_candidates = [int(v) for v in tune.get("refill_candidates", [8, 16, 24, 32, 48, 64])]
    sync_candidates = [int(v) for v in tune.get("sync_candidates", [1, 2, 4, 6, 8])]
    warmup_games = max(32, int(tune.get("warmup_games", 96)))
    measure_games = max(128, int(tune.get("measure_games", 768)))
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "float16"))
    temperature = float(ch.get("temperature", 0.0))
    seed = int(ch.get("seed", 12345))

    def bench(tables: int, refill: int, sync: int) -> float:
        tables = min(max(2, tables), len(refs))
        refill = min(max(1, refill), tables)
        scheduler = stream.GlobalTableScheduler(
            tables,
            ensemble,
            temperature=temperature,
            amp=amp,
            amp_dtype=amp_dtype,
            seed=seed + tables * 1009 + refill * 31 + sync,
        )
        # Używamy tego samego deterministycznego strumienia par w każdym teście.
        jobs = []
        need = tables + warmup_games + measure_games + tables * 2
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
        if start_measure is None:
            return 0.0
        elapsed = max(1e-9, time.perf_counter() - start_measure)
        return measured / elapsed

    baseline_refill = int(adaptive.get("refill_batch", 32))
    baseline_sync = int(adaptive.get("sync_interval_moves", 4))

    print("\n" + "=" * 78)
    print("ALL-RESIDENT SCHEDULER AUTOTUNE")
    print("=" * 78)

    table_scores: list[tuple[float, int]] = []
    for tables in table_candidates:
        score = bench(tables, min(baseline_refill, tables), baseline_sync)
        print(f"[RESIDENT TUNE] tables={tables:>3} | {score:>7.1f} gry/s")
        table_scores.append((score, tables))
    _, best_tables = max(table_scores)

    refill_scores: list[tuple[float, int]] = []
    for refill in refill_candidates:
        if refill > best_tables:
            continue
        score = bench(best_tables, refill, baseline_sync)
        print(f"[RESIDENT TUNE] refill={refill:>3} | {score:>7.1f} gry/s")
        refill_scores.append((score, refill))
    _, best_refill = max(refill_scores)

    sync_scores: list[tuple[float, int]] = []
    for sync in sync_candidates:
        score = bench(best_tables, best_refill, sync)
        print(f"[RESIDENT TUNE] sync={sync:>2} | {score:>7.1f} gry/s")
        sync_scores.append((score, sync))
    _, best_sync = max(sync_scores)

    print(
        f"[RESIDENT WINNER] tables={best_tables} | refill={best_refill} | "
        f"sync={best_sync}\n"
    )
    return best_tables, best_refill, best_sync


def run_resident(
    config_path: str | Path,
    cfg: dict[str, Any],
    *,
    backend_cls: type[stream.DirectIndexedEnsemble],
    dtype_name: str,
) -> bool:
    """Zwraca True gdy turniej został uruchomiony w trybie all-resident.

    Gdy wszystkie modele nie mieszczą się w limicie VRAM, zwraca False i caller
    może przejść do poolowego fallbacku.
    """
    config_path = Path(config_path).resolve()
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}
    resident = adaptive.get("all_resident", {}) or {}
    if not bool(resident.get("enabled", True)):
        return False

    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(root, ch["checkpoint_dir"])
    output_dir = _legacy._resolve_path(root, ch["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = _legacy.discover_checkpoints(checkpoint_dir)
    if len(refs) < 2:
        return False

    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    if device.type != "cuda":
        return False

    limit_gb = float(adaptive.get("vram_limit_gb", 11.0))
    limit_bytes = int(limit_gb * 2**30)
    base._VRAM_LIMIT_BYTES = limit_bytes

    print("\n" + "=" * 78)
    print("ALL-RESIDENT GPU LOAD")
    print("=" * 78)
    print(f"Ładuję {len(refs)} modeli do jednego ensemble GPU | storage={dtype_name}")

    store = base.CNNCheckpointStore(cpu_cache_models=len(refs))
    try:
        lean = [store.get(ref) for ref in refs]
        torch.cuda.empty_cache()
        ensemble = backend_cls(lean, device)
        _cast_ensemble_storage(ensemble, dtype_name)
        torch.cuda.synchronize(device)
        alloc = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        print(
            f"[RESIDENT] po załadowaniu: alloc={alloc / 2**30:.2f} GB | "
            f"reserved={reserved / 2**30:.2f} GB | limit={limit_gb:.2f} GB"
        )
        if max(alloc, reserved) > limit_bytes:
            ensemble.release()
            del ensemble
            torch.cuda.empty_cache()
            print("[RESIDENT] przekroczono limit -> wracam do model pools")
            return False
    except (torch.cuda.OutOfMemoryError, base.AdaptiveBatchResize) as exc:
        torch.cuda.empty_cache()
        print(f"[RESIDENT] nie udało się załadować wszystkich modeli: {exc}")
        return False

    try:
        tuned_ch = copy.deepcopy(ch)
        tuned_ch["amp_dtype"] = dtype_name
        tables, refill_batch, sync_interval = _benchmark_resident_scheduler(
            ensemble,
            refs,
            ch=tuned_ch,
            adaptive=adaptive,
        )

        games_per_pair = int(ch.get("games_per_pair", 2))
        temperature = float(ch.get("temperature", 0.0))
        amp = bool(ch.get("amp", True))
        cpu_cache_models = int(ch.get("cpu_cache_models", 16384))
        seed = int(ch.get("seed", 12345))
        progress_seconds = max(1.0, float(adaptive.get("progress_every_seconds", 5.0)))
        progress_matches = max(1, int(adaptive.get("progress_every_matches", 500)))
        log_seconds = max(0.25, float(adaptive.get("log_every_seconds", 1.0)))

        matches_path = output_dir / "matches.csv"
        ranking_path = output_dir / "ranking.csv"
        html_path = output_dir / "championship.html"
        state_path = output_dir / "state.json"
        _legacy._validate_or_create_state(
            state_path,
            checkpoints=refs,
            games_per_pair=games_per_pair,
            temperature=temperature,
        )
        completed_ids, match_rows = _legacy._load_completed_matches(matches_path)
        total_pairs = len(refs) * (len(refs) - 1) // 2
        stream._write_progress(
            refs,
            match_rows,
            completed_ids,
            total_pairs,
            ranking_path,
            html_path,
            games_per_pair,
            tables,
            temperature,
        )

        scheduler = stream.GlobalTableScheduler(
            tables,
            ensemble,
            temperature=temperature,
            amp=amp,
            amp_dtype=dtype_name,
            seed=seed,
        )
        job_stream = PairJobStream(
            refs,
            completed=completed_ids,
            games_per_pair=games_per_pair,
        )

        initial_jobs = job_stream.take(tables)
        scheduler.refill(list(range(len(initial_jobs))), initial_jobs)

        pair_counters: dict[str, dict[str, int]] = {}
        pair_refs: dict[str, tuple[Any, Any]] = {}
        started_at = time.perf_counter()
        last_log = started_at
        last_progress = started_at
        last_progress_matches = len(completed_ids)
        finished_games_total = 0
        window_started = started_at
        window_games = 0

        print("\n" + "=" * 78)
        print("KING OF CONNECT6 — ALL MODELS RESIDENT")
        print("=" * 78)
        print(
            f"Modele GPU: {len(refs)} | stoły: {tables} | refill: {refill_batch} | "
            f"sync: {sync_interval} | dtype: {dtype_name}"
        )
        print(f"Pary: {len(completed_ids):,}/{total_pairs:,} ukończone")

        while scheduler.active_count > 0 or not job_stream.exhausted:
            finished, free = scheduler.step(sync_interval)
            completed_rows: list[dict[str, Any]] = []

            for _slot, job, winner in finished:
                finished_games_total += 1
                window_games += 1
                pid = _legacy._pair_id(job.a, job.b)
                counter = pair_counters.setdefault(pid, _legacy._empty_counter())
                pair_refs[pid] = (job.a, job.b)
                stream._update_counter(counter, winner, a_is_black=job.a_is_black)
                games_done = counter["a_game_wins"] + counter["b_game_wins"] + counter["draws"]
                if games_done == games_per_pair:
                    row = _legacy._finalize_pair_rows([pair_refs[pid]], [counter], games_per_pair)[0]
                    row["elapsed_seconds"] = 0.0
                    completed_rows.append(row)
                    completed_ids.add(pid)
                    match_rows.append(row)
                    del pair_counters[pid]
                    del pair_refs[pid]

            stream._append_rows(matches_path, completed_rows)

            if not job_stream.exhausted and (len(free) >= refill_batch or scheduler.active_count == 0):
                new_jobs = job_stream.take(len(free))
                if new_jobs:
                    scheduler.refill(free[: len(new_jobs)], new_jobs)

            now = time.perf_counter()
            if (
                now - last_progress >= progress_seconds
                or len(completed_ids) - last_progress_matches >= progress_matches
            ):
                stream._write_progress(
                    refs,
                    match_rows,
                    completed_ids,
                    total_pairs,
                    ranking_path,
                    html_path,
                    games_per_pair,
                    tables,
                    temperature,
                )
                last_progress = now
                last_progress_matches = len(completed_ids)

            if finished and now - last_log >= log_seconds:
                elapsed = max(1e-9, now - started_at)
                window_elapsed = max(1e-9, now - window_started)
                avg_gps = finished_games_total / elapsed
                current_gps = window_games / window_elapsed
                pairs_left = max(0, total_pairs - len(completed_ids))
                eta_seconds = (pairs_left * games_per_pair) / max(current_gps, 1e-9)
                print(
                    f"[RESIDENT STREAM] aktywne {scheduler.active_count:>4}/{tables} | "
                    f"current={current_gps:>7.1f} gry/s | avg={avg_gps:>7.1f} | "
                    f"pary={len(completed_ids):,}/{total_pairs:,} | "
                    f"ETA={eta_seconds / 3600:.2f}h | {_legacy._vram_label(device)}"
                )
                last_log = now
                window_started = now
                window_games = 0

        ranking = stream._write_progress(
            refs,
            match_rows,
            completed_ids,
            total_pairs,
            ranking_path,
            html_path,
            games_per_pair,
            tables,
            temperature,
        )
        print("\n" + "=" * 78)
        print("MISTRZOSTWA ZAKOŃCZONE — ALL RESIDENT")
        if ranking:
            print(f"👑 KING OF CONNECT6: {ranking[0]['model']} | {ranking[0]['points']} pkt")
        print("=" * 78)
        return True
    finally:
        ensemble.release()
        del ensemble
        torch.cuda.empty_cache()
