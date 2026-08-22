from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import torch

from . import championship_cnn as base

_legacy = base._legacy


@dataclass(frozen=True, slots=True)
class GameJob:
    a: Any
    b: Any
    game_index: int
    a_slot: int
    b_slot: int

    @property
    def a_is_black(self) -> bool:
        return self.game_index % 2 == 0


class Cohort:
    """Grupa gier uruchomionych jednocześnie.

    Gry w jednym cohort mają ten sam lokalny numer ruchu, więc układ modeli
    BLACK/WHITE jest stały dla całej grupy. Po synchronizacji martwe pozycje są
    usuwane z inference, bez przebudowywania całej puli modeli na GPU.
    """

    def __init__(
        self,
        jobs: list[GameJob],
        ensemble: base.CNNBatchedPolicyEnsemble,
        *,
        temperature: float,
        amp: bool,
        amp_dtype: str,
        seed: int,
    ) -> None:
        self.jobs = jobs
        self.ensemble = ensemble
        self.device = ensemble.device
        self.temperature = float(temperature)
        self.amp = bool(amp)
        self.amp_dtype = str(amp_dtype)
        self.generator = _legacy._make_generator(self.device, seed)
        self.env = _legacy.VectorConnect6(
            len(jobs),
            ensemble.board_size,
            ensemble.win_length,
            device=self.device,
            debug_checks=False,
        )
        self.active = torch.ones(len(jobs), dtype=torch.bool, device=self.device)
        self.winners = torch.zeros(len(jobs), dtype=torch.int8, device=self.device)
        self.reported = [False] * len(jobs)
        self.move_index = 0
        self.active_indices = torch.arange(len(jobs), device=self.device, dtype=torch.long)

        black_slots = [job.a_slot if job.a_is_black else job.b_slot for job in jobs]
        white_slots = [job.b_slot if job.a_is_black else job.a_slot for job in jobs]
        self.black_slots_all = torch.tensor(black_slots, device=self.device, dtype=torch.long)
        self.white_slots_all = torch.tensor(white_slots, device=self.device, dtype=torch.long)
        self.black_layout = ensemble.prepare_layout(self.black_slots_all)
        self.white_layout = ensemble.prepare_layout(self.white_slots_all)
        self._moves_since_sync = 0

    @property
    def active_count(self) -> int:
        return int(self.active_indices.numel())

    @property
    def finished(self) -> bool:
        return self.active_count == 0

    def _rebuild_active_layouts(self, indices_cpu: list[int]) -> None:
        if not indices_cpu:
            self.active_indices = torch.empty(0, dtype=torch.long, device=self.device)
            return
        self.active_indices = torch.tensor(indices_cpu, device=self.device, dtype=torch.long)
        black = self.black_slots_all.index_select(0, self.active_indices)
        white = self.white_slots_all.index_select(0, self.active_indices)
        self.black_layout = self.ensemble.prepare_layout(black)
        self.white_layout = self.ensemble.prepare_layout(white)

    def sync_finished(self) -> list[tuple[GameJob, int]]:
        active_cpu = self.active.detach().cpu().tolist()
        winners_cpu = self.winners.detach().cpu().tolist()
        newly_finished: list[tuple[GameJob, int]] = []
        remaining: list[int] = []
        for i, is_active in enumerate(active_cpu):
            if is_active:
                remaining.append(i)
            elif not self.reported[i]:
                self.reported[i] = True
                newly_finished.append((self.jobs[i], int(winners_cpu[i])))
        self._rebuild_active_layouts(remaining)
        self._moves_since_sync = 0
        return newly_finished

    @torch.inference_mode()
    def step(self, sync_interval: int) -> list[tuple[GameJob, int]]:
        if self.finished:
            return []

        x_all = self.env.network_input()
        legal_all = self.env.legal_mask()
        x = x_all.index_select(0, self.active_indices)
        legal = legal_all.index_select(0, self.active_indices)
        layout = self.black_layout if base._black_to_move(self.move_index) else self.white_layout

        with _legacy._autocast_context(self.device, self.amp, self.amp_dtype):
            logits = self.ensemble.forward_prepared(x, layout)
        chosen = _legacy._choose_actions(logits, legal, self.temperature, self.generator)

        full_actions = legal_all.to(torch.int8).argmax(dim=1).to(torch.long)
        full_actions.index_copy_(0, self.active_indices, chosen)
        done, winner = base._masked_step(self.env, full_actions, self.active)
        newly_done = self.active & done
        self.winners = torch.where(newly_done, winner, self.winners)
        self.active &= ~done
        self.move_index += 1
        self._moves_since_sync += 1

        if self._moves_since_sync >= max(1, sync_interval) or self.move_index >= self.env.action_size:
            return self.sync_finished()
        return []


def _gpu_timed(device: torch.device, fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return max(1e-9, start.elapsed_time(end) / 1000.0)


def _bench_host_host(
    cp: Any,
    tables: int,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    iterations: int,
) -> float:
    ensemble = base.CNNBatchedPolicyEnsemble([cp] * tables, device)
    try:
        ids = torch.arange(tables, device=device, dtype=torch.long)
        layout = ensemble.prepare_layout(ids)
        x = torch.zeros((tables, 3, ensemble.board_size, ensemble.board_size), device=device)
        with _legacy._autocast_context(device, amp, amp_dtype):
            ensemble.forward_prepared(x, layout)
        torch.cuda.synchronize(device)

        def work() -> None:
            with _legacy._autocast_context(device, amp, amp_dtype):
                for _ in range(iterations):
                    ensemble.forward_prepared(x, layout)

        elapsed = _gpu_timed(device, work)
        return tables * iterations / elapsed
    finally:
        ensemble.release()


def _bench_cross(
    cp: Any,
    tables: int,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    iterations: int,
) -> float:
    hosts = base.CNNBatchedPolicyEnsemble([cp] * tables, device)
    challenger = base.CNNBatchedPolicyEnsemble([cp], device)
    try:
        ids = torch.arange(tables, device=device, dtype=torch.long)
        layout = hosts.prepare_layout(ids)
        x = torch.zeros((tables, 3, hosts.board_size, hosts.board_size), device=device)
        with _legacy._autocast_context(device, amp, amp_dtype):
            hosts.forward_prepared(x, layout)
            challenger.forward_single_model(x, 0)
        torch.cuda.synchronize(device)

        def work() -> None:
            with _legacy._autocast_context(device, amp, amp_dtype):
                for _ in range(iterations):
                    hosts.forward_prepared(x, layout)
                    challenger.forward_single_model(x, 0)

        elapsed = _gpu_timed(device, work)
        return 2 * tables * iterations / elapsed
    finally:
        challenger.release()
        hosts.release()


def _bench_full_loop(
    cp: Any,
    tables: int,
    *,
    device: torch.device,
    amp: bool,
    amp_dtype: str,
    moves: int,
) -> float:
    ensemble = base.CNNBatchedPolicyEnsemble([cp] * tables, device)
    try:
        ids = torch.arange(tables, device=device, dtype=torch.long)
        layout = ensemble.prepare_layout(ids)
        env = _legacy.VectorConnect6(
            tables,
            ensemble.board_size,
            ensemble.win_length,
            device=device,
            debug_checks=False,
        )
        generator = _legacy._make_generator(device, 987654)

        def work() -> None:
            for _ in range(moves):
                x = env.network_input()
                legal = env.legal_mask()
                with _legacy._autocast_context(device, amp, amp_dtype):
                    logits = ensemble.forward_prepared(x, layout)
                actions = _legacy._choose_actions(logits, legal, 0.0, generator)
                env.step(actions)

        # warm-up na osobnym env, aby właściwy pomiar zaczynał się po inicjalizacji cuDNN
        x = env.network_input()
        with _legacy._autocast_context(device, amp, amp_dtype):
            ensemble.forward_prepared(x, layout)
        torch.cuda.synchronize(device)
        elapsed = _gpu_timed(device, work)
        return tables * moves / elapsed
    finally:
        ensemble.release()


def _harmonic_score(hh: float, cross: float, loop: float) -> float:
    # Pełny loop ma największą wagę; harmoniczna karze ustawienie słabe w jednym trybie.
    weights = ((0.20, hh), (0.25, cross), (0.55, loop))
    denom = sum(w / max(rate, 1e-9) for w, rate in weights)
    return 1.0 / denom


def autotune_three_tests(
    config_path: Path,
    cfg: dict[str, Any],
    adaptive: dict[str, Any],
) -> tuple[int, base.AdaptiveTableController, list[int]]:
    ch = cfg.get("championship", cfg)
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(project_root, ch["checkpoint_dir"])
    output_dir = _legacy._resolve_path(project_root, ch["output_dir"])
    refs = _legacy.discover_checkpoints(checkpoint_dir)
    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    if device.type != "cuda":
        fixed = min(len(refs), int(ch.get("tables", 4)))
        controller = base.AdaptiveTableController(
            output_dir / "adaptive_tables.json",
            gpu_name="CPU",
            limit_bytes=1,
            min_tables=2,
            max_tables=max(2, fixed),
        )
        return fixed, controller, [fixed]

    limit_gb = float(adaptive.get("vram_limit_gb", 11.0))
    limit_bytes = int(limit_gb * 2**30)
    gpu_name = torch.cuda.get_device_name(device)
    candidates = base._candidate_tables(adaptive, len(refs))
    controller = base.AdaptiveTableController(
        output_dir / "adaptive_tables.json",
        gpu_name=gpu_name,
        limit_bytes=limit_bytes,
        min_tables=min(candidates),
        max_tables=max(candidates),
    )
    store = base.CNNCheckpointStore(cpu_cache_models=1)
    cp = store.get(refs[0])
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    iterations = max(2, int(adaptive.get("benchmark_iterations", 3)))
    loop_moves = max(4, int(adaptive.get("benchmark_loop_moves", 8)))

    base._VRAM_LIMIT_BYTES = limit_bytes
    print("\n[ADAPTIVE GPU] autotest 3-trybowy: HOST-HOST / CROSS / FULL-LOOP")
    print(f"GPU: {gpu_name} | limit VRAM: {limit_gb:.2f} GB")
    best_tables: int | None = None
    best_score = -1.0

    for tables in candidates:
        if not controller.is_allowed(tables):
            print(f"[AUTOTUNE] {tables:>4} -> pomijam, znane UNSAFE")
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            hh = _bench_host_host(cp, tables, device=device, amp=amp, amp_dtype=amp_dtype, iterations=iterations)
            cross = _bench_cross(cp, tables, device=device, amp=amp, amp_dtype=amp_dtype, iterations=iterations)
            loop = _bench_full_loop(cp, tables, device=device, amp=amp, amp_dtype=amp_dtype, moves=loop_moves)
            peak = torch.cuda.max_memory_reserved(device)
            if peak > limit_bytes:
                raise base.AdaptiveBatchResize(
                    f"peak {peak / 2**30:.2f} GB > {limit_gb:.2f} GB"
                )
        except (base.AdaptiveBatchResize, torch.cuda.OutOfMemoryError) as exc:
            controller.mark_unsafe(tables)
            torch.cuda.empty_cache()
            print(f"[AUTOTUNE] {tables:>4} -> UNSAFE: {exc}")
            break

        score = _harmonic_score(hh, cross, loop)
        controller.benchmarks[tables] = {
            "host_host_dps": hh,
            "cross_dps": cross,
            "full_loop_dps": loop,
            "score": score,
            "peak_vram_gb": peak / 2**30,
        }
        print(
            f"[AUTOTUNE] {tables:>4} | HH {hh:>9,.0f} d/s | "
            f"CROSS {cross:>9,.0f} d/s | LOOP {loop:>9,.0f} d/s | "
            f"score {score:>9,.0f} | VRAM {peak / 2**30:.2f} GB"
        )
        if score > best_score:
            best_score = score
            best_tables = tables

    if best_tables is None:
        best_tables = min(candidates)
    controller.last_selected = best_tables
    controller.save()
    print(f"[AUTOTUNE] wybrano {best_tables} stołów | score {best_score:,.0f}\n")
    return best_tables, controller, candidates


def _update_counter(counter: dict[str, int], winner: int, *, a_is_black: bool) -> None:
    if winner == 0:
        counter["draws"] += 1
        return
    winner_is_a = (winner == 1 and a_is_black) or (winner == -1 and not a_is_black)
    if winner_is_a:
        counter["a_game_wins"] += 1
        counter["a_wins_as_black" if a_is_black else "a_wins_as_white"] += 1
    else:
        counter["b_game_wins"] += 1
        counter["b_wins_as_white" if a_is_black else "b_wins_as_black"] += 1


def _pair_jobs(
    left: list[Any],
    right: list[Any],
    *,
    same_block: bool,
    games_per_pair: int,
    completed: set[str],
    slot_of: dict[str, int],
) -> list[GameJob]:
    pairs = combinations(left, 2) if same_block else product(left, right)
    jobs: list[GameJob] = []
    for a, b in pairs:
        if _legacy._pair_id(a, b) in completed:
            continue
        for game_index in range(games_per_pair):
            jobs.append(
                GameJob(
                    a=a,
                    b=b,
                    game_index=game_index,
                    a_slot=slot_of[a.name],
                    b_slot=slot_of[b.name],
                )
            )
    return jobs


def _write_progress(
    *,
    checkpoints: list[Any],
    match_rows: list[dict[str, Any]],
    completed_ids: set[str],
    total_pairs: int,
    ranking_path: Path,
    html_path: Path,
    games_per_pair: int,
    tables: int,
    temperature: float,
) -> list[dict[str, Any]]:
    ranking = _legacy.build_ranking(checkpoints, match_rows)
    _legacy.write_ranking_csv(ranking_path, ranking)
    _legacy.write_html(
        html_path,
        ranking,
        completed_pairs=len(completed_ids),
        total_pairs=total_pairs,
        checkpoint_count=len(checkpoints),
        games_per_pair=games_per_pair,
        tables=tables,
        temperature=temperature,
        running=len(completed_ids) < total_pairs,
    )
    return ranking


def run_streaming_once(
    config_path: Path,
    cfg: dict[str, Any],
    *,
    tables: int,
    controller: base.AdaptiveTableController,
) -> None:
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}
    project_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    checkpoint_dir = _legacy._resolve_path(project_root, ch["checkpoint_dir"])
    output_dir = _legacy._resolve_path(project_root, ch["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = _legacy.discover_checkpoints(checkpoint_dir)
    games_per_pair = int(ch.get("games_per_pair", 2))
    temperature = float(ch.get("temperature", 0.0))
    amp = bool(ch.get("amp", True))
    amp_dtype = str(ch.get("amp_dtype", "bfloat16"))
    cpu_cache_models = int(ch.get("cpu_cache_models", 16384))
    seed = int(ch.get("seed", 12345))
    sync_interval = max(1, int(adaptive.get("sync_interval_moves", 8)))
    refill_batch = max(1, min(tables, int(adaptive.get("refill_batch", 32))))
    block_default = min(128, max(16, tables // 2))
    model_block_size = max(2, int(adaptive.get("model_pool_block_size", block_default)))
    model_block_size = min(model_block_size, max(2, tables // 2))

    matches_path = output_dir / "matches.csv"
    ranking_path = output_dir / "ranking.csv"
    html_path = output_dir / "championship.html"
    state_path = output_dir / "state.json"
    _legacy._validate_or_create_state(
        state_path,
        checkpoints=checkpoints,
        games_per_pair=games_per_pair,
        temperature=temperature,
    )
    completed_ids, match_rows = _legacy._load_completed_matches(matches_path)
    total_pairs = len(checkpoints) * (len(checkpoints) - 1) // 2
    ranking = _write_progress(
        checkpoints=checkpoints,
        match_rows=match_rows,
        completed_ids=completed_ids,
        total_pairs=total_pairs,
        ranking_path=ranking_path,
        html_path=html_path,
        games_per_pair=games_per_pair,
        tables=tables,
        temperature=temperature,
    )

    device = _legacy._torch_device(str(ch.get("device", "cuda")))
    base._VRAM_LIMIT_BYTES = controller.limit_bytes if device.type == "cuda" else 0
    store = base.CNNCheckpointStore(cpu_cache_models=cpu_cache_models)
    blocks = [checkpoints[i : i + model_block_size] for i in range(0, len(checkpoints), model_block_size)]

    print("=" * 78)
    print("KING OF CONNECT6 — PERSISTENT STREAMING CHAMPIONSHIP")
    print("=" * 78)
    print(f"Stoły: {tables} | refill batch: {refill_batch} | model block: {model_block_size}")
    print(f"Synchronizacja scheduler-a: co {sync_interval} ruchów/cohort")
    print(f"Pary: {len(completed_ids):,}/{total_pairs:,} ukończone")

    pair_counters: dict[str, dict[str, int]] = {}
    pair_refs: dict[str, tuple[Any, Any]] = {}
    started_at = time.perf_counter()
    finished_games_total = 0

    for bi, left in enumerate(blocks):
        for bj in range(bi, len(blocks)):
            right = blocks[bj]
            same = bi == bj
            pool_refs = left if same else left + right
            # Usuwamy duplikaty zachowując kolejność.
            unique: list[Any] = []
            seen: set[str] = set()
            for ref in pool_refs:
                if ref.name not in seen:
                    seen.add(ref.name)
                    unique.append(ref)

            slot_of = {ref.name: i for i, ref in enumerate(unique)}
            jobs = _pair_jobs(
                left,
                right,
                same_block=same,
                games_per_pair=games_per_pair,
                completed=completed_ids,
                slot_of=slot_of,
            )
            if not jobs:
                continue

            print(
                f"\n[MODEL POOL] bloki {bi + 1}/{bj + 1} | modele={len(unique)} | "
                f"gry do wykonania={len(jobs)}"
            )
            lean = [store.get(ref) for ref in unique]
            ensemble = base.CNNBatchedPolicyEnsemble(lean, device)
            pending = 0
            cohorts: list[Cohort] = []

            def add_cohort(count: int) -> None:
                nonlocal pending
                if count <= 0 or pending >= len(jobs):
                    return
                batch = jobs[pending : pending + count]
                pending += len(batch)
                cohorts.append(
                    Cohort(
                        batch,
                        ensemble,
                        temperature=temperature,
                        amp=amp,
                        amp_dtype=amp_dtype,
                        seed=seed + pending + finished_games_total,
                    )
                )

            add_cohort(min(tables, len(jobs)))

            try:
                while cohorts:
                    newly: list[tuple[GameJob, int]] = []
                    for cohort in list(cohorts):
                        newly.extend(cohort.step(sync_interval))

                    if newly:
                        finished_games_total += len(newly)
                        for job, winner in newly:
                            pid = _legacy._pair_id(job.a, job.b)
                            counter = pair_counters.setdefault(pid, _legacy._empty_counter())
                            pair_refs[pid] = (job.a, job.b)
                            _update_counter(counter, winner, a_is_black=job.a_is_black)
                            games_done = counter["a_game_wins"] + counter["b_game_wins"] + counter["draws"]
                            if games_done == games_per_pair:
                                row = _legacy._finalize_pair_rows(
                                    [pair_refs[pid]], [counter], games_per_pair
                                )[0]
                                row["elapsed_seconds"] = 0.0
                                _legacy._append_match(matches_path, row)
                                completed_ids.add(pid)
                                match_rows.append(row)
                                del pair_counters[pid]
                                del pair_refs[pid]

                        ranking = _write_progress(
                            checkpoints=checkpoints,
                            match_rows=match_rows,
                            completed_ids=completed_ids,
                            total_pairs=total_pairs,
                            ranking_path=ranking_path,
                            html_path=html_path,
                            games_per_pair=games_per_pair,
                            tables=tables,
                            temperature=temperature,
                        )

                    cohorts = [c for c in cohorts if not c.finished]
                    active_now = sum(c.active_count for c in cohorts)
                    free = tables - active_now
                    remaining = len(jobs) - pending

                    if remaining > 0 and (free >= refill_batch or not cohorts):
                        refill = min(free, remaining)
                        if refill > 0:
                            add_cohort(refill)

                    if newly:
                        elapsed = max(1e-9, time.perf_counter() - started_at)
                        print(
                            f"[STREAM] aktywne {sum(c.active_count for c in cohorts):>4}/{tables} | "
                            f"cohorty={len(cohorts)} | refill_free={tables - sum(c.active_count for c in cohorts):>3} | "
                            f"gry/s={finished_games_total / elapsed:.2f} | "
                            f"VRAM {_legacy._vram_label(device)}"
                        )
            finally:
                ensemble.release()
                del ensemble

    ranking = _write_progress(
        checkpoints=checkpoints,
        match_rows=match_rows,
        completed_ids=completed_ids,
        total_pairs=total_pairs,
        ranking_path=ranking_path,
        html_path=html_path,
        games_per_pair=games_per_pair,
        tables=tables,
        temperature=temperature,
    )
    print("\n" + "=" * 78)
    print("MISTRZOSTWA ZAKOŃCZONE")
    if ranking:
        print(f"👑 KING OF CONNECT6: {ranking[0]['model']} | {ranking[0]['points']} pkt")
    print(f"Ranking HTML: {html_path}")
    print("=" * 78)


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = base._ORIGINAL_READ_YAML(config_path)
    ch = cfg.get("championship", cfg)
    adaptive = ch.get("adaptive_tables", {}) or {}

    selected, controller, candidates = autotune_three_tests(config_path, cfg, adaptive)
    while True:
        try:
            controller.last_selected = selected
            controller.save()
            run_streaming_once(config_path, cfg, tables=selected, controller=controller)
            return
        except base.AdaptiveBatchResize as exc:
            controller.mark_unsafe(selected)
            new_tables = controller.next_lower(selected, candidates)
            if new_tables >= selected or selected <= controller.min_tables:
                raise
            print(
                f"\n[ADAPTIVE GPU] {selected} stołów przekroczyło bezpieczny budżet: {exc}. "
                f"Zapamiętuję i wznawiam z {new_tables}.\n"
            )
            selected = new_tables
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="King of Connect6 — persistent streaming CNN championship"
    )
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
