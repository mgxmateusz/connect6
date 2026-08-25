from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import championship as legacy
from .checkpoint import load_checkpoint
from .cuda_native import load_native_championship_extension

EXPECTED_KERNELS = (23, 3, 3, 3, 3, 3, 3, 3)
EXPECTED_CHANNELS = (32, 32, 64, 64, 64, 96, 96, 96)
EXPECTED_KPAD = (2128, 288, 288, 576, 576, 576, 864, 864)
NORM_LAYERS = (0, 2, 5, 7)
ARCHITECTURE_VERSION = 6


def _project_root(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _pair_index(i: int, j: int, n: int) -> int:
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def _pair_indices_array(i: np.ndarray, j: np.ndarray, n: int) -> np.ndarray:
    i64 = i.astype(np.int64, copy=False)
    j64 = j.astype(np.int64, copy=False)
    return (i64 * (2 * n - i64 - 1) // 2 + (j64 - i64 - 1)).astype(np.int32)


def _validate_checkpoint_family(payload: dict, ref, *, first_cfg: dict | None = None) -> dict:
    cfg = dict(payload["model_config"])
    cfg.pop("compile", None)
    cfg.pop("compile_mode", None)
    if int(cfg.get("architecture_version", 0)) != ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"Checkpoint {ref.name} ma architecture_version={cfg.get('architecture_version')}; "
            f"native championship wymaga wersji {ARCHITECTURE_VERSION}."
        )
    kernels = tuple(int(v) for v in cfg.get("kernels", EXPECTED_KERNELS))
    channels = tuple(int(v) for v in cfg.get("channels", EXPECTED_CHANNELS))
    if kernels != EXPECTED_KERNELS or channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Native SM120 kernel jest wyspecjalizowany dla kernels="
            f"{EXPECTED_KERNELS}, channels={EXPECTED_CHANNELS}; otrzymano {kernels}, {channels}."
        )
    game_cfg = payload["game_config"]
    if int(game_cfg.get("board_size", 0)) != 19 or int(game_cfg.get("win_length", 6)) != 6:
        raise RuntimeError("Native championship wymaga Connect6 19x19, win_length=6")
    if first_cfg is not None and cfg != first_cfg:
        raise RuntimeError("Checkpointy mają różne konfiguracje modelu")
    return cfg


def _allocate_packed_weights(num_models: int, device: torch.device):
    weights: list[torch.Tensor] = []
    norm_weights: list[torch.Tensor] = []
    norm_biases: list[torch.Tensor] = []
    for out_ch, kpad in zip(EXPECTED_CHANNELS, EXPECTED_KPAD):
        weights.append(torch.zeros((num_models, out_ch, kpad), device=device, dtype=torch.float16))
        norm_weights.append(torch.ones((num_models, out_ch), device=device, dtype=torch.float16))
        norm_biases.append(torch.zeros((num_models, out_ch), device=device, dtype=torch.float16))
    policy = torch.empty((num_models, 96), device=device, dtype=torch.float16)
    return weights, norm_weights, norm_biases, policy


def _pack_checkpoints(refs, device: torch.device, *, chunk_size: int = 32):
    num_models = len(refs)
    weights, norm_weights, norm_biases, policy = _allocate_packed_weights(num_models, device)
    family_cfg: dict | None = None

    print(f"[NATIVE] pakuję {num_models} checkpointów V6 do layoutu WMMA FP16 + GroupNorm")
    for start in range(0, num_models, chunk_size):
        end = min(num_models, start + chunk_size)
        payloads = [load_checkpoint(ref.path, map_location="cpu") for ref in refs[start:end]]
        for ref, payload in zip(refs[start:end], payloads):
            cfg = _validate_checkpoint_family(payload, ref, first_cfg=family_cfg)
            if family_cfg is None:
                family_cfg = cfg

        for layer in range(8):
            raw = torch.stack([p["model_state"][f"convs.{layer}.weight"] for p in payloads], dim=0)
            raw = raw.to(device=device, dtype=torch.float16, non_blocking=False)
            raw = raw.reshape(end - start, EXPECTED_CHANNELS[layer], -1)
            kreal = int(raw.shape[-1])
            if kreal > EXPECTED_KPAD[layer]:
                raise RuntimeError(
                    f"Warstwa {layer}: K={kreal} nie mieści się w KPAD={EXPECTED_KPAD[layer]}"
                )
            weights[layer][start:end, :, :kreal].copy_(raw)
            if kreal < EXPECTED_KPAD[layer]:
                weights[layer][start:end, :, kreal:].zero_()
            del raw

            if layer in NORM_LAYERS:
                nw = torch.stack(
                    [p["model_state"][f"norms.{layer}.weight"] for p in payloads], dim=0
                )
                nb = torch.stack(
                    [p["model_state"][f"norms.{layer}.bias"] for p in payloads], dim=0
                )
                norm_weights[layer][start:end].copy_(nw.to(device=device, dtype=torch.float16))
                norm_biases[layer][start:end].copy_(nb.to(device=device, dtype=torch.float16))
                del nw, nb

        pw = torch.stack([p["model_state"]["policy_output.weight"] for p in payloads], dim=0)
        policy[start:end].copy_(pw.reshape(end - start, 96).to(device=device, dtype=torch.float16))
        del pw, payloads
        gc.collect()
        if end == num_models or end % max(chunk_size, 128) == 0:
            alloc = torch.cuda.memory_allocated(device) / 2**30
            print(f"[NATIVE LOAD] {end:>5}/{num_models} | packed VRAM={alloc:.2f} GB")

    torch.cuda.synchronize(device)
    return weights, norm_weights, norm_biases, policy


def _completed_pair_mask(matches_path: Path, refs) -> np.ndarray:
    n = len(refs)
    total_pairs = n * (n - 1) // 2
    done = np.zeros(total_pairs, dtype=np.bool_)
    if not matches_path.exists():
        return done
    name_to_idx = {ref.name: i for i, ref in enumerate(refs)}
    with matches_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            ia = name_to_idx.get(row.get("model_a", ""))
            ib = name_to_idx.get(row.get("model_b", ""))
            if ia is None or ib is None or ia == ib:
                continue
            done[_pair_index(ia, ib, n)] = True
    return done


def _group_pair_mask(n: int, groups: int) -> tuple[np.ndarray, list[np.ndarray]]:
    total_pairs = n * (n - 1) // 2
    mask = np.zeros(total_pairs, dtype=np.bool_)
    members = [np.arange(g, n, groups, dtype=np.int32) for g in range(groups)]
    for group in members:
        if group.size < 2:
            continue
        li, lj = np.triu_indices(group.size, 1)
        gi = group[li]
        gj = group[lj]
        mask[_pair_indices_array(gi, gj, n)] = True
    return mask, members


def _final_pair_mask(n: int, finalists: np.ndarray) -> np.ndarray:
    total_pairs = n * (n - 1) // 2
    mask = np.zeros(total_pairs, dtype=np.bool_)
    all_models = np.arange(n, dtype=np.int32)
    for finalist in finalists.tolist():
        others = all_models[all_models != finalist]
        lo = np.minimum(others, finalist).astype(np.int32, copy=False)
        hi = np.maximum(others, finalist).astype(np.int32, copy=False)
        mask[_pair_indices_array(lo, hi, n)] = True
    return mask


def _append_match_rows(
    matches_path: Path,
    refs,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    pending_pairs: np.ndarray,
    results: np.ndarray,
    elapsed: float,
) -> None:
    game_results = results.reshape(-1, 2)
    fields = legacy.MATCH_FIELDS
    exists = matches_path.exists() and matches_path.stat().st_size > 0
    per_pair_elapsed = elapsed / max(1, pending_pairs.size)

    with matches_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for k, p in enumerate(pending_pairs):
            i = int(pair_i[p])
            j = int(pair_j[p])
            a = refs[i]
            b = refs[j]
            r0 = int(game_results[k, 0])
            r1 = int(game_results[k, 1])
            a_black = int(r0 == 1)
            a_white = int(r1 == -1)
            b_white = int(r0 == -1)
            b_black = int(r1 == 1)
            draws = int(r0 == 0) + int(r1 == 0)
            aw = a_black + a_white
            bw = b_white + b_black
            a_score = aw + 0.5 * draws
            b_score = bw + 0.5 * draws
            if a_score > b_score:
                ap, bp, mr = 3, 0, "A"
            elif b_score > a_score:
                ap, bp, mr = 0, 3, "B"
            else:
                ap, bp, mr = 1, 1, "DRAW"
            writer.writerow({
                "match_id": legacy._pair_id(a, b),
                "model_a": a.name,
                "update_a": a.update,
                "model_b": b.name,
                "update_b": b.update,
                "games": 2,
                "a_game_wins": aw,
                "draws": draws,
                "b_game_wins": bw,
                "a_wins_as_black": a_black,
                "a_wins_as_white": a_white,
                "b_wins_as_black": b_black,
                "b_wins_as_white": b_white,
                "a_game_score": a_score,
                "b_game_score": b_score,
                "a_points": ap,
                "b_points": bp,
                "match_result": mr,
                "elapsed_seconds": f"{per_pair_elapsed:.9f}",
            })


def _stats_from_matches(matches_path: Path, refs, *, pair_filter: np.ndarray | None = None):
    n = len(refs)
    name_to_idx = {r.name: i for i, r in enumerate(refs)}
    points = np.zeros(n, dtype=np.int64)
    matches = np.zeros(n, dtype=np.int64)
    wins = np.zeros(n, dtype=np.int64)
    draws = np.zeros(n, dtype=np.int64)
    losses = np.zeros(n, dtype=np.int64)
    game_score = np.zeros(n, dtype=np.float64)

    if matches_path.exists():
        with matches_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                ia = name_to_idx.get(row.get("model_a", ""))
                ib = name_to_idx.get(row.get("model_b", ""))
                if ia is None or ib is None or ia == ib:
                    continue
                if pair_filter is not None and not pair_filter[_pair_index(ia, ib, n)]:
                    continue
                ap = int(float(row["a_points"]))
                bp = int(float(row["b_points"]))
                points[ia] += ap
                points[ib] += bp
                matches[ia] += 1
                matches[ib] += 1
                aw = int(row["a_game_wins"])
                bw = int(row["b_game_wins"])
                dr = int(row["draws"])
                wins[ia] += aw
                losses[ia] += bw
                draws[ia] += dr
                wins[ib] += bw
                losses[ib] += aw
                draws[ib] += dr
                game_score[ia] += float(row["a_game_score"])
                game_score[ib] += float(row["b_game_score"])
    return points, matches, wins, draws, losses, game_score


def _select_group_finalists(
    matches_path: Path,
    refs,
    group_mask: np.ndarray,
    group_members: list[np.ndarray],
    advance_per_group: int,
) -> np.ndarray:
    points, _, wins, _, _, game_score = _stats_from_matches(
        matches_path, refs, pair_filter=group_mask
    )
    finalists: list[int] = []
    for group_no, members in enumerate(group_members, 1):
        ordered = sorted(
            (int(i) for i in members),
            key=lambda i: (points[i], game_score[i], wins[i], refs[i].update),
            reverse=True,
        )
        take = min(len(ordered), advance_per_group)
        selected = ordered[:take]
        finalists.extend(selected)
        leader = refs[selected[0]].name if selected else "--"
        print(
            f"[GROUP {group_no}] modele={len(ordered)} | awans={take} | lider={leader}"
        )
    return np.asarray(sorted(set(finalists)), dtype=np.int32)


def _write_ranking(
    matches_path: Path,
    ranking_path: Path,
    html_path: Path,
    refs,
    finalists: np.ndarray,
) -> None:
    points, matches, wins, draws, losses, game_score = _stats_from_matches(matches_path, refs)
    selected = [int(i) for i in finalists]
    order = sorted(
        selected,
        key=lambda i: (points[i], game_score[i], wins[i], refs[i].update),
        reverse=True,
    )
    fields = [
        "rank", "model", "update", "points", "matches", "game_wins", "draws",
        "game_losses", "game_score", "game_score_pct",
    ]
    rows = []
    for rank, i in enumerate(order, 1):
        games = wins[i] + draws[i] + losses[i]
        pct = 100.0 * game_score[i] / max(1, games)
        rows.append({
            "rank": rank,
            "model": refs[i].name,
            "update": refs[i].update,
            "points": int(points[i]),
            "matches": int(matches[i]),
            "game_wins": int(wins[i]),
            "draws": int(draws[i]),
            "game_losses": int(losses[i]),
            "game_score": float(game_score[i]),
            "game_score_pct": pct,
        })
    with ranking_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r[k]))}</td>" for k in fields) + "</tr>"
        for r in rows
    )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Connect6 Championship</title>"
        "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}"
        "td,th{padding:5px 9px;border:1px solid #ccc;text-align:right}"
        "th:nth-child(2),td:nth-child(2){text-align:left}</style>"
        f"<h1>Connect6 Championship</h1><p>Finaliści: {len(rows)}. "
        "Każdy finalista ma wynik przeciw całej puli checkpointów.</p>"
        "<table><thead><tr>"
        + "".join(f"<th>{html.escape(k)}</th>" for k in fields)
        + "</tr></thead><tbody>" + table_rows + "</tbody></table>",
        encoding="utf-8",
    )


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--:--"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _run_stage(
    stage_name: str,
    pending_pairs: np.ndarray,
    *,
    refs,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    matches_path: Path,
    extension,
    weights,
    norm_weights,
    norm_biases,
    policy,
    device: torch.device,
    slots: int,
    progress_chunks: int,
) -> dict:
    if pending_pairs.size == 0:
        print(f"[{stage_name}] brak nowych par do rozegrania")
        return {"pairs": 0, "games": 0, "native_seconds": 0.0, "wall_seconds": 0.0}

    games_np = np.empty(pending_pairs.size * 2, dtype=np.int32)
    games_np[0::2] = pending_pairs * 2
    games_np[1::2] = pending_pairs * 2 + 1
    total_run_pairs = int(pending_pairs.size)
    total_run_games = int(games_np.size)
    num_batches = min(max(1, progress_chunks), total_run_pairs)
    pair_bounds = np.linspace(0, total_run_pairs, num_batches + 1, dtype=np.int64)

    print(
        f"[{stage_name}] pary={total_run_pairs:,} | gry={total_run_games:,} | "
        f"porcje={num_batches}"
    )
    torch.cuda.synchronize(device)
    wall_started = time.perf_counter()
    native_elapsed = 0.0
    completed_total = 0

    for batch_index in range(num_batches):
        start = int(pair_bounds[batch_index])
        end = int(pair_bounds[batch_index + 1])
        if end <= start:
            continue
        pair_batch = pending_pairs[start:end]
        game_batch_np = games_np[2 * start: 2 * end]
        game_ids = torch.from_numpy(game_batch_np).to(device=device, dtype=torch.int32)
        torch.cuda.synchronize(device)
        batch_started = time.perf_counter()
        results_gpu, counters_gpu = extension.run_championship(
            weights,
            norm_weights,
            norm_biases,
            policy,
            game_ids,
            len(refs),
            slots,
        )
        batch_elapsed = time.perf_counter() - batch_started
        native_elapsed += batch_elapsed

        results = results_gpu.cpu().numpy().astype(np.int8, copy=False)
        counters = counters_gpu.cpu().numpy()
        batch_completed = int(counters[2])
        expected_games = int(game_batch_np.size)
        if batch_completed != expected_games:
            raise RuntimeError(
                f"{stage_name}: GPU zakończył {batch_completed}/{expected_games} gier "
                f"w porcji {batch_index + 1}/{num_batches}"
            )
        _append_match_rows(
            matches_path, refs, pair_i, pair_j, pair_batch, results, batch_elapsed
        )

        completed_total += batch_completed
        progress = 100.0 * completed_total / total_run_games
        avg_gps = completed_total / max(native_elapsed, 1e-9)
        eta = (total_run_games - completed_total) / max(avg_gps, 1e-9)
        print(
            f"[{stage_name} {progress:6.2f}%] {completed_total:,}/{total_run_games:,} gier | "
            f"avg={avg_gps:,.1f} g/s | ETA={_format_duration(eta)} | zapisano"
        )
        del game_ids, results_gpu, counters_gpu, results, counters

    wall_elapsed = time.perf_counter() - wall_started
    if completed_total != total_run_games:
        raise RuntimeError(f"{stage_name}: GPU zakończył {completed_total}/{total_run_games} gier")
    return {
        "pairs": total_run_pairs,
        "games": total_run_games,
        "native_seconds": native_elapsed,
        "wall_seconds": wall_elapsed,
    }


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = legacy._read_yaml(config_path)
    ch = cfg.get("championship", cfg)
    native = ch.get("native_gpu", {}) or {}

    if not torch.cuda.is_available():
        raise RuntimeError("Championship jest GPU-native i wymaga CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError(
            f"Ten kernel jest strojony pod SM120; wykryto {torch.cuda.get_device_capability(device)}"
        )
    if int(ch.get("games_per_pair", 2)) != 2:
        raise RuntimeError("Native championship wymaga dokładnie 2 gier na parę")
    if float(ch.get("temperature", 0.0)) != 0.0:
        raise RuntimeError("Native championship obsługuje temperature=0 (deterministyczny argmax)")

    groups = int(ch.get("groups", 4))
    advance_per_group = int(ch.get("advance_per_group", 200))
    if groups < 1:
        raise ValueError("championship.groups musi być >= 1")
    if advance_per_group < 1:
        raise ValueError("championship.advance_per_group musi być >= 1")

    root = _project_root(config_path)
    checkpoint_dir = legacy._resolve_path(root, ch["checkpoint_dir"])
    output_dir = legacy._resolve_path(root, ch["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = legacy.discover_checkpoints(checkpoint_dir)
    if len(refs) < 2:
        raise RuntimeError("Potrzeba co najmniej dwóch checkpointów")
    if groups > len(refs):
        groups = len(refs)

    slots = int(native.get("slots", 4096))
    chunk_size = int(native.get("load_chunk_models", 32))
    progress_chunks = max(1, int(native.get("progress_chunks", 100)))
    matches_path = output_dir / "matches.csv"
    ranking_path = output_dir / "ranking.csv"
    html_path = output_dir / "championship.html"
    state_path = output_dir / "state.json"
    plan_path = output_dir / "tournament_plan.json"

    legacy._validate_or_create_state(
        state_path,
        checkpoints=refs,
        games_per_pair=2,
        temperature=0.0,
    )

    n = len(refs)
    pair_i64, pair_j64 = np.triu_indices(n, 1)
    pair_i = pair_i64.astype(np.int32, copy=False)
    pair_j = pair_j64.astype(np.int32, copy=False)
    del pair_i64, pair_j64
    total_pairs = int(pair_i.size)

    group_mask, group_members = _group_pair_mask(n, groups)
    group_pairs = np.flatnonzero(group_mask).astype(np.int32, copy=False)
    done = _completed_pair_mask(matches_path, refs)
    pending_groups = group_pairs[~done[group_pairs]]

    print("\n" + "=" * 78)
    print("KING OF CONNECT6 — GPU NATIVE SM120 — CNN V6")
    print("=" * 78)
    print(
        f"Modele: {n} | grupy: {groups} | awans/grupa: {advance_per_group} | "
        f"pełne all-vs-all: {total_pairs:,} par"
    )
    print(
        f"Etap grupowy: {group_pairs.size:,} par ({pending_groups.size:,} pozostało) | "
        "przydział round-robin: 1→I, 2→II, ..."
    )
    print(f"GPU slots: {slots} | FP16 WMMA + GroupNorm V6 | argmax")

    extension = load_native_championship_extension(verbose=bool(native.get("compile_verbose", True)))
    weights, norm_weights, norm_biases, policy = _pack_checkpoints(
        refs, device, chunk_size=chunk_size
    )
    print(f"[NATIVE] packed weights ready | alloc={torch.cuda.memory_allocated(device)/2**30:.2f} GB")
    torch.cuda.reset_peak_memory_stats(device)

    group_run = _run_stage(
        "GROUPS",
        pending_groups,
        refs=refs,
        pair_i=pair_i,
        pair_j=pair_j,
        matches_path=matches_path,
        extension=extension,
        weights=weights,
        norm_weights=norm_weights,
        norm_biases=norm_biases,
        policy=policy,
        device=device,
        slots=slots,
        progress_chunks=progress_chunks,
    )

    done = _completed_pair_mask(matches_path, refs)
    if not bool(np.all(done[group_pairs])):
        missing = int(np.count_nonzero(~done[group_pairs]))
        raise RuntimeError(f"Etap grupowy nie jest kompletny: brakuje {missing:,} par")

    finalists = _select_group_finalists(
        matches_path, refs, group_mask, group_members, advance_per_group
    )
    final_mask = _final_pair_mask(n, finalists)
    final_pairs_total = int(np.count_nonzero(final_mask))
    pending_final = np.flatnonzero(final_mask & ~done).astype(np.int32, copy=False)
    already_final = final_pairs_total - int(pending_final.size)
    print(
        f"[FINAL] awansowało {finalists.size} modeli | każdy gra z całą pulą {n} modeli | "
        f"unikalne pary={final_pairs_total:,} | już policzone={already_final:,} | "
        f"pozostało={pending_final.size:,}"
    )

    final_run = _run_stage(
        "FINAL",
        pending_final,
        refs=refs,
        pair_i=pair_i,
        pair_j=pair_j,
        matches_path=matches_path,
        extension=extension,
        weights=weights,
        norm_weights=norm_weights,
        norm_biases=norm_biases,
        policy=policy,
        device=device,
        slots=slots,
        progress_chunks=progress_chunks,
    )

    done = _completed_pair_mask(matches_path, refs)
    if not bool(np.all(done[final_mask])):
        missing = int(np.count_nonzero(~done[final_mask]))
        raise RuntimeError(f"Finał nie jest kompletny: brakuje {missing:,} par")

    print("[POST] generuję ranking wyłącznie modeli, które wyszły z grup")
    _write_ranking(matches_path, ranking_path, html_path, refs, finalists)

    finalist_names = [refs[int(i)].name for i in finalists]
    plan = {
        "models": n,
        "groups": groups,
        "advance_per_group": advance_per_group,
        "group_sizes": [int(g.size) for g in group_members],
        "group_pairs": int(group_pairs.size),
        "finalists": finalist_names,
        "finalist_count": int(finalists.size),
        "final_pairs_total": final_pairs_total,
        "full_round_robin_pairs": total_pairs,
        "pairs_saved_vs_full_if_starting_empty": int(
            total_pairs - np.count_nonzero(group_mask | final_mask)
        ),
    }
    plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    peak = torch.cuda.max_memory_allocated(device) / 2**30
    native_seconds = float(group_run["native_seconds"] + final_run["native_seconds"])
    games_this_run = int(group_run["games"] + final_run["games"])
    meta = {
        "engine": "gpu_native_sm120_wmma_v6_groupnorm",
        "architecture_version": ARCHITECTURE_VERSION,
        "models": n,
        "groups": groups,
        "advance_per_group": advance_per_group,
        "finalists": int(finalists.size),
        "slots": slots,
        "games_completed_this_run": games_this_run,
        "native_seconds": native_seconds,
        "games_per_second": games_this_run / max(native_seconds, 1e-9),
        "peak_vram_gb": peak,
        "group_stage": group_run,
        "final_stage": final_run,
    }
    (output_dir / "native_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"[DONE] finaliści={finalists.size} | nowe gry={games_this_run:,} | "
        f"native={native_seconds:.2f}s | peak VRAM={peak:.2f} GB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect6 GPU-native SM120 championship V6")
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
