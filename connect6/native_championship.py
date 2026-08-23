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
from . import championship_cnn as cnn
from .cuda_native import load_native_championship_extension

EXPECTED_KERNELS = (23, 3, 3, 3, 3, 3, 3, 3)
EXPECTED_CHANNELS = (32, 32, 64, 64, 64, 96, 96, 96)
EXPECTED_KPAD = (1600, 288, 288, 576, 576, 576, 864, 864)


def _project_root(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _pair_index(i: int, j: int, n: int) -> int:
    if i > j:
        i, j = j, i
    return i * (2 * n - i - 1) // 2 + (j - i - 1)


def _validate_checkpoint_family(cp, *, first_cfg: dict | None = None) -> dict:
    cfg = dict(cp.model_config)
    cfg.pop("compile", None)
    cfg.pop("compile_mode", None)
    if int(cfg.get("architecture_version", 0)) != 4:
        raise RuntimeError("Native championship obsługuje wyłącznie CNN architecture_version=4")
    kernels = tuple(int(v) for v in cfg.get("kernels", EXPECTED_KERNELS))
    channels = tuple(int(v) for v in cfg.get("channels", EXPECTED_CHANNELS))
    if kernels != EXPECTED_KERNELS or channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Native SM120 kernel jest wyspecjalizowany dla kernels="
            f"{EXPECTED_KERNELS}, channels={EXPECTED_CHANNELS}; otrzymano {kernels}, {channels}."
        )
    if int(cp.game_config.get("board_size", 0)) != 19 or int(cp.game_config.get("win_length", 6)) != 6:
        raise RuntimeError("Native championship wymaga Connect6 19x19, win_length=6")
    if first_cfg is not None and cfg != first_cfg:
        raise RuntimeError("Checkpointy mają różne konfiguracje modelu")
    return cfg


def _allocate_packed_weights(num_models: int, device: torch.device):
    weights: list[torch.Tensor] = []
    biases: list[torch.Tensor] = []
    for out_ch, kpad in zip(EXPECTED_CHANNELS, EXPECTED_KPAD):
        # [model, output_channel, flattened_K]; this is column-major B when
        # interpreted as KxO by WMMA because each output column owns contiguous K.
        weights.append(torch.zeros((num_models, out_ch, kpad), device=device, dtype=torch.float16))
        biases.append(torch.empty((num_models, out_ch), device=device, dtype=torch.float16))
    policy = torch.empty((num_models, 96), device=device, dtype=torch.float16)
    return weights, biases, policy


def _pack_checkpoints(refs, device: torch.device, *, chunk_size: int = 32):
    num_models = len(refs)
    weights, biases, policy = _allocate_packed_weights(num_models, device)
    store = cnn.CNNCheckpointStore(cpu_cache_models=0)
    family_cfg: dict | None = None

    print(f"[NATIVE] pakuję {num_models} checkpointów bezpośrednio do layoutu WMMA FP16")
    for start in range(0, num_models, chunk_size):
        end = min(num_models, start + chunk_size)
        cps = [store.get(ref) for ref in refs[start:end]]
        for cp in cps:
            cfg = _validate_checkpoint_family(cp, first_cfg=family_cfg)
            if family_cfg is None:
                family_cfg = cfg

        for layer in range(8):
            raw = torch.stack([cp.model_state[f"convs.{layer}.weight"] for cp in cps], dim=0)
            raw = raw.to(device=device, dtype=torch.float16, non_blocking=False)
            raw = raw.reshape(end - start, EXPECTED_CHANNELS[layer], -1)
            kreal = raw.shape[-1]
            weights[layer][start:end, :, :kreal].copy_(raw)
            if kreal < EXPECTED_KPAD[layer]:
                weights[layer][start:end, :, kreal:].zero_()
            del raw

            b = torch.stack([cp.model_state[f"convs.{layer}.bias"] for cp in cps], dim=0)
            biases[layer][start:end].copy_(b.to(device=device, dtype=torch.float16))
            del b

        pw = torch.stack([cp.model_state["policy_output.weight"] for cp in cps], dim=0)
        pw = pw.reshape(end - start, 96)
        policy[start:end].copy_(pw.to(device=device, dtype=torch.float16))
        del pw, cps
        gc.collect()
        if end == num_models or end % max(chunk_size, 128) == 0:
            alloc = torch.cuda.memory_allocated(device) / 2**30
            print(f"[NATIVE LOAD] {end:>5}/{num_models} | packed VRAM={alloc:.2f} GB")

    torch.cuda.synchronize(device)
    return weights, biases, policy


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


def _make_pending_game_ids(matches_path: Path, refs) -> tuple[np.ndarray, np.ndarray]:
    done = _completed_pair_mask(matches_path, refs)
    pending_pairs = np.flatnonzero(~done).astype(np.int32, copy=False)
    games = np.empty(pending_pairs.size * 2, dtype=np.int32)
    games[0::2] = pending_pairs * 2
    games[1::2] = pending_pairs * 2 + 1
    return pending_pairs, games


def _append_match_rows(
    matches_path: Path,
    refs,
    pending_pairs: np.ndarray,
    results: np.ndarray,
    elapsed: float,
) -> None:
    n = len(refs)
    pi, pj = np.triu_indices(n, 1)
    pi = pi.astype(np.int32, copy=False)
    pj = pj.astype(np.int32, copy=False)
    game_results = results.reshape(-1, 2)
    fields = legacy.MATCH_FIELDS
    exists = matches_path.exists() and matches_path.stat().st_size > 0
    per_pair_elapsed = elapsed / max(1, pending_pairs.size)

    with matches_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for k, p in enumerate(pending_pairs):
            i = int(pi[p]); j = int(pj[p])
            a = refs[i]; b = refs[j]
            r0 = int(game_results[k, 0])  # A black, B white
            r1 = int(game_results[k, 1])  # B black, A white
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


def _write_ranking(matches_path: Path, ranking_path: Path, html_path: Path, refs) -> None:
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
                ia = name_to_idx.get(row["model_a"]); ib = name_to_idx.get(row["model_b"])
                if ia is None or ib is None:
                    continue
                ap = int(float(row["a_points"])); bp = int(float(row["b_points"]))
                points[ia] += ap; points[ib] += bp
                matches[ia] += 1; matches[ib] += 1
                aw = int(row["a_game_wins"]); bw = int(row["b_game_wins"]); dr = int(row["draws"])
                wins[ia] += aw; losses[ia] += bw; draws[ia] += dr
                wins[ib] += bw; losses[ib] += aw; draws[ib] += dr
                game_score[ia] += float(row["a_game_score"])
                game_score[ib] += float(row["b_game_score"])

    order = sorted(range(n), key=lambda i: (points[i], game_score[i], wins[i], refs[i].update), reverse=True)
    fields = ["rank", "model", "update", "points", "matches", "game_wins", "draws", "game_losses", "game_score", "game_score_pct"]
    rows = []
    for rank, i in enumerate(order, 1):
        games = wins[i] + draws[i] + losses[i]
        pct = 100.0 * game_score[i] / max(1, games)
        rows.append({
            "rank": rank, "model": refs[i].name, "update": refs[i].update,
            "points": int(points[i]), "matches": int(matches[i]),
            "game_wins": int(wins[i]), "draws": int(draws[i]), "game_losses": int(losses[i]),
            "game_score": float(game_score[i]), "game_score_pct": pct,
        })
    with ranking_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

    table_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(r[k]))}</td>" for k in fields) + "</tr>"
        for r in rows[:500]
    )
    html_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>Connect6 Championship</title>"
        "<style>body{font-family:system-ui;margin:24px}table{border-collapse:collapse}td,th{padding:5px 9px;border:1px solid #ccc;text-align:right}th:nth-child(2),td:nth-child(2){text-align:left}</style>"
        "<h1>Connect6 Championship</h1><p>GPU-native SM120 engine. Top 500.</p><table><thead><tr>"
        + "".join(f"<th>{html.escape(k)}</th>" for k in fields)
        + "</tr></thead><tbody>" + table_rows + "</tbody></table>",
        encoding="utf-8",
    )


def run(config_path: str | Path) -> None:
    config_path = Path(config_path).resolve()
    cfg = cnn._ORIGINAL_READ_YAML(config_path)
    ch = cfg.get("championship", cfg)
    native = ch.get("native_gpu", {}) or {}

    if not torch.cuda.is_available():
        raise RuntimeError("Championship jest teraz GPU-native i wymaga CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError(
            f"Ten kernel jest strojony pod SM120; wykryto {torch.cuda.get_device_capability(device)}"
        )
    if int(ch.get("games_per_pair", 2)) != 2:
        raise RuntimeError("Native championship jest zoptymalizowany dla dokładnie 2 gier na parę")
    if float(ch.get("temperature", 0.0)) != 0.0:
        raise RuntimeError("Native championship obsługuje temperature=0 (deterministyczny argmax)")

    root = _project_root(config_path)
    checkpoint_dir = legacy._resolve_path(root, ch["checkpoint_dir"])
    output_dir = legacy._resolve_path(root, ch["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    refs = legacy.discover_checkpoints(checkpoint_dir)
    if len(refs) < 2:
        raise RuntimeError("Potrzeba co najmniej dwóch checkpointów")

    slots = int(native.get("slots", 4096))
    chunk_size = int(native.get("load_chunk_models", 32))
    matches_path = output_dir / "matches.csv"
    ranking_path = output_dir / "ranking.csv"
    html_path = output_dir / "championship.html"
    state_path = output_dir / "state.json"
    legacy._validate_or_create_state(
        state_path,
        checkpoints=refs,
        games_per_pair=2,
        temperature=0.0,
    )

    pending_pairs, game_ids_np = _make_pending_game_ids(matches_path, refs)
    total_pairs = len(refs) * (len(refs) - 1) // 2
    print("\n" + "=" * 78)
    print("KING OF CONNECT6 — GPU NATIVE SM120")
    print("=" * 78)
    print(f"Modele: {len(refs)} | pary: {total_pairs:,} | pozostało: {pending_pairs.size:,}")
    print(f"GPU slots: {slots} | jobs: {game_ids_np.size:,} | FP16 WMMA | zero host sync per move")
    if game_ids_np.size == 0:
        print("Turniej już ukończony.")
        _write_ranking(matches_path, ranking_path, html_path, refs)
        return

    extension = load_native_championship_extension(verbose=bool(native.get("compile_verbose", True)))
    weights, biases, policy = _pack_checkpoints(refs, device, chunk_size=chunk_size)
    packed_gb = torch.cuda.memory_allocated(device) / 2**30
    print(f"[NATIVE] packed weights ready | alloc={packed_gb:.2f} GB")

    game_ids = torch.from_numpy(game_ids_np).to(device=device, dtype=torch.int32)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    results_gpu, counters_gpu = extension.run_championship(
        weights, biases, policy, game_ids, len(refs), slots
    )
    # Native function returns only after the device-side conditional graph has ended.
    elapsed = time.perf_counter() - started
    results = results_gpu.cpu().numpy().astype(np.int8, copy=False)
    counters = counters_gpu.cpu().numpy()
    completed = int(counters[2])
    if completed != game_ids_np.size:
        raise RuntimeError(f"GPU zakończył {completed}/{game_ids_np.size} gier")

    gps = completed / max(elapsed, 1e-9)
    peak = torch.cuda.max_memory_allocated(device) / 2**30
    print(f"[NATIVE DONE] {completed:,} gier | {elapsed:.2f}s | {gps:,.1f} gier/s | peak VRAM={peak:.2f} GB")

    print("[POST] zapis matches.csv i ranking — CPU pracuje dopiero po zakończeniu GPU")
    _append_match_rows(matches_path, refs, pending_pairs, results, elapsed)
    _write_ranking(matches_path, ranking_path, html_path, refs)
    meta = {
        "engine": "gpu_native_sm120_wmma_v1",
        "models": len(refs),
        "slots": slots,
        "games_completed_this_run": completed,
        "elapsed_seconds": elapsed,
        "games_per_second": gps,
        "peak_vram_gb": peak,
    }
    (output_dir / "native_run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect6 GPU-native SM120 championship")
    parser.add_argument("--config", default="configs/championship.yaml")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
