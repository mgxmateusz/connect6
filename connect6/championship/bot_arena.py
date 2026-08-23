from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import torch

from . import championship as legacy
from .championship_cnn import _black_to_move, _masked_step
from .gpu_bot import GPUTacticalBot
from .history import HistoricalPolicyEnsemble, _load_lean_checkpoint
from .model import mask_logits
from .native_championship import _validate_checkpoint_family
from .vector_env import VectorConnect6


BOT_SIGNATURE = "gpu_tactical_bot_heuristic_v1"
RESULT_FIELDS = [
    "model",
    "update",
    "model_black_result",
    "model_white_result",
    "model_wins",
    "draws",
    "bot_wins",
    "model_wins_as_black",
    "model_wins_as_white",
    "bot_wins_as_black",
    "bot_wins_as_white",
    "model_black_moves",
    "model_white_moves",
    "model_score",
    "model_score_pct",
    "bot_score_pct",
    "elapsed_seconds",
]


def _project_root(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _dtype_from_name(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if value in {"fp16", "float16", "half"}:
        return torch.float16
    if value in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Nieznany inference_dtype: {name}")


def _result_from_winner(winner: int, *, model_is_black: bool) -> str:
    if winner == 0:
        return "DRAW"
    model_won = (winner == 1 and model_is_black) or (winner == -1 and not model_is_black)
    return "WIN" if model_won else "LOSS"


def _score_from_result(result: str) -> float:
    if result == "WIN":
        return 1.0
    if result == "DRAW":
        return 0.5
    return 0.0


def _read_results(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _completed_names(path: Path) -> set[str]:
    return {row.get("model", "") for row in _read_results(path) if row.get("model")}


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def _validate_state(path: Path, *, checkpoint_dir: Path) -> None:
    expected = {
        "format_version": 1,
        "bot_signature": BOT_SIGNATURE,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "games_per_model": 2,
        "temperature": 0.0,
    }
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if current.get(key) != value:
                raise RuntimeError(
                    f"Niezgodny stan bot arena: {key}={current.get(key)!r}, oczekiwano {value!r}. "
                    "Usuń katalog wynikowy albo uruchom z --reset."
                )
        return
    path.write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")


@torch.inference_mode()
def _play_colour_game(
    ensemble: HistoricalPolicyEnsemble,
    bot: GPUTacticalBot,
    *,
    model_is_black: bool,
    sync_interval_moves: int,
) -> tuple[list[int], list[int], float]:
    """Play one game per model in parallel, all with the same colour assignment.

    Board i is controlled by model i whenever that colour is to move. The other
    colour is the fixed tactical CUDA bot. All board state stays on the GPU;
    host synchronisation is only used every few moves to stop once all games ended.
    """
    device = ensemble.device
    n = ensemble.num_models
    env = VectorConnect6(
        num_envs=n,
        board_size=ensemble.board_size,
        win_length=6,
        device=device,
        debug_checks=False,
    )
    env.reset()
    active = torch.ones(n, dtype=torch.bool, device=device)
    winners = torch.zeros(n, dtype=torch.int8, device=device)
    sync_every = max(1, int(sync_interval_moves))

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()

    for move_index in range(env.action_size):
        black_to_move = _black_to_move(move_index)
        model_to_move = black_to_move == bool(model_is_black)

        if model_to_move:
            x = env.network_input()
            logits = ensemble.forward_grouped(x.unsqueeze(1)).squeeze(1)
            logits = mask_logits(logits.float(), env.legal_mask())
            actions = logits.argmax(dim=1)
        else:
            actions = bot.actions(env.boards, env.current_player, env.stones_left)

        done, winner = _masked_step(env, actions, active)
        newly_done = active & done
        winners = torch.where(newly_done, winner, winners)
        active &= ~done

        if (move_index + 1) % sync_every == 0:
            if not bool(active.any().item()):
                break

    if bool(active.any().item()):
        raise RuntimeError("Nie wszystkie partie bot arena zakończyły się po 361 kamieniach")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    winner_list = [int(v) for v in winners.cpu().tolist()]
    move_list = [int(v) for v in env.move_count.cpu().tolist()]
    return winner_list, move_list, elapsed


def _make_chunk_rows(
    refs,
    black_winners: list[int],
    black_moves: list[int],
    white_winners: list[int],
    white_moves: list[int],
    elapsed: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    per_model_elapsed = elapsed / max(1, len(refs))
    for i, ref in enumerate(refs):
        black_result = _result_from_winner(black_winners[i], model_is_black=True)
        white_result = _result_from_winner(white_winners[i], model_is_black=False)
        results = (black_result, white_result)
        model_wins = sum(r == "WIN" for r in results)
        draws = sum(r == "DRAW" for r in results)
        bot_wins = 2 - model_wins - draws
        model_score = sum(_score_from_result(r) for r in results)
        rows.append(
            {
                "model": ref.name,
                "update": int(ref.update),
                "model_black_result": black_result,
                "model_white_result": white_result,
                "model_wins": model_wins,
                "draws": draws,
                "bot_wins": bot_wins,
                "model_wins_as_black": int(black_result == "WIN"),
                "model_wins_as_white": int(white_result == "WIN"),
                "bot_wins_as_black": int(white_result == "LOSS"),
                "bot_wins_as_white": int(black_result == "LOSS"),
                "model_black_moves": int(black_moves[i]),
                "model_white_moves": int(white_moves[i]),
                "model_score": f"{model_score:.1f}",
                "model_score_pct": f"{100.0 * model_score / 2.0:.2f}",
                "bot_score_pct": f"{100.0 - 100.0 * model_score / 2.0:.2f}",
                "elapsed_seconds": f"{per_model_elapsed:.9f}",
            }
        )
    return rows


def _to_numeric_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        try:
            rows.append(
                {
                    **raw,
                    "update": int(raw["update"]),
                    "model_wins": int(raw["model_wins"]),
                    "draws": int(raw["draws"]),
                    "bot_wins": int(raw["bot_wins"]),
                    "model_wins_as_black": int(raw["model_wins_as_black"]),
                    "model_wins_as_white": int(raw["model_wins_as_white"]),
                    "bot_wins_as_black": int(raw["bot_wins_as_black"]),
                    "bot_wins_as_white": int(raw["bot_wins_as_white"]),
                    "model_black_moves": int(raw["model_black_moves"]),
                    "model_white_moves": int(raw["model_white_moves"]),
                    "model_score": float(raw["model_score"]),
                    "model_score_pct": float(raw["model_score_pct"]),
                    "bot_score_pct": float(raw["bot_score_pct"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    rows.sort(key=lambda row: (row["update"], row["model"]))
    return rows


def _rolling(rows: list[dict[str, Any]], window: int = 25) -> None:
    score_window: deque[float] = deque()
    bot_win_window: deque[int] = deque()
    score_sum = 0.0
    bot_win_sum = 0
    for row in rows:
        score_window.append(float(row["model_score"]))
        bot_win_window.append(int(row["bot_wins"]))
        score_sum += float(row["model_score"])
        bot_win_sum += int(row["bot_wins"])
        if len(score_window) > window:
            score_sum -= score_window.popleft()
            bot_win_sum -= bot_win_window.popleft()
        games = 2 * len(score_window)
        row["rolling_model_score_pct"] = 100.0 * score_sum / max(1, games)
        row["rolling_bot_win_pct"] = 100.0 * bot_win_sum / max(1, games)
        row["rolling_window"] = len(score_window)


def _first_update(rows: list[dict[str, Any]], predicate) -> int | None:
    for row in rows:
        if predicate(row):
            return int(row["update"])
    return None


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "models": 0,
            "games": 0,
            "bot_signature": BOT_SIGNATURE,
        }

    _rolling(rows, 25)
    n = len(rows)
    games = 2 * n
    model_wins = sum(int(r["model_wins"]) for r in rows)
    draws = sum(int(r["draws"]) for r in rows)
    bot_wins = sum(int(r["bot_wins"]) for r in rows)

    bot_black_wins = sum(int(r["bot_wins_as_black"]) for r in rows)
    bot_white_wins = sum(int(r["bot_wins_as_white"]) for r in rows)
    bot_black_draws = sum(r["model_white_result"] == "DRAW" for r in rows)
    bot_white_draws = sum(r["model_black_result"] == "DRAW" for r in rows)
    bot_black_losses = sum(int(r["model_wins_as_white"]) for r in rows)
    bot_white_losses = sum(int(r["model_wins_as_black"]) for r in rows)

    last_bot_win_idx = -1
    for idx, row in enumerate(rows):
        if int(row["bot_wins"]) > 0:
            last_bot_win_idx = idx
    if last_bot_win_idx < 0:
        stable_no_bot_win_from = int(rows[0]["update"])
    elif last_bot_win_idx + 1 < len(rows):
        stable_no_bot_win_from = int(rows[last_bot_win_idx + 1]["update"])
    else:
        stable_no_bot_win_from = None

    full_windows = [r for r in rows if int(r.get("rolling_window", 0)) >= 25]
    first_roll50 = _first_update(full_windows, lambda r: r["rolling_model_score_pct"] >= 50.0)
    first_roll75 = _first_update(full_windows, lambda r: r["rolling_model_score_pct"] >= 75.0)

    latest = rows[-1]
    return {
        "bot_signature": BOT_SIGNATURE,
        "models": n,
        "games": games,
        "model_wins": model_wins,
        "draws": draws,
        "bot_wins": bot_wins,
        "model_win_pct": 100.0 * model_wins / games,
        "draw_pct": 100.0 * draws / games,
        "bot_win_pct": 100.0 * bot_wins / games,
        "model_score_pct": 100.0 * (model_wins + 0.5 * draws) / games,
        "bot_score_pct": 100.0 * (bot_wins + 0.5 * draws) / games,
        "bot_as_black": {
            "games": n,
            "wins": bot_black_wins,
            "draws": bot_black_draws,
            "losses": bot_black_losses,
            "win_pct": 100.0 * bot_black_wins / n,
            "score_pct": 100.0 * (bot_black_wins + 0.5 * bot_black_draws) / n,
        },
        "bot_as_white": {
            "games": n,
            "wins": bot_white_wins,
            "draws": bot_white_draws,
            "losses": bot_white_losses,
            "win_pct": 100.0 * bot_white_wins / n,
            "score_pct": 100.0 * (bot_white_wins + 0.5 * bot_white_draws) / n,
        },
        "first_model_win_update": _first_update(rows, lambda r: int(r["model_wins"]) > 0),
        "first_model_sweep_update": _first_update(rows, lambda r: int(r["model_wins"]) == 2),
        "first_checkpoint_without_bot_win": _first_update(rows, lambda r: int(r["bot_wins"]) == 0),
        "last_bot_win_update": None if last_bot_win_idx < 0 else int(rows[last_bot_win_idx]["update"]),
        "stable_no_bot_win_from_update": stable_no_bot_win_from,
        "first_rolling25_model_score_ge_50_update": first_roll50,
        "first_rolling25_model_score_ge_75_update": first_roll75,
        "latest_update": int(latest["update"]),
        "latest_model_black_result": latest["model_black_result"],
        "latest_model_white_result": latest["model_white_result"],
        "latest_model_score_pct": float(latest["model_score_pct"]),
        "latest_rolling25_model_score_pct": float(latest["rolling_model_score_pct"]),
    }


def _fmt_pct(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.1f}%"


def _fmt_update(value: int | None) -> str:
    return "—" if value is None else str(int(value))


def _svg_chart(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>Brak danych do wykresu.</p>"
    width, height = 1180, 330
    left, right, top, bottom = 62, 24, 22, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    u0 = min(r["update"] for r in rows)
    u1 = max(r["update"] for r in rows)
    span = max(1, u1 - u0)

    def xy(update: int, pct: float) -> tuple[float, float]:
        x = left + (update - u0) / span * plot_w
        y = top + (100.0 - max(0.0, min(100.0, pct))) / 100.0 * plot_h
        return x, y

    raw_points = " ".join(
        f"{xy(int(r['update']), float(r['model_score_pct']))[0]:.2f},{xy(int(r['update']), float(r['model_score_pct']))[1]:.2f}"
        for r in rows
    )
    roll_points = " ".join(
        f"{xy(int(r['update']), float(r['rolling_model_score_pct']))[0]:.2f},{xy(int(r['update']), float(r['rolling_model_score_pct']))[1]:.2f}"
        for r in rows
    )
    grid = []
    for pct in (0, 25, 50, 75, 100):
        _, y = xy(u0, pct)
        grid.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width-right}' y2='{y:.2f}' class='grid'/><text x='{left-10}' y='{y+4:.2f}' text-anchor='end'>{pct}%</text>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}' role='img' aria-label='Wynik modeli przeciw botowi po update'>"
        + "".join(grid)
        + f"<polyline points='{raw_points}' class='raw-line'/><polyline points='{roll_points}' class='roll-line'/>"
        + f"<text x='{left}' y='{height-10}'>update {u0}</text><text x='{width-right}' y='{height-10}' text-anchor='end'>update {u1}</text>"
        + "</svg>"
    )


def _write_reports(results_path: Path, output_dir: Path) -> dict[str, Any]:
    rows = _to_numeric_rows(_read_results(results_path))
    summary = _summary(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not rows:
        return summary

    cards = [
        ("Bot wygrywa", _fmt_pct(summary["bot_win_pct"]), f"{summary['bot_wins']}/{summary['games']} gier"),
        ("Bot jako czarny", _fmt_pct(summary["bot_as_black"]["win_pct"]), f"score {_fmt_pct(summary['bot_as_black']['score_pct'])}"),
        ("Bot jako biały", _fmt_pct(summary["bot_as_white"]["win_pct"]), f"score {_fmt_pct(summary['bot_as_white']['score_pct'])}"),
        ("Pierwsza wygrana modelu", _fmt_update(summary["first_model_win_update"]), "update"),
        ("Pierwsze 2:0 modelu", _fmt_update(summary["first_model_sweep_update"]), "update"),
        ("Bot nie wygrywa od", _fmt_update(summary["stable_no_bot_win_from_update"]), "update, jeśli utrzymuje się do końca"),
        ("Najnowszy rolling-25", _fmt_pct(summary["latest_rolling25_model_score_pct"]), "score modelu"),
        ("Najnowszy model", _fmt_pct(summary["latest_model_score_pct"]), f"update {summary['latest_update']}"),
    ]
    card_html = "".join(
        f"<div class='card'><div class='label'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div><div class='sub'>{html.escape(sub)}</div></div>"
        for label, value, sub in cards
    )

    table_rows = []
    for r in reversed(rows):
        table_rows.append(
            "<tr>"
            f"<td>{r['update']}</td>"
            f"<td>{html.escape(str(r['model']))}</td>"
            f"<td class='{str(r['model_black_result']).lower()}'>{r['model_black_result']}</td>"
            f"<td class='{str(r['model_white_result']).lower()}'>{r['model_white_result']}</td>"
            f"<td>{float(r['model_score_pct']):.1f}%</td>"
            f"<td>{float(r['rolling_model_score_pct']):.1f}%</td>"
            f"<td>{r['bot_wins']}</td>"
            f"<td>{r['model_black_moves']}</td>"
            f"<td>{r['model_white_moves']}</td>"
            "</tr>"
        )

    report = f"""<!doctype html>
<meta charset='utf-8'>
<title>Connect6 Bot Arena</title>
<style>
:root{{--bg:#0f131a;--panel:#171d26;--panel2:#202836;--text:#edf2f7;--muted:#9aa7b5;--border:#313b49;--good:#51c878;--bad:#ff6b6b;--draw:#e5ba54;--accent:#6ea1ff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,Segoe UI,sans-serif}}
main{{max-width:1280px;margin:0 auto;padding:28px}} h1{{margin:0 0 4px;font-size:28px}} .muted{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:22px 0}}
.card{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}} .label,.sub{{color:var(--muted)}} .value{{font-size:25px;font-weight:750;margin:3px 0}}
.chart{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:12px;margin:14px 0 20px;overflow-x:auto}} svg{{width:100%;min-width:760px;height:auto}} svg text{{fill:var(--muted);font-size:12px}} .grid{{stroke:#34404f;stroke-width:1}} .raw-line{{fill:none;stroke:#657388;stroke-width:1;opacity:.55}} .roll-line{{fill:none;stroke:var(--accent);stroke-width:3}}
.legend{{display:flex;gap:18px;color:var(--muted);margin:8px 0}} .legend b{{color:var(--text)}}
table{{width:100%;border-collapse:collapse;background:var(--panel);border:1px solid var(--border)}} th,td{{padding:7px 9px;border-bottom:1px solid var(--border);text-align:right}} th{{position:sticky;top:0;background:var(--panel2);color:var(--muted)}} th:nth-child(2),td:nth-child(2){{text-align:left}} .win{{color:var(--good);font-weight:700}} .loss{{color:var(--bad);font-weight:700}} .draw{{color:var(--draw);font-weight:700}}
.table-wrap{{max-height:720px;overflow:auto;border-radius:10px}}
</style>
<main>
<h1>Connect6 — autosave vs GPU Tactical Bot</h1>
<div class='muted'>{len(rows)} modeli • {2*len(rows)} gier • dokładnie 2 gry/model: model raz czarny, raz biały • bot {html.escape(BOT_SIGNATURE)}</div>
<div class='cards'>{card_html}</div>
<div class='chart'><div class='legend'><span><b>Szary:</b> wynik konkretnego checkpointu (2 gry)</span><span><b>Niebieski:</b> rolling score z 25 checkpointów</span></div>{_svg_chart(rows)}</div>
<div class='table-wrap'><table><thead><tr><th>update</th><th>model</th><th>model czarny</th><th>model biały</th><th>score modelu</th><th>rolling-25</th><th>wygrane bota</th><th>ruchy (M czarny)</th><th>ruchy (M biały)</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</main>"""
    (output_dir / "bot_arena.html").write_text(report, encoding="utf-8")
    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    if not summary.get("models"):
        print("Brak wyników.")
        return
    print("\n" + "=" * 78)
    print("BOT ARENA — PODSUMOWANIE")
    print("=" * 78)
    print(
        f"Modele: {summary['models']:,} | gry: {summary['games']:,} | "
        f"bot W/D/L: {summary['bot_wins']}/{summary['draws']}/{summary['model_wins']} | "
        f"bot win={summary['bot_win_pct']:.2f}% | bot score={summary['bot_score_pct']:.2f}%"
    )
    print(
        f"Bot czarny: win={summary['bot_as_black']['win_pct']:.2f}% score={summary['bot_as_black']['score_pct']:.2f}% | "
        f"bot biały: win={summary['bot_as_white']['win_pct']:.2f}% score={summary['bot_as_white']['score_pct']:.2f}%"
    )
    print(
        "Pierwsza wygrana modelu: update " + _fmt_update(summary.get("first_model_win_update"))
        + " | pierwsze 2:0: update " + _fmt_update(summary.get("first_model_sweep_update"))
    )
    print(
        "Ostatnia wygrana bota: update " + _fmt_update(summary.get("last_bot_win_update"))
        + " | bot nie wygrywa już od: update " + _fmt_update(summary.get("stable_no_bot_win_from_update"))
    )
    print(
        f"Najnowszy update {summary['latest_update']}: {summary['latest_model_score_pct']:.1f}% score modelu | "
        f"rolling-25={summary['latest_rolling25_model_score_pct']:.1f}%"
    )


def run(config_path: str | Path, *, reset: bool = False, models_per_batch_override: int | None = None) -> None:
    config_path = Path(config_path).resolve()
    cfg = legacy._read_yaml(config_path)
    arena = cfg.get("bot_arena", cfg)
    root = _project_root(config_path)

    if not torch.cuda.is_available():
        raise RuntimeError("Bot arena wymaga CUDA")
    device = torch.device(arena.get("device", "cuda"))
    if device.type != "cuda":
        raise RuntimeError("Bot arena jest trybem GPU i wymaga device=cuda")

    checkpoint_dir = legacy._resolve_path(root, arena["checkpoint_dir"])
    output_dir = legacy._resolve_path(root, arena["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "bot_matches.csv"
    state_path = output_dir / "state.json"

    if reset:
        for path in (results_path, state_path, output_dir / "summary.json", output_dir / "bot_arena.html"):
            path.unlink(missing_ok=True)

    _validate_state(state_path, checkpoint_dir=checkpoint_dir)
    refs = legacy.discover_checkpoints(checkpoint_dir)
    if not refs:
        raise RuntimeError(f"Nie znaleziono model_update_*.pt w {checkpoint_dir}")

    completed = _completed_names(results_path)
    pending = [ref for ref in refs if ref.name not in completed]
    dtype = _dtype_from_name(arena.get("inference_dtype", "bfloat16"))
    batch_size = int(models_per_batch_override or arena.get("models_per_batch", 256))
    batch_size = max(1, batch_size)
    sync_interval = max(1, int(arena.get("sync_interval_moves", 16)))

    print("\n" + "=" * 78)
    print("CONNECT6 BOT ARENA — ALL AUTOSAVES VS GPU TACTICAL BOT")
    print("=" * 78)
    print(f"Checkpointy: {len(refs):,} | gotowe: {len(completed):,} | pozostało: {len(pending):,}")
    print(f"Każdy model: 2 gry (raz czarny, raz biały) | batch modeli: {batch_size}")
    print(f"GPU: {torch.cuda.get_device_name(device)} | CNN dtype: {dtype} | bot: {BOT_SIGNATURE}")

    if not pending:
        summary = _write_reports(results_path, output_dir)
        _print_summary(summary)
        print(f"HTML: {output_dir / 'bot_arena.html'}")
        return

    bot = GPUTacticalBot(device)
    family_cfg: dict[str, Any] | None = None
    total_started = time.perf_counter()
    completed_this_run = 0

    for chunk_index, start in enumerate(range(0, len(pending), batch_size), 1):
        chunk_refs = pending[start : start + batch_size]
        load_started = time.perf_counter()
        checkpoints = [_load_lean_checkpoint(ref.path) for ref in chunk_refs]
        for cp in checkpoints:
            cfg_norm = _validate_checkpoint_family(cp, first_cfg=family_cfg)
            if family_cfg is None:
                family_cfg = cfg_norm
        load_elapsed = time.perf_counter() - load_started

        try:
            ensemble = HistoricalPolicyEnsemble(checkpoints, device=device, dtype=dtype)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            raise RuntimeError(
                f"OOM przy batchu {len(chunk_refs)} modeli. Zmniejsz bot_arena.models_per_batch "
                "np. do połowy albo użyj --models-per-batch."
            ) from exc

        # Warm-up one model-side forward before timing real games so cuDNN can
        # select/cache convolution algorithms outside the match timing.
        warm_env = VectorConnect6(len(chunk_refs), 19, 6, device=device, debug_checks=False)
        warm_x = warm_env.network_input()
        ensemble.forward_grouped(warm_x.unsqueeze(1)).squeeze(1)
        del warm_env, warm_x
        torch.cuda.synchronize(device)

        black_winners, black_moves, black_elapsed = _play_colour_game(
            ensemble,
            bot,
            model_is_black=True,
            sync_interval_moves=sync_interval,
        )
        white_winners, white_moves, white_elapsed = _play_colour_game(
            ensemble,
            bot,
            model_is_black=False,
            sync_interval_moves=sync_interval,
        )
        game_elapsed = black_elapsed + white_elapsed
        rows = _make_chunk_rows(
            chunk_refs,
            black_winners,
            black_moves,
            white_winners,
            white_moves,
            game_elapsed,
        )
        _append_rows(results_path, rows)
        completed_this_run += len(chunk_refs)

        chunk_bot_wins = sum(int(r["bot_wins"]) for r in rows)
        chunk_model_wins = sum(int(r["model_wins"]) for r in rows)
        chunk_draws = sum(int(r["draws"]) for r in rows)
        games = 2 * len(chunk_refs)
        gps = games / max(game_elapsed, 1e-9)
        progress = 100.0 * completed_this_run / len(pending)
        wall = time.perf_counter() - total_started
        print(
            f"[{progress:6.2f}%] modele {completed_this_run:,}/{len(pending):,} | "
            f"gry={games:,} | {gps:,.1f} g/s | bot/model/draw={chunk_bot_wins}/{chunk_model_wins}/{chunk_draws} | "
            f"load={load_elapsed:.2f}s | play={game_elapsed:.2f}s | wall={wall:.2f}s"
        )

        del ensemble, checkpoints, black_winners, white_winners, black_moves, white_moves, rows
        gc.collect()
        torch.cuda.empty_cache()

        # Rebuild HTML/summary after every chunk. If the process is interrupted,
        # all completed model pairs remain immediately inspectable and resumable.
        _write_reports(results_path, output_dir)

    summary = _write_reports(results_path, output_dir)
    _print_summary(summary)
    total_elapsed = time.perf_counter() - total_started
    print(f"[DONE] {completed_this_run:,} nowych modeli | wall={total_elapsed:.2f}s")
    print(f"CSV:  {results_path}")
    print(f"HTML: {output_dir / 'bot_arena.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6: każdy autosave CNN gra 2 gry przeciw GPU Tactical Bot"
    )
    parser.add_argument("--config", default="configs/bot_arena.yaml")
    parser.add_argument("--reset", action="store_true", help="Usuń poprzednie wyniki i policz od zera")
    parser.add_argument("--models-per-batch", type=int, default=None)
    args = parser.parse_args()
    run(args.config, reset=args.reset, models_per_batch_override=args.models_per_batch)


if __name__ == "__main__":
    main()
