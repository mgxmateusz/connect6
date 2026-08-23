from __future__ import annotations

import argparse
import csv
import gc
import html
import json
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from . import championship as legacy
from .championship_cnn import _black_to_move, _masked_step
from .gpu_bot import GPUTacticalBot, GPUTacticalBotV2
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


@dataclass(frozen=True)
class BotSpec:
    key: str
    label: str
    signature: str
    cls: type


BOT_SPECS = (
    BotSpec("v1", "GPU Tactical Bot V1", BOT_SIGNATURE, GPUTacticalBot),
    BotSpec("v2", "GPU Tactical Bot V2", "gpu_tactical_bot_heuristic_v2", GPUTacticalBotV2),
)


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


def _validate_state(path: Path, *, checkpoint_dir: Path, bot_signature: str) -> None:
    expected = {
        "format_version": 2,
        "bot_signature": bot_signature,
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
    bot,
    *,
    model_is_black: bool,
    sync_interval_moves: int,
) -> tuple[list[int], list[int], float]:
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

        if (move_index + 1) % sync_every == 0 and not bool(active.any().item()):
            break

    if bool(active.any().item()):
        raise RuntimeError("Nie wszystkie partie bot arena zakończyły się po 361 kamieniach")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        [int(v) for v in winners.cpu().tolist()],
        [int(v) for v in env.move_count.cpu().tolist()],
        elapsed,
    )


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


def _summary(
    rows: list[dict[str, Any]],
    bot_signature: str = BOT_SIGNATURE,
) -> dict[str, Any]:
    if not rows:
        return {"models": 0, "games": 0, "bot_signature": bot_signature}

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
    latest = rows[-1]
    return {
        "bot_signature": bot_signature,
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
        "first_rolling25_model_score_ge_50_update": _first_update(
            full_windows, lambda r: r["rolling_model_score_pct"] >= 50.0
        ),
        "first_rolling25_model_score_ge_75_update": _first_update(
            full_windows, lambda r: r["rolling_model_score_pct"] >= 75.0
        ),
        "latest_update": int(latest["update"]),
        "latest_model_black_result": latest["model_black_result"],
        "latest_model_white_result": latest["model_white_result"],
        "latest_model_score_pct": float(latest["model_score_pct"]),
        "latest_rolling25_model_score_pct": float(latest["rolling_model_score_pct"]),
    }


def _fmt_pct(value: float | int | None) -> str:
    return "—" if value is None else f"{float(value):.1f}%"


def _fmt_update(value: int | None) -> str:
    return "—" if value is None else str(int(value))


def _xy_points(rows: list[dict[str, Any]], value_key: str, *, width=1180, height=330) -> str:
    if not rows:
        return ""
    left, right, top, bottom = 62, 24, 22, 44
    plot_w = width - left - right
    plot_h = height - top - bottom
    u0 = min(r["update"] for r in rows)
    u1 = max(r["update"] for r in rows)
    span = max(1, u1 - u0)
    points = []
    for row in rows:
        pct = max(0.0, min(100.0, float(row[value_key])))
        x = left + (int(row["update"]) - u0) / span * plot_w
        y = top + (100.0 - pct) / 100.0 * plot_h
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def _svg_chart(rows_by_key: dict[str, list[dict[str, Any]]]) -> str:
    rows = next((r for r in rows_by_key.values() if r), [])
    if not rows:
        return "<p>Brak danych do wykresu.</p>"
    width, height = 1180, 330
    left, right, top, bottom = 62, 24, 22, 44
    plot_h = height - top - bottom
    u0 = min(r["update"] for r in rows)
    u1 = max(r["update"] for r in rows)

    grid = []
    for pct in (0, 25, 50, 75, 100):
        y = top + (100.0 - pct) / 100.0 * plot_h
        grid.append(
            f"<line x1='{left}' y1='{y:.2f}' x2='{width-right}' y2='{y:.2f}' class='grid'/>"
            f"<text x='{left-10}' y='{y+4:.2f}' text-anchor='end'>{pct}%</text>"
        )
    lines = []
    if rows_by_key.get("v1"):
        lines.append(
            f"<polyline points='{_xy_points(rows_by_key['v1'], 'rolling_model_score_pct')}' class='v1-line'/>"
        )
    if rows_by_key.get("v2"):
        lines.append(
            f"<polyline points='{_xy_points(rows_by_key['v2'], 'rolling_model_score_pct')}' class='v2-line'/>"
        )
    return (
        f"<svg viewBox='0 0 {width} {height}'>"
        + "".join(grid)
        + "".join(lines)
        + f"<text x='{left}' y='{height-10}'>update {u0}</text>"
        + f"<text x='{width-right}' y='{height-10}' text-anchor='end'>update {u1}</text></svg>"
    )


def _write_single_report(
    results_path: Path,
    output_dir: Path,
    spec: BotSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = _to_numeric_rows(_read_results(results_path))
    summary = _summary(rows, spec.signature)
    (output_dir / f"summary_{spec.key}.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not rows:
        return rows, summary

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
<meta charset='utf-8'><title>Connect6 {html.escape(spec.label)} Arena</title>
<style>
body{{background:#0f131a;color:#edf2f7;font:14px system-ui;margin:0}}main{{max-width:1280px;margin:auto;padding:28px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:20px 0}}
.card{{background:#171d26;border:1px solid #313b49;border-radius:10px;padding:14px}}.value{{font-size:25px;font-weight:750}}
.muted{{color:#9aa7b5}}table{{width:100%;border-collapse:collapse;background:#171d26}}th,td{{padding:7px 9px;border-bottom:1px solid #313b49;text-align:right}}
th{{position:sticky;top:0;background:#202836}}th:nth-child(2),td:nth-child(2){{text-align:left}}.win{{color:#51c878;font-weight:700}}.loss{{color:#ff6b6b;font-weight:700}}.draw{{color:#e5ba54;font-weight:700}}.wrap{{max-height:760px;overflow:auto}}
</style><main>
<h1>{html.escape(spec.label)} — autosave arena</h1>
<div class='muted'>{len(rows)} modeli • {2*len(rows)} gier • model raz czarny, raz biały</div>
<div class='cards'>
<div class='card'><div>Bot win</div><div class='value'>{summary['bot_win_pct']:.1f}%</div></div>
<div class='card'><div>Bot score</div><div class='value'>{summary['bot_score_pct']:.1f}%</div></div>
<div class='card'><div>Bot czarny win</div><div class='value'>{summary['bot_as_black']['win_pct']:.1f}%</div></div>
<div class='card'><div>Bot biały win</div><div class='value'>{summary['bot_as_white']['win_pct']:.1f}%</div></div>
<div class='card'><div>Pierwsza wygrana modelu</div><div class='value'>{_fmt_update(summary['first_model_win_update'])}</div></div>
<div class='card'><div>Pierwsze 2:0 modelu</div><div class='value'>{_fmt_update(summary['first_model_sweep_update'])}</div></div>
<div class='card'><div>Latest rolling-25 model</div><div class='value'>{_fmt_pct(summary['latest_rolling25_model_score_pct'])}</div></div>
</div>
<div class='wrap'><table><thead><tr><th>update</th><th>model</th><th>model czarny</th><th>model biały</th><th>score modelu</th><th>rolling-25</th><th>wygrane bota</th><th>ruchy B</th><th>ruchy W</th></tr></thead>
<tbody>{''.join(table_rows)}</tbody></table></div></main>"""
    (output_dir / f"bot_arena_{spec.key}.html").write_text(report, encoding="utf-8")
    return rows, summary


def _write_combined_report(
    output_dir: Path,
    data: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]],
) -> None:
    summaries = {key: summary for key, (_, summary) in data.items()}
    rows_by_key = {key: rows for key, (rows, _) in data.items()}
    comparison: dict[str, Any] = {}
    if summaries.get("v1", {}).get("games") and summaries.get("v2", {}).get("games"):
        comparison = {
            "v2_minus_v1_bot_win_pct": summaries["v2"]["bot_win_pct"] - summaries["v1"]["bot_win_pct"],
            "v2_minus_v1_bot_score_pct": summaries["v2"]["bot_score_pct"] - summaries["v1"]["bot_score_pct"],
            "v2_minus_v1_latest_rolling25_bot_score_pct": (
                100.0 - summaries["v2"]["latest_rolling25_model_score_pct"]
            ) - (
                100.0 - summaries["v1"]["latest_rolling25_model_score_pct"]
            ),
        }
    (output_dir / "summary.json").write_text(
        json.dumps({"bots": summaries, "comparison": comparison}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    cards = []
    for spec in BOT_SPECS:
        summary = summaries.get(spec.key, {})
        if not summary.get("games"):
            continue
        cards.extend(
            [
                (f"{spec.label} win", _fmt_pct(summary["bot_win_pct"])),
                (f"{spec.label} score", _fmt_pct(summary["bot_score_pct"])),
                (f"{spec.label} latest rolling-25", _fmt_pct(100.0 - summary["latest_rolling25_model_score_pct"])),
            ]
        )
    if comparison:
        cards.append(("V2 - V1 win", f"{comparison['v2_minus_v1_bot_win_pct']:+.1f} pp"))

    by_update = {}
    for key, rows in rows_by_key.items():
        for row in rows:
            by_update.setdefault(int(row["update"]), {})[key] = row
    table_rows = []
    for update in sorted(by_update, reverse=True):
        pair = by_update[update]
        v1 = pair.get("v1")
        v2 = pair.get("v2")
        table_rows.append(
            "<tr>"
            f"<td>{update}</td>"
            f"<td>{_fmt_pct(v1['model_score_pct'] if v1 else None)}</td>"
            f"<td>{_fmt_pct(v1['rolling_model_score_pct'] if v1 else None)}</td>"
            f"<td>{_fmt_pct(v2['model_score_pct'] if v2 else None)}</td>"
            f"<td>{_fmt_pct(v2['rolling_model_score_pct'] if v2 else None)}</td>"
            "</tr>"
        )

    card_html = "".join(
        f"<div class='card'><div class='muted'>{html.escape(label)}</div><div class='value'>{html.escape(value)}</div></div>"
        for label, value in cards
    )
    report = f"""<!doctype html>
<meta charset='utf-8'><title>Connect6 Bot Arena V1 vs V2</title>
<style>
:root{{--bg:#0f131a;--panel:#171d26;--panel2:#202836;--text:#edf2f7;--muted:#9aa7b5;--border:#313b49;--v1:#8795aa;--v2:#6ea1ff}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px system-ui}}main{{max-width:1280px;margin:auto;padding:28px}}
.muted{{color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:20px 0}}
.card,.chart{{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:14px}}.value{{font-size:25px;font-weight:750;margin-top:3px}}
.chart{{margin:14px 0 20px;overflow-x:auto}}svg{{width:100%;min-width:760px}}svg text{{fill:var(--muted);font-size:12px}}.grid{{stroke:#34404f}}.v1-line{{fill:none;stroke:var(--v1);stroke-width:3}}.v2-line{{fill:none;stroke:var(--v2);stroke-width:3}}
.legend{{display:flex;gap:20px;margin-bottom:8px}}table{{width:100%;border-collapse:collapse;background:var(--panel)}}th,td{{padding:7px 9px;border-bottom:1px solid var(--border);text-align:right}}th{{position:sticky;top:0;background:var(--panel2)}}.wrap{{max-height:720px;overflow:auto}}
</style><main>
<h1>Connect6 — Bot Arena: V1 vs V2</h1>
<div class='muted'>Każdy checkpoint gra po 2 gry z każdym botem: model raz czarny i raz biały. Łącznie 4 gry/model, jeśli oba zestawy są kompletne.</div>
<div class='cards'>{card_html}</div>
<div class='chart'><div class='legend'><span>V1 rolling-25 model score</span><span>V2 rolling-25 model score</span></div>{_svg_chart(rows_by_key)}</div>
<div class='wrap'><table><thead><tr><th>update</th><th>V1 model score</th><th>V1 rolling</th><th>V2 model score</th><th>V2 rolling</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
</main>"""
    (output_dir / "bot_arena.html").write_text(report, encoding="utf-8")


def _print_summary(spec: BotSpec, summary: dict[str, Any]) -> None:
    if not summary.get("models"):
        print(f"{spec.label}: brak wyników.")
        return
    print("\n" + "=" * 78)
    print(f"{spec.label} — PODSUMOWANIE")
    print("=" * 78)
    print(
        f"Modele: {summary['models']:,} | gry: {summary['games']:,} | "
        f"bot W/D/L: {summary['bot_wins']}/{summary['draws']}/{summary['model_wins']} | "
        f"win={summary['bot_win_pct']:.2f}% | score={summary['bot_score_pct']:.2f}%"
    )
    print(
        f"Bot czarny win={summary['bot_as_black']['win_pct']:.2f}% | "
        f"bot biały win={summary['bot_as_white']['win_pct']:.2f}% | "
        f"latest rolling-25 bot score={100.0-summary['latest_rolling25_model_score_pct']:.2f}%"
    )


def _migrate_legacy_v1(output_dir: Path) -> None:
    legacy_path = output_dir / "bot_matches.csv"
    v1_path = output_dir / "bot_matches_v1.csv"
    if legacy_path.exists() and not v1_path.exists():
        shutil.copy2(legacy_path, v1_path)
        print(f"[MIGRATE] Zachowano stare wyniki V1: {legacy_path.name} -> {v1_path.name}")


def _run_one_bot(
    spec: BotSpec,
    refs,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    sync_interval: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results_path = output_dir / f"bot_matches_{spec.key}.csv"
    state_path = output_dir / f"state_{spec.key}.json"
    _validate_state(state_path, checkpoint_dir=checkpoint_dir, bot_signature=spec.signature)

    completed = _completed_names(results_path)
    pending = [ref for ref in refs if ref.name not in completed]
    print("\n" + "=" * 78)
    print(f"{spec.label.upper()} — ALL AUTOSAVES")
    print("=" * 78)
    print(
        f"Checkpointy: {len(refs):,} | gotowe: {len(completed):,} | "
        f"pozostało: {len(pending):,} | 2 gry/model"
    )

    if not pending:
        data = _write_single_report(results_path, output_dir, spec)
        _print_summary(spec, data[1])
        return data

    bot = spec.cls(device)
    seed_board = torch.zeros((1, 19, 19), dtype=torch.int8, device=device)
    seed_player = torch.ones(1, dtype=torch.int8, device=device)
    seed_left = torch.ones(1, dtype=torch.int8, device=device)
    bot.actions(seed_board, seed_player, seed_left)
    torch.cuda.synchronize(device)

    family_cfg: dict[str, Any] | None = None
    total_started = time.perf_counter()
    completed_this_run = 0

    for start in range(0, len(pending), batch_size):
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
                f"OOM przy batchu {len(chunk_refs)} modeli. Zmniejsz bot_arena.models_per_batch."
            ) from exc

        warm_env = VectorConnect6(len(chunk_refs), 19, 6, device=device, debug_checks=False)
        warm_x = warm_env.network_input()
        ensemble.forward_grouped(warm_x.unsqueeze(1)).squeeze(1)
        del warm_env, warm_x
        torch.cuda.synchronize(device)

        black_winners, black_moves, black_elapsed = _play_colour_game(
            ensemble, bot, model_is_black=True, sync_interval_moves=sync_interval
        )
        white_winners, white_moves, white_elapsed = _play_colour_game(
            ensemble, bot, model_is_black=False, sync_interval_moves=sync_interval
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

        games = 2 * len(chunk_refs)
        gps = games / max(game_elapsed, 1e-9)
        progress = 100.0 * completed_this_run / len(pending)
        print(
            f"[{progress:6.2f}%] {completed_this_run:,}/{len(pending):,} modeli | "
            f"{gps:,.1f} g/s | bot/model/draw="
            f"{sum(int(r['bot_wins']) for r in rows)}/"
            f"{sum(int(r['model_wins']) for r in rows)}/"
            f"{sum(int(r['draws']) for r in rows)} | "
            f"load={load_elapsed:.2f}s | play={game_elapsed:.2f}s"
        )

        del ensemble, checkpoints, rows
        gc.collect()
        torch.cuda.empty_cache()
        _write_single_report(results_path, output_dir, spec)

    data = _write_single_report(results_path, output_dir, spec)
    _print_summary(spec, data[1])
    print(f"[DONE] {spec.label}: {completed_this_run:,} nowych modeli | wall={time.perf_counter()-total_started:.2f}s")
    return data


def run(
    config_path: str | Path,
    *,
    reset: bool = False,
    models_per_batch_override: int | None = None,
) -> None:
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

    if reset:
        for name in (
            "bot_matches.csv", "state.json", "summary.json", "bot_arena.html",
            "bot_matches_v1.csv", "bot_matches_v2.csv",
            "state_v1.json", "state_v2.json",
            "summary_v1.json", "summary_v2.json",
            "bot_arena_v1.html", "bot_arena_v2.html",
        ):
            (output_dir / name).unlink(missing_ok=True)
    else:
        _migrate_legacy_v1(output_dir)

    refs = legacy.discover_checkpoints(checkpoint_dir)
    if not refs:
        raise RuntimeError(f"Nie znaleziono model_update_*.pt w {checkpoint_dir}")

    requested = [str(v).lower() for v in arena.get("bots", ["v1", "v2"])]
    specs = [spec for spec in BOT_SPECS if spec.key in requested]
    unknown = sorted(set(requested) - {spec.key for spec in BOT_SPECS})
    if unknown:
        raise ValueError(f"Nieznane boty w bot_arena.bots: {unknown}; dostępne: v1, v2")
    if not specs:
        raise ValueError("bot_arena.bots nie może być puste")

    dtype = _dtype_from_name(arena.get("inference_dtype", "bfloat16"))
    batch_size = max(1, int(models_per_batch_override or arena.get("models_per_batch", 256)))
    sync_interval = max(1, int(arena.get("sync_interval_moves", 16)))

    print("\n" + "=" * 78)
    print("CONNECT6 BOT ARENA — V1 + V2")
    print("=" * 78)
    print(f"Checkpointy: {len(refs):,} | boty: {', '.join(spec.label for spec in specs)}")
    print(f"Każdy bot: 2 gry/model (model raz czarny, raz biały) | batch={batch_size}")
    print(f"GPU: {torch.cuda.get_device_name(device)} | CNN dtype: {dtype}")

    data: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for spec in specs:
        data[spec.key] = _run_one_bot(
            spec,
            refs,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            sync_interval=sync_interval,
        )
        _write_combined_report(output_dir, data)

    for spec in BOT_SPECS:
        if spec.key not in data:
            path = output_dir / f"bot_matches_{spec.key}.csv"
            if path.exists():
                data[spec.key] = _write_single_report(path, output_dir, spec)
    _write_combined_report(output_dir, data)

    print("\n[DONE] Bot arena")
    for spec in BOT_SPECS:
        path = output_dir / f"bot_matches_{spec.key}.csv"
        if path.exists():
            print(f"CSV {spec.key.upper()}: {path}")
    print(f"HTML comparison: {output_dir / 'bot_arena.html'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6: każdy autosave CNN gra po 2 gry z GPU Tactical Bot V1 i V2"
    )
    parser.add_argument("--config", default="configs/bot_arena.yaml")
    parser.add_argument("--reset", action="store_true", help="Usuń poprzednie wyniki i policz od zera")
    parser.add_argument("--models-per-batch", type=int, default=None)
    args = parser.parse_args()
    run(args.config, reset=args.reset, models_per_batch_override=args.models_per_batch)


if __name__ == "__main__":
    main()
