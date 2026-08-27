from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from itertools import combinations
from pathlib import Path
from typing import Any

import torch

from connect6.bots.gpu_bot import (
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV3,
    GPUTacticalBotV4B8x2,
    GPUTacticalBotV5B8x4,
)
from connect6.championship import bot_arena as model_arena
from connect6.championship import championship as legacy
from connect6.championship.championship_cnn import _black_to_move, _masked_step
from connect6.engine.vector_env import VectorConnect6
from connect6.evaluation import cloudict_arena as cloudict


BOARD_SIZE = 19
BOT_SPECS = (
    model_arena.BotSpec("v1", "GPU Tactical Bot V1", "gpu_tactical_bot_heuristic_v1", GPUTacticalBot),
    model_arena.BotSpec("v2", "GPU Tactical Bot V2", "gpu_tactical_bot_heuristic_v2", GPUTacticalBotV2),
    model_arena.BotSpec("v3", "GPU Tactical Bot V3 D2[8]", "gpu_tactical_bot_v3_d2_b8", GPUTacticalBotV3),
    model_arena.BotSpec("v4", "GPU Tactical Bot V4 D3[8,2]", "gpu_tactical_bot_v4_d3_b8x2", GPUTacticalBotV4B8x2),
    model_arena.BotSpec("v5", "GPU Tactical Bot V5 D3[8,4]", "gpu_tactical_bot_v5_d3_b8x4", GPUTacticalBotV5B8x4),
)
BOT_BY_KEY = {spec.key: spec for spec in BOT_SPECS}
PAIR_FIELDS = [
    "bot_a", "bot_b", "games", "a_wins", "draws", "b_wins",
    "a_score_pct", "b_score_pct", "opening_size", "elapsed_seconds",
]
CLOUDICT_FIELDS = [
    "bot", "depth", "opening_action", "opening_coord", "bot_color",
    "winner", "result", "stones_played", "model_decisions",
    "cloudict_decisions", "model_seconds", "cloudict_seconds", "elapsed_seconds",
]


def _project_root(config_path: Path) -> Path:
    return config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _reset_bot(bot) -> None:
    fn = getattr(bot, "reset", None)
    if callable(fn):
        fn()


def _warm_bot(bot, device: torch.device) -> None:
    board = torch.zeros((1, 19, 19), dtype=torch.int8, device=device)
    player = torch.ones(1, dtype=torch.int8, device=device)
    left = torch.ones(1, dtype=torch.int8, device=device)
    _reset_bot(bot)
    bot.actions(board, player, left)
    torch.cuda.synchronize(device)


def _append(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _center_actions(size: int) -> tuple[int, ...]:
    if size == BOARD_SIZE:
        return tuple(range(BOARD_SIZE * BOARD_SIZE))
    if size <= 0 or size > BOARD_SIZE or size % 2 == 0:
        raise ValueError(f"Nieprawidłowy opening_size={size}")
    radius = size // 2
    center = BOARD_SIZE // 2
    return tuple(
        r * BOARD_SIZE + c
        for r in range(center - radius, center + radius + 1)
        for c in range(center - radius, center + radius + 1)
    )


# Existing model arena is already the desired all-checkpoints gauntlet. Search
# bots only need their pending second stone cleared before every fresh game.
_ORIGINAL_MODEL_GAME = model_arena._play_colour_game


def _model_game_with_reset(ensemble, bot, **kwargs):
    _reset_bot(bot)
    return _ORIGINAL_MODEL_GAME(ensemble, bot, **kwargs)


def _run_models_vs_bots(
    specs,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    models_per_batch: int,
    sync_interval: int,
) -> dict[str, Any]:
    refs = legacy.discover_checkpoints(checkpoint_dir)
    if not refs:
        raise RuntimeError(f"Nie znaleziono model_update_*.pt w {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    old = model_arena._play_colour_game
    model_arena._play_colour_game = _model_game_with_reset
    summaries: dict[str, Any] = {}
    try:
        print("\n# ETAP 1: wszystkie checkpointy vs każdy bot")
        print(f"Modele={len(refs):,} | boty={','.join(s.key for s in specs)} | 2 gry/model/bot")
        for spec in specs:
            _, summary = model_arena._run_one_bot(
                spec,
                refs,
                checkpoint_dir=checkpoint_dir,
                output_dir=output_dir,
                device=device,
                dtype=dtype,
                batch_size=models_per_batch,
                sync_interval=sync_interval,
            )
            summaries[spec.key] = summary
    finally:
        model_arena._play_colour_game = old
    return summaries


@torch.inference_mode()
def _play_pair_colour(
    bot_a,
    bot_b,
    *,
    a_is_black: bool,
    openings: tuple[int, ...],
    device: torch.device,
    sync_interval: int,
) -> list[int]:
    env = VectorConnect6(len(openings), 19, 6, device=device, debug_checks=False)
    env.reset()
    _reset_bot(bot_a)
    _reset_bot(bot_b)
    opening_tensor = torch.tensor(openings, dtype=torch.long, device=device)
    env.step(opening_tensor)
    active = torch.ones(len(openings), dtype=torch.bool, device=device)
    winners = torch.zeros(len(openings), dtype=torch.int8, device=device)

    for move_index in range(1, env.action_size):
        black_to_move = _black_to_move(move_index)
        actor = bot_a if black_to_move == a_is_black else bot_b
        actions = actor.actions(env.boards, env.current_player, env.stones_left)
        done, winner = _masked_step(env, actions, active)
        newly_done = active & done
        winners = torch.where(newly_done, winner, winners)
        active &= ~done
        if (move_index + 1) % sync_interval == 0 and not bool(active.any().item()):
            break

    if bool(active.any().item()):
        raise RuntimeError("Nie wszystkie partie bot-vs-bot zakończyły się")
    return [int(v) for v in winners.cpu().tolist()]


def _count_pair(winners: list[int], *, a_is_black: bool) -> tuple[int, int, int]:
    a_wins = draws = b_wins = 0
    for winner in winners:
        if winner == 0:
            draws += 1
        elif (winner == 1 and a_is_black) or (winner == -1 and not a_is_black):
            a_wins += 1
        else:
            b_wins += 1
    return a_wins, draws, b_wins


def _run_bot_round_robin(
    specs,
    *,
    output_dir: Path,
    device: torch.device,
    opening_size: int,
    sync_interval: int,
) -> list[dict[str, str]]:
    path = output_dir / "bot_vs_bot.csv"
    done = {
        tuple(sorted((row["bot_a"], row["bot_b"])))
        for row in _rows(path)
        if row.get("bot_a") and row.get("bot_b")
    }
    openings = _center_actions(opening_size)
    print("\n# ETAP 2: każdy bot z każdym")
    print(f"Otwarcia={len(openings)} x 2 kolory = {2*len(openings)} gier/parę")

    for a, b in combinations(specs, 2):
        if tuple(sorted((a.key, b.key))) in done:
            print(f"[SKIP] {a.key.upper()} vs {b.key.upper()}")
            continue
        bot_a, bot_b = a.cls(device), b.cls(device)
        _warm_bot(bot_a, device)
        _warm_bot(bot_b, device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        black = _play_pair_colour(
            bot_a, bot_b, a_is_black=True, openings=openings,
            device=device, sync_interval=sync_interval,
        )
        white = _play_pair_colour(
            bot_a, bot_b, a_is_black=False, openings=openings,
            device=device, sync_interval=sync_interval,
        )
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        aw1, d1, bw1 = _count_pair(black, a_is_black=True)
        aw2, d2, bw2 = _count_pair(white, a_is_black=False)
        aw, dr, bw = aw1 + aw2, d1 + d2, bw1 + bw2
        games = 2 * len(openings)
        a_score = 100.0 * (aw + 0.5 * dr) / games
        b_score = 100.0 - a_score
        _append(path, PAIR_FIELDS, {
            "bot_a": a.key, "bot_b": b.key, "games": games,
            "a_wins": aw, "draws": dr, "b_wins": bw,
            "a_score_pct": f"{a_score:.3f}", "b_score_pct": f"{b_score:.3f}",
            "opening_size": opening_size, "elapsed_seconds": f"{elapsed:.6f}",
        })
        print(
            f"{a.key.upper()} vs {b.key.upper()}: {aw} W / {dr} D / {bw} L | "
            f"score={a_score:.2f}% | {games/max(elapsed,1e-9):,.1f} g/s"
        )
    return _rows(path)


class BotAgent:
    def __init__(self, spec, device: torch.device) -> None:
        self.device = device
        self.bot = spec.cls(device)
        _warm_bot(self.bot, device)

    def reset(self) -> None:
        _reset_bot(self.bot)

    @torch.inference_mode()
    def action(self, game) -> tuple[int, float]:
        board = torch.from_numpy(game.board).unsqueeze(0).to(self.device, dtype=torch.int8)
        player = torch.tensor([game.current_player], dtype=torch.int8, device=self.device)
        left = torch.tensor([game.stones_left_in_turn], dtype=torch.int8, device=self.device)
        started = time.perf_counter()
        action = int(self.bot.actions(board, player, left).item())
        torch.cuda.synchronize(self.device)
        return action, time.perf_counter() - started


def _cloudict_done(path: Path) -> set[tuple[str, int, int, str]]:
    result = set()
    for row in _rows(path):
        try:
            result.add((row["bot"], int(row["depth"]), int(row["opening_action"]), row["bot_color"]))
        except (KeyError, ValueError):
            pass
    return result


def _cloudict_stats(rows: list[dict[str, str]], bot: str, depth: int) -> dict[str, Any]:
    selected = [r for r in rows if r.get("bot") == bot and int(r.get("depth", -1)) == depth]
    wins = sum(r.get("result") == "WIN" for r in selected)
    draws = sum(r.get("result") == "DRAW" for r in selected)
    losses = sum(r.get("result") == "LOSS" for r in selected)
    games = len(selected)
    return {
        "games": games, "wins": wins, "draws": draws, "losses": losses,
        "score_pct": 100.0 * (wins + 0.5 * draws) / games if games else 0.0,
    }


def _recoverable_cloudict(exc: BaseException) -> bool:
    text = str(exc)
    return "Cloudict zakonczyl sie" in text or "Przekroczono limit czasu" in text


def _run_cloudict(
    specs,
    *,
    output_dir: Path,
    device: torch.device,
    executable: Path,
    depths: list[int],
    opening_sizes: dict[int, int],
    vcf: bool,
    timeout: float,
    max_restarts: int,
) -> list[dict[str, str]]:
    path = output_dir / "bot_vs_cloudict_games.csv"
    completed = _cloudict_done(path)
    print("\n# ETAP 3: każdy bot vs Cloudict D2/D3/D4")

    for spec in specs:
        agent = BotAgent(spec, device)
        for depth in depths:
            openings = _center_actions(opening_sizes[depth])
            schedule = [(op, black) for op in openings for black in (True, False)]
            remaining = [
                item for item in schedule
                if (spec.key, depth, item[0], "BLACK" if item[1] else "WHITE") not in completed
            ]
            print(
                f"{spec.key.upper()} vs D{depth}: {len(schedule)} gier | "
                f"pozostało={len(remaining)}"
            )
            if not remaining:
                continue

            engine = None
            try:
                for index, (opening, bot_is_black) in enumerate(remaining, 1):
                    retry = 0
                    while True:
                        if engine is None:
                            engine = cloudict.CloudictEngine(
                                executable, depth=depth, vcf=vcf, timeout_seconds=timeout
                            )
                        agent.reset()
                        try:
                            row = cloudict.play_one_game(
                                agent,
                                engine,
                                opening_action=opening,
                                model_is_black=bot_is_black,
                            )
                            break
                        except (RuntimeError, TimeoutError) as exc:
                            limited = max_restarts > 0
                            if not _recoverable_cloudict(exc) or (limited and retry >= max_restarts):
                                raise
                            retry += 1
                            engine.close()
                            engine = None
                            limit_text = str(max_restarts) if limited else "∞"
                            print(f"[CLOUDICT] restart {retry}/{limit_text}: {exc}")
                            time.sleep(0.25)

                    stored = {
                        "bot": spec.key,
                        "depth": depth,
                        "opening_action": row["opening_action"],
                        "opening_coord": row["opening_coord"],
                        "bot_color": row["model_color"],
                        "winner": row["winner"],
                        "result": row["result"],
                        "stones_played": row["stones_played"],
                        "model_decisions": row["model_decisions"],
                        "cloudict_decisions": row["cloudict_decisions"],
                        "model_seconds": row["model_seconds"],
                        "cloudict_seconds": row["cloudict_seconds"],
                        "elapsed_seconds": row["elapsed_seconds"],
                    }
                    _append(path, CLOUDICT_FIELDS, stored)
                    completed.add((spec.key, depth, opening, stored["bot_color"]))
                    if index % 10 == 0 or index == len(remaining):
                        s = _cloudict_stats(_rows(path), spec.key, depth)
                        print(
                            f"  [{index:3d}/{len(remaining)}] W/D/L="
                            f"{s['wins']}/{s['draws']}/{s['losses']} | score={s['score_pct']:.2f}%"
                        )
            finally:
                if engine is not None:
                    engine.close()
    return _rows(path)


def _league_ranking(rows: list[dict[str, str]], specs) -> list[dict[str, Any]]:
    total = {s.key: {"games": 0, "wins": 0, "draws": 0, "losses": 0} for s in specs}
    for row in rows:
        a, b = row["bot_a"], row["bot_b"]
        if a not in total or b not in total:
            continue
        games, aw, dr, bw = int(row["games"]), int(row["a_wins"]), int(row["draws"]), int(row["b_wins"])
        total[a]["games"] += games; total[a]["wins"] += aw; total[a]["draws"] += dr; total[a]["losses"] += bw
        total[b]["games"] += games; total[b]["wins"] += bw; total[b]["draws"] += dr; total[b]["losses"] += aw
    ranking = []
    for spec in specs:
        t = total[spec.key]
        score = t["wins"] + 0.5 * t["draws"]
        ranking.append({"bot": spec.key, **t, "score_pct": 100.0 * score / t["games"] if t["games"] else 0.0})
    ranking.sort(key=lambda r: (-r["score_pct"], r["bot"]))
    return ranking


def _write_summary(
    output_dir: Path,
    specs,
    model_summaries: dict[str, Any],
    pair_rows: list[dict[str, str]],
    cloud_rows: list[dict[str, str]],
    depths: list[int],
) -> None:
    cloud_summary = {
        s.key: {str(d): _cloudict_stats(cloud_rows, s.key, d) for d in depths}
        for s in specs
    }
    summary = {
        "models_vs_bots": model_summaries,
        "bot_round_robin": {"ranking": _league_ranking(pair_rows, specs), "pairs": pair_rows},
        "cloudict": cloud_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        fields = ["stage", "bot", "opponent", "depth", "games", "wins", "draws", "losses", "score_pct"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for spec in specs:
            s = model_summaries.get(spec.key)
            if s and s.get("games"):
                writer.writerow({
                    "stage": "models", "bot": spec.key, "opponent": "all_checkpoints", "depth": "",
                    "games": s["games"], "wins": s["bot_wins"], "draws": s["draws"],
                    "losses": s["model_wins"], "score_pct": s["bot_score_pct"],
                })
        for row in pair_rows:
            writer.writerow({
                "stage": "bot_vs_bot", "bot": row["bot_a"], "opponent": row["bot_b"], "depth": "",
                "games": row["games"], "wins": row["a_wins"], "draws": row["draws"],
                "losses": row["b_wins"], "score_pct": row["a_score_pct"],
            })
        for spec in specs:
            for depth in depths:
                s = cloud_summary[spec.key][str(depth)]
                if s["games"]:
                    writer.writerow({
                        "stage": "cloudict", "bot": spec.key, "opponent": "cloudict", "depth": depth,
                        "games": s["games"], "wins": s["wins"], "draws": s["draws"],
                        "losses": s["losses"], "score_pct": f"{s['score_pct']:.3f}",
                    })

    print("\n=== WYNIK KOŃCOWY ===")
    for i, row in enumerate(summary["bot_round_robin"]["ranking"], 1):
        print(f"{i}. {row['bot'].upper()} score={row['score_pct']:.2f}% W/D/L={row['wins']}/{row['draws']}/{row['losses']}")
    for spec in specs:
        vals = [f"D{d}={cloud_summary[spec.key][str(d)]['score_pct']:.1f}%" for d in depths if cloud_summary[spec.key][str(d)]["games"]]
        if vals:
            print(f"{spec.key.upper()} Cloudict: " + " | ".join(vals))
    print(f"JSON: {output_dir / 'summary.json'}")
    print(f"CSV : {output_dir / 'summary.csv'}")


def run(
    config_path: str | Path,
    *,
    reset: bool = False,
    skip_models: bool = False,
    skip_bot_league: bool = False,
    skip_cloudict: bool = False,
) -> None:
    config_path = Path(config_path).resolve()
    cfg = legacy._read_yaml(config_path)
    arena = cfg.get("bot_strength_arena", cfg)
    root = _project_root(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Ten test wymaga CUDA")
    device = torch.device(arena.get("device", "cuda"))

    requested = [str(x).lower() for x in arena.get("bots", ["v1", "v2", "v3", "v4", "v5"])]
    unknown = sorted(set(requested) - set(BOT_BY_KEY))
    if unknown:
        raise ValueError(f"Nieznane boty: {unknown}")
    specs = [BOT_BY_KEY[key] for key in requested]

    checkpoint_dir = _resolve(root, arena["checkpoint_dir"])
    output_dir = _resolve(root, arena["output_dir"])
    if reset and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtype_name = str(arena.get("inference_dtype", "bfloat16")).lower()
    dtype = torch.float16 if dtype_name in {"fp16", "float16", "half"} else torch.bfloat16
    models_per_batch = max(1, int(arena.get("models_per_batch", 256)))
    sync_interval = max(1, int(arena.get("sync_interval_moves", 16)))

    pair_cfg = arena.get("bot_vs_bot", {})
    opening_size = int(pair_cfg.get("opening_size", 19))

    ccfg = arena.get("cloudict", {})
    executable = _resolve(root, ccfg.get("executable", str(cloudict.DEFAULT_CLOUDICT_EXE)))
    depths = [int(x) for x in ccfg.get("depths", [2, 3, 4])]
    raw_sizes = ccfg.get("opening_sizes", {})
    defaults = {2: 19, 3: 9, 4: 5}
    opening_sizes = {
        d: int(raw_sizes.get(d, raw_sizes.get(str(d), defaults.get(d, 3)))) for d in depths
    }
    vcf = bool(ccfg.get("vcf", False))
    timeout = float(ccfg.get("timeout_seconds", 60.0))
    # max_restarts <= 0 means unlimited recoverable Cloudict restarts.
    max_restarts = int(ccfg.get("max_restarts", 0))

    print("=" * 78)
    print("CONNECT6 BOT STRENGTH ARENA")
    print("=" * 78)
    print(f"Boty: {', '.join(s.key.upper() for s in specs)}")
    print(f"Wyniki: {output_dir}")

    model_dir = output_dir / "models_vs_bots"
    if skip_models:
        model_summaries = {
            s.key: json.loads((model_dir / f"summary_{s.key}.json").read_text(encoding="utf-8"))
            for s in specs if (model_dir / f"summary_{s.key}.json").exists()
        }
    else:
        model_summaries = _run_models_vs_bots(
            specs, checkpoint_dir=checkpoint_dir, output_dir=model_dir,
            device=device, dtype=dtype, models_per_batch=models_per_batch,
            sync_interval=sync_interval,
        )

    pair_rows = _rows(output_dir / "bot_vs_bot.csv") if skip_bot_league else _run_bot_round_robin(
        specs, output_dir=output_dir, device=device,
        opening_size=opening_size, sync_interval=sync_interval,
    )
    cloud_rows = _rows(output_dir / "bot_vs_cloudict_games.csv") if skip_cloudict else _run_cloudict(
        specs, output_dir=output_dir, device=device, executable=executable,
        depths=depths, opening_sizes=opening_sizes, vcf=vcf,
        timeout=timeout, max_restarts=max_restarts,
    )
    _write_summary(output_dir, specs, model_summaries, pair_rows, cloud_rows, depths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="All checkpoints vs bots + bot round-robin + bots vs Cloudict D2/D3/D4"
    )
    parser.add_argument("--config", default="configs/bot_strength_arena.yaml")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-bot-league", action="store_true")
    parser.add_argument("--skip-cloudict", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        reset=args.reset,
        skip_models=args.skip_models,
        skip_bot_league=args.skip_bot_league,
        skip_cloudict=args.skip_cloudict,
    )


if __name__ == "__main__":
    main()
