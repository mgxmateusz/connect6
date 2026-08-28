from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect6.bots.gpu_bot import (
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV2Pro,
    GPUTacticalBotV3Pro,
    GPUTacticalBotV4Pro,
    GPUTacticalBotPairFirst,
    GPUTacticalBotPairFirst32,
    GPUTacticalBotHybrid,
    GPUTacticalBotHybrid32,
    GPUTacticalBotLiveRoad,
    GPUTacticalBotFullPair,
)
from connect6.championship.championship import discover_checkpoints
from connect6.championship.championship_cnn import _black_to_move, _masked_step
from connect6.engine.vector_env import VectorConnect6
from connect6.evaluation.bot_strength_arena_fast import NativePolicyPool


BOARD = 19
CENTER_ROW = BOARD // 2
CENTER_COL = BOARD // 2
CENTER = CENTER_ROW * BOARD + CENTER_COL

BOT_SPECS = (
    ("v1", "GPU Tactical Bot V1", GPUTacticalBot),
    ("v2", "GPU Tactical Bot V2", GPUTacticalBotV2),
    ("v2pro", "GPU Tactical Bot V2 Pro LatentFork", GPUTacticalBotV2Pro),
    ("v3pro", "GPU Tactical Bot V3 Pro Top16 Pair-State", GPUTacticalBotV3Pro),
    ("v4pro", "GPU Tactical Bot V4 Pro Top12 ReplyPair6", GPUTacticalBotV4Pro),
    ("pair", "GPU Tactical Bot PairFirst AllPairs P128", GPUTacticalBotPairFirst),
    ("pair32", "GPU Tactical Bot PairFirst AllPairs P32", GPUTacticalBotPairFirst32),
    ("hybrid", "GPU Tactical Bot Hybrid LiveRoad Pair128", GPUTacticalBotHybrid),
    ("hybrid32", "GPU Tactical Bot Hybrid LiveRoad Pair32", GPUTacticalBotHybrid32),
    ("live", "GPU Tactical Bot LiveRoad Brute Force", GPUTacticalBotLiveRoad),
    ("full", "GPU Tactical Bot Full Pair Brute Force", GPUTacticalBotFullPair),
)


def _reset_bot(bot) -> None:
    reset = getattr(bot, "reset", None)
    if callable(reset):
        reset()


def _winner_name(winner: int, *, cnn_is_black: bool, bot_key: str) -> tuple[str, str]:
    if winner == 0:
        return "draw", "draw"
    winner_color = "black" if winner == 1 else "white"
    cnn_won = (winner == 1 and cnn_is_black) or (winner == -1 and not cnn_is_black)
    return ("cnn" if cnn_won else bot_key), winner_color


def _move_record(
    *,
    ply: int,
    action: int,
    color: str,
    source: str,
    stones_left_before: int,
) -> dict:
    row, col = divmod(int(action), BOARD)
    return {
        "ply": int(ply),
        "action": int(action),
        "row": int(row),
        "col": int(col),
        "color": color,
        "source": source,
        "stones_left_before": int(stones_left_before),
    }


@torch.inference_mode()
def _play_one(
    pool: NativePolicyPool,
    bot,
    *,
    bot_key: str,
    bot_label: str,
    cnn_is_black: bool,
) -> dict:
    device = pool.device
    env = VectorConnect6(1, BOARD, 6, device=device, debug_checks=False)
    env.reset()
    _reset_bot(bot)

    active = torch.ones(1, dtype=torch.bool, device=device)
    winner_value = 0
    moves: list[dict] = []

    opening = torch.tensor([CENTER], dtype=torch.long, device=device)
    opening_left = int(env.stones_left[0].item())
    moves.append(
        _move_record(
            ply=0,
            action=CENTER,
            color="black",
            source="forced_center",
            stones_left_before=opening_left,
        )
    )
    done, winner = _masked_step(env, opening, active)
    if bool(done[0].item()):
        winner_value = int(winner[0].item())
        active[0] = False

    model_ids = torch.zeros(1, dtype=torch.int32, device=device)

    for move_index in range(1, BOARD * BOARD):
        if not bool(active[0].item()):
            break

        black_to_move = _black_to_move(move_index)
        color = "black" if black_to_move else "white"
        cnn_to_move = bool(black_to_move) == bool(cnn_is_black)
        source = "cnn" if cnn_to_move else bot_key
        stones_left_before = int(env.stones_left[0].item())

        if cnn_to_move:
            actions = pool.actions(
                env.boards, env.current_player, env.stones_left, model_ids
            )
        else:
            actions = bot.actions(env.boards, env.current_player, env.stones_left)

        action = int(actions[0].item())
        if action < 0 or action >= BOARD * BOARD:
            raise RuntimeError(
                f"{bot_key}: nielegalny action={action} na ply={move_index}, source={source}"
            )
        row, col = divmod(action, BOARD)
        if int(env.boards[0, row, col].item()) != 0:
            raise RuntimeError(
                f"{bot_key}: zajęte pole action={action} ({row},{col}) na ply={move_index}, source={source}"
            )

        moves.append(
            _move_record(
                ply=move_index,
                action=action,
                color=color,
                source=source,
                stones_left_before=stones_left_before,
            )
        )
        done, winner = _masked_step(env, actions.to(dtype=torch.long), active)
        if bool(done[0].item()):
            winner_value = int(winner[0].item())
            active[0] = False
            break

    if bool(active[0].item()):
        raise RuntimeError(f"{bot_key}: partia nie zakończyła się po maksymalnej liczbie pól")

    winner_name, winner_color = _winner_name(
        winner_value, cnn_is_black=cnn_is_black, bot_key=bot_key
    )
    return {
        "bot": bot_key,
        "bot_label": bot_label,
        "cnn_color": "black" if cnn_is_black else "white",
        "bot_color": "white" if cnn_is_black else "black",
        "forced_center": {"action": CENTER, "row": CENTER_ROW, "col": CENTER_COL},
        "winner": winner_name,
        "winner_color": winner_color,
        "move_count": len(moves),
        "actions": [move["action"] for move in moves],
        "moves": moves,
    }


def _write_output(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Play the newest CNN checkpoint against every active tactical bot: "
            "two center-opening games per bot and save every move."
        )
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path,
        default=Path("runs/connect6_cnn_05/checkpoints"),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("runs/connect6_cnn_05/latest_cnn_vs_all_bots_traces.json"),
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Ten trace runner wymaga CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("Trace runner używa native SM120 i wymaga RTX 50")

    checkpoint_dir = args.checkpoint_dir.resolve()
    output = args.output.resolve()
    refs = discover_checkpoints(checkpoint_dir)
    if not refs:
        raise RuntimeError(f"Brak model_update_*.pt w {checkpoint_dir}")
    latest = refs[-1]

    print("=" * 78)
    print("LATEST CNN VS ALL ACTIVE TACTICAL BOTS — CENTER TRACE")
    print("=" * 78)
    print(f"CNN: {latest.name} (update={latest.update})")
    print(f"Boty: {len(BOT_SPECS)}")
    print(f"Gry: {len(BOT_SPECS) * 2} = dokładnie 2 na bota")
    print(f"Opening: action={CENTER}, row={CENTER_ROW}, col={CENTER_COL}")
    print(f"Output: {output}")

    pool = NativePolicyPool([latest], device, load_chunk=1)

    payload = {
        "format_version": 1,
        "cnn_model": latest.name,
        "cnn_update": int(latest.update),
        "board_size": BOARD,
        "win_length": 6,
        "forced_center_action": CENTER,
        "forced_center_row": CENTER_ROW,
        "forced_center_col": CENTER_COL,
        "games_per_bot": 2,
        "bots": [key for key, _, _ in BOT_SPECS],
        "games": [],
    }
    _write_output(output, payload)

    for bot_index, (bot_key, bot_label, bot_cls) in enumerate(BOT_SPECS, start=1):
        bot = bot_cls(device)
        print(f"\n[{bot_index}/{len(BOT_SPECS)}] {bot_key.upper()} — {bot_label}")
        for cnn_is_black in (True, False):
            game = _play_one(
                pool, bot, bot_key=bot_key, bot_label=bot_label,
                cnn_is_black=cnn_is_black,
            )
            payload["games"].append(game)
            _write_output(output, payload)
            print(
                f"  CNN {game['cnn_color']:<5} vs BOT {game['bot_color']:<5} | "
                f"winner={game['winner']:<8} | moves={game['move_count']}",
                flush=True,
            )

    expected = len(BOT_SPECS) * 2
    if len(payload["games"]) != expected:
        raise RuntimeError(f"Zapisano {len(payload['games'])} gier zamiast {expected}")

    print("\n[DONE]")
    print(f"Zapisano dokładnie {expected} gier.")
    print(f"Plik: {output}")


if __name__ == "__main__":
    main()
