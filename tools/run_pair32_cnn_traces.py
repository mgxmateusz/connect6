from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect6.bots.gpu_bot import GPUTacticalBotPairFirst32
from connect6.championship.championship import discover_checkpoints
from connect6.championship.championship_cnn import _black_to_move, _masked_step
from connect6.engine.vector_env import VectorConnect6
from connect6.evaluation.bot_strength_arena_fast import NativePolicyPool


BOARD = 19
CENTER = (BOARD // 2) * BOARD + (BOARD // 2)
BOT_KEY = "pair32"
BOT_LABEL = "GPU Tactical Bot PairFirst AllPairs P32"
BOT_SIGNATURE = "gpu_tactical_bot_pairfirst_allpairs_p32_v1"


def _completed(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            model = str(row.get("model", ""))
            color = str(row.get("cnn_color", ""))
            if model and color in {"black", "white"}:
                done.add((model, color))
    return done


def _winner_labels(winner: int, *, cnn_is_black: bool) -> tuple[str, str]:
    if winner == 0:
        return "draw", "draw"
    winner_color = "black" if winner == 1 else "white"
    cnn_won = (winner == 1 and cnn_is_black) or (winner == -1 and not cnn_is_black)
    return ("cnn" if cnn_won else BOT_KEY), winner_color


def _append_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        f.flush()


@torch.inference_mode()
def _play_batch(
    pool: NativePolicyPool,
    refs,
    model_ids: torch.Tensor,
    *,
    cnn_is_black: bool,
    bot: GPUTacticalBotPairFirst32,
) -> list[dict]:
    batch = len(refs)
    device = pool.device
    env = VectorConnect6(batch, BOARD, 6, device=device, debug_checks=False)
    env.reset()
    bot.reset()

    active = torch.ones(batch, dtype=torch.bool, device=device)
    winners = torch.zeros(batch, dtype=torch.int8, device=device)
    traces = torch.full((batch, BOARD * BOARD), -1, dtype=torch.int16, device=device)

    # Connect6 opening is fixed for this diagnostic: black always places the
    # very first stone at the exact centre. Neither CNN nor Pair32 chooses it.
    opening = torch.full((batch,), CENTER, dtype=torch.long, device=device)
    traces[:, 0] = CENTER
    done, winner = _masked_step(env, opening, active)
    newly_done = active & done
    winners = torch.where(newly_done, winner, winners)
    active &= ~done

    # move_index is the absolute stone index, so after the forced opening we
    # continue from index 1. Source ownership is determined only by colour.
    for move_index in range(1, BOARD * BOARD):
        model_to_move = _black_to_move(move_index) == bool(cnn_is_black)
        if model_to_move:
            actions = pool.actions(
                env.boards,
                env.current_player,
                env.stones_left,
                model_ids,
            )
        else:
            actions = bot.actions(env.boards, env.current_player, env.stones_left)

        # Finished tables are frozen by _masked_step. Force their placeholder
        # action to an in-range value in case a search bot reports -1 on a
        # terminal/full board.
        actions = actions.to(device=device, dtype=torch.long)
        actions = torch.where(active, actions, torch.zeros_like(actions))
        traces[:, move_index] = torch.where(
            active,
            actions.to(torch.int16),
            torch.full_like(actions, -1, dtype=torch.int16),
        )

        done, winner = _masked_step(env, actions, active)
        newly_done = active & done
        winners = torch.where(newly_done, winner, winners)
        active &= ~done
        if not bool(active.any().item()):
            break

    if bool(active.any().item()):
        raise RuntimeError("Nie wszystkie partie zakończyły się po 361 kamieniach")

    traces_cpu = traces.cpu().tolist()
    winners_cpu = winners.cpu().tolist()
    moves_cpu = env.move_count.cpu().tolist()

    rows: list[dict] = []
    cnn_color = "black" if cnn_is_black else "white"
    bot_color = "white" if cnn_is_black else "black"
    for ref, trace, winner, move_count in zip(refs, traces_cpu, winners_cpu, moves_cpu):
        count = int(move_count)
        actions = [int(a) for a in trace[:count]]
        result, winner_color = _winner_labels(int(winner), cnn_is_black=cnn_is_black)
        rows.append(
            {
                "model": ref.name,
                "update": int(ref.update),
                "cnn_color": cnn_color,
                "bot": BOT_KEY,
                "bot_label": BOT_LABEL,
                "bot_signature": BOT_SIGNATURE,
                "bot_color": bot_color,
                "forced_center": CENTER,
                "winner": result,
                "winner_color": winner_color,
                "move_count": count,
                "actions": actions,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Trace every CNN checkpoint vs latest Pair32 bot: two games/model, "
            "CNN once black and once white, with black's first stone forced to centre."
        )
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("runs/connect6_cnn_05/checkpoints"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/connect6_cnn_05/pair32_vs_cnn_traces.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--native-load-chunk-models", type=int, default=32)
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing trace file before running. Default is resume.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Ten trace runner wymaga CUDA")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("Trace runner używa native SM120 i wymaga RTX 50")
    if args.batch_size <= 0 or args.native_load_chunk_models <= 0:
        raise ValueError("batch sizes muszą być dodatnie")

    checkpoint_dir = args.checkpoint_dir.resolve()
    output = args.output.resolve()
    refs = discover_checkpoints(checkpoint_dir)
    if not refs:
        raise RuntimeError(f"Brak model_update_*.pt w {checkpoint_dir}")

    if args.reset and output.exists():
        output.unlink()

    done = _completed(output)
    total_target = len(refs) * 2
    print("=" * 78)
    print("PAIR32 VS ALL CNN CHECKPOINTS — CENTER-OPENING TRACE")
    print("=" * 78)
    print(f"Checkpointy: {len(refs):,}")
    print(f"Docelowe gry: {total_target:,} (2/model)")
    print(f"Gotowe z pliku: {len(done):,}")
    print(f"Output: {output}")
    print(f"Bot: {BOT_LABEL}")
    print(f"Opening: action={CENTER}, row=9, col=9 (forced black centre)")

    # Pack every CNN checkpoint once with the same fast native policy path used
    # by the strength arena. This avoids loading thousands of .pt files per game.
    pool = NativePolicyPool(
        refs,
        device,
        load_chunk=int(args.native_load_chunk_models),
    )
    bot = GPUTacticalBotPairFirst32(device)
    bot.reset()
    # Force extension build/load before the first timed game batch.
    warm_board = torch.zeros((1, BOARD, BOARD), dtype=torch.int8, device=device)
    warm_player = torch.ones(1, dtype=torch.int8, device=device)
    warm_left = torch.full((1,), 2, dtype=torch.int8, device=device)
    bot.actions(warm_board, warm_player, warm_left)
    bot.reset()

    newly_written = 0
    for cnn_is_black in (True, False):
        color = "black" if cnn_is_black else "white"
        pending_indices = [
            i for i, ref in enumerate(refs) if (ref.name, color) not in done
        ]
        print(f"\nCNN {color.upper()}: pending={len(pending_indices):,}")

        for start in range(0, len(pending_indices), int(args.batch_size)):
            ids = pending_indices[start : start + int(args.batch_size)]
            batch_refs = [refs[i] for i in ids]
            model_ids = torch.tensor(ids, dtype=torch.int32, device=device)
            rows = _play_batch(
                pool,
                batch_refs,
                model_ids,
                cnn_is_black=cnn_is_black,
                bot=bot,
            )
            _append_rows(output, rows)
            newly_written += len(rows)
            done.update((row["model"], row["cnn_color"]) for row in rows)

            cnn_wins = sum(row["winner"] == "cnn" for row in rows)
            bot_wins = sum(row["winner"] == BOT_KEY for row in rows)
            draws = len(rows) - cnn_wins - bot_wins
            print(
                f"[{len(done):>5}/{total_target}] +{len(rows):>3} | "
                f"CNN/BOT/D={cnn_wins}/{bot_wins}/{draws} | "
                f"last={batch_refs[-1].name}",
                flush=True,
            )

    print("\n[DONE]")
    print(f"Nowe gry zapisane: {newly_written:,}")
    print(f"Łącznie w trace: {len(done):,}/{total_target:,}")
    print(f"Plik: {output}")


if __name__ == "__main__":
    main()
