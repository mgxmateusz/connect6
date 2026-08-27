from __future__ import annotations

import argparse
from pathlib import Path

import torch

from connect6.bots.gpu_bot import (
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV3,
    GPUTacticalBotV4,
)
from connect6.engine.checkpoint import load_model_for_inference
from connect6.engine.model import mask_logits
from connect6.engine.vector_env import VectorConnect6, canonical_network_input


def _find_checkpoint(runs_dir: Path) -> Path:
    latest = list(runs_dir.rglob("checkpoints/latest.pt")) if runs_dir.exists() else []
    if latest:
        return max(latest, key=lambda p: p.stat().st_mtime)

    versioned = (
        list(runs_dir.rglob("checkpoints/model_update_*.pt"))
        if runs_dir.exists()
        else []
    )
    if versioned:
        return max(versioned, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(
        f"No checkpoint found under {runs_dir}. Use --checkpoint PATH."
    )


def _parse_batch_sizes(raw: str) -> list[int]:
    result = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not result or any(x <= 0 for x in result):
        raise ValueError("--batch-sizes must contain positive integers")
    return result


def _validate_stone_range(min_stones: int, max_stones: int) -> None:
    if min_stones < 1 or max_stones >= 361 or min_stones > max_stones:
        raise ValueError("stone range must satisfy 1 <= min <= max < 361")
    if min_stones % 2 == 0 or max_stones % 2 == 0:
        raise ValueError("--min-stones and --max-stones must be odd turn-start counts")


def _sample_turn_start_targets(
    count: int,
    device: torch.device,
    min_stones: int,
    max_stones: int,
) -> torch.Tensor:
    min_turn = (min_stones - 1) // 2
    max_turn = (max_stones - 1) // 2
    turns = torch.randint(
        min_turn,
        max_turn + 1,
        (count,),
        dtype=torch.int16,
        device=device,
    )
    return turns * 2 + 1


@torch.no_grad()
def _make_legal_positions(
    batch: int,
    device: torch.device,
    min_stones: int,
    max_stones: int,
    seed: int,
):
    """Generate legal, non-terminal Connect6 positions at a two-stone turn start."""
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    env = VectorConnect6(
        num_envs=batch,
        board_size=19,
        win_length=6,
        device=device,
        debug_checks=False,
    )

    targets = _sample_turn_start_targets(batch, device, min_stones, max_stones)
    captured = torch.zeros(batch, dtype=torch.bool, device=device)
    out_boards = torch.empty_like(env.boards)
    out_players = torch.empty_like(env.current_player)
    out_counts = torch.empty(batch, dtype=torch.int16, device=device)

    max_generation_steps = max(4096, max_stones * 40)
    for _ in range(max_generation_steps):
        legal = env.legal_mask()
        random_scores = torch.rand(
            (batch, env.action_size), dtype=torch.float32, device=device
        )
        random_scores.masked_fill_(~legal, -1.0)
        actions = random_scores.argmax(dim=1)

        step = env.step(actions)

        ready = (
            ~captured
            & ~step.done
            & env.move_count.eq(targets)
            & env.stones_left.eq(2)
        )
        if bool(ready.any()):
            out_boards[ready] = env.boards[ready]
            out_players[ready] = env.current_player[ready]
            out_counts[ready] = env.move_count[ready]
            captured[ready] = True

        if bool(captured.all()):
            break

        if bool(step.done.any()):
            reset_idx = torch.nonzero(step.done, as_tuple=False).flatten()
            env.reset(reset_idx)
            retry = step.done & ~captured
            if bool(retry.any()):
                retry_idx = torch.nonzero(retry, as_tuple=False).flatten()
                targets[retry_idx] = _sample_turn_start_targets(
                    int(retry_idx.numel()), device, min_stones, max_stones
                )
    else:
        missing = int((~captured).sum().item())
        raise RuntimeError(
            f"Could not generate {missing}/{batch} legal benchmark positions"
        )

    stones_left = torch.full((batch,), 2, dtype=torch.int8, device=device)
    return out_boards, out_players, stones_left, out_counts


def _elapsed_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def _elapsed_search_avg_decision_ms(
    bot,
    boards: torch.Tensor,
    players: torch.Tensor,
    *,
    warmup: int,
    iterations: int,
) -> float:
    """Measure full two-stone planning and report average ms per played stone."""
    left_two = torch.full(
        (boards.shape[0],), 2, dtype=torch.int8, device=boards.device
    )
    left_one = torch.ones(
        (boards.shape[0],), dtype=torch.int8, device=boards.device
    )
    bot.reset()

    for _ in range(warmup):
        bot.actions(boards, players, left_two)
        bot.actions(boards, players, left_one)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        bot.actions(boards, players, left_two)
        bot.actions(boards, players, left_one)
    end.record()
    end.synchronize()

    total_ms = float(start.elapsed_time(end))
    return total_ms / (iterations * 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU decision benchmark on legal non-terminal Connect6 positions"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--batch-sizes", default="1,32,128,256,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument(
        "--min-stones",
        type=int,
        default=41,
        help="Minimum odd stone count at the start of a full two-stone turn.",
    )
    parser.add_argument(
        "--max-stones",
        type=int,
        default=81,
        help="Maximum odd stone count at the start of a full two-stone turn.",
    )
    parser.add_argument("--position-seed", type=int, default=12345)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    _validate_stone_range(args.min_stones, args.max_stones)

    device = torch.device("cuda")
    checkpoint = args.checkpoint or _find_checkpoint(args.runs_dir)
    model, payload = load_model_for_inference(checkpoint, device)

    bot_v1 = GPUTacticalBot(device)
    bot_v2 = GPUTacticalBotV2(device)
    bot_v3 = GPUTacticalBotV3(device)
    bot_v4 = GPUTacticalBotV4(device)

    tr_cfg = payload.get("config", {}).get("training", {})
    amp_enabled = bool(tr_cfg.get("amp", True))
    amp_name = str(tr_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = (
        torch.float16
        if amp_name in {"float16", "fp16", "half"}
        else torch.bfloat16
    )

    seed_board = torch.zeros((1, 19, 19), dtype=torch.int8, device=device)
    seed_player = torch.ones(1, dtype=torch.int8, device=device)
    seed_left_one = torch.ones(1, dtype=torch.int8, device=device)
    seed_left_two = torch.full((1,), 2, dtype=torch.int8, device=device)
    bot_v1.actions(seed_board, seed_player, seed_left_one)
    bot_v2.actions(seed_board, seed_player, seed_left_one)
    for bot in (bot_v3, bot_v4):
        bot.reset()
        bot.actions(seed_board, seed_player, seed_left_two)
        bot.actions(seed_board, seed_player, seed_left_one)
    torch.cuda.synchronize()

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Checkpoint: {checkpoint}")
    print(
        "Positions: legal random VectorConnect6 trajectories, non-terminal, "
        f"captured at full-turn start with {args.min_stones}-{args.max_stones} stones."
    )
    print(
        "CNN timing: board -> canonical input -> forward -> legal mask -> argmax "
        f"(autocast={'on' if amp_enabled else 'off'}, dtype={amp_name})"
    )
    print("V1/V2 timing: one native CUDA scoring decision at stones_left=2.")
    print("V3: TOP16 current cells -> all C(16,2)=120 pair states; no reply search.")
    print(
        "V4: TOP12 current cells -> all C(12,2)=66 pair states -> TOP4 own pairs -> "
        "for each pair opponent V2 TOP4 single replies -> exact reply state eval -> maximin."
    )
    print(
        "V3 work: 1 score_all + 120 state evals. V4 work: 1 own score_all + 66 own "
        "state evals + 4 opponent score_all + up to 4*4=16 reply state evals."
    )
    print("Ratios are always bot_ms / CNN_ms; lower is faster relative to CNN.")
    print()

    print(
        f"{'batch':>6} | {'stones':>11} | {'CNN ms':>9} | {'V1 ms':>8} | {'V2 ms':>8} | "
        f"{'V3 Top16':>10} | {'V4 T12/R4':>10} | {'V1/CNN':>7} | "
        f"{'V2/CNN':>7} | {'V3/CNN':>7} | {'V4/CNN':>7}"
    )
    print("-" * 126)

    for batch in _parse_batch_sizes(args.batch_sizes):
        boards, players, left, stone_counts = _make_legal_positions(
            batch,
            device,
            args.min_stones,
            args.max_stones,
            args.position_seed + batch,
        )
        legal = boards.view(batch, -1).eq(0)

        min_count = int(stone_counts.min().item())
        max_count = int(stone_counts.max().item())
        mean_count = float(stone_counts.float().mean().item())
        stone_label = f"{min_count}-{max_count}/{mean_count:.0f}"

        def model_decision():
            network_input = canonical_network_input(boards, players, left)
            with torch.autocast(
                device_type="cuda",
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits, _ = model(network_input)
            logits = mask_logits(logits.float(), legal)
            return logits.argmax(dim=1)

        def bot_v1_decision():
            return bot_v1.actions(boards, players, left)

        def bot_v2_decision():
            return bot_v2.actions(boards, players, left)

        model_ms = _elapsed_ms(model_decision, args.warmup, args.iters)
        v1_ms = _elapsed_ms(bot_v1_decision, args.warmup, args.iters)
        v2_ms = _elapsed_ms(bot_v2_decision, args.warmup, args.iters)
        v3_ms = _elapsed_search_avg_decision_ms(
            bot_v3,
            boards,
            players,
            warmup=args.warmup,
            iterations=args.iters,
        )
        v4_ms = _elapsed_search_avg_decision_ms(
            bot_v4,
            boards,
            players,
            warmup=args.warmup,
            iterations=args.iters,
        )

        print(
            f"{batch:6d} | {stone_label:>11} | {model_ms:9.4f} | {v1_ms:8.4f} | "
            f"{v2_ms:8.4f} | {v3_ms:10.4f} | {v4_ms:10.4f} | "
            f"{v1_ms / model_ms:7.3f} | {v2_ms / model_ms:7.3f} | "
            f"{v3_ms / model_ms:7.3f} | {v4_ms / model_ms:7.3f}"
        )


if __name__ == "__main__":
    main()
