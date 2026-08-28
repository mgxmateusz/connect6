from __future__ import annotations

import argparse
from pathlib import Path

import torch

from connect6.bots.gpu_bot import (
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV2Pro,
    GPUTacticalBotV2Pro2,
    GPUTacticalBotV3,
    GPUTacticalBotV4,
    GPUTacticalBotFullPair,
    GPUTacticalBotPairFirst,
    GPUTacticalBotPairFirst32,
    GPUTacticalBotLiveRoad,
    GPUTacticalBotHybrid,
    GPUTacticalBotHybrid32,
)
from connect6.engine.checkpoint import load_model_for_inference
from connect6.engine.model import mask_logits
from connect6.engine.vector_env import VectorConnect6, canonical_network_input


def _find_checkpoint(runs_dir: Path) -> Path:
    latest = list(runs_dir.rglob("checkpoints/latest.pt")) if runs_dir.exists() else []
    if latest:
        return max(latest, key=lambda p: p.stat().st_mtime)
    versioned = list(runs_dir.rglob("checkpoints/model_update_*.pt")) if runs_dir.exists() else []
    if versioned:
        return max(versioned, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"No checkpoint found under {runs_dir}. Use --checkpoint PATH.")


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


def _sample_turn_start_targets(count, device, min_stones, max_stones):
    min_turn = (min_stones - 1) // 2
    max_turn = (max_stones - 1) // 2
    turns = torch.randint(min_turn, max_turn + 1, (count,), dtype=torch.int16, device=device)
    return turns * 2 + 1


@torch.no_grad()
def _make_legal_positions(batch, device, min_stones, max_stones, seed):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    env = VectorConnect6(batch, 19, 6, device=device, debug_checks=False)
    targets = _sample_turn_start_targets(batch, device, min_stones, max_stones)
    captured = torch.zeros(batch, dtype=torch.bool, device=device)
    out_boards = torch.empty_like(env.boards)
    out_players = torch.empty_like(env.current_player)
    out_counts = torch.empty(batch, dtype=torch.int16, device=device)
    max_generation_steps = max(4096, max_stones * 40)
    for _ in range(max_generation_steps):
        legal = env.legal_mask()
        random_scores = torch.rand((batch, env.action_size), dtype=torch.float32, device=device)
        random_scores.masked_fill_(~legal, -1.0)
        step = env.step(random_scores.argmax(dim=1))
        ready = ~captured & ~step.done & env.move_count.eq(targets) & env.stones_left.eq(2)
        if bool(ready.any()):
            out_boards[ready] = env.boards[ready]
            out_players[ready] = env.current_player[ready]
            out_counts[ready] = env.move_count[ready]
            captured[ready] = True
        if bool(captured.all()):
            break
        if bool(step.done.any()):
            env.reset(torch.nonzero(step.done, as_tuple=False).flatten())
            retry = step.done & ~captured
            if bool(retry.any()):
                retry_idx = torch.nonzero(retry, as_tuple=False).flatten()
                targets[retry_idx] = _sample_turn_start_targets(
                    int(retry_idx.numel()), device, min_stones, max_stones
                )
    else:
        raise RuntimeError(f"Could not generate {int((~captured).sum().item())}/{batch} legal positions")
    stones_left = torch.full((batch,), 2, dtype=torch.int8, device=device)
    return out_boards, out_players, stones_left, out_counts


def _elapsed_ms(fn, warmup, iterations):
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


def _elapsed_search_avg_decision_ms(bot, boards, players, *, warmup, iterations):
    left_two = torch.full((boards.shape[0],), 2, dtype=torch.int8, device=boards.device)
    left_one = torch.ones((boards.shape[0],), dtype=torch.int8, device=boards.device)
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
    return float(start.elapsed_time(end)) / (iterations * 2.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU decision benchmark on legal non-terminal Connect6 positions")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--batch-sizes", default="1,32,128,256,512,1024")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--heavy-iters", type=int, default=50)
    parser.add_argument("--heavy-warmup", type=int, default=1)
    parser.add_argument("--min-stones", type=int, default=41)
    parser.add_argument("--max-stones", type=int, default=81)
    parser.add_argument("--position-seed", type=int, default=12345)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if args.iters <= 0 or args.warmup < 0 or args.heavy_iters <= 0 or args.heavy_warmup < 0:
        raise ValueError("iteration counts must be positive and warmups non-negative")
    _validate_stone_range(args.min_stones, args.max_stones)
    device = torch.device("cuda")
    checkpoint = args.checkpoint or _find_checkpoint(args.runs_dir)
    model, payload = load_model_for_inference(checkpoint, device)

    bot_v1 = GPUTacticalBot(device)
    bot_v2 = GPUTacticalBotV2(device)
    bot_v2pro = GPUTacticalBotV2Pro(device)
    bot_v2pro2 = GPUTacticalBotV2Pro2(device)
    bot_v3 = GPUTacticalBotV3(device)
    bot_v4 = GPUTacticalBotV4(device)
    bot_p128 = GPUTacticalBotPairFirst(device)
    bot_p32 = GPUTacticalBotPairFirst32(device)
    bot_h128 = GPUTacticalBotHybrid(device)
    bot_h32 = GPUTacticalBotHybrid32(device)
    bot_live = GPUTacticalBotLiveRoad(device)
    bot_full = GPUTacticalBotFullPair(device)

    tr_cfg = payload.get("config", {}).get("training", {})
    amp_enabled = bool(tr_cfg.get("amp", True))
    amp_name = str(tr_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.float16 if amp_name in {"float16", "fp16", "half"} else torch.bfloat16

    seed_board = torch.zeros((1, 19, 19), dtype=torch.int8, device=device)
    seed_player = torch.ones(1, dtype=torch.int8, device=device)
    seed_left_one = torch.ones(1, dtype=torch.int8, device=device)
    seed_left_two = torch.full((1,), 2, dtype=torch.int8, device=device)
    bot_v1.actions(seed_board, seed_player, seed_left_one)
    bot_v2.actions(seed_board, seed_player, seed_left_one)
    bot_v2pro.actions(seed_board, seed_player, seed_left_one)
    bot_v2pro2.actions(seed_board, seed_player, seed_left_one)
    for bot in (bot_v3, bot_v4, bot_p128, bot_p32, bot_h128, bot_h32, bot_live, bot_full):
        bot.reset()
        bot.actions(seed_board, seed_player, seed_left_two)
        bot.actions(seed_board, seed_player, seed_left_one)
    torch.cuda.synchronize()

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Checkpoint: {checkpoint}")
    print("V2 Pro: one-cell latent-fork/road-leverage detection.")
    print("V2 Pro2: same V2Pro base plus local X+Y pair-force detection for 2->4/3->5/4->6.")
    print("Pair128/32: every legal pair gets the same cheap pair-aware score; only exact finalist count changes.")
    print("Hybrid128/32: same pure LiveRoad pool + same cheap pair score; only exact finalist count changes.")
    print()
    print(
        f"{'batch':>6} | {'stones':>11} | {'CNN ms':>9} | {'V1 ms':>8} | {'V2 ms':>8} | {'V2Pro':>8} | {'V2Pro2':>8} | "
        f"{'V3 ms':>9} | {'V4 ms':>9} | {'P128':>9} | {'P32':>9} | {'H128':>9} | {'H32':>9} | "
        f"{'LiveRoad':>10} | {'Full':>10} | {'P2/P1':>7} | {'P2/V2':>7} | {'P32/CNN':>8} | {'H32/CNN':>8}"
    )
    print("-" * 207)

    for batch in _parse_batch_sizes(args.batch_sizes):
        boards, players, left, stone_counts = _make_legal_positions(
            batch, device, args.min_stones, args.max_stones, args.position_seed + batch
        )
        legal = boards.view(batch, -1).eq(0)
        stone_label = f"{int(stone_counts.min())}-{int(stone_counts.max())}/{float(stone_counts.float().mean()):.0f}"

        def model_decision():
            network_input = canonical_network_input(boards, players, left)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                logits, _ = model(network_input)
            return mask_logits(logits.float(), legal).argmax(dim=1)

        model_ms = _elapsed_ms(model_decision, args.warmup, args.iters)
        v1_ms = _elapsed_ms(lambda: bot_v1.actions(boards, players, left), args.warmup, args.iters)
        v2_ms = _elapsed_ms(lambda: bot_v2.actions(boards, players, left), args.warmup, args.iters)
        v2pro_ms = _elapsed_ms(lambda: bot_v2pro.actions(boards, players, left), args.warmup, args.iters)
        v2pro2_ms = _elapsed_ms(lambda: bot_v2pro2.actions(boards, players, left), args.warmup, args.iters)
        v3_ms = _elapsed_search_avg_decision_ms(bot_v3, boards, players, warmup=args.warmup, iterations=args.iters)
        v4_ms = _elapsed_search_avg_decision_ms(bot_v4, boards, players, warmup=args.warmup, iterations=args.iters)
        p128_ms = _elapsed_search_avg_decision_ms(bot_p128, boards, players, warmup=args.warmup, iterations=args.iters)
        p32_ms = _elapsed_search_avg_decision_ms(bot_p32, boards, players, warmup=args.warmup, iterations=args.iters)
        h128_ms = _elapsed_search_avg_decision_ms(bot_h128, boards, players, warmup=args.warmup, iterations=args.iters)
        h32_ms = _elapsed_search_avg_decision_ms(bot_h32, boards, players, warmup=args.warmup, iterations=args.iters)
        live_ms = _elapsed_search_avg_decision_ms(bot_live, boards, players, warmup=args.heavy_warmup, iterations=args.heavy_iters)
        full_ms = _elapsed_search_avg_decision_ms(bot_full, boards, players, warmup=args.heavy_warmup, iterations=args.heavy_iters)

        print(
            f"{batch:6d} | {stone_label:>11} | {model_ms:9.4f} | {v1_ms:8.4f} | {v2_ms:8.4f} | {v2pro_ms:8.4f} | {v2pro2_ms:8.4f} | "
            f"{v3_ms:9.4f} | {v4_ms:9.4f} | {p128_ms:9.4f} | {p32_ms:9.4f} | {h128_ms:9.4f} | {h32_ms:9.4f} | "
            f"{live_ms:10.4f} | {full_ms:10.4f} | {v2pro2_ms/v2pro_ms:7.3f} | {v2pro2_ms/v2_ms:7.3f} | {p32_ms/model_ms:8.3f} | {h32_ms/model_ms:8.3f}"
        )


if __name__ == "__main__":
    main()
