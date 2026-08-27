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
from connect6.engine.vector_env import canonical_network_input


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


def _make_positions(batch: int, device: torch.device, occupied_fraction: float):
    rnd = torch.rand((batch, 19, 19), device=device)
    half = occupied_fraction * 0.5
    boards = torch.zeros((batch, 19, 19), dtype=torch.int8, device=device)
    boards[rnd < half] = 1
    boards[(rnd >= half) & (rnd < occupied_fraction)] = -1
    boards[:, 9, 9] = 0
    ids = torch.arange(batch, device=device)
    current_player = torch.where(
        ids.remainder(2).eq(0),
        torch.ones_like(ids, dtype=torch.int8),
        -torch.ones_like(ids, dtype=torch.int8),
    )
    stones_left = torch.where(
        ids.remainder(3).eq(0),
        torch.ones_like(ids, dtype=torch.int8),
        torch.full_like(ids, 2, dtype=torch.int8),
    )
    return boards, current_player, stones_left


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
        description="GPU decision benchmark: CNN vs Tactical Bot V1/V2/V3/V4"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--batch-sizes", default="1,32,128,256,512,1024,2048")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--occupied", type=float, default=0.35)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if not (0.0 <= args.occupied < 1.0):
        raise ValueError("--occupied must be in [0, 1)")

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
        "CNN timing: board -> canonical input -> forward -> legal mask -> argmax "
        f"(autocast={'on' if amp_enabled else 'off'}, dtype={amp_name})"
    )
    print("V1/V2 timing: one native CUDA scoring decision.")
    print("V3: TOP16 current cells -> all C(16,2)=120 pair states; no reply search.")
    print(
        "V4: TOP8 current cells -> all C(8,2)=28 pair states -> TOP4 pairs -> "
        "every legal one-stone opponent reply -> maximin."
    )
    print(
        "V3 work: 1 score_all + 120 state evals. V4 work: 1 score_all + 28 own "
        "state evals + up to 4*(legal opponent cells) reply state evals."
    )
    print("Ratios are always bot_ms / CNN_ms; lower is faster relative to CNN.")
    print()

    print(
        f"{'batch':>6} | {'CNN ms':>9} | {'V1 ms':>8} | {'V2 ms':>8} | "
        f"{'V3 Top16':>10} | {'V4 Reply1':>10} | {'V1/CNN':>7} | "
        f"{'V2/CNN':>7} | {'V3/CNN':>7} | {'V4/CNN':>7}"
    )
    print("-" * 112)

    for batch in _parse_batch_sizes(args.batch_sizes):
        boards, players, left = _make_positions(batch, device, args.occupied)
        legal = boards.view(batch, -1).eq(0)

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
            f"{batch:6d} | {model_ms:9.4f} | {v1_ms:8.4f} | {v2_ms:8.4f} | "
            f"{v3_ms:10.4f} | {v4_ms:10.4f} | {v1_ms / model_ms:7.3f} | "
            f"{v2_ms / model_ms:7.3f} | {v3_ms / model_ms:7.3f} | "
            f"{v4_ms / model_ms:7.3f}"
        )


if __name__ == "__main__":
    main()
