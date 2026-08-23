from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .checkpoint import load_model_for_inference
from .gpu_bot import GPUTacticalBot
from .model import mask_logits
from .vector_env import canonical_network_input


def _find_checkpoint(runs_dir: Path) -> Path:
    latest = list(runs_dir.rglob("checkpoints/latest.pt")) if runs_dir.exists() else []
    if latest:
        return max(latest, key=lambda p: p.stat().st_mtime)

    versioned = list(runs_dir.rglob("checkpoints/model_update_*.pt")) if runs_dir.exists() else []
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
    # Fixed random GPU-resident positions. Timing does not include generating data.
    rnd = torch.rand((batch, 19, 19), device=device)
    half = occupied_fraction * 0.5
    boards = torch.zeros((batch, 19, 19), dtype=torch.int8, device=device)
    boards[rnd < half] = 1
    boards[(rnd >= half) & (rnd < occupied_fraction)] = -1

    # Always leave the centre legal so every synthetic board has a legal action.
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="GPU decision benchmark: CNN policy vs one-pass tactical bot"
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--batch-sizes", default="1,32,128,512,2048")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--occupied", type=float, default=0.35)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    if not (0.0 <= args.occupied < 1.0):
        raise ValueError("--occupied must be in [0, 1)")

    device = torch.device("cuda")
    checkpoint = args.checkpoint or _find_checkpoint(args.runs_dir)
    model, payload = load_model_for_inference(checkpoint, device)
    bot = GPUTacticalBot(device)

    tr_cfg = payload.get("config", {}).get("training", {})
    amp_enabled = bool(tr_cfg.get("amp", True))
    amp_name = str(tr_cfg.get("amp_dtype", "bfloat16")).lower()
    amp_dtype = torch.float16 if amp_name in {"float16", "fp16", "half"} else torch.bfloat16

    # Compile/load the native bot before any timed region.
    seed_board = torch.zeros((1, 19, 19), dtype=torch.int8, device=device)
    seed_player = torch.ones(1, dtype=torch.int8, device=device)
    seed_left = torch.ones(1, dtype=torch.int8, device=device)
    bot.actions(seed_board, seed_player, seed_left)
    torch.cuda.synchronize()

    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Checkpoint: {checkpoint}")
    print(
        "CNN timing: board -> canonical input -> forward -> legal mask -> argmax "
        f"(autocast={'on' if amp_enabled else 'off'}, dtype={amp_name})"
    )
    print("Bot timing: board -> one native CUDA threat-scoring kernel -> argmax")
    print()
    print(
        f"{'batch':>7} | {'CNN ms':>10} | {'BOT ms':>10} | "
        f"{'BOT faster':>11} | {'CNN pos/s':>12} | {'BOT pos/s':>12}"
    )
    print("-" * 77)

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

        def bot_decision():
            return bot.actions(boards, players, left)

        model_ms = _elapsed_ms(model_decision, args.warmup, args.iters)
        bot_ms = _elapsed_ms(bot_decision, args.warmup, args.iters)
        speedup = model_ms / bot_ms if bot_ms > 0 else float("inf")
        model_rate = batch * 1000.0 / model_ms
        bot_rate = batch * 1000.0 / bot_ms

        print(
            f"{batch:7d} | {model_ms:10.4f} | {bot_ms:10.4f} | "
            f"{speedup:10.2f}x | {model_rate:12,.0f} | {bot_rate:12,.0f}"
        )


if __name__ == "__main__":
    main()
