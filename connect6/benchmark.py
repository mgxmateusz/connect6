from __future__ import annotations

import argparse
import time

import torch

from .config import load_config
from .model import build_model, mask_logits
from .vector_env import VectorConnect6


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark wektorowego środowiska Connect6 i inference modelu")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tr, game_cfg, model_cfg = cfg["training"], cfg["game"], cfg["model"]
    device = torch.device(tr.get("device", "cuda"))
    n = int(tr.get("num_envs", 1024))
    size = int(game_cfg.get("board_size", 19))
    env = VectorConnect6(n, size, int(game_cfg.get("win_length", 6)), device)
    model = build_model(model_cfg, size).to(device).eval()

    # Rozgrzewka GPU przed właściwym pomiarem.
    with torch.inference_mode():
        for _ in range(20):
            network_input = env.network_input()
            logits, _ = model(network_input)
            logits = mask_logits(logits.float(), env.legal_mask())
            actions = logits.argmax(1)
            step = env.step(actions)
            env.reset(torch.nonzero(step.done, as_tuple=False).flatten())
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.steps):
            network_input = env.network_input()
            logits, _ = model(network_input)
            logits = mask_logits(logits.float(), env.legal_mask())
            actions = logits.argmax(1)
            step = env.step(actions)
            env.reset(torch.nonzero(step.done, as_tuple=False).flatten())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    positions = args.steps * n
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Parallel boards: {n}")
    print(f"Positions: {positions:,}")
    print(f"Elapsed: {elapsed:.3f}s")
    print(f"Throughput: {positions/elapsed:,.0f} positions/s")


if __name__ == "__main__":
    main()
