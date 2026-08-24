from __future__ import annotations

import time

import torch

from connect6.engine.model import PolicyValueNet
from connect6.engine.native_rollout import NativeRolloutCollector, build_rollout_assignments
from connect6.engine.native_rollout_state import pack_rollout_models


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA jest wymagane")
    device = torch.device("cuda")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError(f"Smoke test jest dla SM120; wykryto {torch.cuda.get_device_capability(device)}")

    # Match the real training collector size from configs/train.yaml.
    # num_envs=2048 and completed_positions_per_update=384, so the benchmark
    # stops at the same completed-position target as one genuine PPO update.
    envs = 2048
    completed_positions_per_update = 384
    target = envs * completed_positions_per_update

    model = PolicyValueNet(board_size=19).to(device).eval()
    collector = NativeRolloutCollector(
        num_envs=envs,
        target_completed_positions=target,
        device=device,
        verbose_build=True,
    )
    packed = pack_rollout_models(model, [], device)
    assignments = build_rollout_assignments(
        envs,
        historical_model_count=0,
        device=device,
        historical_fraction=0.25,
        bot_fraction=0.25,
        bot_v1_fraction=0.50,
    )

    # Build/load the native extension before the timed section. On the first run
    # NVCC compilation can take tens of seconds and would otherwise completely
    # swamp the actual rollout timing, making the throughput number meaningless.
    collector._ext()

    # No untimed gameplay warmup: persistent boards should start exactly like a
    # fresh training run. The measured section therefore represents one complete
    # collector update from empty boards to the configured PPO sample target.
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    stats = collector.collect(
        packed,
        assignments,
        temperature=1.0,
        random_black_opening_fraction=0.50,
        seed=12345,
        symmetry_augmentation=True,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    completed_idx, returns = collector.buffer.completed_samples(gamma=0.98)

    graph_positions = stats.graph_steps * envs
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"envs: {envs:,}")
    print(f"completed_positions_per_update: {completed_positions_per_update:,}")
    print(f"target: {target:,}")
    print(f"completed: {stats.completed_positions:,}")
    print(f"buffer current-policy positions: {stats.generated_positions:,}")
    print(f"completed samples: {completed_idx.numel():,}")
    print(f"games: {stats.games:,}")
    print(f"graph steps: {stats.graph_steps:,}")
    print(f"graph positions processed: {graph_positions:,}")
    print(f"elapsed (rollout only): {elapsed:.4f} s")
    print(f"graph-step rate: {stats.graph_steps / max(elapsed, 1e-9):,.1f} steps/s")
    print(f"all-table throughput: {graph_positions / max(elapsed, 1e-9):,.0f} board-actions/s")
    print(f"generated throughput: {stats.generated_positions / max(elapsed, 1e-9):,.0f} current-policy pos/s")
    print(f"completed throughput: {stats.completed_positions / max(elapsed, 1e-9):,.0f} completed pos/s")
    print(f"returns finite: {bool(torch.isfinite(returns).all().item())}")

    if stats.completed_positions < target:
        raise RuntimeError("Collector nie osiągnął targetu")
    if int(completed_idx.numel()) != stats.completed_positions:
        raise RuntimeError("Licznik completed != liczba pełnych próbek")


if __name__ == "__main__":
    main()
