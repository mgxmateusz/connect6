import argparse
import statistics
import time
from dataclasses import dataclass

import torch
import torch.nn as nn


BOARD = 19
IN_CHANNELS = 4


@dataclass(frozen=True)
class Architecture:
    name: str
    kernels: tuple[int, ...]

    @property
    def receptive_field(self):
        return 1 + sum(k - 1 for k in self.kernels)


ARCHITECTURES = [
    Architecture(
        "19-5-5-3-3-3-3-3",
        (19, 5, 5, 3, 3, 3, 3, 3),
    ),
    Architecture(
        "19-7-5-5-3-3",
        (19, 7, 5, 5, 3, 3),
    ),
    Architecture(
        "23-3-3-3-3-3-3-3",
        (23, 3, 3, 3, 3, 3, 3, 3),
    ),
]


class Connect6CNN(nn.Module):
    def __init__(self, kernels, channels):
        super().__init__()

        layers = []
        in_ch = IN_CHANNELS

        for k in kernels:
            layers.append(
                nn.Conv2d(
                    in_ch,
                    channels,
                    kernel_size=k,
                    padding=k // 2,
                    bias=True,
                )
            )
            layers.append(nn.ReLU(inplace=True))
            in_ch = channels

        self.backbone = nn.Sequential(*layers)
        self.policy = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x):
        x = self.backbone(x)
        x = self.policy(x)
        return x.flatten(1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def make_batch(batch_size, device, dtype):
    x = torch.randn(
        batch_size,
        IN_CHANNELS,
        BOARD,
        BOARD,
        device=device,
        dtype=dtype,
    )
    target = torch.randint(
        0,
        BOARD * BOARD,
        (batch_size,),
        device=device,
    )
    return x, target


def training_step(model, optimizer, criterion, x, target):
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits.float(), target)
    loss.backward()
    optimizer.step()
    return loss


def warmup_training(model, optimizer, criterion, x, target, device, seconds):
    start = time.perf_counter()
    steps = 0

    while True:
        training_step(model, optimizer, criterion, x, target)
        steps += 1

        if steps % 10 == 0:
            sync(device)
            if time.perf_counter() - start >= seconds:
                break

    sync(device)


def benchmark_training(model, optimizer, criterion, x, target, device, seconds):
    batch_size = x.shape[0]

    sync(device)
    start = time.perf_counter()

    steps = 0
    step_times = []

    while True:
        t0 = time.perf_counter()

        training_step(model, optimizer, criterion, x, target)

        sync(device)
        t1 = time.perf_counter()

        step_times.append(t1 - t0)
        steps += 1

        if t1 - start >= seconds:
            break

    total_time = time.perf_counter() - start

    return {
        "steps": steps,
        "total_time_s": total_time,
        "mean_step_ms": statistics.mean(step_times) * 1000.0,
        "median_step_ms": statistics.median(step_times) * 1000.0,
        "samples_per_s": (steps * batch_size) / total_time,
    }


@torch.no_grad()
def benchmark_inference(model, device, dtype, warmup_seconds, seconds):
    model.eval()

    x = torch.randn(
        1,
        IN_CHANNELS,
        BOARD,
        BOARD,
        device=device,
        dtype=dtype,
    )

    start = time.perf_counter()
    steps = 0

    while True:
        _ = model(x)
        steps += 1

        if steps % 50 == 0:
            sync(device)
            if time.perf_counter() - start >= warmup_seconds:
                break

    sync(device)

    start = time.perf_counter()
    steps = 0

    while True:
        _ = model(x)
        steps += 1

        if steps % 50 == 0:
            sync(device)
            if time.perf_counter() - start >= seconds:
                break

    sync(device)

    elapsed = time.perf_counter() - start

    return {
        "decisions_per_s": steps / elapsed,
        "mean_ms": elapsed / steps * 1000.0,
        "steps": steps,
    }


def human(n):
    return f"{n:,.0f}".replace(",", " ")


def main():
    parser = argparse.ArgumentParser(
        description="Connect6 RF37: compare 3 candidate CNN architectures"
    )

    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--train-seconds", type=float, default=30.0)
    parser.add_argument("--inference-seconds", type=float, default=10.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        device = torch.device("cpu")

    if args.fp16:
        if device.type != "cuda":
            raise RuntimeError("--fp16 wymaga CUDA.")
        dtype = torch.float16
    else:
        dtype = torch.float32

    print("=" * 100)
    print("CONNECT6 — RF37 TRIPLET COMPARISON")
    print("=" * 100)
    print(f"PyTorch:          {torch.__version__}")
    print(f"Device:           {device}")
    if device.type == "cuda":
        print(f"GPU:              {torch.cuda.get_device_name(device)}")
        print(f"CUDA:             {torch.version.cuda}")
    print(f"Dtype:            {dtype}")
    print(f"Board:            {BOARD}x{BOARD}")
    print(f"Hidden channels:  {args.channels}")
    print(f"Training batch:   {args.batch}")
    print()

    results = []

    for arch in ARCHITECTURES:
        torch.manual_seed(1234)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(1234)

        model = Connect6CNN(
            arch.kernels,
            args.channels,
        ).to(
            device=device,
            dtype=dtype,
        )

        params = count_parameters(model)

        print("=" * 100)
        print(arch.name)
        print("=" * 100)
        print(f"Kernels:          {list(arch.kernels)}")
        print(f"Layers:           {len(arch.kernels)}")
        print(f"RF:               {arch.receptive_field}x{arch.receptive_field}")
        print(f"Parameters:       {human(params)}")

        print("\nInference batch=1: warmup/test...")

        inference = benchmark_inference(
            model,
            device,
            dtype,
            args.warmup_seconds,
            args.inference_seconds,
        )

        print(
            f"  {inference['mean_ms']:.4f} ms/decyzję | "
            f"{human(inference['decisions_per_s'])} decyzji/s"
        )

        model.train()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
        )

        criterion = nn.CrossEntropyLoss()

        x, target = make_batch(
            args.batch,
            device,
            dtype,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        print("\nTraining: warmup...")

        warmup_training(
            model,
            optimizer,
            criterion,
            x,
            target,
            device,
            args.warmup_seconds,
        )

        print(
            f"Training: test {args.train_seconds:.0f}s...",
            flush=True,
        )

        training = benchmark_training(
            model,
            optimizer,
            criterion,
            x,
            target,
            device,
            args.train_seconds,
        )

        peak_vram = None

        if device.type == "cuda":
            peak_vram = (
                torch.cuda.max_memory_allocated(device) / 1024**3
            )

        print(
            f"  {human(training['samples_per_s'])} pozycji/s | "
            f"{training['mean_step_ms']:.3f} ms/krok | "
            f"{training['steps']} kroków"
        )

        if peak_vram is not None:
            print(f"  Peak VRAM:       {peak_vram:.3f} GB")

        results.append(
            {
                "arch": arch,
                "params": params,
                "inference": inference,
                "training": training,
                "peak_vram": peak_vram,
            }
        )

        del model, optimizer, criterion, x, target

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n\n" + "=" * 100)
    print("RANKING TRENINGU")
    print("=" * 100)

    train_rank = sorted(
        results,
        key=lambda r: r["training"]["samples_per_s"],
        reverse=True,
    )

    leader = train_rank[0]["training"]["samples_per_s"]

    for i, r in enumerate(train_rank, 1):
        speed = r["training"]["samples_per_s"]
        print(
            f"{i}. {r['arch'].name:<24} "
            f"{human(speed):>9} pozycji/s | "
            f"{speed / leader * 100:6.1f}% lidera"
        )

    print("\n" + "=" * 100)
    print("RANKING DECYZJI batch=1")
    print("=" * 100)

    inf_rank = sorted(
        results,
        key=lambda r: r["inference"]["mean_ms"],
    )

    best_ms = inf_rank[0]["inference"]["mean_ms"]

    for i, r in enumerate(inf_rank, 1):
        ms = r["inference"]["mean_ms"]
        print(
            f"{i}. {r['arch'].name:<24} "
            f"{ms:>8.4f} ms | "
            f"{ms / best_ms:6.3f}x czasu lidera"
        )


if __name__ == "__main__":
    main()