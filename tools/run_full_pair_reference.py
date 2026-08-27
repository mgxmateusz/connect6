from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect6.bots.gpu_bot import (
    GPUTacticalBotFullPair,
    GPUTacticalBotSmall,
    GPUTacticalBotV3,
)
from connect6.championship import bot_arena as model_arena
from connect6.evaluation import bot_strength_arena as arena


SPECS = (
    model_arena.BotSpec(
        "small",
        "GPU Tactical Bot Small Top12 Pair-State",
        "gpu_tactical_bot_small_top12_pair_state_v1",
        GPUTacticalBotSmall,
    ),
    model_arena.BotSpec(
        "v3",
        "GPU Tactical Bot V3 Top16 Pair-State",
        "gpu_tactical_bot_v3_top16_pair_state_v1",
        GPUTacticalBotV3,
    ),
    model_arena.BotSpec(
        "full",
        "GPU Tactical Bot Full Pair Brute Force",
        "gpu_tactical_bot_full_pair_bruteforce_v1",
        GPUTacticalBotFullPair,
    ),
)


def _print_decisive(rows: list[dict[str, str]]) -> None:
    print("\n# DECISIVE-ONLY REFERENCE")
    for row in rows:
        if row.get("bot_a") not in {"small", "v3", "full"}:
            continue
        if row.get("bot_b") not in {"small", "v3", "full"}:
            continue
        aw = int(row.get("a_wins", 0))
        dr = int(row.get("draws", 0))
        bw = int(row.get("b_wins", 0))
        decisive = aw + bw
        if decisive:
            ap = 100.0 * aw / decisive
            bp = 100.0 * bw / decisive
            pct = f"{ap:.2f}% / {bp:.2f}%"
        else:
            pct = "N/A (0 decisive games)"
        print(
            f"{row['bot_a'].upper()} vs {row['bot_b'].upper()}: "
            f"{aw} W / {dr} D / {bw} L | decisive={decisive} | {pct}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reference arena for candidate-width recall: Small TOP12 vs V3 TOP16 "
            "vs exhaustive Full Pair C(E,2). No CNN or Cloudict stages."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/connect6_cnn_05/full_pair_reference"),
    )
    parser.add_argument(
        "--opening-size",
        type=int,
        default=19,
        help="19 = all 361 forced first-stone openings; 3 = center 3x3 quick test.",
    )
    parser.add_argument("--sync-interval", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Full Pair reference requires CUDA")

    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("# FULL PAIR REFERENCE")
    print(
        "SMALL: TOP12 -> C(12,2)=66 pairs | "
        "V3: TOP16 -> C(16,2)=120 pairs | "
        "FULL: every legal unordered pair C(E,2)."
    )
    print(
        "FULL uses the same pair-state evaluator as V3/Small; the experiment "
        "isolates candidate pruning only."
    )
    print("No model gauntlet and no Cloudict are run by this script.\n")

    rows = arena._run_bot_round_robin(
        SPECS,
        output_dir=args.output_dir,
        device=device,
        opening_size=args.opening_size,
        sync_interval=args.sync_interval,
    )
    _print_decisive(rows)
    print(f"\nCSV: {args.output_dir / 'bot_vs_bot.csv'}")


if __name__ == "__main__":
    main()
