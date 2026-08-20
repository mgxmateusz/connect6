from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .checkpoint import load_model_for_inference
from .game import BLACK, WHITE, Connect6Game
from .model import mask_logits


@torch.inference_mode()
def evaluate(
    checkpoint_a: str | Path,
    checkpoint_b: str | Path,
    games: int = 100,
    device: str = "cuda",
    temperature: float = 0.0,
) -> dict[str, float]:
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    model_a, payload_a = load_model_for_inference(checkpoint_a, dev)
    model_b, payload_b = load_model_for_inference(checkpoint_b, dev)

    size_a = int(payload_a["game_config"]["board_size"])
    size_b = int(payload_b["game_config"]["board_size"])
    win_a = int(payload_a["game_config"].get("win_length", 6))
    win_b = int(payload_b["game_config"].get("win_length", 6))
    if size_a != size_b or win_a != win_b:
        raise ValueError("Checkpointy używają niezgodnych ustawień gry")

    boards = [Connect6Game(size_a, win_a) for _ in range(games)]
    # W pierwszej połowie A gra czarnymi, w drugiej białymi, aby ograniczyć wpływ pierwszego gracza.
    a_is_black = np.arange(games) < ((games + 1) // 2)
    active = set(range(games))
    a_wins = b_wins = draws = 0

    while active:
        group_a: list[int] = []
        group_b: list[int] = []
        for i in active:
            g = boards[i]
            actor_is_a = (g.current_player == BLACK and a_is_black[i]) or (g.current_player == WHITE and not a_is_black[i])
            (group_a if actor_is_a else group_b).append(i)

        for indices, model in ((group_a, model_a), (group_b, model_b)):
            if not indices:
                continue
            inputs = [boards[i].network_input() for i in indices]
            network_input = torch.from_numpy(np.stack(inputs)).to(dev)
            legal = torch.from_numpy(np.stack([boards[i].legal_mask() for i in indices])).to(dev)
            logits, _ = model(network_input)
            logits = mask_logits(logits.float(), legal)
            if temperature <= 0:
                actions = logits.argmax(1)
            else:
                probs = torch.softmax(logits / max(temperature, 1e-4), dim=1)
                actions = torch.multinomial(probs, 1).squeeze(1)
            actions_cpu = actions.cpu().tolist()
            for i, action in zip(indices, actions_cpu):
                if i not in active:
                    continue
                result = boards[i].step(action)
                if result.done:
                    if result.winner == 0:
                        draws += 1
                    else:
                        winner_is_a = (result.winner == BLACK and a_is_black[i]) or (result.winner == WHITE and not a_is_black[i])
                        if winner_is_a:
                            a_wins += 1
                        else:
                            b_wins += 1
                    active.remove(i)

    return {
        "games": games,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "draws": draws,
        "a_win_rate": a_wins / games,
        "b_win_rate": b_wins / games,
        "draw_rate": draws / games,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Headless test checkpoint kontra checkpoint w Connect6")
    p.add_argument("checkpoint_a")
    p.add_argument("checkpoint_b")
    p.add_argument("--games", type=int, default=100)
    p.add_argument("--device", default="cuda")
    p.add_argument("--temperature", type=float, default=0.0)
    args = p.parse_args()
    result = evaluate(args.checkpoint_a, args.checkpoint_b, args.games, args.device, args.temperature)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
