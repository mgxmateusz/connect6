from __future__ import annotations

import argparse
import tkinter as tk

import torch

from .gui import Connect6GUI
from .model import mask_logits
from .vector_env import canonical_network_input


class Connect6CNNGUI(Connect6GUI):
    @torch.inference_mode()
    def _ai_action(self, source: str) -> int:
        model, _ = self._get_model(source)

        board = torch.from_numpy(self.game.board).to(self.device).unsqueeze(0)
        player = torch.tensor(
            [int(self.game.current_player)],
            dtype=torch.int8,
            device=self.device,
        )
        stones_left = torch.tensor(
            [int(self.game.stones_left_in_turn)],
            dtype=torch.int8,
            device=self.device,
        )

        network_input = canonical_network_input(
            board,
            player,
            stones_left,
        )

        legal = (
            torch.from_numpy(self.game.legal_mask())
            .unsqueeze(0)
            .to(self.device)
        )

        logits, _ = model(network_input)
        logits = mask_logits(logits.float(), legal)

        temp = float(self.temperature.get())
        if temp <= 0:
            return int(logits.argmax(dim=1).item())

        probs = torch.softmax(logits / max(temp, 1e-4), dim=1)
        return int(torch.multinomial(probs, 1).item())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6 GUI: human/model/model-vs-model"
    )
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    root = tk.Tk()
    Connect6CNNGUI(root, args.runs_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
