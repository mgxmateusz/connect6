from __future__ import annotations

import argparse
import threading
import tkinter as tk

import torch

from .gpu_bot import (
    GPU_TACTICAL_BOTS,
    create_gpu_tactical_bot,
)
from .gui import HUMAN, Connect6GUI
from .model import mask_logits
from .vector_env import canonical_network_input


class Connect6CNNGUI(Connect6GUI):
    def __init__(self, root: tk.Tk, runs_dir: str = "runs"):
        self._tactical_bots = {}
        self._bot_loading = False
        self._bot_load_error: BaseException | None = None
        self._bot_load_thread: threading.Thread | None = None
        super().__init__(root, runs_dir)

    def refresh_models(self) -> None:
        # Preserve bot selections because the base implementation knows only
        # Human + checkpoint entries and would otherwise reset them to Human.
        black_before = self.black_source.get()
        white_before = self.white_source.get()
        super().refresh_models()

        values = list(self.black_combo["values"])
        if self.device.type == "cuda":
            insert_at = 1
            for label in GPU_TACTICAL_BOTS:
                if label not in values:
                    values.insert(insert_at, label)
                    insert_at += 1
        self.black_combo["values"] = values
        self.white_combo["values"] = values

        if black_before in GPU_TACTICAL_BOTS and black_before in values:
            self.black_source.set(black_before)
        if white_before in GPU_TACTICAL_BOTS and white_before in values:
            self.white_source.set(white_before)

    def _bot_selected(self) -> bool:
        return any(
            source in GPU_TACTICAL_BOTS
            for source in (self.black_source.get(), self.white_source.get())
        )

    def _bots_ready(self) -> bool:
        return all(label in self._tactical_bots for label in GPU_TACTICAL_BOTS)

    def _start_tactical_bot_loading(self) -> None:
        if self.device.type != "cuda":
            self._bot_load_error = RuntimeError("GPU Tactical Bots require CUDA")
            return
        if self._bots_ready() or self._bot_loading:
            return

        self._bot_loading = True
        self._bot_load_error = None
        self.status.set(
            "Compiling/loading GPU Tactical Bot V1 + V2 in background... "
            "first run can take a minute; GUI remains usable."
        )

        def worker() -> None:
            try:
                # Both versions live in one native extension. Force the build
                # once, then create two lightweight wrappers over that cache.
                first = create_gpu_tactical_bot(
                    GPU_TACTICAL_BOTS[0], self.device, verbose_build=True
                )
                first._ext()
                self._tactical_bots[GPU_TACTICAL_BOTS[0]] = first
                self._tactical_bots[GPU_TACTICAL_BOTS[1]] = create_gpu_tactical_bot(
                    GPU_TACTICAL_BOTS[1], self.device
                )
            except BaseException as exc:  # propagate to Tk thread via polling
                self._bot_load_error = exc
            finally:
                self._bot_loading = False

        self._bot_load_thread = threading.Thread(
            target=worker,
            name="connect6-gpu-bot-loader",
            daemon=True,
        )
        self._bot_load_thread.start()
        self.root.after(100, self._poll_tactical_bot_loading)

    def _poll_tactical_bot_loading(self) -> None:
        if self._bot_loading:
            self.root.after(100, self._poll_tactical_bot_loading)
            return

        if self._bot_load_error is not None:
            exc = self._bot_load_error
            self._bot_load_error = None
            self.status.set("GPU Tactical Bot build failed - see error dialog/console")
            from tkinter import messagebox

            messagebox.showerror("GPU Tactical Bot build error", str(exc))
            return

        if self._bots_ready():
            self.status.set("GPU Tactical Bot V1 + V2 ready")
            self._schedule_ai_if_needed()

    def _get_tactical_bot(self, source: str):
        if self.device.type != "cuda":
            raise RuntimeError("GPU Tactical Bots require CUDA")
        bot = self._tactical_bots.get(source)
        if bot is None:
            raise RuntimeError(f"{source} is still compiling/loading")
        return bot

    @torch.inference_mode()
    def _bot_action(self, source: str) -> int:
        bot = self._get_tactical_bot(source)
        board = torch.from_numpy(self.game.board).to(
            device=self.device,
            dtype=torch.int8,
        ).unsqueeze(0)
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
        action = int(bot.actions(board, player, stones_left)[0].item())
        if action < 0:
            raise RuntimeError(f"{source} did not find a legal move")
        return action

    @torch.inference_mode()
    def _ai_action(self, source: str) -> int:
        if source in GPU_TACTICAL_BOTS:
            return self._bot_action(source)

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

    def _schedule_ai_if_needed(self) -> None:
        self._cancel_ai_job()
        if self.paused or self.game.done or self.current_source() == HUMAN:
            return

        if self.current_source() in GPU_TACTICAL_BOTS and not self._bots_ready():
            self._start_tactical_bot_loading()
            return

        self.ai_job = self.root.after(
            max(0, int(self.delay_ms.get())),
            self._do_ai_step,
        )

    def new_game(self) -> None:
        super().new_game()
        # If either bot is selected for either side, compile the shared native
        # extension immediately in the background.
        if self._bot_selected() and not self._bots_ready():
            self._start_tactical_bot_loading()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6 GUI: human/model/GPU Tactical Bot V1/V2 matches"
    )
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()
    root = tk.Tk()
    Connect6CNNGUI(root, args.runs_dir)
    root.mainloop()


if __name__ == "__main__":
    main()
