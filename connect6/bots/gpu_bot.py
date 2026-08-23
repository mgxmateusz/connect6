from __future__ import annotations

import torch

from .cuda_native.bot_loader import load_native_bot_extension


GPU_TACTICAL_BOT = "GPU Tactical Bot"


class GPUTacticalBot:
    """One-pass deterministic Connect6 threat bot running entirely on CUDA.

    The native kernel scores every empty field independently. It has no search
    tree and no neural network: it detects immediate wins/blocks, one-turn
    two-stone wins, length-6 viable windows, broken patterns, multi-threats,
    contiguous lines and a tiny locality/centre preference.
    """

    def __init__(self, device: str | torch.device = "cuda", *, verbose_build: bool = False):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("GPUTacticalBot requires a CUDA device")
        self.verbose_build = bool(verbose_build)
        self._extension = None

    def _ext(self):
        if self._extension is None:
            with torch.cuda.device(self.device):
                self._extension = load_native_bot_extension(verbose=self.verbose_build)
        return self._extension

    @torch.inference_mode()
    def actions(
        self,
        boards: torch.Tensor,
        current_player: torch.Tensor,
        stones_left: torch.Tensor,
    ) -> torch.Tensor:
        """Return one action per board.

        Args:
            boards: int8 CUDA tensor [B, 19, 19], values {-1, 0, +1}.
            current_player: int8 CUDA tensor [B], values {-1, +1}.
            stones_left: int8 CUDA tensor [B], normally 1 or 2.
        """
        boards = boards.to(device=self.device, dtype=torch.int8, non_blocking=True)
        current_player = current_player.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        )
        stones_left = stones_left.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        )
        return self._ext().tactical_bot_actions(
            boards.contiguous(),
            current_player.contiguous(),
            stones_left.contiguous(),
        )
