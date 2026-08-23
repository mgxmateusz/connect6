from __future__ import annotations

import torch

from .cuda_native.bot_loader import load_native_bot_extension


GPU_TACTICAL_BOT = "GPU Tactical Bot V1"
GPU_TACTICAL_BOT_V2 = "GPU Tactical Bot V2"
GPU_TACTICAL_BOTS = (GPU_TACTICAL_BOT, GPU_TACTICAL_BOT_V2)


class _BaseGPUTacticalBot:
    """Stateless one-action Connect6 bot running entirely on CUDA."""

    entrypoint = "tactical_bot_actions"
    label = GPU_TACTICAL_BOT

    def __init__(self, device: str | torch.device = "cuda", *, verbose_build: bool = False):
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"{self.label} requires a CUDA device")
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
        """Return exactly one action per board.

        Both V1 and V2 stay stateless and match the normal Connect6 engine:
        after the first stone the board is updated and the bot is called again
        for the second stone. There is no stored pair plan or search tree.
        """
        boards = boards.to(device=self.device, dtype=torch.int8, non_blocking=True)
        current_player = current_player.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        )
        stones_left = stones_left.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        )
        fn = getattr(self._ext(), self.entrypoint)
        return fn(
            boards.contiguous(),
            current_player.contiguous(),
            stones_left.contiguous(),
        )


class GPUTacticalBot(_BaseGPUTacticalBot):
    """V1: original one-pass threat/count scorer."""

    entrypoint = "tactical_bot_actions"
    label = GPU_TACTICAL_BOT


class GPUTacticalBotV2(_BaseGPUTacticalBot):
    """V2: richer one-pass scorer, still GPU-friendly and stateless.

    Compared with V1 it distinguishes unique finishing cells, true Connect6
    forks (three independent finishing cells when the opponent gets two blocks),
    multi-direction pressure and the shape/compactness of six-cell patterns.
    """

    entrypoint = "tactical_bot_v2_actions"
    label = GPU_TACTICAL_BOT_V2


def create_gpu_tactical_bot(
    label: str,
    device: str | torch.device = "cuda",
    *,
    verbose_build: bool = False,
) -> _BaseGPUTacticalBot:
    if label == GPU_TACTICAL_BOT:
        return GPUTacticalBot(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V2:
        return GPUTacticalBotV2(device, verbose_build=verbose_build)
    raise ValueError(f"Unknown GPU tactical bot: {label}")
