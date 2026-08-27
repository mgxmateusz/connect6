from __future__ import annotations

import torch

from .cuda_native.bot_loader import load_native_bot_extension


GPU_TACTICAL_BOT = "GPU Tactical Bot V1"
GPU_TACTICAL_BOT_V2 = "GPU Tactical Bot V2"
GPU_TACTICAL_BOT_V3 = "GPU Tactical Bot V3 Pair-State"
# Training/native rollout still only knows the stateless V1/V2 actors.
GPU_TACTICAL_BOTS = (GPU_TACTICAL_BOT, GPU_TACTICAL_BOT_V2)


class _BaseGPUTacticalBot:
    """One-action Connect6 bot running entirely on CUDA."""

    entrypoint = "tactical_bot_actions"
    label = GPU_TACTICAL_BOT

    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        verbose_build: bool = False,
    ):
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
    """V2: richer one-pass Connect6-aware move scorer."""

    entrypoint = "tactical_bot_v2_actions"
    label = GPU_TACTICAL_BOT_V2


class _SearchGPUTacticalBot(_BaseGPUTacticalBot):
    """Two-stone planner with the second action cached entirely on the GPU."""

    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        verbose_build: bool = False,
    ):
        super().__init__(device, verbose_build=verbose_build)
        self._pending_second: torch.Tensor | None = None

    @torch.inference_mode()
    def reset(self) -> None:
        if self._pending_second is not None:
            self._pending_second.fill_(-1)

    def _pending(self, batch: int) -> torch.Tensor:
        if (
            self._pending_second is None
            or self._pending_second.numel() != batch
            or self._pending_second.device != self.device
        ):
            self._pending_second = torch.full(
                (batch,),
                -1,
                dtype=torch.int16,
                device=self.device,
            )
        return self._pending_second

    @torch.inference_mode()
    def actions(
        self,
        boards: torch.Tensor,
        current_player: torch.Tensor,
        stones_left: torch.Tensor,
    ) -> torch.Tensor:
        boards = boards.to(device=self.device, dtype=torch.int8, non_blocking=True)
        current_player = current_player.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        )
        stones_left = stones_left.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        )
        boards = boards.contiguous()
        current_player = current_player.contiguous()
        stones_left = stones_left.contiguous()
        pending = self._pending(int(boards.shape[0]))
        fn = getattr(self._ext(), self.entrypoint)
        return fn(boards, current_player, stones_left, pending)


class GPUTacticalBotV3(_SearchGPUTacticalBot):
    """V3: pair-state search with full-turn opponent replies.

    V2 is used only for move ordering. Up to 32 unique own pairs are evaluated
    by a separate 924-road board evaluator with exact 0/1/2/3+ threat pressure.
    Only the best four pairs receive a full opponent 4x2 pair search, keeping
    worst-case V2-score work below the removed V5 implementation.
    """

    entrypoint = "tactical_bot_v3_actions"
    label = GPU_TACTICAL_BOT_V3


class _RemovedSearchBot(GPUTacticalBotV3):
    """Import-compatibility shell for deleted experimental versions."""

    removed_name = "old search bot"

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            f"{self.removed_name} was removed. Use GPUTacticalBotV3 "
            "(pair-state/full-turn GPU search)."
        )


class GPUTacticalBotV4B8x2(_RemovedSearchBot):
    removed_name = "GPU Tactical Bot V4 D3 B8x2"


class GPUTacticalBotV5B8x4(_RemovedSearchBot):
    removed_name = "GPU Tactical Bot V5 D3 B8x4"


def create_gpu_tactical_bot(
    label: str,
    device: str | torch.device = "cuda",
    *,
    verbose_build: bool = False,
) -> _BaseGPUTacticalBot:
    # V3 keeps per-board pending-second state and is deliberately not inserted
    # into the native training actor mix until rollout reset/state plumbing is
    # added for it.
    if label == GPU_TACTICAL_BOT:
        return GPUTacticalBot(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V2:
        return GPUTacticalBotV2(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V3:
        return GPUTacticalBotV3(device, verbose_build=verbose_build)
    raise ValueError(f"Unknown GPU tactical bot: {label}")
