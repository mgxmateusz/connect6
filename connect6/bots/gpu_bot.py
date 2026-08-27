from __future__ import annotations

import torch

from .cuda_native.bot_loader import load_native_bot_extension


GPU_TACTICAL_BOT = "GPU Tactical Bot V1"
GPU_TACTICAL_BOT_V2 = "GPU Tactical Bot V2"
GPU_TACTICAL_BOT_V3 = "GPU Tactical Bot V3 Top16 Pair-State"
GPU_TACTICAL_BOT_SMALL = "GPU Tactical Bot Small Top12 Pair-State"
GPU_TACTICAL_BOT_V4 = "GPU Tactical Bot V4 Top12 ReplyPair6"
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
        current_player = current_player.to(device=self.device, dtype=torch.int8, non_blocking=True)
        stones_left = stones_left.to(device=self.device, dtype=torch.int8, non_blocking=True)
        fn = getattr(self._ext(), self.entrypoint)
        return fn(boards.contiguous(), current_player.contiguous(), stones_left.contiguous())


class GPUTacticalBot(_BaseGPUTacticalBot):
    entrypoint = "tactical_bot_actions"
    label = GPU_TACTICAL_BOT


class GPUTacticalBotV2(_BaseGPUTacticalBot):
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
                (batch,), -1, dtype=torch.int16, device=self.device
            )
        return self._pending_second

    @torch.inference_mode()
    def actions(
        self,
        boards: torch.Tensor,
        current_player: torch.Tensor,
        stones_left: torch.Tensor,
    ) -> torch.Tensor:
        boards = boards.to(device=self.device, dtype=torch.int8, non_blocking=True).contiguous()
        current_player = current_player.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        ).contiguous()
        stones_left = stones_left.to(
            device=self.device, dtype=torch.int8, non_blocking=True
        ).contiguous()
        pending = self._pending(int(boards.shape[0]))
        fn = getattr(self._ext(), self.entrypoint)
        return fn(boards, current_player, stones_left, pending)


class GPUTacticalBotV3(_SearchGPUTacticalBot):
    """V3: TOP16 current cells -> all C(16,2)=120 pairs -> no reply search."""

    entrypoint = "tactical_bot_v3_actions"
    label = GPU_TACTICAL_BOT_V3


class GPUTacticalBotSmall(_SearchGPUTacticalBot):
    """Control bot: exactly V3-style search, but TOP12 -> C(12,2)=66 pairs."""

    entrypoint = "tactical_bot_small_actions"
    label = GPU_TACTICAL_BOT_SMALL


class GPUTacticalBotV4(_SearchGPUTacticalBot):
    """V4: TOP12 own search plus filtered full opponent turns.

    Own side: TOP12 -> 66 pairs -> TOP4 complete pair states.
    Opponent: V2 TOP6 cells -> all C(6,2)=15 full two-stone replies for each
    of our TOP4 finalists. Choose our pair by maximin.
    """

    entrypoint = "tactical_bot_v4_actions"
    label = GPU_TACTICAL_BOT_V4


# Compatibility name used by older arena modules.
GPUTacticalBotV4B8x2 = GPUTacticalBotV4


class GPUTacticalBotV5B8x4(GPUTacticalBotV3):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "GPU Tactical Bot V5 D3 B8x4 was removed. Use GPUTacticalBotV3, "
            "GPUTacticalBotSmall or GPUTacticalBotV4."
        )


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
    if label == GPU_TACTICAL_BOT_V3:
        return GPUTacticalBotV3(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_SMALL:
        return GPUTacticalBotSmall(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V4:
        return GPUTacticalBotV4(device, verbose_build=verbose_build)
    raise ValueError(f"Unknown GPU tactical bot: {label}")
