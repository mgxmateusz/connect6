from __future__ import annotations

import torch

from .cuda_native.bot_loader import load_native_bot_extension


GPU_TACTICAL_BOT = "GPU Tactical Bot V1"
GPU_TACTICAL_BOT_V2 = "GPU Tactical Bot V2"
GPU_TACTICAL_BOT_V3 = "GPU Tactical Bot V3 Top16 Pair-State"
GPU_TACTICAL_BOT_V4 = "GPU Tactical Bot V4 Top8 Reply1"
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
    """V3: TOP16 current-position cells and exhaustive pair-state selection.

    One V2 pass ranks the 16 strongest legal cells. All C(16,2)=120 unordered
    pairs are evaluated with the 924-road state evaluator. No opponent reply is
    searched. This is the former V4 Top16 algorithm, promoted to V3.
    """

    entrypoint = "tactical_bot_v3_actions"
    label = GPU_TACTICAL_BOT_V3


class GPUTacticalBotV4(_SearchGPUTacticalBot):
    """V4: TOP8 pair search plus exhaustive one-stone opponent reply.

    One V2 pass keeps the 8 strongest current-position cells. All 28 unordered
    own pairs are state-evaluated; only the best four continue. For each of those
    four pairs every legal single opponent stone is tested, and V4 chooses the
    own pair with the best worst-case resulting board state.
    """

    entrypoint = "tactical_bot_v4_actions"
    label = GPU_TACTICAL_BOT_V4


# Compatibility name used by older arena modules.
GPUTacticalBotV4B8x2 = GPUTacticalBotV4


class GPUTacticalBotV5B8x4(GPUTacticalBotV3):
    """Import-compatibility shell for the removed old V5 experiment."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "GPU Tactical Bot V5 D3 B8x4 was removed. Use GPUTacticalBotV3 "
            "or GPUTacticalBotV4."
        )


def create_gpu_tactical_bot(
    label: str,
    device: str | torch.device = "cuda",
    *,
    verbose_build: bool = False,
) -> _BaseGPUTacticalBot:
    # V3/V4 keep per-board pending-second state and are deliberately not inserted
    # into the native training actor mix until rollout reset/state plumbing exists.
    if label == GPU_TACTICAL_BOT:
        return GPUTacticalBot(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V2:
        return GPUTacticalBotV2(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V3:
        return GPUTacticalBotV3(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V4:
        return GPUTacticalBotV4(device, verbose_build=verbose_build)
    raise ValueError(f"Unknown GPU tactical bot: {label}")
