from __future__ import annotations

import torch

from .cuda_native.bot_loader import load_native_bot_extension


GPU_TACTICAL_BOT = "GPU Tactical Bot V1"
GPU_TACTICAL_BOT_V2 = "GPU Tactical Bot V2"
GPU_TACTICAL_BOT_V2_PRO = "GPU Tactical Bot V2 Pro LatentFork"
GPU_TACTICAL_BOT_V3_PRO = "GPU Tactical Bot V3 Pro Top16 Pair-State"
GPU_TACTICAL_BOT_V4_PRO = "GPU Tactical Bot V4 Pro Top12 ReplyPair6"
GPU_TACTICAL_BOT_FULL_PAIR = "GPU Tactical Bot Full Pair Brute Force"
GPU_TACTICAL_BOT_PAIRFIRST = "GPU Tactical Bot PairFirst AllPairs P128"
GPU_TACTICAL_BOT_PAIRFIRST32 = "GPU Tactical Bot PairFirst AllPairs P32"
GPU_TACTICAL_BOT_LIVEROAD = "GPU Tactical Bot LiveRoad Brute Force"
GPU_TACTICAL_BOT_HYBRID = "GPU Tactical Bot Hybrid LiveRoad Pair128"
GPU_TACTICAL_BOT_HYBRID32 = "GPU Tactical Bot Hybrid LiveRoad Pair32"

# Training/native rollout still only knows the original stateless V1/V2 actors.
# V2Pro and search bots stay benchmark/arena/GUI-only until training integration is intentional.
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


class GPUTacticalBotV2Pro(_BaseGPUTacticalBot):
    """V2 architecture with latent 2->3/3->4 fork-anchor road leverage."""

    entrypoint = "tactical_bot_v2_pro_actions"
    label = GPU_TACTICAL_BOT_V2_PRO


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


class GPUTacticalBotV3Pro(_SearchGPUTacticalBot):
    """V3 search with V2Pro TOP16 ordering."""

    entrypoint = "tactical_bot_v3_pro_actions"
    label = GPU_TACTICAL_BOT_V3_PRO


class GPUTacticalBotV4Pro(_SearchGPUTacticalBot):
    """V4 maximin search with V2Pro own TOP12 and opponent TOP6 ordering."""

    entrypoint = "tactical_bot_v4_pro_actions"
    label = GPU_TACTICAL_BOT_V4_PRO


class GPUTacticalBotFullPair(_SearchGPUTacticalBot):
    entrypoint = "tactical_bot_full_pair_actions"
    label = GPU_TACTICAL_BOT_FULL_PAIR


class GPUTacticalBotPairFirst(_SearchGPUTacticalBot):
    entrypoint = "tactical_bot_pairfirst_actions"
    label = GPU_TACTICAL_BOT_PAIRFIRST


class GPUTacticalBotPairFirst32(_SearchGPUTacticalBot):
    entrypoint = "tactical_bot_pairfirst32_actions"
    label = GPU_TACTICAL_BOT_PAIRFIRST32


class GPUTacticalBotLiveRoad(_SearchGPUTacticalBot):
    entrypoint = "tactical_bot_liveroad_actions"
    label = GPU_TACTICAL_BOT_LIVEROAD


class GPUTacticalBotHybrid(_SearchGPUTacticalBot):
    entrypoint = "tactical_bot_hybrid_actions"
    label = GPU_TACTICAL_BOT_HYBRID


class GPUTacticalBotHybrid32(_SearchGPUTacticalBot):
    entrypoint = "tactical_bot_hybrid32_actions"
    label = GPU_TACTICAL_BOT_HYBRID32


# Compatibility names for old callers. They now resolve to the validated Pro variants.
GPUTacticalBotV4B8x2 = GPUTacticalBotV4Pro


class GPUTacticalBotV5B8x4(GPUTacticalBotV3Pro):
    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "GPU Tactical Bot V5 D3 B8x4 was removed. Use a current search bot."
        )


def create_gpu_tactical_bot(
    label: str,
    device: str | torch.device = "cuda",
    *,
    verbose_build: bool = False,
) -> _BaseGPUTacticalBot:
    by_label = {
        GPU_TACTICAL_BOT: GPUTacticalBot,
        GPU_TACTICAL_BOT_V2: GPUTacticalBotV2,
        GPU_TACTICAL_BOT_V2_PRO: GPUTacticalBotV2Pro,
        GPU_TACTICAL_BOT_V3_PRO: GPUTacticalBotV3Pro,
        GPU_TACTICAL_BOT_V4_PRO: GPUTacticalBotV4Pro,
        GPU_TACTICAL_BOT_FULL_PAIR: GPUTacticalBotFullPair,
        GPU_TACTICAL_BOT_PAIRFIRST: GPUTacticalBotPairFirst,
        GPU_TACTICAL_BOT_PAIRFIRST32: GPUTacticalBotPairFirst32,
        GPU_TACTICAL_BOT_LIVEROAD: GPUTacticalBotLiveRoad,
        GPU_TACTICAL_BOT_HYBRID: GPUTacticalBotHybrid,
        GPU_TACTICAL_BOT_HYBRID32: GPUTacticalBotHybrid32,
    }
    cls = by_label.get(label)
    if cls is None:
        raise ValueError(f"Unknown GPU tactical bot: {label}")
    return cls(device, verbose_build=verbose_build)
