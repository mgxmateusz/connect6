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
        """Return exactly one action per board."""
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


class _SearchGPUTacticalBot(_BaseGPUTacticalBot):
    """Benchmark/prototype search bot with one cached second action per board.

    A call with stones_left==2 performs the full search, returns stone #1 and
    stores stone #2 in a tiny int16 GPU tensor. The next call for stones_left==1
    returns that cached action and clears it. If no valid cache exists, the
    native kernel falls back to greedy V2.
    """

    def __init__(
        self,
        device: str | torch.device = "cuda",
        *,
        verbose_build: bool = False,
    ):
        super().__init__(device, verbose_build=verbose_build)
        self._pending_second: torch.Tensor | None = None

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
    """V3: V2 scorer, depth 2, beam [8], cached second stone."""

    entrypoint = "tactical_bot_v3_actions"
    label = "GPU Tactical Bot V3 D2 B8"


class GPUTacticalBotV4B8x2(_SearchGPUTacticalBot):
    """V4: V2 scorer, depth 3, beams [8, 2], one-ply opponent minimax."""

    entrypoint = "tactical_bot_v4_8x2_actions"
    label = "GPU Tactical Bot V4 D3 B8x2"


class GPUTacticalBotV4B4x4(_SearchGPUTacticalBot):
    """V4 alternate: V2 scorer, depth 3, beams [4, 4]."""

    entrypoint = "tactical_bot_v4_4x4_actions"
    label = "GPU Tactical Bot V4 D3 B4x4"


def create_gpu_tactical_bot(
    label: str,
    device: str | torch.device = "cuda",
    *,
    verbose_build: bool = False,
) -> _BaseGPUTacticalBot:
    # V3/V4 are intentionally not wired into training/UI yet; they are
    # benchmark prototypes until their speed/strength is measured.
    if label == GPU_TACTICAL_BOT:
        return GPUTacticalBot(device, verbose_build=verbose_build)
    if label == GPU_TACTICAL_BOT_V2:
        return GPUTacticalBotV2(device, verbose_build=verbose_build)
    raise ValueError(f"Unknown GPU tactical bot: {label}")
