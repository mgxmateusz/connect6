"""Hand-written Connect6 opponents."""

# The tactical bot predates the package split and imports `.cuda_native`.
# Keep that dependency canonical without duplicating CUDA modules.
import sys

import connect6.cuda_native as _cuda_native
from connect6.cuda_native import bot_loader as _bot_loader

sys.modules.setdefault(__name__ + ".cuda_native", _cuda_native)
sys.modules.setdefault(__name__ + ".cuda_native.bot_loader", _bot_loader)

from .gpu_bot import (  # noqa: E402
    GPU_TACTICAL_BOT,
    GPU_TACTICAL_BOT_V2,
    GPU_TACTICAL_BOT_V2_PRO,
    GPU_TACTICAL_BOT_V3,
    GPU_TACTICAL_BOT_SMALL,
    GPU_TACTICAL_BOT_V4,
    GPU_TACTICAL_BOT_FULL_PAIR,
    GPU_TACTICAL_BOTS,
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV2Pro,
    GPUTacticalBotV3,
    GPUTacticalBotSmall,
    GPUTacticalBotV4,
    GPUTacticalBotFullPair,
    create_gpu_tactical_bot,
)

__all__ = [
    "GPU_TACTICAL_BOT",
    "GPU_TACTICAL_BOT_V2",
    "GPU_TACTICAL_BOT_V2_PRO",
    "GPU_TACTICAL_BOT_V3",
    "GPU_TACTICAL_BOT_SMALL",
    "GPU_TACTICAL_BOT_V4",
    "GPU_TACTICAL_BOT_FULL_PAIR",
    "GPU_TACTICAL_BOTS",
    "GPUTacticalBot",
    "GPUTacticalBotV2",
    "GPUTacticalBotV2Pro",
    "GPUTacticalBotV3",
    "GPUTacticalBotSmall",
    "GPUTacticalBotV4",
    "GPUTacticalBotFullPair",
    "create_gpu_tactical_bot",
]
