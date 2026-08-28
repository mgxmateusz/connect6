"""Hand-written Connect6 opponents."""

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
    GPU_TACTICAL_BOT_V3_PRO,
    GPU_TACTICAL_BOT_SMALL,
    GPU_TACTICAL_BOT_V4,
    GPU_TACTICAL_BOT_V4_PRO,
    GPU_TACTICAL_BOT_FULL_PAIR,
    GPU_TACTICAL_BOT_FULL_PAIR_PRO,
    GPU_TACTICAL_BOT_PAIRFIRST,
    GPU_TACTICAL_BOT_PAIRFIRST_PRO,
    GPU_TACTICAL_BOT_PAIRFIRST32,
    GPU_TACTICAL_BOT_PAIRFIRST32_PRO,
    GPU_TACTICAL_BOT_LIVEROAD,
    GPU_TACTICAL_BOT_LIVEROAD_PRO,
    GPU_TACTICAL_BOT_HYBRID,
    GPU_TACTICAL_BOT_HYBRID_PRO,
    GPU_TACTICAL_BOT_HYBRID32,
    GPU_TACTICAL_BOT_HYBRID32_PRO,
    GPU_TACTICAL_BOTS,
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV2Pro,
    GPUTacticalBotV3,
    GPUTacticalBotV3Pro,
    GPUTacticalBotSmall,
    GPUTacticalBotV4,
    GPUTacticalBotV4Pro,
    GPUTacticalBotFullPair,
    GPUTacticalBotFullPairPro,
    GPUTacticalBotPairFirst,
    GPUTacticalBotPairFirstPro,
    GPUTacticalBotPairFirst32,
    GPUTacticalBotPairFirst32Pro,
    GPUTacticalBotLiveRoad,
    GPUTacticalBotLiveRoadPro,
    GPUTacticalBotHybrid,
    GPUTacticalBotHybridPro,
    GPUTacticalBotHybrid32,
    GPUTacticalBotHybrid32Pro,
    create_gpu_tactical_bot,
)

__all__ = [name for name in globals() if name.startswith("GPU_TACTICAL_") or name.startswith("GPUTacticalBot")]
__all__.append("create_gpu_tactical_bot")
