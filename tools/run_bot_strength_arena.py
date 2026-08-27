from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# bot_arena predates the native-championship validator signature. Its lean
# HistoricalCheckpoint objects expose model/game config as attributes, while
# the native validator now expects a raw checkpoint dict plus a CheckpointRef.
# Keep the compatibility shim here so the strength arena can reuse the existing
# highly parallel all-autosaves gauntlet without changing its data format.
from connect6.championship import bot_arena as _model_arena
from connect6.championship import native_championship as _native


def _validate_lean_checkpoint_family(payload, ref=None, *, first_cfg=None):
    if not hasattr(payload, "model_config"):
        if ref is None:
            raise TypeError("raw checkpoint validation requires ref")
        return _native._validate_checkpoint_family(
            payload, ref, first_cfg=first_cfg
        )

    cfg = dict(payload.model_config)
    cfg.pop("compile", None)
    cfg.pop("compile_mode", None)
    name = getattr(getattr(payload, "path", None), "name", "<checkpoint>")

    if int(cfg.get("architecture_version", 0)) != _native.ARCHITECTURE_VERSION:
        raise RuntimeError(
            f"Checkpoint {name} ma architecture_version="
            f"{cfg.get('architecture_version')}; wymagano "
            f"{_native.ARCHITECTURE_VERSION}."
        )

    kernels = tuple(int(v) for v in cfg.get("kernels", _native.EXPECTED_KERNELS))
    channels = tuple(int(v) for v in cfg.get("channels", _native.EXPECTED_CHANNELS))
    if kernels != _native.EXPECTED_KERNELS or channels != _native.EXPECTED_CHANNELS:
        raise RuntimeError(
            f"Checkpoint {name} ma niezgodne kernels/channels: "
            f"{kernels}, {channels}."
        )

    game_cfg = payload.game_config
    if int(game_cfg.get("board_size", 0)) != 19 or int(
        game_cfg.get("win_length", 6)
    ) != 6:
        raise RuntimeError(f"Checkpoint {name} nie jest Connect6 19x19 / win_length=6")

    if first_cfg is not None and cfg != first_cfg:
        raise RuntimeError(f"Checkpoint {name} ma inną konfigurację modelu")
    return cfg


_model_arena._validate_checkpoint_family = _validate_lean_checkpoint_family

from connect6.evaluation.bot_strength_arena import main


if __name__ == "__main__":
    main()
