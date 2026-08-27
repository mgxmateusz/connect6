from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect6.bots.gpu_bot import (
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV3,
    GPUTacticalBotV4,
)
from connect6.championship import bot_arena as _model_arena
from connect6.evaluation import bot_strength_arena as _base
from connect6.evaluation import cloudict_arena as _cloudict


# Register the current experimental ladder explicitly. Signatures differ from
# the deleted historical V3/V4/V5 implementations, so stale state can never be
# resumed as if it belonged to the new algorithms.
_base.BOT_SPECS = (
    _model_arena.BotSpec(
        "v1",
        "GPU Tactical Bot V1",
        "gpu_tactical_bot_heuristic_v1",
        GPUTacticalBot,
    ),
    _model_arena.BotSpec(
        "v2",
        "GPU Tactical Bot V2",
        "gpu_tactical_bot_heuristic_v2",
        GPUTacticalBotV2,
    ),
    _model_arena.BotSpec(
        "v3",
        "GPU Tactical Bot V3 Pair-State Full-Turn",
        "gpu_tactical_bot_v3_pair_state_full_turn_v1",
        GPUTacticalBotV3,
    ),
    _model_arena.BotSpec(
        "v4",
        "GPU Tactical Bot V4 Top16 Pair-State",
        "gpu_tactical_bot_v4_top16_pair_state_v1",
        GPUTacticalBotV4,
    ),
)
_base.BOT_BY_KEY = {spec.key: spec for spec in _base.BOT_SPECS}


# Cloudict has a known failure mode on some board orientations/openings: its
# stdout closes and the process dies before returning a move. The standalone
# Cloudict ladder already works around this by retrying the same logical game in
# another symmetry. Do the same for the strength arena without changing the
# locally observed board or the recorded opening.
def _transform_action(action: int, symmetry: int) -> int:
    r, c = divmod(int(action), 19)
    n = 18
    s = int(symmetry) % 8
    if s == 0:
        rr, cc = r, c
    elif s == 1:
        rr, cc = c, n - r
    elif s == 2:
        rr, cc = n - r, n - c
    elif s == 3:
        rr, cc = n - c, r
    elif s == 4:
        rr, cc = r, n - c
    elif s == 5:
        rr, cc = c, r
    elif s == 6:
        rr, cc = n - r, c
    else:
        rr, cc = n - c, n - r
    return rr * 19 + cc


def _inverse_symmetry(symmetry: int) -> int:
    return (0, 3, 2, 1, 4, 5, 6, 7)[int(symmetry) % 8]


def _transform_move(raw_move: str, symmetry: int) -> str:
    actions = _cloudict.cloudict_move_to_actions(raw_move, 19)
    transformed = [_transform_action(action, symmetry) for action in actions]
    return _cloudict.actions_to_cloudict(transformed, 19)


_OriginalCloudictEngine = _cloudict.CloudictEngine


class _SymmetricRetryCloudictEngine(_OriginalCloudictEngine):
    _engine_serial = 0

    def __init__(self, *args, **kwargs):
        self._protocol_symmetry = type(self)._engine_serial % 8
        type(self)._engine_serial += 1
        super().__init__(*args, **kwargs)
        print(
            f"[CLOUDICT] engine symmetry #{self._protocol_symmetry}",
            flush=True,
        )

    def _reply_to_local(self, reply):
        if reply is None:
            return None
        return _cloudict.SearchReply(
            move=_transform_move(
                reply.move,
                _inverse_symmetry(self._protocol_symmetry),
            ),
            elapsed_seconds=reply.elapsed_seconds,
        )

    def start_game(self, opening_coord: str, *, bot_is_black: bool):
        reply = super().start_game(
            _transform_move(opening_coord, self._protocol_symmetry),
            bot_is_black=bot_is_black,
        )
        return self._reply_to_local(reply)

    def respond(self, opponent_move: str):
        reply = super().respond(
            _transform_move(opponent_move, self._protocol_symmetry)
        )
        return self._reply_to_local(reply)


# _run_cloudict instantiates base.cloudict.CloudictEngine on every restart.
# Replacing it here means retry #1/#2/... automatically rotates the protocol
# while all game bookkeeping remains in the original local orientation.
_base.cloudict.CloudictEngine = _SymmetricRetryCloudictEngine

from connect6.evaluation.bot_strength_arena_fast import main


if __name__ == "__main__":
    main()
