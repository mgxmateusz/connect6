from __future__ import annotations

import sys
import time

import connect6.evaluation.cloudict_arena as arena


RECOVERABLE_MARKERS = (
    "Cloudict zakonczyl sie",
    "Przekroczono limit czasu oczekiwania na Cloudicta",
)
MAX_RESTARTS = 20
BOARD_SIZE = 19

# Cloudict ma stary kod C++ i dla niektorych konkretnych orientacji pozycji
# potrafi wywalic proces. Po awarii ponawiamy te sama lokalna partie, ale
# pokazujemy Cloudictowi cala plansze obrocona/odbita. Dla Connect6 to ta sama
# geometrycznie pozycja; model nadal startuje z dokladnie tego samego pola.
_PROTOCOL_SYMMETRY = 0


def _transform_action(action: int, symmetry: int) -> int:
    r, c = divmod(int(action), BOARD_SIZE)
    n = BOARD_SIZE - 1
    s = int(symmetry) % 8
    if s == 0:      # identity
        rr, cc = r, c
    elif s == 1:    # rot90
        rr, cc = c, n - r
    elif s == 2:    # rot180
        rr, cc = n - r, n - c
    elif s == 3:    # rot270
        rr, cc = n - c, r
    elif s == 4:    # mirror left-right
        rr, cc = r, n - c
    elif s == 5:    # main diagonal
        rr, cc = c, r
    elif s == 6:    # mirror top-bottom
        rr, cc = n - r, c
    else:           # anti-diagonal
        rr, cc = n - c, n - r
    return rr * BOARD_SIZE + cc


def _inverse_symmetry(symmetry: int) -> int:
    # Rot90 <-> Rot270; pozostale transformacje sa samoodwrotne.
    return (0, 3, 2, 1, 4, 5, 6, 7)[int(symmetry) % 8]


def _transform_move(raw_move: str, symmetry: int) -> str:
    actions = _ORIGINAL_PARSE_MOVE(raw_move, BOARD_SIZE)
    transformed = [_transform_action(action, symmetry) for action in actions]
    return _ORIGINAL_ENCODE_MOVE(transformed, BOARD_SIZE)


_ORIGINAL_START_GAME = arena.CloudictEngine.start_game
_ORIGINAL_RESPOND = arena.CloudictEngine.respond
_ORIGINAL_APPLY_MOVE = arena._apply_cloudict_move
_ORIGINAL_PARSE_MOVE = arena.cloudict_move_to_actions
_ORIGINAL_ENCODE_MOVE = arena.actions_to_cloudict


def _start_game_with_symmetry(
    self: arena.CloudictEngine,
    opening_coord: str,
    *,
    bot_is_black: bool,
):
    engine_coord = _transform_move(opening_coord, _PROTOCOL_SYMMETRY)
    return _ORIGINAL_START_GAME(
        self,
        engine_coord,
        bot_is_black=bot_is_black,
    )


def _respond_with_symmetry(
    self: arena.CloudictEngine,
    opponent_move: str,
):
    engine_move = _transform_move(opponent_move, _PROTOCOL_SYMMETRY)
    return _ORIGINAL_RESPOND(self, engine_move)


def _apply_cloudict_move_with_symmetry(game, raw_move: str) -> None:
    local_move = _transform_move(raw_move, _inverse_symmetry(_PROTOCOL_SYMMETRY))
    _ORIGINAL_APPLY_MOVE(game, local_move)


# Patch tylko warstwy protokolu do zewnetrznego exe. Logika lokalnej gry,
# checkpoint i raportowane pole startowe pozostaja bez zmian.
arena.CloudictEngine.start_game = _start_game_with_symmetry
arena.CloudictEngine.respond = _respond_with_symmetry
arena._apply_cloudict_move = _apply_cloudict_move_with_symmetry


def _is_recoverable(exc: BaseException) -> bool:
    text = str(exc)
    return any(marker in text for marker in RECOVERABLE_MARKERS)


def _drop_reset_flag() -> None:
    # --reset ma obowiazywac tylko przy pierwszym uruchomieniu. Po awarii
    # kolejne podejscie ma wznowic zapisany results.csv, a nie kasowac postep.
    sys.argv[:] = [arg for arg in sys.argv if arg != "--reset"]


if __name__ == "__main__":
    restart_count = 0
    while True:
        _PROTOCOL_SYMMETRY = restart_count % 8
        if restart_count:
            print(
                f"[CLOUDICT] retry z symetria planszy #{_PROTOCOL_SYMMETRY} "
                "dla pierwszej niezapisanej partii.",
                flush=True,
            )
        try:
            arena.main()
            break
        except (RuntimeError, TimeoutError) as exc:
            if not _is_recoverable(exc) or restart_count >= MAX_RESTARTS:
                raise
            restart_count += 1
            _drop_reset_flag()
            print(
                f"\n[CLOUDICT] proces padl/przekroczyl timeout: {exc}\n"
                f"[CLOUDICT] restart {restart_count}/{MAX_RESTARTS}; "
                "wznawiam te sama niezapisana partie w innej orientacji...\n",
                flush=True,
            )
            time.sleep(0.25)
