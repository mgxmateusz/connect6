from __future__ import annotations

import sys
import time

from connect6.evaluation.cloudict_arena import main


RECOVERABLE_MARKERS = (
    "Cloudict zakonczyl sie",
    "Przekroczono limit czasu oczekiwania na Cloudicta",
)
MAX_RESTARTS = 20


def _is_recoverable(exc: BaseException) -> bool:
    text = str(exc)
    return any(marker in text for marker in RECOVERABLE_MARKERS)


def _drop_reset_flag() -> None:
    # --reset ma obowiązywać tylko przy pierwszym uruchomieniu. Po awarii
    # kolejne podejście ma wznowić zapisany results.csv, a nie kasować postęp.
    sys.argv[:] = [arg for arg in sys.argv if arg != "--reset"]


if __name__ == "__main__":
    restart_count = 0
    while True:
        try:
            main()
            break
        except (RuntimeError, TimeoutError) as exc:
            if not _is_recoverable(exc) or restart_count >= MAX_RESTARTS:
                raise
            restart_count += 1
            _drop_reset_flag()
            print(
                f"\n[CLOUDICT] proces padl/przekroczyl timeout: {exc}\n"
                f"[CLOUDICT] restart {restart_count}/{MAX_RESTARTS}; "
                "wznawiam od pierwszej niezapisanej partii...\n",
                flush=True,
            )
            time.sleep(0.25)
