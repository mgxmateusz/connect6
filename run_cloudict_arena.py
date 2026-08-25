from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import connect6.evaluation.cloudict_arena as arena


RECOVERABLE_MARKERS = (
    "Cloudict zakonczyl sie",
    "Przekroczono limit czasu oczekiwania na Cloudicta",
)
MAX_RESTARTS = 20
BOARD_SIZE = 19

# Etap 2 areny: Cloudict depth 3, ale tylko 3x3 wokol centrum.
# Daje 9 wymuszonych startow x oba kolory modelu = 18 partii.
DEPTH3 = 3
DEPTH3_CENTER_ACTIONS = tuple(
    row * BOARD_SIZE + col
    for row in range(8, 11)
    for col in range(8, 11)
)
DEFAULT_DEPTH3_OUTPUT_DIR = Path(
    "runs/connect6_cnn_05/cloudict_d3_center3x3_update_00001340"
)

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
    # kolejne podejscie ma wznowic zapisane CSV, a nie kasowac postep.
    sys.argv[:] = [arg for arg in sys.argv if arg != "--reset"]


def _cli_value(flag: str, default: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Brak wartosci po {flag}")
    return sys.argv[index + 1]


def _depth3_schedule() -> list[tuple[int, bool]]:
    return [
        (opening_action, model_is_black)
        for opening_action in DEPTH3_CENTER_ACTIONS
        for model_is_black in (True, False)
    ]


def _depth3_output_dir() -> Path:
    # D2 zachowuje swoj dotychczasowy katalog i wyniki. D3 zawsze dostaje
    # osobny katalog, dzieki czemu nigdy nie nadpisuje 722 partii depth 2.
    return DEFAULT_DEPTH3_OUTPUT_DIR.resolve()


def _print_stage_summary(summary: dict[str, object], *, depth: int) -> None:
    overall = summary["overall"]
    black = summary["model_as_black"]
    white = summary["model_as_white"]
    print()
    print("=" * 78)
    print(f"WYNIK CLOUDICT DEPTH {depth}")
    print("=" * 78)
    print(
        f"RAZEM: model {overall['model_wins']} W / {overall['draws']} D / "
        f"{overall['cloudict_wins']} L | "
        f"winrate={100.0 * overall['model_win_rate']:.2f}% | "
        f"score={100.0 * overall['model_score_rate']:.2f}%"
    )
    print(
        f"MODEL BLACK: {black['model_wins']} W / {black['draws']} D / "
        f"{black['cloudict_wins']} L | "
        f"winrate={100.0 * black['model_win_rate']:.2f}%"
    )
    print(
        f"MODEL WHITE: {white['model_wins']} W / {white['draws']} D / "
        f"{white['cloudict_wins']} L | "
        f"winrate={100.0 * white['model_win_rate']:.2f}%"
    )


def _run_depth3_center_stage() -> None:
    checkpoint = Path(
        _cli_value("--checkpoint", str(arena.DEFAULT_CHECKPOINT))
    ).resolve()
    cloudict_exe = Path(
        _cli_value("--cloudict-exe", str(arena.DEFAULT_CLOUDICT_EXE))
    ).resolve()
    device = _cli_value("--device", "cuda")
    timeout = float(_cli_value("--timeout", "60.0"))
    # "i to samo" co D2: domyslnie VCF OFF. Jezeli ktos jawnie uruchomi
    # cala arene z --vcf, oba etapy dostaja VCF ON.
    vcf = "--vcf" in sys.argv
    reset = "--reset" in sys.argv

    if not checkpoint.is_file():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {checkpoint}")

    output_dir = _depth3_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    if reset:
        for path in (results_path, config_path, summary_path):
            path.unlink(missing_ok=True)

    schedule = _depth3_schedule()
    run_config = {
        "checkpoint": str(checkpoint),
        "cloudict_exe": str(cloudict_exe),
        "depth": DEPTH3,
        "vcf": bool(vcf),
        "device": str(device),
        "board_size": BOARD_SIZE,
        "win_length": arena.WIN_LENGTH,
        "forced_openings": len(DEPTH3_CENTER_ACTIONS),
        "games_per_opening": 2,
        "games_total": len(schedule),
        "model_policy": "argmax",
        "opening_mode": "center_3x3",
        "opening_actions": list(DEPTH3_CENTER_ACTIONS),
    }
    arena._validate_run_config(config_path, run_config)

    completed = arena._read_completed(results_path)
    remaining = [
        item
        for item in schedule
        if (item[0], "BLACK" if item[1] else "WHITE") not in completed
    ]

    coords = ", ".join(
        arena.action_to_cloudict(action) for action in DEPTH3_CENTER_ACTIONS
    )
    print()
    print("#" * 78)
    print("ETAP 2/2 — CLOUDICT DEPTH 3 — SRODKOWE 3x3")
    print("#" * 78)
    print(f"Checkpoint : {checkpoint}")
    print(f"Cloudict   : {cloudict_exe}")
    print(f"Depth      : {DEPTH3} | VCF={'ON' if vcf else 'OFF'}")
    print(
        f"Test       : 9 pol startowych x 2 kolory = {len(schedule)} partii "
        f"({len(remaining)} pozostalo)"
    )
    print(f"Starty     : {coords}")
    print("Model      : argmax, bez losowania")

    if not remaining:
        summary = arena._summarize(arena._load_rows(results_path))
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        _print_stage_summary(summary, depth=DEPTH3)
        print(f"CSV     : {results_path}")
        print(f"Summary : {summary_path}")
        return

    agent = arena.CheckpointAgent(checkpoint, device)
    cloudict = arena.CloudictEngine(
        cloudict_exe,
        depth=DEPTH3,
        vcf=vcf,
        timeout_seconds=timeout,
    )

    started_all = time.perf_counter()
    try:
        done_before = len(schedule) - len(remaining)
        for local_index, (opening_action, model_is_black) in enumerate(
            remaining,
            start=1,
        ):
            row = arena.play_one_game(
                agent,
                cloudict,
                opening_action=opening_action,
                model_is_black=model_is_black,
            )
            game_index = done_before + local_index
            row = {"game_index": game_index, **row}
            arena._append_result(results_path, row)

            rows_now = arena._load_rows(results_path)
            wins = sum(r["result"] == "WIN" for r in rows_now)
            draws = sum(r["result"] == "DRAW" for r in rows_now)
            losses = sum(r["result"] == "LOSS" for r in rows_now)
            print(
                f"[D3 {game_index:2d}/{len(schedule)}] "
                f"start={row['opening_coord']} model={row['model_color']:<5} "
                f"-> {row['result']:<4} | W/D/L={wins}/{draws}/{losses} | "
                f"stones={row['stones_played']} | "
                f"{float(row['elapsed_seconds']):.3f}s",
                flush=True,
            )
    finally:
        cloudict.close()

    rows = arena._load_rows(results_path)
    summary = arena._summarize(rows)
    summary["elapsed_seconds_this_run"] = time.perf_counter() - started_all
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    _print_stage_summary(summary, depth=DEPTH3)
    print(f"CSV     : {results_path}")
    print(f"Summary : {summary_path}")


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
            print()
            print("#" * 78)
            print("ETAP 1/2 — CLOUDICT DEPTH 2 — PELNE 19x19")
            print("#" * 78)
            # Dotychczasowy test D2 pozostaje 1:1 bez zmian. Jezeli 722 wyniki
            # juz istnieja, arena tylko je odczyta i od razu przejdzie do D3.
            arena.main()

            _run_depth3_center_stage()

            print()
            print("#" * 78)
            print("ARENA ZAKONCZONA — D2 + D3")
            print("#" * 78)
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
