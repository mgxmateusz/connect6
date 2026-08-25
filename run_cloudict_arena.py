from __future__ import annotations

import csv
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
_PROTOCOL_SYMMETRY = 0

# Jeden wspolny katalog dla calej drabinki Cloudicta.
DEFAULT_LADDER_OUTPUT_DIR = Path(
    "runs/connect6_cnn_05/cloudict_ladder_update_00001340"
)

# depth -> szerokosc kwadratu startowego wokol centrum.
# D2 zachowuje pelne 19x19; kolejne poziomy dostaja coraz mniejszy wycinek.
STAGE_SIZES = {
    2: 19,
    3: 9,
    4: 5,
    5: 3,
}

# Stare katalogi sa tylko zrodlem migracji. Niczego z nich nie kasujemy.
LEGACY_RESULT_FILES = {
    2: Path("runs/connect6_cnn_05/cloudict_d2_update_00001340/results.csv"),
    3: Path("runs/connect6_cnn_05/cloudict_d3_center3x3_update_00001340/results.csv"),
    4: Path("runs/connect6_cnn_05/cloudict_d4_center3x3_update_00001340/results.csv"),
}


def _transform_action(action: int, symmetry: int) -> int:
    r, c = divmod(int(action), BOARD_SIZE)
    n = BOARD_SIZE - 1
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
    return rr * BOARD_SIZE + cc


def _inverse_symmetry(symmetry: int) -> int:
    return (0, 3, 2, 1, 4, 5, 6, 7)[int(symmetry) % 8]


_ORIGINAL_START_GAME = arena.CloudictEngine.start_game
_ORIGINAL_RESPOND = arena.CloudictEngine.respond
_ORIGINAL_APPLY_MOVE = arena._apply_cloudict_move
_ORIGINAL_PARSE_MOVE = arena.cloudict_move_to_actions
_ORIGINAL_ENCODE_MOVE = arena.actions_to_cloudict


def _transform_move(raw_move: str, symmetry: int) -> str:
    actions = _ORIGINAL_PARSE_MOVE(raw_move, BOARD_SIZE)
    transformed = [_transform_action(action, symmetry) for action in actions]
    return _ORIGINAL_ENCODE_MOVE(transformed, BOARD_SIZE)


def _start_game_with_symmetry(self, opening_coord: str, *, bot_is_black: bool):
    return _ORIGINAL_START_GAME(
        self,
        _transform_move(opening_coord, _PROTOCOL_SYMMETRY),
        bot_is_black=bot_is_black,
    )


def _respond_with_symmetry(self, opponent_move: str):
    return _ORIGINAL_RESPOND(
        self,
        _transform_move(opponent_move, _PROTOCOL_SYMMETRY),
    )


def _apply_cloudict_move_with_symmetry(game, raw_move: str) -> None:
    local_move = _transform_move(raw_move, _inverse_symmetry(_PROTOCOL_SYMMETRY))
    _ORIGINAL_APPLY_MOVE(game, local_move)


arena.CloudictEngine.start_game = _start_game_with_symmetry
arena.CloudictEngine.respond = _respond_with_symmetry
arena._apply_cloudict_move = _apply_cloudict_move_with_symmetry


def _is_recoverable(exc: BaseException) -> bool:
    return any(marker in str(exc) for marker in RECOVERABLE_MARKERS)


def _drop_reset_flag() -> None:
    sys.argv[:] = [arg for arg in sys.argv if arg != "--reset"]


def _cli_value(flag: str, default: str) -> str:
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return default
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Brak wartosci po {flag}")
    return sys.argv[index + 1]


def _center_actions(size: int) -> tuple[int, ...]:
    if size == BOARD_SIZE:
        return tuple(range(BOARD_SIZE * BOARD_SIZE))
    if size <= 0 or size > BOARD_SIZE or size % 2 == 0:
        raise ValueError(f"Nieprawidlowy rozmiar wycinka: {size}")
    radius = size // 2
    center = BOARD_SIZE // 2
    return tuple(
        row * BOARD_SIZE + col
        for row in range(center - radius, center + radius + 1)
        for col in range(center - radius, center + radius + 1)
    )


def _schedule(actions: tuple[int, ...]) -> list[tuple[int, bool]]:
    return [
        (opening_action, model_is_black)
        for opening_action in actions
        for model_is_black in (True, False)
    ]


def _stage_paths(output_dir: Path, depth: int) -> tuple[Path, Path, Path]:
    return (
        output_dir / f"d{depth}_results.csv",
        output_dir / f"d{depth}_config.json",
        output_dir / f"d{depth}_summary.json",
    )


def _migrate_legacy_results(
    *,
    depth: int,
    results_path: Path,
    allowed_actions: set[int],
) -> int:
    if results_path.exists() and results_path.stat().st_size > 0:
        return 0
    legacy = LEGACY_RESULT_FILES.get(depth)
    if legacy is None:
        return 0
    legacy = legacy.resolve()
    if not legacy.exists() or legacy.stat().st_size == 0:
        return 0

    rows = arena._load_rows(legacy)
    filtered = [
        row
        for row in rows
        if int(row.get("opening_action", -1)) in allowed_actions
        and row.get("model_color") in {"BLACK", "WHITE"}
    ]
    if not filtered:
        return 0

    # Usuwamy ewentualne duplikaty po (start, kolor), zachowujac pierwszy wynik.
    unique: dict[tuple[int, str], dict[str, str]] = {}
    for row in filtered:
        key = (int(row["opening_action"]), row["model_color"])
        unique.setdefault(key, row)

    with results_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=arena.RESULT_FIELDS)
        writer.writeheader()
        for index, row in enumerate(unique.values(), start=1):
            cleaned = {field: row.get(field, "") for field in arena.RESULT_FIELDS}
            cleaned["game_index"] = index
            writer.writerow(cleaned)
    return len(unique)


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


def _run_stage(depth: int, stage_no: int, total_stages: int) -> None:
    checkpoint = Path(
        _cli_value("--checkpoint", str(arena.DEFAULT_CHECKPOINT))
    ).resolve()
    cloudict_exe = Path(
        _cli_value("--cloudict-exe", str(arena.DEFAULT_CLOUDICT_EXE))
    ).resolve()
    device = _cli_value("--device", "cuda")
    timeout = float(_cli_value("--timeout", "60.0"))
    vcf = "--vcf" in sys.argv
    reset = "--reset" in sys.argv
    output_dir = Path(
        _cli_value("--output-dir", str(DEFAULT_LADDER_OUTPUT_DIR))
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    size = STAGE_SIZES[depth]
    actions = _center_actions(size)
    schedule = _schedule(actions)
    results_path, config_path, summary_path = _stage_paths(output_dir, depth)

    if reset:
        for path in (results_path, config_path, summary_path):
            path.unlink(missing_ok=True)

    migrated = _migrate_legacy_results(
        depth=depth,
        results_path=results_path,
        allowed_actions=set(actions),
    )

    run_config = {
        "checkpoint": str(checkpoint),
        "cloudict_exe": str(cloudict_exe),
        "depth": depth,
        "vcf": bool(vcf),
        "device": str(device),
        "board_size": BOARD_SIZE,
        "win_length": arena.WIN_LENGTH,
        "opening_square_size": size,
        "forced_openings": len(actions),
        "games_per_opening": 2,
        "games_total": len(schedule),
        "model_policy": "argmax",
        "opening_actions": list(actions),
    }
    arena._validate_run_config(config_path, run_config)

    completed = arena._read_completed(results_path)
    remaining = [
        item
        for item in schedule
        if (item[0], "BLACK" if item[1] else "WHITE") not in completed
    ]

    print()
    print("#" * 78)
    region = "PELNE 19x19" if size == 19 else f"SRODKOWE {size}x{size}"
    print(f"ETAP {stage_no}/{total_stages} — CLOUDICT DEPTH {depth} — {region}")
    print("#" * 78)
    print(f"Checkpoint : {checkpoint}")
    print(f"Cloudict   : {cloudict_exe}")
    print(f"Depth      : {depth} | VCF={'ON' if vcf else 'OFF'}")
    print(
        f"Test       : {len(actions)} pol startowych x 2 kolory = "
        f"{len(schedule)} partii ({len(remaining)} pozostalo)"
    )
    print(f"Wyniki     : {output_dir}")
    if migrated:
        print(f"Migracja   : odzyskano {migrated} starych partii D{depth}")
    print("Model      : argmax, bez losowania")

    if not remaining:
        summary = arena._summarize(arena._load_rows(results_path))
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        _print_stage_summary(summary, depth=depth)
        return

    agent = arena.CheckpointAgent(checkpoint, device)
    cloudict = arena.CloudictEngine(
        cloudict_exe,
        depth=depth,
        vcf=vcf,
        timeout_seconds=timeout,
    )
    started_all = time.perf_counter()
    try:
        done_before = len(schedule) - len(remaining)
        for local_index, (opening_action, model_is_black) in enumerate(
            remaining, start=1
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
                f"[D{depth} {game_index:3d}/{len(schedule)}] "
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
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _print_stage_summary(summary, depth=depth)


def _run_all_stages() -> None:
    depths = tuple(STAGE_SIZES)
    for stage_no, depth in enumerate(depths, start=1):
        _run_stage(depth, stage_no, len(depths))

    output_dir = Path(
        _cli_value("--output-dir", str(DEFAULT_LADDER_OUTPUT_DIR))
    ).resolve()
    print()
    print("#" * 78)
    print("ARENA ZAKONCZONA — D2 + D3 + D4 + D5")
    print("#" * 78)
    print(f"Wszystkie wyniki: {output_dir}")


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
            _run_all_stages()
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
