from __future__ import annotations

import argparse
import csv
import json
import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from connect6.engine.checkpoint import load_model_for_inference
from connect6.engine.game import BLACK, EMPTY, WHITE, Connect6Game
from connect6.engine.model import mask_logits
from connect6.engine.vector_env import canonical_network_input


BOARD_SIZE = 19
WIN_LENGTH = 6
DEFAULT_CHECKPOINT = Path("runs/connect6_cnn_05/checkpoints/model_update_00001340.pt")
DEFAULT_CLOUDICT_EXE = Path(r"C:\PRACA\ConnectMore\engines\cloudict.exe")
DEFAULT_OUTPUT_DIR = Path("runs/connect6_cnn_05/cloudict_d2_update_00001340")
RESULT_FIELDS = [
    "game_index",
    "opening_action",
    "opening_row",
    "opening_col",
    "opening_coord",
    "model_color",
    "winner",
    "result",
    "stones_played",
    "model_decisions",
    "cloudict_decisions",
    "model_seconds",
    "cloudict_seconds",
    "elapsed_seconds",
]


def action_to_cloudict(action: int, board_size: int = BOARD_SIZE) -> str:
    row, col = divmod(int(action), int(board_size))
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Akcja {action} jest poza plansza {board_size}x{board_size}")
    return chr(ord("A") + row) + chr(ord("A") + col)


def cloudict_move_to_actions(raw: str, board_size: int = BOARD_SIZE) -> list[int]:
    move = raw.strip().upper()
    if len(move) not in (2, 4):
        raise ValueError(f"Nieprawidlowy ruch Cloudicta: {raw!r}")
    actions: list[int] = []
    for i in range(0, len(move), 2):
        row = ord(move[i]) - ord("A")
        col = ord(move[i + 1]) - ord("A")
        if not (0 <= row < board_size and 0 <= col < board_size):
            raise ValueError(f"Cloudict zwrocil pole poza plansza: {raw!r}")
        actions.append(row * board_size + col)
    return actions


def actions_to_cloudict(actions: Iterable[int], board_size: int = BOARD_SIZE) -> str:
    return "".join(action_to_cloudict(action, board_size) for action in actions)


def build_match_schedule(board_size: int = BOARD_SIZE) -> list[tuple[int, bool]]:
    """Dwie partie z kazdego pola: model czarny i model bialy."""
    return [
        (opening_action, model_is_black)
        for opening_action in range(board_size * board_size)
        for model_is_black in (True, False)
    ]


@dataclass(slots=True)
class SearchReply:
    move: str
    elapsed_seconds: float


class CloudictEngine:
    def __init__(
        self,
        executable: Path,
        *,
        depth: int,
        vcf: bool,
        timeout_seconds: float,
    ) -> None:
        self.executable = executable.resolve()
        self.depth = int(depth)
        self.vcf = bool(vcf)
        self.timeout_seconds = float(timeout_seconds)
        if not self.executable.is_file():
            raise FileNotFoundError(f"Nie znaleziono Cloudicta: {self.executable}")
        patterns = self.executable.parent / "patterns.in"
        if not patterns.is_file():
            raise FileNotFoundError(
                f"Brak patterns.in obok Cloudicta: {patterns}. "
                "Uruchamiaj exe z katalogu engines ConnectMore albo skopiuj tam patterns.in."
            )

        startupinfo = None
        if hasattr(subprocess, "STARTUPINFO"):
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        self.proc = subprocess.Popen(
            [str(self.executable)],
            cwd=str(self.executable.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            startupinfo=startupinfo,
        )
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("Nie udalo sie otworzyc stdin/stdout Cloudicta")

        self._lines: queue.Queue[str | None] = queue.Queue()
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        # Cloudict nie flushuje calej pomocy startowej. `name` wymusza fflush.
        self.send("name")
        self._wait_for_prefix("name ")
        self.send(f"depth {self.depth}")
        self._wait_for_contains("Set the search depth")
        self.send("vcf" if self.vcf else "unvcf")

    def _reader_loop(self) -> None:
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                self._lines.put(line.rstrip("\r\n"))
        finally:
            self._lines.put(None)

    def send(self, command: str) -> None:
        if self.proc.poll() is not None:
            raise RuntimeError(f"Cloudict zakonczyl sie z kodem {self.proc.returncode}")
        assert self.proc.stdin is not None
        self.proc.stdin.write(command.rstrip("\r\n") + "\n")
        self.proc.stdin.flush()

    def _next_line(self, deadline: float) -> str:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("Przekroczono limit czasu oczekiwania na Cloudicta")
        try:
            line = self._lines.get(timeout=remaining)
        except queue.Empty as exc:
            raise TimeoutError("Przekroczono limit czasu oczekiwania na Cloudicta") from exc
        if line is None:
            raise RuntimeError(f"Cloudict zakonczyl sie z kodem {self.proc.poll()}")
        return line

    def _wait_for_prefix(self, prefix: str) -> str:
        deadline = time.perf_counter() + self.timeout_seconds
        while True:
            line = self._next_line(deadline)
            if line.strip().startswith(prefix):
                return line.strip()

    def _wait_for_contains(self, text: str) -> str:
        deadline = time.perf_counter() + self.timeout_seconds
        while True:
            line = self._next_line(deadline)
            if text in line:
                return line.strip()

    def _search(self, command: str) -> SearchReply:
        started = time.perf_counter()
        self.send(command)
        deadline = started + self.timeout_seconds
        tail: list[str] = []
        while True:
            line = self._next_line(deadline)
            stripped = line.strip()
            if stripped:
                tail.append(stripped)
                if len(tail) > 8:
                    tail.pop(0)
            if stripped.startswith("move "):
                return SearchReply(
                    move=stripped[5:].strip().upper(),
                    elapsed_seconds=time.perf_counter() - started,
                )

    def start_game(self, opening_coord: str, *, bot_is_black: bool) -> SearchReply | None:
        # `new white` resetuje plansze bez automatycznego ruchu JJ.
        self.send("new white")
        if bot_is_black:
            # Wymuszony pierwszy kamien nalezy do Cloudicta, ale nie pozwalamy mu
            # wybrac centrum. `black XX` ustawia jego kolor na BLACK bez searcha.
            self.send(f"black {opening_coord}")
            return None
        # Cloudict jest bialy: podajemy wymuszony pierwszy ruch czarnego i od razu
        # prosimy o odpowiedz bialych.
        return self._search(f"move {opening_coord}")

    def respond(self, opponent_move: str) -> SearchReply:
        return self._search(f"move {opponent_move}")

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.send("quit")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


class CheckpointAgent:
    def __init__(self, checkpoint: Path, device: str) -> None:
        self.device = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.model, self.payload = load_model_for_inference(checkpoint, self.device)
        game_cfg = self.payload["game_config"]
        if int(game_cfg["board_size"]) != BOARD_SIZE:
            raise ValueError("Ten test wymaga checkpointu dla planszy 19x19")
        if int(game_cfg.get("win_length", WIN_LENGTH)) != WIN_LENGTH:
            raise ValueError("Ten test wymaga win_length=6")

        tr_cfg = self.payload.get("config", {}).get("training", {})
        self.amp_enabled = self.device.type == "cuda" and bool(tr_cfg.get("amp", True))
        amp_name = str(tr_cfg.get("amp_dtype", "bfloat16")).lower()
        self.amp_dtype = (
            torch.float16
            if amp_name in {"float16", "fp16", "half"}
            else torch.bfloat16
        )

    @torch.inference_mode()
    def action(self, game: Connect6Game) -> tuple[int, float]:
        started = time.perf_counter()
        board = torch.from_numpy(game.board).unsqueeze(0).to(self.device)
        current_player = torch.tensor(
            [game.current_player], dtype=torch.int8, device=self.device
        )
        stones_left = torch.tensor(
            [game.stones_left_in_turn], dtype=torch.int8, device=self.device
        )
        network_input = canonical_network_input(board, current_player, stones_left)
        legal = board.view(1, -1).eq(0)
        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.amp_enabled,
        ):
            logits, _ = self.model(network_input)
        logits = mask_logits(logits.float(), legal)
        action = int(logits.argmax(dim=1).item())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return action, time.perf_counter() - started


def _apply_cloudict_move(game: Connect6Game, raw_move: str) -> None:
    expected_player = int(game.current_player)
    actions = cloudict_move_to_actions(raw_move, game.board_size)
    if not actions:
        raise RuntimeError("Cloudict zwrocil pusty ruch")
    for action in actions:
        if game.done:
            break
        if int(game.current_player) != expected_player:
            raise RuntimeError(
                f"Cloudict zwrocil za duzo kamieni w ruchu {raw_move!r}; tura juz sie zmienila"
            )
        game.step(action)


def _play_model_turn(
    game: Connect6Game, agent: CheckpointAgent
) -> tuple[list[int], float]:
    actor = int(game.current_player)
    actions: list[int] = []
    elapsed = 0.0
    while not game.done and int(game.current_player) == actor:
        action, action_elapsed = agent.action(game)
        elapsed += action_elapsed
        game.step(action)
        actions.append(action)
    return actions, elapsed


def _result_label(winner: int, *, model_is_black: bool) -> str:
    if winner == EMPTY:
        return "DRAW"
    model_color = BLACK if model_is_black else WHITE
    return "WIN" if winner == model_color else "LOSS"


def play_one_game(
    agent: CheckpointAgent,
    cloudict: CloudictEngine,
    *,
    opening_action: int,
    model_is_black: bool,
) -> dict[str, object]:
    game = Connect6Game(BOARD_SIZE, WIN_LENGTH)
    opening_coord = action_to_cloudict(opening_action)
    game.step(opening_action)  # wymuszony pojedynczy pierwszy kamien czarnych

    model_seconds = 0.0
    cloudict_seconds = 0.0
    model_decisions = 0
    cloudict_decisions = 0
    started = time.perf_counter()

    first_reply = cloudict.start_game(
        opening_coord,
        bot_is_black=not model_is_black,
    )

    if model_is_black:
        if first_reply is None:
            raise RuntimeError("Cloudict bialy nie zwrocil pierwszej odpowiedzi")
        cloudict_seconds += first_reply.elapsed_seconds
        cloudict_decisions += 1
        _apply_cloudict_move(game, first_reply.move)

    while not game.done:
        model_color = BLACK if model_is_black else WHITE
        if int(game.current_player) != model_color:
            raise RuntimeError(
                "Rozjechal sie stan tury pomiedzy lokalna gra a Cloudictem "
                f"(opening={opening_coord}, model_is_black={model_is_black})"
            )

        model_actions, elapsed = _play_model_turn(game, agent)
        model_seconds += elapsed
        model_decisions += len(model_actions)
        if game.done:
            break

        if len(model_actions) != 2:
            raise RuntimeError(
                f"Model powinien wykonac 2 kamienie, dostano {model_actions}"
            )
        reply = cloudict.respond(actions_to_cloudict(model_actions))
        cloudict_seconds += reply.elapsed_seconds
        cloudict_decisions += 1
        _apply_cloudict_move(game, reply.move)

    elapsed_total = time.perf_counter() - started
    result = _result_label(int(game.winner), model_is_black=model_is_black)
    winner = (
        "DRAW"
        if game.winner == EMPTY
        else ("BLACK" if game.winner == BLACK else "WHITE")
    )
    return {
        "opening_action": int(opening_action),
        "opening_row": int(opening_action // BOARD_SIZE),
        "opening_col": int(opening_action % BOARD_SIZE),
        "opening_coord": opening_coord,
        "model_color": "BLACK" if model_is_black else "WHITE",
        "winner": winner,
        "result": result,
        "stones_played": int(game.move_count),
        "model_decisions": int(model_decisions),
        "cloudict_decisions": int(cloudict_decisions),
        "model_seconds": f"{model_seconds:.6f}",
        "cloudict_seconds": f"{cloudict_seconds:.6f}",
        "elapsed_seconds": f"{elapsed_total:.6f}",
    }


def _read_completed(path: Path) -> set[tuple[int, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    completed: set[tuple[int, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            try:
                completed.add((int(row["opening_action"]), row["model_color"]))
            except (KeyError, TypeError, ValueError):
                continue
    return completed


def _append_result(path: Path, row: dict[str, object]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _load_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    def stats(items: list[dict[str, str]]) -> dict[str, object]:
        wins = sum(row["result"] == "WIN" for row in items)
        draws = sum(row["result"] == "DRAW" for row in items)
        losses = sum(row["result"] == "LOSS" for row in items)
        games = len(items)
        score = wins + 0.5 * draws
        return {
            "games": games,
            "model_wins": wins,
            "draws": draws,
            "cloudict_wins": losses,
            "model_win_rate": wins / games if games else 0.0,
            "model_score_rate": score / games if games else 0.0,
        }

    black = [row for row in rows if row.get("model_color") == "BLACK"]
    white = [row for row in rows if row.get("model_color") == "WHITE"]
    by_opening: dict[int, float] = {}
    for row in rows:
        action = int(row["opening_action"])
        value = 1.0 if row["result"] == "WIN" else 0.5 if row["result"] == "DRAW" else 0.0
        by_opening[action] = by_opening.get(action, 0.0) + value
    pair_score_counts = {
        label: sum(abs(score - value) < 1e-9 for score in by_opening.values())
        for label, value in (("2.0", 2.0), ("1.5", 1.5), ("1.0", 1.0), ("0.5", 0.5), ("0.0", 0.0))
    }
    return {
        "overall": stats(rows),
        "model_as_black": stats(black),
        "model_as_white": stats(white),
        "opening_pair_score_counts": pair_score_counts,
    }


def _validate_run_config(path: Path, expected: dict[str, object]) -> None:
    if not path.exists():
        path.write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    current = json.loads(path.read_text(encoding="utf-8"))
    if current != expected:
        raise RuntimeError(
            f"Katalog wynikowy {path.parent} zawiera wyniki innego testu. "
            "Uzyj --reset albo innego --output-dir."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "722 partie checkpointu przeciw Cloudictowi: 361 wymuszonych pol startowych "
            "x oba kolory modelu."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--cloudict-exe", type=Path, default=DEFAULT_CLOUDICT_EXE)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--vcf", action="store_true", help="Wlacz VCF; domyslnie OFF")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.depth <= 9:
        raise ValueError("Cloudict obsluguje depth 1..9")
    checkpoint = args.checkpoint.resolve()
    cloudict_exe = args.cloudict_exe.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Nie znaleziono checkpointu: {checkpoint}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    config_path = output_dir / "run_config.json"
    summary_path = output_dir / "summary.json"
    if args.reset:
        for path in (results_path, config_path, summary_path):
            path.unlink(missing_ok=True)

    run_config = {
        "checkpoint": str(checkpoint),
        "cloudict_exe": str(cloudict_exe),
        "depth": int(args.depth),
        "vcf": bool(args.vcf),
        "device": str(args.device),
        "board_size": BOARD_SIZE,
        "win_length": WIN_LENGTH,
        "forced_openings": BOARD_SIZE * BOARD_SIZE,
        "games_per_opening": 2,
        "games_total": 2 * BOARD_SIZE * BOARD_SIZE,
        "model_policy": "argmax",
    }
    _validate_run_config(config_path, run_config)

    schedule = build_match_schedule()
    completed = _read_completed(results_path)
    remaining = [
        item
        for item in schedule
        if (item[0], "BLACK" if item[1] else "WHITE") not in completed
    ]

    print("=" * 78)
    print("CONNECT6 — CNN V6 vs CLOUDICT")
    print("=" * 78)
    print(f"Checkpoint : {checkpoint}")
    print(f"Cloudict   : {cloudict_exe}")
    print(f"Depth      : {args.depth} | VCF={'ON' if args.vcf else 'OFF'}")
    print(
        f"Test       : 361 pol startowych x 2 kolory = {len(schedule)} partii "
        f"({len(remaining)} pozostalo)"
    )
    print("Otwarcie   : pierwszy kamien czarnych jest wymuszany kolejno na A1..S19/AA..SS")
    print("Model      : argmax, bez losowania")
    print()

    if not remaining:
        summary = _summarize(_load_rows(results_path))
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    agent = CheckpointAgent(checkpoint, args.device)
    cloudict = CloudictEngine(
        cloudict_exe,
        depth=args.depth,
        vcf=args.vcf,
        timeout_seconds=args.timeout,
    )

    started_all = time.perf_counter()
    try:
        done_before = len(schedule) - len(remaining)
        for local_index, (opening_action, model_is_black) in enumerate(remaining, start=1):
            row = play_one_game(
                agent,
                cloudict,
                opening_action=opening_action,
                model_is_black=model_is_black,
            )
            game_index = done_before + local_index
            row = {"game_index": game_index, **row}
            _append_result(results_path, row)

            rows_now = _load_rows(results_path)
            wins = sum(r["result"] == "WIN" for r in rows_now)
            draws = sum(r["result"] == "DRAW" for r in rows_now)
            losses = sum(r["result"] == "LOSS" for r in rows_now)
            print(
                f"[{game_index:3d}/{len(schedule)}] start={row['opening_coord']} "
                f"model={row['model_color']:<5} -> {row['result']:<4} | "
                f"W/D/L={wins}/{draws}/{losses} | stones={row['stones_played']} | "
                f"{float(row['elapsed_seconds']):.3f}s",
                flush=True,
            )
    finally:
        cloudict.close()

    rows = _load_rows(results_path)
    summary = _summarize(rows)
    summary["elapsed_seconds_this_run"] = time.perf_counter() - started_all
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("=" * 78)
    print("WYNIK KONCOWY")
    print("=" * 78)
    overall = summary["overall"]
    black = summary["model_as_black"]
    white = summary["model_as_white"]
    print(
        f"RAZEM: model {overall['model_wins']} W / {overall['draws']} D / "
        f"{overall['cloudict_wins']} L | winrate={100.0 * overall['model_win_rate']:.2f}% | "
        f"score={100.0 * overall['model_score_rate']:.2f}%"
    )
    print(
        f"MODEL BLACK: {black['model_wins']} W / {black['draws']} D / "
        f"{black['cloudict_wins']} L | winrate={100.0 * black['model_win_rate']:.2f}%"
    )
    print(
        f"MODEL WHITE: {white['model_wins']} W / {white['draws']} D / "
        f"{white['cloudict_wins']} L | winrate={100.0 * white['model_win_rate']:.2f}%"
    )
    print(f"CSV     : {results_path}")
    print(f"Summary : {summary_path}")


if __name__ == "__main__":
    main()
