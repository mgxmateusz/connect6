from pathlib import Path
import csv
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from connect6.bots.gpu_bot import (
    GPUTacticalBot,
    GPUTacticalBotV2,
    GPUTacticalBotV3,
    GPUTacticalBotV4,
    GPUTacticalBotFullPair,
    GPUTacticalBotPairFirst,
    GPUTacticalBotLiveRoad,
)
from connect6.championship import bot_arena as _model_arena
from connect6.championship import championship as _legacy
from connect6.evaluation import bot_strength_arena as _base
from connect6.evaluation import cloudict_arena as _cloudict


# The CNN gauntlet intentionally uses only even-numbered autosaves:
# model_update_00000002.pt, 00000004.pt, ... . Bot-vs-bot and Cloudict do not
# use discover_checkpoints(), so they are unaffected by this filter.
_OriginalDiscoverCheckpoints = _legacy.discover_checkpoints


def _discover_even_checkpoints(directory):
    refs = _OriginalDiscoverCheckpoints(directory)
    return [ref for ref in refs if int(ref.update) % 2 == 0]


_legacy.discover_checkpoints = _discover_even_checkpoints


# All registered bots below are normal arena participants. The config decides
# which ones run through the same three stages: CNN gauntlet, full round-robin,
# then Cloudict. Small TOP12 is intentionally no longer part of this test set.
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
        "GPU Tactical Bot V3 Top16 Pair-State",
        "gpu_tactical_bot_v3_top16_pair_state_v1",
        GPUTacticalBotV3,
    ),
    _model_arena.BotSpec(
        "v4",
        "GPU Tactical Bot V4 Top12 ReplyPair6",
        "gpu_tactical_bot_v4_top12_pair_top4_v2reply6_pairs_v1",
        GPUTacticalBotV4,
    ),
    _model_arena.BotSpec(
        "pair",
        "GPU Tactical Bot PairFirst AllPairs P128",
        "gpu_tactical_bot_pairfirst_allpairs_p128_v1",
        GPUTacticalBotPairFirst,
    ),
    _model_arena.BotSpec(
        "live",
        "GPU Tactical Bot LiveRoad Brute Force",
        "gpu_tactical_bot_liveroad_ge2_min16_bruteforce_v1",
        GPUTacticalBotLiveRoad,
    ),
    _model_arena.BotSpec(
        "full",
        "GPU Tactical Bot Full Pair Brute Force",
        "gpu_tactical_bot_full_pair_bruteforce_v1",
        GPUTacticalBotFullPair,
    ),
)
_base.BOT_BY_KEY = {spec.key: spec for spec in _base.BOT_SPECS}


# In bot-vs-bot, draws are diagnostic but should not pull every comparison
# toward 50%. Report the win share among decisive games: wins/(wins+losses).
def _decisive_pair_rows(pair_rows):
    converted = []
    for raw in pair_rows:
        row = dict(raw)
        try:
            a_wins = int(row.get("a_wins", 0))
            b_wins = int(row.get("b_wins", 0))
        except (TypeError, ValueError):
            converted.append(row)
            continue
        decisive = a_wins + b_wins
        if decisive > 0:
            a_pct = 100.0 * a_wins / decisive
            b_pct = 100.0 * b_wins / decisive
        else:
            a_pct = b_pct = 50.0
        row["a_score_pct"] = f"{a_pct:.3f}"
        row["b_score_pct"] = f"{b_pct:.3f}"
        converted.append(row)
    return converted


def _rewrite_pair_csv(path: Path, rows) -> None:
    if not rows:
        return
    fields = list(_base.PAIR_FIELDS)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _print_decisive_pair_summary(rows) -> None:
    if not rows:
        return
    print("\n# BOT-VS-BOT — % WYGRANYCH TYLKO W ROZSTRZYGNIETYCH GRACH")
    for row in rows:
        try:
            aw = int(row.get("a_wins", 0))
            dr = int(row.get("draws", 0))
            bw = int(row.get("b_wins", 0))
        except (TypeError, ValueError):
            continue
        decisive = aw + bw
        if decisive:
            a_pct = float(row["a_score_pct"])
            b_pct = float(row["b_score_pct"])
            pct_text = f"{a_pct:.2f}% / {b_pct:.2f}%"
        else:
            pct_text = "N/A (0 rozstrzygnietych)"
        print(
            f"{row.get('bot_a', '?').upper()} vs {row.get('bot_b', '?').upper()}: "
            f"{aw} W / {dr} D / {bw} L | decisive={decisive} | {pct_text}"
        )


_OriginalWriteSummary = _base._write_summary


def _write_summary_with_decisive_pairs(
    output_dir,
    specs,
    model_summaries,
    pair_rows,
    cloud_rows,
    depths,
):
    converted = _decisive_pair_rows(pair_rows)
    _rewrite_pair_csv(Path(output_dir) / "bot_vs_bot.csv", converted)
    _print_decisive_pair_summary(converted)
    return _OriginalWriteSummary(
        output_dir,
        specs,
        model_summaries,
        converted,
        cloud_rows,
        depths,
    )


_base._write_summary = _write_summary_with_decisive_pairs


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
        print(f"[CLOUDICT] engine symmetry #{self._protocol_symmetry}", flush=True)

    def _reply_to_local(self, reply):
        if reply is None:
            return None
        return _cloudict.SearchReply(
            move=_transform_move(reply.move, _inverse_symmetry(self._protocol_symmetry)),
            elapsed_seconds=reply.elapsed_seconds,
        )

    def start_game(self, opening_coord: str, *, bot_is_black: bool):
        reply = super().start_game(
            _transform_move(opening_coord, self._protocol_symmetry),
            bot_is_black=bot_is_black,
        )
        return self._reply_to_local(reply)

    def respond(self, opponent_move: str):
        reply = super().respond(_transform_move(opponent_move, self._protocol_symmetry))
        return self._reply_to_local(reply)


_base.cloudict.CloudictEngine = _SymmetricRetryCloudictEngine

from connect6.evaluation.bot_strength_arena_fast import main


if __name__ == "__main__":
    main()
