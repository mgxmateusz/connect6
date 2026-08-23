from connect6.bot_arena import _result_from_winner, _summary


def _row(update, black, white):
    results = (black, white)
    model_wins = sum(r == "WIN" for r in results)
    draws = sum(r == "DRAW" for r in results)
    bot_wins = 2 - model_wins - draws
    score = model_wins + 0.5 * draws
    return {
        "model": f"model_update_{update:08d}.pt",
        "update": update,
        "model_black_result": black,
        "model_white_result": white,
        "model_wins": model_wins,
        "draws": draws,
        "bot_wins": bot_wins,
        "model_wins_as_black": int(black == "WIN"),
        "model_wins_as_white": int(white == "WIN"),
        "bot_wins_as_black": int(white == "LOSS"),
        "bot_wins_as_white": int(black == "LOSS"),
        "model_black_moves": 80,
        "model_white_moves": 82,
        "model_score": score,
        "model_score_pct": 100.0 * score / 2.0,
        "bot_score_pct": 100.0 - 100.0 * score / 2.0,
    }


def test_result_from_winner_respects_model_colour():
    assert _result_from_winner(1, model_is_black=True) == "WIN"
    assert _result_from_winner(-1, model_is_black=True) == "LOSS"
    assert _result_from_winner(-1, model_is_black=False) == "WIN"
    assert _result_from_winner(1, model_is_black=False) == "LOSS"
    assert _result_from_winner(0, model_is_black=True) == "DRAW"


def test_summary_colour_stats_and_stable_threshold():
    rows = [
        _row(10, "LOSS", "LOSS"),
        _row(20, "WIN", "LOSS"),
        _row(30, "WIN", "WIN"),
        _row(40, "WIN", "DRAW"),
    ]
    summary = _summary(rows)

    assert summary["models"] == 4
    assert summary["games"] == 8
    assert summary["bot_wins"] == 3
    assert summary["model_wins"] == 4
    assert summary["draws"] == 1
    assert summary["first_model_win_update"] == 20
    assert summary["first_model_sweep_update"] == 30
    assert summary["last_bot_win_update"] == 20
    assert summary["stable_no_bot_win_from_update"] == 30

    # Bot black is the game where the model is white.
    assert summary["bot_as_black"]["wins"] == 2
    # Bot white is the game where the model is black.
    assert summary["bot_as_white"]["wins"] == 1
