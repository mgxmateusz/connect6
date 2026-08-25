from connect6.evaluation.cloudict_arena import (
    action_to_cloudict,
    actions_to_cloudict,
    build_match_schedule,
    cloudict_move_to_actions,
)


def test_schedule_has_two_games_for_every_opening():
    schedule = build_match_schedule(19)
    assert len(schedule) == 722
    for action in range(361):
        assert schedule[2 * action] == (action, True)
        assert schedule[2 * action + 1] == (action, False)


def test_cloudict_coordinates_cover_board_corners_and_center():
    assert action_to_cloudict(0) == "AA"
    assert action_to_cloudict(9 * 19 + 9) == "JJ"
    assert action_to_cloudict(360) == "SS"


def test_cloudict_pair_round_trip():
    actions = [9 * 19 + 10, 10 * 19 + 9]
    encoded = actions_to_cloudict(actions)
    assert encoded == "JKKJ"
    assert cloudict_move_to_actions(encoded) == actions
