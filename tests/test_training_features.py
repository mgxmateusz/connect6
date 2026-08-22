from pathlib import Path

import pytest
import torch

from connect6.history import HistoricalCheckpoint, HistoricalPolicyEnsemble
from connect6.model import build_model
from connect6.train import (
    CompleteGameBuffer,
    _forced_random_opening_mask,
    _historical_layout,
    _historical_table_matrix,
    _inverse_transform_actions,
    _symmetry_for_phase,
    _temperature,
    _transform_actions,
    _transform_boards,
)


def test_gamma_discount_is_per_stone_and_terminal_action_has_distance_zero():
    device = torch.device("cpu")
    buffer = CompleteGameBuffer(
        target_completed_positions=4,
        envs=1,
        board_size=3,
        device=device,
    )

    boards = torch.zeros((2, 3, 3), dtype=torch.int8)
    players = torch.tensor([1, 1], dtype=torch.int8)
    stones_left = torch.tensor([1, 2], dtype=torch.int8)
    move_counts = torch.tensor([0, 2], dtype=torch.int16)
    actions = torch.tensor([0, 1], dtype=torch.long)
    logprobs = torch.zeros(2)
    values = torch.zeros(2)
    episode_ids = torch.tensor([0, 0], dtype=torch.int32)

    buffer.append_batch(
        boards=boards,
        players=players,
        stones_left=stones_left,
        move_counts=move_counts,
        actions=actions,
        logprobs=logprobs,
        values=values,
        episode_ids=episode_ids,
    )

    episode_results = torch.tensor([1, 2, 2, 2, 2], dtype=torch.int8)
    episode_terminal_moves = torch.tensor([4, -1, -1, -1, -1], dtype=torch.int16)

    indices, returns = buffer.completed_samples(
        episode_results,
        episode_terminal_moves,
        gamma=0.5,
    )

    assert indices.tolist() == [0, 1]
    assert returns.tolist() == pytest.approx([0.5**3, 0.5])


def test_temperature_uses_one_based_update_numbers():
    cfg = {
        "temperature_start": 1.0,
        "temperature_end": 0.5,
        "temperature_decay_updates": 10,
    }
    assert _temperature(1, cfg) == pytest.approx(1.0)
    assert _temperature(11, cfg) == pytest.approx(0.5)


def test_historical_fraction_is_split_evenly_and_colors_are_balanced():
    mask, opponent_ids, colors = _historical_layout(
        num_envs=100,
        fraction=0.25,
        historical_models=4,
        device=torch.device("cpu"),
    )

    assert int(mask.sum().item()) == 25
    counts = [int(opponent_ids.eq(i).sum().item()) for i in range(4)]
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == 25
    assert torch.all(opponent_ids[~mask] == -1)
    assert set(colors[mask].tolist()).issubset({-1, 1})

    for model_id in range(4):
        model_mask = opponent_ids.eq(model_id)
        black = int((colors[model_mask] == 1).sum().item())
        white = int((colors[model_mask] == -1).sum().item())
        assert abs(black - white) <= 1

    total_black = int((colors[mask] == 1).sum().item())
    total_white = int((colors[mask] == -1).sum().item())
    assert abs(total_black - total_white) <= 1

    table_matrix, valid = _historical_table_matrix(opponent_ids, 4)
    assert int(valid.sum().item()) == 25
    assert table_matrix.shape[0] == 4
    for model_id in range(4):
        tables = table_matrix[model_id][valid[model_id]]
        assert torch.all(opponent_ids[tables] == model_id)


def test_historical_layout_with_fewer_models_never_drops_tables():
    mask, opponent_ids, colors = _historical_layout(
        num_envs=16,
        fraction=0.5,
        historical_models=3,
        device=torch.device("cpu"),
    )

    assert int(mask.sum().item()) == 8
    counts = [int(opponent_ids.eq(i).sum().item()) for i in range(3)]
    assert counts == [3, 3, 2]
    assert abs(int((colors[mask] == 1).sum()) - int((colors[mask] == -1).sum())) <= 1


def test_grouped_history_ensemble_matches_individual_cnn_models_on_cpu():
    torch.manual_seed(123)
    board_size = 5
    model_cfg = {
        "architecture_version": 4,
        "kernels": [3, 3, 3],
        "channels": [8, 8, 16],
        "compile": False,
        "compile_mode": "default",
    }
    game_cfg = {"board_size": board_size, "win_length": 3}

    models = [build_model(model_cfg, board_size) for _ in range(2)]
    checkpoints = [
        HistoricalCheckpoint(
            path=Path(f"model_update_{i + 1:08d}.pt"),
            update=i + 1,
            model_state=model.state_dict(),
            model_config=model_cfg,
            game_config=game_cfg,
        )
        for i, model in enumerate(models)
    ]

    ensemble = HistoricalPolicyEnsemble(checkpoints, torch.device("cpu"))
    x = torch.randn(2, 5, 3, board_size, board_size)
    grouped = ensemble.forward_grouped(x)

    for i, model in enumerate(models):
        model.eval()
        expected, _ = model(x[i])
        assert torch.allclose(grouped[i], expected, atol=1e-5, rtol=1e-5)


def test_online_symmetry_cycle_visits_all_eight_d4_views_in_order():
    expected = [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]
    assert [_symmetry_for_phase(i) for i in range(8)] == expected
    assert _symmetry_for_phase(8) == expected[0]


def test_online_symmetry_board_action_and_inverse_action_are_consistent():
    board = torch.arange(9, dtype=torch.int8).reshape(1, 3, 3)
    actions = torch.arange(9, dtype=torch.long)
    original_values = board.reshape(-1)[actions]

    for phase in range(8):
        k, flip = _symmetry_for_phase(phase)
        transformed_board = _transform_boards(board, k, flip)
        transformed_actions = _transform_actions(actions, 3, k, flip)
        transformed_values = transformed_board.reshape(-1)[transformed_actions]
        restored_actions = _inverse_transform_actions(
            transformed_actions,
            board_size=3,
            k=k,
            flip=flip,
        )

        assert torch.equal(transformed_values, original_values)
        assert torch.equal(restored_actions, actions)


def test_random_black_opening_mask_only_selects_fresh_black_first_move():
    move_count = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int16)
    current_player = torch.tensor([1, -1, 1, 1, 1], dtype=torch.int8)
    stones_left = torch.tensor([1, 1, 1, 2, 1], dtype=torch.int8)

    forced_all = _forced_random_opening_mask(
        move_count,
        current_player,
        stones_left,
        fraction=1.0,
    )
    forced_none = _forced_random_opening_mask(
        move_count,
        current_player,
        stones_left,
        fraction=0.0,
    )

    assert forced_all.tolist() == [True, False, False, False, True]
    assert not forced_none.any()
