from pathlib import Path

import pytest
import torch

from connect6.history import HistoricalModel
from connect6.train import CompleteGameBuffer, _historical_layout, _temperature


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
    # Sample 0: action at move_count 0, terminal move_count 4 -> 3 later actions.
    # Sample 1: action at move_count 2, terminal move_count 4 -> 1 later action.
    assert returns.tolist() == pytest.approx([0.5**3, 0.5])


def test_temperature_uses_one_based_update_numbers():
    cfg = {
        "temperature_start": 1.0,
        "temperature_end": 0.5,
        "temperature_decay_updates": 10,
    }
    assert _temperature(1, cfg) == pytest.approx(1.0)
    assert _temperature(11, cfg) == pytest.approx(0.5)


def test_historical_fraction_and_colors():
    models = [
        HistoricalModel(Path("model_update_00000010.pt"), 10, torch.nn.Identity()),
        HistoricalModel(Path("model_update_00000020.pt"), 20, torch.nn.Identity()),
    ]

    mask, opponent_ids, colors = _historical_layout(
        num_envs=16,
        fraction=0.125,
        historical_models=models,
        device=torch.device("cpu"),
    )

    assert int(mask.sum().item()) == 2
    assert torch.all(opponent_ids[mask] >= 0)
    assert torch.all(opponent_ids[mask] < 2)
    assert set(colors[mask].tolist()).issubset({-1, 1})
    assert torch.all(opponent_ids[~mask] == -1)
