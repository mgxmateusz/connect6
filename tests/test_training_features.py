from pathlib import Path

import pytest
import torch

from connect6.history import HistoricalCheckpoint, HistoricalPolicyEnsemble
from connect6.model import build_model
from connect6.train import (
    CompleteGameBuffer,
    _historical_layout,
    _historical_table_matrix,
    _temperature,
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


def test_historical_fraction_is_split_evenly_and_fixed_per_model():
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

    table_matrix, valid = _historical_table_matrix(opponent_ids, 4)
    assert int(valid.sum().item()) == 25
    assert table_matrix.shape[0] == 4
    for model_id in range(4):
        tables = table_matrix[model_id][valid[model_id]]
        assert torch.all(opponent_ids[tables] == model_id)


def test_grouped_history_ensemble_matches_individual_models_on_cpu():
    torch.manual_seed(123)
    board_size = 3
    model_cfg = {
        "architecture_version": 3,
        "layers": [
            {"neurons": 12, "norm": "layer", "activation": "silu", "dropout": 0.0},
            {"neurons": 8, "norm": "none", "activation": "silu", "dropout": 0.0},
        ],
        "policy_layers": [],
        "value_layers": [
            {"neurons": 4, "norm": "none", "activation": "silu", "dropout": 0.0},
        ],
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
    x = torch.randn(2, 5, board_size * board_size * 2 + 2)
    grouped = ensemble.forward_grouped(x)

    for i, model in enumerate(models):
        model.eval()
        expected, _ = model(x[i])
        assert torch.allclose(grouped[i], expected, atol=1e-5, rtol=1e-5)
