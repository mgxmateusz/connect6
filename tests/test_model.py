import torch

from connect6.engine.model import PolicyValueNet, build_model
from connect6.engine.train import (
    _inverse_transform_actions,
    _transform_actions,
    _transform_boards,
)
from connect6.engine.vector_env import canonical_network_input


def test_model_shapes():
    model = PolicyValueNet(board_size=19)
    x = torch.zeros(3, 4, 19, 19)
    logits, value = model(x)
    assert logits.shape == (3, 361)
    assert value.shape == (3,)
    assert torch.all(value <= 1) and torch.all(value >= -1)
    assert model.receptive_field == 37


def test_explicit_cnn_architecture_from_config():
    cfg = {
        "architecture_version": 5,
        "kernels": [23, 3, 3, 3, 3, 3, 3, 3],
        "channels": [32, 32, 64, 64, 64, 96, 96, 96],
        "compile": False,
    }
    model = build_model(cfg, 19)

    assert [m.kernel_size[0] for m in model.convs] == [23, 3, 3, 3, 3, 3, 3, 3]
    assert [m.out_channels for m in model.convs] == [32, 32, 64, 64, 64, 96, 96, 96]
    assert model.convs[0].in_channels == 4
    assert all(conv.bias is None for conv in model.convs)
    assert [norm.num_groups for norm in model.norms] == [4, 4, 8, 8, 8, 12, 12, 12]
    assert [norm.num_channels // norm.num_groups for norm in model.norms] == [8] * 8
    assert model.policy_output.kernel_size == (1, 1)
    assert model.policy_output.in_channels == 96
    assert model.policy_output.out_channels == 1
    assert model.value_output.kernel_size == (1, 1)
    assert model.receptive_field == 37


def test_canonical_input_has_board_mask_and_last_stone_broadcast():
    boards = torch.zeros((2, 3, 3), dtype=torch.int8)
    boards[0, 0, 1] = 1
    boards[0, 2, 2] = -1
    current_player = torch.tensor([1, -1], dtype=torch.int8)
    stones_left = torch.tensor([2, 1], dtype=torch.int8)

    x = canonical_network_input(boards, current_player, stones_left)
    assert x.shape == (2, 4, 3, 3)
    assert x[0, 0, 0, 1] == 1
    assert x[0, 1, 2, 2] == 1
    assert torch.all(x[:, 2] == 1)
    assert torch.all(x[0, 3] == 0)
    assert torch.all(x[1, 3] == 1)


def test_groupnorm_requires_eight_channels_per_group():
    try:
        PolicyValueNet(board_size=5, kernels=[3], channels=[12])
    except ValueError as exc:
        assert "podzielną przez 8" in str(exc)
    else:
        raise AssertionError("Kanały niepodzielne przez 8 powinny zostać odrzucone")


def test_symmetry_transform_keeps_action_on_transformed_stone():
    n = 7
    board = torch.zeros(1, n, n, dtype=torch.int8)
    board[0, 1, 2] = 1
    action = torch.tensor([1 * n + 2])
    for k in range(4):
        for flip in (False, True):
            out_board = _transform_boards(board, k, flip)
            out_action = _transform_actions(action, n, k, flip)
            r = int(out_action.item()) // n
            c = int(out_action.item()) % n
            assert int(out_board[0, r, c]) == 1
            restored = _inverse_transform_actions(out_action, n, k, flip)
            assert torch.equal(restored, action)
