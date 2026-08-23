import torch

from connect6.engine.model import PolicyValueNet, build_model
from connect6.engine.train import (
    _inverse_transform_actions,
    _transform_actions,
    _transform_boards,
)


def test_model_shapes():
    model = PolicyValueNet(board_size=19)
    x = torch.zeros(3, 3, 19, 19)
    logits, value = model(x)
    assert logits.shape == (3, 361)
    assert value.shape == (3,)
    assert torch.all(value <= 1) and torch.all(value >= -1)
    assert model.receptive_field == 37


def test_explicit_cnn_architecture_from_config():
    cfg = {
        "architecture_version": 4,
        "kernels": [23, 3, 3, 3, 3, 3, 3, 3],
        "channels": [32, 32, 64, 64, 64, 96, 96, 96],
        "compile": False,
    }
    model = build_model(cfg, 19)

    assert [m.kernel_size[0] for m in model.convs] == [23, 3, 3, 3, 3, 3, 3, 3]
    assert [m.out_channels for m in model.convs] == [32, 32, 64, 64, 64, 96, 96, 96]
    assert model.convs[0].in_channels == 3
    assert model.policy_output.kernel_size == (1, 1)
    assert model.policy_output.in_channels == 96
    assert model.policy_output.out_channels == 1
    assert model.value_output.kernel_size == (1, 1)
    assert model.receptive_field == 37


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
