import torch

from connect6.model import PolicyValueNet, build_model


def test_model_shapes():
    model = PolicyValueNet(
        board_size=19,
        layers=[
            {"neurons": 128, "norm": "layer", "activation": "silu"},
            {"neurons": 64, "norm": "none", "activation": "silu"},
            {"neurons": 32, "norm": "none", "activation": "silu"},
        ],
        policy_layers=[],
        value_layers=[
            {"neurons": 16, "norm": "none", "activation": "silu"},
        ],
    )
    x = torch.zeros(3, 724)
    logits, value = model(x)
    assert logits.shape == (3, 361)
    assert value.shape == (3,)
    assert torch.all(value <= 1) and torch.all(value >= -1)


def test_explicit_layer_widths_from_config():
    cfg = {
        "architecture_version": 3,
        "layers": [
            {"neurons": 128, "norm": "layer", "activation": "silu"},
            {"neurons": 64, "norm": "none", "activation": "silu"},
            {"neurons": 32, "norm": "none", "activation": "silu"},
            {"neurons": 64, "norm": "layer", "activation": "silu"},
        ],
        "policy_layers": [
            {"neurons": 48, "norm": "none", "activation": "silu"},
        ],
        "value_layers": [
            {"neurons": 24, "norm": "none", "activation": "silu"},
        ],
        "compile": False,
    }
    model = build_model(cfg, 19)

    wspolne = [
        m for m in model.layers.modules() if isinstance(m, torch.nn.Linear)
    ]
    assert [m.out_features for m in wspolne] == [128, 64, 32, 64]
    assert model.policy_output.out_features == 361
    assert model.value_output.out_features == 1


def test_symmetry_transform_keeps_action_on_transformed_stone():
    from connect6.train import _transform_board_actions

    n = 7
    board = torch.zeros(1, n, n, dtype=torch.int8)
    board[0, 1, 2] = 1
    action = torch.tensor([1 * n + 2])
    for k in range(4):
        for flip in (False, True):
            out_board, out_action = _transform_board_actions(
                board, action, k, flip
            )
            r = int(out_action.item()) // n
            c = int(out_action.item()) % n
            assert int(out_board[0, r, c]) == 1
