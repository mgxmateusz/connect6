import numpy as np
import torch

from connect6.game import BLACK, WHITE, Connect6Game
from connect6.vector_env import VectorConnect6


def test_opening_and_two_stone_turns():
    g = Connect6Game(board_size=19, win_length=6)
    assert g.current_player == BLACK
    assert g.stones_left_in_turn == 1

    g.step(g.rc_to_action(9, 9))
    assert g.current_player == WHITE
    assert g.stones_left_in_turn == 2

    g.step(g.rc_to_action(0, 0))
    assert g.current_player == WHITE
    assert g.stones_left_in_turn == 1

    g.step(g.rc_to_action(0, 1))
    assert g.current_player == BLACK
    assert g.stones_left_in_turn == 2


def test_horizontal_win_cpu():
    g = Connect6Game(board_size=9, win_length=6)
    g.board[4, 1:6] = BLACK
    g.current_player = BLACK
    g.stones_left_in_turn = 1
    result = g.step(g.rc_to_action(4, 6))
    assert result.done
    assert result.winner == BLACK


def test_diagonal_win_cpu():
    g = Connect6Game(board_size=9, win_length=6)
    for i in range(5):
        g.board[i, i] = WHITE
    g.current_player = WHITE
    result = g.step(g.rc_to_action(5, 5))
    assert result.done
    assert result.winner == WHITE


def test_observation_shape_and_channels():
    g = Connect6Game()
    network_input = g.network_input()
    assert network_input.shape == (3, 19, 19)
    assert network_input.dtype == np.float32
    assert not network_input[0].any()
    assert not network_input[1].any()
    assert network_input[2].all()  # opening = ostatni jedyny kamień tej tury

    g.step(g.rc_to_action(9, 9))
    white_input = g.network_input()
    assert white_input.shape == (3, 19, 19)
    assert white_input[1, 9, 9] == 1.0
    assert not white_input[2].any()  # biały zaczyna turę z dwoma kamieniami


def test_vector_env_cpu_device():
    env = VectorConnect6(
        num_envs=4,
        board_size=9,
        win_length=6,
        device="cpu",
        debug_checks=True,
    )
    actions = torch.tensor([0, 1, 2, 3])
    result = env.step(actions)
    assert result.done.shape == (4,)
    network_input = env.network_input()
    assert network_input.shape == (4, 3, 9, 9)
    assert network_input.dtype == torch.float32
