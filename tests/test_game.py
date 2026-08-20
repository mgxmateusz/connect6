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
    # Tworzymy bezpośrednio stan potrzebny do testu reguły wygranej.
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


def test_observation_shape():
    g = Connect6Game()
    network_input = g.network_input()
    assert network_input.shape == (724,)
    assert network_input.dtype == np.float32


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
    assert network_input.shape == (4, 164)
