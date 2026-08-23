import pytest
import torch
from torch.utils.cpp_extension import CUDA_HOME

from connect6.gpu_bot import GPUTacticalBot


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or CUDA_HOME is None,
    reason="GPU tactical bot tests require CUDA + NVCC",
)


def _action(board: torch.Tensor, player: int = 1, stones_left: int = 1) -> int:
    bot = GPUTacticalBot("cuda")
    boards = board.to(device="cuda", dtype=torch.int8).unsqueeze(0)
    players = torch.tensor([player], dtype=torch.int8, device="cuda")
    left = torch.tensor([stones_left], dtype=torch.int8, device="cuda")
    return int(bot.actions(boards, players, left)[0].item())


def test_gpu_bot_prefers_centre_on_empty_board():
    board = torch.zeros((19, 19), dtype=torch.int8)
    assert _action(board) == 9 * 19 + 9


def test_gpu_bot_takes_immediate_win():
    board = torch.zeros((19, 19), dtype=torch.int8)
    board[9, 4:9] = 1
    action = _action(board, player=1, stones_left=1)
    assert action in {9 * 19 + 3, 9 * 19 + 9}


def test_gpu_bot_blocks_immediate_loss():
    board = torch.zeros((19, 19), dtype=torch.int8)
    board[10, 4:9] = -1
    action = _action(board, player=1, stones_left=1)
    assert action in {10 * 19 + 3, 10 * 19 + 9}


def test_gpu_bot_first_stone_sets_up_two_stone_win():
    board = torch.zeros((19, 19), dtype=torch.int8)
    board[8, 6:10] = 1
    first = _action(board, player=1, stones_left=2)
    r, c = divmod(first, 19)
    board[r, c] = 1
    second = _action(board, player=1, stones_left=1)
    rr, cc = divmod(second, 19)
    board[rr, cc] = 1

    # The greedy second evaluation must now finish at least one six-stone line.
    found = False
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        for row in range(19):
            for col in range(19):
                end_r = row + 5 * dr
                end_c = col + 5 * dc
                if not (0 <= end_r < 19 and 0 <= end_c < 19):
                    continue
                if all(int(board[row + k * dr, col + k * dc]) == 1 for k in range(6)):
                    found = True
                    break
            if found:
                break
        if found:
            break
    assert found
