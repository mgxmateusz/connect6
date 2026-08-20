from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


BLACK = 1
WHITE = -1
EMPTY = 0


@dataclass(slots=True)
class MoveResult:
    reward: float
    done: bool
    winner: int
    player_before: int
    player_after: int
    transition_sign: int


class Connect6Game:
    """Prosta implementacja Connect6 na CPU używana przez GUI i testy.

    Zasady zastosowane w projekcie:
    - standardowa plansza ma 19x19 pól,
    - czarne w pierwszym ruchu stawiają jeden kamień,
    - później każdy gracz stawia po dwa kamienie na turę,
    - sześć lub więcej kolejnych kamieni w jednej linii daje wygraną.

    Ruch składający się z dwóch kamieni jest reprezentowany jako dwie kolejne
    decyzje tej samej sieci. Przeciwnik nie wykonuje ruchu pomiędzy nimi.
    Dzięki temu model potrzebuje tylko 361 wyjść zamiast wszystkich możliwych
    par pustych pól.
    """

    def __init__(self, board_size: int = 19, win_length: int = 6):
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.board = np.zeros(
            (self.board_size, self.board_size), dtype=np.int8
        )
        self.current_player = BLACK
        self.stones_left_in_turn = 1
        self.winner = EMPTY
        self.done = False
        self.move_count = 0
        self.last_action: Optional[int] = None

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    @property
    def input_size(self) -> int:
        return self.action_size * 2 + 2

    def reset(self) -> None:
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.stones_left_in_turn = 1
        self.winner = EMPTY
        self.done = False
        self.move_count = 0
        self.last_action = None

    def clone(self) -> "Connect6Game":
        other = Connect6Game(self.board_size, self.win_length)
        other.board[:] = self.board
        other.current_player = self.current_player
        other.stones_left_in_turn = self.stones_left_in_turn
        other.winner = self.winner
        other.done = self.done
        other.move_count = self.move_count
        other.last_action = self.last_action
        return other

    def action_to_rc(self, action: int) -> tuple[int, int]:
        return divmod(int(action), self.board_size)

    def rc_to_action(self, row: int, col: int) -> int:
        return int(row) * self.board_size + int(col)

    def legal_mask(self) -> np.ndarray:
        return self.board.reshape(-1) == EMPTY

    def legal_actions(self) -> np.ndarray:
        return np.flatnonzero(self.legal_mask())

    def network_input(self) -> np.ndarray:
        """Buduje jeden zwykły wektor wejściowy MLP.

        Dla planszy 19x19 wynik ma 724 wartości:
          0..360   = moje kamienie,
          361..721 = kamienie przeciwnika,
          722      = liczba kamieni pozostałych w turze / 2,
          723      = 1, jeśli aktualny gracz jest czarny, inaczej 0.
        """
        me = (self.board == self.current_player).astype(np.float32).reshape(-1)
        opp = (self.board == -self.current_player).astype(np.float32).reshape(-1)
        dodatkowe = np.asarray(
            [
                self.stones_left_in_turn / 2.0,
                1.0 if self.current_player == BLACK else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate((me, opp, dodatkowe), axis=0)

    def observation(self) -> np.ndarray:
        """Alias zachowany dla zgodności z wcześniejszym kodem."""
        return self.network_input()

    def step(self, action: int) -> MoveResult:
        if self.done:
            raise RuntimeError("Gra jest zakończona. Najpierw wywołaj reset().")

        row, col = self.action_to_rc(action)
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            raise ValueError(f"Akcja {action} znajduje się poza planszą.")
        if self.board[row, col] != EMPTY:
            raise ValueError(f"Pole ({row}, {col}) jest już zajęte.")

        actor = int(self.current_player)
        self.board[row, col] = actor
        self.move_count += 1
        self.last_action = int(action)

        if self._check_win_from(row, col, actor):
            self.done = True
            self.winner = actor
            return MoveResult(1.0, True, actor, actor, actor, 0)

        if self.move_count >= self.action_size:
            self.done = True
            self.winner = EMPTY
            return MoveResult(0.0, True, EMPTY, actor, actor, 0)

        self.stones_left_in_turn -= 1
        if self.stones_left_in_turn > 0:
            player_after = actor
            sign = 1
        else:
            self.current_player = -self.current_player
            self.stones_left_in_turn = 2
            player_after = int(self.current_player)
            sign = -1

        return MoveResult(0.0, False, EMPTY, actor, player_after, sign)

    def _check_win_from(self, row: int, col: int, player: int) -> bool:
        """Sprawdza tylko cztery linie przechodzące przez nowy kamień."""
        for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
            count = 1
            for direction in (-1, 1):
                rr, cc = row, col
                while True:
                    rr += direction * dr
                    cc += direction * dc
                    if not (
                        0 <= rr < self.board_size
                        and 0 <= cc < self.board_size
                    ):
                        break
                    if self.board[rr, cc] != player:
                        break
                    count += 1
                    if count >= self.win_length:
                        return True
        return False
