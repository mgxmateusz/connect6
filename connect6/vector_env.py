from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class StepBatch:
    reward: torch.Tensor
    done: torch.Tensor
    winner: torch.Tensor
    transition_sign: torch.Tensor
    game_lengths: torch.Tensor


class VectorConnect6:
    """Wektorowe środowisko Connect6 działające bezpośrednio na GPU.

    Wszystkie plansze znajdują się na jednym urządzeniu torch. Dzięki temu nie
    trzeba uruchamiać osobnego procesu dla każdej gry ani kopiować plansz między
    CPU i GPU przy każdej decyzji modelu.
    """

    def __init__(
        self,
        num_envs: int,
        board_size: int = 19,
        win_length: int = 6,
        device: str | torch.device = "cuda",
        debug_checks: bool = False,
    ) -> None:
        self.num_envs = int(num_envs)
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        self.device = torch.device(device)
        self.debug_checks = bool(debug_checks)

        self.boards = torch.zeros(
            (self.num_envs, self.board_size, self.board_size),
            dtype=torch.int8,
            device=self.device,
        )
        self.current_player = torch.ones(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self.stones_left = torch.ones(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self.empty_count = torch.full(
            (self.num_envs,),
            self.board_size * self.board_size,
            dtype=torch.int16,
            device=self.device,
        )
        self.move_count = torch.zeros(
            self.num_envs, dtype=torch.int16, device=self.device
        )

        # Przesunięcia używane do szybkiego sprawdzania linii tylko wokół
        # właśnie postawionego kamienia.
        k = torch.arange(
            -(self.win_length - 1), self.win_length, device=self.device
        )
        directions = torch.tensor(
            ((1, 0), (0, 1), (1, 1), (1, -1)), device=self.device
        )
        self._dr = directions[:, 0].view(1, 4, 1) * k.view(1, 1, -1)
        self._dc = directions[:, 1].view(1, 4, 1) * k.view(1, 1, -1)
        self._center_idx = self.win_length - 1

    @property
    def action_size(self) -> int:
        return self.board_size * self.board_size

    @property
    def input_size(self) -> int:
        return self.action_size * 2 + 2

    def reset(self, indices: torch.Tensor | None = None) -> None:
        if indices is None:
            self.boards.zero_()
            self.current_player.fill_(1)
            self.stones_left.fill_(1)
            self.empty_count.fill_(self.action_size)
            self.move_count.zero_()
            return

        if indices.numel() == 0:
            return

        self.boards[indices] = 0
        self.current_player[indices] = 1
        self.stones_left[indices] = 1
        self.empty_count[indices] = self.action_size
        self.move_count[indices] = 0

    def legal_mask(self) -> torch.Tensor:
        """Zwraca maskę wolnych pól [B, liczba_pól]."""
        return self.boards.view(self.num_envs, -1).eq(0)

    def network_input(self) -> torch.Tensor:
        """Buduje zwykły wektor wejściowy MLP dla wszystkich plansz."""
        return canonical_network_input(
            self.boards, self.current_player, self.stones_left
        )

    def observation(self) -> torch.Tensor:
        """Alias zachowany dla zgodności z kodem zewnętrznym."""
        return self.network_input()

    @torch.no_grad()
    def step(self, actions: torch.Tensor) -> StepBatch:
        actions = actions.to(
            device=self.device, dtype=torch.long, non_blocking=True
        )
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"Oczekiwano actions o kształcie {(self.num_envs,)}, "
                f"otrzymano {tuple(actions.shape)}"
            )

        rows = torch.div(actions, self.board_size, rounding_mode="floor")
        cols = actions.remainder(self.board_size)
        batch = torch.arange(self.num_envs, device=self.device)

        if self.debug_checks:
            occupied = self.boards[batch, rows, cols].ne(0)
            if bool(occupied.any()):
                bad = int(torch.nonzero(occupied, as_tuple=False).flatten()[0])
                raise RuntimeError(
                    f"Próba postawienia kamienia na zajętym polu w env {bad}"
                )

        actor = self.current_player.clone()
        self.boards[batch, rows, cols] = actor
        self.empty_count.sub_(1)
        self.move_count.add_(1)

        won = self._check_win_local(rows, cols, actor)
        draw = self.empty_count.eq(0) & ~won
        done = won | draw
        winner = torch.where(won, actor, torch.zeros_like(actor))
        reward = won.to(torch.float32)

        remaining = self.stones_left - 1
        same_player = remaining.gt(0)

        # Stan tur aktualizujemy tylko dla gier, które nadal trwają.
        live = ~done
        switch = live & ~same_player
        stay = live & same_player
        self.stones_left[stay] = remaining[stay]
        self.current_player[switch] = -self.current_player[switch]
        self.stones_left[switch] = 2

        transition_sign = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        transition_sign[stay] = 1
        transition_sign[switch] = -1

        return StepBatch(
            reward=reward,
            done=done,
            winner=winner,
            transition_sign=transition_sign,
            game_lengths=self.move_count.clone(),
        )

    def _check_win_local(
        self,
        rows: torch.Tensor,
        cols: torch.Tensor,
        player: torch.Tensor,
    ) -> torch.Tensor:
        # Po broadcastingu mamy:
        # [B, 4 kierunki, 2*win_length-1 pól wokół nowego kamienia].
        rr = rows.view(-1, 1, 1) + self._dr
        cc = cols.view(-1, 1, 1) + self._dc
        valid = (
            (rr >= 0)
            & (rr < self.board_size)
            & (cc >= 0)
            & (cc < self.board_size)
        )

        rr_safe = rr.clamp_(0, self.board_size - 1)
        cc_safe = cc.clamp_(0, self.board_size - 1)
        flat_idx = rr_safe * self.board_size + cc_safe
        flat_board = self.boards.view(self.num_envs, -1)
        gathered = torch.gather(
            flat_board, 1, flat_idx.view(self.num_envs, -1)
        ).view_as(flat_idx)
        match = valid & gathered.eq(player.view(-1, 1, 1))

        center = self._center_idx
        left_near_to_far = torch.flip(match[..., :center], dims=(-1,))
        right_near_to_far = match[..., center + 1 :]

        # cumprod zamienia np. [1,1,0,1] na [1,1,0,0], więc suma liczy tylko
        # kolejne kamienie bez przerwy, stykające się z nowym kamieniem.
        left_count = left_near_to_far.to(torch.int16).cumprod(-1).sum(-1)
        right_count = right_near_to_far.to(torch.int16).cumprod(-1).sum(-1)
        total = 1 + left_count + right_count
        return total.ge(self.win_length).any(dim=1)


def canonical_network_input(
    boards: torch.Tensor,
    current_player: torch.Tensor,
    stones_left: torch.Tensor,
) -> torch.Tensor:
    """Tworzy jeden zwykły wektor wejściowy dla MLP.

    Dla planszy 19x19 wynik ma kształt [B, 724]:
      0..360   = moje kamienie,
      361..721 = kamienie przeciwnika,
      722      = stones_left / 2,
      723      = czy aktualny gracz jest czarny.
    """
    player = current_player.view(-1, 1, 1)

    me = boards.eq(player).to(torch.float32).flatten(1)
    opp = boards.eq(-player).to(torch.float32).flatten(1)

    dodatkowe = torch.stack(
        (
            stones_left.to(torch.float32) / 2.0,
            current_player.eq(1).to(torch.float32),
        ),
        dim=1,
    )

    return torch.cat((me, opp, dodatkowe), dim=1)


def canonical_observation(
    boards: torch.Tensor,
    current_player: torch.Tensor,
    stones_left: torch.Tensor,
) -> torch.Tensor:
    """Alias zachowany dla zgodności z wcześniejszym kodem."""
    return canonical_network_input(boards, current_player, stones_left)
