from __future__ import annotations

import torch


class FastChampionshipConnect6:
    """Connect6 engine wyspecjalizowany pod championship.

    Zachowuje kontrakt istniejących checkpointów:
      input[:,0] = moje kamienie
      input[:,1] = kamienie przeciwnika
      input[:,2] = 1 gdy to ostatni kamień aktualnej tury

    Optymalizacje względem VectorConnect6:
    - fizyczne maski black/white i legal są utrzymywane inkrementalnie,
    - input CNN nie skanuje board.eq(...) od zera,
    - win-check używa precomputed [361,4,11] lookup zamiast liczyć rr/cc/valid/clamp,
    - scheduler może aktualizować tylko aktywne sloty przez masked_step().
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
        self.action_size = self.board_size * self.board_size

        self.boards = torch.zeros(
            (self.num_envs, self.board_size, self.board_size),
            dtype=torch.int8,
            device=self.device,
        )
        # Cache fizycznych kolorów w bool. Konwersja do dtype CNN następuje dopiero
        # w network_input i nie wymaga porównywania pełnej planszy z graczem.
        self.black = torch.zeros_like(self.boards, dtype=torch.bool)
        self.white = torch.zeros_like(self.boards, dtype=torch.bool)
        self._legal = torch.ones(
            (self.num_envs, self.action_size), dtype=torch.bool, device=self.device
        )

        self.current_player = torch.ones(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self.stones_left = torch.ones(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self.empty_count = torch.full(
            (self.num_envs,), self.action_size, dtype=torch.int16, device=self.device
        )
        self.move_count = torch.zeros(
            self.num_envs, dtype=torch.int16, device=self.device
        )
        self._batch = torch.arange(
            self.num_envs, dtype=torch.long, device=self.device
        )

        self._neighbor_lookup = self._build_neighbor_lookup()
        # Sentinel 0 pozwala gatherować pola poza planszą bez valid/clamp.
        self._flat_with_sentinel = torch.zeros(
            (self.num_envs, self.action_size + 1), dtype=torch.int8, device=self.device
        )

    @property
    def input_channels(self) -> int:
        return 3

    def _build_neighbor_lookup(self) -> torch.Tensor:
        size = self.board_size
        radius = self.win_length - 1
        sentinel = self.action_size
        dirs = ((1, 0), (0, 1), (1, 1), (1, -1))
        table = torch.full(
            (self.action_size, 4, radius * 2 + 1),
            sentinel,
            dtype=torch.long,
        )
        for action in range(self.action_size):
            row, col = divmod(action, size)
            for d, (dr, dc) in enumerate(dirs):
                for oi, off in enumerate(range(-radius, radius + 1)):
                    rr = row + dr * off
                    cc = col + dc * off
                    if 0 <= rr < size and 0 <= cc < size:
                        table[action, d, oi] = rr * size + cc
        return table.to(self.device, non_blocking=True)

    def reset(self, indices: torch.Tensor | None = None) -> None:
        if indices is None:
            self.boards.zero_()
            self.black.zero_()
            self.white.zero_()
            self._legal.fill_(True)
            self._flat_with_sentinel.zero_()
            self.current_player.fill_(1)
            self.stones_left.fill_(1)
            self.empty_count.fill_(self.action_size)
            self.move_count.zero_()
            return
        if indices.numel() == 0:
            return
        self.boards[indices] = 0
        self.black[indices] = False
        self.white[indices] = False
        self._legal[indices] = True
        self._flat_with_sentinel[indices] = 0
        self.current_player[indices] = 1
        self.stones_left[indices] = 1
        self.empty_count[indices] = self.action_size
        self.move_count[indices] = 0

    def legal_mask(self) -> torch.Tensor:
        return self._legal

    def network_input(self) -> torch.Tensor:
        # Ten wynik jest semantycznie identyczny z canonical_network_input().
        player_black = self.current_player.eq(1).view(-1, 1, 1)
        mine = torch.where(player_black, self.black, self.white)
        opp = torch.where(player_black, self.white, self.black)
        last = self.stones_left.eq(1).view(-1, 1, 1).expand(
            -1, self.board_size, self.board_size
        )
        return torch.stack((mine, opp, last), dim=1).to(torch.float32)

    def observation(self) -> torch.Tensor:
        return self.network_input()

    def _check_win_lookup(
        self,
        actions: torch.Tensor,
        actor: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        # [B,4,11] indeksów pól leżących na czterech liniach przez nowy kamień.
        lookup = self._neighbor_lookup.index_select(0, actions)
        gathered = torch.gather(
            self._flat_with_sentinel,
            1,
            lookup.view(self.num_envs, -1),
        ).view(self.num_envs, 4, -1)
        match = gathered.eq(actor.view(-1, 1, 1))

        # Każde Connect6 zawierające właśnie postawiony kamień musi należeć do
        # jednego z maksymalnie 6 okien długości 6 w 11-polowej linii.
        windows = match.unfold(-1, self.win_length, 1)
        won = windows.all(dim=-1).any(dim=-1).any(dim=-1)
        return won & active

    @torch.no_grad()
    def masked_step(
        self,
        actions: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actions = actions.to(self.device, dtype=torch.long, non_blocking=True)
        actor = self.current_player.clone()
        batch = self._batch

        if self.debug_checks:
            occupied = ~self._legal[batch, actions]
            if bool((occupied & active).any()):
                raise RuntimeError("Próba zagrania na zajętym polu")

        # Nieaktywne sloty mają dummy action, ale ich stan pozostaje nietknięty.
        chosen_old = self._flat_with_sentinel[batch, actions]
        placed = torch.where(active, actor, chosen_old)
        self._flat_with_sentinel[batch, actions] = placed

        rows = torch.div(actions, self.board_size, rounding_mode="floor")
        cols = actions.remainder(self.board_size)
        old_board = self.boards[batch, rows, cols]
        self.boards[batch, rows, cols] = torch.where(active, actor, old_board)

        is_black = active & actor.eq(1)
        is_white = active & actor.eq(-1)
        self.black[batch[is_black], rows[is_black], cols[is_black]] = True
        self.white[batch[is_white], rows[is_white], cols[is_white]] = True
        self._legal[batch[active], actions[active]] = False

        active_i16 = active.to(torch.int16)
        self.empty_count.sub_(active_i16)
        self.move_count.add_(active_i16)

        won = self._check_win_lookup(actions, actor, active)
        draw = self.empty_count.eq(0) & ~won & active
        done = won | draw
        winner = torch.where(won, actor, torch.zeros_like(actor))

        remaining = self.stones_left - 1
        live = active & ~done
        stay = live & remaining.gt(0)
        switch = live & ~remaining.gt(0)
        self.stones_left[stay] = remaining[stay]
        self.current_player[switch] = -self.current_player[switch]
        self.stones_left[switch] = 2
        return done, winner


def assert_checkpoint_input_compatibility(device: torch.device) -> None:
    """Szybki test semantyki inputu względem treningowego VectorConnect6.

    Import lokalny unika zależności championship -> trening przy normalnym starcie.
    """
    from .vector_env import canonical_network_input

    size = 19
    boards = torch.zeros((8, size, size), dtype=torch.int8, device=device)
    current = torch.tensor([1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.int8, device=device)
    stones = torch.tensor([1, 1, 2, 2, 1, 2, 1, 2], dtype=torch.int8, device=device)
    # Deterministyczny zestaw różnych pozycji, bez RNG i CPU synchronizacji.
    flat = boards.view(8, -1)
    for b in range(8):
        for j in range(0, 24 + b, 3):
            idx = (j * 17 + b * 29) % (size * size)
            flat[b, idx] = 1 if (j + b) % 2 == 0 else -1

    expected = canonical_network_input(boards, current, stones)
    fast = FastChampionshipConnect6(8, size, 6, device=device)
    fast.boards.copy_(boards)
    fast.current_player.copy_(current)
    fast.stones_left.copy_(stones)
    fast.black.copy_(boards.eq(1))
    fast.white.copy_(boards.eq(-1))
    fast._legal.copy_(boards.view(8, -1).eq(0))
    fast._flat_with_sentinel[:, :-1].copy_(boards.view(8, -1))
    actual = fast.network_input()
    if not torch.equal(expected, actual):
        raise RuntimeError("FastChampionshipConnect6 zmienia semantykę inputu checkpointu")
