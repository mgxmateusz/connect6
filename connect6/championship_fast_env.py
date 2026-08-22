from __future__ import annotations

import torch


class FastChampionshipConnect6:
    """Connect6 engine wyspecjalizowany pod championship.

    Zachowuje dokładny kontrakt istniejących checkpointów:
      input[:,0] = moje kamienie
      input[:,1] = kamienie przeciwnika
      input[:,2] = 1 gdy to ostatni kamień aktualnej tury

    Optymalizacje względem treningowego VectorConnect6:
    - black/white/legal są utrzymywane inkrementalnie,
    - input CNN nie wykonuje board.eq(...) po całej planszy co ruch,
    - dtype wejścia może być od razu BF16/FP16 zgodny z inference AMP,
    - rows/cols oraz sąsiedztwo do win-checku są precomputed,
    - win-check używa lookup [361,4,11] zamiast rr/cc/valid/clamp,
    - masked_step aktualizuje wyłącznie stan potrzebny championship,
    - brak boolean-compaction/nonzero na hot-path.
    """

    network_dtype: torch.dtype = torch.bfloat16

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

        # Int8 board zostaje dla zgodności/debugowania, lecz inference korzysta z
        # inkrementalnych masek black/white i nie skanuje go ponownie.
        self.boards = torch.zeros(
            (self.num_envs, self.board_size, self.board_size),
            dtype=torch.int8,
            device=self.device,
        )
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

        # Precomputed action -> row/col usuwa div/remainder z każdego ruchu.
        action_ids = torch.arange(self.action_size, dtype=torch.long)
        self._action_rows = torch.div(
            action_ids, self.board_size, rounding_mode="floor"
        ).to(self.device, non_blocking=True)
        self._action_cols = action_ids.remainder(self.board_size).to(
            self.device, non_blocking=True
        )

        self._neighbor_lookup = self._build_neighbor_lookup()
        # Sentinel=0 poza planszą. Dzięki temu win-check ma jeden gather i nie
        # potrzebuje valid/clamp ani budowania indeksów geometrycznych w runtime.
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
        # Semantyka jest identyczna z canonical_network_input(); zmienia się tylko
        # sposób materializacji oraz dtype zgodny z wybranym AMP inference.
        player_black = self.current_player.eq(1).view(-1, 1, 1)
        mine = torch.where(player_black, self.black, self.white)
        opp = torch.where(player_black, self.white, self.black)
        last = self.stones_left.eq(1).view(-1, 1, 1).expand(
            -1, self.board_size, self.board_size
        )
        return torch.stack((mine, opp, last), dim=1).to(self.network_dtype)

    def observation(self) -> torch.Tensor:
        return self.network_input()

    def _check_win_lookup(
        self,
        actions: torch.Tensor,
        actor: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        lookup = self._neighbor_lookup.index_select(0, actions)
        gathered = torch.gather(
            self._flat_with_sentinel,
            1,
            lookup.view(self.num_envs, -1),
        ).view(self.num_envs, 4, -1)
        match = gathered.eq(actor.view(-1, 1, 1))

        # Connect6 zawierające nowy kamień musi wystąpić w jednym z sześciu
        # okien długości 6 w 11-polowej linii. unfold jest widokiem, nie kopią.
        won = (
            match.unfold(-1, self.win_length, 1)
            .all(dim=-1)
            .any(dim=-1)
            .any(dim=-1)
        )
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

        # Dummy actions dla inactive mogą mieć dowolną wartość. torch.where
        # zachowuje ich poprzedni stan bez kompakcji indeksów.
        chosen_old = self._flat_with_sentinel[batch, actions]
        self._flat_with_sentinel[batch, actions] = torch.where(
            active, actor, chosen_old
        )

        rows = self._action_rows.index_select(0, actions)
        cols = self._action_cols.index_select(0, actions)
        old_board = self.boards[batch, rows, cols]
        self.boards[batch, rows, cols] = torch.where(active, actor, old_board)

        is_black = active & actor.eq(1)
        is_white = active & actor.eq(-1)
        old_black = self.black[batch, rows, cols]
        old_white = self.white[batch, rows, cols]
        self.black[batch, rows, cols] = old_black | is_black
        self.white[batch, rows, cols] = old_white | is_white

        old_legal = self._legal[batch, actions]
        self._legal[batch, actions] = old_legal & ~active

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
        switch = live & ~stay

        # Pełnowymiarowe where zamiast bool advanced indexing ogranicza liczbę
        # dynamicznych indeksowań i utrzymuje wszystko w stałych kształtach.
        self.stones_left.copy_(
            torch.where(stay, remaining, torch.where(switch, torch.full_like(remaining, 2), self.stones_left))
        )
        self.current_player.copy_(
            torch.where(switch, -self.current_player, self.current_player)
        )
        return done, winner


def assert_checkpoint_input_compatibility(
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Sprawdza dokładną semantykę inputu względem treningowego środowiska."""
    from .vector_env import canonical_network_input

    size = 19
    boards = torch.zeros((8, size, size), dtype=torch.int8, device=device)
    current = torch.tensor(
        [1, -1, 1, -1, 1, -1, 1, -1], dtype=torch.int8, device=device
    )
    stones = torch.tensor(
        [1, 1, 2, 2, 1, 2, 1, 2], dtype=torch.int8, device=device
    )
    flat = boards.view(8, -1)
    for b in range(8):
        for j in range(0, 24 + b, 3):
            idx = (j * 17 + b * 29) % (size * size)
            flat[b, idx] = 1 if (j + b) % 2 == 0 else -1

    expected = canonical_network_input(boards, current, stones).to(dtype)
    old_dtype = FastChampionshipConnect6.network_dtype
    FastChampionshipConnect6.network_dtype = dtype
    fast = FastChampionshipConnect6(8, size, 6, device=device)
    fast.boards.copy_(boards)
    fast.current_player.copy_(current)
    fast.stones_left.copy_(stones)
    fast.black.copy_(boards.eq(1))
    fast.white.copy_(boards.eq(-1))
    fast._legal.copy_(boards.view(8, -1).eq(0))
    fast._flat_with_sentinel[:, :-1].copy_(boards.view(8, -1))
    actual = fast.network_input()
    FastChampionshipConnect6.network_dtype = old_dtype
    if not torch.equal(expected, actual):
        raise RuntimeError("FastChampionshipConnect6 zmienia semantykę inputu checkpointu")


def assert_gameplay_compatibility(device: torch.device) -> None:
    """Porównuje deterministyczne ruchy ze starym VectorConnect6.

    Testuje board/current_player/stones_left/empty_count/move_count oraz done/winner.
    Jest wykonywany raz przy starcie championship, nie w hot-path.
    """
    from .vector_env import VectorConnect6

    envs = 16
    slow = VectorConnect6(envs, 19, 6, device=device, debug_checks=False)
    fast = FastChampionshipConnect6(envs, 19, 6, device=device, debug_checks=False)
    active = torch.ones(envs, dtype=torch.bool, device=device)

    for move in range(48):
        # Deterministyczne, różne legalne akcje. W razie kolizji bierzemy pierwsze
        # wolne pole tak samo dla obu środowisk.
        proposal = (
            torch.arange(envs, device=device, dtype=torch.long) * 37 + move * 53
        ).remainder(361)
        legal = slow.legal_mask()
        proposed_ok = legal.gather(1, proposal.view(-1, 1)).squeeze(1)
        fallback = legal.to(torch.int8).argmax(dim=1).to(torch.long)
        actions = torch.where(proposed_ok, proposal, fallback)

        slow_step = slow.step(actions)
        fast_done, fast_winner = fast.masked_step(actions, active)

        if not torch.equal(slow.boards, fast.boards):
            raise RuntimeError("Fast engine: rozjazd planszy względem VectorConnect6")
        if not torch.equal(slow.current_player, fast.current_player):
            raise RuntimeError("Fast engine: rozjazd current_player")
        if not torch.equal(slow.stones_left, fast.stones_left):
            raise RuntimeError("Fast engine: rozjazd stones_left")
        if not torch.equal(slow.empty_count, fast.empty_count):
            raise RuntimeError("Fast engine: rozjazd empty_count")
        if not torch.equal(slow.move_count, fast.move_count):
            raise RuntimeError("Fast engine: rozjazd move_count")
        if not torch.equal(slow_step.done, fast_done):
            raise RuntimeError("Fast engine: rozjazd detekcji końca gry")
        if not torch.equal(slow_step.winner, fast_winner):
            raise RuntimeError("Fast engine: rozjazd zwycięzcy")

        # Test kończymy przed resetami; chodzi o zgodność reguł i faz tury.
        if bool(slow_step.done.any()):
            break
