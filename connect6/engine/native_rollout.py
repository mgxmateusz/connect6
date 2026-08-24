from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from connect6.bots.gpu_bot import GPUTacticalBot, GPUTacticalBotV2

from .model import mask_logits
from .native_rollout_state import (
    NativeRolloutBuffer,
    NativeRolloutState,
    PackedRolloutModels,
)
from .vector_env import VectorConnect6, canonical_network_input


TABLE_SELF = 0
TABLE_HISTORY = 1
TABLE_BOT = 2

_KERNELS = (23, 3, 3, 3, 3, 3, 3, 3)
_CHANNELS = (32, 32, 64, 64, 64, 96, 96, 96)
_IN_CHANNELS = (4, 32, 32, 64, 64, 64, 96, 96)
_GROUP_CHANNELS = 8
_UNKNOWN_EPISODE_RESULT = 2
_UNKNOWN_TERMINAL_MOVE = -1


@dataclass(slots=True)
class RolloutAssignments:
    table_kind: torch.Tensor
    opponent_model: torch.Tensor
    current_color: torch.Tensor
    bot_version: torch.Tensor
    history_tables: int
    bot_tables: int
    bot_v1_tables: int
    bot_v2_tables: int


@dataclass(slots=True)
class NativeRolloutStats:
    completed_positions: int
    generated_positions: int
    games: int
    black_wins: int
    white_wins: int
    draws: int
    game_length_sum: int
    history_games: int
    history_wins: int
    history_losses: int
    history_draws: int
    bot_games: int
    bot_wins: int
    bot_losses: int
    bot_draws: int
    graph_steps: int
    symmetry_phase: int


def _balanced_colors(count: int, device: torch.device) -> torch.Tensor:
    """Return an as-even-as-possible randomized +/-1 color assignment."""
    count = int(count)
    colors = torch.empty(count, dtype=torch.int8, device=device)
    black = (count + 1) // 2
    colors[:black] = 1
    colors[black:] = -1
    if count > 1:
        colors = colors[torch.randperm(count, device=device)]
    return colors


def build_rollout_assignments(
    num_envs: int,
    historical_model_count: int,
    device: torch.device,
    *,
    historical_fraction: float,
    bot_fraction: float,
    bot_v1_fraction: float = 0.5,
) -> RolloutAssignments:
    """Assign self/history/bot roles without touching persistent board state.

    Packed model id 0 is always the current policy. Historical packed ids start
    at 1. If no compatible history exists yet, its quarter naturally falls back
    to current-vs-current until the first V5 checkpoints become available.
    """
    n = int(num_envs)
    historical_fraction = float(historical_fraction)
    bot_fraction = float(bot_fraction)
    bot_v1_fraction = float(bot_v1_fraction)
    if not 0.0 <= historical_fraction <= 1.0:
        raise ValueError("historical_fraction musi być w [0,1]")
    if not 0.0 <= bot_fraction <= 1.0:
        raise ValueError("bot_fraction musi być w [0,1]")
    if historical_fraction + bot_fraction > 1.0 + 1e-9:
        raise ValueError("historical_fraction + bot_fraction nie może przekraczać 1")
    if not 0.0 <= bot_v1_fraction <= 1.0:
        raise ValueError("bot_v1_fraction musi być w [0,1]")

    history_count = (
        int(round(n * historical_fraction)) if historical_model_count > 0 else 0
    )
    bot_count = int(round(n * bot_fraction))
    if history_count + bot_count > n:
        raise ValueError("Za dużo stołów przeciwników względem num_envs")

    permutation = torch.randperm(n, device=device)
    history_slots = permutation[:history_count]
    bot_slots = permutation[history_count : history_count + bot_count]

    table_kind = torch.zeros(n, dtype=torch.uint8, device=device)
    opponent_model = torch.full((n,), -1, dtype=torch.int32, device=device)
    current_color = torch.ones(n, dtype=torch.int8, device=device)
    bot_version = torch.zeros(n, dtype=torch.uint8, device=device)

    if history_count:
        table_kind[history_slots] = TABLE_HISTORY
        ids = torch.arange(history_count, dtype=torch.int32, device=device)
        ids = ids.remainder(int(historical_model_count))
        if history_count > 1:
            ids = ids[torch.randperm(history_count, device=device)]
        opponent_model[history_slots] = ids + 1
        current_color[history_slots] = _balanced_colors(history_count, device)

    v1_count = min(
        bot_count,
        max(0, int(round(bot_count * bot_v1_fraction))),
    )
    v2_count = bot_count - v1_count
    if bot_count:
        table_kind[bot_slots] = TABLE_BOT
        v1_slots = bot_slots[:v1_count]
        v2_slots = bot_slots[v1_count:]
        if v1_count:
            bot_version[v1_slots] = 1
            current_color[v1_slots] = _balanced_colors(v1_count, device)
        if v2_count:
            bot_version[v2_slots] = 2
            current_color[v2_slots] = _balanced_colors(v2_count, device)

    return RolloutAssignments(
        table_kind=table_kind,
        opponent_model=opponent_model,
        current_color=current_color,
        bot_version=bot_version,
        history_tables=history_count,
        bot_tables=bot_count,
        bot_v1_tables=v1_count,
        bot_v2_tables=v2_count,
    )


def _symmetry_for_phase(phase: int) -> tuple[int, bool]:
    phase = int(phase) & 7
    return phase & 3, phase >= 4


def _transform_boards(boards: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    out = torch.rot90(boards, k=int(k), dims=(-2, -1)) if k else boards
    return torch.flip(out, dims=(-1,)) if flip else out


def _inverse_transform_actions(
    actions: torch.Tensor,
    board_size: int,
    k: int,
    flip: bool,
) -> torch.Tensor:
    n = int(board_size)
    r = torch.div(actions, n, rounding_mode="floor")
    c = actions.remainder(n)
    if flip:
        c = n - 1 - c
    if k == 1:
        r, c = c, n - 1 - r
    elif k == 2:
        r, c = n - 1 - r, n - 1 - c
    elif k == 3:
        r, c = n - 1 - c, r
    return r * n + c


def _forced_random_opening_mask(
    move_count: torch.Tensor,
    current_player: torch.Tensor,
    stones_left: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    fresh_black = move_count.eq(0) & current_player.eq(1) & stones_left.eq(1)
    fraction = float(fraction)
    if fraction <= 0.0:
        return torch.zeros_like(fresh_black)
    if fraction >= 1.0:
        return fresh_black
    return fresh_black & torch.rand(
        move_count.shape, device=move_count.device
    ).lt(fraction)


def _sample_actions(
    logits: torch.Tensor,
    legal: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = mask_logits(logits.float(), legal) / max(float(temperature), 1e-4)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    return actions, dist.log_prob(actions)


def _sample_actions_only(
    logits: torch.Tensor,
    legal: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    logits = mask_logits(logits.float(), legal) / max(float(temperature), 1e-4)
    return Categorical(logits=logits).sample()


def _conv_weight(
    packed: PackedRolloutModels,
    layer: int,
    model_slice: slice | int,
) -> torch.Tensor:
    kernel = _KERNELS[layer]
    cin = _IN_CHANNELS[layer]
    cout = _CHANNELS[layer]
    kreal = cin * kernel * kernel
    raw = packed.conv_weights[layer][model_slice, :, :kreal]
    if isinstance(model_slice, int):
        return raw.reshape(cout, cin, kernel, kernel)
    return raw.reshape(raw.shape[0], cout, cin, kernel, kernel)


def _forward_current(
    packed: PackedRolloutModels,
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Current-policy forward through cuDNN/PyTorch, not the hand-written WMMA path."""
    x = x.to(dtype=torch.float16)
    for layer, (kernel, channels) in enumerate(zip(_KERNELS, _CHANNELS)):
        x = F.conv2d(
            x,
            _conv_weight(packed, layer, 0),
            bias=None,
            stride=1,
            padding=kernel // 2,
        )
        x = F.group_norm(
            x,
            num_groups=channels // _GROUP_CHANNELS,
            weight=packed.norm_weights[layer][0],
            bias=packed.norm_biases[layer][0],
            eps=1e-5,
        )
        x = F.silu(x)

    policy = F.conv2d(
        x,
        packed.policy_weight[0].reshape(1, 96, 1, 1),
        packed.policy_bias[0].reshape(1),
    ).flatten(1)
    value_map = F.conv2d(
        x,
        packed.value_weight[0].reshape(1, 96, 1, 1),
        packed.value_bias[0].reshape(1),
    )
    values = torch.tanh(value_map.mean(dim=(-2, -1)).squeeze(1))
    return policy, values


def _build_history_layout(
    assignments: RolloutAssignments,
    historical_models: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a fixed [models,tables] matrix once per update for grouped cuDNN."""
    device = assignments.table_kind.device
    if historical_models <= 0 or assignments.history_tables <= 0:
        empty = torch.empty((0, 0), dtype=torch.long, device=device)
        return (
            empty,
            torch.empty((0, 0), dtype=torch.bool, device=device),
            torch.empty(0, dtype=torch.long, device=device),
            torch.empty(0, dtype=torch.long, device=device),
        )

    history_mask = assignments.table_kind.eq(TABLE_HISTORY)
    groups: list[torch.Tensor] = []
    for model_id in range(historical_models):
        # assignment stores packed model ids, so history id k is packed id k+1.
        group = torch.nonzero(
            history_mask & assignments.opponent_model.eq(model_id + 1),
            as_tuple=False,
        ).flatten()
        if group.numel() == 0:
            raise RuntimeError(f"History model {model_id} nie ma przypisanego stołu")
        groups.append(group)

    max_tables = max(int(g.numel()) for g in groups)
    matrix = torch.empty(
        (historical_models, max_tables), dtype=torch.long, device=device
    )
    valid = torch.zeros(
        (historical_models, max_tables), dtype=torch.bool, device=device
    )
    for model_id, group in enumerate(groups):
        count = int(group.numel())
        matrix[model_id, :count] = group
        valid[model_id, :count] = True
        if count < max_tables:
            matrix[model_id, count:] = group[0]

    valid_positions = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
    flat_tables = matrix.reshape(-1)[valid_positions]
    return matrix, valid, valid_positions, flat_tables


def _forward_history_grouped(
    packed: PackedRolloutModels,
    x: torch.Tensor,
) -> torch.Tensor:
    """Historical policy forward using grouped cuDNN convolutions.

    x is [M,T,4,H,W]. Packed model 0 is current, so history weights use 1:.
    """
    models, tables, _, height, width = x.shape
    x = x.to(dtype=torch.float16)
    for layer, (kernel, channels, cin) in enumerate(
        zip(_KERNELS, _CHANNELS, _IN_CHANNELS)
    ):
        grouped_x = x.permute(1, 0, 2, 3, 4).reshape(
            tables, models * cin, height, width
        )
        weight = _conv_weight(packed, layer, slice(1, 1 + models)).reshape(
            models * channels, cin, kernel, kernel
        )
        y = F.conv2d(
            grouped_x,
            weight,
            bias=None,
            stride=1,
            padding=kernel // 2,
            groups=models,
        )
        x = y.reshape(tables, models, channels, height, width).permute(
            1, 0, 2, 3, 4
        )
        x = F.group_norm(
            x.reshape(models * tables, channels, height, width),
            num_groups=channels // _GROUP_CHANNELS,
            weight=None,
            bias=None,
            eps=1e-5,
        ).reshape(models, tables, channels, height, width)
        x = (
            x * packed.norm_weights[layer][1 : 1 + models, None, :, None, None]
            + packed.norm_biases[layer][1 : 1 + models, None, :, None, None]
        )
        x = F.silu(x)

    grouped_x = x.permute(1, 0, 2, 3, 4).reshape(
        tables, models * 96, height, width
    )
    policy_weight = packed.policy_weight[1 : 1 + models].reshape(
        models, 96, 1, 1
    )
    policy_bias = packed.policy_bias[1 : 1 + models]
    logits = F.conv2d(
        grouped_x,
        policy_weight,
        policy_bias,
        groups=models,
    )
    return logits.reshape(tables, models, height * width).permute(1, 0, 2)


class NativeRolloutCollector:
    """Fast hybrid GPU collector.

    Game state, policy tensors, bot inference and all rollout storage stay on the
    GPU. The expensive CNN path is deliberately executed by PyTorch/cuDNN because
    measurements on RTX 5070 showed the custom WMMA implementation was ~30x
    slower than the already optimized convolution stack. Python only orchestrates
    one large GPU batch per stone and receives compact terminal counters.
    """

    def __init__(
        self,
        num_envs: int,
        target_completed_positions: int,
        device: torch.device,
        *,
        verbose_build: bool = False,
    ) -> None:
        if device.type != "cuda":
            raise ValueError("NativeRolloutCollector wymaga CUDA")
        self.device = device
        self.num_envs = int(num_envs)
        self.state = NativeRolloutState.create(self.num_envs, device)
        self.env = VectorConnect6(
            num_envs=self.num_envs,
            board_size=19,
            win_length=6,
            device=device,
            debug_checks=False,
        )
        # Share persistent tensors with VectorConnect6 instead of copying state.
        self.env.boards = self.state.boards
        self.env.current_player = self.state.current_player
        self.env.stones_left = self.state.stones_left
        self.env.empty_count = self.state.empty_count
        self.env.move_count = self.state.move_count

        self.buffer = NativeRolloutBuffer(
            target_completed_positions=target_completed_positions,
            envs=self.num_envs,
            device=device,
        )
        self.verbose_build = bool(verbose_build)
        self.symmetry_phase = 0
        self.bot_v1 = GPUTacticalBot(device, verbose_build=verbose_build)
        self.bot_v2 = GPUTacticalBotV2(device, verbose_build=verbose_build)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True

    def _ext(self):
        """Compatibility warmup used by the benchmark: compile/load only bot CUDA."""
        self.bot_v1._ext()
        self.bot_v2._ext()
        return None

    @torch.inference_mode()
    def collect(
        self,
        packed: PackedRolloutModels,
        assignments: RolloutAssignments,
        *,
        temperature: float,
        random_black_opening_fraction: float,
        seed: int,
        symmetry_augmentation: bool,
    ) -> NativeRolloutStats:
        if packed.num_models < 1:
            raise ValueError("packed musi zawierać current policy")
        if assignments.table_kind.numel() != self.num_envs:
            raise ValueError("assignments ma złą liczbę stołów")

        self.buffer.count = 0
        self.buffer.episode_results.fill_(_UNKNOWN_EPISODE_RESULT)
        self.buffer.episode_terminal_moves.fill_(_UNKNOWN_TERMINAL_MOVE)

        current_episode_ids = torch.arange(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        segment_positions = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        next_episode_id = self.num_envs
        current_color = assignments.current_color.clone()

        historical_models = max(0, packed.num_models - 1)
        history_matrix, history_valid, history_valid_positions, history_flat_tables = (
            _build_history_layout(assignments, historical_models)
        )
        history_tables = int(assignments.history_tables)

        v1_slots = torch.nonzero(
            assignments.table_kind.eq(TABLE_BOT) & assignments.bot_version.eq(1),
            as_tuple=False,
        ).flatten()
        v2_slots = torch.nonzero(
            assignments.table_kind.eq(TABLE_BOT) & assignments.bot_version.eq(2),
            as_tuple=False,
        ).flatten()

        completed_positions = 0
        games = 0
        black_wins = 0
        white_wins = 0
        draws = 0
        game_length_sum = 0
        history_games = 0
        history_wins = 0
        history_losses = 0
        history_draws = 0
        bot_games = 0
        bot_wins = 0
        bot_losses = 0
        bot_draws = 0
        steps = 0

        # Keep the seed local to this rollout without touching CPU RNG.
        torch.cuda.manual_seed(int(seed) & 0x7FFFFFFF)

        while completed_positions < self.buffer.target_completed_positions:
            if symmetry_augmentation:
                k, flip = _symmetry_for_phase(self.symmetry_phase)
                view_boards = _transform_boards(self.env.boards, k, flip)
            else:
                k, flip = 0, False
                view_boards = self.env.boards

            network_input = canonical_network_input(
                view_boards, self.env.current_player, self.env.stones_left
            )
            legal = view_boards.reshape(self.num_envs, -1).eq(0)
            forced_opening = _forced_random_opening_mask(
                self.env.move_count,
                self.env.current_player,
                self.env.stones_left,
                random_black_opening_fraction,
            )

            kind = assignments.table_kind
            versus_current = kind.eq(TABLE_SELF) | self.env.current_player.eq(current_color)
            current_actor = versus_current & ~forced_opening
            history_actor = (
                kind.eq(TABLE_HISTORY)
                & self.env.current_player.ne(current_color)
                & ~forced_opening
            )
            bot_actor = (
                kind.eq(TABLE_BOT)
                & self.env.current_player.ne(current_color)
                & ~forced_opening
            )

            # Canonical environment actions. Every slot is filled by exactly one
            # of current/history/bot/random-opening below.
            env_actions = torch.empty(
                self.num_envs, dtype=torch.long, device=self.device
            )

            current_idx = torch.nonzero(current_actor, as_tuple=False).flatten()
            if current_idx.numel():
                current_logits, current_values = _forward_current(
                    packed, network_input[current_idx]
                )
                current_actions_view, current_logprobs = _sample_actions(
                    current_logits,
                    legal[current_idx],
                    temperature,
                )
                if symmetry_augmentation:
                    current_actions_env = _inverse_transform_actions(
                        current_actions_view, 19, k, flip
                    )
                else:
                    current_actions_env = current_actions_view
                env_actions[current_idx] = current_actions_env

                self.buffer.append_batch(
                    boards=view_boards[current_idx],
                    players=self.env.current_player[current_idx],
                    stones_left=self.env.stones_left[current_idx],
                    move_counts=self.env.move_count[current_idx],
                    actions=current_actions_view,
                    logprobs=current_logprobs,
                    values=current_values.float(),
                    episode_ids=current_episode_ids[current_idx],
                )
                segment_positions[current_idx] += 1

            if historical_models and history_tables:
                grouped_input = network_input[history_matrix]
                grouped_legal = legal[history_matrix]
                grouped_logits = _forward_history_grouped(packed, grouped_input)
                grouped_actions_view = _sample_actions_only(
                    grouped_logits,
                    grouped_legal,
                    temperature,
                )
                grouped_players = self.env.current_player[history_matrix]
                grouped_colors = current_color[history_matrix]
                grouped_forced = forced_opening[history_matrix]
                old_turn = (
                    history_valid
                    & grouped_players.ne(grouped_colors)
                    & ~grouped_forced
                )
                valid_old_turn = old_turn.reshape(-1)[history_valid_positions]
                valid_actions = grouped_actions_view.reshape(-1)[history_valid_positions]
                if symmetry_augmentation:
                    valid_actions = _inverse_transform_actions(
                        valid_actions, 19, k, flip
                    )
                previous = env_actions[history_flat_tables]
                env_actions[history_flat_tables] = torch.where(
                    valid_old_turn, valid_actions, previous
                )

            # Bots are cheap. Evaluate their fixed table subsets every step and use
            # the result only on the bot-owned turn; this avoids dynamic bot batches.
            if v1_slots.numel():
                v1_actions = self.bot_v1.actions(
                    self.env.boards[v1_slots],
                    self.env.current_player[v1_slots],
                    self.env.stones_left[v1_slots],
                ).long()
                v1_use = bot_actor[v1_slots]
                env_actions[v1_slots] = torch.where(
                    v1_use, v1_actions, env_actions[v1_slots]
                )
            if v2_slots.numel():
                v2_actions = self.bot_v2.actions(
                    self.env.boards[v2_slots],
                    self.env.current_player[v2_slots],
                    self.env.stones_left[v2_slots],
                ).long()
                v2_use = bot_actor[v2_slots]
                env_actions[v2_slots] = torch.where(
                    v2_use, v2_actions, env_actions[v2_slots]
                )

            if forced_opening.any():
                # Fresh black board is empty, therefore every 0..360 action is legal.
                random_actions = torch.randint(
                    0, 19 * 19, (self.num_envs,), device=self.device
                )
                env_actions = torch.where(forced_opening, random_actions, env_actions)

            step = self.env.step(env_actions)
            self.state.rng_counter.add_(1)
            steps += 1
            if symmetry_augmentation:
                self.symmetry_phase = (self.symmetry_phase + 1) & 7

            done_idx = torch.nonzero(step.done, as_tuple=False).flatten()
            if done_idx.numel():
                done_episode_ids = current_episode_ids[done_idx].long()
                winners = step.winner[done_idx]
                full_lengths = step.game_lengths[done_idx].long()
                completed_segments = segment_positions[done_idx].long()

                self.buffer.episode_results[done_episode_ids] = winners
                self.buffer.episode_terminal_moves[done_episode_ids] = full_lengths.to(
                    torch.int16
                )

                done_kind = assignments.table_kind[done_idx]
                done_colors = current_color[done_idx]
                done_history = done_kind.eq(TABLE_HISTORY)
                done_bot = done_kind.eq(TABLE_BOT)

                batch_stats = torch.stack(
                    (
                        completed_segments.sum(),
                        winners.eq(1).sum(),
                        winners.eq(-1).sum(),
                        winners.eq(0).sum(),
                        full_lengths.sum(),
                        done_history.sum(),
                        (done_history & winners.eq(done_colors)).sum(),
                        (done_history & winners.eq(-done_colors)).sum(),
                        (done_history & winners.eq(0)).sum(),
                        done_bot.sum(),
                        (done_bot & winners.eq(done_colors)).sum(),
                        (done_bot & winners.eq(-done_colors)).sum(),
                        (done_bot & winners.eq(0)).sum(),
                    )
                ).to(torch.int64).cpu().tolist()

                (
                    completed_add,
                    black_add,
                    white_add,
                    draw_add,
                    length_add,
                    history_games_add,
                    history_wins_add,
                    history_losses_add,
                    history_draws_add,
                    bot_games_add,
                    bot_wins_add,
                    bot_losses_add,
                    bot_draws_add,
                ) = [int(v) for v in batch_stats]

                completed_positions += completed_add
                games += int(done_idx.numel())
                black_wins += black_add
                white_wins += white_add
                draws += draw_add
                game_length_sum += length_add
                history_games += history_games_add
                history_wins += history_wins_add
                history_losses += history_losses_add
                history_draws += history_draws_add
                bot_games += bot_games_add
                bot_wins += bot_wins_add
                bot_losses += bot_losses_add
                bot_draws += bot_draws_add

                self.env.reset(done_idx)
                segment_positions[done_idx] = 0
                count_done = int(done_idx.numel())
                if next_episode_id + count_done > self.buffer.episode_capacity:
                    raise RuntimeError("episode result buffer capacity exceeded")
                current_episode_ids[done_idx] = torch.arange(
                    next_episode_id,
                    next_episode_id + count_done,
                    dtype=torch.int32,
                    device=self.device,
                )
                next_episode_id += count_done

                flip_color = done_history | done_bot
                current_color[done_idx] = torch.where(
                    flip_color, -done_colors, done_colors
                )

        torch.cuda.synchronize(self.device)
        return NativeRolloutStats(
            completed_positions=completed_positions,
            generated_positions=self.buffer.count,
            games=games,
            black_wins=black_wins,
            white_wins=white_wins,
            draws=draws,
            game_length_sum=game_length_sum,
            history_games=history_games,
            history_wins=history_wins,
            history_losses=history_losses,
            history_draws=history_draws,
            bot_games=bot_games,
            bot_wins=bot_wins,
            bot_losses=bot_losses,
            bot_draws=bot_draws,
            graph_steps=steps,
            symmetry_phase=self.symmetry_phase,
        )
