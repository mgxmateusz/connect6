from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from connect6.bots.gpu_bot import GPUTacticalBot, GPUTacticalBotV2

from .model import mask_logits
from .native_rollout_state import NativeRolloutBuffer, NativeRolloutState, PackedRolloutModels
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
_STATS_COUNT = 13


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
    """Assign fixed self/history/bot tables for one PPO update."""
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

    history_count = int(round(n * historical_fraction)) if historical_model_count > 0 else 0
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

    v1_count = min(bot_count, max(0, int(round(bot_count * bot_v1_fraction))))
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
    actions: torch.Tensor, board_size: int, k: int, flip: bool
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
    return fresh_black & torch.rand(move_count.shape, device=move_count.device).lt(fraction)


def _sample_actions(
    logits: torch.Tensor, legal: torch.Tensor, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = mask_logits(logits.float(), legal) / max(float(temperature), 1e-4)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    return actions, dist.log_prob(actions)


def _sample_actions_only(
    logits: torch.Tensor, legal: torch.Tensor, temperature: float
) -> torch.Tensor:
    logits = mask_logits(logits.float(), legal) / max(float(temperature), 1e-4)
    return Categorical(logits=logits).sample()


def _conv_weight(
    packed: PackedRolloutModels, layer: int, model_slice: slice | int
) -> torch.Tensor:
    kernel = _KERNELS[layer]
    cin = _IN_CHANNELS[layer]
    cout = _CHANNELS[layer]
    kreal = cin * kernel * kernel
    raw = packed.conv_weights[layer][model_slice, :, :kreal]
    if isinstance(model_slice, int):
        return raw.reshape(cout, cin, kernel, kernel)
    return raw.reshape(raw.shape[0], cout, cin, kernel, kernel)


def _prepare_current_weights(
    packed: PackedRolloutModels,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """Create stable fixed-shape views once, so cuDNN sees one shape every step."""
    conv = [_conv_weight(packed, i, 0).contiguous() for i in range(len(_KERNELS))]
    norm_w = [packed.norm_weights[i][0].contiguous() for i in range(len(_KERNELS))]
    norm_b = [packed.norm_biases[i][0].contiguous() for i in range(len(_KERNELS))]
    return conv, norm_w, norm_b


def _forward_current_fixed(
    x: torch.Tensor,
    conv_weights: list[torch.Tensor],
    norm_weights: list[torch.Tensor],
    norm_biases: list[torch.Tensor],
    packed: PackedRolloutModels,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed full-env forward. No dynamic compaction => no CUDA sync/autotune churn."""
    x = x.to(dtype=torch.float16)
    for layer, (kernel, channels) in enumerate(zip(_KERNELS, _CHANNELS)):
        x = F.conv2d(x, conv_weights[layer], bias=None, stride=1, padding=kernel // 2)
        x = F.group_norm(
            x,
            num_groups=channels // _GROUP_CHANNELS,
            weight=norm_weights[layer],
            bias=norm_biases[layer],
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
    assignments: RolloutAssignments, historical_models: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build fixed [models,tables] history layout once, outside the hot loop."""
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
        group = torch.nonzero(
            history_mask & assignments.opponent_model.eq(model_id + 1), as_tuple=False
        ).flatten()
        if group.numel() == 0:
            raise RuntimeError(f"History model {model_id} nie ma przypisanego stołu")
        groups.append(group)

    max_tables = max(int(g.numel()) for g in groups)
    matrix = torch.empty((historical_models, max_tables), dtype=torch.long, device=device)
    valid = torch.zeros((historical_models, max_tables), dtype=torch.bool, device=device)
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
    packed: PackedRolloutModels, x: torch.Tensor
) -> torch.Tensor:
    """Frozen historical policy forward using fixed grouped CUDA convolutions."""
    models, tables, _, height, width = x.shape
    x = x.to(dtype=torch.float16)
    for layer, (kernel, channels, cin) in enumerate(
        zip(_KERNELS, _CHANNELS, _IN_CHANNELS)
    ):
        grouped_x = x.permute(1, 0, 2, 3, 4).reshape(tables, models * cin, height, width)
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
        x = y.reshape(tables, models, channels, height, width).permute(1, 0, 2, 3, 4)
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

    grouped_x = x.permute(1, 0, 2, 3, 4).reshape(tables, models * 96, height, width)
    policy_weight = packed.policy_weight[1 : 1 + models].reshape(models, 96, 1, 1)
    policy_bias = packed.policy_bias[1 : 1 + models]
    logits = F.conv2d(grouped_x, policy_weight, policy_bias, groups=models)
    return logits.reshape(tables, models, height * width).permute(1, 0, 2)


class NativeRolloutCollector:
    """GPU-resident hybrid collector with a fixed-shape cuDNN hot path.

    The previous hybrid implementation accidentally synchronized CUDA several
    times per move (`torch.nonzero`, bool(cuda_tensor), terminal .cpu()) and fed
    cuDNN a different current-policy batch size almost every step. With
    `cudnn.benchmark=True` that caused algorithm-search churn and left the GPU
    mostly idle. This version intentionally evaluates current policy on all envs,
    writes fixed env-sized slabs, keeps terminal accounting on GPU and performs
    only one unavoidable scalar sync per step to preserve the exact first step
    at which the completed-position target is crossed.
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
        self.bot_v1._ext()
        self.bot_v2._ext()
        return None

    def _compact_slabs(self, valid: torch.Tensor, slab_count: int) -> int:
        """Compact current-policy rows once after rollout instead of every move."""
        valid_idx = torch.nonzero(valid[:slab_count], as_tuple=False).flatten()
        count = int(valid_idx.numel())
        if count == 0:
            self.buffer.count = 0
            return 0

        # Advanced indexing creates temporary GPU tensors, so overlapping source
        # and destination ranges are safe during the one-off compaction.
        self.buffer.boards[:count].copy_(self.buffer.boards[valid_idx])
        self.buffer.players[:count].copy_(self.buffer.players[valid_idx])
        self.buffer.stones_left[:count].copy_(self.buffer.stones_left[valid_idx])
        self.buffer.move_counts[:count].copy_(self.buffer.move_counts[valid_idx])
        self.buffer.actions[:count].copy_(self.buffer.actions[valid_idx])
        self.buffer.logprobs[:count].copy_(self.buffer.logprobs[valid_idx])
        self.buffer.values[:count].copy_(self.buffer.values[valid_idx])
        self.buffer.episode_ids[:count].copy_(self.buffer.episode_ids[valid_idx])
        self.buffer.count = count
        return count

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
        next_episode_id = torch.tensor(
            self.num_envs, dtype=torch.int64, device=self.device
        )
        current_color = assignments.current_color.clone()

        historical_models = max(0, packed.num_models - 1)
        history_matrix, history_valid, history_valid_positions, history_flat_tables = (
            _build_history_layout(assignments, historical_models)
        )
        history_tables = int(assignments.history_tables)

        # Fixed bot subsets are built once. Their nonzero calls are outside the hot loop.
        v1_slots = torch.nonzero(
            assignments.table_kind.eq(TABLE_BOT) & assignments.bot_version.eq(1),
            as_tuple=False,
        ).flatten()
        v2_slots = torch.nonzero(
            assignments.table_kind.eq(TABLE_BOT) & assignments.bot_version.eq(2),
            as_tuple=False,
        ).flatten()

        current_conv, current_norm_w, current_norm_b = _prepare_current_weights(packed)
        slab_valid = torch.empty(self.buffer.capacity, dtype=torch.bool, device=self.device)
        stats_gpu = torch.zeros(_STATS_COUNT, dtype=torch.int64, device=self.device)
        completed_gpu = torch.zeros((), dtype=torch.int64, device=self.device)
        steps = 0

        kind = assignments.table_kind
        history_kind = kind.eq(TABLE_HISTORY)
        bot_kind = kind.eq(TABLE_BOT)
        opponent_kind = history_kind | bot_kind

        torch.cuda.manual_seed(int(seed) & 0x7FFFFFFF)

        while True:
            slab_start = steps * self.num_envs
            slab_end = slab_start + self.num_envs
            if slab_end > self.buffer.capacity:
                raise RuntimeError(
                    "Rollout slab capacity exceeded before target. "
                    f"Potrzeba {slab_end:,}, capacity={self.buffer.capacity:,}."
                )

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

            versus_current = kind.eq(TABLE_SELF) | self.env.current_player.eq(current_color)
            current_actor = versus_current & ~forced_opening
            bot_actor = bot_kind & self.env.current_player.ne(current_color) & ~forced_opening

            # Crucial performance choice: always use the same [num_envs,4,19,19]
            # current-policy batch. Dynamic torch.nonzero batches synchronized the
            # host and caused cuDNN benchmark/autotune to see hundreds of shapes.
            current_logits, current_values = _forward_current_fixed(
                network_input,
                current_conv,
                current_norm_w,
                current_norm_b,
                packed,
            )
            current_actions_view, current_logprobs = _sample_actions(
                current_logits, legal, temperature
            )
            if symmetry_augmentation:
                env_actions = _inverse_transform_actions(current_actions_view, 19, k, flip)
            else:
                env_actions = current_actions_view.clone()

            # Fixed slab write: no dynamic current_idx, no per-step compaction.
            self.buffer.boards[slab_start:slab_end].copy_(view_boards)
            self.buffer.players[slab_start:slab_end].copy_(self.env.current_player)
            self.buffer.stones_left[slab_start:slab_end].copy_(self.env.stones_left)
            self.buffer.move_counts[slab_start:slab_end].copy_(self.env.move_count)
            self.buffer.actions[slab_start:slab_end].copy_(current_actions_view.to(torch.int16))
            self.buffer.logprobs[slab_start:slab_end].copy_(current_logprobs.float())
            self.buffer.values[slab_start:slab_end].copy_(current_values.float())
            self.buffer.episode_ids[slab_start:slab_end].copy_(current_episode_ids)
            slab_valid[slab_start:slab_end].copy_(current_actor)
            segment_positions.add_(current_actor.to(torch.int32))

            if historical_models and history_tables:
                grouped_input = network_input[history_matrix]
                grouped_legal = legal[history_matrix]
                grouped_logits = _forward_history_grouped(packed, grouped_input)
                grouped_actions_view = _sample_actions_only(
                    grouped_logits, grouped_legal, temperature
                )
                grouped_players = self.env.current_player[history_matrix]
                grouped_colors = current_color[history_matrix]
                grouped_forced = forced_opening[history_matrix]
                old_turn = history_valid & grouped_players.ne(grouped_colors) & ~grouped_forced
                valid_old_turn = old_turn.reshape(-1)[history_valid_positions]
                valid_actions = grouped_actions_view.reshape(-1)[history_valid_positions]
                if symmetry_augmentation:
                    valid_actions = _inverse_transform_actions(valid_actions, 19, k, flip)
                env_actions[history_flat_tables] = torch.where(
                    valid_old_turn, valid_actions, env_actions[history_flat_tables]
                )

            # Fixed bot batches; no dynamic selection inside the loop.
            if v1_slots.numel():
                v1_actions = self.bot_v1.actions(
                    self.env.boards[v1_slots],
                    self.env.current_player[v1_slots],
                    self.env.stones_left[v1_slots],
                ).long()
                env_actions[v1_slots] = torch.where(
                    bot_actor[v1_slots], v1_actions, env_actions[v1_slots]
                )
            if v2_slots.numel():
                v2_actions = self.bot_v2.actions(
                    self.env.boards[v2_slots],
                    self.env.current_player[v2_slots],
                    self.env.stones_left[v2_slots],
                ).long()
                env_actions[v2_slots] = torch.where(
                    bot_actor[v2_slots], v2_actions, env_actions[v2_slots]
                )

            # Always execute the fixed-size random generation. bool(cuda_tensor)
            # used here previously forced a full device synchronization every step.
            random_actions = torch.randint(0, 19 * 19, (self.num_envs,), device=self.device)
            env_actions = torch.where(forced_opening, random_actions, env_actions)

            step = self.env.step(env_actions)
            self.state.rng_counter.add_(1)
            steps += 1
            if symmetry_augmentation:
                self.symmetry_phase = (self.symmetry_phase + 1) & 7

            done = step.done
            done_i64 = done.to(torch.int64)
            winners = step.winner
            full_lengths = step.game_lengths.to(torch.int64)
            done_history = done & history_kind
            done_bot = done & bot_kind

            # Episode result writes stay fixed-size and entirely on-device.
            episode_slots = current_episode_ids.long()
            old_results = self.buffer.episode_results[episode_slots]
            old_terminal = self.buffer.episode_terminal_moves[episode_slots]
            self.buffer.episode_results[episode_slots] = torch.where(
                done, winners, old_results
            )
            self.buffer.episode_terminal_moves[episode_slots] = torch.where(
                done, step.game_lengths.to(torch.int16), old_terminal
            )

            completed_add = (segment_positions.to(torch.int64) * done_i64).sum()
            completed_gpu.add_(completed_add)

            # Aggregate all reporting counters on GPU; transfer only once at end.
            stats_gpu[0].add_(done_i64.sum())
            stats_gpu[1].add_((done & winners.eq(1)).sum())
            stats_gpu[2].add_((done & winners.eq(-1)).sum())
            stats_gpu[3].add_((done & winners.eq(0)).sum())
            stats_gpu[4].add_((full_lengths * done_i64).sum())
            stats_gpu[5].add_(done_history.sum())
            stats_gpu[6].add_((done_history & winners.eq(current_color)).sum())
            stats_gpu[7].add_((done_history & winners.eq(-current_color)).sum())
            stats_gpu[8].add_((done_history & winners.eq(0)).sum())
            stats_gpu[9].add_(done_bot.sum())
            stats_gpu[10].add_((done_bot & winners.eq(current_color)).sum())
            stats_gpu[11].add_((done_bot & winners.eq(-current_color)).sum())
            stats_gpu[12].add_((done_bot & winners.eq(0)).sum())

            # Reset completed boards without torch.nonzero / dynamic indices.
            self.env.boards.masked_fill_(done[:, None, None], 0)
            self.env.current_player.masked_fill_(done, 1)
            self.env.stones_left.masked_fill_(done, 1)
            self.env.empty_count.masked_fill_(done, 19 * 19)
            self.env.move_count.masked_fill_(done, 0)
            segment_positions.masked_fill_(done, 0)

            flip_color = done & opponent_kind
            current_color.copy_(torch.where(flip_color, -current_color, current_color))

            # Allocate fresh episode ids with a fixed-size prefix sum, still GPU-only.
            done_rank = done.to(torch.int32).cumsum(0)
            new_ids = (next_episode_id + done_rank.to(torch.int64) - 1).to(torch.int32)
            current_episode_ids.copy_(torch.where(done, new_ids, current_episode_ids))
            next_episode_id.add_(done_rank[-1].to(torch.int64))

            # One scalar sync remains intentionally: the user requested stopping at
            # the exact first rollout step X that reaches the target, while keeping
            # every game that also terminates in that same step X.
            if int(completed_gpu.item()) >= self.buffer.target_completed_positions:
                break

        slab_count = steps * self.num_envs
        generated_positions = self._compact_slabs(slab_valid, slab_count)
        stats = [int(v) for v in stats_gpu.cpu().tolist()]
        completed_positions = int(completed_gpu.item())
        next_episode = int(next_episode_id.item())
        if next_episode > self.buffer.episode_capacity:
            raise RuntimeError(
                f"episode result buffer capacity exceeded: {next_episode:,} > "
                f"{self.buffer.episode_capacity:,}"
            )

        torch.cuda.synchronize(self.device)
        return NativeRolloutStats(
            completed_positions=completed_positions,
            generated_positions=generated_positions,
            games=stats[0],
            black_wins=stats[1],
            white_wins=stats[2],
            draws=stats[3],
            game_length_sum=stats[4],
            history_games=stats[5],
            history_wins=stats[6],
            history_losses=stats[7],
            history_draws=stats[8],
            bot_games=stats[9],
            bot_wins=stats[10],
            bot_losses=stats[11],
            bot_draws=stats[12],
            graph_steps=steps,
            symmetry_phase=self.symmetry_phase,
        )
