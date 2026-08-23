from __future__ import annotations

from dataclasses import dataclass

import torch

from connect6.cuda_native.rollout_loader import load_native_rollout_extension

from .native_rollout_state import (
    NativeRolloutBuffer,
    NativeRolloutState,
    PackedRolloutModels,
)


TABLE_SELF = 0
TABLE_HISTORY = 1
TABLE_BOT = 2


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


class NativeRolloutCollector:
    """One Python call per rollout; the per-move loop lives in a CUDA Graph."""

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
        self.state = NativeRolloutState.create(num_envs, device)
        self.buffer = NativeRolloutBuffer(
            target_completed_positions=target_completed_positions,
            envs=num_envs,
            device=device,
        )
        self.verbose_build = bool(verbose_build)
        self._extension = None
        self.symmetry_phase = 0

    def _ext(self):
        if self._extension is None:
            self._extension = load_native_rollout_extension(
                verbose=self.verbose_build
            )
        return self._extension

    @torch.no_grad()
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
        outputs = self._ext().run_rollout(
            packed.conv_weights,
            packed.norm_weights,
            packed.norm_biases,
            packed.policy_weight,
            packed.policy_bias,
            packed.value_weight,
            packed.value_bias,
            self.state.boards,
            self.state.current_player,
            self.state.stones_left,
            self.state.empty_count,
            self.state.move_count,
            self.state.rng_counter,
            assignments.table_kind,
            assignments.opponent_model,
            assignments.current_color,
            assignments.bot_version,
            self.buffer.boards,
            self.buffer.players,
            self.buffer.stones_left,
            self.buffer.move_counts,
            self.buffer.actions,
            self.buffer.logprobs,
            self.buffer.values,
            self.buffer.episode_ids,
            self.buffer.episode_results,
            self.buffer.episode_terminal_moves,
            self.buffer.target_completed_positions,
            float(temperature),
            float(random_black_opening_fraction),
            int(seed),
            bool(symmetry_augmentation),
            int(self.symmetry_phase),
        )
        counters = [int(v) for v in outputs[0].cpu().tolist()]
        self.symmetry_phase = int(outputs[1].item())

        if counters[17] != 0:
            errors = {
                1: "rollout buffer capacity exceeded",
                3: "native policy/bot returned an illegal action",
                4: "episode id out of result buffer",
                5: "episode result buffer capacity exceeded",
                6: "CUDA graph safety step limit reached",
            }
            raise RuntimeError(
                "Native rollout failed: "
                + errors.get(counters[17], f"error code {counters[17]}")
            )

        self.buffer.count = counters[1]
        return NativeRolloutStats(
            completed_positions=counters[0],
            generated_positions=counters[1],
            games=counters[3],
            black_wins=counters[4],
            white_wins=counters[5],
            draws=counters[6],
            game_length_sum=counters[7],
            history_games=counters[8],
            history_wins=counters[9],
            history_losses=counters[10],
            history_draws=counters[11],
            bot_games=counters[12],
            bot_wins=counters[13],
            bot_losses=counters[14],
            bot_draws=counters[15],
            graph_steps=counters[16],
            symmetry_phase=self.symmetry_phase,
        )
