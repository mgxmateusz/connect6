from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from .history import HistoricalCheckpoint


BOARD = 19
HW = BOARD * BOARD
EXPECTED_KERNELS = (23, 3, 3, 3, 3, 3, 3, 3)
EXPECTED_CHANNELS = (32, 32, 64, 64, 64, 96, 96, 96)
PACKED_K = (2128, 288, 288, 576, 576, 576, 864, 864)
NORM_LAYERS = (0, 2, 5, 7)


@dataclass(slots=True)
class NativeRolloutState:
    boards: torch.Tensor
    current_player: torch.Tensor
    stones_left: torch.Tensor
    empty_count: torch.Tensor
    move_count: torch.Tensor
    rng_counter: torch.Tensor

    @classmethod
    def create(cls, num_envs: int, device: torch.device) -> "NativeRolloutState":
        if device.type != "cuda":
            raise ValueError("NativeRolloutState wymaga CUDA")
        n = int(num_envs)
        return cls(
            boards=torch.zeros((n, BOARD, BOARD), dtype=torch.int8, device=device),
            current_player=torch.ones(n, dtype=torch.int8, device=device),
            stones_left=torch.ones(n, dtype=torch.int8, device=device),
            empty_count=torch.full((n,), HW, dtype=torch.int16, device=device),
            move_count=torch.zeros(n, dtype=torch.int16, device=device),
            rng_counter=torch.zeros(n, dtype=torch.int64, device=device),
        )

    @property
    def num_envs(self) -> int:
        return int(self.boards.shape[0])


class NativeRolloutBuffer:
    def __init__(self, target_completed_positions: int, envs: int, device: torch.device) -> None:
        self.target_completed_positions = int(target_completed_positions)
        self.envs = int(envs)
        self.capacity = self.target_completed_positions + self.envs * HW + self.envs
        self.episode_capacity = self.capacity + self.envs + 1
        self.count = 0
        self.boards = torch.empty((self.capacity, BOARD, BOARD), dtype=torch.int8, device=device)
        self.players = torch.empty(self.capacity, dtype=torch.int8, device=device)
        self.stones_left = torch.empty(self.capacity, dtype=torch.int8, device=device)
        self.move_counts = torch.empty(self.capacity, dtype=torch.int16, device=device)
        self.actions = torch.empty(self.capacity, dtype=torch.int16, device=device)
        self.logprobs = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.values = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.episode_ids = torch.empty(self.capacity, dtype=torch.int32, device=device)
        self.episode_results = torch.empty(self.episode_capacity, dtype=torch.int8, device=device)
        self.episode_terminal_moves = torch.empty(self.episode_capacity, dtype=torch.int16, device=device)

    def append_batch(self, *, boards: torch.Tensor, players: torch.Tensor, stones_left: torch.Tensor,
                     move_counts: torch.Tensor, actions: torch.Tensor, logprobs: torch.Tensor,
                     values: torch.Tensor, episode_ids: torch.Tensor) -> None:
        batch = int(boards.shape[0])
        if batch == 0:
            return
        start = self.count
        end = start + batch
        if end > self.capacity:
            raise RuntimeError(f"Rollout buffer capacity exceeded: {end:,} > {self.capacity:,}")
        self.boards[start:end].copy_(boards)
        self.players[start:end].copy_(players)
        self.stones_left[start:end].copy_(stones_left)
        self.move_counts[start:end].copy_(move_counts.to(torch.int16))
        self.actions[start:end].copy_(actions.to(torch.int16))
        self.logprobs[start:end].copy_(logprobs.float())
        self.values[start:end].copy_(values.float())
        self.episode_ids[start:end].copy_(episode_ids.to(torch.int32))
        self.count = end

    def completed_samples(self, gamma: float) -> tuple[torch.Tensor, torch.Tensor]:
        used_episode_ids = self.episode_ids[: self.count].long()
        winners = self.episode_results[used_episode_ids]
        indices = torch.nonzero(winners.ne(2), as_tuple=False).flatten()
        if indices.numel() == 0:
            return indices, torch.empty(0, dtype=torch.float32, device=self.boards.device)
        sample_episode_ids = self.episode_ids[indices].long()
        sample_winners = self.episode_results[sample_episode_ids].to(torch.float32)
        actor_outcomes = sample_winners * self.players[indices].to(torch.float32)
        if gamma == 1.0:
            return indices, actor_outcomes
        terminal_moves = self.episode_terminal_moves[sample_episode_ids].long()
        sample_move_counts = self.move_counts[indices].long()
        distance_after_action = (terminal_moves - sample_move_counts - 1).clamp_min(0)
        discounts = torch.pow(
            torch.full_like(distance_after_action, float(gamma), dtype=torch.float32),
            distance_after_action,
        )
        return indices, actor_outcomes * discounts


@dataclass(slots=True)
class PackedRolloutModels:
    conv_weights: list[torch.Tensor]
    norm_weights: list[torch.Tensor]
    norm_biases: list[torch.Tensor]
    policy_weight: torch.Tensor
    policy_bias: torch.Tensor
    value_weight: torch.Tensor
    value_bias: torch.Tensor
    num_models: int


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _validate_current_model(model: torch.nn.Module) -> torch.nn.Module:
    model = _unwrap_model(model)
    kernels = tuple(int(v) for v in getattr(model, "kernels", ()))
    channels = tuple(int(v) for v in getattr(model, "channels", ()))
    if kernels != EXPECTED_KERNELS or channels != EXPECTED_CHANNELS:
        raise RuntimeError(
            "Native rollout jest wyspecjalizowany dla kernels="
            f"{EXPECTED_KERNELS}, channels={EXPECTED_CHANNELS}; otrzymano {kernels}, {channels}."
        )
    if int(getattr(model, "input_channels", 0)) != 4:
        raise RuntimeError("Native rollout wymaga 4 kanałów wejściowych")
    if tuple(getattr(model, "norm_layers", ())) != NORM_LAYERS:
        raise RuntimeError(f"Native rollout V6 wymaga GroupNorm tylko na warstwach {NORM_LAYERS}")
    for layer, norm in enumerate(model.norms):
        if layer in NORM_LAYERS and not isinstance(norm, nn.GroupNorm):
            raise RuntimeError(f"Warstwa {layer} musi mieć GroupNorm")
        if layer not in NORM_LAYERS and not isinstance(norm, nn.Identity):
            raise RuntimeError(f"Warstwa {layer} nie może wykonywać GroupNorm")
    return model


def _validate_historical_checkpoint(cp: HistoricalCheckpoint) -> None:
    cfg = dict(cp.model_config)
    if int(cfg.get("architecture_version", 0)) != 6:
        raise RuntimeError(f"{cp.path.name}: native rollout przyjmuje tylko architecture_version=6")
    kernels = tuple(int(v) for v in cfg.get("kernels", EXPECTED_KERNELS))
    channels = tuple(int(v) for v in cfg.get("channels", EXPECTED_CHANNELS))
    if kernels != EXPECTED_KERNELS or channels != EXPECTED_CHANNELS:
        raise RuntimeError(f"{cp.path.name}: niezgodna architektura CNN V6")
    if int(cp.game_config.get("board_size", 0)) != BOARD:
        raise RuntimeError(f"{cp.path.name}: native rollout wymaga planszy 19x19")
    if int(cp.game_config.get("win_length", 6)) != 6:
        raise RuntimeError(f"{cp.path.name}: native rollout wymaga win_length=6")


def _copy_padded_conv(destination: torch.Tensor, destination_slice: slice, raw: torch.Tensor) -> None:
    out_channels = int(raw.shape[1])
    flat = raw.reshape(raw.shape[0], out_channels, -1)
    kreal = int(flat.shape[-1])
    destination[destination_slice, :, :kreal].copy_(flat)
    if kreal < destination.shape[-1]:
        destination[destination_slice, :, kreal:].zero_()


@torch.no_grad()
def pack_rollout_models(
    model: torch.nn.Module,
    historical: Sequence[HistoricalCheckpoint],
    device: torch.device,
    *,
    chunk_size: int = 16,
) -> PackedRolloutModels:
    if device.type != "cuda":
        raise ValueError("Native rollout packing wymaga CUDA")
    base = _validate_current_model(model)
    historical = list(historical)
    for cp in historical:
        _validate_historical_checkpoint(cp)

    num_models = 1 + len(historical)
    dtype = torch.float16
    conv_weights = [
        torch.zeros((num_models, out_c, kpad), dtype=dtype, device=device)
        for out_c, kpad in zip(EXPECTED_CHANNELS, PACKED_K)
    ]
    # Skipped norm layers keep identity affine values in the packed representation;
    # the collector does not launch GroupNorm for those layers.
    norm_weights = [torch.ones((num_models, out_c), dtype=dtype, device=device) for out_c in EXPECTED_CHANNELS]
    norm_biases = [torch.zeros((num_models, out_c), dtype=dtype, device=device) for out_c in EXPECTED_CHANNELS]
    policy_weight = torch.empty((num_models, 96), dtype=dtype, device=device)
    policy_bias = torch.empty(num_models, dtype=dtype, device=device)
    value_weight = torch.empty((num_models, 96), dtype=dtype, device=device)
    value_bias = torch.empty(num_models, dtype=dtype, device=device)

    for layer in range(8):
        current_w = base.convs[layer].weight.detach().reshape(1, EXPECTED_CHANNELS[layer], -1).to(device=device, dtype=dtype)
        _copy_padded_conv(conv_weights[layer], slice(0, 1), current_w)
        if layer in NORM_LAYERS:
            norm = base.norms[layer]
            assert isinstance(norm, nn.GroupNorm)
            norm_weights[layer][0].copy_(norm.weight.detach().to(device=device, dtype=dtype))
            norm_biases[layer][0].copy_(norm.bias.detach().to(device=device, dtype=dtype))

    policy_weight[0].copy_(base.policy_output.weight.detach().reshape(-1).to(device=device, dtype=dtype))
    policy_bias[0].copy_(base.policy_output.bias.detach().reshape(()).to(device=device, dtype=dtype))
    value_weight[0].copy_(base.value_output.weight.detach().reshape(-1).to(device=device, dtype=dtype))
    value_bias[0].copy_(base.value_output.bias.detach().reshape(()).to(device=device, dtype=dtype))

    chunk_size = max(1, int(chunk_size))
    for start in range(0, len(historical), chunk_size):
        chunk = historical[start : start + chunk_size]
        dst = slice(1 + start, 1 + start + len(chunk))
        states = [cp.model_state for cp in chunk]
        for layer in range(8):
            raw = torch.stack([state[f"convs.{layer}.weight"] for state in states], dim=0).to(
                device=device, dtype=dtype, non_blocking=True
            )
            _copy_padded_conv(conv_weights[layer], dst, raw)
            if layer in NORM_LAYERS:
                norm_weights[layer][dst].copy_(
                    torch.stack([state[f"norms.{layer}.weight"] for state in states], dim=0).to(
                        device=device, dtype=dtype, non_blocking=True
                    )
                )
                norm_biases[layer][dst].copy_(
                    torch.stack([state[f"norms.{layer}.bias"] for state in states], dim=0).to(
                        device=device, dtype=dtype, non_blocking=True
                    )
                )

        policy_weight[dst].copy_(torch.stack(
            [state["policy_output.weight"].reshape(-1) for state in states], dim=0
        ).to(device=device, dtype=dtype, non_blocking=True))
        policy_bias[dst].copy_(torch.stack(
            [state["policy_output.bias"].reshape(()) for state in states], dim=0
        ).to(device=device, dtype=dtype, non_blocking=True))
        value_weight[dst].copy_(torch.stack(
            [state["value_output.weight"].reshape(-1) for state in states], dim=0
        ).to(device=device, dtype=dtype, non_blocking=True))
        value_bias[dst].copy_(torch.stack(
            [state["value_output.bias"].reshape(()) for state in states], dim=0
        ).to(device=device, dtype=dtype, non_blocking=True))

    return PackedRolloutModels(
        conv_weights=conv_weights,
        norm_weights=norm_weights,
        norm_biases=norm_biases,
        policy_weight=policy_weight,
        policy_bias=policy_bias,
        value_weight=value_weight,
        value_bias=value_bias,
        num_models=num_models,
    )
