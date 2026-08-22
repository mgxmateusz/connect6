from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from .checkpoint import CheckpointManager, load_checkpoint
from .config import load_config
from .history import (
    HistoricalCheckpointCache,
    HistoricalPolicyEnsemble,
    load_random_historical_checkpoints,
)
from .logger import TrainingLogger
from .model import build_model, mask_logits
from .utils import resolve_device, seed_everything
from .vector_env import VectorConnect6, canonical_network_input


UNKNOWN_EPISODE_RESULT = 2
UNKNOWN_TERMINAL_MOVE = -1


class CompleteGameBuffer:
    """GPU buffer containing only decisions made by the current policy."""

    def __init__(
        self,
        target_completed_positions: int,
        envs: int,
        board_size: int,
        device: torch.device,
    ) -> None:
        self.target_completed_positions = int(target_completed_positions)
        self.envs = int(envs)
        self.board_size = int(board_size)
        self.max_game_positions = self.board_size * self.board_size
        self.capacity = (
            self.target_completed_positions
            + self.envs * self.max_game_positions
            + self.envs
        )

        self.boards = torch.empty(
            (self.capacity, board_size, board_size), dtype=torch.int8, device=device
        )
        self.players = torch.empty(self.capacity, dtype=torch.int8, device=device)
        self.stones_left = torch.empty(self.capacity, dtype=torch.int8, device=device)
        self.move_counts = torch.empty(self.capacity, dtype=torch.int16, device=device)
        self.actions = torch.empty(self.capacity, dtype=torch.int16, device=device)
        self.logprobs = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.values = torch.empty(self.capacity, dtype=torch.float32, device=device)
        self.episode_ids = torch.empty(self.capacity, dtype=torch.int32, device=device)
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def append_batch(
        self,
        boards: torch.Tensor,
        players: torch.Tensor,
        stones_left: torch.Tensor,
        move_counts: torch.Tensor,
        actions: torch.Tensor,
        logprobs: torch.Tensor,
        values: torch.Tensor,
        episode_ids: torch.Tensor,
    ) -> None:
        batch = int(boards.shape[0])
        start = self.count
        end = start + batch
        if end > self.capacity:
            raise RuntimeError(
                "Przepełnienie CompleteGameBuffer. Zwiększ bufor albo sprawdź collector."
            )

        self.boards[start:end].copy_(boards)
        self.players[start:end].copy_(players)
        self.stones_left[start:end].copy_(stones_left)
        self.move_counts[start:end].copy_(move_counts.to(torch.int16))
        self.actions[start:end].copy_(actions.to(torch.int16))
        self.logprobs[start:end].copy_(logprobs)
        self.values[start:end].copy_(values)
        self.episode_ids[start:end].copy_(episode_ids)
        self.count = end

    def completed_samples(
        self,
        episode_results: torch.Tensor,
        episode_terminal_moves: torch.Tensor,
        gamma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        used_episode_ids = self.episode_ids[: self.count].long()
        winners = episode_results[used_episode_ids]
        complete_mask = winners.ne(UNKNOWN_EPISODE_RESULT)
        indices = torch.nonzero(complete_mask, as_tuple=False).flatten()

        if indices.numel() == 0:
            return indices, torch.empty(0, dtype=torch.float32, device=self.boards.device)

        sample_episode_ids = self.episode_ids[indices].long()
        sample_winners = episode_results[sample_episode_ids].to(torch.float32)
        actor_outcomes = sample_winners * self.players[indices].to(torch.float32)

        terminal_moves = episode_terminal_moves[sample_episode_ids].long()
        sample_move_counts = self.move_counts[indices].long()
        distance_after_action = (terminal_moves - sample_move_counts - 1).clamp_min(0)

        if gamma == 1.0:
            return indices, actor_outcomes

        discounts = torch.pow(
            torch.full_like(distance_after_action, float(gamma), dtype=torch.float32),
            distance_after_action,
        )
        return indices, actor_outcomes * discounts


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if dtype_name.lower() == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _history_dtype(device: torch.device, dtype_name: str) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    name = dtype_name.lower()
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(
        "historical_inference_dtype musi być jednym z: bfloat16 | float16 | float32"
    )


def _schedule_progress(update: int, total_updates: int) -> float:
    if total_updates <= 1:
        return 1.0
    return min(1.0, max(0.0, (update - 1) / (total_updates - 1)))


def _temperature(update: int, cfg: dict[str, Any]) -> float:
    start = float(cfg.get("temperature_start", 1.0))
    end = float(cfg.get("temperature_end", 0.25))
    decay = max(1, int(cfg.get("temperature_decay_updates", 5000)))
    progress = min(1.0, max(0.0, (update - 1) / decay))
    return start + (end - start) * progress


def _learning_rate(
    update: int,
    total_updates: int,
    base_lr: float,
    schedule: str,
) -> float:
    if schedule == "constant":
        return base_lr
    if schedule == "cosine":
        progress = _schedule_progress(update, total_updates)
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Nieznany lr_schedule: {schedule}")


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = percentile * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _sample_actions(
    logits: torch.Tensor,
    legal: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = mask_logits(logits.float(), legal) / max(temperature, 1e-4)
    dist = Categorical(logits=logits)
    actions = dist.sample()
    return actions, dist.log_prob(actions)


def _sample_actions_only(
    logits: torch.Tensor,
    legal: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Sample frozen-opponent moves without computing unused log-probabilities."""
    logits = mask_logits(logits.float(), legal) / max(temperature, 1e-4)
    return Categorical(logits=logits).sample()


def _symmetry_for_phase(phase: int) -> tuple[int, bool]:
    """Eight-step D4 cycle: 0/90/180/270, mirror, then 4 rotations mirrored."""
    phase = int(phase) % 8
    return phase % 4, phase >= 4


def _transform_boards(
    boards: torch.Tensor,
    k: int,
    flip: bool,
) -> torch.Tensor:
    """Transform board coordinates exactly as seen by a policy."""
    k = int(k) % 4
    out = torch.rot90(boards, k=k, dims=(-2, -1)) if k else boards
    return torch.flip(out, dims=(-1,)) if flip else out


def _transform_actions(
    actions: torch.Tensor,
    board_size: int,
    k: int,
    flip: bool,
) -> torch.Tensor:
    """Map canonical actions into coordinates of `_transform_boards`."""
    n = int(board_size)
    k = int(k) % 4
    r = torch.div(actions, n, rounding_mode="floor")
    c = actions.remainder(n)
    if k == 1:
        r, c = n - 1 - c, r
    elif k == 2:
        r, c = n - 1 - r, n - 1 - c
    elif k == 3:
        r, c = c, n - 1 - r
    if flip:
        c = n - 1 - c
    return r * n + c


def _inverse_transform_actions(
    actions: torch.Tensor,
    board_size: int,
    k: int,
    flip: bool,
) -> torch.Tensor:
    """Map policy-view actions back into canonical environment coordinates."""
    n = int(board_size)
    k = int(k) % 4
    r = torch.div(actions, n, rounding_mode="floor")
    c = actions.remainder(n)

    # Forward transform is rotate first, mirror second, therefore inverse must
    # undo the mirror first and then apply the inverse rotation.
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
    """Select fresh black openings that must be random and excluded from PPO."""
    fresh_black = (
        move_count.eq(0)
        & current_player.eq(1)
        & stones_left.eq(1)
    )
    fraction = float(fraction)
    if fraction <= 0.0:
        return torch.zeros_like(fresh_black)
    if fraction >= 1.0:
        return fresh_black
    return fresh_black & torch.rand(
        move_count.shape,
        device=move_count.device,
    ).lt(fraction)


def _model_count(models_or_count: Any) -> int:
    if isinstance(models_or_count, int):
        return max(0, models_or_count)
    return len(models_or_count)


def _historical_layout(
    num_envs: int,
    fraction: float,
    historical_models: Any,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed table->opponent assignment for one PPO update.

    The percentage remains the user-facing control. Once the model pool is known,
    historical tables are split as evenly as possible between models and each
    table keeps the same opponent checkpoint until the next PPO update.
    """

    fraction = min(1.0, max(0.0, float(fraction)))
    model_count = _model_count(historical_models)
    count = int(round(num_envs * fraction)) if model_count else 0
    if count and model_count > count:
        model_count = count

    historical_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    opponent_ids = torch.full((num_envs,), -1, dtype=torch.int16, device=device)
    current_colors = torch.zeros(num_envs, dtype=torch.int8, device=device)

    if count == 0:
        return historical_mask, opponent_ids, current_colors

    historical_mask[:count] = True
    base = count // model_count
    remainder = count % model_count
    counts = torch.full((model_count,), base, dtype=torch.long, device=device)
    if remainder:
        counts[:remainder] += 1

    ids = torch.repeat_interleave(
        torch.arange(model_count, dtype=torch.int16, device=device), counts
    )
    opponent_ids[:count] = ids

    # Each opponent is balanced as closely as mathematically possible. If a model
    # gets an odd number of tables, the extra color alternates between models so
    # the entire historical pool is also globally balanced to <= 1 table.
    offset = 0
    for model_id in range(model_count):
        n = int(counts[model_id].item())
        if n % 2 == 0:
            black_count = n // 2
        elif model_id % 2 == 0:
            black_count = n // 2 + 1
        else:
            black_count = n // 2
        block = torch.empty(n, dtype=torch.int8, device=device)
        block[:black_count] = 1
        block[black_count:] = -1
        block = block[torch.randperm(n, device=device)]
        current_colors[offset : offset + n] = block
        offset += n

    return historical_mask, opponent_ids, current_colors


def _historical_table_matrix(
    opponent_ids: torch.Tensor,
    model_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build [models, max_tables] table indices once per update."""

    if model_count <= 0:
        empty = torch.empty((0, 0), dtype=torch.long, device=opponent_ids.device)
        return empty, torch.empty((0, 0), dtype=torch.bool, device=opponent_ids.device)

    groups = [
        torch.nonzero(opponent_ids.eq(i), as_tuple=False).flatten()
        for i in range(model_count)
    ]
    max_tables = max(int(group.numel()) for group in groups)
    matrix = torch.empty(
        (model_count, max_tables), dtype=torch.long, device=opponent_ids.device
    )
    valid = torch.zeros(
        (model_count, max_tables), dtype=torch.bool, device=opponent_ids.device
    )

    for i, group in enumerate(groups):
        n = int(group.numel())
        matrix[i, :n] = group
        valid[i, :n] = True
        if n < max_tables:
            matrix[i, n:] = group[0]

    return matrix, valid


def train(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    run_cfg = cfg["run"]
    game_cfg = cfg["game"]
    model_cfg = cfg["model"]
    tr = cfg["training"]

    seed_everything(int(run_cfg.get("seed", 42)))
    device = resolve_device(str(tr.get("device", "cuda")))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    run_dir = Path(run_cfg.get("root_dir", "runs")) / str(
        run_cfg.get("name", "connect6")
    )
    checkpoint_mgr = CheckpointManager(run_dir / "checkpoints")
    logger = TrainingLogger(run_dir)

    board_size = int(game_cfg.get("board_size", 19))
    win_length = int(game_cfg.get("win_length", 6))
    num_envs = int(tr.get("num_envs", 1024))
    minibatch_size = int(tr.get("minibatch_size", 2048))
    completed_target = int(tr.get("completed_positions_per_update", 256)) * minibatch_size
    total_updates = int(tr.get("updates", 100000))

    if completed_target <= 0:
        raise ValueError("completed_positions_per_update musi być większe od 0")
    if total_updates <= 0:
        raise ValueError("updates musi być większe od 0")

    gamma = float(tr.get("gamma", 1.0))
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma musi należeć do przedziału (0, 1]")

    random_black_opening_fraction = float(
        tr.get("random_black_opening_fraction", 0.0)
    )
    if not 0.0 <= random_black_opening_fraction <= 1.0:
        raise ValueError(
            "random_black_opening_fraction musi należeć do przedziału [0, 1]"
        )
    symmetry_augmentation = bool(tr.get("symmetry_augmentation", False))

    historical_fraction = float(tr.get("historical_fraction", 0.0))
    if not 0.0 <= historical_fraction <= 1.0:
        raise ValueError("historical_fraction musi należeć do przedziału [0, 1]")
    historical_models_per_update = max(
        0, int(tr.get("historical_models_per_update", 0))
    )
    historical_ram_cache_models = max(
        0, int(tr.get("historical_ram_cache_models", 256))
    )
    historical_inference_dtype_name = str(
        tr.get("historical_inference_dtype", "bfloat16")
    )
    historical_inference_dtype = _history_dtype(
        device, historical_inference_dtype_name
    )

    cuda_cache_clear_every_percent = float(
        tr.get("cuda_cache_clear_every_percent", 1.0)
    )
    if cuda_cache_clear_every_percent < 0.0:
        raise ValueError("cuda_cache_clear_every_percent nie może być ujemne")
    cuda_cache_clear_interval = (
        max(
            1,
            int(round(total_updates * cuda_cache_clear_every_percent / 100.0)),
        )
        if cuda_cache_clear_every_percent > 0.0
        else 0
    )

    model = build_model(model_cfg, board_size).to(device)
    base_lr = float(tr.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        eps=float(tr.get("adam_eps", 1e-5)),
    )

    use_amp = bool(tr.get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(tr.get("amp_dtype", "bfloat16"))
    scaler_enabled = use_amp and amp_dtype_name.lower() == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)

    start_update = 1
    global_step = 0
    resume_mode = str(run_cfg.get("resume", "auto"))
    resume_path: Path | None = None

    if resume_mode.lower() == "auto":
        resume_path = checkpoint_mgr.find_latest()
    elif resume_mode.lower() not in ("none", "off", "false", ""):
        resume_path = Path(resume_mode)

    if resume_path and resume_path.exists():
        payload = load_checkpoint(resume_path, map_location="cpu")
        checkpoint_model_cfg = payload.get("model_config", {})
        checkpoint_version = int(checkpoint_model_cfg.get("architecture_version", 1))
        current_version = int(model_cfg.get("architecture_version", 3))
        if checkpoint_version != current_version:
            raise RuntimeError(f"Checkpoint {resume_path} używa innej architektury modelu.")

        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scaler_state") and scaler_enabled:
            scaler.load_state_dict(payload["scaler_state"])
        start_update = int(payload.get("update", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        print(f"[resume] {resume_path} -> update {start_update}, global_step {global_step}")
        del payload

    if bool(model_cfg.get("compile", False)):
        compile_mode = str(model_cfg.get("compile_mode", "default"))
        print(f"[compile] torch.compile(mode={compile_mode!r})")
        model = torch.compile(model, mode=compile_mode)

    env = VectorConnect6(
        num_envs=num_envs,
        board_size=board_size,
        win_length=win_length,
        device=device,
        debug_checks=bool(game_cfg.get("debug_checks", False)),
    )
    buffer = CompleteGameBuffer(completed_target, num_envs, board_size, device)

    clip_coef = float(tr.get("clip_coef", 0.2))
    value_coef = float(tr.get("value_coef", 0.5))
    entropy_coef = float(tr.get("entropy_coef", 0.01))
    max_grad_norm = float(tr.get("max_grad_norm", 1.0))
    ppo_epochs = int(tr.get("ppo_epochs", 4))
    target_kl = float(tr.get("target_kl", 0.03))
    kl_window = max(1, int(tr.get("kl_window", 32)))
    kl_hard_multiplier = max(1.0, float(tr.get("kl_hard_multiplier", 3.0)))
    normalize_adv = bool(tr.get("normalize_advantage", True))
    checkpoint_every = max(1, int(tr.get("checkpoint_every_updates", 25)))
    dashboard_every = max(1, int(tr.get("dashboard_every_updates", 5)))
    lr_schedule = str(tr.get("lr_schedule", "cosine")).lower()

    games_total = 0
    historical_games_total = 0
    historical_ensemble: HistoricalPolicyEnsemble | None = None
    history_ram_cache = HistoricalCheckpointCache(historical_ram_cache_models)
    symmetry_phase = 0

    print(f"[urządzenie] {device}")
    if device.type == "cuda":
        print(f"[gpu] {torch.cuda.get_device_name(device)} | CUDA runtime {torch.version.cuda}")
    print(
        f"[środowisko] {num_envs} parallel boards | terminal target >= "
        f"{completed_target:,} current-policy positions/update"
    )
    print(
        f"[history] fraction={historical_fraction:.3f} | "
        f"models/update={historical_models_per_update} | grouped GPU ensemble | "
        f"dtype={historical_inference_dtype_name} | RAM cache={historical_ram_cache_models}"
    )
    if device.type == "cuda" and cuda_cache_clear_interval:
        print(
            f"[cuda-cache] empty_cache co {cuda_cache_clear_every_percent:g}% treningu "
            f"(~{cuda_cache_clear_interval:,} update'ów)"
        )
    print(
        f"[PPO] minibatch={minibatch_size:,} | target_kl={target_kl:.4f} | "
        f"rolling_window={kl_window} | hard={kl_hard_multiplier:.1f}x"
    )
    print(f"[credit] gamma={gamma:.6f} per stone/action")
    print(
        f"[opening] random black opening={random_black_opening_fraction:.0%} | "
        "forced move is never stored in PPO"
    )
    print(
        "[symmetry] online D4 cycle: "
        + ("0/90/180/270 + mirror rotations" if symmetry_augmentation else "off")
    )
    print(
        f"[model] parameters: "
        f"{sum(p.numel() for p in getattr(model, '_orig_mod', model).parameters()):,}"
    )
    print(f"[run] {run_dir.resolve()}")

    try:
        for update in range(start_update, total_updates + 1):
            update_started = time.perf_counter()
            temperature = _temperature(update, tr)
            lr = _learning_rate(update, total_updates, base_lr, lr_schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Release previous live ensemble. CUDA's caching allocator may keep the
            # freed blocks reserved for reuse; a controlled empty_cache below makes
            # that visible memory return to the driver only at rare milestones.
            historical_ensemble = None
            cuda_cache_cleared = 0.0
            cuda_cache_clear_seconds = 0.0
            if (
                device.type == "cuda"
                and cuda_cache_clear_interval > 0
                and update % cuda_cache_clear_interval == 0
            ):
                clear_started = time.perf_counter()
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
                cuda_cache_clear_seconds = time.perf_counter() - clear_started
                cuda_cache_cleared = 1.0

            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            requested_history_tables = int(round(num_envs * historical_fraction))
            requested_models = min(
                historical_models_per_update,
                requested_history_tables,
            )

            cache_hits_before = history_ram_cache.hits
            cache_misses_before = history_ram_cache.misses
            history_load_started = time.perf_counter()
            historical_checkpoints = []
            if requested_models > 0:
                historical_checkpoints = load_random_historical_checkpoints(
                    checkpoint_mgr.dir,
                    current_update=update,
                    requested_count=requested_models,
                    ram_cache=history_ram_cache,
                )
            history_load_seconds = time.perf_counter() - history_load_started
            history_cache_hits = history_ram_cache.hits - cache_hits_before
            history_cache_misses = history_ram_cache.misses - cache_misses_before

            history_build_started = time.perf_counter()
            if historical_checkpoints:
                historical_ensemble = HistoricalPolicyEnsemble(
                    historical_checkpoints,
                    device=device,
                    dtype=historical_inference_dtype,
                )
            history_build_seconds = time.perf_counter() - history_build_started

            historical_model_count = (
                historical_ensemble.num_models if historical_ensemble is not None else 0
            )
            historical_mask, historical_opponent_ids, historical_current_colors = (
                _historical_layout(
                    num_envs,
                    historical_fraction,
                    historical_model_count,
                    device,
                )
            )
            history_table_matrix, history_table_valid = _historical_table_matrix(
                historical_opponent_ids,
                historical_model_count,
            )
            historical_idx = torch.nonzero(historical_mask, as_tuple=False).flatten()
            if historical_idx.numel():
                env.reset(historical_idx)

            history_tables = int(historical_idx.numel())
            history_valid_flat_positions = torch.empty(
                0, dtype=torch.long, device=device
            )
            history_flat_tables = torch.empty(0, dtype=torch.long, device=device)
            if historical_model_count:
                history_valid_flat_positions = torch.nonzero(
                    history_table_valid.reshape(-1), as_tuple=False
                ).flatten()
                history_flat_tables = history_table_matrix.reshape(-1)[
                    history_valid_flat_positions
                ]

            history_tables_per_model_min = 0
            history_tables_per_model_max = 0
            if historical_ensemble is not None:
                tables_per_model = history_table_valid.sum(dim=1)
                history_tables_per_model_min = int(tables_per_model.min().item())
                history_tables_per_model_max = int(tables_per_model.max().item())
                print(
                    f"[history] u={update:06d} loaded={historical_model_count} "
                    f"tables={history_tables} per_model="
                    f"{history_tables_per_model_min}-{history_tables_per_model_max} "
                    f"ram={len(history_ram_cache)}/{historical_ram_cache_models} "
                    f"hit/miss={history_cache_hits}/{history_cache_misses}"
                )

            buffer.reset()
            model.eval()

            update_games = 0
            update_black = 0
            update_white = 0
            update_draws = 0
            update_hist_games = 0
            update_hist_wins = 0
            update_hist_losses = 0
            update_hist_draws = 0
            completed_positions = 0
            game_length_sum = 0

            current_segment_lengths = torch.zeros(
                num_envs, dtype=torch.int32, device=device
            )
            current_episode_ids = torch.arange(
                num_envs, dtype=torch.int32, device=device
            )
            next_episode_id = num_envs

            episode_results = torch.full(
                (buffer.capacity + num_envs + 1,),
                UNKNOWN_EPISODE_RESULT,
                dtype=torch.int8,
                device=device,
            )
            episode_terminal_moves = torch.full(
                (buffer.capacity + num_envs + 1,),
                UNKNOWN_TERMINAL_MOVE,
                dtype=torch.int16,
                device=device,
            )

            collection_started = time.perf_counter()

            with torch.inference_mode():
                while completed_positions < completed_target:
                    # ONLINE augmentation: the policy really sees this transformed
                    # board. Its sampled action is kept in the same transformed
                    # coordinates in PPO, then mapped back only for env.step().
                    if symmetry_augmentation:
                        symmetry_k, symmetry_flip = _symmetry_for_phase(
                            symmetry_phase
                        )
                        view_boards = _transform_boards(
                            env.boards,
                            symmetry_k,
                            symmetry_flip,
                        )
                    else:
                        symmetry_k, symmetry_flip = 0, False
                        view_boards = env.boards

                    network_input = canonical_network_input(
                        view_boards,
                        env.current_player,
                        env.stones_left,
                    )
                    legal = view_boards.reshape(num_envs, -1).eq(0)

                    # Exactly once per fresh game we make an independent Bernoulli
                    # choice. Forced black openings use a uniform board action and
                    # are excluded from current_idx, so they can never enter PPO.
                    forced_opening_mask = _forced_random_opening_mask(
                        env.move_count,
                        env.current_player,
                        env.stones_left,
                        random_black_opening_fraction,
                    )

                    current_actor_mask = ~historical_mask
                    if history_tables:
                        current_actor_mask = current_actor_mask | (
                            historical_mask
                            & env.current_player.eq(historical_current_colors)
                        )
                    current_actor_mask = current_actor_mask & ~forced_opening_mask

                    if random_black_opening_fraction > 0.0:
                        # Random values remain active only on forced fresh openings.
                        # Every other table is overwritten below by current/history.
                        actions_view = torch.randint(
                            0,
                            board_size * board_size,
                            (num_envs,),
                            dtype=torch.long,
                            device=device,
                        )
                    else:
                        actions_view = torch.empty(
                            num_envs,
                            dtype=torch.long,
                            device=device,
                        )

                    # This dynamic compaction is intentionally retained for safety:
                    # only actual current-policy decisions may enter PPO.
                    current_idx = torch.nonzero(
                        current_actor_mask, as_tuple=False
                    ).flatten()
                    if current_idx.numel():
                        with _autocast_context(device, use_amp, amp_dtype_name):
                            current_logits, current_values = model(
                                network_input[current_idx]
                            )
                        current_actions, current_logprobs = _sample_actions(
                            current_logits,
                            legal[current_idx],
                            temperature,
                        )
                        actions_view[current_idx] = current_actions
                        buffer.append_batch(
                            boards=view_boards[current_idx],
                            players=env.current_player[current_idx],
                            stones_left=env.stones_left[current_idx],
                            move_counts=env.move_count[current_idx],
                            actions=current_actions,
                            logprobs=current_logprobs,
                            values=current_values.float(),
                            episode_ids=current_episode_ids[current_idx],
                        )
                        current_segment_lengths[current_idx] += 1

                    # History sees the exact same transformed coordinates as current.
                    # Forced opening rows are deliberately not overwritten by either
                    # policy, regardless of which side owns Black on that table.
                    if historical_ensemble is not None and history_tables:
                        grouped_input = network_input[history_table_matrix]
                        grouped_legal = legal[history_table_matrix]
                        with _autocast_context(device, use_amp, amp_dtype_name):
                            grouped_logits = historical_ensemble.forward_grouped(
                                grouped_input
                            )
                        grouped_actions = _sample_actions_only(
                            grouped_logits,
                            grouped_legal,
                            temperature,
                        )

                        grouped_players = env.current_player[history_table_matrix]
                        grouped_current_colors = historical_current_colors[
                            history_table_matrix
                        ]
                        grouped_forced_opening = forced_opening_mask[
                            history_table_matrix
                        ]
                        old_turn = (
                            history_table_valid
                            & grouped_players.ne(grouped_current_colors)
                            & ~grouped_forced_opening
                        )

                        valid_old_turn = old_turn.reshape(-1)[
                            history_valid_flat_positions
                        ]
                        valid_history_actions = grouped_actions.reshape(-1)[
                            history_valid_flat_positions
                        ]
                        actions_view[history_flat_tables] = torch.where(
                            valid_old_turn,
                            valid_history_actions,
                            actions_view[history_flat_tables],
                        )

                    if symmetry_augmentation:
                        env_actions = _inverse_transform_actions(
                            actions_view,
                            board_size,
                            symmetry_k,
                            symmetry_flip,
                        )
                    else:
                        env_actions = actions_view

                    step = env.step(env_actions)
                    if symmetry_augmentation:
                        symmetry_phase = (symmetry_phase + 1) % 8

                    # `done_idx` is retained because reset/episode IDs are variable
                    # length state. We avoid adding many EXTRA synchronizations on
                    # top of this required compaction.
                    done_idx = torch.nonzero(step.done, as_tuple=False).flatten()
                    if done_idx.numel():
                        done_episode_ids = current_episode_ids[done_idx].long()
                        winners = step.winner[done_idx]
                        full_game_lengths = step.game_lengths[done_idx].long()
                        segment_lengths = current_segment_lengths[done_idx].long()

                        episode_results[done_episode_ids] = winners
                        episode_terminal_moves[done_episode_ids] = full_game_lengths.to(
                            torch.int16
                        )

                        done_hist_mask = historical_mask[done_idx]
                        done_hist_colors = historical_current_colors[done_idx]

                        # One small GPU->CPU transfer replaces many .item() calls
                        # plus the former full game_lengths.cpu().tolist().
                        batch_stats = torch.stack(
                            (
                                segment_lengths.sum(),
                                winners.eq(1).sum(),
                                winners.eq(-1).sum(),
                                winners.eq(0).sum(),
                                full_game_lengths.sum(),
                                done_hist_mask.sum(),
                                (done_hist_mask & winners.eq(done_hist_colors)).sum(),
                                (done_hist_mask & winners.eq(-done_hist_colors)).sum(),
                                (done_hist_mask & winners.eq(0)).sum(),
                            )
                        ).to(torch.int64)
                        (
                            completed_add,
                            black_add,
                            white_add,
                            draw_add,
                            game_length_add,
                            hist_games_add,
                            hist_wins_add,
                            hist_losses_add,
                            hist_draws_add,
                        ) = [int(v) for v in batch_stats.cpu().tolist()]

                        completed_positions += completed_add
                        count_done = int(done_idx.numel())
                        update_games += count_done
                        update_black += black_add
                        update_white += white_add
                        update_draws += draw_add
                        game_length_sum += game_length_add
                        update_hist_games += hist_games_add
                        update_hist_wins += hist_wins_add
                        update_hist_losses += hist_losses_add
                        update_hist_draws += hist_draws_add

                        env.reset(done_idx)
                        current_segment_lengths[done_idx] = 0

                        new_ids = torch.arange(
                            next_episode_id,
                            next_episode_id + count_done,
                            dtype=torch.int32,
                            device=device,
                        )
                        current_episode_ids[done_idx] = new_ids
                        next_episode_id += count_done

                        # Fixed table keeps the same opponent for the entire update.
                        # Only finished historical games flip current color.
                        historical_current_colors[done_idx] = torch.where(
                            done_hist_mask,
                            -done_hist_colors,
                            done_hist_colors,
                        )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            collection_elapsed = max(
                1e-9, time.perf_counter() - collection_started
            )

            # Release the last collector outputs before the much larger PPO pass.
            del (
                network_input,
                legal,
                actions_view,
                env_actions,
                view_boards,
                forced_opening_mask,
                current_actor_mask,
                current_idx,
            )
            if history_tables:
                del (
                    grouped_input,
                    grouped_legal,
                    grouped_logits,
                    grouped_actions,
                    grouped_players,
                    grouped_current_colors,
                    grouped_forced_opening,
                    old_turn,
                    valid_old_turn,
                    valid_history_actions,
                )

            completed_idx, returns = buffer.completed_samples(
                episode_results,
                episode_terminal_moves,
                gamma,
            )
            train_size = int(completed_idx.numel())
            generated_positions = int(buffer.count)
            discarded_positions = generated_positions - train_size

            if train_size != completed_positions:
                raise RuntimeError(
                    "Collector accounting mismatch: "
                    f"terminal current-policy positions={completed_positions}, "
                    f"completed samples={train_size}."
                )

            unfinished_games_discarded = int(
                current_segment_lengths.gt(0).sum().item()
            )

            advantages = returns - buffer.values[completed_idx]
            if normalize_adv and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

            # -----------------------------------------------------------------
            # PPO with KL watchdog BEFORE backward / optimizer.step.
            # -----------------------------------------------------------------
            ppo_started = time.perf_counter()
            model.train()
            kls: list[float] = []
            clipfracs: list[float] = []

            stop_for_kl = False
            kl_stop_kind = "none"
            minibatches_per_epoch = max(1, math.ceil(train_size / minibatch_size))
            planned_ppo_minibatches = ppo_epochs * minibatches_per_epoch
            checked_ppo_minibatches = 0
            completed_ppo_minibatches = 0
            ppo_stop_epoch = 0
            ppo_stop_minibatch = 0
            kl_rolling_last = 0.0

            ppo_values = torch.empty(
                (planned_ppo_minibatches, 4),
                dtype=torch.float32,
                device=device,
            )
            grad_norm_values = torch.empty(
                planned_ppo_minibatches, dtype=torch.float32, device=device
            )
            grad_scale_values = torch.empty_like(grad_norm_values)
            grad_clipped_values = torch.empty_like(grad_norm_values)

            for _epoch in range(ppo_epochs):
                order = torch.randperm(train_size, device=device)
                for start in range(0, train_size, minibatch_size):
                    local_idx = order[start : start + minibatch_size]
                    idx = completed_idx[local_idx]

                    mb_boards = buffer.boards[idx]
                    mb_players = buffer.players[idx]
                    mb_stones = buffer.stones_left[idx]
                    mb_actions = buffer.actions[idx].long()
                    mb_input = canonical_network_input(
                        mb_boards, mb_players, mb_stones
                    )
                    mb_legal = mb_boards.reshape(mb_boards.shape[0], -1).eq(0)

                    with _autocast_context(device, use_amp, amp_dtype_name):
                        new_logits, new_values = model(mb_input)
                        new_logits = mask_logits(new_logits, mb_legal) / max(
                            temperature, 1e-4
                        )
                        dist = Categorical(logits=new_logits.float())
                        new_logprob = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()

                        old_logprob = buffer.logprobs[idx]
                        logratio = new_logprob - old_logprob
                        ratio = logratio.exp()
                        mb_adv = advantages[local_idx]
                        pg1 = -mb_adv * ratio
                        pg2 = -mb_adv * torch.clamp(
                            ratio, 1.0 - clip_coef, 1.0 + clip_coef
                        )
                        policy_loss = torch.maximum(pg1, pg2).mean()

                        value_pred = new_values.float()
                        value_loss = 0.5 * (
                            value_pred - returns[local_idx]
                        ).pow(2).mean()
                        loss = (
                            policy_loss
                            + value_coef * value_loss
                            - entropy_coef * entropy
                        )

                    with torch.no_grad():
                        approx_kl_tensor = ((ratio - 1.0) - logratio).mean()
                        clipfrac_tensor = (
                            ((ratio - 1.0).abs() > clip_coef)
                            .float()
                            .mean()
                        )
                        # One required sync for the KL watchdog also transfers
                        # clipfrac; no second .item() synchronization.
                        kl_clip_cpu = torch.stack(
                            (approx_kl_tensor, clipfrac_tensor)
                        ).cpu()
                        approx_kl = float(kl_clip_cpu[0])
                        clipfrac = float(kl_clip_cpu[1])

                    checked_ppo_minibatches += 1
                    kls.append(approx_kl)
                    clipfracs.append(clipfrac)
                    kl_rolling_last = _mean(kls[-kl_window:])

                    hard_limit = target_kl * kl_hard_multiplier
                    hard_stop = target_kl > 0 and approx_kl > hard_limit
                    soft_stop = (
                        target_kl > 0
                        and len(kls) >= kl_window
                        and kl_rolling_last > target_kl
                    )
                    if hard_stop or soft_stop:
                        stop_for_kl = True
                        kl_stop_kind = "hard" if hard_stop else "rolling"
                        ppo_stop_epoch = _epoch + 1
                        ppo_stop_minibatch = (start // minibatch_size) + 1
                        optimizer.zero_grad(set_to_none=True)
                        break

                    optimizer.zero_grad(set_to_none=True)
                    if scaler_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()

                    grad_norm_tensor = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm, error_if_nonfinite=True
                    ).detach().float()
                    grad_scale_tensor = torch.clamp(
                        torch.as_tensor(
                            max_grad_norm,
                            dtype=torch.float32,
                            device=device,
                        )
                        / (grad_norm_tensor + 1e-6),
                        max=1.0,
                    )
                    slot = completed_ppo_minibatches
                    grad_norm_values[slot] = grad_norm_tensor
                    grad_scale_values[slot] = grad_scale_tensor
                    grad_clipped_values[slot] = grad_norm_tensor.gt(
                        max_grad_norm
                    ).float()
                    ppo_values[slot, 0] = loss.detach().float()
                    ppo_values[slot, 1] = policy_loss.detach().float()
                    ppo_values[slot, 2] = value_loss.detach().float()
                    ppo_values[slot, 3] = entropy.detach().float()

                    if scaler_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    completed_ppo_minibatches += 1

                if stop_for_kl:
                    break

            ppo_completion_fraction = completed_ppo_minibatches / max(
                1, planned_ppo_minibatches
            )
            ppo_epochs_equivalent = completed_ppo_minibatches / max(
                1, minibatches_per_epoch
            )

            if completed_ppo_minibatches:
                applied_values = ppo_values[:completed_ppo_minibatches]
                applied_grads = grad_norm_values[:completed_ppo_minibatches]
                applied_scales = grad_scale_values[:completed_ppo_minibatches]
                applied_clipped = grad_clipped_values[:completed_ppo_minibatches]
                ppo_summary = torch.stack(
                    (
                        applied_values[:, 0].mean(),
                        applied_values[:, 1].mean(),
                        applied_values[:, 2].mean(),
                        applied_values[:, 3].mean(),
                        applied_grads.mean(),
                        torch.quantile(applied_grads, 0.95),
                        applied_grads.max(),
                        applied_clipped.mean(),
                        applied_scales.mean(),
                    )
                ).cpu().tolist()
            else:
                ppo_summary = [0.0] * 9

            (
                loss_mean,
                policy_loss_mean,
                value_loss_mean,
                entropy_mean,
                grad_norm_mean,
                grad_norm_p95,
                grad_norm_max,
                grad_clip_fraction,
                grad_scale_mean,
            ) = [float(v) for v in ppo_summary]
            ppo_elapsed = max(1e-9, time.perf_counter() - ppo_started)

            global_step += train_size
            games_total += update_games
            historical_games_total += update_hist_games
            elapsed = max(1e-9, time.perf_counter() - update_started)
            denom = max(1, update_games)
            hist_denom = max(1, update_hist_games)

            if device.type == "cuda":
                gpu_allocated_gb = torch.cuda.memory_allocated(device) / 1e9
                gpu_reserved_gb = torch.cuda.memory_reserved(device) / 1e9
                gpu_peak_allocated_gb = (
                    torch.cuda.max_memory_allocated(device) / 1e9
                )
                gpu_peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9
            else:
                gpu_allocated_gb = 0.0
                gpu_reserved_gb = 0.0
                gpu_peak_allocated_gb = 0.0
                gpu_peak_reserved_gb = 0.0

            cache_requests = history_cache_hits + history_cache_misses
            history_cache_hit_rate = (
                history_cache_hits / cache_requests if cache_requests else 0.0
            )

            metrics = {
                "update": update,
                "global_step": global_step,
                "loss": loss_mean,
                "policy_loss": policy_loss_mean,
                "value_loss": value_loss_mean,
                "entropy": entropy_mean,
                "approx_kl": _mean(kls),
                "approx_kl_mean": _mean(kls),
                "approx_kl_p95": _percentile(kls, 0.95),
                "approx_kl_max": max(kls) if kls else 0.0,
                "approx_kl_last": kls[-1] if kls else 0.0,
                "approx_kl_rolling_last": kl_rolling_last,
                "target_kl": target_kl,
                "kl_window": kl_window,
                "kl_hard_limit": target_kl * kl_hard_multiplier,
                "ppo_early_stop": 1.0 if stop_for_kl else 0.0,
                "ppo_kl_hard_stop": 1.0 if kl_stop_kind == "hard" else 0.0,
                "ppo_kl_rolling_stop": 1.0 if kl_stop_kind == "rolling" else 0.0,
                "ppo_epochs_target": ppo_epochs,
                "ppo_epochs_equivalent": ppo_epochs_equivalent,
                "ppo_minibatches_checked": checked_ppo_minibatches,
                "ppo_minibatches_completed": completed_ppo_minibatches,
                "ppo_minibatches_possible": planned_ppo_minibatches,
                "ppo_completion_fraction": ppo_completion_fraction,
                "ppo_stop_epoch": ppo_stop_epoch,
                "ppo_stop_minibatch": ppo_stop_minibatch,
                "clip_fraction": _mean(clipfracs),
                "grad_norm_mean": grad_norm_mean,
                "grad_norm_p95": grad_norm_p95,
                "grad_norm_max": grad_norm_max,
                "grad_clip_fraction": grad_clip_fraction,
                "grad_scale_mean": grad_scale_mean,
                "grad_limit": max_grad_norm,
                "learning_rate": lr,
                "temperature": temperature,
                "gamma": gamma,
                "games_completed": games_total,
                "games_this_update": update_games,
                "black_win_rate": update_black / denom,
                "white_win_rate": update_white / denom,
                "draw_rate": update_draws / denom,
                "mean_game_length": game_length_sum / denom,
                "historical_fraction": historical_fraction,
                "historical_tables": history_tables,
                "historical_models_loaded": historical_model_count,
                "historical_tables_per_model_min": history_tables_per_model_min,
                "historical_tables_per_model_max": history_tables_per_model_max,
                "historical_games_completed": historical_games_total,
                "historical_games_this_update": update_hist_games,
                "historical_wins": update_hist_wins,
                "historical_losses": update_hist_losses,
                "historical_draws": update_hist_draws,
                "historical_win_rate": update_hist_wins / hist_denom,
                "historical_score_rate": (
                    update_hist_wins + 0.5 * update_hist_draws
                ) / hist_denom,
                "historical_ram_cache_models": len(history_ram_cache),
                "historical_ram_cache_limit": historical_ram_cache_models,
                "historical_ram_cache_hits": history_cache_hits,
                "historical_ram_cache_misses": history_cache_misses,
                "historical_ram_cache_hit_rate": history_cache_hit_rate,
                "history_load_seconds": history_load_seconds,
                "history_build_seconds": history_build_seconds,
                "completed_positions_this_update": train_size,
                "generated_positions_this_update": generated_positions,
                "discarded_positions_this_update": discarded_positions,
                "unfinished_games_discarded": unfinished_games_discarded,
                "discard_fraction": discarded_positions / max(1, generated_positions),
                "collector_seconds": collection_elapsed,
                "ppo_seconds": ppo_elapsed,
                "update_seconds": elapsed,
                "selfplay_positions_per_second": generated_positions / collection_elapsed,
                "positions_per_second": train_size / elapsed,
                "gpu_memory_gb": gpu_peak_allocated_gb,
                "gpu_allocated_gb": gpu_allocated_gb,
                "gpu_reserved_gb": gpu_reserved_gb,
                "gpu_peak_allocated_gb": gpu_peak_allocated_gb,
                "gpu_peak_reserved_gb": gpu_peak_reserved_gb,
                "cuda_cache_cleared": cuda_cache_cleared,
                "cuda_cache_clear_seconds": cuda_cache_clear_seconds,
            }

            logger.log(
                metrics,
                write_dashboard=(update % dashboard_every == 0),
            )

            print(
                f"u={update:06d} step={global_step:,} "
                f"loss={metrics['loss']:.4f} ent={metrics['entropy']:.3f} "
                f"KL={metrics['approx_kl_mean']:.4f}/"
                f"{metrics['approx_kl_max']:.4f} roll={kl_rolling_last:.4f} "
                f"PPO={metrics['ppo_completion_fraction']:.0%}"
                f"{' ' + kl_stop_kind.upper() + '-STOP' if stop_for_kl else ''} "
                f"games={update_games:4d} complete={train_size:,} "
                f"discard={discarded_positions:,} "
                f"B/W/D={metrics['black_win_rate']:.2f}/"
                f"{metrics['white_win_rate']:.2f}/{metrics['draw_rate']:.2f} "
                f"histWR={metrics['historical_win_rate']:.1%} "
                f"hist={historical_model_count}x{history_tables_per_model_min} "
                f"selfplay={metrics['selfplay_positions_per_second']:,.0f} pos/s "
                f"VRAM={gpu_allocated_gb:.2f}/{gpu_reserved_gb:.2f}GB "
                f"total={metrics['positions_per_second']:,.0f} train-pos/s"
            )

            if update % checkpoint_every == 0:
                path = checkpoint_mgr.save(
                    update=update,
                    model=model,
                    optimizer=optimizer,
                    config=cfg,
                    global_step=global_step,
                    scaler_state=(scaler.state_dict() if scaler_enabled else None),
                    extra={"metrics": metrics},
                )
                print(f"[checkpoint] {path.name}")

            # Make the historical weights eligible for allocator reuse immediately
            # after the update instead of retaining a live reference until the next
            # iteration. RAM LRU copies remain intentionally cached on the CPU.
            historical_ensemble = None
            historical_checkpoints = []

        final_update = max(start_update, total_updates)
        checkpoint_mgr.save(
            update=final_update,
            model=model,
            optimizer=optimizer,
            config=cfg,
            global_step=global_step,
            scaler_state=(scaler.state_dict() if scaler_enabled else None),
            extra={"final": True},
        )
        logger.write_dashboard()

    except KeyboardInterrupt:
        print("\n[przerwano] Zapisywanie najnowszego checkpointu...")
        interrupted_update = update if "update" in locals() else max(1, start_update)
        checkpoint_mgr.save(
            update=interrupted_update,
            model=model,
            optimizer=optimizer,
            config=cfg,
            global_step=global_step,
            scaler_state=(scaler.state_dict() if scaler_enabled else None),
            extra={"interrupted": True},
        )
        logger.write_dashboard()
        print("[przerwano] Zapisano.")


def _transform_board_actions(
    boards: torch.Tensor,
    actions: torch.Tensor,
    k: int,
    flip: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply one D4 transform consistently to board and action coordinates."""
    return (
        _transform_boards(boards, k, flip),
        _transform_actions(actions, boards.shape[-1], k, flip),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6 PPO: self-play + grouped frozen historical opponents"
    )
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
