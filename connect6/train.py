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
from .history import HistoricalModel, load_random_historical_models
from .logger import TrainingLogger
from .model import build_model, mask_logits
from .utils import resolve_device, seed_everything
from .vector_env import VectorConnect6, canonical_network_input


UNKNOWN_EPISODE_RESULT = 2  # poprawne wyniki: -1, 0, +1
UNKNOWN_TERMINAL_MOVE = -1


class CompleteGameBuffer:
    """GPU buffer containing only decisions made by the current policy.

    For current-v-current games every move is stored. For current-v-history games
    moves made by frozen historical opponents are deliberately not stored.
    """

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
        """Returns terminal samples and gamma-discounted terminal targets.

        Gamma is applied per placed stone/action, not per full two-stone turn.
        The terminal action itself has distance 0 and therefore multiplier 1.
        """

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
        returns = actor_outcomes * discounts
        return indices, returns


def _autocast_context(device: torch.device, enabled: bool, dtype_name: str):
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if dtype_name.lower() == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


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


def _historical_layout(
    num_envs: int,
    fraction: float,
    historical_models: list[HistoricalModel],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create fixed historical tables plus per-game opponent/color assignment."""

    fraction = min(1.0, max(0.0, float(fraction)))
    count = int(round(num_envs * fraction)) if historical_models else 0

    historical_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if count:
        historical_mask[:count] = True

    opponent_ids = torch.full((num_envs,), -1, dtype=torch.int16, device=device)
    current_colors = torch.zeros(num_envs, dtype=torch.int8, device=device)

    if count:
        idx = torch.arange(count, device=device)
        opponent_ids[idx] = torch.randint(
            0, len(historical_models), (count,), device=device, dtype=torch.int16
        )
        current_colors[idx] = torch.where(
            torch.rand(count, device=device) < 0.5,
            torch.ones(count, dtype=torch.int8, device=device),
            -torch.ones(count, dtype=torch.int8, device=device),
        )

    return historical_mask, opponent_ids, current_colors


def _reassign_historical_games(
    indices: torch.Tensor,
    opponent_ids: torch.Tensor,
    current_colors: torch.Tensor,
    historical_models: list[HistoricalModel],
) -> None:
    if indices.numel() == 0 or not historical_models:
        return
    device = indices.device
    n = int(indices.numel())
    opponent_ids[indices] = torch.randint(
        0, len(historical_models), (n,), device=device, dtype=torch.int16
    )
    current_colors[indices] = torch.where(
        torch.rand(n, device=device) < 0.5,
        torch.ones(n, dtype=torch.int8, device=device),
        -torch.ones(n, dtype=torch.int8, device=device),
    )


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

    historical_fraction = float(tr.get("historical_fraction", 0.0))
    if not 0.0 <= historical_fraction <= 1.0:
        raise ValueError("historical_fraction musi należeć do przedziału [0, 1]")
    historical_models_per_update = max(
        0, int(tr.get("historical_models_per_update", 0))
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

    # -------------------------------------------------------------------------
    # RESUME. Fresh runs start at update 1. Existing zero-based checkpoints are
    # still compatible: checkpoint 4099 resumes as update 4100.
    # -------------------------------------------------------------------------
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
            raise RuntimeError(
                f"Checkpoint {resume_path} używa innej architektury modelu."
            )

        model.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scaler_state") and scaler_enabled:
            scaler.load_state_dict(payload["scaler_state"])

        start_update = int(payload.get("update", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        print(
            f"[resume] {resume_path} -> update {start_update}, global_step {global_step}"
        )
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
    normalize_adv = bool(tr.get("normalize_advantage", True))
    checkpoint_every = max(1, int(tr.get("checkpoint_every_updates", 25)))
    dashboard_every = max(1, int(tr.get("dashboard_every_updates", 5)))
    lr_schedule = str(tr.get("lr_schedule", "cosine")).lower()

    if bool(tr.get("symmetry_augmentation", False)):
        print(
            "[uwaga] symmetry_augmentation jest ignorowane: post-hoc transformacja "
            "nie zachowuje old_logprob PPO."
        )

    games_total = 0
    historical_games_total = 0
    historical_models: list[HistoricalModel] = []

    print(f"[urządzenie] {device}")
    if device.type == "cuda":
        print(
            f"[gpu] {torch.cuda.get_device_name(device)} | CUDA runtime {torch.version.cuda}"
        )
    print(
        f"[środowisko] {num_envs} parallel boards | terminal target >= "
        f"{completed_target:,} current-policy positions/update"
    )
    print(
        f"[history] fraction={historical_fraction:.3f} | "
        f"models/update={historical_models_per_update}"
    )
    print(f"[credit] gamma={gamma:.6f} per stone/action")
    print(
        f"[model] parameters: {sum(p.numel() for p in getattr(model, '_orig_mod', model).parameters()):,}"
    )
    print(f"[run] {run_dir.resolve()}")

    try:
        for update in range(start_update, total_updates + 1):
            update_started = time.perf_counter()
            temperature = _temperature(update, tr)
            lr = _learning_rate(update, total_updates, base_lr, lr_schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # -----------------------------------------------------------------
            # Choose a new frozen historical pool once per PPO update.
            # Old pool references are dropped before new models are loaded.
            # -----------------------------------------------------------------
            historical_models = []
            if device.type == "cuda":
                # Frees tensors from the previous pool for reuse by the allocator.
                torch.cuda.empty_cache()

            if historical_fraction > 0.0 and historical_models_per_update > 0:
                historical_models = load_random_historical_models(
                    checkpoint_mgr.dir,
                    current_update=update,
                    requested_count=historical_models_per_update,
                    device=device,
                )

            historical_mask, historical_opponent_ids, historical_current_colors = (
                _historical_layout(
                    num_envs,
                    historical_fraction,
                    historical_models,
                    device,
                )
            )
            historical_idx = torch.nonzero(
                historical_mask, as_tuple=False
            ).flatten()

            # A history table must not silently switch old opponent in the middle
            # of a game when a new pool is sampled. Reset only those tables.
            if historical_idx.numel():
                env.reset(historical_idx)

            history_tables = int(historical_idx.numel())
            if historical_models:
                print(
                    f"[history] u={update:06d} loaded={len(historical_models)} "
                    f"tables={history_tables}"
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
            game_lengths: list[float] = []

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
                    network_input = env.network_input()
                    legal = env.legal_mask()

                    # Current policy owns all ordinary self-play moves and only
                    # its assigned color on historical tables.
                    current_actor_mask = ~historical_mask
                    if history_tables:
                        current_actor_mask = current_actor_mask | (
                            historical_mask
                            & env.current_player.eq(historical_current_colors)
                        )

                    actions = torch.empty(num_envs, dtype=torch.long, device=device)

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
                        actions[current_idx] = current_actions

                        # Store ONLY actions from the trainable current policy.
                        buffer.append_batch(
                            boards=env.boards[current_idx],
                            players=env.current_player[current_idx],
                            stones_left=env.stones_left[current_idx],
                            move_counts=env.move_count[current_idx],
                            actions=current_actions,
                            logprobs=current_logprobs,
                            values=current_values.float(),
                            episode_ids=current_episode_ids[current_idx],
                        )
                        current_segment_lengths[current_idx] += 1

                    # Frozen historical policies only act on their own assigned
                    # history tables. Their states/actions/logprobs never enter PPO.
                    for historical_id, historical in enumerate(historical_models):
                        old_actor_mask = (
                            historical_mask
                            & ~current_actor_mask
                            & historical_opponent_ids.eq(historical_id)
                        )
                        old_idx = torch.nonzero(
                            old_actor_mask, as_tuple=False
                        ).flatten()
                        if old_idx.numel() == 0:
                            continue

                        with _autocast_context(device, use_amp, amp_dtype_name):
                            old_logits, _ = historical.model(network_input[old_idx])
                        old_actions, _ = _sample_actions(
                            old_logits,
                            legal[old_idx],
                            temperature,
                        )
                        actions[old_idx] = old_actions

                    step = env.step(actions)
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
                        completed_positions += int(segment_lengths.sum().item())

                        update_games += int(done_idx.numel())
                        update_black += int(winners.eq(1).sum().item())
                        update_white += int(winners.eq(-1).sum().item())
                        update_draws += int(winners.eq(0).sum().item())
                        game_lengths.extend(full_game_lengths.float().cpu().tolist())

                        done_hist_mask = historical_mask[done_idx]
                        hist_done_idx = done_idx[done_hist_mask]
                        if hist_done_idx.numel():
                            hist_winners = step.winner[hist_done_idx]
                            hist_current_colors = historical_current_colors[hist_done_idx]
                            update_hist_games += int(hist_done_idx.numel())
                            update_hist_wins += int(
                                hist_winners.eq(hist_current_colors).sum().item()
                            )
                            update_hist_losses += int(
                                hist_winners.eq(-hist_current_colors).sum().item()
                            )
                            update_hist_draws += int(hist_winners.eq(0).sum().item())

                        env.reset(done_idx)
                        current_segment_lengths[done_idx] = 0

                        count_done = int(done_idx.numel())
                        new_ids = torch.arange(
                            next_episode_id,
                            next_episode_id + count_done,
                            dtype=torch.int32,
                            device=device,
                        )
                        current_episode_ids[done_idx] = new_ids
                        next_episode_id += count_done

                        # New history game: fresh old opponent and current color.
                        _reassign_historical_games(
                            hist_done_idx,
                            historical_opponent_ids,
                            historical_current_colors,
                            historical_models,
                        )

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            collection_elapsed = max(
                1e-9, time.perf_counter() - collection_started
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
            # PPO
            # -----------------------------------------------------------------
            model.train()
            losses: list[float] = []
            policy_losses: list[float] = []
            value_losses: list[float] = []
            entropies: list[float] = []
            kls: list[float] = []
            clipfracs: list[float] = []
            grad_norms: list[float] = []
            grad_clip_scales: list[float] = []
            grad_was_clipped: list[float] = []
            stop_for_kl = False

            minibatches_per_epoch = max(1, math.ceil(train_size / minibatch_size))
            planned_ppo_minibatches = ppo_epochs * minibatches_per_epoch
            completed_ppo_minibatches = 0
            ppo_stop_epoch = 0
            ppo_stop_minibatch = 0

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

                    optimizer.zero_grad(set_to_none=True)
                    if scaler_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()

                    grad_norm_tensor = nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm, error_if_nonfinite=True
                    )
                    grad_norm = float(grad_norm_tensor.detach().item())
                    grad_scale = min(
                        1.0, max_grad_norm / (grad_norm + 1e-6)
                    )
                    grad_norms.append(grad_norm)
                    grad_clip_scales.append(grad_scale)
                    grad_was_clipped.append(
                        1.0 if grad_norm > max_grad_norm else 0.0
                    )

                    if scaler_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    with torch.no_grad():
                        approx_kl = (((ratio - 1.0) - logratio).mean().item())
                        clipfrac = (
                            ((ratio - 1.0).abs() > clip_coef)
                            .float()
                            .mean()
                            .item()
                        )

                    losses.append(float(loss.item()))
                    policy_losses.append(float(policy_loss.item()))
                    value_losses.append(float(value_loss.item()))
                    entropies.append(float(entropy.item()))
                    kls.append(float(approx_kl))
                    clipfracs.append(float(clipfrac))
                    completed_ppo_minibatches += 1

                    if target_kl > 0 and approx_kl > target_kl:
                        stop_for_kl = True
                        ppo_stop_epoch = _epoch + 1
                        ppo_stop_minibatch = (start // minibatch_size) + 1
                        break
                if stop_for_kl:
                    break

            ppo_completion_fraction = completed_ppo_minibatches / max(
                1, planned_ppo_minibatches
            )
            ppo_epochs_equivalent = completed_ppo_minibatches / max(
                1, minibatches_per_epoch
            )

            # -----------------------------------------------------------------
            # Metrics
            # -----------------------------------------------------------------
            global_step += train_size
            games_total += update_games
            historical_games_total += update_hist_games
            elapsed = max(1e-9, time.perf_counter() - update_started)
            denom = max(1, update_games)
            hist_denom = max(1, update_hist_games)

            metrics = {
                "update": update,
                "global_step": global_step,
                "loss": _mean(losses),
                "policy_loss": _mean(policy_losses),
                "value_loss": _mean(value_losses),
                "entropy": _mean(entropies),
                "approx_kl": _mean(kls),
                "approx_kl_mean": _mean(kls),
                "approx_kl_p95": _percentile(kls, 0.95),
                "approx_kl_max": max(kls) if kls else 0.0,
                "approx_kl_last": kls[-1] if kls else 0.0,
                "target_kl": target_kl,
                "ppo_early_stop": 1.0 if stop_for_kl else 0.0,
                "ppo_epochs_target": ppo_epochs,
                "ppo_epochs_equivalent": ppo_epochs_equivalent,
                "ppo_minibatches_completed": completed_ppo_minibatches,
                "ppo_minibatches_possible": planned_ppo_minibatches,
                "ppo_completion_fraction": ppo_completion_fraction,
                "ppo_stop_epoch": ppo_stop_epoch,
                "ppo_stop_minibatch": ppo_stop_minibatch,
                "clip_fraction": _mean(clipfracs),
                "grad_norm_mean": _mean(grad_norms),
                "grad_norm_p95": _percentile(grad_norms, 0.95),
                "grad_norm_max": max(grad_norms) if grad_norms else 0.0,
                "grad_clip_fraction": _mean(grad_was_clipped),
                "grad_scale_mean": _mean(grad_clip_scales),
                "grad_limit": max_grad_norm,
                "learning_rate": lr,
                "temperature": temperature,
                "gamma": gamma,
                "games_completed": games_total,
                "games_this_update": update_games,
                "black_win_rate": update_black / denom,
                "white_win_rate": update_white / denom,
                "draw_rate": update_draws / denom,
                "mean_game_length": _mean(game_lengths),
                "historical_fraction": historical_fraction,
                "historical_tables": history_tables,
                "historical_models_loaded": len(historical_models),
                "historical_games_completed": historical_games_total,
                "historical_games_this_update": update_hist_games,
                "historical_wins": update_hist_wins,
                "historical_losses": update_hist_losses,
                "historical_draws": update_hist_draws,
                "historical_win_rate": update_hist_wins / hist_denom,
                "historical_score_rate": (
                    update_hist_wins + 0.5 * update_hist_draws
                ) / hist_denom,
                "completed_positions_this_update": train_size,
                "generated_positions_this_update": generated_positions,
                "discarded_positions_this_update": discarded_positions,
                "unfinished_games_discarded": unfinished_games_discarded,
                "discard_fraction": discarded_positions
                / max(1, generated_positions),
                "selfplay_positions_per_second": generated_positions
                / collection_elapsed,
                "positions_per_second": train_size / elapsed,
                "gpu_memory_gb": (
                    torch.cuda.max_memory_allocated(device) / 1e9
                    if device.type == "cuda"
                    else 0.0
                ),
            }

            logger.log(
                metrics,
                write_dashboard=(update % dashboard_every == 0),
            )

            print(
                f"u={update:06d} step={global_step:,} "
                f"loss={metrics['loss']:.4f} ent={metrics['entropy']:.3f} "
                f"KL={metrics['approx_kl_mean']:.4f}/"
                f"{metrics['approx_kl_max']:.4f} target={target_kl:.4f} "
                f"PPO={metrics['ppo_completion_fraction']:.0%}"
                f"{' EARLY-STOP' if stop_for_kl else ''} "
                f"games={update_games:4d} complete={train_size:,} "
                f"discard={discarded_positions:,} "
                f"B/W/D={metrics['black_win_rate']:.2f}/"
                f"{metrics['white_win_rate']:.2f}/{metrics['draw_rate']:.2f} "
                f"histWR={metrics['historical_win_rate']:.1%} "
                f"grad={metrics['grad_norm_mean']:.2f} "
                f"p95={metrics['grad_norm_p95']:.2f} "
                f"selfplay={metrics['selfplay_positions_per_second']:,.0f} pos/s "
                f"total={metrics['positions_per_second']:,.0f} train-pos/s"
            )

            # With one-based update numbering, every 10 means 10,20,30,...
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
        interrupted_update = (
            update if "update" in locals() else max(1, start_update)
        )
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
    """D4 transform retained for tests; not used by PPO collection."""
    n = boards.shape[-1]
    out = torch.rot90(boards, k=k, dims=(-2, -1)) if k else boards
    r = torch.div(actions, n, rounding_mode="floor")
    c = actions.remainder(n)
    if k == 1:
        r, c = n - 1 - c, r
    elif k == 2:
        r, c = n - 1 - r, n - 1 - c
    elif k == 3:
        r, c = c, n - 1 - r
    if flip:
        out = torch.flip(out, dims=(-1,))
        c = n - 1 - c
    return out, r * n + c


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Connect6 PPO: self-play + frozen historical opponents"
    )
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
