from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from .checkpoint import CheckpointManager, load_checkpoint
from .config import load_config
from .history import HistoricalCheckpointCache, load_random_historical_checkpoints
from .logger import TrainingLogger
from .model import build_model, mask_logits
from .native_rollout import NativeRolloutCollector, build_rollout_assignments
from .native_rollout_state import pack_rollout_models
from .train import _autocast_context, _learning_rate, _mean, _percentile, _temperature
from .utils import resolve_device, seed_everything
from .vector_env import canonical_network_input


def _validate_fraction(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} musi należeć do [0, 1]")
    return value


def train(config_path: str | Path) -> None:
    cfg = load_config(config_path)
    run_cfg = cfg["run"]
    game_cfg = cfg["game"]
    model_cfg = cfg["model"]
    tr = cfg["training"]

    if int(model_cfg.get("architecture_version", 0)) != 6:
        raise RuntimeError("GPU training wymaga architecture_version=6")

    seed = int(run_cfg.get("seed", 42))
    seed_everything(seed)
    device = resolve_device(str(tr.get("device", "cuda")))
    if device.type != "cuda":
        raise RuntimeError("GPU collector wymaga CUDA")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError(
            "Collector jest aktualnie strojony pod SM120/RTX 50; "
            f"wykryto {torch.cuda.get_device_capability(device)}"
        )
    torch.set_float32_matmul_precision("high")

    board_size = int(game_cfg.get("board_size", 19))
    win_length = int(game_cfg.get("win_length", 6))
    if board_size != 19 or win_length != 6:
        raise RuntimeError("Rollout wymaga Connect6 19x19, win_length=6")

    num_envs = int(tr.get("num_envs", 2048))
    completed_per_env = int(tr.get("completed_positions_per_update", 384))
    if num_envs <= 0 or completed_per_env <= 0:
        raise ValueError("num_envs i completed_positions_per_update muszą być > 0")
    completed_target = num_envs * completed_per_env
    minibatch_size = int(tr.get("minibatch_size", 1024))
    total_updates = int(tr.get("updates", 100000))
    if minibatch_size <= 0 or total_updates <= 0:
        raise ValueError("minibatch_size i updates muszą być > 0")

    historical_fraction = _validate_fraction("historical_fraction", tr.get("historical_fraction", 0.25))
    bot_fraction = _validate_fraction("bot_fraction", tr.get("bot_fraction", 0.25))
    bot_v1_fraction = _validate_fraction("bot_v1_fraction", tr.get("bot_v1_fraction", 0.50))
    if historical_fraction + bot_fraction > 1.0 + 1e-9:
        raise ValueError("historical_fraction + bot_fraction nie może przekroczyć 1")

    historical_models_per_update = max(0, int(tr.get("historical_models_per_update", 128)))
    historical_ram_cache_models = max(0, int(tr.get("historical_ram_cache_models", 1024)))
    random_black_opening_fraction = _validate_fraction(
        "random_black_opening_fraction", tr.get("random_black_opening_fraction", 0.0)
    )
    symmetry_augmentation = bool(tr.get("symmetry_augmentation", False))

    gamma = float(tr.get("gamma", 1.0))
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma musi należeć do (0, 1]")

    run_dir = Path(run_cfg.get("root_dir", "runs")) / str(run_cfg.get("name", "connect6_v6"))
    checkpoint_mgr = CheckpointManager(run_dir / "checkpoints")
    logger = TrainingLogger(run_dir)

    model = build_model(model_cfg, board_size).to(device)
    model = model.to(memory_format=torch.channels_last)
    base_lr = float(tr.get("learning_rate", 3e-4))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=float(tr.get("weight_decay", 1e-4)),
        eps=float(tr.get("adam_eps", 1e-5)),
        fused=True,
    )

    use_amp = bool(tr.get("amp", True))
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
        cp_cfg = payload.get("model_config", {})
        if int(cp_cfg.get("architecture_version", 0)) != 6:
            raise RuntimeError("Resume przyjmuje wyłącznie checkpoint CNN V6")
        model.load_state_dict(payload["model_state"])
        model.to(memory_format=torch.channels_last)
        optimizer.load_state_dict(payload["optimizer_state"])
        if payload.get("scaler_state") and scaler_enabled:
            scaler.load_state_dict(payload["scaler_state"])
        start_update = int(payload.get("update", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        print(f"[resume] {resume_path} -> update {start_update}, step {global_step}")
        del payload

    if bool(model_cfg.get("compile", False)):
        mode = str(model_cfg.get("compile_mode", "default"))
        print(f"[compile] torch.compile(mode={mode!r})")
        model = torch.compile(model, mode=mode)

    collector = NativeRolloutCollector(
        num_envs=num_envs,
        target_completed_positions=completed_target,
        device=device,
        verbose_build=bool(tr.get("native_compile_verbose", False)),
    )
    history_cache = HistoricalCheckpointCache(historical_ram_cache_models)

    clip_coef = float(tr.get("clip_coef", 0.2))
    value_coef = float(tr.get("value_coef", 0.5))
    entropy_coef = float(tr.get("entropy_coef", 0.01))
    max_grad_norm = float(tr.get("max_grad_norm", 1.0))
    ppo_epochs = int(tr.get("ppo_epochs", 4))
    target_kl = float(tr.get("target_kl", 0.03))
    kl_window = max(1, int(tr.get("kl_window", 32)))
    kl_hard_multiplier = max(1.0, float(tr.get("kl_hard_multiplier", 3.0)))
    kl_sync_interval = max(1, int(tr.get("kl_sync_interval", 32)))
    normalize_adv = bool(tr.get("normalize_advantage", True))
    checkpoint_every = max(1, int(tr.get("checkpoint_every_updates", 25)))
    dashboard_every = max(1, int(tr.get("dashboard_every_updates", 5)))
    lr_schedule = str(tr.get("lr_schedule", "cosine")).lower()

    games_total = 0
    history_games_total = 0
    bot_games_total = 0

    print(f"[urządzenie] {device} | {torch.cuda.get_device_name(device)}")
    print("[model] CNN V6 dense | GroupNorm layers=0,2,5,7 | channels_last=on")
    print("[optimizer] fused AdamW | foreach gradient clipping")
    print(f"[ppo] KL watchdog sync co {kl_sync_interval} minibatchy")
    print(
        f"[collector] envs={num_envs:,} | target={num_envs:,}*{completed_per_env:,}="
        f"{completed_target:,} pełnych pozycji"
    )
    print(
        f"[mix] self={1-historical_fraction-bot_fraction:.0%} | history={historical_fraction:.0%} | "
        f"bot={bot_fraction:.0%} (V1={bot_v1_fraction:.0%}, V2={1-bot_v1_fraction:.0%})"
    )
    print("[history] sampling with replacement: 50% all / 25% latest half / 25% latest quarter")
    print("[buffer] unfinished boards persist across PPO; unfinished trajectory prefixes are discarded")
    print(f"[run] {run_dir.resolve()}")

    try:
        for update in range(start_update, total_updates + 1):
            update_started = time.perf_counter()
            temperature = _temperature(update, tr)
            lr = _learning_rate(update, total_updates, base_lr, lr_schedule)
            for group in optimizer.param_groups:
                group["lr"] = lr

            torch.cuda.reset_peak_memory_stats(device)
            requested_history_tables = int(round(num_envs * historical_fraction))
            requested_models = min(historical_models_per_update, requested_history_tables)
            hits_before = history_cache.hits
            misses_before = history_cache.misses
            history_load_started = time.perf_counter()
            historical = load_random_historical_checkpoints(
                checkpoint_mgr.dir,
                current_update=update,
                requested_count=requested_models,
                required_model_config=model_cfg,
                required_game_config=game_cfg,
                ram_cache=history_cache,
            ) if requested_models else []
            history_load_seconds = time.perf_counter() - history_load_started
            history_hits = history_cache.hits - hits_before
            history_misses = history_cache.misses - misses_before

            pack_started = time.perf_counter()
            packed = pack_rollout_models(model, historical, device)
            pack_seconds = time.perf_counter() - pack_started
            historical_models_loaded = len(historical)

            assignments = build_rollout_assignments(
                num_envs,
                historical_model_count=historical_models_loaded,
                device=device,
                historical_fraction=historical_fraction,
                bot_fraction=bot_fraction,
                bot_v1_fraction=bot_v1_fraction,
            )

            model.eval()
            collection_started = time.perf_counter()
            rollout = collector.collect(
                packed,
                assignments,
                temperature=temperature,
                random_black_opening_fraction=random_black_opening_fraction,
                seed=seed + update * 0x9E3779B1,
                symmetry_augmentation=symmetry_augmentation,
            )
            collection_elapsed = max(1e-9, time.perf_counter() - collection_started)
            buffer = collector.buffer

            completed_idx, returns = buffer.completed_samples(gamma)
            train_size = int(completed_idx.numel())
            if train_size != rollout.completed_positions:
                raise RuntimeError(
                    "Collector accounting mismatch: "
                    f"counter={rollout.completed_positions}, samples={train_size}"
                )
            if train_size < completed_target:
                raise RuntimeError(f"Collector zatrzymał się za wcześnie: {train_size} < {completed_target}")

            generated_positions = rollout.generated_positions
            discarded_positions = generated_positions - train_size
            advantages = returns - buffer.values[completed_idx]
            if normalize_adv and advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            del packed, historical

            ppo_started = time.perf_counter()
            model.train()
            kls: list[float] = []
            clipfracs: list[float] = []
            stop_for_kl = False
            kl_stop_kind = "none"
            minibatches_per_epoch = max(1, math.ceil(train_size / minibatch_size))
            planned = ppo_epochs * minibatches_per_epoch
            checked = 0
            completed = 0
            synced_until = 0
            kl_rolling_last = 0.0
            ppo_stop_epoch = 0
            ppo_stop_minibatch = 0

            ppo_values = torch.empty((planned, 4), dtype=torch.float32, device=device)
            grad_norm_values = torch.empty(planned, dtype=torch.float32, device=device)
            grad_scale_values = torch.empty_like(grad_norm_values)
            grad_clipped_values = torch.empty_like(grad_norm_values)
            kl_values = torch.empty((planned, 2), dtype=torch.float32, device=device)

            def sync_kl_watchdog(current_epoch: int, current_mb: int) -> bool:
                nonlocal synced_until, kl_rolling_last, stop_for_kl, kl_stop_kind
                nonlocal ppo_stop_epoch, ppo_stop_minibatch
                if checked <= synced_until:
                    return False
                chunk = kl_values[synced_until:checked].cpu().tolist()
                base_index = synced_until
                synced_until = checked
                for offset, (approx_kl, clipfrac) in enumerate(chunk):
                    kls.append(float(approx_kl))
                    clipfracs.append(float(clipfrac))
                    kl_rolling_last = _mean(kls[-kl_window:])
                    hard_stop = target_kl > 0 and approx_kl > target_kl * kl_hard_multiplier
                    soft_stop = target_kl > 0 and len(kls) >= kl_window and kl_rolling_last > target_kl
                    if hard_stop or soft_stop:
                        stop_for_kl = True
                        kl_stop_kind = "hard" if hard_stop else "rolling"
                        absolute_checked = base_index + offset
                        ppo_stop_epoch = absolute_checked // minibatches_per_epoch + 1
                        ppo_stop_minibatch = absolute_checked % minibatches_per_epoch + 1
                        return True
                return False

            for epoch in range(ppo_epochs):
                order = torch.randperm(train_size, device=device)
                for start in range(0, train_size, minibatch_size):
                    local_idx = order[start : start + minibatch_size]
                    idx = completed_idx[local_idx]
                    mb_boards = buffer.boards[idx]
                    mb_players = buffer.players[idx]
                    mb_stones = buffer.stones_left[idx]
                    mb_actions = buffer.actions[idx].long()
                    mb_input = canonical_network_input(mb_boards, mb_players, mb_stones)
                    mb_input = mb_input.contiguous(memory_format=torch.channels_last)
                    mb_legal = mb_boards.reshape(mb_boards.shape[0], -1).eq(0)

                    with _autocast_context(device, use_amp, amp_dtype_name):
                        logits, values = model(mb_input)
                        logits = mask_logits(logits, mb_legal) / max(temperature, 1e-4)
                        dist = Categorical(logits=logits.float())
                        new_logprob = dist.log_prob(mb_actions)
                        entropy = dist.entropy().mean()
                        old_logprob = buffer.logprobs[idx]
                        logratio = new_logprob - old_logprob
                        ratio = logratio.exp()
                        mb_adv = advantages[local_idx]
                        pg1 = -mb_adv * ratio
                        pg2 = -mb_adv * torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
                        policy_loss = torch.maximum(pg1, pg2).mean()
                        value_loss = 0.5 * (values.float() - returns[local_idx]).pow(2).mean()
                        loss = policy_loss + value_coef * value_loss - entropy_coef * entropy

                    with torch.no_grad():
                        approx_kl_t = ((ratio - 1.0) - logratio).mean()
                        clipfrac_t = ((ratio - 1.0).abs() > clip_coef).float().mean()
                        kl_values[checked, 0] = approx_kl_t
                        kl_values[checked, 1] = clipfrac_t

                    checked += 1
                    end_of_epoch = start + minibatch_size >= train_size
                    should_sync = (checked - synced_until) >= kl_sync_interval or end_of_epoch
                    if should_sync and sync_kl_watchdog(epoch + 1, start // minibatch_size + 1):
                        optimizer.zero_grad(set_to_none=True)
                        break

                    optimizer.zero_grad(set_to_none=True)
                    if scaler_enabled:
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                    else:
                        loss.backward()

                    grad_norm = nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_grad_norm,
                        error_if_nonfinite=True,
                        foreach=True,
                    ).detach().float()
                    grad_scale = torch.clamp(
                        torch.as_tensor(max_grad_norm, device=device) / (grad_norm + 1e-6), max=1.0
                    )
                    slot = completed
                    grad_norm_values[slot] = grad_norm
                    grad_scale_values[slot] = grad_scale
                    grad_clipped_values[slot] = grad_norm.gt(max_grad_norm).float()
                    ppo_values[slot, 0] = loss.detach().float()
                    ppo_values[slot, 1] = policy_loss.detach().float()
                    ppo_values[slot, 2] = value_loss.detach().float()
                    ppo_values[slot, 3] = entropy.detach().float()

                    if scaler_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    completed += 1

                if stop_for_kl:
                    break

            sync_kl_watchdog(ppo_epochs, minibatches_per_epoch)

            if completed:
                vals = ppo_values[:completed]
                grads = grad_norm_values[:completed]
                summary = torch.stack((
                    vals[:, 0].mean(), vals[:, 1].mean(), vals[:, 2].mean(), vals[:, 3].mean(),
                    grads.mean(), torch.quantile(grads, 0.95), grads.max(),
                    grad_clipped_values[:completed].mean(), grad_scale_values[:completed].mean(),
                )).cpu().tolist()
            else:
                summary = [0.0] * 9
            (
                loss_mean, policy_loss_mean, value_loss_mean, entropy_mean,
                grad_norm_mean, grad_norm_p95, grad_norm_max,
                grad_clip_fraction, grad_scale_mean,
            ) = [float(v) for v in summary]
            ppo_elapsed = max(1e-9, time.perf_counter() - ppo_started)

            global_step += train_size
            games_total += rollout.games
            history_games_total += rollout.history_games
            bot_games_total += rollout.bot_games
            elapsed = max(1e-9, time.perf_counter() - update_started)
            game_denom = max(1, rollout.games)
            history_denom = max(1, rollout.history_games)
            bot_denom = max(1, rollout.bot_games)
            bot_v1_denom = max(1, rollout.bot_v1_games)
            bot_v2_denom = max(1, rollout.bot_v2_games)
            cache_requests = history_hits + history_misses

            gpu_allocated_gb = torch.cuda.memory_allocated(device) / 1e9
            gpu_reserved_gb = torch.cuda.memory_reserved(device) / 1e9
            gpu_peak_allocated_gb = torch.cuda.max_memory_allocated(device) / 1e9
            gpu_peak_reserved_gb = torch.cuda.max_memory_reserved(device) / 1e9

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
                "kl_sync_interval": kl_sync_interval,
                "ppo_early_stop": float(stop_for_kl),
                "ppo_kl_hard_stop": float(kl_stop_kind == "hard"),
                "ppo_kl_rolling_stop": float(kl_stop_kind == "rolling"),
                "ppo_epochs_target": ppo_epochs,
                "ppo_epochs_equivalent": completed / max(1, minibatches_per_epoch),
                "ppo_minibatches_checked": checked,
                "ppo_minibatches_completed": completed,
                "ppo_minibatches_possible": planned,
                "ppo_completion_fraction": completed / max(1, planned),
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
                "games_this_update": rollout.games,
                "black_win_rate": rollout.black_wins / game_denom,
                "white_win_rate": rollout.white_wins / game_denom,
                "draw_rate": rollout.draws / game_denom,
                "mean_game_length": rollout.game_length_sum / game_denom,
                "historical_fraction": historical_fraction,
                "historical_tables": assignments.history_tables,
                "historical_models_loaded": historical_models_loaded,
                "historical_games_completed": history_games_total,
                "historical_games_this_update": rollout.history_games,
                "historical_wins": rollout.history_wins,
                "historical_losses": rollout.history_losses,
                "historical_draws": rollout.history_draws,
                "historical_win_rate": rollout.history_wins / history_denom,
                "historical_score_rate": (rollout.history_wins + 0.5 * rollout.history_draws) / history_denom,
                "historical_ram_cache_models": len(history_cache),
                "historical_ram_cache_limit": historical_ram_cache_models,
                "historical_ram_cache_hits": history_hits,
                "historical_ram_cache_misses": history_misses,
                "historical_ram_cache_hit_rate": history_hits / cache_requests if cache_requests else 0.0,
                "history_load_seconds": history_load_seconds,
                "history_pack_seconds": pack_seconds,
                "bot_fraction": bot_fraction,
                "bot_tables": assignments.bot_tables,
                "bot_v1_tables": assignments.bot_v1_tables,
                "bot_v2_tables": assignments.bot_v2_tables,
                "bot_games_completed": bot_games_total,
                "bot_games_this_update": rollout.bot_games,
                "bot_wins": rollout.bot_wins,
                "bot_losses": rollout.bot_losses,
                "bot_draws": rollout.bot_draws,
                "bot_win_rate": rollout.bot_wins / bot_denom,
                "bot_score_rate": (rollout.bot_wins + 0.5 * rollout.bot_draws) / bot_denom,
                "bot_v1_games_this_update": rollout.bot_v1_games,
                "bot_v1_wins": rollout.bot_v1_wins,
                "bot_v1_losses": rollout.bot_v1_losses,
                "bot_v1_draws": rollout.bot_v1_draws,
                "bot_v1_win_rate": rollout.bot_v1_wins / bot_v1_denom,
                "bot_v1_score_rate": (rollout.bot_v1_wins + 0.5 * rollout.bot_v1_draws) / bot_v1_denom,
                "bot_v2_games_this_update": rollout.bot_v2_games,
                "bot_v2_wins": rollout.bot_v2_wins,
                "bot_v2_losses": rollout.bot_v2_losses,
                "bot_v2_draws": rollout.bot_v2_draws,
                "bot_v2_win_rate": rollout.bot_v2_wins / bot_v2_denom,
                "bot_v2_score_rate": (rollout.bot_v2_wins + 0.5 * rollout.bot_v2_draws) / bot_v2_denom,
                "completed_positions_this_update": train_size,
                "generated_positions_this_update": generated_positions,
                "discarded_positions_this_update": discarded_positions,
                "discard_fraction": discarded_positions / max(1, generated_positions),
                "collector_graph_steps": rollout.graph_steps,
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
            }

            logger.log(metrics, write_dashboard=(update % dashboard_every == 0))
            print(
                f"u={update:06d} step={global_step:,} loss={loss_mean:.4f} "
                f"ent={entropy_mean:.3f} KL={_mean(kls):.4f}/{metrics['approx_kl_max']:.4f} "
                f"PPO={metrics['ppo_completion_fraction']:.0%}"
                f"{' '+kl_stop_kind.upper()+'-STOP' if stop_for_kl else ''} "
                f"games={rollout.games:,} complete={train_size:,} discard={discarded_positions:,} "
                f"hist={assignments.history_tables}/{len(history_cache)} "
                f"bots={assignments.bot_v1_tables}+{assignments.bot_v2_tables} "
                f"V1={metrics['bot_v1_win_rate']:.1%} V2={metrics['bot_v2_win_rate']:.1%} "
                f"collect={metrics['selfplay_positions_per_second']:,.0f} pos/s "
                f"PPO={ppo_elapsed:.1f}s VRAM={gpu_allocated_gb:.2f}/{gpu_reserved_gb:.2f}GB"
            )

            if update % checkpoint_every == 0:
                path = checkpoint_mgr.save(
                    update=update,
                    model=model,
                    optimizer=optimizer,
                    config=cfg,
                    global_step=global_step,
                    scaler_state=scaler.state_dict() if scaler_enabled else None,
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
            scaler_state=scaler.state_dict() if scaler_enabled else None,
            extra={"final": True},
        )
        logger.write_dashboard()

    except KeyboardInterrupt:
        print("\n[przerwano] Zapisywanie checkpointu...")
        interrupted_update = update if "update" in locals() else max(1, start_update)
        checkpoint_mgr.save(
            update=interrupted_update,
            model=model,
            optimizer=optimizer,
            config=cfg,
            global_step=global_step,
            scaler_state=scaler.state_dict() if scaler_enabled else None,
            extra={"interrupted": True},
        )
        logger.write_dashboard()
        print("[przerwano] Zapisano.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect6 CNN V6 PPO with optimized GPU rollout")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
