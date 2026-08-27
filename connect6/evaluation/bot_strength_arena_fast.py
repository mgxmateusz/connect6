from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import torch

from connect6.championship import bot_arena as model_arena
from connect6.championship import championship as legacy
from connect6.championship import native_championship as native_champ
from connect6.championship.championship_cnn import _black_to_move, _masked_step
from connect6.cuda_native.policy_loader import load_native_policy_extension
from connect6.engine.history import _load_lean_checkpoint
from connect6.engine.vector_env import VectorConnect6
from connect6.evaluation import bot_strength_arena as base


@torch.inference_mode()
def _pack_checkpoints_lean(refs, device: torch.device, *, chunk_size: int = 32):
    """Pack all V6 checkpoints once, using lean history cache when available."""
    n = len(refs)
    weights, norm_weights, norm_biases, policy = native_champ._allocate_packed_weights(n, device)
    family_cfg: dict[str, Any] | None = None

    print(f"[NATIVE PACK] {n:,} modeli -> FP16 WMMA layout (lean cache)")
    started = time.perf_counter()
    for start in range(0, n, chunk_size):
        end = min(n, start + chunk_size)
        cps = [_load_lean_checkpoint(ref.path) for ref in refs[start:end]]

        for ref, cp in zip(refs[start:end], cps):
            cfg = native_champ._validate_checkpoint_family(
                {"model_config": cp.model_config, "game_config": cp.game_config},
                ref,
                first_cfg=family_cfg,
            )
            if family_cfg is None:
                family_cfg = cfg

        for layer in range(8):
            raw = torch.stack([cp.model_state[f"convs.{layer}.weight"] for cp in cps], dim=0)
            raw = raw.to(device=device, dtype=torch.float16, non_blocking=False)
            raw = raw.reshape(end - start, native_champ.EXPECTED_CHANNELS[layer], -1)
            kreal = int(raw.shape[-1])
            kpad = native_champ.EXPECTED_KPAD[layer]
            if kreal > kpad:
                raise RuntimeError(f"Warstwa {layer}: K={kreal} > KPAD={kpad}")
            weights[layer][start:end, :, :kreal].copy_(raw)
            if kreal < kpad:
                weights[layer][start:end, :, kreal:].zero_()

            if layer in native_champ.NORM_LAYERS:
                nw = torch.stack([cp.model_state[f"norms.{layer}.weight"] for cp in cps], dim=0)
                nb = torch.stack([cp.model_state[f"norms.{layer}.bias"] for cp in cps], dim=0)
                norm_weights[layer][start:end].copy_(nw.to(device=device, dtype=torch.float16))
                norm_biases[layer][start:end].copy_(nb.to(device=device, dtype=torch.float16))

        pw = torch.stack([cp.model_state["policy_output.weight"] for cp in cps], dim=0)
        policy[start:end].copy_(pw.reshape(end - start, 96).to(device=device, dtype=torch.float16))
        del cps, raw, pw
        gc.collect()

        if end == n or end % max(128, chunk_size) == 0:
            print(
                f"[NATIVE PACK] {end:>5}/{n} | VRAM={torch.cuda.memory_allocated(device)/2**30:.2f} GB",
                flush=True,
            )

    torch.cuda.synchronize(device)
    print(f"[NATIVE PACK] gotowe w {time.perf_counter()-started:.2f}s")
    return weights, norm_weights, norm_biases, policy


class NativePolicyPool:
    def __init__(self, refs, device: torch.device, *, load_chunk: int = 32) -> None:
        self.refs = refs
        self.device = device
        self.extension = load_native_policy_extension(verbose=False)
        self.weights, self.norm_weights, self.norm_biases, self.policy = _pack_checkpoints_lean(
            refs, device, chunk_size=load_chunk
        )

    @torch.inference_mode()
    def actions(
        self,
        boards: torch.Tensor,
        current_player: torch.Tensor,
        stones_left: torch.Tensor,
        model_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.extension.policy_actions_dense(
            self.weights,
            self.norm_weights,
            self.norm_biases,
            self.policy,
            boards.contiguous(),
            current_player.contiguous(),
            stones_left.contiguous(),
            model_ids.contiguous(),
        )


@torch.inference_mode()
def _play_native_colour_game(
    pool: NativePolicyPool,
    bot,
    model_ids: torch.Tensor,
    *,
    model_is_black: bool,
    sync_interval_moves: int,
) -> tuple[list[int], list[int], float]:
    n = int(model_ids.numel())
    device = pool.device
    env = VectorConnect6(n, 19, 6, device=device, debug_checks=False)
    env.reset()
    base._reset_bot(bot)
    active = torch.ones(n, dtype=torch.bool, device=device)
    winners = torch.zeros(n, dtype=torch.int8, device=device)
    sync_every = max(1, int(sync_interval_moves))

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for move_index in range(env.action_size):
        model_to_move = _black_to_move(move_index) == bool(model_is_black)
        if model_to_move:
            actions = pool.actions(
                env.boards,
                env.current_player,
                env.stones_left,
                model_ids,
            )
        else:
            actions = bot.actions(env.boards, env.current_player, env.stones_left)

        done, winner = _masked_step(env, actions, active)
        newly_done = active & done
        winners = torch.where(newly_done, winner, winners)
        active &= ~done
        if (move_index + 1) % sync_every == 0 and not bool(active.any().item()):
            break

    if bool(active.any().item()):
        raise RuntimeError("Nie wszystkie partie native bot gauntlet zakończyły się po 361 kamieniach")
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    return (
        [int(v) for v in winners.cpu().tolist()],
        [int(v) for v in env.move_count.cpu().tolist()],
        elapsed,
    )


def _run_one_bot_native(
    spec,
    refs,
    pool: NativePolicyPool,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    batch_size: int,
    sync_interval: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"bot_matches_{spec.key}.csv"
    state_path = output_dir / f"state_{spec.key}.json"
    model_arena._validate_state(
        state_path,
        checkpoint_dir=checkpoint_dir,
        bot_signature=spec.signature + "_native_policy_sm120_v1",
    )

    completed = model_arena._completed_names(results_path)
    pending_indices = [i for i, ref in enumerate(refs) if ref.name not in completed]
    print("\n" + "=" * 78)
    print(f"{spec.label.upper()} — NATIVE ALL AUTOSAVES")
    print("=" * 78)
    print(
        f"Checkpointy={len(refs):,} | gotowe={len(completed):,} | "
        f"pozostało={len(pending_indices):,} | 2 gry/model | batch={batch_size}"
    )

    if not pending_indices:
        return model_arena._write_single_report(results_path, output_dir, spec)

    bot = spec.cls(pool.device)
    base._warm_bot(bot, pool.device)
    started_all = time.perf_counter()
    completed_this_run = 0

    for start in range(0, len(pending_indices), batch_size):
        ids = pending_indices[start : start + batch_size]
        chunk_refs = [refs[i] for i in ids]
        model_ids = torch.tensor(ids, dtype=torch.int32, device=pool.device)

        black_winners, black_moves, black_elapsed = _play_native_colour_game(
            pool,
            bot,
            model_ids,
            model_is_black=True,
            sync_interval_moves=sync_interval,
        )
        white_winners, white_moves, white_elapsed = _play_native_colour_game(
            pool,
            bot,
            model_ids,
            model_is_black=False,
            sync_interval_moves=sync_interval,
        )
        elapsed = black_elapsed + white_elapsed
        rows = model_arena._make_chunk_rows(
            chunk_refs,
            black_winners,
            black_moves,
            white_winners,
            white_moves,
            elapsed,
        )
        model_arena._append_rows(results_path, rows)
        completed_this_run += len(ids)

        games = 2 * len(ids)
        actual_done = len(completed) + completed_this_run
        print(
            f"[{100.0*actual_done/len(refs):6.2f}%] {actual_done:,}/{len(refs):,} modeli | "
            f"{games/max(elapsed,1e-9):,.1f} g/s | bot/model/draw="
            f"{sum(int(r['bot_wins']) for r in rows)}/"
            f"{sum(int(r['model_wins']) for r in rows)}/"
            f"{sum(int(r['draws']) for r in rows)} | play={elapsed:.3f}s",
            flush=True,
        )
        del model_ids, rows

    data = model_arena._write_single_report(results_path, output_dir, spec)
    model_arena._print_summary(spec, data[1])
    print(
        f"[DONE] {spec.label}: {completed_this_run:,} nowych modeli | "
        f"wall={time.perf_counter()-started_all:.2f}s"
    )
    return data


def _run_models_vs_bots_native(
    specs,
    *,
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    sync_interval: int,
    load_chunk: int,
) -> dict[str, Any]:
    refs = legacy.discover_checkpoints(checkpoint_dir)
    if not refs:
        raise RuntimeError(f"Nie znaleziono model_update_*.pt w {checkpoint_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n# ETAP 1 FAST: wszystkie checkpointy vs każdy bot — NATIVE SM120")
    print(f"Modele={len(refs):,} | boty={','.join(s.key for s in specs)} | 2 gry/model/bot")
    print("CNN: native FP16 WMMA + GroupNorm; bez HistoricalPolicyEnsemble/PyTorch forward")
    pool = NativePolicyPool(refs, device, load_chunk=load_chunk)

    summaries: dict[str, Any] = {}
    for spec in specs:
        _, summary = _run_one_bot_native(
            spec,
            refs,
            pool,
            checkpoint_dir=checkpoint_dir,
            output_dir=output_dir,
            batch_size=batch_size,
            sync_interval=sync_interval,
        )
        summaries[spec.key] = summary
    return summaries


def run(
    config_path: str | Path,
    *,
    reset: bool = False,
    skip_models: bool = False,
    skip_bot_league: bool = False,
    skip_cloudict: bool = False,
) -> None:
    config_path = Path(config_path).resolve()
    cfg = legacy._read_yaml(config_path)
    arena = cfg.get("bot_strength_arena", cfg)
    root = base._project_root(config_path)

    if not torch.cuda.is_available():
        raise RuntimeError("Ten test wymaga CUDA")
    device = torch.device(arena.get("device", "cuda"))
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise RuntimeError("Fast arena wymaga SM120 / RTX 50")

    requested = [str(x).lower() for x in arena.get("bots", ["v1", "v2", "v3", "v4", "v5"])]
    unknown = sorted(set(requested) - set(base.BOT_BY_KEY))
    if unknown:
        raise ValueError(f"Nieznane boty: {unknown}")
    specs = [base.BOT_BY_KEY[key] for key in requested]

    checkpoint_dir = base._resolve(root, arena["checkpoint_dir"])
    output_dir = base._resolve(root, arena["output_dir"])
    if reset and output_dir.exists():
        # Preserve nothing when user explicitly requests a clean rerun.
        import shutil
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = max(1, int(arena.get("models_per_batch", 512)))
    sync_interval = max(1, int(arena.get("sync_interval_moves", 16)))
    load_chunk = max(1, int(arena.get("native_load_chunk_models", 32)))

    pair_cfg = arena.get("bot_vs_bot", {})
    opening_size = int(pair_cfg.get("opening_size", 19))
    ccfg = arena.get("cloudict", {})
    executable = base._resolve(root, ccfg.get("executable", str(base.cloudict.DEFAULT_CLOUDICT_EXE)))
    depths = [int(x) for x in ccfg.get("depths", [2, 3, 4])]
    raw_sizes = ccfg.get("opening_sizes", {})
    defaults = {2: 19, 3: 9, 4: 5}
    opening_sizes = {d: int(raw_sizes.get(d, raw_sizes.get(str(d), defaults.get(d, 3)))) for d in depths}
    vcf = bool(ccfg.get("vcf", False))
    timeout = float(ccfg.get("timeout_seconds", 60.0))
    max_restarts = int(ccfg.get("max_restarts", 20))

    print("=" * 78)
    print("CONNECT6 BOT STRENGTH ARENA — FAST NATIVE MODEL GAUNTLET")
    print("=" * 78)
    print(f"Boty: {', '.join(s.key.upper() for s in specs)}")
    print(f"Wyniki: {output_dir}")

    # Separate native directory prevents mixing previous BF16/PyTorch results
    # with FP16/WMMA decisions if the old slow arena was already partially run.
    model_dir = output_dir / "models_vs_bots_native"
    if skip_models:
        model_summaries = {
            s.key: json.loads((model_dir / f"summary_{s.key}.json").read_text(encoding="utf-8"))
            for s in specs
            if (model_dir / f"summary_{s.key}.json").exists()
        }
    else:
        model_summaries = _run_models_vs_bots_native(
            specs,
            checkpoint_dir=checkpoint_dir,
            output_dir=model_dir,
            device=device,
            batch_size=batch_size,
            sync_interval=sync_interval,
            load_chunk=load_chunk,
        )

    pair_rows = (
        base._rows(output_dir / "bot_vs_bot.csv")
        if skip_bot_league
        else base._run_bot_round_robin(
            specs,
            output_dir=output_dir,
            device=device,
            opening_size=opening_size,
            sync_interval=sync_interval,
        )
    )
    cloud_rows = (
        base._rows(output_dir / "bot_vs_cloudict_games.csv")
        if skip_cloudict
        else base._run_cloudict(
            specs,
            output_dir=output_dir,
            device=device,
            executable=executable,
            depths=depths,
            opening_sizes=opening_sizes,
            vcf=vcf,
            timeout=timeout,
            max_restarts=max_restarts,
        )
    )
    base._write_summary(output_dir, specs, model_summaries, pair_rows, cloud_rows, depths)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fast native all-checkpoints vs bots + bot round-robin + Cloudict D2/D3/D4"
    )
    parser.add_argument("--config", default="configs/bot_strength_arena.yaml")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-bot-league", action="store_true")
    parser.add_argument("--skip-cloudict", action="store_true")
    args = parser.parse_args()
    run(
        args.config,
        reset=args.reset,
        skip_models=args.skip_models,
        skip_bot_league=args.skip_bot_league,
        skip_cloudict=args.skip_cloudict,
    )


if __name__ == "__main__":
    main()
