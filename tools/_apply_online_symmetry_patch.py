from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected exactly 1 match, got {count}")
    return text.replace(old, new, 1)


train_path = Path("connect6/train.py")
text = train_path.read_text(encoding="utf-8")

old = '''def _sample_actions_only(
    logits: torch.Tensor,
    legal: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Sample frozen-opponent moves without computing unused log-probabilities."""
    logits = mask_logits(logits.float(), legal) / max(temperature, 1e-4)
    return Categorical(logits=logits).sample()


def _model_count(models_or_count: Any) -> int:
'''
new = '''def _sample_actions_only(
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
'''
text = replace_once(text, old, new, "insert symmetry helpers")

old = '''    gamma = float(tr.get("gamma", 1.0))
    if not 0.0 < gamma <= 1.0:
        raise ValueError("gamma musi należeć do przedziału (0, 1]")

    historical_fraction = float(tr.get("historical_fraction", 0.0))
'''
new = '''    gamma = float(tr.get("gamma", 1.0))
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
'''
text = replace_once(text, old, new, "parse opening/symmetry config")

old = '''    if bool(tr.get("symmetry_augmentation", False)):
        print(
            "[uwaga] symmetry_augmentation jest ignorowane: post-hoc transformacja "
            "nie zachowuje old_logprob PPO."
        )

    games_total = 0
'''
new = '''    games_total = 0
'''
text = replace_once(text, old, new, "remove obsolete symmetry warning")

old = '''    historical_ensemble: HistoricalPolicyEnsemble | None = None
    history_ram_cache = HistoricalCheckpointCache(historical_ram_cache_models)

    print(f"[urządzenie] {device}")
'''
new = '''    historical_ensemble: HistoricalPolicyEnsemble | None = None
    history_ram_cache = HistoricalCheckpointCache(historical_ram_cache_models)
    symmetry_phase = 0

    print(f"[urządzenie] {device}")
'''
text = replace_once(text, old, new, "add persistent symmetry phase")

old = '''    print(f"[credit] gamma={gamma:.6f} per stone/action")
    print(
        f"[model] parameters: "
'''
new = '''    print(f"[credit] gamma={gamma:.6f} per stone/action")
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
'''
text = replace_once(text, old, new, "add opening/symmetry startup diagnostics")

old = '''            with torch.inference_mode():
                while completed_positions < completed_target:
                    network_input = env.network_input()
                    legal = env.legal_mask()

                    current_actor_mask = ~historical_mask
                    if history_tables:
                        current_actor_mask = current_actor_mask | (
                            historical_mask
                            & env.current_player.eq(historical_current_colors)
                        )

                    actions = torch.empty(num_envs, dtype=torch.long, device=device)

                    # This dynamic compaction is intentionally retained for safety:
                    # only current-policy moves may enter PPO. Removing it would
                    # require changing buffer semantics, not merely an optimization.
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

                    # One fixed-shape history forward. We sample all valid history
                    # slots too, then use a GPU torch.where to override ONLY turns
                    # actually owned by the frozen opponent. No CUDA->CPU .any().
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
                        old_turn = (
                            history_table_valid
                            & grouped_players.ne(grouped_current_colors)
                        )

                        valid_old_turn = old_turn.reshape(-1)[
                            history_valid_flat_positions
                        ]
                        valid_history_actions = grouped_actions.reshape(-1)[
                            history_valid_flat_positions
                        ]
                        actions[history_flat_tables] = torch.where(
                            valid_old_turn,
                            valid_history_actions,
                            actions[history_flat_tables],
                        )

                    step = env.step(actions)
'''
new = '''            with torch.inference_mode():
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
'''
text = replace_once(text, old, new, "replace collector with online symmetry/opening")

old = '''            # Release the last collector outputs before the much larger PPO pass.
            del network_input, legal, actions, current_actor_mask, current_idx
            if history_tables:
                del (
                    grouped_input,
                    grouped_legal,
                    grouped_logits,
                    grouped_actions,
                    grouped_players,
                    grouped_current_colors,
                    old_turn,
                    valid_old_turn,
                    valid_history_actions,
                )
'''
new = '''            # Release the last collector outputs before the much larger PPO pass.
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
'''
text = replace_once(text, old, new, "update collector cleanup")

old = '''def _transform_board_actions(
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
'''
new = '''def _transform_board_actions(
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
'''
text = replace_once(text, old, new, "reuse tested symmetry primitives")

train_path.write_text(text, encoding="utf-8")

config_path = Path("configs/train.yaml")
config = config_path.read_text(encoding="utf-8")
old = '''  symmetry_augmentation: false
'''
new = '''  # ONLINE D4: każdy kolejny krok collectora zmienia widok planszy w cyklu:
  # 0°, 90°, 180°, 270°, mirror, mirror+90°, mirror+180°, mirror+270°.
  # Do PPO trafia dokładnie plansza i akcja widziana przez model. Dopiero akcja
  # wysyłana do środowiska jest przeliczana z powrotem na realne współrzędne.
  symmetry_augmentation: true

  # W tylu nowych partiach pierwszy kamień Czarnych jest wymuszony losowo na
  # całej planszy. Ten ruch NIE jest decyzją policy i nigdy nie trafia do PPO.
  random_black_opening_fraction: 0.50
'''
config = replace_once(config, old, new, "enable online symmetry/random opening")
config_path.write_text(config, encoding="utf-8")

test_path = Path("tests/test_training_features.py")
tests = test_path.read_text(encoding="utf-8")
old = '''from connect6.train import (
    CompleteGameBuffer,
    _historical_layout,
    _historical_table_matrix,
    _temperature,
)
'''
new = '''from connect6.train import (
    CompleteGameBuffer,
    _forced_random_opening_mask,
    _historical_layout,
    _historical_table_matrix,
    _inverse_transform_actions,
    _symmetry_for_phase,
    _temperature,
    _transform_board_actions,
)
'''
tests = replace_once(tests, old, new, "extend training feature imports")

tests += '''\n\ndef test_online_symmetry_cycle_visits_all_eight_d4_views_in_order():
    expected = [
        (0, False),
        (1, False),
        (2, False),
        (3, False),
        (0, True),
        (1, True),
        (2, True),
        (3, True),
    ]
    assert [_symmetry_for_phase(i) for i in range(8)] == expected
    assert _symmetry_for_phase(8) == expected[0]


def test_online_symmetry_board_action_and_inverse_action_are_consistent():
    board = torch.arange(9, dtype=torch.int8).reshape(1, 3, 3)
    actions = torch.arange(9, dtype=torch.long)
    original_values = board.reshape(-1)[actions]

    for phase in range(8):
        k, flip = _symmetry_for_phase(phase)
        transformed_board, transformed_actions = _transform_board_actions(
            board,
            actions,
            k,
            flip,
        )
        transformed_values = transformed_board.reshape(-1)[transformed_actions]
        restored_actions = _inverse_transform_actions(
            transformed_actions,
            board_size=3,
            k=k,
            flip=flip,
        )

        assert torch.equal(transformed_values, original_values)
        assert torch.equal(restored_actions, actions)


def test_random_black_opening_mask_only_selects_fresh_black_first_move():
    move_count = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int16)
    current_player = torch.tensor([1, -1, 1, 1, 1], dtype=torch.int8)
    stones_left = torch.tensor([1, 1, 1, 2, 1], dtype=torch.int8)

    forced_all = _forced_random_opening_mask(
        move_count,
        current_player,
        stones_left,
        fraction=1.0,
    )
    forced_none = _forced_random_opening_mask(
        move_count,
        current_player,
        stones_left,
        fraction=0.0,
    )

    assert forced_all.tolist() == [True, False, False, False, True]
    assert not forced_none.any()
'''
test_path.write_text(tests, encoding="utf-8")
