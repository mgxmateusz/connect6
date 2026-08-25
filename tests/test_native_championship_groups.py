from __future__ import annotations

import numpy as np

from connect6.championship.native_championship import (
    _final_pair_mask,
    _group_pair_mask,
    _pair_index,
)


def test_group_assignment_is_round_robin_and_only_internal_pairs() -> None:
    n = 12
    mask, members = _group_pair_mask(n, 4)

    assert [group.tolist() for group in members] == [
        [0, 4, 8],
        [1, 5, 9],
        [2, 6, 10],
        [3, 7, 11],
    ]
    assert int(mask.sum()) == 12  # 4 grupy * C(3, 2)

    for i in range(n):
        for j in range(i + 1, n):
            expected = (i % 4) == (j % 4)
            assert bool(mask[_pair_index(i, j, n)]) is expected


def test_finalists_are_scheduled_against_every_model_without_duplicate_pairs() -> None:
    n = 12
    finalists = np.asarray([0, 5, 10, 11], dtype=np.int32)
    mask = _final_pair_mask(n, finalists)

    # Pary incydentne do F finalistów: F*(N-F) + C(F,2).
    assert int(mask.sum()) == 4 * 8 + 6

    finalist_set = set(finalists.tolist())
    for i in range(n):
        for j in range(i + 1, n):
            expected = i in finalist_set or j in finalist_set
            assert bool(mask[_pair_index(i, j, n)]) is expected


def test_group_plus_final_stage_skips_matches_already_played_in_groups() -> None:
    n = 20
    group_mask, _ = _group_pair_mask(n, 4)
    finalists = np.asarray([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    final_mask = _final_pair_mask(n, finalists)

    already_played = int(np.count_nonzero(group_mask & final_mask))
    new_final_pairs = int(np.count_nonzero(final_mask & ~group_mask))

    assert already_played > 0
    assert new_final_pairs + already_played == int(final_mask.sum())
    assert int(np.count_nonzero(group_mask | final_mask)) < n * (n - 1) // 2
