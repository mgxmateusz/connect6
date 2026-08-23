from __future__ import annotations

from connect6.championship.championship_cnn import AdaptiveTableController, _black_to_move


def test_connect6_move_schedule() -> None:
    expected_black = [
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert [_black_to_move(i) for i in range(len(expected_black))] == expected_black


def test_adaptive_controller_remembers_unsafe_ceiling(tmp_path) -> None:
    state = tmp_path / "adaptive_tables.json"
    controller = AdaptiveTableController(
        state,
        gpu_name="TEST GPU",
        limit_bytes=11 * 2**30,
        min_tables=16,
        max_tables=2048,
    )
    controller.mark_unsafe(512)

    assert controller.is_allowed(256)
    assert not controller.is_allowed(512)
    assert not controller.is_allowed(1024)

    restored = AdaptiveTableController(
        state,
        gpu_name="TEST GPU",
        limit_bytes=11 * 2**30,
        min_tables=16,
        max_tables=2048,
    )
    assert restored.unsafe_from == 512
    assert restored.next_lower(512, [16, 32, 64, 128, 256, 512, 1024]) == 256
