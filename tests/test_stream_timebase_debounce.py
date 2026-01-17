from collections import deque

import pytest

from utils.stream_timebase import clamp_lsl_timestamp, should_segment_break_backwards


def test_soft_backwards_is_clamped_not_hard():
    result = clamp_lsl_timestamp(
        10.0, 9.995, epsilon_s=0.010, hard_backwards_s=0.200
    )
    assert result.clamped is True
    assert result.is_hard_backwards is False
    assert result.backwards_delta_s == pytest.approx(0.005)


def test_hard_backwards_is_flagged():
    result = clamp_lsl_timestamp(
        10.0, 9.5, epsilon_s=0.010, hard_backwards_s=0.200
    )
    assert result.clamped is True
    assert result.is_hard_backwards is True
    assert result.backwards_delta_s == pytest.approx(0.5)


def test_backwards_burst_breaks_after_limit():
    events = deque(maxlen=256)
    assert (
        should_segment_break_backwards(
            events,
            1.0,
            soft_limit=3,
            window_s=1.0,
            hard_backwards=False,
        )
        is False
    )
    assert (
        should_segment_break_backwards(
            events,
            1.2,
            soft_limit=3,
            window_s=1.0,
            hard_backwards=False,
        )
        is False
    )
    assert (
        should_segment_break_backwards(
            events,
            1.4,
            soft_limit=3,
            window_s=1.0,
            hard_backwards=False,
        )
        is True
    )
