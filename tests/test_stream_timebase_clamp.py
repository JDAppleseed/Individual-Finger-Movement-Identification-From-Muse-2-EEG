import pytest

from utils.stream_timebase import clamp_lsl_timestamp


def test_clamp_prev_none():
    result = clamp_lsl_timestamp(None, 10.0, epsilon_s=0.01, hard_backwards_s=0.2)
    assert result.clamped is False
    assert result.backwards_delta_s == pytest.approx(0.0)
    assert result.is_soft_backwards is False
    assert result.is_hard_backwards is False


def test_clamp_forward_time():
    result = clamp_lsl_timestamp(10.0, 10.01, epsilon_s=0.01, hard_backwards_s=0.2)
    assert result.clamped is False
    assert result.is_soft_backwards is False
    assert result.is_hard_backwards is False


def test_clamp_tiny_backwards():
    result = clamp_lsl_timestamp(10.0, 9.999, epsilon_s=0.01, hard_backwards_s=0.2)
    assert result.clamped is True
    assert result.is_soft_backwards is False
    assert result.is_hard_backwards is False


def test_clamp_soft_backwards():
    result = clamp_lsl_timestamp(10.0, 9.98, epsilon_s=0.01, hard_backwards_s=0.2)
    assert result.clamped is True
    assert result.is_soft_backwards is True
    assert result.is_hard_backwards is False


def test_clamp_hard_backwards():
    result = clamp_lsl_timestamp(10.0, 9.5, epsilon_s=0.01, hard_backwards_s=0.2)
    assert result.clamped is True
    assert result.is_soft_backwards is False
    assert result.is_hard_backwards is True
