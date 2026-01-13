from utils.timebase import clamp_monotonic_time


def test_clamp_monotonic_time():
    value, clamped = clamp_monotonic_time(None, None)
    assert value is None
    assert clamped is False

    value, clamped = clamp_monotonic_time(None, 1.0)
    assert value == 1.0
    assert clamped is False

    value, clamped = clamp_monotonic_time(2.0, 1.5)
    assert value == 2.0
    assert clamped is True

    value, clamped = clamp_monotonic_time(2.0, 2.0)
    assert value == 2.0
    assert clamped is False
