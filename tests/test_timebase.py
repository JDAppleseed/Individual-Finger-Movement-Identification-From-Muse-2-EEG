from demo_backend.timebase import StreamClock, clamp_monotonic_window


def test_clamp_monotonic_window():
    start_s, end_s, clamped = clamp_monotonic_window(
        prev_end_s=1.0,
        start_s=0.9,
        end_s=0.95,
        eps=1e-6,
    )
    assert clamped is True
    assert end_s >= 1.0
    assert abs((end_s - start_s) - (0.95 - 0.9)) < 1e-6


def test_no_clamp_when_forward():
    start_s, end_s, clamped = clamp_monotonic_window(
        prev_end_s=1.0,
        start_s=1.1,
        end_s=1.2,
        eps=1e-6,
    )
    assert clamped is False
    assert start_s == 1.1
    assert end_s == 1.2


def test_stream_clock_extrapolation_non_negative():
    clock = StreamClock()
    assert clock.estimate_stream_ms_now(perf_s=10.0) is None
    assert clock.update_from_window_end_s(1.25, perf_s=100.0) == 1250
    assert clock.estimate_stream_ms_now(perf_s=100.5) == 1750
    assert clock.estimate_stream_ms_now(perf_s=99.0) == 1250
