from muse_streaming.muse_lsl_streamer import FUTURE_TOL_S, MuseLslStreamer


def _make_streamer() -> MuseLslStreamer:
    streamer = MuseLslStreamer(simulate=True)
    streamer._lsl_offset = 0.0
    return streamer


def _is_strictly_increasing(seq) -> bool:
    return all(seq[i] > seq[i - 1] for i in range(1, len(seq)))


def test_timebase_controller_adjusts_backward() -> None:
    streamer = _make_streamer()
    fs = streamer.config.rate
    now = 1000.0
    streamer._now_mono = lambda: now
    ts = streamer._build_timestamps(4)
    assert _is_strictly_increasing(ts)

    expected_end = streamer._timebase_t0_mono + (
        (streamer._sample_index + 4 - 1) / fs
    )
    now = expected_end - (FUTURE_TOL_S + 0.05)
    prev_adjust = streamer._t0_adjust_total
    ts = streamer._build_timestamps(4)
    assert _is_strictly_increasing(ts)
    assert streamer._t0_adjust_total < prev_adjust
    assert streamer._last_time_err_s is not None and streamer._last_time_err_s < 0.0


def test_timebase_controller_adjusts_forward() -> None:
    streamer = _make_streamer()
    fs = streamer.config.rate
    now = 2000.0
    streamer._now_mono = lambda: now
    ts = streamer._build_timestamps(4)
    assert _is_strictly_increasing(ts)

    expected_end = streamer._timebase_t0_mono + (
        (streamer._sample_index + 4 - 1) / fs
    )
    now = expected_end + 0.1
    prev_adjust = streamer._t0_adjust_total
    ts = streamer._build_timestamps(4)
    assert _is_strictly_increasing(ts)
    assert streamer._t0_adjust_total > prev_adjust
    assert streamer._last_time_err_s is not None and streamer._last_time_err_s > 0.0
