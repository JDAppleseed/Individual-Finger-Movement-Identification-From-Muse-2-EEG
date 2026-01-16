import numpy as np

from utils.stream_timebase import clamp_lsl_timestamp, is_gap


def test_clamp_lsl_timestamp_repairs_backwards():
    raw = [1.0, 1.1, 1.05, 1.2]
    mono = []
    prev = None
    backwards = 0
    for ts in raw:
        mono_ts, clamped = clamp_lsl_timestamp(prev, ts)
        if clamped:
            backwards += 1
        mono.append(mono_ts)
        prev = mono_ts

    assert mono == [1.0, 1.1, 1.1, 1.2]
    assert backwards == 1


def test_gap_detection_threshold():
    nominal_dt = 1.0 / 256.0
    assert is_gap(0.3, nominal_dt) is True
    assert is_gap(nominal_dt * 1.5, nominal_dt) is False
