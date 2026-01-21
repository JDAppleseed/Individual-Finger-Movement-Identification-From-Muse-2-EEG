from __future__ import annotations

import numpy as np

from muse_streaming.timebase import (
    absolute_v1_time_s,
    check_timebase_invariants,
    latency_ms,
)


def test_timebase_invariants_ok():
    lsl_ts = np.linspace(100.0, 101.0, 10)
    check = check_timebase_invariants(lsl_ts, max_gap_s=0.2)
    assert check.ok
    assert check.warnings == []


def test_timebase_absolute_v1_and_latency():
    stream_start = 100.0
    sample_ts = 100.5
    time_s = absolute_v1_time_s(sample_ts, stream_start)
    assert time_s == 0.5
    assert latency_ms(101.0, sample_ts) == 500.0
