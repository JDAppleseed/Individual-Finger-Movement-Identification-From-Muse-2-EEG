from __future__ import annotations

import numpy as np

from muse_streaming.timebase import (
    absolute_v1_time_s,
    check_timebase_invariants,
    latency_ms,
    lsl_from_local,
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


def test_lsl_from_local_matches_clock_offset_formula():
    stream_start_lsl_ts = 123.456
    local_at_start = 120.0
    clock_offset = stream_start_lsl_ts - local_at_start
    assert lsl_from_local(local_at_start, clock_offset) == stream_start_lsl_ts


def test_event_and_feature_times_share_stream_start_reference():
    stream_start_lsl_ts = 2000.0
    feature_lsl_ts = 2000.25
    event_local_ts = 500.0
    clock_offset = 1500.0

    event_lsl_ts = lsl_from_local(event_local_ts, clock_offset)
    feature_time_s = absolute_v1_time_s(feature_lsl_ts, stream_start_lsl_ts)
    event_time_s = absolute_v1_time_s(event_lsl_ts, stream_start_lsl_ts)

    assert event_lsl_ts == 2000.0
    assert feature_time_s == 0.25
    assert event_time_s == 0.0
