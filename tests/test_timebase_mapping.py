from utils.session_timebase import compute_event_lsl_ts


def test_clock_offset_formula():
    stream_start_lsl_ts = 123.456
    local_at_start = 120.0
    clock_offset = stream_start_lsl_ts - local_at_start
    event_lsl_ts = compute_event_lsl_ts(local_at_start, clock_offset)
    assert event_lsl_ts == stream_start_lsl_ts


def test_event_local_to_lsl_conversion():
    local_ts = 1000.0
    clock_offset = 5.25
    event_lsl_ts = compute_event_lsl_ts(local_ts, clock_offset)
    assert event_lsl_ts == local_ts + clock_offset


def test_feature_and_event_share_stream_start():
    stream_start_lsl_ts = 2000.0
    feature_lsl_ts = 2000.25
    local_ts = 500.0
    clock_offset = 1500.0
    event_lsl_ts = compute_event_lsl_ts(local_ts, clock_offset)
    assert event_lsl_ts == 2000.0

    feature_time_s = feature_lsl_ts - stream_start_lsl_ts
    event_time_s = event_lsl_ts - stream_start_lsl_ts
    assert feature_time_s == 0.25
    assert event_time_s == 0.0
