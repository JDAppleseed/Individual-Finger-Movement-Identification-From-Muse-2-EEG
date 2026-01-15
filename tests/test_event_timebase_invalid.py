import math

from utils.session_timebase import compute_event_lsl_ts


def test_compute_event_lsl_ts_invalid_mapping_returns_none():
    assert compute_event_lsl_ts(None, 1.0) is None
    assert compute_event_lsl_ts(1.0, None) is None
    assert compute_event_lsl_ts(math.nan, 1.0) is None
    assert compute_event_lsl_ts(1.0, math.inf) is None
