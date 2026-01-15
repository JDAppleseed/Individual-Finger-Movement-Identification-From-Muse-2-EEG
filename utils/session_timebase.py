from __future__ import annotations

import math
from typing import Optional


def compute_run_time_s(run_start_lsl_ts: float, lsl_ts: float) -> float:
    return float(lsl_ts - run_start_lsl_ts)


def compute_time_s(
    total_elapsed_s: float, run_start_lsl_ts: float, lsl_ts: float
) -> float:
    return float(total_elapsed_s + compute_run_time_s(run_start_lsl_ts, lsl_ts))


def compute_event_lsl_ts(
    local_ts: Optional[float], clock_offset: Optional[float]
) -> Optional[float]:
    if local_ts is None or clock_offset is None:
        return None
    if not math.isfinite(local_ts) or not math.isfinite(clock_offset):
        return None
    event_lsl_ts = float(local_ts + clock_offset)
    if not math.isfinite(event_lsl_ts):
        return None
    return event_lsl_ts


def compute_event_time_s(
    total_elapsed_s: float,
    run_start_lsl_ts: float,
    event_lsl_ts: float,
) -> float:
    return float(total_elapsed_s + (event_lsl_ts - run_start_lsl_ts))


def has_timebase_fields(
    total_elapsed_s: Optional[float],
    run_start_lsl_ts: Optional[float],
) -> bool:
    return total_elapsed_s is not None and run_start_lsl_ts is not None
