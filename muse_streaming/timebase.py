from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class TimebaseCheck:
    ok: bool
    warnings: List[str]
    max_gap_s: Optional[float]
    max_backward_s: Optional[float]


def absolute_v1_time_s(lsl_ts: float, stream_start_lsl_ts: float) -> float:
    return float(lsl_ts - stream_start_lsl_ts)


def lsl_from_local(local_ts: float, clock_offset: float) -> float:
    return float(local_ts + clock_offset)


def clamp_monotonic(
    lsl_ts: float,
    last_lsl_ts: Optional[float],
    epsilon: float = 1e-6,
) -> tuple[float, bool]:
    if last_lsl_ts is None:
        return float(lsl_ts), False
    if lsl_ts <= last_lsl_ts:
        return float(last_lsl_ts + epsilon), True
    return float(lsl_ts), False


def check_timebase_invariants(
    lsl_timestamps: Iterable[float],
    *,
    max_gap_s: float,
    allow_backwards_s: float = 0.0,
) -> TimebaseCheck:
    ts_list = [float(ts) for ts in lsl_timestamps if ts is not None]
    warnings: List[str] = []
    if len(ts_list) < 2:
        return TimebaseCheck(ok=True, warnings=warnings, max_gap_s=None, max_backward_s=None)

    diffs = [b - a for a, b in zip(ts_list[:-1], ts_list[1:])]
    max_gap = max(diffs)
    min_gap = min(diffs)

    max_backward = None
    if min_gap < 0:
        max_backward = float(abs(min_gap))
        if max_backward > allow_backwards_s:
            warnings.append("backwards_timestamp_detected")

    if max_gap > max_gap_s:
        warnings.append("gap_exceeds_threshold")

    ok = not warnings
    return TimebaseCheck(
        ok=ok,
        warnings=warnings,
        max_gap_s=float(max_gap),
        max_backward_s=max_backward,
    )


def latency_ms(lsl_now: float, lsl_event_ts: float) -> float:
    return float((lsl_now - lsl_event_ts) * 1000.0)


def is_finite(ts: Optional[float]) -> bool:
    return ts is not None and math.isfinite(ts)
