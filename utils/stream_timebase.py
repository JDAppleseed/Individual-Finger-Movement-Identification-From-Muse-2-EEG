from __future__ import annotations

from dataclasses import dataclass
from typing import Deque, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class GapSummary:
    count: int
    max_gap_s: Optional[float]
    p95_gap_s: Optional[float]
    p99_gap_s: Optional[float]


@dataclass(frozen=True)
class ClampResult:
    mono_ts: float
    clamped: bool
    backwards_delta_s: float
    is_hard_backwards: bool
    is_soft_backwards: bool


def clamp_lsl_timestamp(
    prev_mono: Optional[float],
    raw_lsl_ts: float,
    *,
    epsilon_s: float = 0.010,
    hard_backwards_s: float = 0.200,
) -> ClampResult:
    if prev_mono is None:
        return ClampResult(
            mono_ts=float(raw_lsl_ts),
            clamped=False,
            backwards_delta_s=0.0,
            is_hard_backwards=False,
            is_soft_backwards=False,
        )
    if raw_lsl_ts < prev_mono:
        backwards_delta = float(prev_mono - raw_lsl_ts)
        is_hard = backwards_delta >= float(hard_backwards_s)
        is_soft = backwards_delta > float(epsilon_s) and not is_hard
        return ClampResult(
            mono_ts=float(prev_mono),
            clamped=True,
            backwards_delta_s=backwards_delta,
            is_hard_backwards=is_hard,
            is_soft_backwards=is_soft,
        )
    return ClampResult(
        mono_ts=float(raw_lsl_ts),
        clamped=False,
        backwards_delta_s=0.0,
        is_hard_backwards=False,
        is_soft_backwards=False,
    )


def should_segment_break_backwards(
    backwards_events_ts_monotonic: Deque[float],
    now_monotonic: float,
    *,
    soft_limit: int = 6,
    window_s: float = 1.0,
    hard_backwards: bool,
) -> bool:
    if hard_backwards:
        return True
    backwards_events_ts_monotonic.append(float(now_monotonic))
    cutoff = float(now_monotonic) - float(window_s)
    while backwards_events_ts_monotonic and backwards_events_ts_monotonic[0] < cutoff:
        backwards_events_ts_monotonic.popleft()
    return len(backwards_events_ts_monotonic) >= int(soft_limit)


def gap_threshold_s(nominal_dt: float) -> float:
    return max(2.5 * float(nominal_dt), 0.25)


def is_gap(dt_s: float, nominal_dt: float) -> bool:
    return float(dt_s) > gap_threshold_s(nominal_dt)


def summarize_gaps(gaps_s: Iterable[float]) -> GapSummary:
    gaps = [float(g) for g in gaps_s if g is not None]
    if not gaps:
        return GapSummary(count=0, max_gap_s=None, p95_gap_s=None, p99_gap_s=None)
    arr = np.asarray(gaps, dtype=float)
    return GapSummary(
        count=int(arr.size),
        max_gap_s=float(np.max(arr)),
        p95_gap_s=float(np.percentile(arr, 95)),
        p99_gap_s=float(np.percentile(arr, 99)),
    )
