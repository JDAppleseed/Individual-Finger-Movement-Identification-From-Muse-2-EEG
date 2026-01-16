from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class GapSummary:
    count: int
    max_gap_s: Optional[float]
    p95_gap_s: Optional[float]
    p99_gap_s: Optional[float]


def clamp_lsl_timestamp(
    prev_mono: Optional[float], raw_lsl_ts: float
) -> Tuple[float, bool]:
    if prev_mono is None:
        return float(raw_lsl_ts), False
    if raw_lsl_ts < prev_mono:
        return float(prev_mono), True
    return float(raw_lsl_ts), False


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
