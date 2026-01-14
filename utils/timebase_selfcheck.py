from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TimebaseSelfCheck:
    max_abs_delta_s: Optional[float]
    mean_abs_delta_s: Optional[float]
    warn_threshold_s: float
    error_threshold_s: float
    warn: bool
    error: bool


def evaluate_timebase_alignment(
    recent_sample_times: Sequence[float],
    event_times: Iterable[float],
    warn_threshold_s: float = 0.05,
    error_threshold_s: float = 0.2,
) -> TimebaseSelfCheck:
    sample_times = np.asarray(recent_sample_times, dtype=float)
    event_times_arr = np.asarray(list(event_times), dtype=float)
    if sample_times.size == 0 or event_times_arr.size == 0:
        return TimebaseSelfCheck(
            max_abs_delta_s=None,
            mean_abs_delta_s=None,
            warn_threshold_s=warn_threshold_s,
            error_threshold_s=error_threshold_s,
            warn=False,
            error=False,
        )

    deltas = []
    for event_time in event_times_arr:
        nearest = sample_times[np.argmin(np.abs(sample_times - event_time))]
        deltas.append(float(event_time - nearest))
    deltas_arr = np.abs(np.asarray(deltas, dtype=float))
    max_abs = float(np.max(deltas_arr)) if deltas_arr.size else None
    mean_abs = float(np.mean(deltas_arr)) if deltas_arr.size else None
    warn = max_abs is not None and max_abs > warn_threshold_s
    error = max_abs is not None and max_abs > error_threshold_s
    return TimebaseSelfCheck(
        max_abs_delta_s=max_abs,
        mean_abs_delta_s=mean_abs,
        warn_threshold_s=warn_threshold_s,
        error_threshold_s=error_threshold_s,
        warn=warn,
        error=error,
    )
