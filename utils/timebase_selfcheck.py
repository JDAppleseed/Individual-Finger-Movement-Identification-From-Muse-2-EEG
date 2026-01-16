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


@dataclass(frozen=True)
class TimebaseConsistencyCheck:
    max_abs_delta_s: Optional[float]
    mean_abs_delta_s: Optional[float]
    warn_threshold_s: float
    error_threshold_s: float
    warn: bool
    error: bool
    sustained_warns: int
    sustained_warn_limit: int


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
    warn = False
    error = False
    if max_abs is not None and max_abs > warn_threshold_s:
        warn = True
    if mean_abs is not None and mean_abs > warn_threshold_s:
        warn = True
    if max_abs is not None and max_abs > error_threshold_s:
        error = True
    if mean_abs is not None and mean_abs > error_threshold_s:
        error = True
    return TimebaseSelfCheck(
        max_abs_delta_s=max_abs,
        mean_abs_delta_s=mean_abs,
        warn_threshold_s=warn_threshold_s,
        error_threshold_s=error_threshold_s,
        warn=warn,
        error=error,
    )


def _summarize_deltas(
    deltas: np.ndarray,
    warn_threshold_s: float,
    error_threshold_s: float,
    sustained_warn_limit: int,
) -> TimebaseConsistencyCheck:
    deltas_abs = np.abs(deltas)
    max_abs = float(np.max(deltas_abs)) if deltas_abs.size else None
    mean_abs = float(np.mean(deltas_abs)) if deltas_abs.size else None
    warn = False
    error = False
    if max_abs is not None and max_abs > warn_threshold_s:
        warn = True
    if mean_abs is not None and mean_abs > warn_threshold_s:
        warn = True
    if max_abs is not None and max_abs > error_threshold_s:
        error = True
    if mean_abs is not None and mean_abs > error_threshold_s:
        error = True
    sustained_warns = int(np.sum(deltas_abs > warn_threshold_s))
    if sustained_warns >= sustained_warn_limit:
        error = True
    return TimebaseConsistencyCheck(
        max_abs_delta_s=max_abs,
        mean_abs_delta_s=mean_abs,
        warn_threshold_s=warn_threshold_s,
        error_threshold_s=error_threshold_s,
        warn=warn,
        error=error,
        sustained_warns=sustained_warns,
        sustained_warn_limit=sustained_warn_limit,
    )


def evaluate_timebase_consistency(
    time_s: Sequence[float],
    lsl_timestamp_mono: Sequence[float],
    stream_start_lsl_ts: float,
    *,
    warn_threshold_s: float = 0.01,
    error_threshold_s: float = 0.05,
    sustained_warn_limit: int = 3,
) -> TimebaseConsistencyCheck:
    time_arr = np.asarray(time_s, dtype=float)
    lsl_arr = np.asarray(lsl_timestamp_mono, dtype=float)
    if time_arr.size == 0 or lsl_arr.size == 0:
        return _summarize_deltas(
            np.asarray([], dtype=float),
            warn_threshold_s=warn_threshold_s,
            error_threshold_s=error_threshold_s,
            sustained_warn_limit=sustained_warn_limit,
        )
    expected = lsl_arr - float(stream_start_lsl_ts)
    deltas = time_arr - expected
    return _summarize_deltas(
        deltas,
        warn_threshold_s=warn_threshold_s,
        error_threshold_s=error_threshold_s,
        sustained_warn_limit=sustained_warn_limit,
    )


def evaluate_event_time_consistency(
    onset_s: Sequence[float],
    onset_lsl: Sequence[float],
    stream_start_lsl_ts: float,
    *,
    warn_threshold_s: float = 0.01,
    error_threshold_s: float = 0.05,
    sustained_warn_limit: int = 3,
) -> TimebaseConsistencyCheck:
    onset_arr = np.asarray(onset_s, dtype=float)
    lsl_arr = np.asarray(onset_lsl, dtype=float)
    if onset_arr.size == 0 or lsl_arr.size == 0:
        return _summarize_deltas(
            np.asarray([], dtype=float),
            warn_threshold_s=warn_threshold_s,
            error_threshold_s=error_threshold_s,
            sustained_warn_limit=sustained_warn_limit,
        )
    expected = lsl_arr - float(stream_start_lsl_ts)
    deltas = onset_arr - expected
    return _summarize_deltas(
        deltas,
        warn_threshold_s=warn_threshold_s,
        error_threshold_s=error_threshold_s,
        sustained_warn_limit=sustained_warn_limit,
    )
