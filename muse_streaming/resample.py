from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class AlignmentReport:
    ok: bool
    reason: Optional[str]
    window_size: int
    max_gap_s: Optional[float]
    monotonic: bool


def resample_window(
    times: np.ndarray,
    values: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    target_fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if times.size < 2:
        raise ValueError("Not enough samples to resample.")
    if not np.all(np.diff(times) > 0):
        order = np.argsort(times)
        times = times[order]
        values = values[order]
    sample_count = int(round((end_s - start_s) * target_fs))
    grid = np.linspace(start_s, end_s, sample_count, endpoint=False, dtype=float)
    window = np.zeros((grid.size, values.shape[1]), dtype=float)
    for ch_idx in range(values.shape[1]):
        window[:, ch_idx] = np.interp(grid, times, values[:, ch_idx])
    return grid, window


def verify_alignment(
    times: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    target_fs: float,
    max_gap_s: float,
) -> AlignmentReport:
    if times.size < 2:
        return AlignmentReport(False, "insufficient_samples", 0, None, False)
    diffs = np.diff(times)
    monotonic = bool(np.all(diffs > 0))
    max_gap = float(np.max(diffs)) if diffs.size else None
    expected = int(round((end_s - start_s) * target_fs))
    if max_gap is not None and max_gap > max_gap_s:
        return AlignmentReport(False, "gap_exceeds_threshold", expected, max_gap, monotonic)
    if not monotonic:
        return AlignmentReport(False, "non_monotonic", expected, max_gap, monotonic)
    return AlignmentReport(True, None, expected, max_gap, monotonic)
