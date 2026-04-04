from __future__ import annotations

import numpy as np

from muse_streaming.resample import resample_window, verify_alignment


def test_window_fixed_size_from_irregular_samples():
    rng = np.random.default_rng(0)
    target_fs = 128.0
    window_sec = 1.0
    start_s = 0.0
    end_s = start_s + window_sec
    times = np.cumsum(rng.uniform(0.005, 0.012, size=200))
    times = times[(times >= start_s) & (times < end_s)]
    values = rng.normal(size=(times.size, 4))

    alignment = verify_alignment(
        times,
        start_s=start_s,
        end_s=end_s,
        target_fs=target_fs,
        max_gap_s=0.05,
    )
    assert alignment.ok

    grid, window = resample_window(
        times,
        values,
        start_s=start_s,
        end_s=end_s,
        target_fs=target_fs,
    )
    assert window.shape[0] == int(round(window_sec * target_fs))
    assert window.shape[0] == grid.size


def test_verify_alignment_rejects_missing_window_start_support():
    times = np.array([0.11, 0.15, 0.20, 0.24], dtype=float)
    values = np.zeros((times.size, 4), dtype=float)

    alignment = verify_alignment(
        times,
        start_s=0.0,
        end_s=0.25,
        target_fs=128.0,
        max_gap_s=0.05,
    )

    assert alignment.ok is False
    assert alignment.reason == "start_gap_exceeds_threshold"

    grid, window = resample_window(
        times,
        values,
        start_s=0.0,
        end_s=0.25,
        target_fs=128.0,
    )
    assert grid.size == window.shape[0]


def test_verify_alignment_rejects_missing_window_end_support():
    times = np.array([-0.01, 0.02, 0.08, 0.12], dtype=float)

    alignment = verify_alignment(
        times,
        start_s=0.0,
        end_s=0.25,
        target_fs=128.0,
        max_gap_s=0.05,
    )

    assert alignment.ok is False
    assert alignment.reason == "end_gap_exceeds_threshold"
