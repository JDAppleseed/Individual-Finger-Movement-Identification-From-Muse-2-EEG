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
