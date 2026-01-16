import importlib.util
from pathlib import Path

import numpy as np


def _load_extract_module():
    module_path = Path(__file__).resolve().parents[1] / "1b_extract_windows.py"
    spec = importlib.util.spec_from_file_location("extract_windows", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_gap_window_rejection():
    mod = _load_extract_module()
    times = np.array([0.0, 0.01, 0.5], dtype=float)
    gap_flag, gap_fraction, max_dt = mod.compute_gap_metrics(
        times, gap_threshold_s=0.05, window_sec=0.25
    )
    assert gap_flag == 1
    assert gap_fraction > 0.0
    assert mod.should_drop_gap(
        gap_flag,
        max_dt,
        allow_gaps=False,
        allow_gap_interp=False,
        gap_interp_max_s=0.05,
    )
