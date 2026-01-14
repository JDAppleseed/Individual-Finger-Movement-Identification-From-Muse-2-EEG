import importlib.util
import json
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


def _load_extract_module():
    module_path = Path(__file__).resolve().parents[1] / "1b_extract_windows.py"
    spec = importlib.util.spec_from_file_location("extract_windows_report", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extraction_report_contents(tmp_path, monkeypatch):
    mod = _load_extract_module()
    features_path = tmp_path / "features.csv"
    events_path = tmp_path / "events.csv"

    times = np.arange(0.0, 1.0, 0.01)
    df_features = pd.DataFrame(
        {
            "lsl_timestamp": 10.0 + times,
            "time_s": times,
            "ch1": np.sin(times),
            "ch2": np.cos(times),
            "ch3": np.sin(times * 2.0),
            "ch4": np.cos(times * 2.0),
        }
    )
    df_features.to_csv(features_path, index=False)

    events = [
        {
            "onset_s": 0.1,
            "duration_s": 0.2,
            "action_id": 1,
            "finger_id": 2,
            "type": "open",
            "event_id": 1,
        },
        {
            "onset_s": 0.1,
            "duration_s": 0.2,
            "action_id": 1,
            "finger_id": 2,
            "type": "open",
            "event_id": 1,
        },
        {
            "onset_s": 0.6,
            "duration_s": 0.2,
            "action_id": 0,
            "finger_id": 0,
            "type": "rest",
            "event_id": 99,
        },
    ]
    df_events = pd.DataFrame(events)
    df_events.to_csv(events_path, index=False)

    monkeypatch.chdir(tmp_path)
    mod.KEEP_BASELINE_REST_EVENTS = 1
    mod.REST_SUBSAMPLE_PROB = 1.0
    mod.REST_MAX_WINDOWS = None

    argv = [
        "1b_extract_windows.py",
        "--features",
        str(features_path),
        "--events",
        str(events_path),
        "--seed",
        "123",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0

    report_path = tmp_path / "extraction_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())

    assert "drop_counts" in report
    assert report["drop_counts"]["ambiguous"] > 0
    assert "rest_policy" in report
    assert report["rest_policy"]["seed"] == 123
    assert report["rest_policy"]["baseline_rest_event_indices_count"] == 1

    windows_df = pd.read_csv(tmp_path / "eeg_windows.csv")
    assert (windows_df["assigned_event_type"] == "baseline_rest").any()
