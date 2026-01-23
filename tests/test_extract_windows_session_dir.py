import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pandas")


def _load_extract_module():
    module_path = Path(__file__).resolve().parents[1] / "1b_extract_windows.py"
    spec = importlib.util.spec_from_file_location("extract_windows_session", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_from_session_dir(tmp_path, monkeypatch):
    mod = _load_extract_module()
    session_dir = tmp_path / "subject_session"
    raw_dir = session_dir / "raw"
    events_dir = session_dir / "events"
    raw_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)

    meta = {
        "subject_id": "subject",
        "session_id": "session",
        "sampling_rate": 100.0,
        "channel_labels": ["ch1", "ch2", "ch3", "ch4"],
        "timebase_version": "absolute_v1",
    }
    (session_dir / "meta.json").write_text(json.dumps(meta))
    (session_dir / "manifest.json").write_text(
        json.dumps(
            {
                "seq_min": 0,
                "seq_max": 99,
                "expected_sample_count": 100,
                "actual_sample_count": 100,
                "missing_seq_count": 0,
                "termination_reason": "normal",
            }
        )
    )

    dtype = np.dtype(
        [
            ("seq", "<i8"),
            ("lsl_ts_raw", "<f8"),
            ("lsl_ts_mono", "<f8"),
            ("local_ts", "<f8"),
            ("flags", "<i8"),
            ("segment_id", "<i8"),
            ("clamped", "<i1"),
            ("sample", "<f8", (4,)),
        ]
    )
    raw = np.zeros((100,), dtype=dtype)
    raw["seq"] = np.arange(100)
    raw["lsl_ts_raw"] = 1.0 + np.arange(100) * 0.01
    raw["lsl_ts_mono"] = raw["lsl_ts_raw"]
    raw["local_ts"] = raw["lsl_ts_raw"]
    raw["sample"] = np.random.randn(100, 4)
    np.save(raw_dir / "eeg_raw_shard_000.npy", raw)

    event = {
        "onset_s": 0.2,
        "duration_s": 0.2,
        "action_id": 1,
        "finger_id": 2,
        "type": "open",
        "event_id": 1,
    }
    (events_dir / "events.jsonl").write_text(json.dumps(event) + "\n")

    monkeypatch.chdir(tmp_path)
    argv = [
        "1b_extract_windows.py",
        "--session-dir",
        str(session_dir),
        "--seed",
        "123",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert mod.main() == 0
    assert (tmp_path / "eeg_windows.csv").exists()
