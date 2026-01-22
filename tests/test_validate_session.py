import json
from pathlib import Path

import numpy as np

from muse_streaming.validate_session import validate_session_dir


def _write_session(tmp_path: Path, *, missing: bool = False) -> Path:
    session_dir = tmp_path / "subject_session"
    raw_dir = session_dir / "raw"
    events_dir = session_dir / "events"
    raw_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)

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
    raw = np.zeros((10,), dtype=dtype)
    raw["seq"] = np.arange(10)
    if missing:
        raw["seq"][5:] += 1
    raw["lsl_ts_raw"] = 1.0 + np.arange(10) * 0.01
    raw["lsl_ts_mono"] = raw["lsl_ts_raw"]
    raw["local_ts"] = raw["lsl_ts_raw"]
    raw["sample"] = np.random.randn(10, 4)
    np.save(raw_dir / "eeg_raw_shard_000.npy", raw)

    (events_dir / "events.jsonl").write_text(json.dumps({"onset_s": 0.1}) + "\n")

    manifest = {
        "seq_min": int(raw["seq"][0]),
        "seq_max": int(raw["seq"][-1]),
        "expected_sample_count": int(raw["seq"][-1] - raw["seq"][0] + 1),
        "actual_sample_count": int(raw.shape[0]),
        "missing_seq_count": 0 if not missing else 1,
        "termination_reason": "normal",
        "shard_list": [{"path": str(raw_dir / "eeg_raw_shard_000.npy")}],
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest))
    (session_dir / "meta.json").write_text(json.dumps({"session_id": "session"}))
    return session_dir


def test_validate_session_ok(tmp_path):
    session_dir = _write_session(tmp_path)
    report = validate_session_dir(session_dir)
    assert report["ok"] is True


def test_validate_session_missing_seq(tmp_path):
    session_dir = _write_session(tmp_path, missing=True)
    report = validate_session_dir(session_dir)
    assert report["ok"] is False
