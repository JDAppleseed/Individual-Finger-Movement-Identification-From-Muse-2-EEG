import json

import numpy as np
from scipy.io import loadmat

from utils.eeglab_export import (
    default_eeglab_export_path,
    export_session_to_eeglab,
    resolve_eeglab_export_events_path,
)


def _write_session(tmp_path):
    session_dir = tmp_path / "subject_session"
    raw_dir = session_dir / "raw"
    events_dir = session_dir / "events"
    raw_dir.mkdir(parents=True)
    events_dir.mkdir(parents=True)

    meta = {
        "subject_id": "S1",
        "session_id": "20260319_120000",
        "sampling_rate": 100.0,
        "channel_labels": ["TP9", "AF7", "AF8", "TP10"],
    }
    (session_dir / "meta.json").write_text(json.dumps(meta))

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
    raw["lsl_ts_raw"] = 1.0 + np.arange(10) * 0.01
    raw["lsl_ts_mono"] = raw["lsl_ts_raw"]
    raw["local_ts"] = raw["lsl_ts_raw"]
    raw["sample"] = np.arange(40, dtype=np.float64).reshape(10, 4)
    np.save(raw_dir / "eeg_raw_shard_000.npy", raw)

    event = {
        "onset_s": 0.02,
        "duration_s": 0.1,
        "type": "index_open",
    }
    (events_dir / "events.jsonl").write_text(json.dumps(event) + "\n")
    return session_dir


def test_export_session_to_eeglab_writes_set_file(tmp_path):
    session_dir = _write_session(tmp_path)
    out_path = default_eeglab_export_path(session_dir)

    summary = export_session_to_eeglab(session_dir, out_path)

    assert summary.out_path == out_path
    assert summary.sample_count == 10
    assert summary.channel_count == 4
    assert summary.event_count == 1
    assert out_path.exists()

    payload = loadmat(out_path, squeeze_me=True, struct_as_record=False)
    eeg = payload["EEG"]
    assert eeg.setname == "S1_20260319_120000"
    assert eeg.nbchan == 4
    assert eeg.pnts == 10
    assert eeg.srate == 100.0
    assert eeg.data.shape == (4, 10)
    assert eeg.chanlocs[0].labels == "TP9"
    assert eeg.event.type == "index_open"
    assert eeg.event.latency == 3.0


def test_export_session_to_eeglab_skips_out_of_range_events(tmp_path):
    session_dir = _write_session(tmp_path)
    (session_dir / "events" / "events.jsonl").write_text(
        json.dumps({"onset_s": 99.0, "duration_s": 0.0, "type": "late"}) + "\n"
    )

    summary = export_session_to_eeglab(session_dir)

    assert summary.event_count == 0
    assert summary.skipped_event_count == 1


def test_export_session_to_eeglab_uses_metadata_event_path_fallback(tmp_path):
    session_dir = _write_session(tmp_path)
    custom_events = session_dir / "logs" / "captured_events.jsonl"
    custom_events.parent.mkdir(parents=True, exist_ok=True)
    custom_events.write_text((session_dir / "events" / "events.jsonl").read_text())
    (session_dir / "events" / "events.jsonl").unlink()

    meta = json.loads((session_dir / "meta.json").read_text())
    meta["events_jsonl_path"] = str(custom_events)
    (session_dir / "meta.json").write_text(json.dumps(meta))

    assert resolve_eeglab_export_events_path(session_dir) == custom_events.resolve()

    summary = export_session_to_eeglab(session_dir)

    assert summary.event_count == 1


def test_export_session_to_eeglab_without_events_still_writes_set_file(tmp_path):
    session_dir = _write_session(tmp_path)
    (session_dir / "events" / "events.jsonl").unlink()

    summary = export_session_to_eeglab(session_dir)

    assert summary.out_path.exists()
    assert summary.event_count == 0
    assert summary.skipped_event_count == 0
