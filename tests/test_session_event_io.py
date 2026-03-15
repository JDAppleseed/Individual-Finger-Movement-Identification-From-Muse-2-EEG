import json

from utils.session_event_io import (
    event_row_to_payload,
    load_events_dataframe,
    resolve_raw_shard_paths,
)


def test_load_events_dataframe_reads_metadata_backed_events(tmp_path):
    events_path = tmp_path / "events.jsonl"
    payload = {
        "event_time_s": 1.25,
        "label": "thumb_open",
        "lsl_ts_mono": 123.4,
        "metadata": {
            "duration_s": 0.5,
            "action_id": 1,
            "finger_id": 1,
            "source": "keyboard",
            "notes": "captured",
            "trial": 7,
            "block": 2,
            "mode": "physical",
        },
    }
    events_path.write_text(json.dumps(payload) + "\n")

    events = load_events_dataframe(events_path)

    assert len(events) == 1
    row = events.iloc[0]
    assert row["onset_s"] == 1.25
    assert row["duration_s"] == 0.5
    assert row["type"] == "thumb_open"
    assert row["action_id"] == 1
    assert row["finger_id"] == 1
    assert row["trial_id"] == 7
    assert row["block_id"] == 2
    assert row["session_mode"] == "physical"
    assert row["source"] == "keyboard"
    assert row["notes"] == "captured"


def test_event_row_to_payload_preserves_metadata_fields():
    row = {
        "onset_s": 2.0,
        "duration_s": 0.75,
        "type": "index_open",
        "action_id": 1,
        "finger_id": 2,
        "source": "keyboard",
        "notes": "edited",
        "_source_payload": {
            "label": "index_open",
            "metadata": {
                "custom_field": "keep-me",
                "trial_id": 3,
            },
        },
    }

    payload = event_row_to_payload(row)

    assert payload["onset_s"] == 2.0
    assert payload["event_time_s"] == 2.0
    assert payload["duration_s"] == 0.75
    assert payload["end_s"] == 2.75
    assert payload["type"] == "index_open"
    assert payload["label"] == "index_open"
    assert payload["action_id"] == 1
    assert payload["finger_id"] == 2
    assert payload["notes"] == "edited"
    assert payload["metadata"]["custom_field"] == "keep-me"
    assert payload["metadata"]["action_id"] == 1
    assert payload["metadata"]["finger_id"] == 2
    assert payload["metadata"]["source"] == "keyboard"


def test_resolve_raw_shard_paths_prefers_manifest_order(tmp_path):
    session_dir = tmp_path / "session"
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True)
    shard_a = raw_dir / "eeg_raw_shard_000.npy"
    shard_b = raw_dir / "eeg_raw_shard_001.npy"
    shard_a.write_bytes(b"a")
    shard_b.write_bytes(b"b")
    manifest = {
        "shard_list": [
            {"path": "raw/eeg_raw_shard_001.npy"},
            {"path": "raw/eeg_raw_shard_000.npy"},
        ]
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest))

    shard_paths = resolve_raw_shard_paths(session_dir)

    assert shard_paths == [shard_b.resolve(), shard_a.resolve()]
