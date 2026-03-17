from pathlib import Path

from tools.analyze_live_predictions import (
    build_segments,
    resolve_latest_live_infer_dir,
    resolve_prediction_log_path,
    summarize_records,
)


def test_build_segments_and_summary():
    records = [
        {
            "window_start_s": 0.00,
            "window_end_s": 0.25,
            "ts_utc": 1000.0,
            "alignment_ok": True,
            "committed_action_id": 0,
            "committed_finger_id": 0,
            "action_conf": 0.91,
            "finger_conf": 1.0,
            "joint_conf": 0.91,
            "decision_reason": "commit",
            "actuation_sent": False,
            "uncertainty_gate_ok": True,
            "latency_ms": 80.0,
        },
        {
            "window_start_s": 0.05,
            "window_end_s": 0.30,
            "ts_utc": 1000.05,
            "alignment_ok": True,
            "committed_action_id": 0,
            "committed_finger_id": 0,
            "action_conf": 0.88,
            "finger_conf": 1.0,
            "joint_conf": 0.88,
            "decision_reason": "commit",
            "actuation_sent": False,
            "uncertainty_gate_ok": True,
            "latency_ms": 82.0,
        },
        {
            "window_start_s": 0.10,
            "window_end_s": 0.35,
            "ts_utc": 1000.10,
            "alignment_ok": True,
            "committed_action_id": 1,
            "committed_finger_id": 1,
            "action_conf": 0.77,
            "finger_conf": 0.72,
            "joint_conf": 0.72,
            "decision_reason": "commit",
            "actuation_sent": True,
            "uncertainty_gate_ok": True,
            "latency_ms": 84.0,
        },
        {
            "window_start_s": 0.15,
            "window_end_s": 0.40,
            "ts_utc": 1000.15,
            "alignment_ok": True,
            "committed_action_id": 1,
            "committed_finger_id": 1,
            "action_conf": 0.79,
            "finger_conf": 0.75,
            "joint_conf": 0.75,
            "decision_reason": "commit",
            "actuation_sent": False,
            "uncertainty_gate_ok": True,
            "latency_ms": 86.0,
        },
    ]

    segments = build_segments(records)
    assert len(segments) == 2
    assert segments[0]["pair_label"] == "REST+NONE"
    assert segments[1]["pair_label"] == "OPEN+THUMB"
    assert abs(segments[0]["duration_s"] - 0.30) < 1e-9
    assert abs(segments[1]["duration_s"] - 0.30) < 1e-9
    assert segments[1]["actuation_sent_count"] == 1
    assert abs(segments[1]["mean_joint_conf"] - 0.735) < 1e-9
    assert abs(segments[1]["max_joint_conf"] - 0.75) < 1e-9

    result = summarize_records(records, short_segment_sec=0.35)
    summary = result["summary"]
    assert summary["record_count"] == 4
    assert summary["valid_window_count"] == 4
    assert summary["alignment_fail_count"] == 0
    assert summary["segment_count"] == 2
    assert summary["actuatable_segment_count"] == 1
    assert summary["short_actuatable_segment_count"] == 1
    assert summary["actuation_sent_count"] == 1
    assert summary["pair_counts"]["REST+NONE"] == 2
    assert summary["pair_counts"]["OPEN+THUMB"] == 2


def test_resolve_prediction_log_from_latest_live_dir(tmp_path: Path):
    session_dir = tmp_path / "session"
    processed = session_dir / "processed"
    older = processed / "live_infer"
    newer = processed / "live_infer_v2"
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    (older / "predictions.jsonl").write_text('{"committed_action_id":0}\n')
    (newer / "predictions.jsonl").write_text('{"committed_action_id":1}\n')

    latest = resolve_latest_live_infer_dir(session_dir)
    assert latest == newer

    pred_log = resolve_prediction_log_path(
        pred_log=None,
        session_dir=session_dir,
        config_dir=None,
    )
    assert pred_log == (newer / "predictions.jsonl").resolve()
