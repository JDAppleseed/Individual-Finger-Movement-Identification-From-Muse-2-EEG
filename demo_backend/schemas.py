from __future__ import annotations

from datetime import datetime, timezone

TICK_SCHEMA = {
    "type": "object",
    "required": [
        "type",
        "ts_utc",
        "mode",
        "session",
        "prediction",
        "safety",
        "diagnostics",
    ],
    "properties": {
        "type": {"const": "tick"},
        "ts_utc": {"type": "string"},
        "mode": {"type": "string", "enum": ["replay", "live", "idle"]},
        "session": {
            "type": "object",
            "required": [
                "subject_id",
                "experiment_hash",
                "window_index",
                "window_start_s",
                "window_end_s",
                "timebase_version",
            ],
            "properties": {
                "subject_id": {"type": "string"},
                "experiment_hash": {"type": "string"},
                "window_index": {"type": "integer"},
                "window_start_s": {"type": "number"},
                "window_end_s": {"type": "number"},
                "timebase_version": {"type": "string"},
            },
        },
        "prediction": {
            "type": "object",
            "required": [
                "action_id",
                "action_name",
                "finger_id",
                "finger_name",
                "action_confidence",
                "action_uncertainty",
                "finger_confidence",
                "finger_uncertainty",
            ],
            "properties": {
                "action_id": {"type": "integer"},
                "action_name": {"type": "string"},
                "finger_id": {"type": "integer"},
                "finger_name": {"type": "string"},
                "action_confidence": {"type": "number"},
                "action_uncertainty": {"type": "number"},
                "finger_confidence": {"type": "number"},
                "finger_uncertainty": {"type": "number"},
            },
        },
        "safety": {
            "type": "object",
            "required": [
                "base_threshold",
                "adaptive_threshold",
                "allow_actuation",
                "stability_frames",
                "stability_ok",
                "velocity",
            ],
            "properties": {
                "base_threshold": {"type": "number"},
                "adaptive_threshold": {"type": "number"},
                "allow_actuation": {"type": "boolean"},
                "stability_frames": {"type": "integer"},
                "stability_ok": {"type": "boolean"},
                "velocity": {"type": "number"},
            },
        },
        "diagnostics": {
            "type": "object",
            "required": [
                "latency_ms",
                "fps_target",
                "fps_actual",
                "health_score",
                "lsl_connected",
                "artifact_suppression",
                "notes",
            ],
            "properties": {
                "latency_ms": {"type": "number"},
                "fps_target": {"type": "number"},
                "fps_actual": {"type": "number"},
                "health_score": {"type": "number"},
                "lsl_connected": {"type": "boolean"},
                "artifact_suppression": {"type": ["boolean", "null"]},
                "notes": {"type": "string"},
                "smoothing_enabled": {"type": "boolean"},
                "smoothing_method": {"type": "string"},
                "smoothing_window": {"type": "integer"},
                "hysteresis_enabled": {"type": "boolean"},
                "hysteresis_frames": {"type": "integer"},
                "threshold_action": {"type": "number"},
                "threshold_finger": {"type": "number"},
                "adjacency_enabled": {"type": "boolean"},
                "decision_reason": {"type": "string"},
                "raw_top_action_id": {"type": "integer"},
                "raw_top_finger_id": {"type": "integer"},
                "committed_action_id": {"type": "integer"},
                "committed_finger_id": {"type": "integer"},
                "smoothed_action_id": {"type": "integer"},
                "smoothed_finger_id": {"type": "integer"},
                "frames_in_state": {"type": "integer"},
            },
        },
        "telemetry": {
            "type": ["object", "null"],
            "properties": {
                "servo_rail_v": {"type": ["number", "null"]},
                "rail_current_a": {"type": ["number", "null"]},
                "thermal_c": {"type": ["number", "null"]},
                "fingertip_contact": {"type": ["boolean", "null"]},
            },
        },
        "nnvis": {"type": ["object", "null"]},
    },
}

STATUS_SCHEMA = {
    "type": "object",
    "required": ["type", "ts_utc", "level", "message", "details"],
    "properties": {
        "type": {"const": "status"},
        "ts_utc": {"type": "string"},
        "level": {"type": "string", "enum": ["info", "warning", "error"]},
        "message": {"type": "string"},
        "details": {"type": "object"},
    },
}


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def sample_tick():
    return {
        "type": "tick",
        "ts_utc": _now_iso(),
        "mode": "replay",
        "session": {
            "subject_id": "1-M17",
            "experiment_hash": "abc123def456",
            "window_index": 1234,
            "window_start_s": 12.3,
            "window_end_s": 12.55,
            "timebase_version": "absolute_v1",
        },
        "prediction": {
            "action_id": 1,
            "action_name": "OPEN",
            "finger_id": 2,
            "finger_name": "INDEX",
            "action_confidence": 0.82,
            "action_uncertainty": 0.05,
            "finger_confidence": 0.77,
            "finger_uncertainty": 0.08,
        },
        "safety": {
            "base_threshold": 0.75,
            "adaptive_threshold": 0.79,
            "allow_actuation": True,
            "stability_frames": 3,
            "stability_ok": True,
            "velocity": 0.78,
        },
        "diagnostics": {
            "latency_ms": 12.4,
            "fps_target": 20,
            "fps_actual": 19.6,
            "health_score": 0.91,
            "lsl_connected": False,
            "artifact_suppression": None,
            "notes": "",
            "smoothing_enabled": True,
            "smoothing_method": "vote",
            "smoothing_window": 5,
            "hysteresis_enabled": True,
            "hysteresis_frames": 3,
            "threshold_action": 0.75,
            "threshold_finger": 0.75,
            "adjacency_enabled": True,
            "decision_reason": "vote_commit",
            "raw_top_action_id": 1,
            "raw_top_finger_id": 2,
            "committed_action_id": 1,
            "committed_finger_id": 2,
            "smoothed_action_id": 1,
            "smoothed_finger_id": 2,
            "frames_in_state": 3,
        },
        "telemetry": {
            "servo_rail_v": None,
            "rail_current_a": None,
            "thermal_c": None,
            "fingertip_contact": None,
        },
    }


def sample_status():
    return {
        "type": "status",
        "ts_utc": _now_iso(),
        "level": "info",
        "message": "LSL stream not found; staying idle.",
        "details": {},
    }


def schema_bundle():
    return {
        "tick": TICK_SCHEMA,
        "status": STATUS_SCHEMA,
        "sample_tick": sample_tick(),
        "sample_status": sample_status(),
    }
