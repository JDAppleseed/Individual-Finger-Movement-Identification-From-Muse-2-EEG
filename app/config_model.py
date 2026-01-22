from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = "1.0"
TIMEBASE_VERSION = "absolute_v1"


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ConfigFile:
    schema_version: str
    created_at: str
    project_name: str
    subject_id: str
    session_id: str
    timebase_version: str
    settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["TIMEBASE_VERSION"] = self.timebase_version
        return data


@dataclass
class SessionSnapshot:
    schema_version: str
    created_at: str
    project_name: str
    subject_id: str
    session_id: str
    timebase_version: str
    steps: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["TIMEBASE_VERSION"] = self.timebase_version
        return data


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def default_step1_settings() -> Dict[str, Any]:
    return {
        "MODE": "train_record",
        "ALLOW_DROP": False,
        "TRAINING_MODE": False,
        "DEMO_MODE": False,
        "ENABLE_PLOT": False,
        "SAVE_TO_DISK": True,
        "SAVE_RAW": True,
        "ENABLE_FEATURES": False,
        "ENABLE_INFERENCE": False,
        "ENABLE_ICA": False,
        "ICA_WARMUP_S": 10.0,
        "ICA_MIN_SAMPLES": 256 * 5,
        "ICA_MIN_VAR": 1e-8,
        "ICA_FAIL_POLICY": "skip",
        "ICA_MAX_RETRIES_PER_SESSION": 1,
        "LOG_ICA_DIAGNOSTICS": True,
        "DATA_STREAM_TIMEOUT_S": 5.0,
        "DATA_STREAM_CHECK_INTERVAL_S": 0.5,
        "PROCESSING_QUEUE_MAXSIZE": 2048,
        "RAW_QUEUE_MAXSIZE": 4096,
        "MAX_BACKPRESSURE_S": 3.0,
        "QUEUE_PUT_TIMEOUT_S": 0.1,
        "RAW_SHARD_SAMPLES": 2048,
        "SAMPLING_RATE": 256,
        "WINDOW_SEC": 0.25,
        "CHANNELS": 4,
        "N_FINGERS": 6,
        "N_ACTIONS": 3,
        "TIMEBASE_VERSION": TIMEBASE_VERSION,
        "MODEL_PATH": "finger_action_model.pt",
        "SCALER_PATH": "scaler.save",
        "BASE_CONF_THRESH": 0.75,
        "UNCERTAINTY_WEIGHT": 0.5,
        "STABILITY_FRAMES": 3,
        "ENABLE_ACTUATION": False,
        "MC_DROPOUT_PASSES": 10,
        "EVENT_MARKING_ENABLED": True,
        "EVENTS_CSV_PATH": None,
        "EVENTS_AUTOSAVE_PATH": None,
        "EVENTS_CHANNEL": "n/a",
        "LSL_STREAM_NAME": "Muse2-EEG",
        "LSL_STREAM_TYPE": "EEG",
        "CSV_OFFLINE_PATH": None,
        "subject_id": None,
        "force_new_session": False,
        "init_only": False,
        "SESSION_ID_OVERRIDE": None,
        "HARD_STOP_AFTER_UNHEALTHY_S": 2.0,
        "FAILED_WRITE_WINDOW_S": 5.0,
        "FAILED_DIR": "data/failed",
        "REQUIRED_LSL_LABELS": ["TP9", "AF7", "AF8", "TP10"],
        "REQUIRE_EXACTLY_4_CHANNELS": True,
        "LIVE_VIZ_ENABLED": False,
        "LIVE_VIZ_FPS": 2,
        "STREAMER_INTERNAL": True,
        "STREAMER_STREAM_NAME": "Muse2-EEG",
        "STREAMER_STREAM_TYPE": "EEG",
        "LABEL_CHECK_ACKNOWLEDGED": False,
        "LABEL_CHECK_FOUND_LABELS": None,
        "LABEL_CHECK_EXPECTED_LABELS": ["TP9", "AF7", "AF8", "TP10"],
    }


def default_step1b_settings() -> Dict[str, Any]:
    return {
        "session_dir": None,
        "features": None,
        "events": None,
        "subject_id": "8-M16",
        "target_fs": None,
        "allow_gaps": False,
        "allow_partial": False,
        "ignore_misalignment": False,
        "WINDOW_SEC": 0.25,
        "SOURCE_FS_DEFAULT": 256,
        "TARGET_FS_DEFAULT": 256.0,
        "WINDOW_SEC_DEFAULT": 0.25,
        "STEP_SEC": 0.05,
        "PAD_SEC": 0.05,
        "GAP_THRESHOLD_SEC": 0.10,
        "DEDUP_POLICY": "keep_last",
        "INTERPOLATION_POLICY": "np.interp.linear",
        "LABEL_GATED": True,
        "KEEP_BASELINE_REST_EVENTS": 2,
        "MIN_OVERLAP_RATIO": 0.20,
        "GUARD_BAND_SEC": 0.00,
        "ARTIFACT_MIN_OVERLAP_FRAC": 0.20,
        "OUT_FILE": "eeg_windows.csv",
        "OUT_NPZ": "eeg_windows.npz",
    }


def default_train_settings() -> Dict[str, Any]:
    return {
        "npz": "eeg_windows.npz",
        "subject_id": "8-M16",
        "epochs": 60,
        "batch_size": 64,
        "lr": 0.001,
        "seed": 42,
        "loss_action_weight": 1.0,
        "rest_weight": 0.2,
        "test_size": 0.2,
        "non_rest_only": False,
        "save_model": "finger_action_model.pt",
        "save_scaler": "scaler.save",
        "save_preds": "test_predictions.npz",
        "N_FINGERS": 6,
        "N_ACTIONS": 3,
    }


def default_infer_settings() -> Dict[str, Any]:
    return {
        "model_path": "models/finger_action_model.pt",
        "scaler_path": "scaler.save",
        "stream_name": "Muse2-EEG",
        "stream_type": "EEG",
        "window_sec": 0.25,
        "hop_sec": 0.05,
        "target_fs": 256.0,
        "allow_drop": False,
        "latency_threshold_ms": 250.0,
        "latency_policy": "warn",
        "log_every": 1.0,
        "enable_actuation": False,
        "bluetooth_target": "",
        "record_raw": False,
    }


def default_preprocess_settings() -> Dict[str, Any]:
    return {}


def default_export_settings() -> Dict[str, Any]:
    return {}


def build_config(
    project_name: str,
    subject_id: str,
    session_id: str,
    settings: Dict[str, Any],
    timebase_version: str = TIMEBASE_VERSION,
) -> ConfigFile:
    return ConfigFile(
        schema_version=SCHEMA_VERSION,
        created_at=now_utc_iso(),
        project_name=project_name,
        subject_id=subject_id,
        session_id=session_id,
        timebase_version=timebase_version,
        settings=settings,
    )
