from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

SCHEMA_VERSION = "1.0"
TIMEBASE_VERSION = "absolute_v1"
DEFAULT_TARGET_FS = 256.0


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
        "ENABLE_PLOT": True,
        "PLOT_SCALE_MODE": "fixed",
        "PLOT_FIXED_YLIM_MIN": -200.0,
        "PLOT_FIXED_YLIM_MAX": 200.0,
        "PLOT_ROBUST_WINDOW_SEC": 5.0,
        "PLOT_ROBUST_EMA": 0.2,
        "PLOT_REFERENCE_OVERLAY": False,
        "PLOT_WINDOW_SEC": 5.0,
        "EVENT_MARKING_ENABLED": True,
        "EVENT_KEYMAP": "space:mark,1:thumb,2:index,3:middle,4:ring,5:pinky,o:open,c:close,r:rest",
        "SAMPLING_RATE": 256,
        "CHANNELS": 4,
        "RAW_QUEUE_MAXSIZE": 4096,
        "RAW_SHARD_SAMPLES": 2048,
        "TIMEBASE_VERSION": TIMEBASE_VERSION,
        "LSL_STREAM_NAME": "Muse2-EEG",
        "LSL_STREAM_TYPE": "EEG",
        "raw_dir": "data/raw",
        "subject_id": None,
        "session_id": None,
        "force_new_session": False,
        "init_only": False,
    }


def default_step1b_settings() -> Dict[str, Any]:
    return {
        "session_dir": None,
        "features": None,
        "events": None,
        "subject_id": "2-M16",
        "target_fs": DEFAULT_TARGET_FS,
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
        "subject_id": "2-M16",
        "epochs": 60,
        "batch_size": 64,
        "lr": 0.001,
        "seed": 42,
        "loss_action_weight": 1.0,
        "rest_weight": 0.8,
        "test_size": 0.2,
        "non_rest_only": False,
        "save_model": "finger_action_model.pt",
        "save_scaler": "scaler.npz",
        "save_preds": "test_predictions.npz",
        "N_FINGERS": 6,
        "N_ACTIONS": 3,
    }


def default_infer_settings() -> Dict[str, Any]:
    return {
        "model_path": "models/finger_action_model.pt",
        "scaler_path": "scaler.npz",
        "stream_name": "Muse2-EEG",
        "stream_type": "EEG",
        "window_sec": 0.25,
        "hop_sec": 0.05,
        "target_fs": DEFAULT_TARGET_FS,
        "allow_drop": False,
        "latency_threshold_ms": 250.0,
        "latency_policy": "warn",
        "log_every": 1.0,
        "enable_actuation": False,
        "bluetooth_target": "",
        "no_file_io": False,
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
    settings = dict(settings or {})
    if subject_id:
        settings["subject_id"] = subject_id
        if "DEFAULT_SUBJECT_ID" in settings:
            settings["DEFAULT_SUBJECT_ID"] = subject_id
    if "window_sec" in settings and "target_fs" in settings:
        try:
            window_sec = float(settings.get("window_sec"))
            target_fs = float(settings.get("target_fs"))
            window_samples = int(round(window_sec * target_fs))
        except Exception:
            window_samples = 0
        if window_samples < 1:
            settings["target_fs"] = DEFAULT_TARGET_FS
            try:
                window_sec = float(settings.get("window_sec", 0.0))
                target_fs = float(settings.get("target_fs"))
                window_samples = int(round(window_sec * target_fs))
            except Exception:
                window_samples = 0
            if window_samples < 1:
                settings["window_sec"] = 0.25
    return ConfigFile(
        schema_version=SCHEMA_VERSION,
        created_at=now_utc_iso(),
        project_name=project_name,
        subject_id=subject_id,
        session_id=session_id,
        timebase_version=timebase_version,
        settings=settings,
    )
