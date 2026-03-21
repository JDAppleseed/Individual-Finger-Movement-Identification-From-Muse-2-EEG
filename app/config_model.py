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
        "PLOT_FIXED_UV": 200.0,
        "PLOT_FIXED_YLIM_MIN": -200.0,
        "PLOT_FIXED_YLIM_MAX": 200.0,
        "PLOT_ROBUST_WINDOW_SEC": 5.0,
        "PLOT_ROBUST_EMA": 0.2,
        "PLOT_REFERENCE_OVERLAY": False,
        "PLOT_REFERENCE_LINES": False,
        "PLOT_WINDOW_SEC": 5.0,
        "PLOT_FPS": 20.0,
        "PLOT_DISPLAY_FS": 64.0,
        "PLOT_STARTUP_TIMEOUT_S": 0.75,
        "PLOT_CHANNEL_SPACING_UV": 120.0,
        "EVENT_MARKING_ENABLED": True,
        "EVENT_KEYMAP": "space:mark,1:thumb,2:index,3:middle,4:ring,5:pinky,o:open,c:close,r:rest",
        "SAMPLING_RATE": 256,
        "CHANNELS": 4,
        "RAW_QUEUE_MAXSIZE": 4096,
        "RAW_SHARD_SAMPLES": 2048,
        "RAW_SHARD_FLUSH_INTERVAL_S": 2.0,
        "MAX_BACKPRESSURE_S": 3.0,
        "QUEUE_PUT_TIMEOUT_S": 0.1,
        "LSL_RESOLVE_TIMEOUT": 2.0,
        "LSL_INLET_MAX_BUFLEN_SEC": 2,
        "LSL_INLET_MAX_CHUNKLEN": 1,
        "HEARTBEAT_INTERVAL_S": 5.0,
        "NO_SAMPLE_TIMEOUT_S": 5.0,
        "WRITE_STALL_TIMEOUT_S": 5.0,
        "WARMUP_SAMPLE_COUNT": 3,
        "WARMUP_TIMEOUT_S": 8.0,
        "EVENT_FLUSH_INTERVAL_S": 1.0,
        "HARD_STOP_AFTER_UNHEALTHY_S": 8.0,
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
        "REST_POLICY": "label_gated",
        "KEEP_BASELINE_REST_EVENTS": -1,
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
        "rest_weight": 1.0,
        "action_weights": None,
        "rest_balance_mode": "core_event_equalized",
        "active_finger_head": True,
        "finger_applicability_head": True,
        "rest_finger_loss_weight": 0.0,
        "applicability_loss_weight": 0.5,
        "finger_weights": None,
        "window_preprocess": "center_detrend",
        "test_size": 0.2,
        "calibration_size": 0.1,
        "threshold_applicability": 0.5,
        "split_mode": "group_trial",
        "purge_seconds": 0.0,
        "hop_seconds": None,
        "window_idx_leak_threshold": 0.65,
        "strict_leakage": False,
        "non_rest_only": False,
        "device": "auto",
        "num_workers": 0,
        "pin_memory": False,
        "save_model": "finger_action_model.pt",
        "save_scaler": "scaler.npz",
        "save_preds": "test_predictions.npz",
        "save_temperature": "temperature_scaling.json",
        "run_dir": None,
    }


def default_evaluate_settings() -> Dict[str, Any]:
    return {
        "run_dir": None,
        "max_samples": None,
        "batch_size": 256,
        "device": "auto",
        "amp_mode": "off",
        "split_seed": 42,
        "save_manifest": None,
        "no_manifest": False,
        "export_test_pred": False,
        "deterministic": True,
        "smooth": False,
        "smooth_action_only": False,
        "smooth_method": "vote",
        "smooth_window": 5,
        "hysteresis": False,
        "hysteresis_frames": 3,
        "threshold_action": 0.75,
        "threshold_finger": 0.75,
        "threshold_applicability": 0.5,
        "adjacency": False,
    }


def default_evaluate_deepchecks_settings() -> Dict[str, Any]:
    return {
        "run_dir": None,
        "max_samples": None,
        "batch_size": 1024,
        "device": "auto",
        "amp_mode": "off",
        "split_mode": None,
        "purge_seconds": 0.0,
        "hop_seconds": None,
    }


def default_evaluate_figures_settings() -> Dict[str, Any]:
    return {
        "run_dir": None,
        "show_plots": False,
        "batch_size": 1024,
        "device": "auto",
        "amp_mode": "off",
        "mc_samples": 30,
        "seed": 42,
    }


def default_topomap_settings() -> Dict[str, Any]:
    return {
        "session_dir": None,
        "npz": None,
        "out_dir": None,
        "suite": True,
        "group_by": "action",
        "metric": "log_absolute",
        "split_halves": False,
        "include_none": False,
        "band_low": 8.0,
        "band_high": 12.0,
        "blur_sigma": 0.0,
        "robust_quantile": 0.05,
        "out": None,
        "summary_out": None,
        "summary_json_out": None,
    }


def default_infer_settings() -> Dict[str, Any]:
    return {
        "model_path": "models/finger_action_model.pt",
        "scaler_path": "scaler.npz",
        "stream_name": "Muse2-EEG",
        "stream_type": "EEG",
        "LIVE_VIZ_ENABLED": False,
        "LIVE_VIZ_FPS": 2,
        "window_sec": 0.25,
        "hop_sec": 0.05,
        "target_fs": DEFAULT_TARGET_FS,
        "allow_drop": False,
        "latency_threshold_ms": 750.0,
        "latency_policy": "warn",
        "log_every": 5.0,
        "enable_actuation": False,
        "serial_port": None,
        "serial_baud": 9600,
        "bluetooth_target": "",
        "no_file_io": False,
        "actuation_min_prob": 0.2,
        "actuation_stability": 3,
        "actuation_cooldown_ms": 250,
        "actuation_repeat_ms": 500,
        "actuation_min_speed": 0.5,
        "modulate_actuation_speed": True,
        "actuation_speed_gamma": 1.0,
        "postprocess": True,
        "smoothing_enabled": True,
        "smoothing_method": "ema",
        "smoothing_window": 5,
        "hysteresis_enabled": False,
        "hysteresis_frames": 3,
        "threshold_action": 0.05,
        "threshold_finger": 0.20,
        "threshold_applicability": 0.40,
        "adjacency_enabled": False,
        "hysteresis_margin": 0.05,
        "finger_delta": 0.05,
        "finger_mode": "raw",
        "use_inference_engine": False,
        "mc_passes": 10,
        "uncertainty_base_threshold": 0.75,
        "uncertainty_weight": 0.5,
        "pred_log": None,
    }


def default_live_review_settings() -> Dict[str, Any]:
    return {
        "session_dir": None,
        "pred_log": None,
        "out_json": "live_prediction_summary.json",
        "segments_csv": "predicted_segments.csv",
        "review_csv": "predicted_segments_review.csv",
        "video_offset_s": 0.0,
        "short_segment_sec": 0.25,
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
