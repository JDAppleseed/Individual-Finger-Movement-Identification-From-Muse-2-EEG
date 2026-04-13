from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from utils.default_recipe import (
    ARTIFACT_DEFAULTS,
    EVAL_RECIPE_DEFAULTS,
    TRAIN_RECIPE_DEFAULTS,
    WINDOW_EXTRACTION_DEFAULTS,
)
from utils.step7_config import default_step7_settings

SCHEMA_VERSION = "1.0"
TIMEBASE_VERSION = "absolute_v1"
DEFAULT_TARGET_FS = float(WINDOW_EXTRACTION_DEFAULTS["target_fs"])


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
        "subject_id": None,
        "target_fs": DEFAULT_TARGET_FS,
        "allow_gaps": False,
        "allow_partial": False,
        "ignore_misalignment": False,
        "WINDOW_SEC": float(WINDOW_EXTRACTION_DEFAULTS["window_sec"]),
        "SOURCE_FS_DEFAULT": 256,
        "TARGET_FS_DEFAULT": DEFAULT_TARGET_FS,
        "WINDOW_SEC_DEFAULT": float(WINDOW_EXTRACTION_DEFAULTS["window_sec"]),
        "STEP_SEC": float(WINDOW_EXTRACTION_DEFAULTS["step_sec"]),
        "PAD_SEC": float(WINDOW_EXTRACTION_DEFAULTS["pad_sec"]),
        "GAP_THRESHOLD_SEC": float(WINDOW_EXTRACTION_DEFAULTS["gap_threshold_sec"]),
        "DEDUP_POLICY": str(WINDOW_EXTRACTION_DEFAULTS["dedupe_policy"]),
        "INTERPOLATION_POLICY": str(WINDOW_EXTRACTION_DEFAULTS["interpolation_policy"]),
        "LABEL_GATED": True,
        "REST_POLICY": str(WINDOW_EXTRACTION_DEFAULTS["rest_policy"]),
        "KEEP_BASELINE_REST_EVENTS": int(
            WINDOW_EXTRACTION_DEFAULTS["keep_baseline_rest_events"]
        ),
        "MIN_OVERLAP_RATIO": float(WINDOW_EXTRACTION_DEFAULTS["min_overlap_ratio"]),
        "GUARD_BAND_SEC": float(WINDOW_EXTRACTION_DEFAULTS["guard_band_sec"]),
        "ARTIFACT_MIN_OVERLAP_FRAC": float(
            WINDOW_EXTRACTION_DEFAULTS["artifact_min_overlap_frac"]
        ),
        "OUT_FILE": "eeg_windows.csv",
        "OUT_NPZ": str(ARTIFACT_DEFAULTS["windows_npz"]),
    }


def default_train_settings() -> Dict[str, Any]:
    return {
        "npz": str(ARTIFACT_DEFAULTS["windows_npz"]),
        "subject_id": None,
        "epochs": int(TRAIN_RECIPE_DEFAULTS["epochs"]),
        "batch_size": int(TRAIN_RECIPE_DEFAULTS["batch_size"]),
        "lr": float(TRAIN_RECIPE_DEFAULTS["lr"]),
        "seed": int(TRAIN_RECIPE_DEFAULTS["seed"]),
        "loss_action_weight": float(TRAIN_RECIPE_DEFAULTS["loss_action_weight"]),
        "rest_weight": float(TRAIN_RECIPE_DEFAULTS["rest_weight"]),
        "action_weights": TRAIN_RECIPE_DEFAULTS["action_weights"],
        "rest_balance_mode": str(TRAIN_RECIPE_DEFAULTS["rest_balance_mode"]),
        "active_finger_head": bool(TRAIN_RECIPE_DEFAULTS["active_finger_head"]),
        "finger_applicability_head": bool(
            TRAIN_RECIPE_DEFAULTS["finger_applicability_head"]
        ),
        "rest_finger_loss_weight": float(TRAIN_RECIPE_DEFAULTS["rest_finger_loss_weight"]),
        "applicability_loss_weight": float(TRAIN_RECIPE_DEFAULTS["applicability_loss_weight"]),
        "finger_weights": TRAIN_RECIPE_DEFAULTS["finger_weights"],
        "window_preprocess": str(TRAIN_RECIPE_DEFAULTS["window_preprocess"]),
        "test_size": float(TRAIN_RECIPE_DEFAULTS["test_size"]),
        "calibration_size": float(TRAIN_RECIPE_DEFAULTS["calibration_size"]),
        "threshold_applicability": float(TRAIN_RECIPE_DEFAULTS["threshold_applicability"]),
        "split_mode": str(TRAIN_RECIPE_DEFAULTS["split_mode"]),
        "aux_rest_session_policy": str(TRAIN_RECIPE_DEFAULTS["aux_rest_session_policy"]),
        "purge_seconds": float(TRAIN_RECIPE_DEFAULTS["purge_seconds"]),
        "hop_seconds": TRAIN_RECIPE_DEFAULTS["hop_seconds"],
        "window_idx_leak_threshold": float(TRAIN_RECIPE_DEFAULTS["window_idx_leak_threshold"]),
        "strict_leakage": bool(TRAIN_RECIPE_DEFAULTS["strict_leakage"]),
        "non_rest_only": bool(TRAIN_RECIPE_DEFAULTS["non_rest_only"]),
        "device": "auto",
        "num_workers": 0,
        "pin_memory": False,
        "save_model": str(ARTIFACT_DEFAULTS["model"]),
        "save_scaler": str(ARTIFACT_DEFAULTS["scaler"]),
        "save_preds": str(ARTIFACT_DEFAULTS["preds"]),
        "save_temperature": str(ARTIFACT_DEFAULTS["temperature"]),
        "run_dir": None,
    }


def default_evaluate_settings() -> Dict[str, Any]:
    return {
        "run_dir": None,
        "max_samples": None,
        "batch_size": 256,
        "device": "auto",
        "amp_mode": "off",
        "split_seed": int(EVAL_RECIPE_DEFAULTS["split_seed"]),
        "save_manifest": None,
        "no_manifest": False,
        "export_test_pred": False,
        "deterministic": bool(EVAL_RECIPE_DEFAULTS["deterministic"]),
        "smooth": bool(EVAL_RECIPE_DEFAULTS["smooth"]),
        "smooth_action_only": bool(EVAL_RECIPE_DEFAULTS["smooth_action_only"]),
        "smooth_method": str(EVAL_RECIPE_DEFAULTS["smooth_method"]),
        "smooth_window": int(EVAL_RECIPE_DEFAULTS["smooth_window"]),
        "hysteresis": bool(EVAL_RECIPE_DEFAULTS["hysteresis"]),
        "hysteresis_frames": int(EVAL_RECIPE_DEFAULTS["hysteresis_frames"]),
        "threshold_action": float(EVAL_RECIPE_DEFAULTS["threshold_action"]),
        "threshold_finger": float(EVAL_RECIPE_DEFAULTS["threshold_finger"]),
        "threshold_applicability": float(EVAL_RECIPE_DEFAULTS["threshold_applicability"]),
        "adjacency": bool(EVAL_RECIPE_DEFAULTS["adjacency"]),
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
    return default_step7_settings()


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
