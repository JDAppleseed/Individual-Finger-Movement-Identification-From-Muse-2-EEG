from __future__ import annotations

from typing import Any, Dict

# Canonical recipe source:
# Projects/2-M16/subjects/2-M16/winning_model
#   - model_run/train_config.json
#   - configs/infer.json
#   - configs/pseudo_live.json
#   - session_report/eval_manifest.json
#   - source processed/eeg_windows.npz metadata

WINNING_RECIPE_SOURCE: Dict[str, str] = {
    "subject_id": "2-M16",
    "session_id": "combined_20260319_081200_pruned_rest_events_0_1_2",
    "run_id": "20260319_075520",
    "snapshot_dir": "Projects/2-M16/subjects/2-M16/winning_model",
}

ARTIFACT_DEFAULTS: Dict[str, str] = {
    "windows_npz": "eeg_windows.npz",
    "model": "finger_action_model.pt",
    "scaler": "scaler.npz",
    "preds": "test_predictions.npz",
    "temperature": "temperature_scaling.json",
}

WINDOW_EXTRACTION_DEFAULTS: Dict[str, Any] = {
    "target_fs": 256.0,
    "window_sec": 0.25,
    "step_sec": 0.05,
    "pad_sec": 0.05,
    "gap_threshold_sec": 0.25,
    "gap_interp_max_s": 0.05,
    "allow_gap_interp": False,
    "dedupe_policy": "keep_last",
    "interpolation_policy": "np.interp.linear",
    "min_overlap_ratio": 0.20,
    "guard_band_sec": 0.0,
    "artifact_min_overlap_frac": 0.20,
    "rest_policy": "label_gated",
    "keep_baseline_rest_events": -1,
    "rest_subsample_prob": 1.0,
    "rest_subsample_seed": 42,
    "rest_max_windows": None,
}

# The winning train artifact recorded 0.5 in train_config.json. We keep that
# value in Step 2 defaults so new runs preserve the same metadata and
# reproducibility trail.
HISTORICAL_TRAIN_ARTIFACT_APPLICABILITY_THRESHOLD = 0.5

# The winning deployed recipe actually evaluated and inferred with 0.4, as
# captured in session_report/eval_manifest.json and configs/infer.json. This is
# the canonical postprocess default for new Step 3 / Step 7 configs.
CANONICAL_DEPLOYMENT_APPLICABILITY_THRESHOLD = 0.4

TRAIN_RECIPE_DEFAULTS: Dict[str, Any] = {
    "seed": 43,
    "epochs": 60,
    "batch_size": 64,
    "lr": 0.001,
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
    "threshold_applicability": HISTORICAL_TRAIN_ARTIFACT_APPLICABILITY_THRESHOLD,
    "split_mode": "group_trial",
    "aux_rest_session_policy": "auto_train_only",
    "purge_seconds": 0.0,
    "hop_seconds": None,
    "window_idx_leak_threshold": 0.65,
    "strict_leakage": False,
    "non_rest_only": False,
    "amp_mode": "off",
}

EVAL_RECIPE_DEFAULTS: Dict[str, Any] = {
    "split_seed": int(TRAIN_RECIPE_DEFAULTS["seed"]),
    "deterministic": True,
    "smooth": False,
    "smooth_action_only": False,
    "smooth_method": "vote",
    "smooth_window": 5,
    "hysteresis": False,
    "hysteresis_frames": 3,
    "threshold_action": 0.75,
    "threshold_finger": 0.75,
    "threshold_applicability": CANONICAL_DEPLOYMENT_APPLICABILITY_THRESHOLD,
    "adjacency": False,
}

LIVE_INFER_RECIPE_DEFAULTS: Dict[str, Any] = {
    "LIVE_EEG_PLOT_ENABLED": True,
    "LIVE_VIZ_ENABLED": False,
    "LIVE_VIZ_FPS": 2.0,
    "window_sec": float(WINDOW_EXTRACTION_DEFAULTS["window_sec"]),
    "hop_sec": float(WINDOW_EXTRACTION_DEFAULTS["step_sec"]),
    "target_fs": float(WINDOW_EXTRACTION_DEFAULTS["target_fs"]),
    "alignment_internal_max_gap_s": 0.06,
    "latency_threshold_ms": 750.0,
    "latency_policy": "warn",
    "allow_drop": False,
    "log_every": 5.0,
    "enable_actuation": False,
    "serial_port": None,
    "serial_baud": 9600,
    "force_no_serial": False,
    "serial_write_timeout_s": 0.03,
    "serial_max_hz": 10.0,
    "serial_settle_s": 1.2,
    "serial_movement_warmup_enabled": False,
    "lsl_acquirer_queue_max_chunks": 32,
    "bluetooth_target": None,
    "no_file_io": False,
    "postprocess": True,
    "smoothing_enabled": True,
    "smoothing_method": "ema",
    "smoothing_window": 5,
    "hysteresis_enabled": False,
    "hysteresis_frames": 3,
    "threshold_action": 0.05,
    "threshold_finger": 0.2,
    "threshold_applicability": CANONICAL_DEPLOYMENT_APPLICABILITY_THRESHOLD,
    "adjacency_enabled": False,
    "hysteresis_margin": 0.05,
    "finger_delta": 0.05,
    "finger_mode": "raw",
    "actuation_min_prob": 0.2,
    "actuation_stability": 3,
    "actuation_cooldown_ms": 250,
    "actuation_repeat_ms": 100,
    "actuation_min_speed": 0.5,
    "modulate_actuation_speed": True,
    "actuation_speed_gamma": 1.0,
    "rest_bias_correction_enabled": False,
    "rest_bias_strength": 1.5,
    "rest_bias_min_windows": 10,
    "live_quality_enabled": True,
    "input_clip_abs_z": 6.0,
    "bad_channel_rms_z": 4.0,
    "bad_channel_abs_p95_z": 6.0,
    "bad_channel_clipped_frac": 0.05,
    "bad_window_clipped_frac": 0.10,
    "bad_window_max_masked_channels": 1,
    "use_inference_engine": False,
    "mc_passes": 10,
    "uncertainty_base_threshold": 0.75,
    "uncertainty_weight": 0.5,
    "parity_capture_enabled": True,
    "parity_capture_max_windows": 128,
    "parity_capture_flush_every": 25,
}

PSEUDO_LIVE_RECIPE_DEFAULTS: Dict[str, Any] = {
    "latency_mode": "ignore",
    "fixed_latency_ms": None,
    "reset_on_trial_change": True,
    "deterministic": True,
}


def extraction_recipe_defaults() -> Dict[str, Any]:
    return dict(WINDOW_EXTRACTION_DEFAULTS)


def train_recipe_defaults() -> Dict[str, Any]:
    return dict(TRAIN_RECIPE_DEFAULTS)


def eval_recipe_defaults() -> Dict[str, Any]:
    return dict(EVAL_RECIPE_DEFAULTS)


def live_infer_recipe_defaults() -> Dict[str, Any]:
    return dict(LIVE_INFER_RECIPE_DEFAULTS)


def pseudo_live_recipe_defaults() -> Dict[str, Any]:
    return dict(PSEUDO_LIVE_RECIPE_DEFAULTS)
