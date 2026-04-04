from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


def _validate_bool(
    settings: Dict[str, Any], key: str, errors: List[str]
) -> None:
    if key in settings and not isinstance(settings.get(key), bool):
        errors.append(f"{key} must be a boolean.")


def _validate_int_min(
    settings: Dict[str, Any], key: str, minimum: int, errors: List[str]
) -> None:
    if key not in settings:
        return
    try:
        value = int(settings.get(key))
    except Exception:
        errors.append(f"{key} must be an integer.")
        return
    if value < minimum:
        errors.append(f"{key} must be >= {minimum}.")


def _validate_float_min(
    settings: Dict[str, Any], key: str, minimum: float, errors: List[str]
) -> None:
    if key not in settings:
        return
    try:
        value = float(settings.get(key))
    except Exception:
        errors.append(f"{key} must be numeric.")
        return
    if value < minimum:
        errors.append(f"{key} must be >= {minimum}.")


def _validate_unit_interval(
    settings: Dict[str, Any], key: str, errors: List[str]
) -> None:
    if key not in settings:
        return
    try:
        value = float(settings.get(key))
    except Exception:
        errors.append(f"{key} must be numeric.")
        return
    if value < 0.0 or value > 1.0:
        errors.append(f"{key} must be in [0.0, 1.0].")


def validate_train_record(settings: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    mode = settings.get("MODE")
    if mode and mode != "train_record":
        errors.append("MODE must be train_record for lossless capture.")
    if "ENABLE_ACTUATION" in settings:
        errors.append(
            "Legacy key 'ENABLE_ACTUATION' is not supported; migrate to 'enable_actuation'."
        )
    if "enable_actuation" in settings:
        if not isinstance(settings.get("enable_actuation"), bool):
            errors.append("enable_actuation must be a boolean.")
        elif settings.get("enable_actuation") is True:
            errors.append("enable_actuation can only be true for the live_infer step.")
    if settings.get("ALLOW_DROP"):
        errors.append("ALLOW_DROP is forbidden in train_record mode.")
    if not settings.get("SAVE_RAW", True):
        errors.append("SAVE_RAW must remain enabled for lossless capture.")
    if settings.get("ENABLE_FEATURES") or settings.get("ENABLE_INFERENCE"):
        warnings.append("Feature/inference flags will be ignored in train_record mode.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_live_infer(settings: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    if settings.get("MODE") and settings.get("MODE") != "live_infer":
        errors.append("MODE must be live_infer for deployment.")
    if settings.get("ALLOW_DROP"):
        warnings.append("ALLOW_DROP is enabled; dropped windows will be logged.")
    if "ENABLE_ACTUATION" in settings:
        errors.append(
            "Legacy key 'ENABLE_ACTUATION' is not supported; migrate to 'enable_actuation'."
        )
    for key in (
        "enable_actuation",
        "modulate_actuation_speed",
        "use_inference_engine",
        "postprocess",
        "smoothing_enabled",
        "hysteresis_enabled",
        "adjacency_enabled",
        "no_file_io",
        "live_quality_enabled",
        "rest_bias_correction_enabled",
    ):
        _validate_bool(settings, key, errors)

    for key in ("mc_passes", "serial_baud", "smoothing_window", "hysteresis_frames", "actuation_stability"):
        _validate_int_min(settings, key, 1, errors)
    for key in ("actuation_cooldown_ms", "actuation_repeat_ms"):
        _validate_int_min(settings, key, 0, errors)

    for key in (
        "threshold_action",
        "threshold_finger",
        "threshold_applicability",
        "uncertainty_base_threshold",
        "actuation_min_prob",
        "actuation_min_speed",
        "hysteresis_margin",
        "finger_delta",
    ):
        _validate_unit_interval(settings, key, errors)
    for key in ("uncertainty_weight", "actuation_speed_gamma"):
        _validate_float_min(settings, key, 0.0, errors)
    for key in (
        "input_clip_abs_z",
        "bad_channel_rms_z",
        "bad_channel_abs_p95_z",
        "rest_bias_strength",
    ):
        _validate_float_min(settings, key, 0.0, errors)
    for key in ("bad_channel_clipped_frac", "bad_window_clipped_frac"):
        _validate_unit_interval(settings, key, errors)
    _validate_int_min(settings, "bad_window_max_masked_channels", 0, errors)
    _validate_int_min(settings, "rest_bias_min_windows", 1, errors)

    if "smoothing_method" in settings:
        value = str(settings.get("smoothing_method"))
        if value not in {"vote", "ema"}:
            errors.append("smoothing_method must be 'vote' or 'ema'.")
    if "finger_mode" in settings:
        value = str(settings.get("finger_mode"))
        if value not in {"raw", "smooth"}:
            errors.append("finger_mode must be 'raw' or 'smooth'.")
    if "latency_policy" in settings:
        value = str(settings.get("latency_policy"))
        if value not in {"ignore", "warn", "enforce"}:
            errors.append("latency_policy must be 'ignore', 'warn', or 'enforce'.")

    if settings.get("enable_actuation") and not str(settings.get("serial_port") or "").strip():
        warnings.append(
            "enable_actuation is true and serial_port is blank; Step 7 will auto-detect a port. Set serial_port explicitly for live hardware when possible."
        )
    if settings.get("no_file_io"):
        warnings.append(
            "no_file_io is enabled; prediction logs and other live outputs will not be written."
        )
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_train(settings: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    _validate_bool(settings, "active_finger_head", errors)
    _validate_bool(settings, "finger_applicability_head", errors)
    if "rest_balance_mode" in settings:
        value = str(settings.get("rest_balance_mode"))
        if value not in {"none", "session_equalized", "core_event_equalized"}:
            errors.append(
                "rest_balance_mode must be 'none', 'session_equalized', or 'core_event_equalized'."
            )
    if "rest_finger_loss_weight" in settings:
        try:
            if float(settings.get("rest_finger_loss_weight")) < 0.0:
                errors.append("rest_finger_loss_weight must be >= 0.")
        except Exception:
            errors.append("rest_finger_loss_weight must be numeric.")
    if "applicability_loss_weight" in settings:
        try:
            if float(settings.get("applicability_loss_weight")) < 0.0:
                errors.append("applicability_loss_weight must be >= 0.")
        except Exception:
            errors.append("applicability_loss_weight must be numeric.")
    if "threshold_applicability" in settings:
        _validate_unit_interval(settings, "threshold_applicability", errors)
    if "action_weights" in settings and settings.get("action_weights") not in {None, "", "none", "null"}:
        raw = settings.get("action_weights")
        try:
            if isinstance(raw, str) and raw.strip().startswith(("[", "{")):
                parsed = json.loads(raw)
            elif isinstance(raw, str):
                parsed = [float(v) for v in raw.replace(" ", ",").split(",") if v]
            else:
                parsed = raw
            if isinstance(parsed, dict):
                pass
            elif not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
                errors.append("action_weights must contain exactly 3 values for REST, OPEN, CLOSE.")
        except Exception:
            errors.append("action_weights must be parseable as CSV/JSON with 3 values.")
    if "window_preprocess" in settings:
        value = str(settings.get("window_preprocess"))
        if value not in {"none", "center", "center_detrend"}:
            errors.append(
                "window_preprocess must be one of: none, center, center_detrend."
            )
    if "calibration_size" in settings:
        try:
            calibration_size = float(settings.get("calibration_size"))
            if calibration_size < 0.0 or calibration_size >= 1.0:
                errors.append("calibration_size must be in [0.0, 1.0).")
        except Exception:
            errors.append("calibration_size must be numeric.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_topomaps(settings: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
    try:
        band_low = float(settings.get("band_low", 8.0))
        band_high = float(settings.get("band_high", 12.0))
        if band_low >= band_high:
            errors.append("band_low must be strictly less than band_high.")
    except Exception:
        errors.append("band_low and band_high must be numeric.")

    if "blur_sigma" in settings:
        try:
            if float(settings.get("blur_sigma")) < 0.0:
                errors.append("blur_sigma must be >= 0.")
        except Exception:
            errors.append("blur_sigma must be numeric.")

    if "robust_quantile" in settings:
        try:
            robust_quantile = float(settings.get("robust_quantile"))
            if robust_quantile < 0.0 or robust_quantile >= 0.5:
                errors.append("robust_quantile must be in [0.0, 0.5).")
        except Exception:
            errors.append("robust_quantile must be numeric.")

    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_step_settings(step_id: str, settings: Dict[str, Any]) -> ValidationResult:
    if step_id == "step1":
        return validate_train_record(settings)
    if step_id == "train":
        return validate_train(settings)
    if step_id == "topomaps":
        return validate_topomaps(settings)
    if step_id == "infer":
        return validate_live_infer(settings)
    errors: List[str] = []
    warnings: List[str] = []
    if "ENABLE_ACTUATION" in settings:
        errors.append(
            "Legacy key 'ENABLE_ACTUATION' is not supported; migrate to 'enable_actuation'."
        )
    if "enable_actuation" in settings:
        if not isinstance(settings.get("enable_actuation"), bool):
            errors.append("enable_actuation must be a boolean.")
        elif settings.get("enable_actuation") is True:
            errors.append("enable_actuation can only be true for the live_infer step.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
