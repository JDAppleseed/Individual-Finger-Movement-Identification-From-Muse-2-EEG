from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


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
    if "enable_actuation" in settings and not isinstance(
        settings.get("enable_actuation"), bool
    ):
        errors.append("enable_actuation must be a boolean.")
    if "modulate_actuation_speed" in settings and not isinstance(
        settings.get("modulate_actuation_speed"), bool
    ):
        errors.append("modulate_actuation_speed must be a boolean.")
    if "use_inference_engine" in settings and not isinstance(
        settings.get("use_inference_engine"), bool
    ):
        errors.append("use_inference_engine must be a boolean.")
    if "mc_passes" in settings:
        try:
            if int(settings.get("mc_passes")) < 1:
                errors.append("mc_passes must be >= 1.")
        except Exception:
            errors.append("mc_passes must be an integer.")
    for key in (
        "uncertainty_base_threshold",
        "uncertainty_weight",
        "actuation_speed_gamma",
    ):
        if key in settings:
            try:
                float(settings.get(key))
            except Exception:
                errors.append(f"{key} must be numeric.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_train(settings: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []
    warnings: List[str] = []
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


def validate_step_settings(step_id: str, settings: Dict[str, Any]) -> ValidationResult:
    if step_id == "step1":
        return validate_train_record(settings)
    if step_id == "train":
        return validate_train(settings)
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
    return ValidationResult(ok=True, errors=[], warnings=[])
