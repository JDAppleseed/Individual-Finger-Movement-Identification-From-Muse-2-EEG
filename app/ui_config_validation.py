from __future__ import annotations

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
            "Legacy key 'ENABLE_ACTUATION' is not supported; use 'enable_actuation'."
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
            "Legacy key 'ENABLE_ACTUATION' is not supported; use 'enable_actuation'."
        )
    if "enable_actuation" in settings and not isinstance(
        settings.get("enable_actuation"), bool
    ):
        errors.append("enable_actuation must be a boolean.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)


def validate_step_settings(step_id: str, settings: Dict[str, Any]) -> ValidationResult:
    if step_id == "step1":
        return validate_train_record(settings)
    if step_id == "infer":
        return validate_live_infer(settings)
    errors: List[str] = []
    warnings: List[str] = []
    if "ENABLE_ACTUATION" in settings:
        errors.append(
            "Legacy key 'ENABLE_ACTUATION' is not supported; use 'enable_actuation'."
        )
    if "enable_actuation" in settings:
        if not isinstance(settings.get("enable_actuation"), bool):
            errors.append("enable_actuation must be a boolean.")
        elif settings.get("enable_actuation") is True:
            errors.append("enable_actuation can only be true for the live_infer step.")
    return ValidationResult(ok=not errors, errors=errors, warnings=warnings)
    return ValidationResult(ok=True, errors=[], warnings=[])
