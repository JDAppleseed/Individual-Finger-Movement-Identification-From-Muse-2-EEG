from __future__ import annotations

from typing import Any, Mapping, Optional

LIVE_PREFLIGHT_REPORT_VERSION = 1
LIVE_PREFLIGHT_LAUNCH_PLAN_SCHEMA_VERSION = 1
LIVE_PREFLIGHT_LAUNCH_PLAN_REQUIRED_KEYS = (
    "selected_session_dir",
    "model_path",
    "scaler_path",
    "out_dir",
)


def serialize_live_preflight_launch_plan(launch_plan: Any) -> dict[str, Any] | None:
    if launch_plan is None:
        return None
    return {
        "schema_version": LIVE_PREFLIGHT_LAUNCH_PLAN_SCHEMA_VERSION,
        "project_name": getattr(launch_plan, "project_name", None),
        "subject_id": getattr(launch_plan, "subject_id", None),
        "selection_source": getattr(launch_plan, "selection_source", None),
        "session_dir_inferred": bool(
            getattr(launch_plan, "session_dir_inferred", False)
        ),
        "selected_session_dir": (
            str(getattr(launch_plan, "selected_session_dir", ""))
            if getattr(launch_plan, "selected_session_dir", None) is not None
            else None
        ),
        "explicit_overrides": list(
            getattr(launch_plan, "explicit_overrides", ()) or ()
        ),
        "chosen_run_dir": (
            str(getattr(launch_plan, "chosen_run_dir", ""))
            if getattr(launch_plan, "chosen_run_dir", None) is not None
            else None
        ),
        "model_path": (
            str(getattr(launch_plan, "model_path", ""))
            if getattr(launch_plan, "model_path", None) is not None
            else None
        ),
        "scaler_path": (
            str(getattr(launch_plan, "scaler_path", ""))
            if getattr(launch_plan, "scaler_path", None) is not None
            else None
        ),
        "temperature_path": (
            str(getattr(launch_plan, "temperature_path", ""))
            if getattr(launch_plan, "temperature_path", None) is not None
            else None
        ),
        "out_dir": (
            str(getattr(launch_plan, "out_dir", ""))
            if getattr(launch_plan, "out_dir", None) is not None
            else None
        ),
        "no_file_io": bool(getattr(launch_plan, "no_file_io", False)),
        "record_raw": bool(getattr(launch_plan, "record_raw", False)),
    }


def extract_live_preflight_launch_plan(
    report: Mapping[str, Any] | None,
) -> tuple[Optional[dict[str, Any]], dict[str, Any]]:
    diagnostics: dict[str, Any] = {
        "reason": "ok",
        "report_keys": sorted(report.keys()) if isinstance(report, Mapping) else [],
        "launch_plan_key_exists": False,
        "launch_plan_present": False,
        "launch_plan_type": None,
        "launch_plan_is_empty": False,
        "required_keys": list(LIVE_PREFLIGHT_LAUNCH_PLAN_REQUIRED_KEYS),
        "missing_keys": [],
        "empty_keys": [],
        "schema_version": None,
        "raw_launch_plan_preview": None,
    }
    if not isinstance(report, Mapping):
        diagnostics["reason"] = "preflight_launch_plan_missing"
        return None, diagnostics
    if "launch_plan" not in report:
        diagnostics["reason"] = "preflight_launch_plan_missing"
        return None, diagnostics
    diagnostics["launch_plan_key_exists"] = True

    raw_launch_plan = report.get("launch_plan")
    diagnostics["launch_plan_type"] = type(raw_launch_plan).__name__
    diagnostics["raw_launch_plan_preview"] = repr(raw_launch_plan)[:200]
    if raw_launch_plan is None:
        diagnostics["reason"] = "preflight_launch_plan_empty"
        diagnostics["launch_plan_is_empty"] = True
        return None, diagnostics
    diagnostics["launch_plan_present"] = True
    if not isinstance(raw_launch_plan, Mapping):
        diagnostics["reason"] = "preflight_launch_plan_schema_mismatch"
        return None, diagnostics

    plan = dict(raw_launch_plan)
    diagnostics["launch_plan_is_empty"] = not bool(plan)
    diagnostics["schema_version"] = plan.get("schema_version")
    if not plan:
        diagnostics["reason"] = "preflight_launch_plan_empty"
        return None, diagnostics
    missing_keys = [
        key for key in LIVE_PREFLIGHT_LAUNCH_PLAN_REQUIRED_KEYS if key not in plan
    ]
    empty_keys = [
        key
        for key in LIVE_PREFLIGHT_LAUNCH_PLAN_REQUIRED_KEYS
        if key in plan and not str(plan.get(key) or "").strip()
    ]
    diagnostics["missing_keys"] = missing_keys
    diagnostics["empty_keys"] = empty_keys
    if missing_keys or empty_keys:
        diagnostics["reason"] = "preflight_launch_plan_schema_mismatch"
        return None, diagnostics
    return plan, diagnostics
