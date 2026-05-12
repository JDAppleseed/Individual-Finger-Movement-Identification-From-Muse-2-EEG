from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

from utils.default_recipe import ARTIFACT_DEFAULTS, LIVE_INFER_RECIPE_DEFAULTS

_DEFAULT_REQUIRED_LSL_LABELS = ["TP9", "AF7", "AF8", "TP10"]


def default_step7_settings() -> dict[str, Any]:
    return {
        "deployment_session_dir": None,
        "session_dir": None,
        "model_path": f"models/{ARTIFACT_DEFAULTS['model']}",
        "scaler_path": str(ARTIFACT_DEFAULTS["scaler"]),
        "out_dir": None,
        "device": "auto",
        "stream_name": "Muse2-EEG",
        "stream_type": "EEG",
        "lsl_source_id": None,
        "LSL_RESOLVE_TIMEOUT": 25.0,
        "REQUIRED_LSL_LABELS": list(_DEFAULT_REQUIRED_LSL_LABELS),
        "REQUIRE_EXACTLY_4_CHANNELS": True,
        "LIVE_EEG_PLOT_ENABLED": bool(
            LIVE_INFER_RECIPE_DEFAULTS["LIVE_EEG_PLOT_ENABLED"]
        ),
        "LIVE_EEG_PLOT_DISPLAY_FS": float(
            LIVE_INFER_RECIPE_DEFAULTS["LIVE_EEG_PLOT_DISPLAY_FS"]
        ),
        "LIVE_EEG_PLOT_FPS": float(LIVE_INFER_RECIPE_DEFAULTS["LIVE_EEG_PLOT_FPS"]),
        "LIVE_PREDICTION_TEXT_ENABLED": bool(
            LIVE_INFER_RECIPE_DEFAULTS["LIVE_PREDICTION_TEXT_ENABLED"]
        ),
        "LIVE_PREDICTION_TEXT_FPS": float(
            LIVE_INFER_RECIPE_DEFAULTS["LIVE_PREDICTION_TEXT_FPS"]
        ),
        "LIVE_VIZ_ENABLED": bool(LIVE_INFER_RECIPE_DEFAULTS["LIVE_VIZ_ENABLED"]),
        "LIVE_VIZ_FPS": int(LIVE_INFER_RECIPE_DEFAULTS["LIVE_VIZ_FPS"]),
        "window_sec": float(LIVE_INFER_RECIPE_DEFAULTS["window_sec"]),
        "hop_sec": float(LIVE_INFER_RECIPE_DEFAULTS["hop_sec"]),
        "target_fs": float(LIVE_INFER_RECIPE_DEFAULTS["target_fs"]),
        "live_buffer_sec": float(LIVE_INFER_RECIPE_DEFAULTS["live_buffer_sec"]),
        "live_max_window_lag_s": float(
            LIVE_INFER_RECIPE_DEFAULTS["live_max_window_lag_s"]
        ),
        "alignment_internal_max_gap_s": float(
            LIVE_INFER_RECIPE_DEFAULTS["alignment_internal_max_gap_s"]
        ),
        "allow_drop": bool(LIVE_INFER_RECIPE_DEFAULTS["allow_drop"]),
        "latency_threshold_ms": float(LIVE_INFER_RECIPE_DEFAULTS["latency_threshold_ms"]),
        "latency_policy": str(LIVE_INFER_RECIPE_DEFAULTS["latency_policy"]),
        "log_every": float(LIVE_INFER_RECIPE_DEFAULTS["log_every"]),
        "enable_actuation": bool(LIVE_INFER_RECIPE_DEFAULTS["enable_actuation"]),
        "serial_port": LIVE_INFER_RECIPE_DEFAULTS["serial_port"],
        "serial_baud": int(LIVE_INFER_RECIPE_DEFAULTS["serial_baud"]),
        "force_no_serial": bool(LIVE_INFER_RECIPE_DEFAULTS["force_no_serial"]),
        "serial_write_timeout_s": float(
            LIVE_INFER_RECIPE_DEFAULTS["serial_write_timeout_s"]
        ),
        "serial_max_hz": float(LIVE_INFER_RECIPE_DEFAULTS["serial_max_hz"]),
        "serial_settle_s": float(LIVE_INFER_RECIPE_DEFAULTS["serial_settle_s"]),
        "serial_movement_warmup_enabled": bool(
            LIVE_INFER_RECIPE_DEFAULTS["serial_movement_warmup_enabled"]
        ),
        "lsl_acquirer_queue_max_chunks": int(
            LIVE_INFER_RECIPE_DEFAULTS["lsl_acquirer_queue_max_chunks"]
        ),
        "bluetooth_target": LIVE_INFER_RECIPE_DEFAULTS["bluetooth_target"],
        "no_file_io": bool(LIVE_INFER_RECIPE_DEFAULTS["no_file_io"]),
        "actuation_min_prob": float(LIVE_INFER_RECIPE_DEFAULTS["actuation_min_prob"]),
        "actuation_stability": int(LIVE_INFER_RECIPE_DEFAULTS["actuation_stability"]),
        "actuation_cooldown_ms": int(LIVE_INFER_RECIPE_DEFAULTS["actuation_cooldown_ms"]),
        "actuation_repeat_ms": int(LIVE_INFER_RECIPE_DEFAULTS["actuation_repeat_ms"]),
        "actuation_min_speed": float(LIVE_INFER_RECIPE_DEFAULTS["actuation_min_speed"]),
        "modulate_actuation_speed": bool(
            LIVE_INFER_RECIPE_DEFAULTS["modulate_actuation_speed"]
        ),
        "actuation_speed_gamma": float(LIVE_INFER_RECIPE_DEFAULTS["actuation_speed_gamma"]),
        "postprocess": bool(LIVE_INFER_RECIPE_DEFAULTS["postprocess"]),
        "smoothing_enabled": bool(LIVE_INFER_RECIPE_DEFAULTS["smoothing_enabled"]),
        "smoothing_method": str(LIVE_INFER_RECIPE_DEFAULTS["smoothing_method"]),
        "smoothing_window": int(LIVE_INFER_RECIPE_DEFAULTS["smoothing_window"]),
        "hysteresis_enabled": bool(LIVE_INFER_RECIPE_DEFAULTS["hysteresis_enabled"]),
        "hysteresis_frames": int(LIVE_INFER_RECIPE_DEFAULTS["hysteresis_frames"]),
        "threshold_action": float(LIVE_INFER_RECIPE_DEFAULTS["threshold_action"]),
        "threshold_finger": float(LIVE_INFER_RECIPE_DEFAULTS["threshold_finger"]),
        "threshold_applicability": float(
            LIVE_INFER_RECIPE_DEFAULTS["threshold_applicability"]
        ),
        "adjacency_enabled": bool(LIVE_INFER_RECIPE_DEFAULTS["adjacency_enabled"]),
        "hysteresis_margin": float(LIVE_INFER_RECIPE_DEFAULTS["hysteresis_margin"]),
        "finger_delta": float(LIVE_INFER_RECIPE_DEFAULTS["finger_delta"]),
        "finger_mode": str(LIVE_INFER_RECIPE_DEFAULTS["finger_mode"]),
        "rest_bias_correction_enabled": bool(
            LIVE_INFER_RECIPE_DEFAULTS["rest_bias_correction_enabled"]
        ),
        "rest_bias_strength": float(LIVE_INFER_RECIPE_DEFAULTS["rest_bias_strength"]),
        "rest_bias_min_windows": int(LIVE_INFER_RECIPE_DEFAULTS["rest_bias_min_windows"]),
        "live_quality_enabled": bool(LIVE_INFER_RECIPE_DEFAULTS["live_quality_enabled"]),
        "input_clip_abs_z": float(LIVE_INFER_RECIPE_DEFAULTS["input_clip_abs_z"]),
        "bad_channel_rms_z": float(LIVE_INFER_RECIPE_DEFAULTS["bad_channel_rms_z"]),
        "bad_channel_abs_p95_z": float(
            LIVE_INFER_RECIPE_DEFAULTS["bad_channel_abs_p95_z"]
        ),
        "bad_channel_clipped_frac": float(
            LIVE_INFER_RECIPE_DEFAULTS["bad_channel_clipped_frac"]
        ),
        "bad_window_clipped_frac": float(
            LIVE_INFER_RECIPE_DEFAULTS["bad_window_clipped_frac"]
        ),
        "bad_window_max_masked_channels": int(
            LIVE_INFER_RECIPE_DEFAULTS["bad_window_max_masked_channels"]
        ),
        "use_inference_engine": bool(LIVE_INFER_RECIPE_DEFAULTS["use_inference_engine"]),
        "mc_passes": int(LIVE_INFER_RECIPE_DEFAULTS["mc_passes"]),
        "uncertainty_base_threshold": float(
            LIVE_INFER_RECIPE_DEFAULTS["uncertainty_base_threshold"]
        ),
        "uncertainty_weight": float(LIVE_INFER_RECIPE_DEFAULTS["uncertainty_weight"]),
        "parity_capture_enabled": bool(LIVE_INFER_RECIPE_DEFAULTS["parity_capture_enabled"]),
        "parity_capture_max_windows": int(LIVE_INFER_RECIPE_DEFAULTS["parity_capture_max_windows"]),
        "parity_capture_flush_every": int(
            LIVE_INFER_RECIPE_DEFAULTS["parity_capture_flush_every"]
        ),
        "pred_log": None,
    }


def load_step7_config(path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).expanduser().resolve().read_text())
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if isinstance(settings, dict):
        return payload, dict(settings)
    if isinstance(payload, dict):
        return payload, dict(payload)
    raise ValueError(f"Unsupported Step 7 config format: {path}")


def resolve_subject_step7_config_path(subject_dir: Path) -> Path:
    subject_dir = Path(subject_dir).expanduser().resolve()
    candidates = [
        subject_dir / "winning_model" / "configs" / "infer.json",
        subject_dir / "config" / "infer.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0]


def coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def build_step7_postprocess_settings(
    infer_settings: Mapping[str, Any],
):
    from utils.postprocess import PostprocessSettings

    defaults = PostprocessSettings()
    kwargs = {
        field.name: infer_settings.get(field.name, getattr(defaults, field.name))
        for field in fields(PostprocessSettings)
    }
    return PostprocessSettings(**kwargs)


def build_step7_replay_runtime_config(
    infer_settings: Mapping[str, Any],
    *,
    settings: Mapping[str, Any] | None = None,
    latency_mode: str | None = None,
    fixed_latency_ms: float | None = None,
    reset_on_trial_change: bool | None = None,
    deterministic: bool | None = None,
):
    from utils.live_infer_common import ReplayRuntimeConfig

    config_overrides = settings or {}
    defaults = ReplayRuntimeConfig()
    kwargs = {field.name: getattr(defaults, field.name) for field in fields(ReplayRuntimeConfig)}
    for key in kwargs:
        if key in infer_settings:
            kwargs[key] = infer_settings[key]
        if key in config_overrides:
            kwargs[key] = config_overrides[key]
    kwargs["latency_mode"] = (
        str(latency_mode)
        if latency_mode is not None
        else str(config_overrides.get("latency_mode", kwargs["latency_mode"]))
    )
    kwargs["fixed_latency_ms"] = (
        float(fixed_latency_ms)
        if fixed_latency_ms is not None
        else (
            float(config_overrides["fixed_latency_ms"])
            if config_overrides.get("fixed_latency_ms") is not None
            else kwargs["fixed_latency_ms"]
        )
    )
    kwargs["reset_on_trial_change"] = (
        bool(reset_on_trial_change)
        if reset_on_trial_change is not None
        else coerce_bool(
            config_overrides.get("reset_on_trial_change"),
            kwargs["reset_on_trial_change"],
        )
    )
    kwargs["deterministic"] = (
        bool(deterministic)
        if deterministic is not None
        else coerce_bool(config_overrides.get("deterministic"), kwargs["deterministic"])
    )
    latency_policy = str(infer_settings.get("latency_policy", "warn")).strip().lower()
    if latency_mode is None and settings is None and latency_policy == "warn":
        kwargs["latency_mode"] = "ignore"
    kwargs["modulate_actuation_speed"] = coerce_bool(
        kwargs["modulate_actuation_speed"], True
    )
    kwargs["use_inference_engine"] = coerce_bool(kwargs["use_inference_engine"], False)
    kwargs["live_quality_enabled"] = coerce_bool(kwargs["live_quality_enabled"], True)
    if str(kwargs["latency_mode"]).strip().lower() != "fixed":
        kwargs["fixed_latency_ms"] = None
    if (
        str(kwargs["latency_mode"]).strip().lower() == "fixed"
        and kwargs["fixed_latency_ms"] is None
    ):
        raise ValueError("fixed_latency_ms is required when latency_mode='fixed'")
    return ReplayRuntimeConfig(
        window_sec=float(kwargs["window_sec"]),
        hop_sec=float(kwargs["hop_sec"]),
        latency_threshold_ms=float(kwargs["latency_threshold_ms"]),
        actuation_min_prob=float(kwargs["actuation_min_prob"]),
        actuation_stability=int(kwargs["actuation_stability"]),
        actuation_cooldown_ms=int(kwargs["actuation_cooldown_ms"]),
        actuation_repeat_ms=int(kwargs["actuation_repeat_ms"]),
        actuation_min_speed=float(kwargs["actuation_min_speed"]),
        modulate_actuation_speed=bool(kwargs["modulate_actuation_speed"]),
        actuation_speed_gamma=float(kwargs["actuation_speed_gamma"]),
        use_inference_engine=bool(kwargs["use_inference_engine"]),
        mc_passes=int(kwargs["mc_passes"]),
        uncertainty_base_threshold=float(kwargs["uncertainty_base_threshold"]),
        uncertainty_weight=float(kwargs["uncertainty_weight"]),
        live_quality_enabled=bool(kwargs["live_quality_enabled"]),
        input_clip_abs_z=float(kwargs["input_clip_abs_z"]),
        bad_channel_rms_z=float(kwargs["bad_channel_rms_z"]),
        bad_channel_abs_p95_z=float(kwargs["bad_channel_abs_p95_z"]),
        bad_channel_clipped_frac=float(kwargs["bad_channel_clipped_frac"]),
        bad_window_clipped_frac=float(kwargs["bad_window_clipped_frac"]),
        bad_window_max_masked_channels=int(kwargs["bad_window_max_masked_channels"]),
        latency_mode=str(kwargs["latency_mode"]),
        fixed_latency_ms=(
            float(kwargs["fixed_latency_ms"])
            if kwargs["fixed_latency_ms"] is not None
            else None
        ),
        reset_on_trial_change=bool(kwargs["reset_on_trial_change"]),
        deterministic=bool(kwargs["deterministic"]),
    )


def diff_step7_settings(
    base_settings: Mapping[str, Any],
    runtime_settings: Mapping[str, Any],
) -> dict[str, Any]:
    diff: dict[str, Any] = {}
    for key, value in runtime_settings.items():
        if _normalize_for_compare(base_settings.get(key)) != _normalize_for_compare(value):
            diff[str(key)] = value
    return diff


def _normalize_for_compare(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_for_compare(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_compare(item) for item in value]
    return value
