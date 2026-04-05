#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.live_infer_common import (
    ActuationDecision,
    applicability_gate_passed,
    debounced_should_send,
    finger_gate_passed,
    load_model_artifacts_from_files,
    uncertainty_gate_passed,
)
from utils.live_parity import load_capture_records, load_json, sha256_file, write_json
from utils.postprocess import PostprocessSettings, PostprocessState
from utils.runtime_utils import resolve_device


def _load_live_module():
    module_path = REPO_ROOT / "7_live_infer_and_actuate.py"
    spec = importlib.util.spec_from_file_location("live_infer_replay", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_max_abs_diff(expected: Any, actual: Any) -> Optional[float]:
    if expected is None or actual is None:
        return None
    exp_arr = np.asarray(expected, dtype=np.float64)
    act_arr = np.asarray(actual, dtype=np.float64)
    if exp_arr.shape != act_arr.shape:
        return float("inf")
    if exp_arr.size == 0 and act_arr.size == 0:
        return 0.0
    try:
        diff = np.abs(exp_arr - act_arr)
    except Exception:
        return float("inf")
    if diff.size == 0:
        return 0.0
    return float(np.max(diff))


def _safe_scalar_diff(expected: Any, actual: Any) -> Optional[float]:
    if expected is None or actual is None:
        return None
    try:
        exp_val = float(expected)
        act_val = float(actual)
    except Exception:
        return float("inf")
    if not math.isfinite(exp_val) or not math.isfinite(act_val):
        return float("inf")
    return abs(exp_val - act_val)


def _runtime_manifest_path(
    capture_dir: Path,
    capture_manifest: dict[str, Any],
) -> Optional[Path]:
    manifest_seed = capture_manifest.get("manifest_seed", {})
    if isinstance(manifest_seed, dict):
        raw = manifest_seed.get("runtime_manifest_path")
        if raw:
            path = Path(str(raw)).expanduser()
            if path.exists():
                return path.resolve()
    candidate = capture_dir.parent / "live_runtime_manifest.json"
    if candidate.exists():
        return candidate.resolve()
    return None


def _validate_artifact_hash(
    *,
    label: str,
    path_value: Any,
    expected_sha256: Any,
) -> None:
    if not path_value:
        raise RuntimeError(f"Runtime manifest missing {label} path")
    actual_sha256 = sha256_file(path_value)
    if actual_sha256 is None:
        raise FileNotFoundError(f"{label} file not found: {path_value}")
    if expected_sha256 and str(expected_sha256) != str(actual_sha256):
        raise RuntimeError(
            f"{label} hash mismatch for {path_value}: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )


def _validate_capture_manifest(
    capture_dir: Path,
    capture_manifest: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    if not isinstance(capture_manifest, dict) or not capture_manifest:
        raise RuntimeError("Capture manifest is missing or malformed")
    if not isinstance(records, list) or not records:
        raise RuntimeError("Captured windows payload is empty or malformed")
    missing = [
        key
        for key in (
            "record_count",
            "records_sha256",
            "records_candidate_indices",
            "manifest_seed",
        )
        if key not in capture_manifest
    ]
    if missing:
        raise RuntimeError(
            "Capture manifest is incomplete; missing field(s): "
            + ", ".join(missing)
        )
    records_path = capture_dir / "captured_windows.json"
    actual_records_sha256 = sha256_file(records_path)
    expected_records_sha256 = capture_manifest.get("records_sha256")
    if actual_records_sha256 is None:
        raise FileNotFoundError(f"Captured windows payload not found: {records_path}")
    if str(expected_records_sha256) != str(actual_records_sha256):
        raise RuntimeError(
            "Capture manifest records_sha256 does not match captured_windows.json: "
            f"expected {expected_records_sha256}, got {actual_records_sha256}"
        )
    expected_count = capture_manifest.get("record_count")
    if expected_count is not None and int(expected_count) != int(len(records)):
        raise RuntimeError(
            f"Capture manifest record_count={expected_count} does not match "
            f"captured_windows record count={len(records)}"
        )
    expected_indices = capture_manifest.get("records_candidate_indices")
    if expected_indices is not None:
        actual_indices = [
            int(row.get("candidate_index"))
            for row in records
            if row.get("candidate_index") is not None
        ]
        if list(expected_indices) != actual_indices:
            raise RuntimeError(
                "Capture manifest candidate index list does not match captured windows payload"
            )


def _require_record_fields(record: dict[str, Any], *, index: int) -> None:
    required_top = [
        "candidate_index",
        "segment_id",
        "window_start_s",
        "window_end_s",
        "raw_window_times",
        "raw_window_values",
        "resampled_window",
        "prepared_window",
        "quality",
        "inference",
        "decision",
        "actuation",
    ]
    missing = [key for key in required_top if key not in record]
    if missing:
        raise RuntimeError(
            f"Captured window {index} is missing required field(s): {', '.join(missing)}"
        )
    for key in ("quality", "inference", "decision", "actuation"):
        if not isinstance(record.get(key), dict):
            raise RuntimeError(f"Captured window {index} field '{key}' must be an object")
    nested_required = {
        "quality": [
            "window_quality_bad",
            "quality_bad_reason",
            "masked_channel_count",
            "masked_channel_ids",
        ],
        "inference": [
            "action_logits",
            "finger_logits",
            "applicability_logit",
            "action_probs",
            "model_raw_finger_probs",
            "finger_probs",
        ],
        "decision": [
            "raw_top_action_id",
            "raw_top_finger_id",
            "smoothed_action_id",
            "smoothed_finger_id",
            "committed_action_id",
            "committed_finger_id",
            "finger_gate_ok",
            "applicability_gate_ok",
            "uncertainty_gate_ok",
            "decision_reason",
        ],
        "actuation": [
            "vote_reason",
            "vote_finger_counts",
            "vote_action_counts",
            "vote_pair_counts",
            "target_action_id",
            "target_finger_id",
            "speed_scalar",
            "suppressed_reason",
            "sent",
            "latency_gate_ok",
        ],
    }
    for section, keys in nested_required.items():
        missing_nested = [
            key for key in keys if key not in record.get(section, {})
        ]
        if missing_nested:
            raise RuntimeError(
                f"Captured window {index} section '{section}' is missing: "
                f"{', '.join(missing_nested)}"
            )


def _build_runtime_args(runtime_manifest: dict[str, Any]) -> SimpleNamespace:
    runtime = runtime_manifest.get("runtime", {}) if isinstance(runtime_manifest, dict) else {}
    post_settings = runtime.get("postprocess_settings", {})
    quality = runtime.get("quality_thresholds", {})
    rest_bias = runtime.get("rest_bias", {})
    actuation = runtime.get("actuation", {})
    return SimpleNamespace(
        window_sec=float(runtime.get("window_sec", 0.25)),
        hop_sec=float(runtime.get("hop_sec", 0.05)),
        target_fs=float(runtime.get("target_fs", 64.0)),
        latency_policy=str(runtime.get("latency_policy", "warn")),
        latency_threshold_ms=float(runtime.get("latency_threshold_ms", 250.0)),
        live_quality_enabled=bool(runtime.get("live_quality_enabled", False)),
        input_clip_abs_z=float(quality.get("input_clip_abs_z", 8.0)),
        bad_channel_rms_z=float(quality.get("bad_channel_rms_z", 6.0)),
        bad_channel_abs_p95_z=float(quality.get("bad_channel_abs_p95_z", 8.0)),
        bad_channel_clipped_frac=float(quality.get("bad_channel_clipped_frac", 0.2)),
        bad_window_clipped_frac=float(quality.get("bad_window_clipped_frac", 0.1)),
        bad_window_max_masked_channels=int(
            quality.get("bad_window_max_masked_channels", 1)
        ),
        postprocess=bool(runtime.get("postprocess_enabled", False)),
        mc_passes=int(runtime.get("mc_passes", 1)),
        uncertainty_base_threshold=float(
            runtime.get("uncertainty_base_threshold", 0.0)
        ),
        uncertainty_weight=float(runtime.get("uncertainty_weight", 0.0)),
        threshold_action=float(post_settings.get("threshold_action", 0.5)),
        threshold_finger=float(post_settings.get("threshold_finger", 0.5)),
        threshold_applicability=float(
            post_settings.get("threshold_applicability", 0.5)
        ),
        rest_bias_correction_enabled=bool(rest_bias.get("enabled", False)),
        rest_bias_strength=float(rest_bias.get("strength", 0.0)),
        rest_bias_min_windows=int(rest_bias.get("min_rest_windows", 10)),
        enable_actuation=bool(actuation.get("enabled", False)),
        actuation_min_prob=float(actuation.get("actuation_min_prob", 0.0)),
        actuation_stability=int(actuation.get("actuation_stability", 1)),
        actuation_cooldown_ms=int(actuation.get("actuation_cooldown_ms", 0)),
        actuation_repeat_ms=int(actuation.get("actuation_repeat_ms", 0)),
        actuation_min_speed=float(actuation.get("actuation_min_speed", 0.0)),
        modulate_actuation_speed=bool(
            actuation.get("modulate_actuation_speed", False)
        ),
        actuation_speed_gamma=float(actuation.get("actuation_speed_gamma", 1.0)),
    )


def _compare_dict_fields(
    expected: dict[str, Any],
    actual: dict[str, Any],
    *,
    keys: list[str],
) -> list[str]:
    mismatches: list[str] = []
    for key in keys:
        if expected.get(key) != actual.get(key):
            mismatches.append(str(key))
    return mismatches


def replay_capture(
    *,
    capture_dir: Path,
    device_name: str,
    tolerance: float,
) -> dict[str, Any]:
    capture_manifest, records = load_capture_records(capture_dir)
    _validate_capture_manifest(capture_dir, capture_manifest, records)
    runtime_manifest_path = _runtime_manifest_path(capture_dir, capture_manifest)
    if runtime_manifest_path is None:
        raise FileNotFoundError(
            f"Runtime manifest not found for capture dir {capture_dir}"
        )
    runtime_manifest = load_json(runtime_manifest_path)
    artifacts = runtime_manifest.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise RuntimeError("Runtime manifest missing artifacts block")
    model_path = artifacts.get("model_path")
    scaler_path = artifacts.get("scaler_path")
    temperature_path = artifacts.get("temperature_path")
    if not model_path or not scaler_path:
        raise RuntimeError("Runtime manifest missing model_path/scaler_path")
    _validate_artifact_hash(
        label="model",
        path_value=model_path,
        expected_sha256=artifacts.get("model_sha256"),
    )
    _validate_artifact_hash(
        label="scaler",
        path_value=scaler_path,
        expected_sha256=artifacts.get("scaler_sha256"),
    )
    if temperature_path:
        _validate_artifact_hash(
            label="temperature",
            path_value=temperature_path,
            expected_sha256=artifacts.get("temperature_sha256"),
        )

    live_mod = _load_live_module()
    runtime_args = _build_runtime_args(runtime_manifest)
    device = resolve_device(device_name or str(runtime_manifest.get("runtime", {}).get("device", "auto")))

    first_values = np.asarray(records[0].get("raw_window_values"), dtype=np.float32)
    if first_values.ndim != 2 or first_values.shape[1] <= 0:
        raise RuntimeError("Captured windows are missing raw_window_values")
    n_channels = int(first_values.shape[1])
    model, scaler, temperature_state = load_model_artifacts_from_files(
        model_path=Path(str(model_path)),
        scaler_path=Path(str(scaler_path)),
        device=device,
        n_channels=n_channels,
        temperature_path=Path(str(temperature_path)) if temperature_path else None,
    )
    inference_engine = live_mod._build_inference_engine(
        model,
        scaler,
        device,
        runtime_args,
        temperature_state,
    )
    direct_engine = (
        None
        if inference_engine is not None
        else live_mod._build_direct_inference_engine(
            model, scaler, device, temperature_state
        )
    )
    postprocess_settings = PostprocessSettings(
        **dict(runtime_manifest.get("runtime", {}).get("postprocess_settings", {}))
    )
    post_state = PostprocessState()
    rest_bias = live_mod.RestFingerBiasCorrection(
        enabled=bool(runtime_args.rest_bias_correction_enabled),
        min_rest_windows=max(1, int(runtime_args.rest_bias_min_windows)),
        strength=float(runtime_args.rest_bias_strength),
    )
    actuation_speed_mapper = live_mod._build_actuation_speed_mapper(runtime_args)
    actuation_command_shaper = live_mod._build_actuation_command_shaper(runtime_args)
    actuation_history: deque[ActuationDecision] = deque(
        maxlen=max(1, int(runtime_args.actuation_stability))
    )
    last_sent: Optional[tuple[int, int]] = None
    last_send_time_ms: Optional[float] = None
    prev_segment_id: Optional[int] = None

    max_diffs: dict[str, float] = {
        "resampled_window": 0.0,
        "prepared_window": 0.0,
        "action_logits": 0.0,
        "finger_logits": 0.0,
        "applicability_logit": 0.0,
        "action_probs": 0.0,
        "finger_probs": 0.0,
        "model_raw_finger_probs": 0.0,
        "finger_applicable_prob": 0.0,
        "action_uncertainty": 0.0,
        "finger_uncertainty": 0.0,
        "applicability_uncertainty": 0.0,
        "adaptive_threshold": 0.0,
    }
    mismatch_counts: dict[str, int] = {
        "quality": 0,
        "decision": 0,
        "actuation": 0,
    }
    per_window: list[dict[str, Any]] = []

    for record in records:
        _require_record_fields(record, index=int(record.get("candidate_index", len(per_window))))
        candidate_index = int(record.get("candidate_index", len(per_window)))
        segment_id = int(record.get("segment_id", 0))
        if prev_segment_id is None:
            prev_segment_id = segment_id
        elif segment_id != prev_segment_id:
            actuation_history.clear()
            actuation_command_shaper.reset()
            post_state.reset()
            last_sent = None
            last_send_time_ms = None
            prev_segment_id = segment_id

        raw_times = np.asarray(record.get("raw_window_times"), dtype=float)
        raw_values = np.asarray(record.get("raw_window_values"), dtype=float)
        start_s = float(record.get("window_start_s"))
        end_s = float(record.get("window_end_s"))
        replay_resampled = live_mod._resample_window(
            raw_times,
            raw_values,
            start_s=start_s,
            end_s=end_s,
            target_fs=float(runtime_args.target_fs),
        )
        if replay_resampled is None:
            raise RuntimeError(f"Replay resampling failed for candidate {candidate_index}")

        quality = live_mod._sanitize_live_window(
            replay_resampled,
            scaler=scaler,
            enabled=bool(runtime_args.live_quality_enabled),
            input_clip_abs_z=float(runtime_args.input_clip_abs_z),
            bad_channel_rms_z=float(runtime_args.bad_channel_rms_z),
            bad_channel_abs_p95_z=float(runtime_args.bad_channel_abs_p95_z),
            bad_channel_clipped_frac=float(runtime_args.bad_channel_clipped_frac),
            bad_window_clipped_frac=float(runtime_args.bad_window_clipped_frac),
            bad_window_max_masked_channels=int(
                runtime_args.bad_window_max_masked_channels
            ),
        )
        inference_result = live_mod._predict_window(
            replay_resampled,
            scaler=scaler,
            model=model,
            device=device,
            inference_engine=inference_engine,
            direct_engine=direct_engine,
            temperature_state=temperature_state,
            emit_viz=False,
            prepared_window=quality.prepared_window,
        )
        action_probs = np.asarray(inference_result["action_probs"], dtype=float)
        model_raw_finger_probs = np.asarray(inference_result["finger_probs"], dtype=float)
        rest_bias.update(action_probs, model_raw_finger_probs)
        finger_probs = np.asarray(rest_bias.apply(model_raw_finger_probs), dtype=float)
        finger_applicable_prob = inference_result.get("finger_applicable_prob")
        decision_info = live_mod._postprocess_decision(
            action_probs,
            finger_probs,
            enabled=bool(runtime_args.postprocess),
            settings=postprocess_settings,
            state=post_state,
            finger_applicable_prob=(
                float(finger_applicable_prob)
                if finger_applicable_prob is not None
                else None
            ),
        )
        decision = ActuationDecision(
            finger_id=int(decision_info.get("committed_finger_id", 0)),
            action_id=int(decision_info.get("committed_action_id", 0)),
            prob=float(
                min(
                    float(decision_info.get("action_conf", 0.0)),
                    float(decision_info.get("finger_conf", 0.0)),
                )
            ),
        )
        action_uncertainty = float(inference_result.get("action_uncertainty", 0.0) or 0.0)
        finger_gate_ok = finger_gate_passed(decision_info)
        applicability_gate_ok = applicability_gate_passed(decision_info)
        uncertainty_gate_ok = uncertainty_gate_passed(decision_info, inference_result)
        actuation_speed_scalar = live_mod._compute_actuation_speed_scalar(
            decision.prob,
            action_uncertainty,
            actuation_speed_mapper,
            min_speed=float(runtime_args.actuation_min_speed),
        )
        latency_ms = float(record.get("latency_ms", record.get("prediction_latency_ms", 0.0)) or 0.0)
        actuation_vote = live_mod._resolve_live_actuation_vote(
            actuation_history,
            decision,
            required_pair_stability=int(runtime_args.actuation_stability),
            ignore_window=bool(quality.window_quality_bad),
            ignore_reason="quality_gate",
        )
        voted_decision = actuation_vote["decision"]
        actuation_target_finger_id = int(voted_decision.finger_id)
        actuation_target_action_id = int(voted_decision.action_id)
        actuation_suppressed_reason = None
        actuation_sent = False
        latency_policy = str(runtime_args.latency_policy).strip().lower()
        actuation_latency_gate_ok = (
            True
            if latency_policy == "warn"
            else live_mod._latency_gate_passed(
                latency_ms, float(runtime_args.latency_threshold_ms)
            )
        )
        if bool(runtime_args.enable_actuation):
            if quality.window_quality_bad:
                actuation_suppressed_reason = "quality_gate"
            elif not actuation_latency_gate_ok:
                actuation_suppressed_reason = "latency_gate"
            elif not finger_gate_ok:
                actuation_suppressed_reason = "finger_gate"
            elif not applicability_gate_ok:
                actuation_suppressed_reason = "applicability_gate"
            elif live_mod._is_noop_decision(
                voted_decision.finger_id,
                voted_decision.action_id,
            ):
                actuation_suppressed_reason = str(actuation_vote.get("reason", "noop"))
            elif not uncertainty_gate_ok:
                actuation_suppressed_reason = "uncertainty_gate"
            else:
                window_center_stream_s = start_s + (float(runtime_args.window_sec) / 2.0)
                shaped_command = actuation_command_shaper.shape(
                    action_id=int(voted_decision.action_id),
                    finger_id=int(voted_decision.finger_id),
                    action_conf=float(voted_decision.prob),
                    speed_scalar_override=float(actuation_speed_scalar),
                    timestamp_stream_ms=int(round(window_center_stream_s * 1000.0)),
                    stability_ok=True,
                    timebase_ms=int(round(window_center_stream_s * 1000.0)),
                )
                actuation_target_finger_id = int(shaped_command.finger_id)
                actuation_target_action_id = int(shaped_command.action_id)
                actuation_speed_scalar = float(shaped_command.speed_scalar)
                actuation_decision = ActuationDecision(
                    finger_id=actuation_target_finger_id,
                    action_id=actuation_target_action_id,
                    prob=float(voted_decision.prob),
                )
                current_time_ms = int(round(end_s * 1000.0))
                if live_mod._is_noop_decision(
                    actuation_decision.finger_id,
                    actuation_decision.action_id,
                ):
                    actuation_suppressed_reason = "min_prob"
                elif debounced_should_send(
                    actuation_decision,
                    last_sent=last_sent,
                    stable_count=1,
                    required_stability=1,
                    last_send_time_ms=last_send_time_ms,
                    current_time_ms=float(current_time_ms),
                    cooldown_ms=int(runtime_args.actuation_cooldown_ms),
                    repeat_same_ms=int(runtime_args.actuation_repeat_ms),
                ):
                    last_sent = (
                        int(actuation_decision.finger_id),
                        int(actuation_decision.action_id),
                    )
                    last_send_time_ms = float(current_time_ms)
                    actuation_sent = True
                else:
                    actuation_suppressed_reason = "cooldown_or_duplicate"

        replay_quality = {
            "window_quality_bad": bool(quality.window_quality_bad),
            "quality_bad_reason": quality.quality_bad_reason,
            "masked_channel_count": int(len(quality.masked_channel_ids)),
            "masked_channel_ids": list(quality.masked_channel_ids),
        }
        replay_decision = {
            "raw_top_action_id": int(decision_info.get("raw_top_action_id", 0)),
            "raw_top_finger_id": int(decision_info.get("raw_top_finger_id", 0)),
            "smoothed_action_id": int(decision_info.get("smoothed_action_id", 0)),
            "smoothed_finger_id": int(decision_info.get("smoothed_finger_id", 0)),
            "committed_action_id": int(decision_info.get("committed_action_id", 0)),
            "committed_finger_id": int(decision_info.get("committed_finger_id", 0)),
            "finger_gate_ok": bool(decision_info.get("finger_gate_ok", True)),
            "applicability_gate_ok": bool(
                decision_info.get("applicability_gate_ok", True)
            ),
            "uncertainty_gate_ok": bool(uncertainty_gate_ok),
            "decision_reason": str(decision_info.get("decision_reason", "")),
        }
        replay_actuation = {
            "vote_reason": str(actuation_vote.get("reason", "")),
            "vote_finger_counts": {
                str(key): int(value)
                for key, value in dict(actuation_vote.get("finger_votes", {})).items()
            },
            "vote_action_counts": {
                str(key): int(value)
                for key, value in dict(actuation_vote.get("action_votes", {})).items()
            },
            "vote_pair_counts": {
                str(key): int(value)
                for key, value in dict(actuation_vote.get("pair_votes", {})).items()
            },
            "target_action_id": int(actuation_target_action_id),
            "target_finger_id": int(actuation_target_finger_id),
            "speed_scalar": float(actuation_speed_scalar),
            "suppressed_reason": actuation_suppressed_reason,
            "sent": bool(actuation_sent),
            "latency_gate_ok": bool(actuation_latency_gate_ok),
        }

        expected_quality = dict(record.get("quality", {}))
        expected_decision = dict(record.get("decision", {}))
        expected_actuation = dict(record.get("actuation", {}))
        quality_mismatches = _compare_dict_fields(
            expected_quality,
            replay_quality,
            keys=[
                "window_quality_bad",
                "quality_bad_reason",
                "masked_channel_count",
                "masked_channel_ids",
            ],
        )
        decision_mismatches = _compare_dict_fields(
            expected_decision,
            replay_decision,
            keys=[
                "raw_top_action_id",
                "raw_top_finger_id",
                "smoothed_action_id",
                "smoothed_finger_id",
                "committed_action_id",
                "committed_finger_id",
                "finger_gate_ok",
                "applicability_gate_ok",
                "uncertainty_gate_ok",
                "decision_reason",
            ],
        )
        actuation_mismatches = _compare_dict_fields(
            expected_actuation,
            replay_actuation,
            keys=[
                "vote_reason",
                "vote_finger_counts",
                "vote_action_counts",
                "vote_pair_counts",
                "target_action_id",
                "target_finger_id",
                "speed_scalar",
                "suppressed_reason",
                "sent",
                "latency_gate_ok",
            ],
        )
        if quality_mismatches:
            mismatch_counts["quality"] += 1
        if decision_mismatches:
            mismatch_counts["decision"] += 1
        if actuation_mismatches:
            mismatch_counts["actuation"] += 1

        diffs = {
            "resampled_window": _safe_max_abs_diff(
                record.get("resampled_window"), replay_resampled
            ),
            "prepared_window": _safe_max_abs_diff(
                record.get("prepared_window"), quality.prepared_window
            ),
            "action_logits": _safe_max_abs_diff(
                record.get("inference", {}).get("action_logits"),
                inference_result.get("action_logits"),
            ),
            "finger_logits": _safe_max_abs_diff(
                record.get("inference", {}).get("finger_logits"),
                inference_result.get("finger_logits"),
            ),
            "applicability_logit": _safe_scalar_diff(
                record.get("inference", {}).get("applicability_logit"),
                inference_result.get("applicability_logit"),
            ),
            "action_probs": _safe_max_abs_diff(
                record.get("inference", {}).get("action_probs"),
                action_probs,
            ),
            "finger_probs": _safe_max_abs_diff(
                record.get("inference", {}).get("finger_probs"),
                finger_probs,
            ),
            "model_raw_finger_probs": _safe_max_abs_diff(
                record.get("inference", {}).get("model_raw_finger_probs"),
                model_raw_finger_probs,
            ),
            "finger_applicable_prob": _safe_scalar_diff(
                record.get("inference", {}).get("finger_applicable_prob"),
                inference_result.get("finger_applicable_prob"),
            ),
            "action_uncertainty": _safe_scalar_diff(
                record.get("inference", {}).get("action_uncertainty"),
                inference_result.get("action_uncertainty"),
            ),
            "finger_uncertainty": _safe_scalar_diff(
                record.get("inference", {}).get("finger_uncertainty"),
                inference_result.get("finger_uncertainty"),
            ),
            "applicability_uncertainty": _safe_scalar_diff(
                record.get("inference", {}).get("applicability_uncertainty"),
                inference_result.get("applicability_uncertainty"),
            ),
            "adaptive_threshold": _safe_scalar_diff(
                record.get("inference", {}).get("adaptive_threshold"),
                inference_result.get("adaptive_threshold"),
            ),
        }
        for key, value in diffs.items():
            if value is not None and math.isfinite(value):
                max_diffs[key] = max(float(max_diffs[key]), float(value))

        per_window.append(
            {
                "candidate_index": candidate_index,
                "segment_id": segment_id,
                "diffs": diffs,
                "quality_mismatches": quality_mismatches,
                "decision_mismatches": decision_mismatches,
                "actuation_mismatches": actuation_mismatches,
            }
        )

    parity = {
        "resampled_window": {
            "ok": float(max_diffs["resampled_window"]) <= float(tolerance),
            "max_abs_diff": float(max_diffs["resampled_window"]),
        },
        "preprocessed_tensor_values": {
            "ok": float(max_diffs["prepared_window"]) <= float(tolerance),
            "max_abs_diff": float(max_diffs["prepared_window"]),
        },
        "logits": {
            "ok": max(
                float(max_diffs["action_logits"]),
                float(max_diffs["finger_logits"]),
                float(max_diffs["applicability_logit"]),
            )
            <= float(tolerance),
            "max_abs_diff": {
                "action_logits": float(max_diffs["action_logits"]),
                "finger_logits": float(max_diffs["finger_logits"]),
                "applicability_logit": float(max_diffs["applicability_logit"]),
            },
        },
        "probabilities": {
            "ok": max(
                float(max_diffs["action_probs"]),
                float(max_diffs["finger_probs"]),
                float(max_diffs["model_raw_finger_probs"]),
            )
            <= float(tolerance),
            "max_abs_diff": {
                "action_probs": float(max_diffs["action_probs"]),
                "finger_probs": float(max_diffs["finger_probs"]),
                "model_raw_finger_probs": float(max_diffs["model_raw_finger_probs"]),
            },
        },
        "diagnostics": {
            "ok": max(
                float(max_diffs["finger_applicable_prob"]),
                float(max_diffs["action_uncertainty"]),
                float(max_diffs["finger_uncertainty"]),
                float(max_diffs["applicability_uncertainty"]),
                float(max_diffs["adaptive_threshold"]),
            )
            <= float(tolerance),
            "max_abs_diff": {
                "finger_applicable_prob": float(max_diffs["finger_applicable_prob"]),
                "action_uncertainty": float(max_diffs["action_uncertainty"]),
                "finger_uncertainty": float(max_diffs["finger_uncertainty"]),
                "applicability_uncertainty": float(max_diffs["applicability_uncertainty"]),
                "adaptive_threshold": float(max_diffs["adaptive_threshold"]),
            },
        },
        "decoded_outputs": {
            "ok": sum(mismatch_counts.values()) == 0,
            "mismatch_counts": mismatch_counts,
        },
    }
    report = {
        "status": "ok",
        "capture_dir": str(capture_dir),
        "runtime_manifest_path": str(runtime_manifest_path),
        "record_count": int(len(records)),
        "tolerance": float(tolerance),
        "inference_backend": str(
            runtime_manifest.get("runtime", {}).get("inference_backend", "unknown")
        ),
        "mc_passes": int(runtime_manifest.get("runtime", {}).get("mc_passes", 1)),
        "notes": [
            "Replay preserves postprocess and actuation state across windows and resets postprocess/actuation state on segment_id changes.",
            "Rest-bias correction is preserved across segment_id changes to mirror Step 7 runtime behavior.",
        ],
        "parity": parity,
        "per_window": per_window,
    }
    parity_checks = [
        bool(value.get("ok"))
        for value in parity.values()
        if isinstance(value, dict) and "ok" in value
    ]
    if not parity_checks or not all(parity_checks):
        report["status"] = "parity_failure"
    return report


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay captured Step 7 live windows through the same preprocessing and inference path."
    )
    parser.add_argument(
        "--capture-dir",
        required=True,
        type=str,
        help="Path to processed/live_infer/parity_capture.",
    )
    parser.add_argument(
        "--report-out",
        type=str,
        default=None,
        help="Optional output path for parity_report.json. Defaults beside the capture dir.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device override.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-5,
        help="Numeric max-abs-diff tolerance for parity checks.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    capture_dir = Path(args.capture_dir).expanduser().resolve()
    report_out = (
        Path(args.report_out).expanduser().resolve()
        if args.report_out
        else capture_dir.parent / "parity_report.json"
    )
    try:
        report = replay_capture(
            capture_dir=capture_dir,
            device_name=str(args.device),
            tolerance=float(args.tolerance),
        )
    except Exception as exc:
        failure_report = {
            "status": "error",
            "capture_dir": str(capture_dir),
            "tolerance": float(args.tolerance),
            "error": str(exc),
        }
        write_json(report_out, failure_report)
        print(f"Replay failed: {exc}", file=sys.stderr)
        print(f"Parity report written: {report_out}", file=sys.stderr)
        return 1
    write_json(report_out, report)
    print(json.dumps(report["parity"], indent=2, sort_keys=True))
    print(f"Parity report written: {report_out}")
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
