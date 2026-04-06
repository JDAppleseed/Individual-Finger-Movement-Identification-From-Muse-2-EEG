from __future__ import annotations

import importlib.util
import json
import sys
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from tools.audit_live_parity import audit_live_dir
from tools.replay_live_capture import replay_capture
from utils.live_infer_common import ActuationDecision
from utils.live_infer_common import load_model_artifacts_from_files
from utils.live_parity import (
    LiveParityCapture,
    ParityCaptureSettings,
    load_json,
    sha256_file,
    write_json,
)
from utils.postprocess import PostprocessSettings, PostprocessState
from utils.runtime_utils import (
    TemperatureScalingState,
    save_normalizer,
    save_temperature_scaling,
)


def _load_live_module():
    module_path = Path(__file__).resolve().parents[1] / "7_live_infer_and_actuate.py"
    spec = importlib.util.spec_from_file_location("live_infer_parity_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_deterministic_artifacts(run_dir: Path) -> tuple[Path, Path, Path]:
    model_path = run_dir / "finger_action_model.pt"
    scaler_path = run_dir / "scaler.npz"
    temperature_path = run_dir / "temperature_scaling.json"

    model = CNNLSTMFingerActionNet(
        n_channels=4,
        n_fingers=5,
        n_actions=3,
        finger_applicability_head=True,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.finger_head.bias[0] = 3.0
        model.action_head.bias[1] = 2.5
        model.finger_applicability_head.bias.fill_(2.0)
    torch.save(model.state_dict(), model_path)

    save_normalizer(
        scaler_path,
        {
            "type": "per_channel",
            "mean": np.zeros((4,), dtype=np.float32),
            "std": np.ones((4,), dtype=np.float32),
            "channels": 4,
            "preprocess": {
                "per_window_center": False,
                "per_window_detrend": False,
            },
        },
    )
    save_temperature_scaling(
        temperature_path,
        TemperatureScalingState(
            action_temperature=1.0,
            finger_temperature=1.0,
            applicability_temperature=1.0,
            has_applicability_temperature=True,
            source="test",
        ),
    )
    return model_path, scaler_path, temperature_path


def _modern_runtime_manifest(
    *,
    run_dir: Path,
    model_path: Path,
    scaler_path: Path,
    temperature_path: Path,
    post_settings: PostprocessSettings | None = None,
    target_fs: float = 256.0,
    live_quality_enabled: bool = True,
    actuation_enabled: bool = False,
    actuation_stability: int = 2,
    channel_labels: list[str] | None = None,
    reorder: list[int] | None = None,
) -> dict[str, Any]:
    post_settings = post_settings or PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.0,
        threshold_applicability=0.0,
        adjacency_enabled=False,
    )
    channel_labels = channel_labels or ["TP9", "AF7", "AF8", "TP10"]
    reorder = reorder or list(range(len(channel_labels)))
    return {
        "artifacts": {
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "model_sha256": sha256_file(model_path),
            "scaler_path": str(scaler_path),
            "scaler_sha256": sha256_file(scaler_path),
            "temperature_path": str(temperature_path),
            "temperature_sha256": sha256_file(temperature_path),
        },
        "deployment": {"run_dir": str(run_dir)},
        "stream_resolution": {
            "requested_source_id": "fresh-id",
            "selected_source_id": "fresh-id",
            "source_id_source": "env",
            "selection_matched_by_source_id": True,
            "channel_labels": list(channel_labels),
        },
        "stream_selection": {
            "expected_channel_labels": list(channel_labels),
        },
        "stream_contract": {
            "expected": {
                "required_labels": list(channel_labels),
                "sampling_rate": float(target_fs),
            },
            "resolved": {
                "channel_labels": list(channel_labels),
                "channel_reorder_to_model_order": list(reorder),
                "channel_reorder_applied": bool(
                    list(reorder) != list(range(len(reorder)))
                ),
            },
            "contract_ok": True,
            "mismatches": [],
        },
        "runtime": {
            "device": "cpu",
            "inference_backend": "direct",
            "mc_passes": 1,
            "uncertainty_base_threshold": 0.0,
            "uncertainty_weight": 0.0,
            "postprocess_enabled": False,
            "postprocess_settings": asdict(post_settings),
            "latency_policy": "warn",
            "latency_threshold_ms": 250.0,
            "window_sec": 0.25,
            "hop_sec": 0.05,
            "target_fs": float(target_fs),
            "alignment_internal_max_gap_s": 0.06,
            "alignment_edge_max_gap_s": 1.0 / float(target_fs) * 4.0,
            "live_quality_enabled": bool(live_quality_enabled),
            "quality_thresholds": {
                "input_clip_abs_z": 6.0,
                "bad_channel_rms_z": 10.0,
                "bad_channel_abs_p95_z": 10.0,
                "bad_channel_clipped_frac": 1.0,
                "bad_window_clipped_frac": 1.0,
                "bad_window_max_masked_channels": 1,
            },
            "rest_bias": {
                "enabled": False,
                "strength": 0.0,
                "min_rest_windows": 2,
            },
            "actuation": {
                "enabled": bool(actuation_enabled),
                "actuation_min_prob": 0.24,
                "actuation_stability": int(actuation_stability),
                "actuation_cooldown_ms": 0,
                "actuation_repeat_ms": 0,
                "actuation_min_speed": 0.0,
                "modulate_actuation_speed": False,
                "actuation_speed_gamma": 1.0,
            },
        },
    }


def _capture_record(
    live_mod,
    *,
    model,
    scaler,
    temperature_state,
    device,
    runtime_args,
    post_settings,
    post_state,
    rest_bias,
    actuation_history,
    start_s: float,
    end_s: float,
    raw_times: np.ndarray,
    raw_values: np.ndarray,
    latency_ms: float,
):
    resampled = live_mod._resample_window(
        raw_times,
        raw_values,
        start_s=start_s,
        end_s=end_s,
        target_fs=float(runtime_args.target_fs),
    )
    quality = live_mod._sanitize_live_window(
        resampled,
        scaler=scaler,
        enabled=bool(runtime_args.live_quality_enabled),
        input_clip_abs_z=float(runtime_args.input_clip_abs_z),
        bad_channel_rms_z=float(runtime_args.bad_channel_rms_z),
        bad_channel_abs_p95_z=float(runtime_args.bad_channel_abs_p95_z),
        bad_channel_clipped_frac=float(runtime_args.bad_channel_clipped_frac),
        bad_window_clipped_frac=float(runtime_args.bad_window_clipped_frac),
        bad_window_max_masked_channels=int(runtime_args.bad_window_max_masked_channels),
    )
    direct_engine = live_mod._build_direct_inference_engine(
        model, scaler, device, temperature_state
    )
    inference_result = live_mod._predict_window(
        resampled,
        scaler=scaler,
        model=model,
        device=device,
        inference_engine=None,
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
        settings=post_settings,
        state=post_state,
        finger_applicable_prob=(
            float(finger_applicable_prob)
            if finger_applicable_prob is not None
            else None
        ),
    )
    decision = ActuationDecision(
        finger_id=int(decision_info["committed_finger_id"]),
        action_id=int(decision_info["committed_action_id"]),
        prob=float(
            min(
                float(decision_info["action_conf"]),
                float(decision_info["finger_conf"]),
            )
        ),
    )
    actuation_vote = live_mod._resolve_live_actuation_vote(
        actuation_history,
        decision,
        required_pair_stability=int(runtime_args.actuation_stability),
        ignore_window=bool(quality.window_quality_bad),
        ignore_reason="quality_gate",
    )
    voted_decision = actuation_vote["decision"]
    return {
        "window_start_s": float(start_s),
        "window_end_s": float(end_s),
        "raw_window_times": raw_times.tolist(),
        "raw_window_values": raw_values.tolist(),
        "resampled_window": resampled.tolist(),
        "prepared_window": quality.prepared_window.tolist(),
        "latency_ms": float(latency_ms),
        "quality": {
            "window_quality_bad": bool(quality.window_quality_bad),
            "quality_bad_reason": quality.quality_bad_reason,
            "masked_channel_count": int(len(quality.masked_channel_ids)),
            "masked_channel_ids": list(quality.masked_channel_ids),
        },
        "inference": {
            "backend": str(inference_result.get("backend", "direct")),
            "action_logits": np.asarray(
                inference_result.get("action_logits"), dtype=float
            ).tolist(),
            "finger_logits": np.asarray(
                inference_result.get("finger_logits"), dtype=float
            ).tolist(),
            "applicability_logit": float(inference_result.get("applicability_logit")),
            "action_probs": action_probs.tolist(),
            "model_raw_finger_probs": model_raw_finger_probs.tolist(),
            "finger_probs": finger_probs.tolist(),
            "finger_applicable_prob": float(finger_applicable_prob),
            "action_uncertainty": float(inference_result.get("action_uncertainty", 0.0)),
            "finger_uncertainty": float(inference_result.get("finger_uncertainty", 0.0)),
            "applicability_uncertainty": inference_result.get(
                "applicability_uncertainty"
            ),
            "adaptive_threshold": inference_result.get("adaptive_threshold"),
        },
        "decision": {
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
            "uncertainty_gate_ok": True,
            "decision_reason": str(decision_info.get("decision_reason", "")),
        },
        "actuation": {
            "latency_gate_ok": True,
            "vote_reason": str(actuation_vote.get("reason", "")),
            "vote_finger_counts": actuation_vote.get("finger_votes", {}),
            "vote_action_counts": actuation_vote.get("action_votes", {}),
            "vote_pair_counts": actuation_vote.get("pair_votes", {}),
            "target_action_id": int(voted_decision.action_id),
            "target_finger_id": int(voted_decision.finger_id),
            "speed_scalar": 1.0,
            "suppressed_reason": None,
            "sent": False,
        },
    }


def test_replay_live_capture_matches_deterministic_fixture(tmp_path: Path):
    live_mod = _load_live_module()
    live_dir = tmp_path / "live_infer"
    capture_dir = live_dir / "parity_capture"
    run_dir = tmp_path / "run"
    live_dir.mkdir(parents=True)
    capture_dir.mkdir(parents=True)
    run_dir.mkdir()
    model_path, scaler_path, temperature_path = _write_deterministic_artifacts(run_dir)

    device = torch.device("cpu")
    model, scaler, temperature_state = load_model_artifacts_from_files(
        model_path=model_path,
        scaler_path=scaler_path,
        device=device,
        n_channels=4,
        temperature_path=temperature_path,
    )
    runtime_args = type(
        "Args",
        (),
        {
            "target_fs": 256.0,
            "live_quality_enabled": True,
            "input_clip_abs_z": 6.0,
            "bad_channel_rms_z": 10.0,
            "bad_channel_abs_p95_z": 10.0,
            "bad_channel_clipped_frac": 1.0,
            "bad_window_clipped_frac": 1.0,
            "bad_window_max_masked_channels": 1,
            "postprocess": False,
            "actuation_stability": 2,
        },
    )()
    post_settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.0,
        threshold_applicability=0.0,
        adjacency_enabled=False,
    )
    post_state = PostprocessState()
    rest_bias = live_mod.RestFingerBiasCorrection(enabled=False, min_rest_windows=2, strength=0.0)
    actuation_history: deque[ActuationDecision] = deque(maxlen=2)

    base_times = (np.arange(64, dtype=np.float64) / 256.0).astype(np.float64)
    base_values = np.stack(
        [
            np.linspace(0.0, 0.5, 64, dtype=np.float64),
            np.linspace(0.1, 0.6, 64, dtype=np.float64),
            np.linspace(-0.2, 0.2, 64, dtype=np.float64),
            np.linspace(0.3, -0.1, 64, dtype=np.float64),
        ],
        axis=1,
    )
    records = []
    for idx, start_s in enumerate((0.00, 0.05)):
        end_s = start_s + 0.25
        record = _capture_record(
            live_mod,
            model=model,
            scaler=scaler,
            temperature_state=temperature_state,
            device=device,
            runtime_args=runtime_args,
            post_settings=post_settings,
            post_state=post_state,
            rest_bias=rest_bias,
            actuation_history=actuation_history,
            start_s=start_s,
            end_s=end_s,
            raw_times=base_times + start_s,
            raw_values=base_values,
            latency_ms=80.0 + idx,
        )
        record["candidate_index"] = idx
        record["segment_id"] = 0
        records.append(record)

    runtime_manifest = _modern_runtime_manifest(
        run_dir=run_dir,
        model_path=model_path,
        scaler_path=scaler_path,
        temperature_path=temperature_path,
        post_settings=post_settings,
        actuation_enabled=False,
        actuation_stability=2,
    )
    write_json(live_dir / "live_runtime_manifest.json", runtime_manifest)
    write_json(capture_dir / "captured_windows.json", {"records": records})
    write_json(
        capture_dir / "capture_manifest.json",
        {
            "record_count": len(records),
            "records_sha256": sha256_file(capture_dir / "captured_windows.json"),
            "records_candidate_indices": [0, 1],
            "manifest_seed": {
                "runtime_manifest_path": str(live_dir / "live_runtime_manifest.json")
            },
        },
    )

    report = replay_capture(
        capture_dir=capture_dir,
        device_name="cpu",
        tolerance=1e-5,
    )

    assert report["record_count"] == 2
    assert report["parity"]["preprocessed_tensor_values"]["ok"] is True
    assert report["parity"]["logits"]["ok"] is True
    assert report["parity"]["probabilities"]["ok"] is True
    assert report["parity"]["decoded_outputs"]["ok"] is True
    assert report["per_window"][1]["decision_mismatches"] == []
    assert report["per_window"][1]["actuation_mismatches"] == []


def test_live_parity_capture_persists_initial_state_and_first_window(tmp_path: Path):
    capture = LiveParityCapture(
        root_dir=tmp_path,
        settings=ParityCaptureSettings(enabled=True, max_windows=4, flush_every=8),
        manifest_seed={"runtime_manifest_path": str(tmp_path / "live_runtime_manifest.json")},
    )

    manifest = load_json(capture.manifest_path)
    records_payload = load_json(capture.records_path)
    assert manifest["record_count"] == 0
    assert records_payload["records"] == []

    capture.add({"candidate_index": 7, "segment_id": 2, "decision": {"decision_reason": "test"}})

    manifest = load_json(capture.manifest_path)
    records_payload = load_json(capture.records_path)
    assert manifest["record_count"] == 1
    assert manifest["records_candidate_indices"] == [7]
    assert records_payload["records"][0]["candidate_index"] == 7


def test_replay_live_capture_rejects_records_sha_mismatch(tmp_path: Path):
    capture_dir = tmp_path / "live_infer" / "parity_capture"
    capture_dir.mkdir(parents=True)
    write_json(capture_dir / "captured_windows.json", {"records": [{"candidate_index": 1}]})
    write_json(
        capture_dir / "capture_manifest.json",
        {
            "record_count": 1,
            "records_sha256": "not-the-real-hash",
            "records_candidate_indices": [1],
            "manifest_seed": {},
        },
    )

    with pytest.raises(RuntimeError, match="records_sha256"):
        replay_capture(
            capture_dir=capture_dir,
            device_name="cpu",
            tolerance=1e-5,
        )


def test_replay_live_capture_requires_modern_runtime_manifest_by_default(tmp_path: Path):
    live_mod = _load_live_module()
    live_dir = tmp_path / "live_infer"
    capture_dir = live_dir / "parity_capture"
    run_dir = tmp_path / "run"
    live_dir.mkdir(parents=True)
    capture_dir.mkdir(parents=True)
    run_dir.mkdir()
    model_path, scaler_path, temperature_path = _write_deterministic_artifacts(run_dir)

    device = torch.device("cpu")
    model, scaler, temperature_state = load_model_artifacts_from_files(
        model_path=model_path,
        scaler_path=scaler_path,
        device=device,
        n_channels=4,
        temperature_path=temperature_path,
    )
    runtime_args = type(
        "Args",
        (),
        {
            "target_fs": 256.0,
            "live_quality_enabled": True,
            "input_clip_abs_z": 6.0,
            "bad_channel_rms_z": 10.0,
            "bad_channel_abs_p95_z": 10.0,
            "bad_channel_clipped_frac": 1.0,
            "bad_window_clipped_frac": 1.0,
            "bad_window_max_masked_channels": 1,
            "postprocess": False,
            "actuation_stability": 2,
        },
    )()
    post_settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.0,
        threshold_applicability=0.0,
        adjacency_enabled=False,
    )
    post_state = PostprocessState()
    rest_bias = live_mod.RestFingerBiasCorrection(enabled=False, min_rest_windows=2, strength=0.0)
    actuation_history: deque[ActuationDecision] = deque(maxlen=2)
    base_times = (np.arange(64, dtype=np.float64) / 256.0).astype(np.float64)
    base_values = np.stack(
        [
            np.linspace(0.0, 0.5, 64, dtype=np.float64),
            np.linspace(0.1, 0.6, 64, dtype=np.float64),
            np.linspace(-0.2, 0.2, 64, dtype=np.float64),
            np.linspace(0.3, -0.1, 64, dtype=np.float64),
        ],
        axis=1,
    )
    record = _capture_record(
        live_mod,
        model=model,
        scaler=scaler,
        temperature_state=temperature_state,
        device=device,
        runtime_args=runtime_args,
        post_settings=post_settings,
        post_state=post_state,
        rest_bias=rest_bias,
        actuation_history=actuation_history,
        start_s=0.00,
        end_s=0.25,
        raw_times=base_times,
        raw_values=base_values,
        latency_ms=80.0,
    )
    record["candidate_index"] = 0
    record["segment_id"] = 0
    write_json(live_dir / "live_runtime_manifest.json", {
        "artifacts": {
            "run_dir": str(run_dir),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "temperature_path": str(temperature_path),
        },
        "runtime": {
            "window_sec": 0.25,
            "hop_sec": 0.05,
            "target_fs": 256.0,
            "live_quality_enabled": True,
            "quality_thresholds": {
                "input_clip_abs_z": 6.0,
                "bad_channel_rms_z": 10.0,
                "bad_channel_abs_p95_z": 10.0,
                "bad_channel_clipped_frac": 1.0,
                "bad_window_clipped_frac": 1.0,
                "bad_window_max_masked_channels": 1,
            },
            "postprocess_enabled": False,
            "postprocess_settings": {
                "threshold_action": 0.0,
                "threshold_finger": 0.0,
                "threshold_applicability": 0.0,
            },
            "rest_bias": {
                "enabled": False,
                "strength": 0.0,
                "min_rest_windows": 2,
            },
            "actuation": {
                "enabled": False,
                "actuation_min_prob": 0.24,
                "actuation_stability": 2,
                "actuation_cooldown_ms": 0,
                "actuation_repeat_ms": 0,
                "actuation_min_speed": 0.0,
                "modulate_actuation_speed": False,
                "actuation_speed_gamma": 1.0,
            },
        },
    })
    write_json(capture_dir / "captured_windows.json", {"records": [record]})
    write_json(
        capture_dir / "capture_manifest.json",
        {
            "record_count": 1,
            "records_sha256": sha256_file(capture_dir / "captured_windows.json"),
            "records_candidate_indices": [0],
            "manifest_seed": {
                "runtime_manifest_path": str(live_dir / "live_runtime_manifest.json")
            },
        },
    )

    with pytest.raises(RuntimeError, match="Modern runtime manifest fields"):
        replay_capture(
            capture_dir=capture_dir,
            device_name="cpu",
            tolerance=1e-5,
        )

    report = replay_capture(
        capture_dir=capture_dir,
        device_name="cpu",
        tolerance=1e-5,
        allow_legacy_runtime_defaults=True,
    )

    assert report["evidence_mode"] == "legacy_partial"
    assert "artifacts.model_sha256" in report["missing_modern_runtime_fields"]


def test_audit_live_parity_classifies_window_and_actuation_failures(tmp_path: Path):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    predictions = [
        {
            "window_start_s": 0.00,
            "window_end_s": 0.25,
            "ts_utc": 1000.0,
            "alignment_ok": True,
            "committed_action_id": 1,
            "committed_finger_id": 1,
            "actuation_sent": False,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 0,
            "actuation_suppressed_reason": "pair_stability",
            "window_quality_bad": False,
            "masked_channel_ids": [1],
            "applicability_gate_ok": True,
            "uncertainty_gate_ok": True,
        },
        {
            "window_start_s": 0.05,
            "window_end_s": 0.30,
            "ts_utc": 1000.05,
            "alignment_ok": True,
            "committed_action_id": 1,
            "committed_finger_id": 1,
            "actuation_sent": False,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 0,
            "actuation_suppressed_reason": "pair_stability",
            "window_quality_bad": False,
            "masked_channel_ids": [1],
            "applicability_gate_ok": True,
            "uncertainty_gate_ok": True,
        },
    ]
    (live_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions)
    )
    (live_dir / "window_audit.jsonl").write_text(
        json.dumps({"status": "accepted"}) + "\n"
        + json.dumps({"status": "dropped", "drop_reason": "gap_exceeds_threshold"}) + "\n"
    )
    (live_dir / "segment_breaks.jsonl").write_text("")
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 2,
            "accepted_window_count": 1,
            "dropped_window_reason_counts": {"gap_exceeds_threshold": 1},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
            "masked_window_count": 2,
            "window_quality_bad_count": 0,
            "actuation_suppressed_counts": {"pair_stability": 2},
        },
    )
    write_json(
        live_dir / "live_runtime_manifest.json",
        {
            "stream_resolution": {
                "requested_source_id": "fresh-id",
                "selected_source_id": "fresh-id",
                "source_id_source": "env",
                "source_id_match_mode": "exact_match",
                "selection_matched_by_source_id": True,
                "recovery_used": False,
            },
            "stream_contract": {
                "expected_name": "Muse2-EEG",
                "expected_type": "EEG",
            },
            "artifacts": {
                "run_dir": str(tmp_path / "run"),
                "model_path": str(tmp_path / "run" / "finger_action_model.pt"),
                "scaler_path": str(tmp_path / "run" / "scaler.npz"),
                "temperature_path": str(tmp_path / "run" / "temperature_scaling.json"),
                "model_sha256": "modelhash",
                "scaler_sha256": "scalerhash",
                "temperature_sha256": "temphash",
            },
            "deployment": {"run_dir": str(tmp_path / "run")},
            "runtime": {
                "inference_backend": "direct",
                "live_quality_enabled": True,
                "postprocess_enabled": False,
                "actuation": {
                    "actuation_stability": 2,
                },
            },
        },
    )
    write_json(
        live_dir / "parity_report.json",
        {
            "parity": {
                "preprocessed_tensor_values": {"ok": True},
                "logits": {"ok": True},
                "probabilities": {"ok": True},
                "decoded_outputs": {"ok": True},
            }
        },
    )
    connector_log = tmp_path / "connector.log"
    connector_log.write_text("partial_dropped=4\n")

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[connector_log],
        parity_report_path=live_dir / "parity_report.json",
        distribution_report_path=None,
    )
    statuses = {
        row["issue"]: row["status"] for row in report["suspected_issues"]
    }

    assert statuses["stale_lsl_source_id_wrong_stream"] == "ruled_out"
    assert statuses["strict_live_window_alignment_drops_windows"] == "confirmed"
    assert statuses["streamer_partial_packet_drops_drive_gap_rejection"] == "confirmed"
    assert statuses["segment_break_logic_clears_state_frequently"] == "ruled_out"
    assert (
        statuses["live_quality_enabled_changes_tensors_relative_to_raw_live_windows"]
        == "confirmed"
    )
    assert statuses["true_model_inference_parity_failure"] == "ruled_out"
    assert statuses["commit_actuation_layer_suppresses_otherwise_valid_predictions"] == "confirmed"
    assert statuses["exact_pair_stability_required_for_actuation"] == "confirmed"


def test_audit_live_parity_uses_distribution_report_for_upstream_limiter(tmp_path: Path):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "window_start_s": 0.0,
                "window_end_s": 0.25,
                "alignment_ok": True,
                "committed_action_id": 0,
                "committed_finger_id": 0,
                "actuation_sent": False,
            }
        )
        + "\n"
    )
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 20,
            "accepted_window_count": 18,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
            "raw_top_non_rest_count": 1,
            "committed_valid_non_rest_count": 0,
            "non_rest_sent_count": 0,
            "non_rest_suppressed_counts": {},
            "non_rest_suppressed_reason_counts": {},
        },
    )
    write_json(
        live_dir / "live_runtime_manifest.json",
        {
            "runtime": {"actuation": {"actuation_stability": 2}},
            "artifacts": {},
        },
    )
    write_json(
        live_dir / "parity_report.json",
        {
            "evidence_mode": "decisive",
            "parity": {
                "preprocessed_tensor_values": {"ok": True},
                "logits": {"ok": True},
                "probabilities": {"ok": True},
                "decoded_outputs": {"ok": True},
            },
        },
    )
    write_json(
        live_dir / "live_input_distribution_report.json",
        {
            "distribution_match": {
                "decisive": True,
                "verdict": "shifted_low_amplitude",
                "reason": "quiet capture",
                "median_rms_ratio": 0.58,
            }
        },
    )

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=live_dir / "parity_report.json",
        distribution_report_path=live_dir / "live_input_distribution_report.json",
    )
    statuses = {
        row["issue"]: row["status"] for row in report["suspected_issues"]
    }

    assert statuses["distribution_match_shifted_low_amplitude"] == "confirmed"
    assert statuses["distribution_match_shifted_high_amplitude"] == "ruled_out"
    assert report["dominant_limiter"] == "upstream_signal"


def test_audit_live_parity_uses_non_rest_flow_for_pair_stability_limiter(tmp_path: Path):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps(
            {
                "window_start_s": 0.0,
                "window_end_s": 0.25,
                "alignment_ok": True,
                "raw_top_action_id": 1,
                "raw_top_finger_id": 1,
                "committed_action_id": 1,
                "committed_finger_id": 1,
                "committed_pair_valid": True,
                "actuation_sent": False,
                "actuation_target_action_id": 0,
                "actuation_target_finger_id": 0,
                "actuation_suppressed_reason": "pair_stability",
            }
        )
        + "\n"
    )
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 10,
            "accepted_window_count": 10,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
            "raw_top_non_rest_count": 4,
            "committed_valid_non_rest_count": 4,
            "non_rest_sent_count": 1,
            "non_rest_suppressed_counts": {"pair_stability": 3},
            "non_rest_suppressed_reason_counts": {"pair_stability": 3},
            "actuation_suppressed_counts": {"pair_stability": 3},
        },
    )
    write_json(
        live_dir / "live_runtime_manifest.json",
        {
            "runtime": {"actuation": {"actuation_stability": 2}},
            "artifacts": {},
        },
    )
    write_json(
        live_dir / "parity_report.json",
        {
            "evidence_mode": "decisive",
            "parity": {
                "preprocessed_tensor_values": {"ok": True},
                "logits": {"ok": True},
                "probabilities": {"ok": True},
                "decoded_outputs": {"ok": True},
            },
        },
    )
    write_json(
        live_dir / "live_input_distribution_report.json",
        {
            "distribution_match": {
                "decisive": True,
                "verdict": "nominal",
                "reason": "nominal",
                "recovered_vs_strict_count": 0,
            }
        },
    )

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=live_dir / "parity_report.json",
        distribution_report_path=live_dir / "live_input_distribution_report.json",
    )

    assert report["dominant_limiter"] == "downstream_pair_stability"


def test_audit_live_dir_missing_distribution_report_downgrades_completeness(tmp_path: Path):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps({"alignment_ok": True, "actuation_sent": False}) + "\n"
    )
    (live_dir / "window_audit.jsonl").write_text(
        json.dumps({"status": "accepted"}) + "\n"
    )
    (live_dir / "segment_breaks.jsonl").write_text("")
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 1,
            "accepted_window_count": 1,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
        },
    )
    write_json(
        live_dir / "live_runtime_manifest.json",
        {
            "runtime": {},
            "artifacts": {},
        },
    )
    write_json(
        live_dir / "parity_report.json",
        {
            "evidence_mode": "decisive",
            "parity": {
                "preprocessed_tensor_values": {"ok": True},
                "logits": {"ok": True},
                "probabilities": {"ok": True},
                "decoded_outputs": {"ok": True},
            },
        },
    )

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=live_dir / "parity_report.json",
        distribution_report_path=None,
    )

    assert report["evidence"]["completeness"] == "partial"
    assert report["evidence"]["distribution_evidence"] == "none"
    assert any(
        "live_input_distribution_report.json" in limitation
        for limitation in report["evidence"]["limitations"]
    )


def test_audit_live_dir_with_distribution_report_is_eligible_for_complete_completeness(
    tmp_path: Path,
):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps({"alignment_ok": True, "actuation_sent": False}) + "\n"
    )
    (live_dir / "window_audit.jsonl").write_text(
        json.dumps({"status": "accepted"}) + "\n"
    )
    (live_dir / "segment_breaks.jsonl").write_text("")
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 1,
            "accepted_window_count": 1,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
        },
    )
    write_json(
        live_dir / "live_runtime_manifest.json",
        {
            "runtime": {},
            "artifacts": {},
            "finalization": {
                "termination_reason": "ok",
                "required_outputs_ok": True,
            },
        },
    )
    write_json(
        live_dir / "parity_report.json",
        {
            "evidence_mode": "decisive",
            "parity": {
                "preprocessed_tensor_values": {"ok": True},
                "logits": {"ok": True},
                "probabilities": {"ok": True},
                "decoded_outputs": {"ok": True},
            },
        },
    )
    write_json(
        live_dir / "live_input_distribution_report.json",
        {
            "distribution_match": {
                "decisive": True,
                "verdict": "nominal",
                "reason": "nominal",
            }
        },
    )

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=live_dir / "parity_report.json",
        distribution_report_path=live_dir / "live_input_distribution_report.json",
    )

    assert report["evidence"]["completeness"] == "complete"
    assert report["evidence"]["distribution_evidence"] == "confirmed"
    assert report["evidence"]["decisive_evidence_complete"] is True


def test_audit_live_dir_runtime_finalization_failure_blocks_decisive_completeness(
    tmp_path: Path,
):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps({"alignment_ok": True, "actuation_sent": False}) + "\n"
    )
    (live_dir / "window_audit.jsonl").write_text(
        json.dumps({"status": "accepted"}) + "\n"
    )
    (live_dir / "segment_breaks.jsonl").write_text("")
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 1,
            "accepted_window_count": 1,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
        },
    )
    write_json(
        live_dir / "live_runtime_manifest.json",
        {
            "runtime": {},
            "artifacts": {},
            "finalization": {
                "termination_reason": "required_output_error",
                "required_outputs_ok": False,
                "required_output_errors": ["missing distribution report"],
            },
        },
    )
    write_json(
        live_dir / "parity_report.json",
        {
            "evidence_mode": "decisive",
            "parity": {
                "preprocessed_tensor_values": {"ok": True},
                "logits": {"ok": True},
                "probabilities": {"ok": True},
                "decoded_outputs": {"ok": True},
            },
        },
    )
    write_json(
        live_dir / "live_input_distribution_report.json",
        {
            "distribution_match": {
                "decisive": True,
                "verdict": "nominal",
                "reason": "nominal",
            }
        },
    )

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=live_dir / "parity_report.json",
        distribution_report_path=live_dir / "live_input_distribution_report.json",
    )

    assert report["evidence"]["completeness"] == "partial"
    assert report["evidence"]["decisive_evidence_complete"] is False
    assert any(
        "termination_reason=required_output_error" in failure
        for failure in report["evidence"]["decisive_failures"]
    )
    assert any(
        "required_output_error" in err for err in report["blocking_errors"]
    )


def test_audit_live_dir_legacy_partial_parity_blocks_decisive_completeness(
    tmp_path: Path,
):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps({"alignment_ok": True, "actuation_sent": False}) + "\n"
    )
    (live_dir / "window_audit.jsonl").write_text(
        json.dumps({"status": "accepted"}) + "\n"
    )
    (live_dir / "segment_breaks.jsonl").write_text("")
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 1,
            "accepted_window_count": 1,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
        },
    )
    write_json(
        live_dir / "live_runtime_manifest.json",
        {
            "runtime": {},
            "artifacts": {},
            "finalization": {
                "termination_reason": "ok",
                "required_outputs_ok": True,
            },
        },
    )
    write_json(
        live_dir / "parity_report.json",
        {
            "evidence_mode": "legacy_partial",
            "parity": {
                "preprocessed_tensor_values": {"ok": True},
                "logits": {"ok": True},
                "probabilities": {"ok": True},
                "decoded_outputs": {"ok": True},
            },
        },
    )
    write_json(
        live_dir / "live_input_distribution_report.json",
        {
            "distribution_match": {
                "decisive": True,
                "verdict": "nominal",
                "reason": "nominal",
            }
        },
    )

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=live_dir / "parity_report.json",
        distribution_report_path=live_dir / "live_input_distribution_report.json",
    )

    assert report["evidence"]["completeness"] == "partial"
    assert report["evidence"]["accepted_window_parity_evidence"] == "partial"
    assert report["evidence"]["decisive_evidence_complete"] is False
    assert any(
        "Accepted-window parity evidence is partial or legacy"
        in failure
        for failure in report["evidence"]["decisive_failures"]
    )


def test_audit_live_dir_marks_legacy_outputs_as_partial_evidence(tmp_path: Path):
    live_dir = tmp_path / "legacy_live"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps({"alignment_ok": True, "actuation_sent": False}) + "\n"
    )
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 1,
            "accepted_window_count": 1,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
        },
    )

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=None,
        distribution_report_path=None,
    )

    assert report["evidence"]["completeness"] == "partial"
    assert report["evidence"]["accepted_window_parity_evidence"] == "none"
    assert any(
        "live_runtime_manifest.json" in limitation
        for limitation in report["evidence"]["limitations"]
    )
    assert report["evidence"]["decisive_evidence_complete"] is False
    assert any(
        "Missing required decisive-evidence artifacts" in err
        for err in report["blocking_errors"]
    )


def test_audit_live_dir_reports_blocking_errors_for_malformed_window_audit(tmp_path: Path):
    live_dir = tmp_path / "live_infer"
    live_dir.mkdir()
    (live_dir / "predictions.jsonl").write_text(
        json.dumps({"alignment_ok": True, "actuation_sent": False}) + "\n"
    )
    write_json(
        live_dir / "live_prediction_summary.json",
        {
            "candidate_window_count": 1,
            "accepted_window_count": 1,
            "dropped_window_reason_counts": {},
            "segment_break_count": 0,
            "segment_break_reason_counts": {},
        },
    )
    write_json(live_dir / "live_runtime_manifest.json", {"runtime": {}, "artifacts": {}})
    (live_dir / "window_audit.jsonl").write_text("{not-json}\n")
    (live_dir / "segment_breaks.jsonl").write_text("")

    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[],
        parity_report_path=None,
        distribution_report_path=None,
    )

    assert any("window_audit.jsonl contains 1 malformed line" in err for err in report["blocking_errors"])
    assert report["metrics"]["window_audit_parse_errors"][0]["line_no"] == 1
    assert report["evidence"]["completeness"] == "partial"
