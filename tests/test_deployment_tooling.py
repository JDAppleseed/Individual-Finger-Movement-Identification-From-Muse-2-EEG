import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.live_infer_common import require_deployable_run
from utils.runtime_utils import TemperatureScalingState, save_temperature_scaling


def _load_module(relative_path: str, name: str):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_reference_dataset_path_sanitizer_strips_any_checkout_root():
    mod = _load_module(
        "tools/build_2m16_reference_dataset.py",
        "build_2m16_reference_dataset_test",
    )
    repo_name = Path(__file__).resolve().parents[1].name

    local_path = mod.REPO_ROOT / "Projects" / "2-M16" / "artifact.npz"
    other_checkout = (
        f"/home/example/work/{repo_name}/Projects/2-M16/subjects/2-M16/"
        "sessions/source/processed/eeg_windows.npz"
    )

    assert mod._repo_relative_text(str(local_path)) == "Projects/2-M16/artifact.npz"
    assert mod._repo_relative_text(other_checkout).startswith("Projects/2-M16/")
    assert mod._repo_relative_text("Projects/2-M16/artifact.npz") == "Projects/2-M16/artifact.npz"


def test_pseudo_live_replay_asserts_deployment_invariant():
    mod = _load_module("tools/pseudo_live_replay.py", "pseudo_live_replay_test")

    with pytest.raises(SystemExit, match="Deployment replay invariant failed"):
        mod._assert_deployment_replay_ok(
            target_session_dir=Path("/tmp/session"),
            summary={
                "committed_non_rest_none_count": 1,
                "committed_rest_non_none_count": 0,
                "sent_non_rest_none_count": 0,
                "sent_rest_non_none_count": 0,
                "deployment_pair_invariant_ok": False,
            },
            replay_metrics={
                "committed_non_rest_none_count": 0,
                "committed_rest_non_none_count": 1,
                "sent_non_rest_none_count": 0,
                "sent_rest_non_none_count": 0,
                "deployment_pair_invariant_ok": True,
            },
        )


def test_cnn_lstm_model_emits_applicability_head_when_enabled():
    model = CNNLSTMFingerActionNet(
        n_channels=4,
        n_fingers=5,
        n_actions=3,
        finger_applicability_head=True,
    )
    xb = torch.zeros((2, 64, 4), dtype=torch.float32)

    finger_logits, action_logits, applicability_logits = model(xb)

    assert tuple(finger_logits.shape) == (2, 5)
    assert tuple(action_logits.shape) == (2, 3)
    assert tuple(applicability_logits.shape) == (2,)
    assert "finger_applicability_head.weight" in model.state_dict()


def test_require_deployable_run_rejects_missing_applicability_temperature(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_config.json").write_text(
        json.dumps(
            {
                "active_finger_head": True,
                "finger_applicability_head": True,
            }
        )
    )
    torch.save(
        CNNLSTMFingerActionNet(
            n_channels=4,
            n_fingers=5,
            n_actions=3,
            finger_applicability_head=True,
        ).state_dict(),
        run_dir / "finger_action_model.pt",
    )

    with pytest.raises(RuntimeError, match="applicability_temperature"):
        require_deployable_run(run_dir)


def test_smoke_inference_infers_active_finger_head_from_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    mod = _load_module("tools/smoke_inference.py", "smoke_inference_test")

    np.savez(
        tmp_path / "eeg_windows.npz",
        X=np.zeros((1, 64, 4), dtype=np.float32),
        y_action=np.array([0], dtype=np.int64),
        y_finger=np.array([0], dtype=np.int64),
    )
    np.savez(
        tmp_path / "scaler.npz",
        mean=np.zeros((4,), dtype=np.float32),
        std=np.ones((4,), dtype=np.float32),
        channels=np.array(4, dtype=np.int64),
    )
    torch.save(
        CNNLSTMFingerActionNet(
            n_channels=4,
            n_fingers=5,
            n_actions=3,
            finger_applicability_head=True,
        ).state_dict(),
        tmp_path / "finger_action_model.pt",
    )
    (tmp_path / "train_config.json").write_text(
        json.dumps(
            {
                "active_finger_head": True,
                "finger_applicability_head": True,
            }
        )
    )
    save_temperature_scaling(
        tmp_path / "temperature_scaling.json",
        TemperatureScalingState(
            action_temperature=1.0,
            finger_temperature=1.0,
            applicability_temperature=1.0,
            has_applicability_temperature=True,
            source="test",
        ),
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "smoke_inference.py",
            "--npz",
            str(tmp_path / "eeg_windows.npz"),
            "--model",
            str(tmp_path / "finger_action_model.pt"),
            "--scaler",
            str(tmp_path / "scaler.npz"),
            "--device",
            "cpu",
        ],
    )

    mod.main()
    stdout = capsys.readouterr().out
    assert "Smoke inference OK" in stdout
    assert "applicability_prob=" in stdout
    assert "applicability_gate_ok=" in stdout


def test_live_prediction_summary_includes_full_runtime_metrics(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_summary_builder_test")

    pred_log = tmp_path / "predictions.jsonl"
    records = [
        {
            "window_start_s": 0.00,
            "window_end_s": 0.25,
            "ts_utc": 1000.0,
            "alignment_ok": True,
            "raw_top_action_id": 0,
            "raw_top_finger_id": 5,
            "committed_action_id": 0,
            "committed_finger_id": 0,
            "decision_reason": "raw_argmax_gated",
            "actuation_vote_reason": "pair_stability",
            "actuation_suppressed_reason": "pair_stability",
            "actuation_sent": False,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 0,
            "latency_ms": 80.0,
            "joint_conf": 0.0,
            "action_uncertainty": 0.0,
            "window_quality_bad": False,
            "masked_channel_ids": [],
            "committed_pair_valid": True,
            "uncertainty_gate_ok": True,
            "applicability_gate_ok": True,
        },
        {
            "window_start_s": 0.05,
            "window_end_s": 0.30,
            "ts_utc": 1000.05,
            "alignment_ok": True,
            "raw_top_action_id": 1,
            "raw_top_finger_id": 1,
            "committed_action_id": 1,
            "committed_finger_id": 1,
            "decision_reason": "raw_argmax_gated",
            "actuation_vote_reason": "exact_pair_stability",
            "actuation_suppressed_reason": None,
            "actuation_sent": True,
            "actuation_target_action_id": 1,
            "actuation_target_finger_id": 1,
            "latency_ms": 90.0,
            "joint_conf": 0.72,
            "action_uncertainty": 0.0,
            "window_quality_bad": False,
            "masked_channel_ids": [2],
            "committed_pair_valid": True,
            "uncertainty_gate_ok": True,
            "applicability_gate_ok": True,
        },
        {
            "window_start_s": 0.10,
            "window_end_s": 0.35,
            "ts_utc": 1000.10,
            "alignment_ok": False,
            "raw_top_action_id": None,
            "raw_top_finger_id": None,
            "committed_action_id": 0,
            "committed_finger_id": 0,
            "decision_reason": "alignment_fail",
            "actuation_vote_reason": None,
            "actuation_suppressed_reason": None,
            "actuation_sent": False,
            "actuation_target_action_id": 0,
            "actuation_target_finger_id": 0,
            "latency_ms": None,
            "joint_conf": None,
            "action_uncertainty": None,
            "window_quality_bad": None,
            "masked_channel_ids": None,
            "committed_pair_valid": True,
            "uncertainty_gate_ok": True,
            "applicability_gate_ok": True,
        },
    ]
    pred_log.write_text("".join(json.dumps(row) + "\n" for row in records))

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    shard_dtype = np.dtype(
        [
            ("seq", "<i8"),
            ("lsl_ts_raw", "<f8"),
            ("lsl_ts_mono", "<f8"),
            ("local_ts", "<f8"),
            ("flags", "<i8"),
            ("segment_id", "<i8"),
            ("clamped", "i1"),
            ("sample", "<f8", (4,)),
        ]
    )
    shard = np.zeros(2, dtype=shard_dtype)
    shard["seq"] = np.asarray([0, 1], dtype=np.int64)
    shard["flags"] = np.asarray([0, mod.RAW_FLAG_NONFINITE], dtype=np.int64)
    shard["sample"][0] = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    shard["sample"][1] = np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float64)
    np.save(raw_dir / "eeg_raw_shard_000.npy", shard, allow_pickle=False)

    window_audit_path = tmp_path / "window_audit.jsonl"
    window_audit_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "candidate_index": 0,
                        "status": "accepted",
                        "masked_channel_count": 0,
                    }
                ),
                json.dumps(
                    {
                        "candidate_index": 1,
                        "status": "accepted",
                        "masked_channel_count": 1,
                    }
                ),
                json.dumps(
                    {
                        "candidate_index": 2,
                        "status": "dropped",
                        "drop_reason": "gap_exceeds_threshold",
                    }
                ),
            ]
        )
        + "\n"
    )
    segment_break_path = tmp_path / "segment_breaks.jsonl"
    segment_break_path.write_text(
        json.dumps({"reason": "stream_gap"}) + "\n",
    )
    runtime_manifest_path = tmp_path / "live_runtime_manifest.json"
    runtime_manifest_path.write_text(
        json.dumps(
            {
                "stream_resolution": {
                    "requested_source_id": "fresh-id",
                    "selected_source_id": "fresh-id",
                    "source_id_source": "env",
                    "selection_matched_by_source_id": True,
                    "channel_labels": ["AF7", "TP9", "AF8", "TP10"],
                },
                "stream_selection": {
                    "expected_channel_labels": ["TP9", "AF7", "AF8", "TP10"],
                },
                "stream_contract": {
                    "expected": {
                        "required_labels": ["TP9", "AF7", "AF8", "TP10"],
                    },
                    "resolved": {
                        "channel_labels": ["AF7", "TP9", "AF8", "TP10"],
                        "channel_reorder_to_model_order": [1, 0, 2, 3],
                        "channel_reorder_applied": True,
                    },
                    "contract_ok": True,
                    "mismatches": [],
                },
                "artifacts": {
                    "run_dir": str(tmp_path),
                    "model_path": str(tmp_path / "finger_action_model.pt"),
                    "model_sha256": "modelhash",
                    "scaler_path": str(tmp_path / "scaler.npz"),
                    "scaler_sha256": "scalerhash",
                    "temperature_path": str(tmp_path / "temperature_scaling.json"),
                    "temperature_sha256": "temphash",
                },
                "runtime": {
                    "actuation": {
                        "enabled": True,
                        "actuation_min_prob": 0.2,
                        "actuation_stability": 3,
                    },
                },
            }
        )
    )

    summary_path = tmp_path / "live_prediction_summary.json"
    mod._build_live_prediction_summary(
        pred_log_path=pred_log,
        summary_path=summary_path,
        raw_dir=raw_dir,
        dropped_windows=3,
        dropped_nonfinite_samples=4,
        dropped_nonfinite_windows=1,
        segment_break_count=0,
        candidate_window_count=3,
        accepted_window_count=2,
        window_audit_path=window_audit_path,
        segment_break_path=segment_break_path,
        runtime_manifest_path=runtime_manifest_path,
    )

    summary = json.loads(summary_path.read_text())
    assert summary["record_count"] == 3
    assert summary["valid_window_count"] == 2
    assert summary["alignment_fail_count"] == 1
    assert summary["actuation_sent_count"] == 1
    assert summary["pair_counts"]["REST+NONE"] == 2
    assert summary["pair_counts"]["OPEN+THUMB"] == 1
    assert summary["latency_ms"]["p50"] == pytest.approx(85.0)
    assert summary["raw_action_counts"]["1"] == 1
    assert summary["actuation_sent_pair_counts"]["1:1"] == 1
    assert summary["actuation_vote_reason_counts"]["exact_pair_stability"] == 1
    assert summary["actuation_suppressed_counts"]["pair_stability"] == 1
    assert summary["dropped_windows"] == 3
    assert summary["candidate_window_count"] == 3
    assert summary["accepted_window_count"] == 2
    assert summary["dropped_window_reason_counts"]["gap_exceeds_threshold"] == 1
    assert summary["segment_break_count"] == 1
    assert summary["segment_break_reason_counts"]["stream_gap"] == 1
    assert summary["stream_resolution"]["requested_source_id"] == "fresh-id"
    assert summary["artifact_provenance"]["model_sha256"] == "modelhash"
    assert summary["actuation_enabled"] is True
    assert summary["actuation_runtime"]["actuation_stability"] == 3
    assert summary["runtime_manifest_path"] == str(runtime_manifest_path)
    assert "segment_break_count_vs_segment_break_log" in summary["reconciliation"]["mismatches"]
    assert summary["raw_channel_stats"]["flagged_nonfinite_rows"] == 1
    assert summary["raw_channel_stats"]["raw_stream_order"]["channel_labels"] == [
        "AF7",
        "TP9",
        "AF8",
        "TP10",
    ]
    assert summary["raw_channel_stats"]["model_order"]["channel_labels"] == [
        "TP9",
        "AF7",
        "AF8",
        "TP10",
    ]
    assert summary["raw_channel_stats"]["model_order"]["channels"][0]["mean"] == pytest.approx(2.0)
    assert summary["raw_top_non_rest_count"] == 1
    assert summary["committed_valid_non_rest_count"] == 1
    assert summary["non_rest_sent_count"] == 1


def test_live_prediction_summary_finalization_sync_persists_runtime_manifest_state(
    tmp_path: Path,
):
    mod = _load_module("7_live_infer_and_actuate.py", "live_summary_sync_test")

    summary_path = tmp_path / "live_prediction_summary.json"
    summary_path.write_text(json.dumps({"record_count": 1}))
    mod._sync_summary_finalization(
        summary_path=summary_path,
        runtime_manifest_finalization={
            "finalized_at": "2026-04-05T00:00:00Z",
            "termination_reason": "required_output_error",
            "summary_path": str(summary_path),
            "summary_write_error": None,
            "distribution_report_path": str(
                tmp_path / "live_input_distribution_report.json"
            ),
            "distribution_report_write_error": None,
            "cleanup_errors": ["window_audit_log_close_error"],
            "required_outputs_ok": False,
            "required_output_errors": ["missing parity report"],
            "post_run_commands": {
                "replay": "python replay.py",
                "audit": "python audit.py",
            },
            "output_hashes": {"summary_sha256": "ignored"},
        },
    )

    payload = json.loads(summary_path.read_text())
    assert payload["runtime_manifest_finalization"] == {
        "finalized_at": "2026-04-05T00:00:00Z",
        "termination_reason": "required_output_error",
        "summary_path": str(summary_path),
        "summary_write_error": None,
        "distribution_report_path": str(
            tmp_path / "live_input_distribution_report.json"
        ),
        "distribution_report_write_error": None,
        "parity_report_path": None,
        "parity_report_write_error": None,
        "cleanup_errors": ["window_audit_log_close_error"],
        "required_outputs_ok": False,
        "required_output_errors": ["missing parity report"],
        "post_run_commands": {
            "replay": "python replay.py",
            "audit": "python audit.py",
        },
    }


def test_live_preflight_distribution_probe_assessment_nominal():
    mod = _load_module("tools/live_preflight.py", "live_preflight_assess_nominal")

    errors, warnings, compact = mod._assess_distribution_probe(
        {
            "distribution_claim_decisive": True,
            "distribution_match": {
                "decisive": True,
                "verdict": "nominal",
                "catastrophic": False,
                "median_rms_ratio": 0.98,
                "recovered_vs_strict_count": 1,
            },
            "alignment": {
                "relaxed": {
                    "accepted_count": 18,
                    "candidate_count": 20,
                    "quality_bad_rate": 0.05,
                }
            },
        }
    )

    assert errors == []
    assert warnings == []
    assert compact["verdict"] == "nominal"


def test_live_preflight_distribution_probe_assessment_non_decisive_is_blocking():
    mod = _load_module("tools/live_preflight.py", "live_preflight_assess_nondecisive")

    errors, warnings, compact = mod._assess_distribution_probe(
        {
            "distribution_claim_decisive": False,
            "distribution_match": {
                "decisive": False,
                "verdict": "nominal",
                "catastrophic": False,
                "reason": "missing reorder proof",
            },
            "alignment": {
                "relaxed": {
                    "accepted_count": 18,
                    "candidate_count": 18,
                    "quality_bad_rate": 0.0,
                }
            },
        }
    )

    assert any("non-decisive" in error for error in errors)
    assert warnings == []
    assert compact["decisive"] is False


def test_live_preflight_distribution_probe_assessment_shifted_low_amplitude():
    mod = _load_module("tools/live_preflight.py", "live_preflight_assess_quiet")

    errors, warnings, compact = mod._assess_distribution_probe(
        {
            "distribution_claim_decisive": True,
            "distribution_match": {
                "decisive": True,
                "verdict": "shifted_low_amplitude",
                "catastrophic": False,
                "reason": "quiet capture",
                "median_rms_ratio": 0.58,
                "recovered_vs_strict_count": 2,
            },
            "alignment": {
                "relaxed": {
                    "accepted_count": 16,
                    "candidate_count": 20,
                    "quality_bad_rate": 0.10,
                }
            },
        }
    )

    assert errors == []
    assert any("shifted_low_amplitude" in warning for warning in warnings)
    assert compact["verdict"] == "shifted_low_amplitude"


def test_live_raw_distribution_flags_channel_local_low_amplitude():
    mod = _load_module(
        "tools/analyze_live_raw_inputs.py",
        "live_raw_distribution_channel_low",
    )

    report = mod._classify_distribution_match(
        relaxed_stats={
            "accepted_count": 20,
            "accepted_rate": 1.0,
            "quality_bad_rate": 0.0,
            "masked_window_rate": 0.0,
            "recovered_vs_strict_count": 0,
            "prepared_summary": {
                "prepared_rms_mean": [0.45, 0.90, 0.95, 0.70],
                "prepared_total_clip_mean": 0.0,
                "spectral_proxies": {},
            },
        },
        strict_stats={"accepted_rate": 1.0},
        offline_reference={
            "prepared_rms_mean": [1.0, 1.0, 1.0, 1.0],
            "prepared_total_clip_mean": 0.0,
            "spectral_proxies": {},
        },
        decisive=True,
    )

    assert report["verdict"] == "shifted_low_amplitude"
    assert report["min_rms_ratio"] == pytest.approx(0.45)
    assert report["low_rms_channel_count"] == 2


def test_live_preflight_distribution_probe_assessment_catastrophic():
    mod = _load_module("tools/live_preflight.py", "live_preflight_assess_noisy")

    errors, warnings, compact = mod._assess_distribution_probe(
        {
            "distribution_claim_decisive": True,
            "distribution_match": {
                "decisive": True,
                "verdict": "catastrophic",
                "catastrophic": True,
                "reason": "gross multi-channel deviation",
                "median_rms_ratio": 2.8,
                "recovered_vs_strict_count": 0,
            },
            "alignment": {
                "relaxed": {
                    "accepted_count": 2,
                    "candidate_count": 20,
                    "quality_bad_rate": 0.85,
                }
            },
        }
    )

    assert any("catastrophic" in error for error in errors)
    assert any("quality rejection is overwhelming" in error for error in errors)
    assert compact["verdict"] == "catastrophic"


def test_live_preflight_probe_stream_records_step7_channel_reorder(
    monkeypatch: pytest.MonkeyPatch,
):
    preflight_mod = _load_module("tools/live_preflight.py", "live_preflight_reorder_test")
    live_mod = _load_module("7_live_infer_and_actuate.py", "live_preflight_reorder_live_mod")

    resolved = SimpleNamespace(
        resolution={
            "name": "Muse2-EEG",
            "type": "EEG",
            "nominal_srate": 256.0,
            "channel_count": 4,
            "channel_labels": ["AF7", "TP9", "AF8", "TP10"],
            "requested_source_id": "fresh-id",
            "selected_source_id": "fresh-id",
            "selection_matched_by_source_id": True,
            "source_id_source": "env",
        },
        inlet=object(),
    )
    monkeypatch.setattr(live_mod, "_resolve_lsl_inlet", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(
        live_mod,
        "_resolve_expected_channel_labels",
        lambda settings, deployment_run_dir: (
            ["TP9", "AF7", "AF8", "TP10"],
            "test.expected_labels",
        ),
    )
    monkeypatch.setattr(live_mod, "_load_train_config", lambda deployment_run_dir: {})
    monkeypatch.setattr(
        live_mod,
        "_resolve_effective_target_fs",
        lambda train_config, window_sec, requested_target_fs: (256.0, {}),
    )

    _, _, stream_contract = preflight_mod._probe_stream(
        live_mod=live_mod,
        settings={
            "stream_name": "Muse2-EEG",
            "stream_type": "EEG",
            "window_sec": 0.25,
            "target_fs": 256.0,
        },
        deployment_run_dir=Path("/tmp/fake_run"),
        cli_source_id=None,
        env_source_id="fresh-id",
        config_source_id=None,
    )

    assert stream_contract["resolved"]["channel_reorder_to_model_order"] == [1, 0, 2, 3]
    assert stream_contract["resolved"]["channel_reorder_applied"] is True


def test_live_preflight_probe_stream_requires_expected_channel_labels(
    monkeypatch: pytest.MonkeyPatch,
):
    preflight_mod = _load_module("tools/live_preflight.py", "live_preflight_require_labels")
    live_mod = _load_module("7_live_infer_and_actuate.py", "live_preflight_require_labels_mod")

    resolved = SimpleNamespace(
        resolution={
            "name": "Muse2-EEG",
            "type": "EEG",
            "nominal_srate": 256.0,
            "channel_count": 4,
            "channel_labels": ["AF7", "TP9", "AF8", "TP10"],
            "requested_source_id": "fresh-id",
            "selected_source_id": "fresh-id",
            "selection_matched_by_source_id": True,
            "source_id_source": "env",
        },
        inlet=object(),
    )
    monkeypatch.setattr(live_mod, "_resolve_lsl_inlet", lambda *args, **kwargs: resolved)
    monkeypatch.setattr(
        live_mod,
        "_resolve_expected_channel_labels",
        lambda settings, deployment_run_dir: ([], None),
    )

    with pytest.raises(RuntimeError, match="cannot prove model-order channel mapping"):
        preflight_mod._probe_stream(
            live_mod=live_mod,
            settings={
                "stream_name": "Muse2-EEG",
                "stream_type": "EEG",
                "window_sec": 0.25,
                "target_fs": 256.0,
            },
            deployment_run_dir=Path("/tmp/fake_run"),
            cli_source_id=None,
            env_source_id="fresh-id",
            config_source_id=None,
        )


def test_run_live_preflight_fails_without_source_id_or_parity_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_module("tools/live_preflight.py", "live_preflight_report_failclosed")

    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessA"
    run_dir = session_dir / "processed" / "models" / "run_001"
    out_dir = session_dir / "processed" / "live_infer"
    run_dir.mkdir(parents=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "finger_action_model.pt"
    scaler_path = run_dir / "scaler.npz"
    temperature_path = run_dir / "temperature_scaling.json"
    model_path.write_text("model")
    scaler_path.write_text("scaler")
    temperature_path.write_text("{}")

    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "project_name": "Demo",
                "subject_id": "S01",
                "settings": {
                    "session_dir": str(session_dir),
                    "parity_capture_enabled": False,
                    "lsl_source_id": None,
                    "stream_name": "Muse2-EEG",
                    "stream_type": "EEG",
                    "REQUIRED_LSL_LABELS": ["TP9", "AF7", "AF8", "TP10"],
                    "REQUIRE_EXACTLY_4_CHANNELS": True,
                    "alignment_internal_max_gap_s": 0.06,
                    "latency_policy": "drop",
                    "latency_threshold_ms": 150.0,
                },
            }
        )
    )

    fake_live_mod = SimpleNamespace(
        _build_arg_parser=lambda: (None, {"parity_capture_enabled": False}),
        _resolve_expected_channel_labels=lambda settings, deployment_run_dir: (
            ["TP9", "AF7", "AF8", "TP10"],
            "config.REQUIRED_LSL_LABELS",
        ),
        _require_expected_channel_labels=lambda labels, source: list(labels),
        resolve_live_launch_plan=lambda **kwargs: SimpleNamespace(
            project_name="Demo",
            subject_id="S01",
            selection_source="config",
            session_dir_inferred=False,
            selected_session_dir=session_dir,
            explicit_overrides=[],
            chosen_run_dir=run_dir,
            model_path=model_path,
            scaler_path=scaler_path,
            temperature_path=temperature_path,
            out_dir=out_dir,
            no_file_io=False,
            record_raw=False,
        ),
    )
    monkeypatch.setattr(mod, "_load_live_module", lambda: fake_live_mod)
    monkeypatch.delenv("LSL_SOURCE_ID", raising=False)

    report = mod.run_live_preflight(config_path=config_path, skip_smoke=True)

    assert report["ready"] is False
    assert any("No explicit live LSL source_id is pinned" in err for err in report["errors"])
    assert any("parity_capture_enabled is not enabled" in err for err in report["errors"])
    assert report["effective_contract"]["requested_source_id"] is None
    assert report["effective_contract"]["parity_capture_enabled"] is False
    assert report["report_version"] == 1
    assert report["launch_plan_resolution_succeeded"] is True
    assert report["launch_plan_contract_status"] == "ok"
    assert report["launch_plan"]["schema_version"] == 1
    assert report["launch_plan"]["selected_session_dir"] == str(session_dir)
    assert report["launch_plan"]["model_path"] == str(model_path)
    assert report["launch_plan"]["scaler_path"] == str(scaler_path)
    assert report["launch_plan"]["out_dir"] == str(out_dir)


def test_run_live_preflight_accepts_env_source_id_and_emits_recommended_commands(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_module("tools/live_preflight.py", "live_preflight_report_ready")

    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessB"
    run_dir = session_dir / "processed" / "models" / "run_002"
    out_dir = session_dir / "processed" / "live_infer"
    run_dir.mkdir(parents=True)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    model_path = run_dir / "finger_action_model.pt"
    scaler_path = run_dir / "scaler.npz"
    temperature_path = run_dir / "temperature_scaling.json"
    model_path.write_text("model")
    scaler_path.write_text("scaler")
    temperature_path.write_text("{}")

    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "project_name": "Demo",
                "subject_id": "S01",
                "settings": {
                    "session_dir": str(session_dir),
                    "parity_capture_enabled": True,
                    "lsl_source_id": None,
                    "stream_name": "Muse2-EEG",
                    "stream_type": "EEG",
                    "REQUIRED_LSL_LABELS": ["TP9", "AF7", "AF8", "TP10"],
                    "REQUIRE_EXACTLY_4_CHANNELS": True,
                    "alignment_internal_max_gap_s": 0.06,
                    "latency_policy": "drop",
                    "latency_threshold_ms": 150.0,
                },
            }
        )
    )

    fake_live_mod = SimpleNamespace(
        _build_arg_parser=lambda: (None, {"parity_capture_enabled": True}),
        _resolve_expected_channel_labels=lambda settings, deployment_run_dir: (
            ["TP9", "AF7", "AF8", "TP10"],
            "config.REQUIRED_LSL_LABELS",
        ),
        _require_expected_channel_labels=lambda labels, source: list(labels),
        resolve_live_launch_plan=lambda **kwargs: SimpleNamespace(
            project_name="Demo",
            subject_id="S01",
            selection_source="config",
            session_dir_inferred=False,
            selected_session_dir=session_dir,
            explicit_overrides=[],
            chosen_run_dir=run_dir,
            model_path=model_path,
            scaler_path=scaler_path,
            temperature_path=temperature_path,
            out_dir=out_dir,
            no_file_io=False,
            record_raw=False,
        ),
    )
    monkeypatch.setattr(mod, "_load_live_module", lambda: fake_live_mod)
    monkeypatch.setenv("LSL_SOURCE_ID", "env-source-123")

    report = mod.run_live_preflight(config_path=config_path, skip_smoke=True)

    assert report["ready"] is True
    assert report["errors"] == []
    assert report["report_version"] == 1
    assert report["launch_plan_resolution_succeeded"] is True
    assert report["launch_plan_contract_status"] == "ok"
    assert report["launch_plan"]["schema_version"] == 1
    assert report["launch_plan"]["selected_session_dir"] == str(session_dir)
    assert report["effective_contract"]["requested_source_id"] == "env-source-123"
    assert report["effective_contract"]["source_id_source"] == "env"
    assert report["effective_contract"]["parity_capture_enabled"] is True
    assert "--lsl-source-id env-source-123" in report["recommended_commands"]["live"]
    assert f"--model-path {model_path}" in report["recommended_commands"]["live"]
    assert f"--scaler-path {scaler_path}" in report["recommended_commands"]["live"]
    assert "--parity-capture-enabled" in report["recommended_commands"]["live"]
    assert not any("lsl_source_id is blank in config" in warning for warning in report["warnings"])


def test_run_live_preflight_classifies_nonfresh_out_dir_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_module("tools/live_preflight.py", "live_preflight_outdir_classification")
    config_path = tmp_path / "infer.json"
    config_path.write_text(json.dumps({"settings": {}}))

    fake_live_mod = SimpleNamespace(
        _build_arg_parser=lambda: (None, {"parity_capture_enabled": True}),
        resolve_live_launch_plan=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "Output dir already exists and is not empty: /tmp/live_infer. Choose a fresh --out-dir for an unambiguous live run."
            )
        ),
    )
    monkeypatch.setattr(mod, "_load_live_module", lambda: fake_live_mod)

    report = mod.run_live_preflight(
        config_path=config_path,
        allow_no_source_id=True,
        allow_no_parity_capture=True,
        skip_smoke=True,
    )

    assert report["ready"] is False
    assert any(
        error.startswith("preflight_out_dir_not_fresh:")
        for error in report["errors"]
    )


def test_run_live_preflight_flags_empty_launch_plan_after_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_module("tools/live_preflight.py", "live_preflight_launch_plan_contract")
    config_path = tmp_path / "infer.json"
    config_path.write_text(json.dumps({"settings": {}}))

    launch_plan = SimpleNamespace(
        project_name="Demo",
        subject_id="S01",
        selection_source="config",
        session_dir_inferred=False,
        selected_session_dir=tmp_path / "session",
        explicit_overrides=[],
        chosen_run_dir=tmp_path / "run",
        model_path=tmp_path / "run" / "finger_action_model.pt",
        scaler_path=tmp_path / "run" / "scaler.npz",
        temperature_path=tmp_path / "run" / "temperature_scaling.json",
        out_dir=tmp_path / "live_infer",
        no_file_io=False,
        record_raw=False,
    )

    fake_live_mod = SimpleNamespace(
        _build_arg_parser=lambda: (None, {"parity_capture_enabled": True}),
        resolve_live_launch_plan=lambda **kwargs: launch_plan,
    )
    monkeypatch.setattr(mod, "_load_live_module", lambda: fake_live_mod)
    monkeypatch.setattr(mod, "serialize_live_preflight_launch_plan", lambda lp: {})

    report = mod.run_live_preflight(
        config_path=config_path,
        allow_no_source_id=True,
        allow_no_parity_capture=True,
        skip_smoke=True,
    )

    assert report["launch_plan_resolution_succeeded"] is True
    assert report["launch_plan"] == {}
    assert report["launch_plan_contract_status"] == "preflight_launch_plan_empty"
    assert any(
        "preflight_launch_plan_contract_violation" in error
        for error in report["errors"]
    )


def test_run_live_preflight_preserves_resolved_plan_when_out_dir_validation_fails(
    tmp_path: Path,
):
    mod = _load_module("tools/live_preflight.py", "live_preflight_preserve_plan_outdir")

    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    model_path = artifact_dir / "finger_action_model.pt"
    scaler_path = artifact_dir / "scaler.npz"
    temperature_path = artifact_dir / "temperature_scaling.json"
    model_path.write_text("model")
    scaler_path.write_text("scaler")
    temperature_path.write_text("{}")
    out_dir = session_dir / "processed" / "live_infer_001"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.txt").write_text("stale")

    config_path = tmp_path / "infer.json"
    config_path.write_text(json.dumps({"settings": {}}))

    report = mod.run_live_preflight(
        config_path=config_path,
        session_dir=str(session_dir),
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        out_dir=str(out_dir),
        allow_no_source_id=True,
        allow_no_parity_capture=True,
        skip_smoke=True,
    )

    assert report["ready"] is False
    assert report["launch_plan_resolution_succeeded"] is True
    assert report["launch_plan_resolved_before_validation"] is True
    assert report["launch_plan_contract_status"] == "ok"
    assert report["launch_plan"]["selected_session_dir"] == str(session_dir.resolve())
    assert report["launch_plan"]["model_path"] == str(model_path.resolve())
    assert report["launch_plan"]["scaler_path"] == str(scaler_path.resolve())
    assert report["launch_plan"]["out_dir"] == str(out_dir.resolve())
    assert any(
        error.startswith("preflight_out_dir_not_fresh:")
        for error in report["errors"]
    )
    assert any(
        error.startswith("preflight_out_dir_not_fresh:")
        for error in report["launch_plan_validation_errors"]
    )
    assert report["launch_plan_inputs"]["model_path"]["source"] == "cli_override"
    assert report["launch_plan_inputs"]["out_dir"]["source"] == "cli_override"


def test_run_live_preflight_preserves_resolved_plan_when_model_path_is_invalid(
    tmp_path: Path,
):
    mod = _load_module("tools/live_preflight.py", "live_preflight_preserve_plan_model")

    session_dir = tmp_path / "session"
    session_dir.mkdir(parents=True)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    model_path = artifact_dir / "missing_model.pt"
    scaler_path = artifact_dir / "scaler.npz"
    scaler_path.write_text("scaler")
    out_dir = session_dir / "processed" / "live_infer_002"
    config_path = tmp_path / "infer.json"
    config_path.write_text(json.dumps({"settings": {}}))

    report = mod.run_live_preflight(
        config_path=config_path,
        session_dir=str(session_dir),
        model_path=str(model_path),
        scaler_path=str(scaler_path),
        out_dir=str(out_dir),
        allow_no_source_id=True,
        allow_no_parity_capture=True,
        skip_smoke=True,
    )

    assert report["ready"] is False
    assert report["launch_plan_resolution_succeeded"] is True
    assert report["launch_plan_contract_status"] == "ok"
    assert report["launch_plan"]["model_path"] == str(model_path.resolve())
    assert any(
        error.startswith("preflight_model_path_invalid:")
        for error in report["errors"]
    )
    assert any(
        error.startswith("preflight_model_path_invalid:")
        for error in report["launch_plan_validation_errors"]
    )
    assert report["launch_plan_inputs"]["model_path"]["source"] == "cli_override"
    assert report["launch_plan_inputs"]["session_dir"]["source"] == "cli_override"


def test_live_preflight_main_writes_report_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    mod = _load_module("tools/live_preflight.py", "live_preflight_main_report")
    config_path = tmp_path / "infer.json"
    config_path.write_text("{}")
    report_path = tmp_path / "live_preflight_report.json"

    monkeypatch.setattr(
        mod,
        "run_live_preflight",
        lambda **kwargs: {
            "ready": True,
            "warnings": [],
            "errors": [],
            "launch_plan": {"out_dir": str(tmp_path / "live_infer")},
            "effective_contract": {},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "live_preflight.py",
            "--config",
            str(config_path),
            "--report-path",
            str(report_path),
        ],
    )

    exit_code = mod.main()

    assert exit_code == 0
    assert json.loads(report_path.read_text())["ready"] is True
    assert capsys.readouterr().out == ""


class _FakeDesc:
    def __init__(self, labels: list[str], index: int = 0):
        self._labels = labels
        self._index = index

    def child(self, name: str):
        if name == "channels":
            return self
        if name == "channel":
            return _FakeDesc(self._labels, 0)
        return _FakeDesc([], 0)

    def child_value(self, name: str) -> str:
        if name == "label" and self._index < len(self._labels):
            return str(self._labels[self._index])
        return ""

    def next_sibling(self):
        return _FakeDesc(self._labels, self._index + 1)


class _FakeStream:
    def __init__(
        self,
        *,
        name: str,
        stream_type: str,
        source_id: str,
        uid: str,
        labels: list[str],
    ):
        self._name = name
        self._type = stream_type
        self._source_id = source_id
        self._uid = uid
        self._labels = labels

    def name(self) -> str:
        return self._name

    def type(self) -> str:
        return self._type

    def channel_count(self) -> int:
        return len(self._labels)

    def nominal_srate(self) -> float:
        return 256.0

    def source_id(self) -> str:
        return self._source_id

    def uid(self) -> str:
        return self._uid

    def desc(self):
        return _FakeDesc(self._labels)


class _FakeInlet:
    def __init__(self, stream, max_chunklen: int = 64):
        self._stream = stream

    def info(self):
        return self._stream

    def pull_sample(self, timeout: float = 0.1):
        return [0.1, 0.2, 0.3, 0.4], 1.234


def test_step7_resolve_lsl_inlet_prefers_env_over_stale_config(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_module("7_live_infer_and_actuate.py", "live_lsl_resolution_env_test")
    streams = [
        _FakeStream(
            name="Muse2-EEG",
            stream_type="EEG",
            source_id="stale-id",
            uid="uid-stale",
            labels=["TP9", "AF7", "AF8", "TP10"],
        ),
        _FakeStream(
            name="Muse2-EEG",
            stream_type="EEG",
            source_id="fresh-id",
            uid="uid-fresh",
            labels=["TP9", "AF7", "AF8", "TP10"],
        ),
    ]
    monkeypatch.setattr(mod, "LSL_AVAILABLE", True)
    monkeypatch.setattr(mod, "resolve_streams", lambda wait_time=0.1: streams)
    monkeypatch.setattr(mod, "resolve_byprop", None)
    monkeypatch.setattr(mod, "StreamInlet", _FakeInlet)

    resolved = mod._resolve_lsl_inlet(
        "Muse2-EEG",
        "EEG",
        timeout_s=0.1,
        cli_source_id=None,
        env_source_id="fresh-id",
        config_source_id="stale-id",
    )

    assert resolved.resolution["source_id_source"] == "env"
    assert resolved.resolution["selected_source_id"] == "fresh-id"
    assert resolved.resolution["selection_matched_by_source_id"] is True


def test_step7_resolve_lsl_inlet_refuses_ambiguous_type_only_recovery(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_module("7_live_infer_and_actuate.py", "live_lsl_resolution_ambiguous_test")
    streams = [
        _FakeStream(
            name="Muse-A",
            stream_type="EEG",
            source_id="fresh-a",
            uid="uid-a",
            labels=["TP9", "AF7", "AF8", "TP10"],
        ),
        _FakeStream(
            name="Muse-B",
            stream_type="EEG",
            source_id="fresh-b",
            uid="uid-b",
            labels=["TP9", "AF7", "AF8", "TP10"],
        ),
    ]
    monkeypatch.setattr(mod, "LSL_AVAILABLE", True)
    monkeypatch.setattr(mod, "resolve_streams", lambda wait_time=0.1: streams)
    monkeypatch.setattr(mod, "resolve_byprop", None)
    monkeypatch.setattr(mod, "StreamInlet", _FakeInlet)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="ambiguous recovery"):
        mod._resolve_lsl_inlet(
            "Muse2-EEG",
            "EEG",
            timeout_s=0.1,
            cli_source_id=None,
            env_source_id="stale-id",
            config_source_id=None,
        )


def test_step7_resolve_lsl_inlet_refuses_stale_single_candidate_recovery(
    monkeypatch: pytest.MonkeyPatch,
):
    mod = _load_module("7_live_infer_and_actuate.py", "live_lsl_resolution_stale_single_test")
    streams = [
        _FakeStream(
            name="Muse-Only",
            stream_type="EEG",
            source_id="fresh-id",
            uid="uid-fresh",
            labels=["TP9", "AF7", "AF8", "TP10"],
        ),
    ]
    monkeypatch.setattr(mod, "LSL_AVAILABLE", True)
    monkeypatch.setattr(mod, "resolve_streams", lambda wait_time=0.1: streams)
    monkeypatch.setattr(mod, "resolve_byprop", None)
    monkeypatch.setattr(mod, "StreamInlet", _FakeInlet)
    monkeypatch.setattr(mod.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError, match="refusing single-candidate recovery"):
        mod._resolve_lsl_inlet(
            "Muse2-EEG",
            "EEG",
            timeout_s=0.1,
            cli_source_id=None,
            env_source_id="stale-id",
            config_source_id=None,
        )


def test_resolve_live_launch_plan_rejects_nonempty_output_dir(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_launch_plan_nonempty_out")

    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessA"
    models_run = session_dir / "processed" / "models" / "run_001"
    models_run.mkdir(parents=True)
    out_dir = session_dir / "processed" / "live_infer"
    out_dir.mkdir(parents=True)
    (out_dir / "stale.txt").write_text("stale")
    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"project_name": "Demo", "subject_id": "S01"}))

    with pytest.raises(RuntimeError, match="Choose a fresh --out-dir"):
        mod.resolve_live_launch_plan(
            config_path=config_path,
            config_payload={"project_name": "Demo", "subject_id": "S01"},
            config_settings={"session_dir": str(session_dir)},
            session_dir_override=str(session_dir),
            project_name_override=None,
            subject_id_override=None,
            model_path_override=None,
            scaler_path_override=None,
            out_dir_override=None,
            allow_outside_base=False,
        )


def test_resolve_live_launch_plan_allows_reserved_preflight_files_only(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_launch_plan_reserved_preflight_out")

    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessReserved"
    run_dir = session_dir / "processed" / "models" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "finger_action_model.pt").write_text("model")
    (run_dir / "scaler.npz").write_text("scaler")
    out_dir = session_dir / "processed" / "live_infer"
    out_dir.mkdir(parents=True)
    (out_dir / "step7_launch_config.json").write_text("{}")
    (out_dir / "live_preflight_report.json").write_text("{}")
    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"project_name": "Demo", "subject_id": "S01"}))

    plan = mod.resolve_live_launch_plan(
        config_path=config_path,
        config_payload={"project_name": "Demo", "subject_id": "S01"},
        config_settings={"session_dir": str(session_dir)},
        session_dir_override=str(session_dir),
        project_name_override=None,
        subject_id_override=None,
        model_path_override=None,
        scaler_path_override=None,
        out_dir_override=None,
        allow_outside_base=False,
    )

    assert plan.out_dir == out_dir.resolve()


def test_resolve_live_launch_plan_requires_session_run_or_explicit_artifacts(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_launch_plan_missing_run")

    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessB"
    session_dir.mkdir(parents=True)
    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"project_name": "Demo", "subject_id": "S01"}))

    with pytest.raises(RuntimeError, match="Selected session has no model run directory"):
        mod.resolve_live_launch_plan(
            config_path=config_path,
            config_payload={"project_name": "Demo", "subject_id": "S01"},
            config_settings={"session_dir": str(session_dir)},
            session_dir_override=str(session_dir),
            project_name_override=None,
            subject_id_override=None,
            model_path_override=None,
            scaler_path_override=None,
            out_dir_override=None,
            allow_outside_base=False,
        )


def test_resolve_live_launch_plan_does_not_borrow_run_from_other_session(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_launch_plan_no_cross_session_drift")

    selected_session = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessPinned"
    selected_session.mkdir(parents=True)
    other_run = (
        tmp_path
        / "Projects"
        / "Demo"
        / "subjects"
        / "S01"
        / "sessions"
        / "sessOther"
        / "processed"
        / "models"
        / "run_001"
    )
    other_run.mkdir(parents=True)
    (other_run / "finger_action_model.pt").write_text("model")
    (other_run / "scaler.npz").write_text("scaler")
    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"project_name": "Demo", "subject_id": "S01"}))

    with pytest.raises(RuntimeError, match="Selected session has no model run directory"):
        mod.resolve_live_launch_plan(
            config_path=config_path,
            config_payload={"project_name": "Demo", "subject_id": "S01"},
            config_settings={"session_dir": str(selected_session)},
            session_dir_override=str(selected_session),
            project_name_override=None,
            subject_id_override=None,
            model_path_override=None,
            scaler_path_override=None,
            out_dir_override=None,
            allow_outside_base=False,
        )


def test_resolve_live_launch_plan_accepts_explicit_artifacts_without_session_run(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_launch_plan_explicit_artifacts")

    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessC"
    session_dir.mkdir(parents=True)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    model_path = artifact_dir / "finger_action_model.pt"
    scaler_path = artifact_dir / "custom_scaler.npz"
    model_path.write_text("model")
    scaler_path.write_text("scaler")
    out_dir = session_dir / "processed" / "live_infer_explicit"
    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"project_name": "Demo", "subject_id": "S01"}))

    plan = mod.resolve_live_launch_plan(
        config_path=config_path,
        config_payload={"project_name": "Demo", "subject_id": "S01"},
        config_settings={
            "session_dir": str(session_dir),
            "model_path": str(model_path),
            "scaler_path": str(scaler_path),
            "out_dir": str(out_dir),
        },
        session_dir_override=str(session_dir),
        project_name_override=None,
        subject_id_override=None,
        model_path_override=str(model_path),
        scaler_path_override=str(scaler_path),
        out_dir_override=str(out_dir),
        allow_outside_base=False,
    )

    assert plan.selected_session_dir == session_dir.resolve()
    assert plan.chosen_run_dir is None
    assert plan.model_path == model_path.resolve()
    assert plan.scaler_path == scaler_path.resolve()
    assert plan.temperature_path == (artifact_dir / "temperature_scaling.json").resolve()
    assert plan.out_dir == out_dir.resolve()
    assert set(plan.explicit_overrides) == {"model_path", "scaler_path", "out_dir"}


def test_resolve_live_launch_plan_accepts_repo_root_relative_artifact_paths(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_launch_plan_repo_relative")

    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessD"
    session_dir.mkdir(parents=True)
    artifact_dir = (
        tmp_path
        / "Projects"
        / "Demo"
        / "subjects"
        / "S01"
        / "sessions"
        / "trained"
        / "processed"
        / "models"
        / "run_001"
    )
    artifact_dir.mkdir(parents=True)
    model_path = artifact_dir / "finger_action_model.pt"
    scaler_path = artifact_dir / "scaler.npz"
    model_path.write_text("model")
    scaler_path.write_text("scaler")
    out_dir = session_dir / "processed" / "live_infer_repo_relative"
    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"project_name": "Demo", "subject_id": "S01"}))

    plan = mod.resolve_live_launch_plan(
        config_path=config_path,
        config_payload={"project_name": "Demo", "subject_id": "S01"},
        config_settings={
            "session_dir": str(session_dir),
            "model_path": "Projects/Demo/subjects/S01/sessions/trained/processed/models/run_001/finger_action_model.pt",
            "scaler_path": "Projects/Demo/subjects/S01/sessions/trained/processed/models/run_001/scaler.npz",
            "out_dir": str(out_dir),
        },
        session_dir_override=str(session_dir),
        project_name_override=None,
        subject_id_override=None,
        model_path_override=None,
        scaler_path_override=None,
        out_dir_override=str(out_dir),
        allow_outside_base=False,
    )

    assert plan.model_path == model_path.resolve()
    assert plan.scaler_path == scaler_path.resolve()


def test_resolve_live_launch_plan_accepts_repo_root_relative_out_dir_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    mod = _load_module("7_live_infer_and_actuate.py", "live_launch_plan_repo_relative_outdir")

    monkeypatch.chdir(tmp_path)
    session_dir = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "sessions" / "sessE"
    run_dir = session_dir / "processed" / "models" / "run_001"
    run_dir.mkdir(parents=True)
    (run_dir / "finger_action_model.pt").write_text("model")
    (run_dir / "scaler.npz").write_text("scaler")
    config_path = tmp_path / "Projects" / "Demo" / "subjects" / "S01" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps({"project_name": "Demo", "subject_id": "S01"}))
    out_dir_text = "Projects/Demo/subjects/S01/sessions/sessE/processed/live_infer_repo_relative_new"

    plan = mod.resolve_live_launch_plan(
        config_path=config_path,
        config_payload={"project_name": "Demo", "subject_id": "S01"},
        config_settings={
            "session_dir": str(session_dir),
            "out_dir": out_dir_text,
        },
        session_dir_override=str(session_dir),
        project_name_override=None,
        subject_id_override=None,
        model_path_override=None,
        scaler_path_override=None,
        out_dir_override=out_dir_text,
        allow_outside_base=False,
    )

    assert plan.out_dir == (tmp_path / out_dir_text).resolve()


def test_collect_required_output_status_flags_missing_required_files(tmp_path: Path):
    mod = _load_module("7_live_infer_and_actuate.py", "live_required_output_status")

    out_dir = tmp_path / "live_infer"
    out_dir.mkdir()
    live_log = out_dir / "live_infer.log"
    pred_log = out_dir / "predictions.jsonl"
    window_audit = out_dir / "window_audit.jsonl"
    segment_breaks = out_dir / "segment_breaks.jsonl"
    parity_dir = out_dir / "parity_capture"
    parity_dir.mkdir()
    live_log.write_text("log")
    pred_log.write_text("{}\n")
    window_audit.write_text("{}\n")
    segment_breaks.write_text("{}\n")
    (parity_dir / "capture_manifest.json").write_text("{}\n")

    output_hashes, errors = mod._collect_required_output_status(
        no_file_io=False,
        out_dir=out_dir,
        pred_log_path=pred_log,
        window_audit_path=window_audit,
        segment_break_path=segment_breaks,
        summary_path=out_dir / "live_prediction_summary.json",
        distribution_report_path=out_dir / "live_input_distribution_report.json",
        parity_report_path=out_dir / "parity_report.json",
        parity_capture=SimpleNamespace(
            manifest_path=parity_dir / "capture_manifest.json",
            records_path=parity_dir / "captured_windows.json",
        ),
        parity_capture_required=True,
        cleanup_errors=["prediction_log_close_error: broken pipe"],
        summary_write_error="summary build failed",
        distribution_report_write_error="distribution report failed",
        parity_report_write_error="parity replay failed",
    )

    assert output_hashes["live_log_sha256"] is not None
    assert any("summary_write_error" in err for err in errors)
    assert any("summary_missing_or_unreadable" in err for err in errors)
    assert any("distribution_report_write_error" in err for err in errors)
    assert any("distribution_report_missing_or_unreadable" in err for err in errors)
    assert any("parity_report_write_error" in err for err in errors)
    assert any("parity_report_missing_or_unreadable" in err for err in errors)
    assert any("parity_capture_records_missing_or_unreadable" in err for err in errors)
    assert any("prediction_log_close_error" in err for err in errors)
