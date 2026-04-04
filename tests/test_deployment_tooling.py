import importlib.util
import json
import sys
from pathlib import Path

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

    summary_path = tmp_path / "live_prediction_summary.json"
    mod._build_live_prediction_summary(
        pred_log_path=pred_log,
        summary_path=summary_path,
        raw_dir=raw_dir,
        dropped_windows=3,
        dropped_nonfinite_samples=4,
        dropped_nonfinite_windows=1,
        segment_break_count=0,
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
    assert summary["raw_channel_stats"]["flagged_nonfinite_rows"] == 1
