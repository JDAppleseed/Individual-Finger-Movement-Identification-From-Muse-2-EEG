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
