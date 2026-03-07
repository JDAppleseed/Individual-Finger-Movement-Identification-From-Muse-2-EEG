import importlib.util
from pathlib import Path

import numpy as np
import torch

from utils.inference import InferenceConfig, InferenceEngine
from utils.postprocess import PostprocessSettings, PostprocessState


def _load_live_module():
    module_path = Path(__file__).resolve().parents[1] / "7_live_infer_and_actuate.py"
    spec = importlib.util.spec_from_file_location("live_infer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_postprocess_decision_deterministic():
    mod = _load_live_module()
    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.5,
        threshold_finger=0.5,
        adjacency_enabled=False,
    )
    state = PostprocessState()
    action_probs = np.array([0.1, 0.8, 0.1], dtype=float)
    finger_probs = np.array([0.05, 0.05, 0.85, 0.02, 0.02, 0.01], dtype=float)

    out = mod._postprocess_decision(
        action_probs,
        finger_probs,
        enabled=True,
        settings=settings,
        state=state,
    )

    assert out["committed_action_id"] == 1
    assert out["committed_finger_id"] == 2


class _DummyMCModel(torch.nn.Module):
    def forward(self, x):
        finger_logits = torch.tensor(
            [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        action_logits = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        return finger_logits, action_logits

    def mc_forward(self, x, passes=20):
        return {
            "finger_mean": torch.tensor(
                [[0.05, 0.05, 0.80, 0.04, 0.03, 0.03]], dtype=torch.float32
            ),
            "action_mean": torch.tensor([[0.10, 0.75, 0.15]], dtype=torch.float32),
            "finger_std": torch.tensor(
                [[0.01, 0.01, 0.05, 0.01, 0.01, 0.01]], dtype=torch.float32
            ),
            "action_std": torch.tensor([[0.02, 0.10, 0.03]], dtype=torch.float32),
        }


def test_predict_window_uses_inference_engine_backend():
    mod = _load_live_module()
    model = _DummyMCModel()
    engine = InferenceEngine(
        model=model,
        normalizer=None,
        device=torch.device("cpu"),
        action_names={},
        finger_names={},
        config=InferenceConfig(
            base_threshold=0.7,
            uncertainty_weight=0.5,
            stability_frames=2,
            mc_passes=5,
        ),
    )

    out = mod._predict_window(
        np.zeros((64, 4), dtype=np.float32),
        scaler=None,
        model=model,
        device=torch.device("cpu"),
        inference_engine=engine,
        emit_viz=False,
    )

    assert out["backend"] == "inference_engine"
    assert np.isclose(out["action_probs"][1], 0.75)
    assert np.isclose(out["finger_probs"][2], 0.80)
    assert np.isclose(out["action_uncertainty"], np.mean([0.02, 0.10, 0.03]))
    assert np.isclose(
        out["finger_uncertainty"], np.mean([0.01, 0.01, 0.05, 0.01, 0.01, 0.01])
    )
    assert out["adaptive_threshold"] > 0.7


def test_compute_actuation_speed_scalar_uses_uncertainty():
    mod = _load_live_module()
    mapper = mod._build_actuation_speed_mapper(
        type(
            "Args",
            (),
            {"modulate_actuation_speed": True, "actuation_speed_gamma": 1.0},
        )()
    )

    speed = mod._compute_actuation_speed_scalar(
        decision_prob=0.8,
        action_uncertainty=0.25,
        speed_mapper=mapper,
    )

    assert np.isclose(speed, 0.6)
