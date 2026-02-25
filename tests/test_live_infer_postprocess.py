import importlib.util
from pathlib import Path

import numpy as np

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
