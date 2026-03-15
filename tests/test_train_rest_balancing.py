import importlib.util
from pathlib import Path

import numpy as np


def _load_train_module():
    module_path = Path(__file__).resolve().parents[1] / "2_train_model.py"
    spec = importlib.util.spec_from_file_location("step2_train_model", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_session_equalized_rest_weights_balance_rest_mass():
    mod = _load_train_module()
    y_action = np.array([0] * 8 + [0] * 2 + [1] * 6, dtype=np.int64)
    meta = {
        "session_id": np.array(
            ["rest_heavy"] * 8 + ["rest_light"] * 2 + ["rest_heavy"] * 6,
            dtype="U",
        )
    }

    weights, summary = mod._build_train_sample_weights(
        y_action,
        meta,
        balance_mode="session_equalized",
    )

    assert weights is not None
    assert summary["enabled"] is True
    rest_heavy_mass = float(np.sum(weights[(y_action == 0) & (meta["session_id"] == "rest_heavy")]))
    rest_light_mass = float(np.sum(weights[(y_action == 0) & (meta["session_id"] == "rest_light")]))
    assert np.isclose(rest_heavy_mass, rest_light_mass)
    assert np.allclose(weights[y_action != 0], 1.0)


def test_session_equalized_rest_weights_disable_without_multiple_rest_sessions():
    mod = _load_train_module()
    y_action = np.array([0, 0, 0, 1, 2], dtype=np.int64)
    meta = {"session_id": np.array(["only"] * len(y_action), dtype="U")}

    weights, summary = mod._build_train_sample_weights(
        y_action,
        meta,
        balance_mode="session_equalized",
    )

    assert weights is None
    assert summary["enabled"] is False
    assert summary["reason"] == "single_rest_session"
