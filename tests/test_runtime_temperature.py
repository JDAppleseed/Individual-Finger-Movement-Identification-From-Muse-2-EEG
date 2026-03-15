import numpy as np

from pathlib import Path

from utils.runtime_utils import (
    TemperatureScalingState,
    apply_channel_normalizer,
    load_temperature_scaling,
    load_normalizer,
    save_normalizer,
    save_temperature_scaling,
)


def test_save_and_load_temperature_scaling(tmp_path):
    path = Path(tmp_path) / "temperature_scaling.json"
    state = TemperatureScalingState(
        action_temperature=1.5,
        finger_temperature=2.0,
        fit_sample_count=123,
        fit_non_rest_count=100,
        source="fit_on_holdout",
        metrics={"action": {"nll_before": 1.0, "nll_after": 0.9}},
    )

    save_temperature_scaling(path, state)
    loaded = load_temperature_scaling(path)

    assert loaded is not None
    assert loaded.action_temperature == 1.5
    assert loaded.finger_temperature == 2.0
    assert loaded.fit_sample_count == 123
    assert loaded.fit_non_rest_count == 100
    assert loaded.source == "fit_on_holdout"
    assert loaded.metrics["action"]["nll_after"] == 0.9


def test_save_and_load_normalizer_with_preprocess(tmp_path):
    path = Path(tmp_path) / "scaler.npz"
    normalizer = {
        "type": "per_channel",
        "mean": np.array([0.0, 0.0], dtype=np.float32),
        "std": np.array([1.0, 1.0], dtype=np.float32),
        "channels": 2,
        "preprocess": {
            "per_window_center": True,
            "per_window_detrend": True,
        },
    }

    save_normalizer(path, normalizer)
    loaded = load_normalizer(path)

    assert loaded is not None
    assert loaded["preprocess"]["per_window_center"] is True
    assert loaded["preprocess"]["per_window_detrend"] is True

    window = np.array(
        [
            [10.0, -2.0],
            [12.0, 0.0],
            [14.0, 2.0],
            [16.0, 4.0],
        ],
        dtype=np.float32,
    )
    out = apply_channel_normalizer(window, loaded)
    assert out.shape == window.shape
    assert np.all(np.isfinite(out))
    assert np.allclose(out.mean(axis=0), [0.0, 0.0], atol=1e-5)
