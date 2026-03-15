from pathlib import Path

from utils.runtime_utils import (
    TemperatureScalingState,
    load_temperature_scaling,
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
