import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest


def module(filename):
    path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_calibration_keeps_events_separate_and_purges_overlapping_signal():
    train = module('2_train_model.py')
    n = 120
    starts = np.arange(n) * .05
    groups = np.repeat(np.arange(12), 10)
    meta = dict(event_id=groups, session_id=np.full(n, 'one'),
                window_start=starts, window_end=starts + .25)
    fit, calibration = train._split_calibration_indices(
        np.arange(n), groups % 3, np.where(groups % 3 == 0, 0, 1), meta,
        calibration_size=.2, random_state=43, split_mode='group_trial')
    assert len(fit) and len(calibration)
    assert not np.intersect1d(groups[fit], groups[calibration]).size
    assert np.min(np.abs(starts[fit, None] - starts[calibration])) > .25


def test_calibration_refuses_random_window_fallback():
    train = module('2_train_model.py')
    with pytest.raises(ValueError, match='metadata'):
        train._split_calibration_indices(np.arange(30), np.ones(30), np.ones(30), {},
            calibration_size=.2, random_state=43, split_mode='group_trial')


def test_calibration_cannot_take_only_same_session_rest_event(monkeypatch):
    train = module('2_train_model.py')
    n = 30
    actions = np.repeat([0, 1, 2], 10)
    events = np.repeat([10, 20, 30], 10)
    starts = np.arange(n, dtype=float)
    meta = dict(event_id=events, session_id=np.full(n, 'movement'),
                window_start=starts, window_end=starts + .25)
    # Reproduce the failure: all available core REST assigned to calibration.
    monkeypatch.setattr(train, 'split_indices', lambda *a, **kw:
        (np.arange(10, 30), np.arange(10)))
    fit, calibration = train._split_calibration_indices(
        np.arange(n), actions, np.where(actions == 0, 0, 1), meta,
        calibration_size=.1, random_state=43, split_mode='group_trial')
    assert np.sum(actions[fit] == 0) == 10
    assert len(calibration) == 0  # Prefer fitting over an invalid calibration claim.
    assert np.array_equal(fit, np.arange(n))


def test_fit_coverage_survives_temporal_purging():
    from utils.calibration_split import preserve_fit_coverage
    fit, calibration = preserve_fit_coverage(
        calibration=np.array([1, 2, 4]), groups=np.array([0, 1, 1, 2, 3]),
        sessions=np.full(5, 'one'), actions=np.array([0, 1, 1, 2, 2]),
        starts=np.array([0., .1, .2, 3., 6.]), purge_seconds=.25)
    assert 0 in fit
    assert np.array_equal(calibration, [4])


def test_partial_class_calibration_uses_identity_temperature():
    import torch
    train = module('2_train_model.py')
    temperature, metrics = train._fit_temperature(
        np.ones((8, 3)), np.array([1, 2] * 4), device=torch.device('cpu'))
    assert temperature == 1.
    assert metrics['skipped_reason'] == 'missing_calibration_classes'


@pytest.mark.parametrize('nested', [True, False])
def test_live_rejects_archived_ensemble_config(tmp_path, nested):
    live = module('7_live_infer_and_actuate.py')
    config = {'ensemble_manifest': 'old-candidate.json'}
    path = tmp_path / 'infer.json'
    path.write_text(json.dumps({'settings': config} if nested else config))
    with pytest.raises(ValueError, match='CNN-LSTM'):
        live._load_config_file(path)
