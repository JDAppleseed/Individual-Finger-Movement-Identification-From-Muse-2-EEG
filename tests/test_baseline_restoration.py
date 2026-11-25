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


@pytest.mark.parametrize('nested', [True, False])
def test_live_rejects_archived_ensemble_config(tmp_path, nested):
    live = module('7_live_infer_and_actuate.py')
    config = {'ensemble_manifest': 'old-candidate.json'}
    path = tmp_path / 'infer.json'
    path.write_text(json.dumps({'settings': config} if nested else config))
    with pytest.raises(ValueError, match='CNN-LSTM'):
        live._load_config_file(path)
