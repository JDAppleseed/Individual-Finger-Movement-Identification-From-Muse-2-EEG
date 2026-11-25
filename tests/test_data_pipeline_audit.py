import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from utils.saved_folds import load_saved_folds
from utils.sequence_data import _as_1d_int64


def load_module(name):
    path = Path(__file__).resolve().parents[1] / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('labels', [np.array([[1], [2]]), np.array([1.5, 2]), np.array([np.nan]), np.array([-1])])
def test_loader_rejects_malformed_labels(labels):
    with pytest.raises(ValueError):
        _as_1d_int64(labels, 'labels')


def write_folds(tmp_path, **extra):
    dataset = tmp_path / 'dataset.npz'
    np.savez(dataset, X=np.zeros((10, 64, 4)))
    payload = dict(train=np.array([1, 3, 5]), fit=np.array([1, 3]),
                   calibration=np.array([5]), test=np.array([7, 9]),
                   index_space=np.array('source_npz'),
                   dataset_sha256=np.array(hashlib.sha256(dataset.read_bytes()).hexdigest()))
    payload.update(extra)
    np.savez(tmp_path / 'split_indices.npz', **payload)
    return dataset


def test_saved_source_indices_map_after_filtering_and_reordering(tmp_path):
    path = write_folds(tmp_path)
    folds = load_saved_folds(tmp_path, path, np.array([9, 3, 7, 5]), 10)
    assert folds['test'].tolist() == [2, 0]
    assert folds['fit'].tolist() == [1]
    assert folds['calibration'].tolist() == [3]


@pytest.mark.parametrize('bad', [np.array([-1]), np.array([10]), np.array([7, 7]), np.array([1.5])])
def test_saved_invalid_test_indices_rejected(tmp_path, bad):
    path = write_folds(tmp_path, test=bad)
    with pytest.raises(ValueError):
        load_saved_folds(tmp_path, path, np.arange(10), 10)


def test_saved_overlap_rejected(tmp_path):
    path = write_folds(tmp_path, fit=np.array([1, 7]))
    with pytest.raises(ValueError, match='overlap'):
        load_saved_folds(tmp_path, path, np.arange(10), 10)


def test_saved_dataset_change_rejected(tmp_path):
    path = write_folds(tmp_path)
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='different dataset'):
        load_saved_folds(tmp_path, path, np.arange(10), 10)


def test_missing_original_fold_never_resplits(tmp_path):
    path = tmp_path / 'data'; path.write_bytes(b'data')
    with pytest.raises(ValueError, match='refusing to re-split'):
        load_saved_folds(tmp_path, path, np.arange(10), 10)


def test_temperature_fits_on_cpu_even_with_accelerator_request():
    import torch
    train = load_module('2_train_model.py')
    # CPU fitting must not allocate tensors on an unavailable accelerator.
    temp, metrics = train._fit_temperature(np.array([[5., 0.], [5., 0.]]),
        np.array([0, 1]), device=torch.device('cuda'), max_iter=5)
    assert temp > 1 and metrics['nll_after'] < metrics['nll_before']
    temp, metrics = train._fit_binary_temperature(np.array([5., 5.]),
        np.array([0, 1]), device=torch.device('cuda'), max_iter=5)
    assert temp > 1 and metrics['nll_after'] < metrics['nll_before']


def test_smoothing_receives_applicability_gate():
    from utils.postprocess import PostprocessSettings
    evaluate = load_module('3_evaluate_model.py')
    settings = PostprocessSettings(threshold_applicability=.5, threshold_action=0,
                                  threshold_finger=0, hysteresis_frames=1, finger_mode='raw')
    diagnostics = {}
    a, f = evaluate.apply_postprocess_sequence(np.array([[0., 1., 0.]]),
        np.array([[1., 0., 0., 0., 0.]]), np.array([0]), settings,
        applicability_probs=np.array([0.]), diagnostics=diagnostics)
    assert f.tolist() == [1]
    assert diagnostics["applicability_gate_passed"] == [False]


def test_appended_training_rest_is_purged_if_signal_overlaps(monkeypatch):
    import utils.splitting as splitting
    plan = dict(train_idx=np.array([0, 1]), test_idx=np.array([2]),
                train_locked_idx=np.array([1]), core_train_idx=np.array([0, 1]),
                aux_train_idx=np.array([], dtype=int), main_split_idx=np.arange(3),
                main_train_local=np.array([0]))
    monkeypatch.setattr(splitting, '_compose_split_indices_unchecked', lambda *a, **kw: plan)
    result = splitting.compose_split_indices(np.array([1, 0, 1]), np.array([1, 0, 1]),
        meta=dict(window_start=np.array([0., 1., 1.1]), window_end=np.array([.25, 1.25, 1.35]),
                  session_id=np.full(3, 'same')))
    assert result['train_idx'].tolist() == [0]
    assert result['train_locked_idx'].size == 0


def test_final_calibration_guard_detects_locked_rest_overlap():
    from utils.saved_folds import assert_signal_disjoint
    meta = dict(window_start=np.array([0., .1, 1.]), window_end=np.array([.25, .35, 1.25]),
                session_id=np.full(3, 'same'))
    with pytest.raises(ValueError, match='fit/calibration share raw signal'):
        assert_signal_disjoint(meta, fit=np.array([0]), calibration=np.array([1]), test=np.array([2]))
    meta['session_id'] = np.array(['one', 'two', 'one'])
    assert_signal_disjoint(meta, fit=np.array([0]), calibration=np.array([1]), test=np.array([2]))


def test_rest_event_ids_are_scoped_to_recording_session():
    evaluate = load_module('3_evaluate_model.py')
    breakdown, flags = evaluate._build_rest_event_breakdown(
        indices=np.arange(4), y_action_true=np.zeros(4, dtype=int),
        action_probs=np.array([[1., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 1., 0.]]),
        action_preds=np.array([0, 0, 1, 1]), finger_preds=np.array([0, 0, 1, 1]),
        meta=dict(event_id=np.ones(4, dtype=int), session_id=np.array(['A', 'A', 'B', 'B'])))
    assert len(breakdown) == 2
    assert {r['session_id']: r['rest_tpr'] for r in breakdown} == {'A': 1., 'B': 0.}
    assert all(flag['recommended_action'] == 'independent_signal_and_label_review' for flag in flags)
