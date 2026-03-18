import numpy as np
import pytest

from utils.label_schema import (
    ACTION_OPEN,
    ACTION_REST,
    FINGER_INDEX,
    FINGER_NONE,
    FINGER_THUMB,
    decode_active_finger_predictions,
    decode_finger_prediction,
    decode_finger_predictions_for_actions,
    decode_prediction_pair,
    enforce_prediction_pair,
    enforce_prediction_pairs,
    is_valid_action_finger,
    prediction_pair_diagnostics,
)


def test_enforce_prediction_pair_forces_rest_to_none():
    action_id, finger_id = enforce_prediction_pair(ACTION_REST, FINGER_INDEX)
    assert action_id == ACTION_REST
    assert finger_id == FINGER_NONE


def test_is_valid_action_finger_rejects_non_rest_none():
    assert is_valid_action_finger(ACTION_REST, FINGER_NONE)
    assert is_valid_action_finger(ACTION_OPEN, FINGER_INDEX)
    assert not is_valid_action_finger(ACTION_OPEN, FINGER_NONE)


def test_decode_prediction_pair_gates_rest_and_commits_active_finger():
    rest_action, rest_finger = decode_prediction_pair(
        np.array([0.9, 0.05, 0.05], dtype=float),
        np.array([0.01, 0.80, 0.02, 0.02, 0.10, 0.05], dtype=float),
    )
    assert rest_action == ACTION_REST
    assert rest_finger == FINGER_NONE

    open_action, open_finger = decode_prediction_pair(
        np.array([0.05, 0.90, 0.05], dtype=float),
        np.array([0.85, 0.05, 0.03, 0.02, 0.03, 0.02], dtype=float),
    )
    assert open_action == ACTION_OPEN
    assert open_finger == FINGER_THUMB


def test_decode_prediction_pair_maps_active_only_finger_head_to_true_label_ids():
    open_action, open_finger = decode_prediction_pair(
        np.array([0.05, 0.90, 0.05], dtype=float),
        np.array([0.85, 0.05, 0.03, 0.02, 0.05], dtype=float),
    )
    assert open_action == ACTION_OPEN
    assert open_finger == FINGER_THUMB


def test_decode_finger_prediction_maps_active_only_output():
    finger_id = decode_finger_prediction(
        np.array([0.01, 0.80, 0.05, 0.04, 0.10], dtype=float)
    )
    assert finger_id == FINGER_INDEX


def test_decode_active_finger_predictions_legacy_head_ignores_none_slot():
    decoded = decode_active_finger_predictions(
        np.array(
            [
                [0.90, 0.04, 0.03, 0.01, 0.01, 0.01],
                [0.05, 0.10, 0.70, 0.05, 0.05, 0.05],
            ],
            dtype=float,
        )
    )
    assert decoded.tolist() == [FINGER_THUMB, FINGER_INDEX]


def test_decode_finger_predictions_for_actions_respects_action_conditioning():
    decoded = decode_finger_predictions_for_actions(
        np.array([ACTION_REST, ACTION_OPEN], dtype=int),
        np.array(
            [
                [0.05, 0.80, 0.05, 0.04, 0.03, 0.03],
                [0.90, 0.04, 0.03, 0.01, 0.01, 0.01],
            ],
            dtype=float,
        ),
    )
    assert decoded.tolist() == [FINGER_NONE, FINGER_THUMB]


def test_prediction_pair_diagnostics_reports_raw_and_committed_counts():
    diagnostics = prediction_pair_diagnostics(
        np.array([ACTION_REST, ACTION_OPEN, ACTION_OPEN], dtype=int),
        np.array([FINGER_INDEX, FINGER_NONE, FINGER_INDEX], dtype=int),
        committed_finger_ids=np.array([FINGER_NONE, FINGER_THUMB, FINGER_INDEX], dtype=int),
    )

    assert diagnostics["raw_rest_non_none_count"] == 1
    assert diagnostics["raw_non_rest_none_count"] == 1
    assert diagnostics["committed_non_rest_none_count"] == 0
    assert diagnostics["deployment_pair_invariant_ok"] is True


def test_enforce_prediction_pairs_vectorized():
    action_ids, finger_ids = enforce_prediction_pairs(
        np.array([ACTION_REST, ACTION_OPEN, ACTION_REST], dtype=int),
        np.array([FINGER_INDEX, FINGER_NONE, FINGER_INDEX], dtype=int),
    )
    assert action_ids.tolist() == [ACTION_REST, ACTION_OPEN, ACTION_REST]
    assert finger_ids.tolist() == [FINGER_NONE, FINGER_NONE, FINGER_NONE]


def test_enforce_prediction_pairs_shape_mismatch_raises():
    with pytest.raises(ValueError):
        enforce_prediction_pairs(np.array([0, 1]), np.array([0]))
