import numpy as np
import pytest

from utils.label_schema import (
    ACTION_OPEN,
    ACTION_REST,
    FINGER_INDEX,
    FINGER_NONE,
    decode_prediction_pair,
    enforce_prediction_pair,
    enforce_prediction_pairs,
)


def test_enforce_prediction_pair_forces_rest_to_none():
    action_id, finger_id = enforce_prediction_pair(ACTION_REST, FINGER_INDEX)
    assert action_id == ACTION_REST
    assert finger_id == FINGER_NONE


def test_decode_prediction_pair_gates_rest_but_keeps_non_rest_none():
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
    assert open_finger == FINGER_NONE


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
