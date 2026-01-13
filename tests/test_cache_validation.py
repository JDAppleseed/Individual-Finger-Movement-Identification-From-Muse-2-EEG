import pytest

np = pytest.importorskip("numpy")

from utils.eval_utils import validate_cached_predictions


def test_cached_predictions_reject_duplicate_indices():
    action_probs = np.zeros((3, 2), dtype=np.float32)
    finger_probs = np.zeros((3, 2), dtype=np.float32)
    y_action = np.array([0, 1, 0])
    y_finger = np.array([0, 1, 0])
    test_idx = np.array([0, 0, 1])

    assert not validate_cached_predictions(
        action_probs=action_probs,
        finger_probs=finger_probs,
        y_action_test=y_action,
        y_finger_test=y_finger,
        test_idx=test_idx,
        n_actions=2,
        n_fingers=2,
        n_samples=5,
    )


def test_cached_predictions_accept_unique_indices():
    action_probs = np.zeros((2, 2), dtype=np.float32)
    finger_probs = np.zeros((2, 2), dtype=np.float32)
    y_action = np.array([0, 1])
    y_finger = np.array([0, 1])
    test_idx = np.array([1, 3])

    assert validate_cached_predictions(
        action_probs=action_probs,
        finger_probs=finger_probs,
        y_action_test=y_action,
        y_finger_test=y_finger,
        test_idx=test_idx,
        n_actions=2,
        n_fingers=2,
        n_samples=10,
    )
