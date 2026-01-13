from __future__ import annotations

from typing import Mapping, Optional

import numpy as np


def validate_cached_predictions(
    action_probs: np.ndarray,
    finger_probs: np.ndarray,
    y_action_test: np.ndarray,
    y_finger_test: np.ndarray,
    test_idx: np.ndarray,
    n_actions: int,
    n_fingers: int,
    n_samples: Optional[int] = None,
) -> bool:
    if action_probs.shape != (len(test_idx), n_actions):
        return False
    if finger_probs.shape != (len(test_idx), n_fingers):
        return False
    if y_action_test.shape != (len(test_idx),):
        return False
    if y_finger_test.shape != (len(test_idx),):
        return False
    if len(test_idx) != len(np.unique(test_idx)):
        return False
    if n_samples is not None:
        if len(test_idx) and (test_idx.min() < 0 or test_idx.max() >= n_samples):
            return False
    if len(y_action_test) > 0:
        try:
            if y_action_test.min() < 0 or y_action_test.max() >= n_actions:
                return False
        except Exception:
            return False
    if len(y_finger_test) > 0:
        try:
            if y_finger_test.min() < 0 or y_finger_test.max() >= n_fingers:
                return False
        except Exception:
            return False
    return True


def resolve_cached_test_indices(payload: Mapping[str, object]) -> Optional[np.ndarray]:
    if "test_indices_local" in payload:
        return np.asarray(payload["test_indices_local"]).astype(np.int64)
    if "test_indices" in payload:
        return np.asarray(payload["test_indices"]).astype(np.int64)
    return None
