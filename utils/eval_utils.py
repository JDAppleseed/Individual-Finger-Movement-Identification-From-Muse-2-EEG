from __future__ import annotations

from typing import Mapping, Optional, Tuple, List, Any, Dict

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
    if "test_indices" in payload:
        return np.asarray(payload["test_indices"]).astype(np.int64)
    if "test_indices_local" in payload:
        return np.asarray(payload["test_indices_local"]).astype(np.int64)
    return None


def validate_cached_predictions_with_dataset_info(
    *,
    action_probs: np.ndarray,
    finger_probs: np.ndarray,
    y_action_test: np.ndarray,
    y_finger_test: np.ndarray,
    test_idx: np.ndarray,
    n_actions: int,
    n_fingers: int,
    n_samples_current: int,
    dataset_info_cache: Optional[Dict[str, Any]],
    dataset_info_current: Dict[str, Any],
    y_action_current: np.ndarray,
    y_finger_current: np.ndarray,
    spotcheck_k: int = 10,
    rng_seed: int = 0,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    cache_ok = validate_cached_predictions(
        action_probs=action_probs,
        finger_probs=finger_probs,
        y_action_test=y_action_test,
        y_finger_test=y_finger_test,
        test_idx=test_idx,
        n_actions=n_actions,
        n_fingers=n_fingers,
        n_samples=n_samples_current,
    )
    if not cache_ok:
        reasons.append("cache_shape_invalid")
        return False, reasons

    if dataset_info_cache is None:
        return False, ["legacy_cache_missing_dataset_info"]

    cache_filters = dataset_info_cache.get("filters", {}) or {}
    current_filters = dataset_info_current.get("filters", {}) or {}

    cache_subject_id = (
        ""
        if cache_filters.get("subject_id") is None
        else str(cache_filters.get("subject_id"))
    )
    current_subject_id = (
        ""
        if current_filters.get("subject_id") is None
        else str(current_filters.get("subject_id"))
    )
    if cache_subject_id != current_subject_id:
        reasons.append("filter_subject_id_mismatch")

    cache_max_samples = cache_filters.get("max_samples")
    current_max_samples = current_filters.get("max_samples")
    if cache_max_samples != current_max_samples:
        reasons.append("filter_max_samples_mismatch")

    if dataset_info_cache.get("experiment_hash") != dataset_info_current.get(
        "experiment_hash"
    ):
        reasons.append("experiment_hash_mismatch")

    if dataset_info_cache.get("n_samples") != dataset_info_current.get("n_samples"):
        reasons.append("n_samples_mismatch")

    cache_sha = dataset_info_cache.get("npz_sha256")
    current_sha = dataset_info_current.get("npz_sha256")
    if cache_sha is not None or current_sha is not None:
        if cache_sha is not None and current_sha is None:
            reasons.append("npz_sha256_unavailable_current")
        elif cache_sha is None and current_sha is not None:
            reasons.append("npz_identity_insufficient")
        elif cache_sha != current_sha:
            reasons.append("npz_sha256_mismatch")
    else:
        cache_size = dataset_info_cache.get("npz_size_bytes")
        current_size = dataset_info_current.get("npz_size_bytes")
        if cache_size is not None and current_size is not None:
            if int(cache_size) != int(current_size):
                reasons.append("npz_size_mismatch")
        else:
            reasons.append("npz_identity_insufficient")

    if reasons:
        return False, reasons

    if spotcheck_k > 0 and len(test_idx) > 0:
        rng = np.random.default_rng(rng_seed)
        sample_count = min(int(spotcheck_k), len(test_idx))
        positions = rng.choice(len(test_idx), size=sample_count, replace=False)
        for pos in positions:
            idx = int(test_idx[pos])
            if (
                y_action_test[pos] != y_action_current[idx]
                or y_finger_test[pos] != y_finger_current[idx]
            ):
                reasons.append("spotcheck_label_mismatch")
                break

    return len(reasons) == 0, reasons
