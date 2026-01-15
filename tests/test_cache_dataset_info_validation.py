import pytest

np = pytest.importorskip("numpy")


def _base_payload():
    n_actions = 3
    n_fingers = 2
    test_idx = np.array([1, 3, 4], dtype=np.int64)
    y_action_current = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
    y_finger_current = np.array([0, 1, 0, 1, 1, 0], dtype=np.int64)
    y_action_test = y_action_current[test_idx].copy()
    y_finger_test = y_finger_current[test_idx].copy()
    action_probs = np.zeros((len(test_idx), n_actions), dtype=np.float32)
    finger_probs = np.zeros((len(test_idx), n_fingers), dtype=np.float32)

    dataset_info_current = {
        "npz_path": "/tmp/data.npz",
        "npz_sha256": "abc",
        "npz_size_bytes": 123,
        "experiment_hash": "exp",
        "n_samples": int(len(y_action_current)),
        "filters": {"subject_id": "", "max_samples": None},
        "created_utc": "2020-01-01T00:00:00Z",
    }
    dataset_info_cache = dict(dataset_info_current)
    dataset_info_cache["filters"] = dict(dataset_info_current["filters"])

    return {
        "action_probs": action_probs,
        "finger_probs": finger_probs,
        "y_action_test": y_action_test,
        "y_finger_test": y_finger_test,
        "test_idx": test_idx,
        "n_actions": n_actions,
        "n_fingers": n_fingers,
        "n_samples_current": len(y_action_current),
        "dataset_info_cache": dataset_info_cache,
        "dataset_info_current": dataset_info_current,
        "y_action_current": y_action_current,
        "y_finger_current": y_finger_current,
    }


def test_cache_accepts_matching_dataset_info():
    payload = _base_payload()
    from utils.eval_utils import validate_cached_predictions_with_dataset_info

    ok, reasons = validate_cached_predictions_with_dataset_info(
        **payload, spotcheck_k=10, rng_seed=0
    )
    assert ok is True
    assert reasons == []


def test_cache_rejects_subject_filter_mismatch():
    payload = _base_payload()
    payload["dataset_info_cache"]["filters"]["subject_id"] = "subject-x"
    from utils.eval_utils import validate_cached_predictions_with_dataset_info

    ok, reasons = validate_cached_predictions_with_dataset_info(
        **payload, spotcheck_k=10, rng_seed=0
    )
    assert ok is False
    assert "filter_subject_id_mismatch" in reasons


def test_cache_rejects_max_samples_mismatch():
    payload = _base_payload()
    payload["dataset_info_cache"]["filters"]["max_samples"] = 50
    from utils.eval_utils import validate_cached_predictions_with_dataset_info

    ok, reasons = validate_cached_predictions_with_dataset_info(
        **payload, spotcheck_k=10, rng_seed=0
    )
    assert ok is False
    assert "filter_max_samples_mismatch" in reasons


def test_cache_rejects_experiment_hash_mismatch():
    payload = _base_payload()
    payload["dataset_info_cache"]["experiment_hash"] = "different"
    from utils.eval_utils import validate_cached_predictions_with_dataset_info

    ok, reasons = validate_cached_predictions_with_dataset_info(
        **payload, spotcheck_k=10, rng_seed=0
    )
    assert ok is False
    assert "experiment_hash_mismatch" in reasons


def test_cache_rejects_legacy_missing_dataset_info():
    payload = _base_payload()
    payload["dataset_info_cache"] = None
    from utils.eval_utils import validate_cached_predictions_with_dataset_info

    ok, reasons = validate_cached_predictions_with_dataset_info(
        **payload, spotcheck_k=10, rng_seed=0
    )
    assert ok is False
    assert reasons == ["legacy_cache_missing_dataset_info"]


def test_cache_rejects_spotcheck_mismatch():
    payload = _base_payload()
    payload["y_action_test"][1] = 2
    from utils.eval_utils import validate_cached_predictions_with_dataset_info

    ok, reasons = validate_cached_predictions_with_dataset_info(
        **payload, spotcheck_k=10, rng_seed=0
    )
    assert ok is False
    assert "spotcheck_label_mismatch" in reasons


def test_cache_rejects_sha256_mismatch():
    payload = _base_payload()
    payload["dataset_info_cache"]["npz_sha256"] = "def"
    from utils.eval_utils import validate_cached_predictions_with_dataset_info

    ok, reasons = validate_cached_predictions_with_dataset_info(
        **payload, spotcheck_k=10, rng_seed=0
    )
    assert ok is False
    assert "npz_sha256_mismatch" in reasons
