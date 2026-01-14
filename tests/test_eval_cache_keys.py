import pytest

np = pytest.importorskip("numpy")


def test_resolve_cached_test_indices_prefers_test_indices():
    payload = {
        "test_indices_local": np.array([1, 2, 3]),
        "test_indices": np.array([9, 8, 7]),
    }
    from utils.eval_utils import resolve_cached_test_indices

    resolved = resolve_cached_test_indices(payload)
    assert resolved.tolist() == [9, 8, 7]


def test_resolve_cached_test_indices_falls_back():
    payload = {
        "test_indices": np.array([4, 5]),
    }
    from utils.eval_utils import resolve_cached_test_indices

    resolved = resolve_cached_test_indices(payload)
    assert resolved.tolist() == [4, 5]
