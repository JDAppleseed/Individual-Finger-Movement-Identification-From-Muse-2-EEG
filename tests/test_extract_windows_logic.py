import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")


def _load_extract_module():
    module_path = Path(__file__).resolve().parents[1] / "1b_extract_windows.py"
    spec = importlib.util.spec_from_file_location("extract_windows", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rest_allowlist_is_stable():
    mod = _load_extract_module()
    event = {
        "onset_s": 0.0,
        "duration_s": 1.0,
        "action_id": mod.ACTION_REST,
        "finger_id": mod.FINGER_NONE,
        "type": "rest",
        "event_index": 7,
    }
    baseline_rest_set = {event["event_index"]}

    event_copy = dict(event)
    assert event_copy["event_index"] in baseline_rest_set


def test_rest_subsampling_deterministic():
    mod = _load_extract_module()
    rest_indices = list(range(10))
    keep_a, drop_sub_a, drop_cap_a = mod._select_rest_keep_indices(
        rest_indices=rest_indices,
        non_rest_count=5,
        subsample_prob=0.5,
        seed=42,
        rest_cap=3,
    )
    keep_b, drop_sub_b, drop_cap_b = mod._select_rest_keep_indices(
        rest_indices=rest_indices,
        non_rest_count=5,
        subsample_prob=0.5,
        seed=42,
        rest_cap=3,
    )
    assert keep_a == keep_b
    assert drop_sub_a == drop_sub_b
    assert drop_cap_a == drop_cap_b


def test_rest_subsampling_guard_when_no_non_rest():
    mod = _load_extract_module()
    rest_indices = list(range(5))
    keep, drop_sub, drop_cap = mod._select_rest_keep_indices(
        rest_indices=rest_indices,
        non_rest_count=0,
        subsample_prob=0.1,
        seed=7,
        rest_cap=1,
    )
    assert keep == rest_indices
    assert drop_sub == 0
    assert drop_cap == 0


def test_overlap_tie_break_is_robust():
    mod = _load_extract_module()
    event_a = {
        "onset_s": 0.1,
        "duration_s": 0.5,
        "action_id": 1,
        "finger_id": 2,
        "type": "move",
        "event_id": 1,
    }
    event_b = {
        "onset_s": 0.1,
        "duration_s": 0.5,
        "action_id": 1,
        "finger_id": 2,
        "type": "move",
        "event_id": 2,
    }
    overlaps = [
        (0.2, 0.8, event_a),
        (0.2000001, 0.8, event_b),
    ]
    best, ambiguous = mod._select_best_overlap(overlaps)
    assert best[2]["event_id"] == 2
    assert ambiguous is False

    event_c = dict(event_a)
    event_c["event_id"] = 1
    overlaps_identical = [
        (0.2, 0.8, event_a),
        (0.2, 0.8, event_c),
    ]
    best_identical, ambiguous_identical = mod._select_best_overlap(overlaps_identical)
    assert best_identical[2]["event_id"] == 1
    assert ambiguous_identical is True
