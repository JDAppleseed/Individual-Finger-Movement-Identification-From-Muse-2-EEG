import numpy as np

from utils.splitting import (
    _split_score,
    split_indices,
    infer_groups,
    assert_no_group_overlap,
    resolve_auxiliary_rest_sessions,
)


def test_group_overlap_empty():
    n_trials = 100
    windows_per_trial = 10
    n = n_trials * windows_per_trial

    trial_id = np.repeat(np.arange(n_trials), windows_per_trial)
    y_action = np.random.randint(0, 3, size=n)
    y_finger = np.random.randint(0, 5, size=n)
    meta = {"trial_id": trial_id}

    train_idx, test_idx = split_indices(
        y_action,
        y_finger,
        meta=meta,
        test_size=0.2,
        random_state=0,
    )

    groups = infer_groups(meta, n)
    assert_no_group_overlap(groups, train_idx, test_idx)


def test_purge_removes_boundary_windows():
    n_per_trial = 10
    n = n_per_trial * 2

    trial_id = np.repeat([0, 1], n_per_trial)
    window_start = np.arange(n, dtype=float)
    session_id = np.array(["S1"] * n, dtype="U")

    y_action = np.random.randint(0, 2, size=n)
    y_finger = np.random.randint(0, 2, size=n)

    meta = {
        "trial_id": trial_id,
        "window_start": window_start,
        "session_id": session_id,
        "hop_s": 1.0,
    }

    train_idx, test_idx = split_indices(
        y_action,
        y_finger,
        meta=meta,
        test_size=0.5,
        random_state=0,
        purge_seconds=2.0,
        hop_seconds=1.0,
    )

    train_starts = window_start[train_idx]
    test_starts = window_start[test_idx]
    if train_starts.size and test_starts.size:
        diffs = np.abs(train_starts[:, None] - test_starts[None, :])
        assert np.all(diffs > 2.0)

    assert len(train_idx) == n_per_trial - 2


def test_infer_groups_uses_event_id():
    windows_per_event = 5
    event_ids = np.repeat(np.arange(4), windows_per_event)
    n = len(event_ids)
    y_action = np.random.randint(0, 3, size=n)
    y_finger = np.random.randint(0, 5, size=n)
    meta = {"event_id": event_ids}

    groups = infer_groups(meta, n)
    assert len(np.unique(groups)) == 4
    assert np.all(groups == event_ids)


def test_split_score_prefers_representative_rest_session_mix():
    y_action = np.array([0] * 8 + [0] * 8 + [1] * 8 + [2] * 8, dtype=np.int64)
    y_finger = np.array([0] * 16 + [1] * 8 + [2] * 8, dtype=np.int64)
    label_ids = y_action * 10 + y_finger
    session_ids = np.array(["A"] * 8 + ["B"] * 8 + ["A"] * 8 + ["B"] * 8, dtype="U")

    balanced_test = np.array([0, 1, 8, 9, 16, 17, 24, 25], dtype=np.int64)
    rest_skewed_test = np.array([0, 1, 2, 3, 16, 17, 24, 25], dtype=np.int64)

    balanced_score = _split_score(
        label_ids,
        y_action,
        balanced_test,
        test_size=0.25,
        session_ids=session_ids,
    )
    skewed_score = _split_score(
        label_ids,
        y_action,
        rest_skewed_test,
        test_size=0.25,
        session_ids=session_ids,
    )

    assert balanced_score < skewed_score


def test_resolve_auxiliary_rest_sessions_marks_rest_only_sessions_train_only():
    y_action = np.array([0, 1, 2, 1, 2, 0, 0, 0], dtype=np.int64)
    meta = {
        "session_id": np.array(
            ["move_a"] * 3 + ["move_b"] * 2 + ["rest_only"] * 3,
            dtype="U",
        )
    }

    plan = resolve_auxiliary_rest_sessions(
        y_action,
        meta,
        policy="auto_train_only",
    )

    assert plan["enabled"] is True
    assert plan["aux_sessions"] == ["rest_only"]
    assert sorted(plan["core_sessions"]) == ["move_a", "move_b"]
    assert np.array_equal(plan["core_idx"], np.array([0, 1, 2, 3, 4], dtype=np.int64))
    assert np.array_equal(plan["aux_idx"], np.array([5, 6, 7], dtype=np.int64))
