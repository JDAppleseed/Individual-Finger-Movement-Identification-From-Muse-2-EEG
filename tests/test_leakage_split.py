import numpy as np

from utils.splitting import split_indices, infer_groups, assert_no_group_overlap


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
