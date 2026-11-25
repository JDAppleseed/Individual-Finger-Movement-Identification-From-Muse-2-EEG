"""Preserve fitting class coverage without splitting calibration event groups."""
from __future__ import annotations

import numpy as np


def preserve_fit_coverage(*, calibration, groups, sessions, actions, starts,
                          purge_seconds, by_session=True):
    """Return purged fitting/calibration indices relative to the candidate pool.

    A class represented by one event cannot be independently fitted and calibrated.
    Prefer keeping that event in fitting. Move whole groups only, and also move
    neighboring calibration groups if purging would remove the last fitting
    example. Calibration may become empty; it must never consume the only fitting
    support for a source/action pair. Session holdouts preserve global class
    coverage instead of requiring the held-out session in fitting.
    """
    groups = np.asarray(groups)
    sessions = np.asarray(sessions).astype(str)
    actions = np.asarray(actions)
    starts = np.asarray(starts, dtype=float)
    calibration = np.asarray(calibration, dtype=np.int64)
    all_indices = np.arange(len(actions), dtype=np.int64)
    domains = sessions if by_session else np.full(len(actions), 'GLOBAL')
    required = sorted(set(zip(domains.tolist(), actions.tolist())))

    def near_calibration(indices):
        keep = np.ones(len(indices), dtype=bool)
        for j, i in enumerate(indices):
            same = calibration[sessions[calibration] == sessions[i]]
            if len(same) and np.any(np.abs(starts[same] - starts[i]) <= purge_seconds):
                keep[j] = False
        return indices[keep]

    while True:
        fit = near_calibration(np.setdiff1d(all_indices, calibration))
        observed = set(zip(domains[fit].tolist(), actions[fit].tolist()))
        missing = [key for key in required if key not in observed]
        if not missing:
            return fit, calibration
        domain, action = missing[0]
        candidates = all_indices[(domains == domain) & (actions == action)]
        # Deterministic choice independent of model performance or test labels.
        target_group = groups[candidates[0]]
        protected = all_indices[groups == target_group]
        move_groups = {target_group}
        for i in protected:
            nearby = calibration[(sessions[calibration] == sessions[i]) &
                (np.abs(starts[calibration] - starts[i]) <= purge_seconds)]
            move_groups.update(groups[nearby].tolist())
        remaining = calibration[~np.isin(groups[calibration], list(move_groups))]
        if len(remaining) >= len(calibration):
            raise ValueError('Cannot preserve fitting class coverage after calibration purge.')
        calibration = remaining


def split_coverage(meta, actions, **folds):
    """JSON-compatible per-source action counts for every saved fold."""
    sessions = np.asarray(meta.get('session_id', np.full(len(actions), 'GLOBAL'))).astype(str)
    return {name: {str(sid): np.bincount(
        np.asarray(actions)[np.asarray(indices)[sessions[np.asarray(indices)] == sid]],
        minlength=3).tolist() for sid in np.unique(sessions)}
        for name, indices in folds.items()}
