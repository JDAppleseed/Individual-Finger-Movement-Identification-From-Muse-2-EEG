"""
utils/splitting.py

Leakage-resistant split utilities for EEG window datasets.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

try:
    from utils.label_schema import ACTION_REST
except Exception:  # pragma: no cover - optional dependency
    ACTION_REST = None


_TRIAL_KEYS = ["event_id", "event_index", "trial_id", "event_trial_id", "trial"]
_BLOCK_KEYS = ["block_id", "event_block_id", "block"]
_WINDOW_START_KEYS = ["window_start", "start_s", "onset_s"]
_WINDOW_INDEX_KEYS = ["window_idx", "global_window_idx"]
_SESSION_KEYS = ["session_id"]
_HOP_KEYS = [
    "hop_s",
    "hop_seconds",
    "window_hop_s",
    "window_hop",
    "hop",
    "step_s",
    "stride_s",
]

_IDENTIFIER_KEYS = {
    "window_idx",
    "global_window_idx",
    "trial_id",
    "event_trial_id",
    "event_id",
    "event_index",
    "block_id",
    "event_block_id",
    "session_id",
    "timestamp",
    "timestamp_s",
    "window_start",
    "window_end",
    "start_s",
    "end_s",
    "onset_s",
    "offset_s",
}


def _as_1d_int64(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after reshape; got shape {arr.shape}")
    return arr


def _get_meta_array(meta: Optional[Dict[str, Any]], keys, n: int) -> Optional[np.ndarray]:
    if not meta:
        return None
    for key in keys:
        if key not in meta:
            continue
        try:
            arr = np.asarray(meta[key])
        except Exception:
            continue
        if arr.ndim == 0:
            continue
        try:
            if len(arr) != n:
                continue
        except Exception:
            continue
        return arr.reshape(-1)
    return None


def _is_usable_group(arr: np.ndarray) -> bool:
    arr = np.asarray(arr).reshape(-1)
    if arr.ndim != 1 or len(arr) == 0:
        return False
    if arr.dtype.kind in "OUS":
        arr_u = np.asarray(arr).astype("U")
        unique = np.unique(arr_u)
        if len(unique) <= 1:
            return False
        nonempty = (arr_u != "") & (arr_u != "UNKNOWN")
        if nonempty.any():
            unique_nonempty = np.unique(arr_u[nonempty])
            if len(unique_nonempty) <= 1:
                return False
        return True
    unique = np.unique(arr)
    if len(unique) <= 1:
        return False
    if np.all(arr == -1):
        return False
    return True


def _infer_hop_seconds(meta: Optional[Dict[str, Any]], window_start: np.ndarray) -> Optional[float]:
    if meta:
        for key in _HOP_KEYS:
            if key not in meta:
                continue
            try:
                val = np.asarray(meta[key])
                if isinstance(val, np.ndarray) and val.ndim == 0:
                    val = val.item()
                hop = float(val)
            except Exception:
                continue
            if np.isfinite(hop) and hop > 0:
                return hop
    diffs = np.diff(window_start.astype(float))
    diffs = diffs[diffs > 0]
    if diffs.size:
        hop = float(np.median(diffs))
        if np.isfinite(hop) and hop > 0:
            return hop
    return None


def _block_time_groups(
    block_id: np.ndarray, window_start: np.ndarray, hop_seconds: Optional[float]
) -> np.ndarray:
    n = len(block_id)
    groups = np.empty(n, dtype=np.int64)
    block_id = np.asarray(block_id)
    window_start = np.asarray(window_start, dtype=float)

    order = []
    seen = set()
    for val in block_id:
        key = val if isinstance(val, (str, int, float, np.integer, np.floating)) else str(val)
        if key in seen:
            continue
        seen.add(key)
        order.append(val)

    next_gid = 0
    for val in order:
        mask = block_id == val
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        idx.sort()
        starts = window_start[idx]
        hop = hop_seconds
        if hop is None or not np.isfinite(hop) or hop <= 0:
            diffs = np.diff(starts)
            diffs = diffs[diffs > 0]
            hop = float(np.median(diffs)) if diffs.size else None
        gap_thr = hop * 1.5 if hop is not None and np.isfinite(hop) else None

        seg = 0
        last = None
        for j, start in zip(idx, starts):
            if last is not None:
                if gap_thr is not None and (start - last) > gap_thr:
                    seg += 1
                elif start < last:
                    seg += 1
            groups[j] = next_gid + seg
            last = start
        next_gid += seg + 1

    return groups


def infer_groups(meta: Dict[str, Any], n: int) -> np.ndarray:
    """
    Infer grouping IDs for leakage-safe splitting.

    Priority:
      1) trial/event IDs
      2) block IDs + contiguous window_start segments
      3) raise ValueError with actionable guidance
    """
    if not meta:
        raise ValueError(
            "No trial/event metadata found. Add trial_id/event_id to NPZ meta during window extraction."
        )

    for key in _TRIAL_KEYS:
        arr = _get_meta_array(meta, [key], n)
        if arr is None:
            continue
        # Prefer event_id/event_index, but avoid collapsing all no-event windows into one group.
        if key in {"event_id", "event_index"}:
            try:
                arr_int = np.asarray(arr).astype(np.int64).reshape(-1)
            except Exception:
                arr_int = None
            if arr_int is not None:
                if np.all(arr_int == -1):
                    # No usable event IDs at all; fall through to block/time groups.
                    continue
                if np.any(arr_int == -1):
                    rest_mask = arr_int == -1
                    window_start = _get_meta_array(meta, _WINDOW_START_KEYS, n)
                    if window_start is None:
                        continue
                    hop = _infer_hop_seconds(meta, np.asarray(window_start, dtype=float))
                    rest_block = np.zeros(int(rest_mask.sum()), dtype=np.int64)
                    rest_groups = _block_time_groups(
                        rest_block, np.asarray(window_start, dtype=float)[rest_mask], hop
                    )
                    offset = int(arr_int[~rest_mask].max() + 1) if np.any(~rest_mask) else 0
                    groups = arr_int.copy()
                    groups[rest_mask] = rest_groups + offset
                    if _is_usable_group(groups):
                        return groups
                if _is_usable_group(arr_int):
                    return arr_int
        if _is_usable_group(arr):
            return np.asarray(arr).reshape(-1)

    block_id = _get_meta_array(meta, _BLOCK_KEYS, n)
    window_start = _get_meta_array(meta, _WINDOW_START_KEYS, n)
    if block_id is not None and window_start is not None:
        hop = _infer_hop_seconds(meta, window_start)
        return _block_time_groups(block_id, window_start, hop)

    raise ValueError(
        "No trial/event metadata found. Add trial_id/event_id to NPZ meta during window extraction."
    )


def _build_joint_label(y_action: np.ndarray, y_finger: np.ndarray) -> np.ndarray:
    y_action = np.asarray(y_action).astype(int)
    y_finger = np.asarray(y_finger).astype(int)
    max_finger = int(np.max(y_finger)) if y_finger.size else 0
    if ACTION_REST is None:
        return y_action * (max_finger + 1) + y_finger
    finger_adj = np.where(y_action == ACTION_REST, -1, y_finger)
    return y_action * (max_finger + 2) + (finger_adj + 1)


def _factorize(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels).reshape(-1)
    _, inv = np.unique(labels, return_inverse=True)
    return inv


def resolve_auxiliary_rest_sessions(
    y_action: np.ndarray,
    meta: Optional[Dict[str, Any]],
    *,
    policy: str = "none",
) -> Dict[str, Any]:
    y_action = _as_1d_int64(y_action, "y_action")
    result: Dict[str, Any] = {
        "policy": str(policy or "none"),
        "enabled": False,
        "reason": "not_requested",
        "core_idx": np.arange(len(y_action), dtype=np.int64),
        "aux_idx": np.array([], dtype=np.int64),
        "core_sessions": [],
        "aux_sessions": [],
        "session_action_counts": {},
    }
    policy = str(policy or "none").strip().lower()
    if policy == "none":
        return result
    if policy != "auto_train_only":
        raise ValueError(f"Unsupported auxiliary rest session policy: {policy}")
    if ACTION_REST is None:
        result["reason"] = "missing_action_rest"
        return result

    session_id = _get_meta_array(meta, _SESSION_KEYS, len(y_action))
    if session_id is None:
        result["reason"] = "missing_session_id"
        return result
    session_id = np.asarray(session_id).astype("U")

    valid_mask = (session_id != "") & (session_id != "UNKNOWN")
    unique_sessions = np.unique(session_id[valid_mask]) if np.any(valid_mask) else np.array([])
    if len(unique_sessions) < 2:
        result["reason"] = "insufficient_sessions"
        return result

    aux_sessions = []
    core_sessions = []
    session_action_counts: Dict[str, Dict[int, int]] = {}
    for sid in unique_sessions.tolist():
        sid_mask = session_id == sid
        sid_actions = y_action[sid_mask]
        unique_actions, counts = np.unique(sid_actions, return_counts=True)
        session_action_counts[str(sid)] = {
            int(action_id): int(count)
            for action_id, count in zip(unique_actions.tolist(), counts.tolist())
        }
        if len(unique_actions) == 1 and int(unique_actions[0]) == int(ACTION_REST):
            aux_sessions.append(str(sid))
        else:
            core_sessions.append(str(sid))

    result["session_action_counts"] = session_action_counts
    result["core_sessions"] = core_sessions
    result["aux_sessions"] = aux_sessions

    if not aux_sessions:
        result["reason"] = "no_rest_only_sessions"
        return result
    if not core_sessions:
        result["reason"] = "all_sessions_rest_only"
        return result

    aux_mask = np.isin(session_id, np.asarray(aux_sessions, dtype="U"))
    result["enabled"] = True
    result["reason"] = "rest_only_sessions_train_only"
    result["core_idx"] = np.flatnonzero(~aux_mask).astype(np.int64)
    result["aux_idx"] = np.flatnonzero(aux_mask).astype(np.int64)
    return result


def _split_score(
    label_ids: np.ndarray,
    y_action: np.ndarray,
    test_idx: np.ndarray,
    test_size: float,
    session_ids: Optional[np.ndarray] = None,
    groups: Optional[np.ndarray] = None,
) -> float:
    overall = np.bincount(label_ids)
    test_counts = np.bincount(label_ids[test_idx], minlength=len(overall))
    overall_freq = overall / max(1, overall.sum())
    test_freq = test_counts / max(1, test_counts.sum())
    score = float(np.sum(np.abs(test_freq - overall_freq)))

    action_ids = np.asarray(y_action).astype(int)
    action_overall = np.bincount(action_ids)
    action_test = np.bincount(action_ids[test_idx], minlength=len(action_overall))
    missing_actions = int(np.sum((action_overall > 0) & (action_test == 0)))
    if missing_actions:
        score += 5.0 * missing_actions

    actual_test = len(test_idx) / max(1, len(label_ids))
    score += abs(actual_test - test_size)

    if session_ids is not None:
        session_ids = np.asarray(session_ids).reshape(-1)
        session_label_ids = _factorize(session_ids)
        overall_sessions = np.bincount(session_label_ids)
        test_sessions = np.bincount(
            session_label_ids[test_idx], minlength=len(overall_sessions)
        )
        overall_session_freq = overall_sessions / max(1, overall_sessions.sum())
        test_session_freq = test_sessions / max(1, test_sessions.sum())
        score += 0.5 * float(np.sum(np.abs(test_session_freq - overall_session_freq)))

        if ACTION_REST is not None:
            rest_mask = np.asarray(y_action).astype(int) == int(ACTION_REST)
            if np.any(rest_mask):
                overall_rest_sessions = np.bincount(
                    session_label_ids[rest_mask], minlength=len(overall_sessions)
                )
                test_rest_idx = test_idx[rest_mask[test_idx]]
                if len(test_rest_idx) == 0:
                    score += 5.0
                else:
                    test_rest_sessions = np.bincount(
                        session_label_ids[test_rest_idx], minlength=len(overall_sessions)
                    )
                    overall_rest_freq = overall_rest_sessions / max(
                        1, overall_rest_sessions.sum()
                    )
                    test_rest_freq = test_rest_sessions / max(
                        1, test_rest_sessions.sum()
                    )
                    score += 2.0 * float(
                        np.sum(np.abs(test_rest_freq - overall_rest_freq))
                    )

    if groups is not None and ACTION_REST is not None:
        groups = np.asarray(groups).reshape(-1)
        rest_mask = np.asarray(y_action).astype(int) == int(ACTION_REST)
        if np.any(rest_mask):
            rest_groups_all = np.unique(groups[rest_mask])
            total_rest_groups = int(len(rest_groups_all))
            if total_rest_groups > 0:
                rest_test_idx = test_idx[rest_mask[test_idx]]
                rest_test_groups = np.unique(groups[rest_test_idx])
                test_rest_group_count = int(len(rest_test_groups))

                # When multiple rest events exist, require the test split to cover
                # more than one of them whenever possible. This avoids brittle
                # evaluations where all REST performance is determined by a single
                # held-out baseline-rest block.
                target_rest_groups = min(
                    total_rest_groups - 1,
                    max(1, int(np.ceil(total_rest_groups * float(test_size)))),
                )
                if total_rest_groups >= 3:
                    target_rest_groups = max(2, target_rest_groups)

                if test_rest_group_count < target_rest_groups:
                    score += 10.0 * float(target_rest_groups - test_rest_group_count)
                else:
                    score += 0.25 * abs(test_rest_group_count - target_rest_groups)
    return score


def _choose_group_split(
    indices: np.ndarray,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
    session_ids: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    groups = np.asarray(groups).reshape(-1)
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        raise ValueError("Need at least 2 groups for group-aware splitting.")

    joint = _build_joint_label(y_action, y_finger)
    label_ids = _factorize(joint)

    n_splits = int(min(50, max(10, n_groups)))
    splitter = GroupShuffleSplit(
        n_splits=n_splits, test_size=test_size, random_state=random_state
    )

    best_score = None
    best_split = None
    for train_idx, test_idx in splitter.split(indices, y_action, groups=groups):
        score = _split_score(
            label_ids,
            y_action,
            test_idx,
            test_size,
            session_ids=session_ids,
            groups=groups,
        )
        if best_score is None or score < best_score:
            best_score = score
            best_split = (train_idx, test_idx)

    if best_split is None:
        train_idx, test_idx = next(splitter.split(indices, y_action, groups=groups))
        return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)

    train_idx, test_idx = best_split
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def _fallback_split(
    y_action: np.ndarray,
    y_finger: np.ndarray,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(y_action), dtype=np.int64)
    max_finger = int(np.max(y_finger)) if len(y_finger) else 0
    stratify_labels = (y_action.astype(int) * (max_finger + 1)) + y_finger.astype(int)
    try:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            stratify=stratify_labels,
            random_state=random_state,
        )
    except ValueError:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=None
        )
    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def _purge_train_indices(
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    meta: Dict[str, Any],
    purge_seconds: float,
    hop_seconds: Optional[float],
) -> np.ndarray:
    if purge_seconds <= 0:
        return train_idx

    n_total = len(train_idx) + len(test_idx)
    window_start = _get_meta_array(meta, _WINDOW_START_KEYS, n_total)
    if window_start is None:
        if hop_seconds is None or not np.isfinite(hop_seconds) or hop_seconds <= 0:
            hop_seconds = _infer_hop_seconds(meta, np.array([], dtype=float))
        if hop_seconds is None or not np.isfinite(hop_seconds) or hop_seconds <= 0:
            return train_idx
        window_idx = _get_meta_array(meta, _WINDOW_INDEX_KEYS, n_total)
        if window_idx is None:
            return train_idx
        window_start = np.asarray(window_idx, dtype=float) * float(hop_seconds)

    if window_start is None:
        return train_idx

    window_start = np.asarray(window_start, dtype=float).reshape(-1)
    if len(window_start) == 0:
        return train_idx

    session = _get_meta_array(meta, _SESSION_KEYS, len(window_start))
    if session is None:
        session = np.array(["GLOBAL"] * len(window_start), dtype="U")
    else:
        session = np.asarray(session).astype("U")

    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    keep_mask = np.ones(len(train_idx), dtype=bool)
    test_starts = window_start[test_idx]
    test_sessions = session[test_idx]

    for i, idx in enumerate(train_idx):
        sess = session[idx]
        if not np.any(test_sessions == sess):
            continue
        start = window_start[idx]
        diffs = np.abs(test_starts[test_sessions == sess] - start)
        if diffs.size and np.min(diffs) <= purge_seconds:
            keep_mask[i] = False

    return train_idx[keep_mask]


def split_indices(
    y_action: np.ndarray,
    y_finger: np.ndarray,
    meta: Optional[Dict[str, Any]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    split_mode: str = "group_trial",
    purge_seconds: float = 0.0,
    hop_seconds: Optional[float] = None,
    allow_fallback: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (train_idx, test_idx) as int64 arrays using group-aware splitting.

    Default split_mode="group_trial" uses trial/event groups. If missing, raises ValueError
    unless allow_fallback=True.
    """
    y_action = _as_1d_int64(y_action, "y_action")
    y_finger = _as_1d_int64(y_finger, "y_finger")
    n = len(y_action)
    indices = np.arange(n, dtype=np.int64)

    if not meta:
        if allow_fallback:
            return _fallback_split(y_action, y_finger, test_size, random_state)
        raise ValueError(
            "No trial/event metadata found. Add trial_id/event_id to NPZ meta during window extraction."
        )

    if split_mode == "holdout_session":
        session = _get_meta_array(meta, _SESSION_KEYS, n)
        if session is None:
            if allow_fallback:
                return _fallback_split(y_action, y_finger, test_size, random_state)
            raise ValueError("split_mode=holdout_session requires session_id in meta.")
        if not _is_usable_group(session):
            if allow_fallback:
                return _fallback_split(y_action, y_finger, test_size, random_state)
            raise ValueError("session_id in meta is not usable for splitting.")
        groups = session
    else:
        groups = infer_groups(meta, n)
    session_ids = _get_meta_array(meta, _SESSION_KEYS, n)
    if session_ids is not None and not _is_usable_group(session_ids):
        session_ids = None

    train_idx, test_idx = _choose_group_split(
        indices,
        y_action,
        y_finger,
        groups,
        test_size,
        random_state,
        session_ids=session_ids,
    )

    if purge_seconds > 0:
        train_idx = _purge_train_indices(
            train_idx, test_idx, meta, purge_seconds, hop_seconds
        )

    return np.asarray(train_idx, dtype=np.int64), np.asarray(test_idx, dtype=np.int64)


def assert_no_group_overlap(
    groups: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray
) -> None:
    groups = np.asarray(groups).reshape(-1)
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    overlap = np.intersect1d(train_groups, test_groups)
    if overlap.size:
        sample = ", ".join([str(v) for v in overlap[:10]])
        raise RuntimeError(
            f"Group leakage detected: {len(overlap)} overlapping groups (sample: {sample})"
        )


def assert_identifier_not_in_X(meta: Optional[Dict[str, Any]], X_keys_or_columns) -> None:
    if X_keys_or_columns is None:
        return
    keys = [str(k) for k in X_keys_or_columns]
    overlap = sorted(set(keys) & _IDENTIFIER_KEYS)
    if overlap:
        raise RuntimeError(
            f"Identifier fields present in features: {overlap}. Remove identifier columns before training/evaluation."
        )
