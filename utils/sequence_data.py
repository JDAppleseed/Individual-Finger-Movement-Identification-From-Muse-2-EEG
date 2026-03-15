"""
utils/sequence_data.py

Robust NPZ loader + split utilities for EEG window datasets produced by Step 1b.

Key goals:
- Preserve *all* useful metadata from Step 1b NPZ (not just a short whitelist).
- Keep y_action/y_finger strict (1D int64) and validate lengths.
- Support memmap loading (don't cast X if mmap_mode is set).
- Prefer leakage-resistant splits (trial/event groups first; block segments if needed).
- Stratify when possible, but never break groups.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

from utils.runtime_utils import (
    apply_channel_normalizer as _apply_saved_normalizer,
    normalize_preprocess_config,
    preprocess_eeg_windows,
)
from utils.splitting import split_indices as _safe_split_indices


# -------------------------
# Helpers
# -------------------------

_REQUIRED_KEYS = ("X", "y_action", "y_finger")


_STRING_META_KEYS = {
    "subject_id",
    "experiment_hash",
    "session_id",
    "channel_names",
    "assigned_event_type",
    "event_source",
    "session_mode",
    "features_path",
    "events_path",
    "source_features_path",
    "source_events_path",
    "timebase_version",
    "interpolation_policy",
    "gap_policy",
}

# Many Step-1b keys are numeric; we keep dtype as-is unless we normalize a known string key.
# We also keep JSON config blobs as-is (often dtype "U" with one entry).
_JSONISH_KEYS = {"config", "dataset_info"}


def _as_1d_int64(x, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.int64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D after reshape; got shape {arr.shape}")
    return arr


def _normalize_string_array(arr: np.ndarray) -> np.ndarray:
    # Normalize to unicode array. Handles object arrays of strings too.
    try:
        return np.asarray(arr).astype("U")
    except Exception:
        # Best-effort fallback: stringify elementwise
        return np.array([str(v) for v in np.asarray(arr).reshape(-1)], dtype="U")


def _maybe_scalar(arr: np.ndarray):
    # Some NPZ scalars are stored as 0-d arrays.
    if isinstance(arr, np.ndarray) and arr.ndim == 0:
        try:
            return arr.item()
        except Exception:
            return arr
    return arr


def _unique_nonempty(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).astype("U")
    # Filter empty / UNKNOWN
    keep = (values != "") & (values != "UNKNOWN")
    if keep.any():
        return np.unique(values[keep])
    return np.unique(values)


# -------------------------
# Public API
# -------------------------


def load_sequence_npz(
    path: str | Path = "eeg_windows.npz", mmap_mode: Optional[str] = None
):
    """
    Load an EEG window dataset from .npz.

    Returns:
        X: (N,T,C) or (N,C,T) float32 (cast only if mmap_mode is None)
        y_action: (N,) int64
        y_finger: (N,) int64
        meta: dict of all other NPZ keys (arrays preserved)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sequence window file not found: {path}")

    data = np.load(path, allow_pickle=True, mmap_mode=mmap_mode)
    keys = set(data.files)

    missing = [k for k in _REQUIRED_KEYS if k not in keys]
    if missing:
        raise KeyError(
            f"Missing required keys in NPZ {path}: {missing}. Available keys: {sorted(keys)}"
        )

    X = data["X"]
    if mmap_mode is None:
        # When not memmapping, force float32 for torch friendliness.
        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if X.dtype != np.float32:
            X = X.astype(np.float32)

    # Labels: strict 1D int64
    y_action = _as_1d_int64(data["y_action"], "y_action")
    y_finger = _as_1d_int64(data["y_finger"], "y_finger")

    # Validate X shape (don’t transpose here; Step 2 has ensure_X_shape)
    X_arr = X if isinstance(X, np.ndarray) else np.asarray(X)
    if X_arr.ndim != 3:
        raise ValueError(
            f"Expected X to be 3D (N,T,C) or (N,C,T) in {path}, got shape {X_arr.shape}"
        )

    n = int(X_arr.shape[0])
    if len(y_action) != n or len(y_finger) != n:
        raise ValueError(
            f"Dataset length mismatch in {path}: X has N={n}, "
            f"y_action={len(y_action)}, y_finger={len(y_finger)}"
        )

    # Meta: keep everything except X/y arrays
    meta: Dict[str, Any] = {}
    for k in data.files:
        if k in _REQUIRED_KEYS:
            continue
        v = data[k]

        # Normalize known string-ish arrays
        if k in _STRING_META_KEYS:
            v = _normalize_string_array(v)
        else:
            # Preserve scalars as python types when convenient
            v = _maybe_scalar(v)

        meta[k] = v

    # If some expected keys are missing, that's okay; Step 2 handles optional meta.
    # But ensure subject_id/expt/session are in a predictable dtype if present.
    for k in ("subject_id", "experiment_hash", "session_id"):
        if k in meta and isinstance(meta[k], np.ndarray):
            meta[k] = meta[k].astype("U")

    return X, y_action, y_finger, meta


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
    Return (train_idx, test_idx) as int64 arrays using leakage-aware, group-safe splitting.
    """
    return _safe_split_indices(
        y_action=y_action,
        y_finger=y_finger,
        meta=meta,
        test_size=test_size,
        random_state=random_state,
        split_mode=split_mode,
        purge_seconds=purge_seconds,
        hop_seconds=hop_seconds,
        allow_fallback=allow_fallback,
    )


def fit_channel_normalizer(
    X_train: np.ndarray, preprocess: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Per-channel z-score stats over (N,T, C). Works with memmaps.
    """
    X_train = np.asarray(X_train)
    if X_train.ndim != 3:
        raise ValueError(f"X_train must be 3D (N,T,C), got shape {X_train.shape}")

    preprocess_cfg = normalize_preprocess_config(preprocess)
    X_proc = preprocess_eeg_windows(X_train, preprocess_cfg)
    mean = X_proc.mean(axis=(0, 1)).astype(np.float32)
    std = X_proc.std(axis=(0, 1)).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    return {
        "type": "per_channel",
        "mean": mean,
        "std": std,
        "channels": int(X_train.shape[-1]),
        "preprocess": preprocess_cfg,
    }


def apply_channel_normalizer(X: np.ndarray, normalizer: Dict[str, Any]) -> np.ndarray:
    """
    Apply per-channel normalization. Returns float32.
    """
    return _apply_saved_normalizer(X, normalizer)


def summarize_windows(X: np.ndarray) -> pd.DataFrame:
    """
    Convert (N,T,C) windows into compact per-channel summary features for
    diagnostic tooling such as Deepchecks.

    These features are intentionally simpler than the raw sequence input used by
    the CNN+LSTM model. Favor lower-redundancy summaries over multiple nearly
    equivalent magnitude statistics so the diagnostic report surfaces more
    meaningful train/test differences.
    """
    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"X must be 3D (N,T,C), got shape {X.shape}")

    means = X.mean(axis=1)
    stds = X.std(axis=1)
    line_length = np.mean(np.abs(np.diff(X, axis=1)), axis=1)
    centered = X - means[:, None, :]
    zero_cross_rate = np.mean(
        centered[:, :-1, :] * centered[:, 1:, :] < 0.0,
        axis=1,
    )

    feats = []
    names = []
    C = int(X.shape[2])
    for idx in range(C):
        feats.append(means[:, idx])
        names.append(f"ch{idx + 1}_mean")
        feats.append(stds[:, idx])
        names.append(f"ch{idx + 1}_std")
        feats.append(line_length[:, idx])
        names.append(f"ch{idx + 1}_line_length")
        feats.append(zero_cross_rate[:, idx])
        names.append(f"ch{idx + 1}_zero_cross_rate")

    feat_mat = np.stack(feats, axis=1)
    return pd.DataFrame(feat_mat, columns=names)
