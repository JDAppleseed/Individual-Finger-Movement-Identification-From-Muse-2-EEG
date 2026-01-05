import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, GroupShuffleSplit


def load_sequence_npz(path="eeg_windows.npz", mmap_mode=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sequence window file not found: {path}")
    data = np.load(path, allow_pickle=True, mmap_mode=mmap_mode)
    X = data["X"]
    if X.dtype != np.float32 and mmap_mode is None:
        X = X.astype(np.float32)
    y_action = np.asarray(data["y_action"], dtype=np.int64)
    y_finger = np.asarray(data["y_finger"], dtype=np.int64)
    meta = {}
    for key in [
        "subject_id",
        "experiment_hash",
        "window_start",
        "window_end",
        "confidence_hint",
        "artifact_flag",
        "trial_id",
        "block_id",
    ]:
        if key in data:
            meta[key] = data[key]
    return X, y_action, y_finger, meta


def split_indices(y_action, y_finger, meta=None, test_size=0.2, random_state=42):
    n = len(y_action)
    indices = np.arange(n)
    groups = None
    if meta and "subject_id" in meta:
        subject_ids = np.array(meta["subject_id"])
        unique_subjects = np.unique(subject_ids)
        if len(unique_subjects) > 1 and not (len(unique_subjects) == 1 and unique_subjects[0] == "UNKNOWN"):
            groups = subject_ids

    if groups is not None:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(splitter.split(indices, y_action, groups=groups))
        return train_idx, test_idx

    stratify_labels = (y_action.astype(int) * 100) + y_finger.astype(int)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        stratify=stratify_labels,
        random_state=random_state,
    )
    return train_idx, test_idx


def fit_channel_normalizer(X_train):
    mean = X_train.mean(axis=(0, 1))
    std = X_train.std(axis=(0, 1))
    std = np.where(std < 1e-6, 1.0, std)
    return {
        "type": "per_channel",
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "channels": X_train.shape[-1],
    }


def apply_channel_normalizer(X, normalizer):
    mean = normalizer["mean"]
    std = normalizer["std"]
    return (X - mean) / std


def summarize_windows(X):
    means = X.mean(axis=1)
    stds = X.std(axis=1)
    rms = np.sqrt(np.mean(np.square(X), axis=1))
    ptp = np.ptp(X, axis=1)

    feats = []
    names = []
    for idx in range(X.shape[2]):
        feats.append(means[:, idx])
        names.append(f"ch{idx+1}_mean")
        feats.append(stds[:, idx])
        names.append(f"ch{idx+1}_std")
        feats.append(rms[:, idx])
        names.append(f"ch{idx+1}_rms")
        feats.append(ptp[:, idx])
        names.append(f"ch{idx+1}_ptp")

    feat_mat = np.stack(feats, axis=1)
    return pd.DataFrame(feat_mat, columns=names)
