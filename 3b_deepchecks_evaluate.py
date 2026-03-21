"""
STEP 3b — Deepchecks Evaluation (SDS-aligned)
Deterministic model behavior (Dropout OFF)
"""

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch

from deepchecks.tabular import Dataset, Suite
from deepchecks.tabular.suites import (
    data_integrity,
    train_test_validation,
    model_evaluation,
)

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.model_outputs import infer_output_dims_from_state_dict, unpack_model_outputs
from utils.label_schema import ACTION_NAMES, ACTION_REST, FINGER_NAMES
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    summarize_windows,
)
from utils.runtime_utils import (
    apply_channel_normalizer as apply_saved_channel_normalizer,
    load_normalizer,
)
from utils.splitting import infer_groups, assert_no_group_overlap, assert_identifier_not_in_X
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir

# Pipeline handoff: this is a diagnostic companion to Step 3; it reuses the same
# Step 2 run artifacts and split logic, then emits a Deepchecks HTML report.
try:
    from projects.session_finder import latest_session_for_subject
    from projects.session_paths import get_session_paths
except Exception:
    # PATCHED: session-aware path (fallback when projects package is unavailable)
    def latest_session_for_subject(project: str, subject: str):
        sessions_root = (
            Path(__file__).resolve().parent
            / "Projects"
            / project
            / "subjects"
            / subject
            / "sessions"
        )
        if not sessions_root.exists():
            return None
        sessions = [p for p in sessions_root.iterdir() if p.is_dir()]
        if not sessions:
            return None
        return max(sessions, key=lambda p: p.stat().st_mtime).name

    def get_session_paths(project: str, subject: str, session: str):
        session_dir = (
            Path(__file__).resolve().parent
            / "Projects"
            / project
            / "subjects"
            / subject
            / "sessions"
            / session
        )
        return _session_paths_from_dir(session_dir)


@dataclass(frozen=True)
class _SessionPathsCompat:
    windows_npz: Path
    model_file: Path
    scaler_file: Path
    test_predictions_npz: Path
    eval_dir: Path


def _session_paths_from_dir(session_dir: Path) -> _SessionPathsCompat:
    session_dir = resolve_session_dir(session_dir)
    layout = SessionLayout(session_dir)
    run_dir = resolve_latest_run_dir(session_dir)
    run_name = run_dir.name if run_dir is not None else "latest"
    model_base = run_dir if run_dir is not None else layout.models_root
    return _SessionPathsCompat(
        windows_npz=layout.windows_npz,
        model_file=model_base / "finger_action_model.pt",
        scaler_file=model_base / "scaler.npz",
        test_predictions_npz=model_base / "test_predictions.npz",
        eval_dir=layout.reports_root / run_name,
    )


def _session_paths_from_dir_with_run_override(
    session_dir: Path, run_dir: Optional[Path]
) -> _SessionPathsCompat:
    session_dir = resolve_session_dir(session_dir)
    layout = SessionLayout(session_dir)
    resolved_run_dir = None
    if run_dir is not None:
        resolved_run_dir = resolve_session_dir(run_dir)
        if resolved_run_dir.name == "models":
            resolved_run_dir = resolve_latest_run_dir(session_dir)
    else:
        resolved_run_dir = resolve_latest_run_dir(session_dir)
    run_name = resolved_run_dir.name if resolved_run_dir is not None else "latest"
    model_base = resolved_run_dir if resolved_run_dir is not None else layout.models_root
    return _SessionPathsCompat(
        windows_npz=layout.windows_npz,
        model_file=model_base / "finger_action_model.pt",
        scaler_file=model_base / "scaler.npz",
        test_predictions_npz=model_base / "test_predictions.npz",
        eval_dir=layout.reports_root / run_name,
    )


def _session_dir_from_run_dir(run_dir: Path) -> Optional[Path]:
    p = resolve_session_dir(run_dir)
    if p.name == "models":
        try:
            return p.parent.parent
        except Exception:
            return None
    if p.parent.name != "models":
        return None
    processed_dir = p.parent.parent
    if processed_dir.name != "processed":
        return None
    return processed_dir.parent


def _infer_context_from_session_dir(session_dir: Path) -> Tuple[Optional[str], Optional[str], str]:
    resolved = session_dir.expanduser().resolve()
    parts = resolved.parts
    for idx in range(0, len(parts) - 5):
        if (
            parts[idx].lower() == "projects"
            and parts[idx + 2].lower() == "subjects"
            and parts[idx + 4].lower() == "sessions"
        ):
            return parts[idx + 1], parts[idx + 3], parts[idx + 5]
    return None, None, resolved.name

# =========================
# ===== CONFIG ============
# =========================

SEED = 42
MIN_TEST_SAMPLES = 30
MAX_SPLIT_ATTEMPTS = 8
DEFAULT_BATCH_SIZE = 1024
DEFAULT_DEVICE = "auto"
DEFAULT_AMP_MODE = "off"


@dataclass
class InferenceTiming:
    calls: int = 0
    batches: int = 0
    windows: int = 0
    transfer_sec: float = 0.0
    model_sec: float = 0.0
    total_sec: float = 0.0


def _set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(requested: str) -> torch.device:
    requested = str(requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _resolve_amp_dtype(amp_mode: str, device: torch.device) -> Optional[torch.dtype]:
    amp_mode = str(amp_mode or "off").strip().lower()
    if amp_mode == "off":
        return None
    if amp_mode == "float16":
        if device.type in {"cuda", "mps"}:
            return torch.float16
        return None
    raise ValueError(f"Unsupported amp mode: {amp_mode}")


def _autocast_context(device: torch.device, amp_dtype: Optional[torch.dtype]):
    if amp_dtype is None:
        return contextlib.nullcontext()
    if not hasattr(torch, "amp") or not hasattr(torch.amp, "autocast"):
        return contextlib.nullcontext()
    return torch.amp.autocast(device_type=device.type, dtype=amp_dtype)


def _device_synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
        torch.mps.synchronize()


def _format_label_counts(values: np.ndarray, name_map: Optional[dict] = None):
    if values.size == 0:
        return "none"
    unique, counts = np.unique(values, return_counts=True)
    parts = []
    for val, count in zip(unique, counts):
        label = str(name_map.get(int(val), val)) if name_map else str(val)
        parts.append(f"{label}({int(val)})={int(count)}")
    return ", ".join(parts)


def _print_label_summary(prefix: str, y_action: np.ndarray, y_finger: np.ndarray):
    print(f"📊 {prefix} action labels: {_format_label_counts(y_action, ACTION_NAMES)}")
    print(f"📊 {prefix} finger labels: {_format_label_counts(y_finger, FINGER_NAMES)}")
    non_rest = y_action != ACTION_REST
    if np.any(non_rest):
        print(
            f"📊 {prefix} finger (non-REST): "
            f"{_format_label_counts(y_finger[non_rest], FINGER_NAMES)}"
        )
    else:
        print(f"📊 {prefix} finger (non-REST): none")


def _unique_non_rest_fingers(y_action: np.ndarray, y_finger: np.ndarray):
    mask = y_action != ACTION_REST
    if not np.any(mask):
        return 0
    return len(np.unique(y_finger[mask]))


def _apply_sample_limit(
    X, y_action, y_finger, meta, max_samples: Optional[int], seed: int
):
    if not max_samples or len(y_action) <= max_samples:
        return X, y_action, y_finger, meta

    indices = np.arange(len(y_action))
    stratify_labels = (y_action.astype(int) * 100) + y_finger.astype(int)
    try:
        from sklearn.model_selection import StratifiedShuffleSplit

        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=max_samples, random_state=seed
        )
        keep_idx, _ = next(splitter.split(indices, stratify_labels))
    except Exception:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(indices, size=max_samples, replace=False)

    keep_idx = np.sort(keep_idx)
    X = X[keep_idx]
    y_action = y_action[keep_idx]
    y_finger = y_finger[keep_idx]
    if meta:
        meta = {
            key: (
                np.asarray(val)[keep_idx]
                if isinstance(val, np.ndarray) and len(val) == len(indices)
                else val
            )
            for key, val in meta.items()
        }
    return X, y_action, y_finger, meta


def _mask_meta(meta, mask, n_before: int):
    if not meta:
        return {}
    mask = np.asarray(mask)
    out = {}
    for key, val in meta.items():
        if isinstance(val, dict) or isinstance(val, (str, bytes)):
            out[key] = val
            continue
        try:
            arr = np.asarray(val)
        except Exception:
            out[key] = val
            continue
        if arr.ndim == 0:
            out[key] = val
            continue
        try:
            if len(arr) == n_before:
                out[key] = arr[mask]
            else:
                out[key] = val
        except Exception:
            out[key] = val
    return out


def _ensure_X_shape(X, meta):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N,T,C), got shape {X.shape}")

    channel_count = None
    if meta and "channel_names" in meta:
        try:
            channel_count = int(len(np.asarray(meta["channel_names"])))
        except Exception:
            channel_count = None

    if channel_count is not None and channel_count > 0:
        if X.shape[2] == channel_count:
            return X
        if X.shape[1] == channel_count and X.shape[2] != channel_count:
            return np.transpose(X, (0, 2, 1))
        raise ValueError(
            f"Cannot infer X layout: expected channels in dim=2 or dim=1 to equal {channel_count}, got {X.shape}"
        )

    if X.shape[1] <= 16 and X.shape[2] > 16:
        return np.transpose(X, (0, 2, 1))
    return X


def _load_train_config(run_dir: Path) -> dict:
    path = run_dir / "train_config.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _split_with_checks(
    y_action,
    y_finger,
    meta,
    seed: int,
    split_mode: str,
    purge_seconds: float,
    hop_seconds,
    test_size: float,
):
    overall_action_unique = len(np.unique(y_action))
    overall_finger_unique = _unique_non_rest_fingers(y_action, y_finger)

    for attempt in range(MAX_SPLIT_ATTEMPTS):
        train_idx, test_idx = split_indices(
            y_action,
            y_finger,
            meta=meta,
            test_size=test_size,
            random_state=seed + attempt * 11,
            split_mode=split_mode,
            purge_seconds=purge_seconds,
            hop_seconds=hop_seconds,
            allow_fallback=False,
        )

        if len(test_idx) < MIN_TEST_SAMPLES:
            continue

        action_train_unique = len(np.unique(y_action[train_idx]))
        action_test_unique = len(np.unique(y_action[test_idx]))
        finger_train_unique = _unique_non_rest_fingers(
            y_action[train_idx], y_finger[train_idx]
        )
        finger_test_unique = _unique_non_rest_fingers(
            y_action[test_idx], y_finger[test_idx]
        )

        action_ok = overall_action_unique < 2 or (
            action_train_unique >= 2 and action_test_unique >= 2
        )
        finger_ok = overall_finger_unique < 2 or (
            finger_train_unique >= 2 and finger_test_unique >= 2
        )

        if action_ok and finger_ok:
            return train_idx, test_idx

    return None, None


def _dataset_kwargs():
    return {
        "index_name": "row_id",
        "set_index_from_dataframe_index": True,
    }


def _build_eeg_deepchecks_suite() -> Suite:
    excluded = {
        "FeatureFeatureCorrelation",
        "IdentifierLabelCorrelation",
    }
    checks = []
    for base_suite in (data_integrity(), train_test_validation(), model_evaluation()):
        for check in base_suite.checks.values():
            if type(check).__name__ in excluded:
                continue
            checks.append(check)
    return Suite("EEG Data Integrity Suite", *checks)


def _prepare_deepchecks_split(df, labels, lookup_ids, seed: int, index_start: int = 0):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(df))
    df = df.iloc[order].reset_index(drop=True).copy()
    labels = np.asarray(labels)[order]
    lookup_ids = np.asarray(lookup_ids, dtype=np.int64)[order]
    row_ids = np.arange(index_start, index_start + len(df), dtype=np.int64)
    df.index = pd.Index(rng.permutation(row_ids), name="row_id")
    return df, labels, lookup_ids


def _row_lookup_key(values) -> bytes:
    arr = np.asarray(values, dtype=np.float32)
    return np.round(arr, decimals=6).tobytes()


def _build_feature_row_lookup(df, lookup_ids, feature_names):
    mapping = {}
    values = np.asarray(df[feature_names], dtype=np.float32)
    for row, lookup_id in zip(values, np.asarray(lookup_ids, dtype=np.int64)):
        mapping.setdefault(_row_lookup_key(row), []).append(int(lookup_id))
    return mapping


def _lookup_window_idx_from_features(
    X_tabular, feature_names, feature_row_lookup
) -> np.ndarray:
    if not hasattr(X_tabular, "__getitem__"):
        raise KeyError("Deepchecks input is missing required lookup information.")
    try:
        values = np.asarray(X_tabular[feature_names], dtype=np.float32)
    except Exception as exc:
        raise KeyError("Deepchecks input is missing required lookup information.") from exc

    used_counts = {}
    resolved = np.zeros((len(values),), dtype=np.int64)
    for i, row in enumerate(values):
        key = _row_lookup_key(row)
        candidates = feature_row_lookup.get(key)
        if not candidates:
            raise KeyError("Could not map Deepchecks rows back to window tensors.")
        pos = used_counts.get(key, 0)
        if pos >= len(candidates):
            raise KeyError("Deepchecks lookup exhausted for repeated feature rows.")
        resolved[i] = candidates[pos]
        used_counts[key] = pos + 1
    return resolved


def _extract_window_idx(X_tabular) -> np.ndarray:
    if hasattr(X_tabular, "columns") and "window_idx" in X_tabular:
        return X_tabular["window_idx"].to_numpy().astype(np.int64)
    if hasattr(X_tabular, "index") and getattr(X_tabular.index, "name", None) == "window_idx":
        try:
            return np.asarray(X_tabular.index, dtype=np.int64)
        except Exception as exc:
            raise KeyError(
                "Deepchecks input is missing required window lookup indices."
            ) from exc
    raise KeyError("Deepchecks input is missing required window lookup indices.")


parser = argparse.ArgumentParser(
    description=(
        "Step 3b: run Deepchecks diagnostics on a trained Step 2 run using "
        "the same dataset and split logic as Step 3."
    )
)
selection_group = parser.add_argument_group("input selection")
selection_group.add_argument(
    "--run-dir",
    type=str,
    default=None,
    metavar="PATH",
    help="Specific Step 2 run directory to inspect (for example: .../processed/models/<run_id>).",
)
selection_group.add_argument(
    "--project",
    type=str,
    required=False,
    metavar="NAME",
    help="Project identifier used to resolve a session when --run-dir is not provided.",
)
selection_group.add_argument(
    "--subject",
    type=str,
    required=False,
    metavar="ID",
    help="Subject identifier used to resolve a session when --run-dir is not provided.",
)
selection_group.add_argument(
    "--session",
    type=str,
    required=False,
    metavar="ID",
    help="Session identifier to inspect. Defaults to the latest session for the subject.",
)
selection_group.add_argument(
    "--session-dir",
    type=str,
    default=None,
    metavar="PATH",
    help="Legacy session directory override used by the UI.",
)
runtime_group = parser.add_argument_group("runtime and split overrides")
runtime_group.add_argument(
    "--max-samples",
    type=int,
    default=None,
    metavar="N",
    help="Cap the number of windows passed to Deepchecks.",
)
runtime_group.add_argument(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE,
    help="Inference batch size.",
)
runtime_group.add_argument(
    "--device",
    type=str,
    default=DEFAULT_DEVICE,
    choices=["auto", "cpu", "cuda", "mps"],
    help="Inference device. 'auto' prefers CUDA, then MPS, then CPU.",
)
runtime_group.add_argument(
    "--amp-mode",
    type=str,
    default=DEFAULT_AMP_MODE,
    choices=["off", "float16"],
    help=(
        "Experimental mixed precision inference mode. 'off' preserves current numerics. "
        "'float16' is opt-in for CUDA/MPS benchmarking."
    ),
)
runtime_group.add_argument(
    "--test-size",
    type=float,
    default=None,
    help="Fraction reserved for the test split. Defaults to train_config.json when available.",
)
runtime_group.add_argument(
    "--split-seed",
    type=int,
    default=None,
    help="Random seed used when rebuilding the split. Defaults to train_config.json when available.",
)
runtime_group.add_argument(
    "--split-mode",
    type=str,
    default=None,
    choices=["group_trial", "holdout_session"],
    help="Split strategy for train/test partitions. Defaults to train_config.json when available.",
)
runtime_group.add_argument(
    "--purge-seconds",
    type=float,
    default=None,
    help="Drop training windows within this many seconds of any test window from the same session.",
)
runtime_group.add_argument(
    "--hop-seconds",
    type=float,
    default=None,
    help="Window hop size, in seconds, used by leakage-purge heuristics when needed.",
)
args = parser.parse_args()

# Keep session/run resolution aligned with Step 3 so diagnostics and core eval
# describe the same trained model and window dataset.
# PATCHED: session-aware path
legacy_session_dir: Optional[Path] = None
run_dir_override: Optional[Path] = None
if args.run_dir:
    run_dir_override = Path(args.run_dir).expanduser()
    if not run_dir_override.exists():
        print(f"Run dir not found: {run_dir_override}")
        raise SystemExit(2)
    legacy_session_dir = _session_dir_from_run_dir(run_dir_override)
    if legacy_session_dir is None:
        print(
            "Could not infer session dir from --run-dir; "
            "pass --project/--subject/--session or use --session-dir."
        )
        raise SystemExit(2)
    inferred_project, inferred_subject, inferred_session = _infer_context_from_session_dir(
        legacy_session_dir
    )
    if not args.project and inferred_project:
        args.project = inferred_project
    if not args.subject and inferred_subject:
        args.subject = inferred_subject
    if args.session is None:
        args.session = inferred_session
if args.session_dir:
    legacy_session_dir = resolve_session_dir(args.session_dir)
    inferred_project, inferred_subject, inferred_session = _infer_context_from_session_dir(
        legacy_session_dir
    )
    if not args.project and inferred_project:
        args.project = inferred_project
    if not args.subject and inferred_subject:
        args.subject = inferred_subject
    if args.session is None:
        args.session = inferred_session

if not args.project or not args.subject:
    print("Missing --project/--subject (or provide --session-dir under Projects/.../subjects/.../sessions/...).")
    raise SystemExit(2)

if args.session is None:
    args.session = latest_session_for_subject(args.project, args.subject)
if args.session is None:
    print(f"No session found for subject {args.subject} in project {args.project}.")
    raise SystemExit(2)

paths = (
    _session_paths_from_dir_with_run_override(legacy_session_dir, run_dir_override)
    if legacy_session_dir is not None
    else get_session_paths(args.project, args.subject, args.session)
)
# PATCHED: session-aware path
os.makedirs(paths.eval_dir, exist_ok=True)
print(
    f"[✓] Evaluating session {args.session} for subject {args.subject} in project {args.project}"
)
print(f"[✓] Loading model: {paths.model_file}")
print(f"[✓] Saving results to: {paths.eval_dir}")

npz_path = Path(paths.windows_npz).expanduser()
model_path = Path(paths.model_file).expanduser()
scaler_path = Path(paths.scaler_file).expanduser()
run_dir = model_path.parent
if not npz_path.exists():
    print(f"NPZ file not found: {npz_path}")
    raise SystemExit(2)
if not scaler_path.exists():
    print(f"Scaler file not found: {scaler_path}")
    raise SystemExit(2)
if not model_path.exists():
    print(f"Model file not found: {model_path}")
    raise SystemExit(2)

try:
    device = _resolve_device(args.device)
except Exception as exc:
    print(f"Invalid --device: {exc}")
    raise SystemExit(2)
try:
    amp_dtype = _resolve_amp_dtype(args.amp_mode, device)
except Exception as exc:
    print(f"Invalid --amp-mode: {exc}")
    raise SystemExit(2)
if str(args.amp_mode).lower() != "off" and amp_dtype is None:
    print(
        f"⚠️ AMP requested via --amp-mode={args.amp_mode}, but device={device} does not support it. Disabling AMP."
    )
print(
    f"Runtime backend: device={device.type}, amp_mode={'off' if amp_dtype is None else args.amp_mode}, "
    f"batch_size={max(1, int(args.batch_size))}"
)

X, y_action, y_finger, meta = load_sequence_npz(str(npz_path), mmap_mode="r")
try:
    X = _ensure_X_shape(X, meta if isinstance(meta, dict) else {})
except ValueError as exc:
    print(str(exc))
    raise SystemExit(2)
if isinstance(X, np.memmap) and X.dtype != np.float32:
    print(f"ℹ️ X dtype is {X.dtype}; casting to float32 per batch.")

train_cfg = _load_train_config(run_dir)
cli_flags = {
    "split_seed": "--split-seed" in sys.argv,
    "test_size": "--test-size" in sys.argv,
    "split_mode": "--split-mode" in sys.argv,
    "purge_seconds": "--purge-seconds" in sys.argv,
    "hop_seconds": "--hop-seconds" in sys.argv,
}

split_seed = (
    args.split_seed
    if cli_flags["split_seed"]
    else int(train_cfg.get("seed", SEED) or SEED)
)
test_size = (
    float(args.test_size)
    if cli_flags["test_size"] and args.test_size is not None
    else float(train_cfg.get("test_size", 0.2))
)
if not (0.0 < float(test_size) < 1.0):
    print(f"⚠️ Invalid test_size={test_size}; defaulting to 0.2.")
    test_size = 0.2
split_mode = (
    args.split_mode
    if cli_flags["split_mode"] and args.split_mode is not None
    else str(train_cfg.get("split_mode", "group_trial"))
)
split_mode = str(split_mode).strip()
if split_mode not in {"group_trial", "holdout_session"}:
    print(f"⚠️ Unknown split_mode={split_mode!r}; defaulting to 'group_trial'.")
    split_mode = "group_trial"
purge_seconds = (
    float(args.purge_seconds)
    if cli_flags["purge_seconds"] and args.purge_seconds is not None
    else float(train_cfg.get("purge_seconds", 0.0) or 0.0)
)
hop_seconds = (
    float(args.hop_seconds)
    if cli_flags["hop_seconds"] and args.hop_seconds is not None
    else train_cfg.get("hop_seconds", None)
)
_set_deterministic(int(split_seed))
print(
    "Split config: "
    f"test_size={test_size}, seed={split_seed}, mode={split_mode}, "
    f"purge_seconds={purge_seconds}, hop_seconds={hop_seconds}"
)

X, y_action, y_finger, meta = _apply_sample_limit(
    X, y_action, y_finger, meta, args.max_samples, split_seed
)

finite_mask = np.isfinite(X).all(axis=tuple(range(1, X.ndim)))
bad = int((~finite_mask).sum())
if bad > 0:
    print(f"⚠️ Dropping {bad}/{len(X)} samples with NaN/Inf in X (matching Step 2).")
    n_before = len(X)
    X = X[finite_mask]
    y_action = y_action[finite_mask]
    y_finger = y_finger[finite_mask]
    if isinstance(meta, dict):
        meta = _mask_meta(meta, finite_mask, n_before)
    if len(X) == 0:
        print("❌ All samples dropped due to NaN/Inf. Aborting Deepchecks.")
        raise SystemExit(2)

_print_label_summary("Filtered", y_action, y_finger)

# =========================
# ===== SAME SPLIT =========
# =========================

train_idx, test_idx = _split_with_checks(
    y_action,
    y_finger,
    meta=meta,
    seed=int(split_seed),
    split_mode=split_mode,
    purge_seconds=purge_seconds,
    hop_seconds=hop_seconds,
    test_size=test_size,
)
if train_idx is None or test_idx is None:
    print("⚠️ Unable to create a split with multiple classes. Aborting Deepchecks.")
    raise SystemExit(2)
try:
    if split_mode == "holdout_session":
        if not meta or "session_id" not in meta:
            raise ValueError("split_mode=holdout_session requires session_id in meta.")
        groups = np.asarray(meta["session_id"]).reshape(-1)
    else:
        groups = infer_groups(meta, len(y_action))
    assert_no_group_overlap(groups, train_idx, test_idx)
except Exception as exc:
    print(f"❌ Leakage guard tripped: {exc}")
    raise SystemExit(2)
X_train = X[train_idx]
X_test = X[test_idx]
y_train = y_action[train_idx]
y_test = y_action[test_idx]

if len(test_idx) < MIN_TEST_SAMPLES:
    print(f"⚠️ Test set too small ({len(test_idx)} samples). Aborting Deepchecks.")
    raise SystemExit(2)

_print_label_summary("Train split", y_action[train_idx], y_finger[train_idx])
_print_label_summary("Test split", y_action[test_idx], y_finger[test_idx])

overall_action_unique = len(np.unique(y_action))
action_train_unique = len(np.unique(y_train)) if len(y_train) else 0
action_test_unique = len(np.unique(y_test)) if len(y_test) else 0
finger_train_unique = _unique_non_rest_fingers(y_action[train_idx], y_finger[train_idx])
finger_test_unique = _unique_non_rest_fingers(y_action[test_idx], y_finger[test_idx])

if overall_action_unique < 2:
    print("⚠️ Action labels are single-class overall. Aborting Deepchecks.")
    raise SystemExit(2)
if action_train_unique < 2 or action_test_unique < 2:
    print("⚠️ Action labels collapsed in train/test split. Aborting Deepchecks.")
    raise SystemExit(2)

assert action_train_unique > 1, "Action target collapsed in train set."
assert action_test_unique > 1, "Action target collapsed in test set."

if finger_train_unique < 2 or finger_test_unique < 2:
    print(
        "⚠️ Finger labels collapsed in train/test split; Deepchecks will ignore finger labels."
    )

# =========================
# ===== REUSE SCALER ======
# =========================

normalizer = load_normalizer(scaler_path)
if normalizer is None:
    print(f"Failed to load normalizer: {scaler_path}")
    raise SystemExit(2)
X_train = np.ascontiguousarray(X_train, dtype=np.float32)
X_test = np.ascontiguousarray(X_test, dtype=np.float32)
apply_saved_channel_normalizer(X_train, normalizer, out=X_train)
apply_saved_channel_normalizer(X_test, normalizer, out=X_test)
# =========================
# ===== TABULAR SUMMARY ===
# =========================
# Deepchecks needs tabular data; we summarize windows here but
# run the model on normalized window tensors via synthetic lookup indices.

X_lookup = np.ascontiguousarray(np.concatenate([X_train, X_test], axis=0), dtype=np.float32)
train_lookup_ids = np.arange(len(X_train), dtype=np.int64)
test_lookup_ids = np.arange(len(X_train), len(X_train) + len(X_test), dtype=np.int64)

train_df = summarize_windows(X_train)
test_df = summarize_windows(X_test)
train_df, y_train, train_lookup_ids = _prepare_deepchecks_split(
    train_df, y_train, train_lookup_ids, int(split_seed) + 101, index_start=0
)
test_df, y_test, test_lookup_ids = _prepare_deepchecks_split(
    test_df,
    y_test,
    test_lookup_ids,
    int(split_seed) + 202,
    index_start=len(train_df),
)
train_labels = pd.Series(y_train, index=train_df.index)
test_labels = pd.Series(y_test, index=test_df.index)

assert len(train_df) == len(y_train) == len(X_train)
assert len(test_df) == len(y_test) == len(X_test)
assert y_action.min() >= 0
assert set(np.unique(y_action)).issubset(set(ACTION_NAMES.keys()))
assert max(ACTION_NAMES.keys()) >= int(y_action.max())

feature_names = list(train_df.columns)
class_names = [ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]
assert_identifier_not_in_X(meta, feature_names)
print(f"Deepchecks tabular features: {', '.join(feature_names)}")
feature_row_lookup = _build_feature_row_lookup(
    train_df, train_lookup_ids, feature_names
)
for key, values in _build_feature_row_lookup(
    test_df, test_lookup_ids, feature_names
).items():
    feature_row_lookup.setdefault(key, []).extend(values)

train_ds = Dataset(
    train_df,
    label=train_labels,
    features=feature_names,
    label_type="multiclass",
    label_classes=class_names,
    cat_features=[],
    **_dataset_kwargs(),
)

test_ds = Dataset(
    test_df,
    label=test_labels,
    features=feature_names,
    label_type="multiclass",
    label_classes=class_names,
    cat_features=[],
    **_dataset_kwargs(),
)

# =========================
# ===== MODEL (MATCH STEP 2)
# =========================

state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
n_fingers, n_actions, has_applicability_head = infer_output_dims_from_state_dict(
    state_dict
)

model = CNNLSTMFingerActionNet(
    n_channels=X.shape[2],
    n_fingers=n_fingers,
    n_actions=n_actions,
    finger_applicability_head=bool(has_applicability_head),
)
model.load_state_dict(state_dict)
model.to(device)
model.eval()


class TorchModelWrapper:
    """
    Deepchecks-compatible wrapper.
    Exposes sklearn-like predict / predict_proba over deterministic action logits.
    """

    def __init__(self):
        self.classes_ = np.asarray(sorted(ACTION_NAMES.keys()), dtype=np.int64)
        self._all_probs: Optional[np.ndarray] = None
        self._timing = InferenceTiming()

    def _window_idx(self, X_tabular) -> np.ndarray:
        try:
            return _extract_window_idx(X_tabular)
        except KeyError:
            return _lookup_window_idx_from_features(
                X_tabular, feature_names, feature_row_lookup
            )

    def _ensure_all_probs(self) -> np.ndarray:
        if self._all_probs is not None:
            return self._all_probs

        n_windows = int(len(X_lookup))
        if n_windows == 0:
            self._all_probs = np.zeros((0, n_actions), dtype=np.float32)
            return self._all_probs

        batch_size = max(1, min(int(args.batch_size), n_windows))
        probs_out = np.empty((n_windows, n_actions), dtype=np.float32)
        device_batch: Optional[torch.Tensor] = None
        self._timing.calls += 1
        self._timing.windows += n_windows
        total_start = time.perf_counter()

        with torch.inference_mode():
            for start in range(0, n_windows, batch_size):
                end = min(start + batch_size, n_windows)
                host_view = X_lookup[start:end]
                host_tensor = torch.from_numpy(host_view)
                if device.type == "cpu":
                    batch_tensor = host_tensor
                else:
                    if (
                        device_batch is None
                        or device_batch.shape[1:] != host_tensor.shape[1:]
                        or device_batch.shape[0] < host_tensor.shape[0]
                    ):
                        device_batch = torch.empty(
                            (batch_size, *host_tensor.shape[1:]),
                            dtype=torch.float32,
                            device=device,
                        )
                    transfer_start = time.perf_counter()
                    device_batch[: host_tensor.shape[0]].copy_(host_tensor)
                    batch_tensor = device_batch[: host_tensor.shape[0]]
                    _device_synchronize(device)
                    self._timing.transfer_sec += time.perf_counter() - transfer_start
                model_start = time.perf_counter()
                with _autocast_context(device, amp_dtype):
                    _, action_logits, _ = unpack_model_outputs(model(batch_tensor))
                    probs = torch.softmax(action_logits, dim=1)
                _device_synchronize(device)
                self._timing.model_sec += time.perf_counter() - model_start
                probs_out[start:end] = probs.float().cpu().numpy()
                self._timing.batches += 1

        self._timing.total_sec += time.perf_counter() - total_start
        self._all_probs = probs_out
        return self._all_probs

    def predict_proba(self, X_tabular):
        window_idx = self._window_idx(X_tabular)
        if (window_idx < 0).any() or (window_idx >= len(X_lookup)).any():
            raise ValueError("window_idx contains out-of-bounds indices for X_lookup.")
        return self._ensure_all_probs()[window_idx]

    def predict(self, X_tabular):
        probs = self.predict_proba(X_tabular)
        class_pos = np.argmax(probs, axis=1)
        return self.classes_[class_pos]


print("🔍 Running Deepchecks suites...")

suite = _build_eeg_deepchecks_suite()
wrapped_model = TorchModelWrapper()
result = suite.run(train_ds, test_ds, model=wrapped_model)

out_dir = Path(paths.eval_dir)
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "deepchecks_eeg_report.html"
result.save_as_html(out_path.as_posix(), as_widget=False)
latest_html = max(
    out_dir.glob("deepchecks_eeg_report*.html"),
    key=lambda path: path.stat().st_mtime,
)
if latest_html != out_path:
    out_path.write_text(latest_html.read_text(errors="ignore"))
print(
    "Inference timing: "
    f"calls={wrapped_model._timing.calls}, "
    f"windows={wrapped_model._timing.windows}, "
    f"batches={wrapped_model._timing.batches}, "
    f"transfer={wrapped_model._timing.transfer_sec:.2f}s, "
    f"model={wrapped_model._timing.model_sec:.2f}s, "
    f"total={wrapped_model._timing.total_sec:.2f}s"
)
print(f"Saving report to: {out_path}")
print(f"✅ Deepchecks report saved: {out_path}")
