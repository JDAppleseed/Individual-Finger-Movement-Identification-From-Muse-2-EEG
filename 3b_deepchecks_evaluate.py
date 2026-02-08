"""
STEP 3b — Deepchecks Evaluation (SDS-aligned)
Deterministic model behavior (Dropout OFF)
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import (
    data_integrity,
    train_test_validation,
    model_evaluation,
)

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import ACTION_NAMES, ACTION_REST, FINGER_NAMES
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    apply_channel_normalizer,
    summarize_windows,
)
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir

# =========================
# ===== CONFIG ============
# =========================

SEED = 42
MIN_TEST_SAMPLES = 30
MAX_SPLIT_ATTEMPTS = 8
DEFAULT_BATCH_SIZE = 256

# Deterministic setup
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


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


def _split_with_checks(y_action, y_finger, meta, seed: int):
    overall_action_unique = len(np.unique(y_action))
    overall_finger_unique = _unique_non_rest_fingers(y_action, y_finger)

    for attempt in range(MAX_SPLIT_ATTEMPTS):
        train_idx, test_idx = split_indices(
            y_action,
            y_finger,
            meta=meta,
            test_size=0.2,
            random_state=seed + attempt * 11,
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
    try:
        import inspect

        if "index_name" in inspect.signature(Dataset).parameters:
            return {"index_name": "window_idx"}
    except Exception:
        pass
    return {}


# =========================
# ===== LOAD DATA =========
# =========================

def _resolve_path(path_str: str, base_dir: Optional[Path]) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        base = base_dir if base_dir is not None else Path.cwd()
        candidate = (base / candidate).resolve()
    return candidate


parser = argparse.ArgumentParser()
parser.add_argument(
    "--session-dir",
    type=str,
    default=None,
    help="Canonical session directory (resolves npz + latest model run by default).",
)
parser.add_argument(
    "--run-dir",
    type=str,
    default=None,
    help="Model run directory (defaults to latest under <session_dir>/processed/models/).",
)
parser.add_argument(
    "--npz", type=str, default="eeg_windows.npz", help="Sequence npz file"
)
parser.add_argument(
    "--model", type=str, default="finger_action_model.pt", help="Model weights path"
)
parser.add_argument(
    "--scaler", type=str, default="scaler.save", help="Normalizer/scaler path"
)
parser.add_argument(
    "--max-samples", type=int, default=None, help="Limit samples for Deepchecks"
)
parser.add_argument(
    "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size"
)
args = parser.parse_args()

explicit_npz = args.npz not in ("eeg_windows.npz", "./eeg_windows.npz")
explicit_model = args.model not in ("finger_action_model.pt", "./finger_action_model.pt")
explicit_scaler = args.scaler not in ("scaler.save", "./scaler.save")
explicit_run_dir = bool(getattr(args, "run_dir", None))
explicit_overrides = [
    name
    for name, is_explicit in [
        ("run_dir", explicit_run_dir),
        ("npz", explicit_npz),
        ("model", explicit_model),
        ("scaler", explicit_scaler),
    ]
    if is_explicit
]

out_dir_override: Optional[Path] = None
selection_source = "legacy_explicit"
if getattr(args, "session_dir", None):
    session_dir_path = resolve_session_dir(str(args.session_dir))
    if not session_dir_path.exists():
        print("Session selection source: session_dir")
        print(f"Session dir not found: {session_dir_path}")
        raise SystemExit(2)
    base_dir = session_dir_path
    if explicit_overrides:
        print(
            f"⚠️ Explicit paths provided with --session-dir; using overrides: {explicit_overrides}"
        )
        selection_source = "legacy_explicit"
    else:
        selection_source = "session_dir"
    run_dir_path = (
        Path(str(args.run_dir)).expanduser()
        if explicit_run_dir
        else resolve_latest_run_dir(session_dir_path)
    )
    if run_dir_path is None or not run_dir_path.exists():
        print("Session selection source: session_dir")
        print(
            "No model run directory found. Train a model first (Step 2), or pass --run-dir explicitly."
        )
        raise SystemExit(2)
    layout = SessionLayout(session_dir_path)
    if explicit_npz:
        args.npz = str(_resolve_path(args.npz, base_dir))
    else:
        args.npz = str(layout.windows_npz)
    if explicit_model:
        args.model = str(_resolve_path(args.model, base_dir))
    else:
        args.model = str(run_dir_path / "finger_action_model.pt")
    if explicit_scaler:
        args.scaler = str(_resolve_path(args.scaler, base_dir))
    else:
        args.scaler = str(run_dir_path / "scaler.save")
    out_dir_override = layout.reports_root / run_dir_path.name
else:
    base_dir = Path.cwd()
    if not explicit_npz:
        print("Session selection source: legacy_explicit")
        print("❌ Missing --session-dir. Provide --session-dir or explicit --npz PATH.")
        raise SystemExit(2)
    if explicit_run_dir:
        run_dir_path = Path(str(args.run_dir)).expanduser()
        if not explicit_model:
            args.model = str(run_dir_path / "finger_action_model.pt")
        if not explicit_scaler:
            args.scaler = str(run_dir_path / "scaler.save")
    else:
        if not explicit_model or not explicit_scaler:
            print("Session selection source: legacy_explicit")
            print(
                "❌ Missing --session-dir. Provide explicit --model and --scaler (or --run-dir)."
            )
            raise SystemExit(2)
    args.npz = str(_resolve_path(args.npz, base_dir))
    if explicit_model:
        args.model = str(_resolve_path(args.model, base_dir))
    if explicit_scaler:
        args.scaler = str(_resolve_path(args.scaler, base_dir))

print(f"Session selection source: {selection_source}")
print(f"Using NPZ file: {args.npz}")
print(f"Using model file: {args.model}")
print(f"Using scaler file: {args.scaler}")

if not Path(args.npz).exists():
    print(f"NPZ file not found: {args.npz}")
    raise SystemExit(2)
if not Path(args.scaler).exists():
    print(f"Scaler file not found: {args.scaler}")
    raise SystemExit(2)
if not Path(args.model).exists():
    print(f"Model file not found: {args.model}")
    raise SystemExit(2)

X, y_action, y_finger, meta = load_sequence_npz(args.npz, mmap_mode="r")
if isinstance(X, np.memmap) and X.dtype != np.float32:
    print(f"ℹ️ X dtype is {X.dtype}; casting to float32 per batch.")

X, y_action, y_finger, meta = _apply_sample_limit(
    X, y_action, y_finger, meta, args.max_samples, SEED
)

_print_label_summary("Filtered", y_action, y_finger)

# =========================
# ===== SAME SPLIT =========
# =========================

train_idx, test_idx = _split_with_checks(y_action, y_finger, meta=meta, seed=SEED)
if train_idx is None or test_idx is None:
    print("⚠️ Unable to create a split with multiple classes. Aborting Deepchecks.")
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

normalizer = joblib.load(args.scaler)
X_train = apply_channel_normalizer(X_train, normalizer)
X_test = apply_channel_normalizer(X_test, normalizer)
# =========================
# ===== TABULAR SUMMARY ===
# =========================
# Deepchecks needs tabular data; we summarize windows here but
# run the model on raw window tensors via window_idx in predict().

train_df = summarize_windows(X_train)
train_df["window_idx"] = train_idx
test_df = summarize_windows(X_test)
test_df["window_idx"] = test_idx

assert "window_idx" in train_df.columns and "window_idx" in test_df.columns
assert test_idx.max() < len(X) and train_idx.max() < len(X)
assert y_action.min() >= 0
assert set(np.unique(y_action)).issubset(set(ACTION_NAMES.keys()))
assert max(ACTION_NAMES.keys()) >= int(y_action.max())

feature_names = [c for c in train_df.columns if c != "window_idx"]
class_names = [ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]

train_ds = Dataset(
    train_df,
    label=y_train,
    features=feature_names,
    label_type="multiclass",
    label_classes=class_names,
    cat_features=[],
    **_dataset_kwargs(),
)

test_ds = Dataset(
    test_df,
    label=y_test,
    features=feature_names,
    label_type="multiclass",
    label_classes=class_names,
    cat_features=[],
    **_dataset_kwargs(),
)

# =========================
# ===== MODEL (MATCH STEP 2)
# =========================

n_fingers = int(y_finger.max()) + 1
n_actions = int(y_action.max()) + 1

model = CNNLSTMFingerActionNet(
    n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions
)
model_path = args.model
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()


class TorchModelWrapper:
    """
    Deepchecks-compatible wrapper.
    Returns deterministic action probabilities.
    """

    def predict(self, X_tabular):
        if "window_idx" not in X_tabular:
            raise KeyError("Deepchecks input is missing required column 'window_idx'.")
        window_idx = X_tabular["window_idx"].to_numpy().astype(np.int64)
        if (window_idx < 0).any() or (window_idx >= len(X)).any():
            raise ValueError("window_idx contains out-of-bounds indices for X.")

        batch_size = max(1, int(args.batch_size))
        probs_out = np.zeros((len(window_idx), n_actions), dtype=np.float32)
        device = next(model.parameters()).device

        with torch.no_grad():
            for start in range(0, len(window_idx), batch_size):
                end = min(start + batch_size, len(window_idx))
                batch_idx = window_idx[start:end]
                X_batch = np.asarray(X[batch_idx], dtype=np.float32)
                X_batch = apply_channel_normalizer(X_batch, normalizer)
                X_tensor = torch.tensor(X_batch, dtype=torch.float32, device=device)
                _, action_logits = model(X_tensor)
                probs = torch.softmax(action_logits, dim=1)
                probs_out[start:end] = probs.detach().cpu().numpy()
        return probs_out


print("🔍 Running Deepchecks suites...")

suite = data_integrity().add(train_test_validation()).add(model_evaluation())

result = suite.run(train_ds, test_ds, model=TorchModelWrapper())

out_dir = out_dir_override or Path("reports") / "subjects" / "UNKNOWN" / "UNKNOWN"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "deepchecks_eeg_report.html"
result.save_as_html(out_path.as_posix(), as_widget=False)
print(f"Saving report to: {out_path}")
print(f"✅ Deepchecks report saved: {out_path}")
