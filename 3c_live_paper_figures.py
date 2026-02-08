"""
STEP 3c — Interactive Evaluation Figures (SDS-aligned)
Includes confidence & uncertainty visualization
"""

import argparse
import os
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from typing import Optional

from sklearn.metrics import confusion_matrix, accuracy_score

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import ACTION_REST, ACTION_NAMES, FINGER_NAMES
from utils.experiment_logger import get_latest_experiment_hash, LOG_DIR
from utils.per_subject_calibration import plot_subject_calibration
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    apply_channel_normalizer,
)

# =========================
# ===== CONFIG ============
# =========================

MC_SAMPLES = 30
SEED = 42
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

# =========================
# ===== LOAD DATA =========
# =========================

parser = argparse.ArgumentParser()
parser.add_argument("--session-dir", type=str, default=None)
parser.add_argument("--run-dir", type=str, default=None)
parser.add_argument("--npz", type=str, default="eeg_windows.npz")
parser.add_argument("--model", type=str, default="finger_action_model.pt")
parser.add_argument("--scaler", type=str, default="scaler.save")
parser.add_argument(
    "--out-dir",
    type=str,
    default=None,
    help="Directory for figure outputs (defaults to <session_dir>/processed/reports/<run_id>/).",
)
args = parser.parse_args()

def _resolve_path(path_str: str, base_dir: Optional[Path]) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        base = base_dir if base_dir is not None else Path.cwd()
        candidate = (base / candidate).resolve()
    return candidate


out_dir_override = Path(args.out_dir).expanduser() if args.out_dir else None
explicit_npz = args.npz not in ("eeg_windows.npz", "./eeg_windows.npz")
explicit_model = args.model not in ("finger_action_model.pt", "./finger_action_model.pt")
explicit_scaler = args.scaler not in ("scaler.save", "./scaler.save")
explicit_run_dir = bool(args.run_dir)
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

session_dir_path = resolve_session_dir(args.session_dir) if args.session_dir else None
run_dir_path = None
selection_source = "legacy_explicit"
if session_dir_path:
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
        Path(args.run_dir).expanduser()
        if explicit_run_dir
        else resolve_latest_run_dir(session_dir_path)
    )
    if run_dir_path is None or not run_dir_path.exists():
        print("Session selection source: session_dir")
        print(
            "No model run directory found. Train a model first (Step 2), or pass --run-dir."
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
    if out_dir_override is None:
        out_dir_override = layout.reports_root / run_dir_path.name
else:
    base_dir = Path.cwd()
    if not explicit_npz:
        print("Session selection source: legacy_explicit")
        print("❌ Missing --session-dir. Provide --session-dir or explicit --npz PATH.")
        raise SystemExit(2)
    if explicit_run_dir:
        run_dir_path = Path(args.run_dir).expanduser()
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

npz_path = Path(args.npz).expanduser()
model_path = Path(args.model).expanduser()
scaler_path = Path(args.scaler).expanduser()
report_dir = out_dir_override or Path("reports/subjects")
report_dir.mkdir(parents=True, exist_ok=True)
run_tag = run_dir_path.name if run_dir_path is not None else None
print(f"Saving figures to: {report_dir}")

X, y_action, y_finger, meta = load_sequence_npz(str(npz_path))

subject_ids = meta.get("subject_id")
experiment_hashes = meta.get("experiment_hash")
exp_hash = str(experiment_hashes[0]) if experiment_hashes is not None else "UNKNOWN"

# =========================
# ===== SAME SPLIT =========
# =========================

train_idx, test_idx = split_indices(
    y_action, y_finger, meta=meta, test_size=0.2, random_state=SEED
)
X_test = X[test_idx]
y_action_test = y_action[test_idx]
y_finger_test = y_finger[test_idx]

# =========================
# ===== SCALE (REUSE) =====
# =========================

if not scaler_path.exists():
    print(f"Scaler file not found: {scaler_path}")
    raise SystemExit(2)
normalizer = joblib.load(str(scaler_path))
X_test = apply_channel_normalizer(X_test, normalizer)

# =========================
# ===== MODEL (MATCH STEP 2)
# =========================

n_fingers = int(y_finger.max()) + 1
n_actions = int(y_action.max()) + 1

model = CNNLSTMFingerActionNet(
    n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions
)
if not model_path.exists():
    print(f"Model file not found: {model_path}")
    raise SystemExit(2)
model.load_state_dict(torch.load(str(model_path), map_location="cpu"))

# =========================
# ===== MC DROPOUT =========
# =========================

X_t = torch.tensor(X_test, dtype=torch.float32)
mc = model.mc_forward(X_t, passes=MC_SAMPLES)

action_mean = mc["action_mean"].cpu().numpy()
finger_mean = mc["finger_mean"].cpu().numpy()

action_std = mc["action_std"].cpu().numpy()
finger_std = mc["finger_std"].cpu().numpy()

action_preds = np.argmax(action_mean, axis=1)
action_conf = np.max(action_mean, axis=1)
action_uncertainty = np.mean(action_std, axis=1)

finger_preds = np.argmax(finger_mean, axis=1)
finger_conf = np.max(finger_mean, axis=1)
finger_uncertainty = np.mean(finger_std, axis=1)

# =========================
# ===== METRICS ===========
# =========================


def reliability_bins(conf, preds, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_accs = []
    for i in range(n_bins):
        idx = (conf > bins[i]) & (conf <= bins[i + 1])
        if np.sum(idx) == 0:
            bin_accs.append(np.nan)
        else:
            bin_accs.append(np.mean(preds[idx] == labels[idx]))
    return bin_centers, bin_accs


action_acc = accuracy_score(y_action_test, action_preds)
mask = y_action_test != ACTION_REST
finger_acc = (
    accuracy_score(y_finger_test[mask], finger_preds[mask]) if mask.any() else 0.0
)

print(f"\n🎯 Action Accuracy: {action_acc * 100:.2f}%")
print(f"🎯 Finger Accuracy (non-REST): {finger_acc * 100:.2f}%")

# =========================
# ===== PLOTS =============
# =========================

fig, axs = plt.subplots(3, 2, figsize=(14, 14))

# --- Action Confusion Matrix ---
action_cm = confusion_matrix(y_action_test, action_preds)
action_labels = [ACTION_NAMES[i] for i in sorted(ACTION_NAMES.keys())]
sns.heatmap(
    action_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=action_labels,
    yticklabels=action_labels,
    ax=axs[0, 0],
)
axs[0, 0].set_title("Action Confusion Matrix")
axs[0, 0].set_xlabel("Predicted")
axs[0, 0].set_ylabel("True")

# --- Finger Confusion Matrix (non-REST) ---
if mask.any():
    finger_label_ids = [i for i in sorted(FINGER_NAMES.keys()) if i != 0]
    finger_cm = confusion_matrix(
        y_finger_test[mask], finger_preds[mask], labels=finger_label_ids
    )
    finger_labels = [FINGER_NAMES[i] for i in finger_label_ids]
    sns.heatmap(
        finger_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=finger_labels,
        yticklabels=finger_labels,
        ax=axs[0, 1],
    )
    axs[0, 1].set_title("Finger Confusion Matrix (non-REST)")
    axs[0, 1].set_xlabel("Predicted")
    axs[0, 1].set_ylabel("True")
else:
    axs[0, 1].set_axis_off()

# --- Action Confidence/Uncertainty ---
axs[1, 0].hist(action_conf, bins=20, alpha=0.8, label="Confidence")
axs[1, 0].hist(action_uncertainty, bins=20, alpha=0.6, label="Uncertainty")
axs[1, 0].set_title("Action Confidence vs Uncertainty")
axs[1, 0].legend()

# --- Finger Confidence/Uncertainty (non-REST) ---
axs[1, 1].hist(finger_conf[mask], bins=20, alpha=0.8, label="Confidence")
axs[1, 1].hist(finger_uncertainty[mask], bins=20, alpha=0.6, label="Uncertainty")
axs[1, 1].set_title("Finger Confidence vs Uncertainty")
axs[1, 1].legend()

# --- Reliability Diagram (Action) ---
bin_centers, bin_accs = reliability_bins(
    action_conf, action_preds, y_action_test, n_bins=10
)
axs[2, 0].plot([0, 1], [0, 1], "--", color="gray")
axs[2, 0].bar(bin_centers, bin_accs, width=0.08, alpha=0.7)
axs[2, 0].set_title("Action Reliability Diagram")
axs[2, 0].set_xlabel("Confidence")
axs[2, 0].set_ylabel("Accuracy")

# --- Reliability Diagram (Finger, non-REST) ---
if mask.any():
    f_bin_centers, f_bin_accs = reliability_bins(
        finger_conf[mask], finger_preds[mask], y_finger_test[mask], n_bins=10
    )
    axs[2, 1].plot([0, 1], [0, 1], "--", color="gray")
    axs[2, 1].bar(f_bin_centers, f_bin_accs, width=0.08, alpha=0.7, color="green")
    axs[2, 1].set_title("Finger Reliability (non-REST)")
    axs[2, 1].set_xlabel("Confidence")
    axs[2, 1].set_ylabel("Accuracy")
else:
    axs[2, 1].set_axis_off()

plt.tight_layout()

tag = run_tag or exp_hash
fig_path = report_dir / f"mc_eval_{tag}.png"
plt.savefig(fig_path)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# Optional confidence vs uncertainty scatter
plt.figure(figsize=(6, 5))
plt.scatter(action_conf, action_uncertainty, alpha=0.5, s=10)
plt.xlabel("Action Confidence")
plt.ylabel("Action Uncertainty")
plt.title("Confidence vs Uncertainty")
scatter_path = report_dir / f"mc_scatter_{tag}.png"
plt.tight_layout()
plt.savefig(scatter_path)
plt.close()

# --- PER-SUBJECT CALIBRATION PLOT ---
try:
    exp_hash = get_latest_experiment_hash()
    logs = json.loads((LOG_DIR / f"{exp_hash}.json").read_text())
    subject_id = logs["subject_id"]
    plot_subject_calibration(subject_id=subject_id, experiment_hash=exp_hash)
except Exception as e:
    print(f"⚠️ Calibration plot skipped: {e}")
