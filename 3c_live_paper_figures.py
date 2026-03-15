"""
STEP 3c — Interactive Evaluation Figures (SDS-aligned)
Includes confidence & uncertainty visualization
"""

import argparse
import os
import json
from dataclasses import dataclass
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple

from sklearn.metrics import confusion_matrix, accuracy_score

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import (
    ACTION_REST,
    ACTION_NAMES,
    FINGER_NAMES,
    enforce_prediction_pairs,
)
from utils.experiment_logger import get_latest_experiment_hash, LOG_DIR
from utils.per_subject_calibration import plot_subject_calibration
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    apply_channel_normalizer,
)
from utils.runtime_utils import load_normalizer

# Fixed label ordering for standardized confusion matrices
ACTION_LABELS = sorted(ACTION_NAMES.keys())
ACTION_TICK_LABELS = [ACTION_NAMES[label] for label in ACTION_LABELS]
FINGER_LABELS = sorted(FINGER_NAMES.keys())
FINGER_TICK_LABELS = [FINGER_NAMES[label] for label in FINGER_LABELS]

# Pipeline handoff: Step 3c reuses Step 2 model/scaler artifacts and produces
# figure outputs that align with Step 3 evaluation semantics.
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

MC_SAMPLES = 30
SEED = 42
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

# =========================
# ===== LOAD DATA =========
# =========================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run-dir",
    type=str,
    default=None,
    help="Model run directory override (e.g. .../processed/models/<run_id>).",
)
parser.add_argument(
    "--project",
    type=str,
    required=False,
    help="Project identifier",
)
parser.add_argument("--subject", type=str, required=False, help="Subject identifier")
parser.add_argument(
    "--session",
    type=str,
    required=False,
    help="Session identifier (defaults to latest for subject)",
)
parser.add_argument(
    "--session-dir",
    type=str,
    default=None,
    help="Legacy session directory override from UI",
)
args = parser.parse_args()

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
if not npz_path.exists():
    print(f"NPZ file not found: {npz_path}")
    raise SystemExit(2)
report_dir = Path(paths.eval_dir)
report_dir.mkdir(parents=True, exist_ok=True)
run_tag = str(args.session)
print(f"Saving figures to: {report_dir}")

X, y_action, y_finger, meta = load_sequence_npz(str(npz_path))

subject_ids = meta.get("subject_id")
experiment_hashes = meta.get("experiment_hash")
exp_hash = str(experiment_hashes[0]) if experiment_hashes is not None else "UNKNOWN"

# =========================
# ===== SAME SPLIT =========
# =========================

# Use the same seeded split helper as training/eval to keep figure metrics comparable.
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
normalizer = load_normalizer(scaler_path)
if normalizer is None:
    print(f"Failed to load normalizer: {scaler_path}")
    raise SystemExit(2)
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
model.load_state_dict(torch.load(str(model_path), map_location="cpu", weights_only=True))

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

_, finger_preds = enforce_prediction_pairs(action_preds, np.argmax(finger_mean, axis=1))
finger_conf = finger_mean[np.arange(len(finger_mean)), finger_preds]
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
action_cm = confusion_matrix(y_action_test, action_preds, labels=ACTION_LABELS)
action_labels = ACTION_TICK_LABELS
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
    finger_cm = confusion_matrix(
        y_finger_test[mask], finger_preds[mask], labels=FINGER_LABELS
    )
    finger_labels = FINGER_TICK_LABELS
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
