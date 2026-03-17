"""
STEP 3c — Full-Dataset Hero Figures
Board-facing visual summary over the entire combined dataset using the saved model.
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
from scipy.ndimage import gaussian_filter

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
    apply_channel_normalizer,
)
from utils.runtime_utils import (
    apply_temperature_to_logits,
    load_normalizer,
    load_temperature_scaling,
)

# Fixed label ordering for standardized confusion matrices
ACTION_LABELS = sorted(ACTION_NAMES.keys())
ACTION_TICK_LABELS = [ACTION_NAMES[label] for label in ACTION_LABELS]
FINGER_LABELS = sorted(FINGER_NAMES.keys())
FINGER_TICK_LABELS = [FINGER_NAMES[label] for label in FINGER_LABELS]

# Pipeline handoff: Step 3c reuses Step 2 model/scaler artifacts but serves a
# different purpose than Step 3. This script produces a broad, board-ready
# visual summary across the full dataset instead of the strict canonical split.
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
BATCH_SIZE = 256
SEED = 42
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

# =========================
# ===== LOAD DATA =========
# =========================

parser = argparse.ArgumentParser(
    description=(
        "Step 3c: generate interactive and paper-style evaluation figures from "
        "a trained Step 2 run and its Step 3 outputs."
    )
)
selection_group = parser.add_argument_group("input selection")
selection_group.add_argument(
    "--run-dir",
    type=str,
    default=None,
    metavar="PATH",
    help="Specific Step 2 run directory to visualize (for example: .../processed/models/<run_id>).",
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
    help="Session identifier to visualize. Defaults to the latest session for the subject.",
)
selection_group.add_argument(
    "--session-dir",
    type=str,
    default=None,
    metavar="PATH",
    help="Legacy session directory override used by the UI.",
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
# ===== FULL DATASET ======
# =========================

X_eval = X
y_action_eval = y_action
y_finger_eval = y_finger
session_id_eval = (
    np.asarray(meta.get("session_id")).astype("U")
    if meta.get("session_id") is not None
    else np.array(["unknown"] * len(y_action_eval), dtype="U")
)

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
X_eval = apply_channel_normalizer(X_eval, normalizer)
temperature_state = load_temperature_scaling(model_path.parent / "temperature_scaling.json")

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

def _mc_predict_dataset(X_arr: np.ndarray):
    action_mean_parts = []
    finger_mean_parts = []
    action_std_parts = []
    finger_std_parts = []

    was_training = model.training
    model.train()
    with torch.no_grad():
        for start in range(0, len(X_arr), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(X_arr))
            xb = torch.tensor(X_arr[start:end], dtype=torch.float32)
            action_passes = []
            finger_passes = []
            for _ in range(MC_SAMPLES):
                finger_logits, action_logits = model(xb)
                if temperature_state is not None:
                    action_logits = apply_temperature_to_logits(
                        action_logits, temperature_state.action_temperature
                    )
                    finger_logits = apply_temperature_to_logits(
                        finger_logits, temperature_state.finger_temperature
                    )
                action_passes.append(torch.softmax(action_logits, dim=1))
                finger_passes.append(torch.softmax(finger_logits, dim=1))
            action_stack = torch.stack(action_passes, dim=0)
            finger_stack = torch.stack(finger_passes, dim=0)
            action_mean_parts.append(action_stack.mean(dim=0).cpu().numpy())
            finger_mean_parts.append(finger_stack.mean(dim=0).cpu().numpy())
            action_std_parts.append(action_stack.std(dim=0).cpu().numpy())
            finger_std_parts.append(finger_stack.std(dim=0).cpu().numpy())
    if not was_training:
        model.eval()

    return (
        np.concatenate(action_mean_parts, axis=0),
        np.concatenate(finger_mean_parts, axis=0),
        np.concatenate(action_std_parts, axis=0),
        np.concatenate(finger_std_parts, axis=0),
    )


action_mean, finger_mean, action_std, finger_std = _mc_predict_dataset(X_eval)

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


action_acc = accuracy_score(y_action_eval, action_preds)
mask = y_action_eval != ACTION_REST
finger_acc = (
    accuracy_score(y_finger_eval[mask], finger_preds[mask]) if mask.any() else 0.0
)

rest_only_mask = y_action_eval == ACTION_REST
rest_recall = float(np.mean(action_preds[rest_only_mask] == ACTION_REST)) if np.any(rest_only_mask) else 0.0
unique_sessions = sorted(set(session_id_eval.tolist()))

print(f"\n📷 Full-dataset visual summary")
print(f"🎯 Action Accuracy (all windows): {action_acc * 100:.2f}%")
print(f"🎯 Finger Accuracy (non-REST, all windows): {finger_acc * 100:.2f}%")
print(f"🎯 REST Recall (all REST windows): {rest_recall * 100:.2f}%")

# =========================
# ===== PLOTS =============
# =========================

fig, axes = plt.subplots(2, 2, figsize=(15.5, 11.5), constrained_layout=True)
fig.patch.set_facecolor("white")
ax_action_cm, ax_finger_cm = axes[0]
ax_action_rel, ax_finger_rel = axes[1]

# --- Action Confusion Matrix ---
action_cm = confusion_matrix(y_action_eval, action_preds, labels=ACTION_LABELS)
action_labels = ACTION_TICK_LABELS
sns.heatmap(
    action_cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=action_labels,
    yticklabels=action_labels,
    ax=ax_action_cm,
)
ax_action_cm.set_title("Action Confusion Matrix (All Windows)")
ax_action_cm.set_xlabel("Predicted")
ax_action_cm.set_ylabel("True")

# --- Finger Confusion Matrix (non-REST) ---
if mask.any():
    finger_cm = confusion_matrix(
        y_finger_eval[mask], finger_preds[mask], labels=FINGER_LABELS
    )
    finger_labels = FINGER_TICK_LABELS
    sns.heatmap(
        finger_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=finger_labels,
        yticklabels=finger_labels,
        ax=ax_finger_cm,
    )
    ax_finger_cm.set_title("Finger Confusion Matrix (Non-REST)")
    ax_finger_cm.set_xlabel("Predicted")
    ax_finger_cm.set_ylabel("True")
else:
    ax_finger_cm.set_axis_off()

colors = ["#5B8FF9", "#61DDAA", "#F6BD16"]

# --- Reliability Diagram (Action) ---
bin_centers, bin_accs = reliability_bins(
    action_conf, action_preds, y_action_eval, n_bins=10
)
ax_action_rel.plot([0, 1], [0, 1], "--", color="gray")
ax_action_rel.bar(bin_centers, bin_accs, width=0.08, alpha=0.7)
ax_action_rel.set_title("Action Reliability")
ax_action_rel.set_xlabel("Confidence")
ax_action_rel.set_ylabel("Accuracy")

# --- Reliability Diagram (Finger, non-REST) ---
if mask.any():
    f_bin_centers, f_bin_accs = reliability_bins(
        finger_conf[mask], finger_preds[mask], y_finger_eval[mask], n_bins=10
    )
    ax_finger_rel.plot([0, 1], [0, 1], "--", color="gray")
    ax_finger_rel.bar(f_bin_centers, f_bin_accs, width=0.08, alpha=0.7, color="green")
    ax_finger_rel.set_title("Finger Reliability (Non-REST)")
    ax_finger_rel.set_xlabel("Confidence")
    ax_finger_rel.set_ylabel("Accuracy")
else:
    ax_finger_rel.set_axis_off()

tag = run_tag or exp_hash
fig_path = report_dir / f"mc_eval_{tag}.png"
plt.savefig(fig_path, dpi=200)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()

# Optional supplemental density-colored scatter export
fig, ax = plt.subplots(figsize=(7.6, 6.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("#f8fafc")
x = np.asarray(action_conf, dtype=float)
y = np.asarray(action_uncertainty, dtype=float)

# Estimate local density with a finer smoothed 2D histogram, then remap the
# color scale so lower-density regions retain visible structure.
x_bins = np.linspace(max(0.0, float(np.min(x))), min(1.0, float(np.max(x))), 80)
y_min = float(np.min(y))
y_max = float(np.max(y))
if not np.isfinite(y_min) or not np.isfinite(y_max) or y_min == y_max:
    y_min, y_max = 0.0, 1.0
y_bins = np.linspace(y_min, y_max, 80)
hist, x_edges, y_edges = np.histogram2d(x, y, bins=(x_bins, y_bins))
hist = gaussian_filter(hist.astype(np.float32), sigma=1.0)
x_idx = np.clip(np.digitize(x, x_edges) - 1, 0, hist.shape[0] - 1)
y_idx = np.clip(np.digitize(y, y_edges) - 1, 0, hist.shape[1] - 1)
density = hist[x_idx, y_idx].astype(float)
raw_color = np.log10(np.maximum(density, 1.0))
color_floor = float(np.nanpercentile(raw_color, 1.0))
color_ceiling = float(np.nanpercentile(raw_color, 99.7))
if not np.isfinite(color_floor):
    color_floor = float(np.nanmin(raw_color))
if not np.isfinite(color_ceiling) or color_ceiling <= color_floor:
    color_ceiling = color_floor + 1.0
color_values = np.clip((raw_color - color_floor) / (color_ceiling - color_floor), 0.0, 1.0)
color_values = np.power(color_values, 0.82) * 3.0
order = np.argsort(color_values)

sc = plt.scatter(
    x[order],
    y[order],
    c=color_values[order],
    cmap="jet",
    alpha=0.58,
    s=9,
    linewidths=0,
)
fig.subplots_adjust(left=0.11, right=0.885, bottom=0.115, top=0.84)
cb = fig.colorbar(sc, ax=ax, pad=0.018, fraction=0.048)
cb.set_label("Relative density", fontsize=11)
cb.ax.tick_params(labelsize=10)

x_low = float(np.nanpercentile(x, 1))
x_high = float(np.nanpercentile(x, 99.8))
y_low = 0.0
y_high = float(np.nanpercentile(y, 99.5))
if not np.isfinite(x_low):
    x_low = 0.35
if not np.isfinite(x_high):
    x_high = 1.0
if not np.isfinite(y_high) or y_high <= 0:
    y_high = float(np.max(y)) if len(y) else 0.35
x_low = max(0.35, x_low - 0.01)
x_high = min(1.0, x_high + 0.005)
y_high = min(max(y_high + 0.01, 0.08), 0.35)

ax.set_xlim(x_low, x_high)
ax.set_ylim(y_low, y_high)
ax.set_xlabel("Action confidence", fontsize=12)
ax.set_ylabel("Action uncertainty", fontsize=12)
ax.tick_params(axis="both", labelsize=10)
ax.grid(True, color="#d7e3f0", linewidth=0.8, alpha=0.5)
ax.set_axisbelow(True)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
ax.spines["left"].set_color("#94a3b8")
ax.spines["bottom"].set_color("#94a3b8")

plot_bbox = ax.get_position()
plot_center_x = (plot_bbox.x0 + plot_bbox.x1) / 2.0

fig.suptitle(
    "Confidence vs Uncertainty",
    fontsize=18,
    fontweight="bold",
    y=0.96,
    x=plot_center_x,
)
ax.set_title(
    "Action predictions across all windows",
    fontsize=12,
    color="#475569",
    pad=12,
    x=0.5,
)

hi_idx = int(np.argmax(color_values)) if len(color_values) else 0
if len(color_values):
    ann_x = float(x[hi_idx])
    ann_y = float(y[hi_idx])
    ax.annotate(
        "Dense region\nHigh confidence, low uncertainty",
        xy=(ann_x, ann_y),
        xytext=(0.37, 0.17),
        textcoords="axes fraction",
        ha="center",
        va="center",
        fontsize=12.5,
        color="#0f172a",
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.36", fc="white", ec="#cbd5e1", alpha=0.97),
    )

scatter_path = report_dir / f"mc_scatter_{tag}.png"
fig.savefig(scatter_path, dpi=180, bbox_inches="tight", pad_inches=0.08)
plt.close(fig)

# Session action composition export
session_names = list(unique_sessions)
counts = np.zeros((len(session_names), len(ACTION_LABELS)), dtype=np.int64)
for row, sid in enumerate(session_names):
    sid_mask = session_id_eval == sid
    counts[row] = np.bincount(y_action_eval[sid_mask], minlength=len(ACTION_LABELS))
plt.figure(figsize=(8, max(3.5, 1.2 * len(session_names))))
bottom = np.zeros(len(session_names), dtype=np.int64)
for idx, label in enumerate(ACTION_LABELS):
    plt.barh(
        session_names,
        counts[:, idx],
        left=bottom,
        color=colors[idx],
        label=ACTION_NAMES[label],
    )
    bottom += counts[:, idx]
plt.title("Session Action Composition")
plt.xlabel("Window Count")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
session_mix_path = report_dir / f"session_action_mix_{tag}.png"
plt.savefig(session_mix_path)
plt.close()

# --- PER-SUBJECT CALIBRATION PLOT ---
try:
    exp_hash = get_latest_experiment_hash()
    logs = json.loads((LOG_DIR / f"{exp_hash}.json").read_text())
    subject_id = logs["subject_id"]
    plot_subject_calibration(subject_id=subject_id, experiment_hash=exp_hash)
except Exception as e:
    print(f"⚠️ Calibration plot skipped: {e}")
