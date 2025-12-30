"""
STEP 3 — Model Evaluation + Confidence Calibration
ISEF / Paper-Ready
(Deterministic evaluation — Dropout OFF)
"""

import os
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from pathlib import Path

from sklearn.metrics import confusion_matrix, accuracy_score

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import ACTION_REST, ACTION_NAMES, FINGER_NAMES
from utils.sequence_data import load_sequence_npz, split_indices, apply_channel_normalizer

# =========================
# ===== CONFIG ============
# =========================

N_BINS = 10
SEED = 42
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"

# =========================
# ===== LOAD DATA =========
# =========================

X, y_action, y_finger, meta = load_sequence_npz("eeg_windows.npz")

subject_ids = meta.get("subject_id")
experiment_hashes = meta.get("experiment_hash")
exp_hash = str(experiment_hashes[0]) if experiment_hashes is not None else "UNKNOWN"

# =========================
# ===== SAME SPLIT =========
# =========================

train_idx, test_idx = split_indices(y_action, y_finger, meta=meta, test_size=0.2, random_state=SEED)
X_test = X[test_idx]
y_action_test = y_action[test_idx]
y_finger_test = y_finger[test_idx]

# =========================
# ===== SCALE (REUSE) =====
# =========================

normalizer = joblib.load("scaler.save")
X_test = apply_channel_normalizer(X_test, normalizer)

# =========================
# ===== MODEL (MATCH STEP 2)
# =========================

n_fingers = int(y_finger.max()) + 1
n_actions = int(y_action.max()) + 1

model = CNNLSTMFingerActionNet(n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions)
model_path = "finger_action_model.pt"
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# =========================
# ===== INFERENCE =========
# =========================

with torch.no_grad():
    X_t = torch.tensor(X_test, dtype=torch.float32)
    finger_logits, action_logits = model(X_t)
    action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()
    finger_probs = torch.softmax(finger_logits, dim=1).cpu().numpy()

    action_preds = np.argmax(action_probs, axis=1)
    action_conf = np.max(action_probs, axis=1)

    finger_preds = np.argmax(finger_probs, axis=1)
    finger_conf = np.max(finger_probs, axis=1)

# =========================
# ===== METRICS ===========
# =========================

action_acc = accuracy_score(y_action_test, action_preds)
mask = y_action_test != ACTION_REST
finger_acc = accuracy_score(y_finger_test[mask], finger_preds[mask]) if mask.any() else 0.0

print(f"\n🎯 Action Accuracy: {action_acc*100:.2f}%")
print(f"🎯 Finger Accuracy (non-REST): {finger_acc*100:.2f}%\n")

# =========================
# ===== ECE COMPUTATION ===
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


def expected_calibration_error(conf, preds, labels, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (conf > bins[i]) & (conf <= bins[i + 1])
        if np.sum(idx) == 0:
            continue
        bin_acc = np.mean(preds[idx] == labels[idx])
        bin_conf = np.mean(conf[idx])
        ece += np.abs(bin_acc - bin_conf) * (np.sum(idx) / len(conf))
    return ece

action_ece = expected_calibration_error(action_conf, action_preds, y_action_test, N_BINS)
print(f"📏 Action ECE: {action_ece:.4f}")

if mask.any():
    finger_ece = expected_calibration_error(
        finger_conf[mask], finger_preds[mask], y_finger_test[mask], N_BINS
    )
    print(f"📏 Finger ECE (non-REST): {finger_ece:.4f}")

if subject_ids is not None:
    unique_subjects = sorted(set(subject_ids))
    if len(unique_subjects) > 1:
        print("\nPer-subject ECE (test set):")
        for subj in unique_subjects:
            subj_mask = (subject_ids[test_idx] == subj)
            if not subj_mask.any():
                continue
            subj_action_ece = expected_calibration_error(
                action_conf[subj_mask], action_preds[subj_mask], y_action_test[subj_mask], N_BINS
            )
            if mask.any():
                subj_finger_mask = subj_mask & (y_action_test != ACTION_REST)
                if subj_finger_mask.any():
                    subj_finger_ece = expected_calibration_error(
                        finger_conf[subj_finger_mask], finger_preds[subj_finger_mask], y_finger_test[subj_finger_mask], N_BINS
                    )
                else:
                    subj_finger_ece = float("nan")
            else:
                subj_finger_ece = float("nan")
            print(f"  {subj}: action_ece={subj_action_ece:.4f}, finger_ece={subj_finger_ece:.4f}")

# =========================
# ===== PLOTS =============
# =========================

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

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
    ax=axs[0, 0]
)
axs[0, 0].set_title("Action Confusion Matrix")
axs[0, 0].set_xlabel("Predicted")
axs[0, 0].set_ylabel("True")

# --- Finger Confusion Matrix (non-REST) ---
if mask.any():
    finger_label_ids = [i for i in sorted(FINGER_NAMES.keys()) if i != 0]
    finger_cm = confusion_matrix(
        y_finger_test[mask],
        finger_preds[mask],
        labels=finger_label_ids
    )
    finger_labels = [FINGER_NAMES[i] for i in finger_label_ids]
    sns.heatmap(
        finger_cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=finger_labels,
        yticklabels=finger_labels,
        ax=axs[0, 1]
    )
    axs[0, 1].set_title("Finger Confusion Matrix (non-REST)")
    axs[0, 1].set_xlabel("Predicted")
    axs[0, 1].set_ylabel("True")
else:
    axs[0, 1].set_axis_off()

# --- Reliability Diagram (Action) ---
bin_centers, bin_accs = reliability_bins(action_conf, action_preds, y_action_test, N_BINS)
axs[1, 0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
axs[1, 0].bar(bin_centers, bin_accs, width=0.08, alpha=0.7)
axs[1, 0].set_title("Action Reliability Diagram")
axs[1, 0].set_xlabel("Confidence")
axs[1, 0].set_ylabel("Accuracy")

# --- Reliability Diagram (Finger, non-REST) ---
if mask.any():
    f_bin_centers, f_bin_accs = reliability_bins(
        finger_conf[mask], finger_preds[mask], y_finger_test[mask], N_BINS
    )
    axs[1, 1].plot([0, 1], [0, 1], "--", color="gray")
    axs[1, 1].bar(f_bin_centers, f_bin_accs, width=0.08, alpha=0.7, color="green")
    axs[1, 1].set_title("Finger Reliability (non-REST)")
    axs[1, 1].set_xlabel("Confidence")
    axs[1, 1].set_ylabel("Accuracy")
else:
    axs[1, 1].set_axis_off()

plt.tight_layout()

# Save figure for reports
report_dir = Path("reports/subjects")
report_dir.mkdir(parents=True, exist_ok=True)
out_path = report_dir / f"eval_{exp_hash}.png"
plt.savefig(out_path)
if SHOW_PLOTS:
    plt.show()
else:
    plt.close()
