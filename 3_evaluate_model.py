"""
STEP 3 — Model Evaluation + Confidence Calibration
ISEF / Paper-Ready
(Deterministic evaluation — Dropout OFF)
"""

# Work around duplicate libomp on macOS (MKL/torch/scipy); must be set before imports.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from pathlib import Path

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, accuracy_score

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import ACTION_REST, ACTION_NAMES, FINGER_NAMES
from utils.sequence_data import load_sequence_npz, split_indices, apply_channel_normalizer
from demo_backend.postprocess import PostprocessSettings, PostprocessState, postprocess_predictions

# =========================
# ===== CONFIG ============
# =========================

N_BINS = 10
SEED = 42
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"


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


def apply_postprocess_sequence(action_probs, finger_probs, order, settings):
    """
    Apply stateful postprocess across a sequence order.
    Returns committed_action, committed_finger aligned to `order` output.
    """
    state = PostprocessState()
    committed_action = np.zeros(len(order), dtype=np.int64)
    committed_finger = np.zeros(len(order), dtype=np.int64)

    for out_idx, sample_idx in enumerate(order):
        post = postprocess_predictions(
            action_probs[sample_idx],
            finger_probs[sample_idx],
            settings,
            state,
        )
        committed_action[out_idx] = int(post["committed_action_id"])
        committed_finger[out_idx] = int(post["committed_finger_id"])

    return committed_action, committed_finger


def _safe_label_list(name_map: dict, n_classes: int):
    """
    Build a label list of length n_classes using a mapping like ACTION_NAMES/FINGER_NAMES.
    If a key is missing, uses f"CLASS_{i}" fallback.
    """
    labels = []
    for i in range(n_classes):
        if i in name_map:
            labels.append(str(name_map[i]))
        else:
            labels.append(f"CLASS_{i}")
    return labels


def _load_predictions_if_present(path: Path):
    """
    If test_predictions.npz exists, load it.
    Expected keys (from Step 2 training script):
      - action_probs, finger_probs, y_action, y_finger, test_indices
    Returns dict or None.
    """
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=True)
        required = {"action_probs", "finger_probs", "y_action", "y_finger", "test_indices"}
        if not required.issubset(set(d.files)):
            return None
        return {
            "action_probs": d["action_probs"],
            "finger_probs": d["finger_probs"],
            "y_action": d["y_action"],
            "y_finger": d["y_finger"],
            "test_indices": d["test_indices"],
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="eeg_windows.npz", help="Sequence npz file")
    parser.add_argument("--pred-npz", type=str, default="test_predictions.npz", help="Optional cached test predictions")
    parser.add_argument("--model", type=str, default="finger_action_model.pt", help="Model weights path")
    parser.add_argument("--scaler", type=str, default="scaler.save", help="Normalizer/scaler path")
    parser.add_argument("--subject-id", type=str, default="5-M16", help="Filter evaluation to a single subject_id")

    parser.add_argument("--smooth", action="store_true", help="Enable postprocess smoothing")
    parser.add_argument("--smooth-method", type=str, default="vote", choices=["vote", "ema"])
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--hysteresis", action="store_true", help="Enable action hysteresis")
    parser.add_argument("--hysteresis-frames", type=int, default=3)
    parser.add_argument("--threshold-action", type=float, default=0.75)
    parser.add_argument("--threshold-finger", type=float, default=0.75)
    parser.add_argument("--adjacency", action="store_true", help="Enable adjacency assist (finger correction)")
    args = parser.parse_args()

    npz_path = Path(args.npz)
    pred_npz_path = Path(args.pred_npz)
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)

    # =========================
    # ===== LOAD DATA =========
    # =========================
    X, y_action, y_finger, meta = load_sequence_npz(str(npz_path))
    subject_filtered = False
    if args.subject_id:
        if "subject_id" not in meta:
            print("subject_id not found in dataset metadata; cannot filter.")
            return 2
        subject_ids_all = np.asarray(meta["subject_id"]).astype(str)
        mask = subject_ids_all == args.subject_id
        kept = int(mask.sum())
        if kept == 0:
            print(f"No windows found for subject_id={args.subject_id}")
            return 2
        X = X[mask]
        y_action = y_action[mask]
        y_finger = y_finger[mask]
        meta = {
            key: (np.array(val)[mask] if isinstance(val, np.ndarray) and len(val) == len(mask) else val)
            for key, val in meta.items()
        }
        subject_filtered = True

    subject_ids = meta.get("subject_id", None)
    experiment_hashes = meta.get("experiment_hash", None)
    exp_hash = str(experiment_hashes[0]) if experiment_hashes is not None else "UNKNOWN"

    # Determine class counts from the full dataset (matches Step 2)
    n_fingers = int(np.max(y_finger)) + 1
    n_actions = int(np.max(y_action)) + 1

    # =========================
    # ===== TEST SPLIT =========
    # =========================
    # Priority 1: If cached predictions exist, trust their test_indices + labels for reproducibility.
    cached = _load_predictions_if_present(pred_npz_path)
    if subject_filtered and cached is not None:
        print("ℹ️ Ignoring cached predictions because --subject-id filtering is enabled.")
        cached = None

    if cached is not None:
        action_probs = np.asarray(cached["action_probs"])
        finger_probs = np.asarray(cached["finger_probs"])
        y_action_test = np.asarray(cached["y_action"])
        y_finger_test = np.asarray(cached["y_finger"])
        test_idx = np.asarray(cached["test_indices"]).astype(np.int64)

        # Sanity: ensure probabilities align with label lengths
        if len(action_probs) != len(y_action_test) or len(finger_probs) != len(y_finger_test):
            raise RuntimeError("test_predictions.npz shapes do not align (probs vs labels).")

        # We'll use meta for ordering (time) if possible, via test_idx.
        print(f"✅ Using cached predictions: {pred_npz_path}")
    else:
        # Priority 2: reproduce the same split and run deterministic inference.
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
            raise FileNotFoundError(f"Missing scaler/normalizer file: {scaler_path}")
        normalizer = joblib.load(str(scaler_path))
        X_test = apply_channel_normalizer(X_test, normalizer)

        # =========================
        # ===== MODEL (MATCH STEP 2)
        # =========================
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights: {model_path}")

        model = CNNLSTMFingerActionNet(
            n_channels=X.shape[2],
            n_fingers=n_fingers,
            n_actions=n_actions,
        )
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()

        # =========================
        # ===== INFERENCE =========
        # =========================
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            finger_logits, action_logits = model(X_t)
            action_probs = torch.softmax(action_logits, dim=1).cpu().numpy()
            finger_probs = torch.softmax(finger_logits, dim=1).cpu().numpy()

        print("✅ Ran deterministic inference (no cached test_predictions.npz found).")

    # =========================
    # ===== PREDICTIONS =========
    # =========================
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
    # ===== OPTIONAL SMOOTHED METRICS (stateful) ==========
    # =========================
    if args.smooth:
        settings = PostprocessSettings(
            smoothing_enabled=True,
            smoothing_method=args.smooth_method,
            smoothing_window=int(args.smooth_window),
            hysteresis_enabled=bool(args.hysteresis),
            hysteresis_frames=int(args.hysteresis_frames),
            threshold_action=float(args.threshold_action),
            threshold_finger=float(args.threshold_finger),
            adjacency_enabled=bool(args.adjacency),
        )

        # Postprocess is stateful, so order matters.
        # Best: sort by window_start time if meta has it.
        order = np.arange(len(action_probs), dtype=np.int64)
        if "window_start" in meta:
            try:
                # meta["window_start"] is for the full dataset; map via test_idx.
                starts = np.asarray(meta["window_start"])[test_idx]
                order = np.argsort(starts).astype(np.int64)
            except Exception:
                # Fall back to the default sequential order
                order = np.arange(len(action_probs), dtype=np.int64)

        smoothed_action, smoothed_finger = apply_postprocess_sequence(
            action_probs, finger_probs, order, settings
        )

        action_acc_s = accuracy_score(y_action_test[order], smoothed_action)
        mask_s = y_action_test[order] != ACTION_REST
        finger_acc_s = (
            accuracy_score(y_finger_test[order][mask_s], smoothed_finger[mask_s])
            if mask_s.any()
            else 0.0
        )

        print(f"🎯 Smoothed Action Accuracy: {action_acc_s*100:.2f}%")
        print(f"🎯 Smoothed Finger Accuracy (non-REST): {finger_acc_s*100:.2f}%\n")

    # =========================
    # ===== ECE COMPUTATION ===
    # =========================
    action_ece = expected_calibration_error(action_conf, action_preds, y_action_test, N_BINS)
    print(f"📏 Action ECE: {action_ece:.4f}")

    if mask.any():
        finger_ece = expected_calibration_error(
            finger_conf[mask], finger_preds[mask], y_finger_test[mask], N_BINS
        )
        print(f"📏 Finger ECE (non-REST): {finger_ece:.4f}")

    # Optional per-subject ECE (on the same test set)
    if subject_ids is not None:
        try:
            subj_test = np.asarray(subject_ids)[test_idx]
            unique_subjects = sorted(set(subj_test.tolist()))
            if len(unique_subjects) > 1:
                print("\nPer-subject ECE (test set):")
                for subj in unique_subjects:
                    subj_mask = (subj_test == subj)
                    if not np.any(subj_mask):
                        continue
                    subj_action_ece = expected_calibration_error(
                        action_conf[subj_mask], action_preds[subj_mask], y_action_test[subj_mask], N_BINS
                    )

                    subj_finger_mask = subj_mask & (y_action_test != ACTION_REST)
                    if np.any(subj_finger_mask):
                        subj_finger_ece = expected_calibration_error(
                            finger_conf[subj_finger_mask],
                            finger_preds[subj_finger_mask],
                            y_finger_test[subj_finger_mask],
                            N_BINS,
                        )
                    else:
                        subj_finger_ece = float("nan")

                    print(f"  {subj}: action_ece={subj_action_ece:.4f}, finger_ece={subj_finger_ece:.4f}")
        except Exception:
            pass

    # =========================
    # ===== PLOTS =============
    # =========================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # --- Action Confusion Matrix ---
    action_cm = confusion_matrix(y_action_test, action_preds, labels=list(range(n_actions)))
    action_labels = _safe_label_list(ACTION_NAMES, n_actions)

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
        # Finger confusion is only meaningful for non-rest windows.
        # We exclude finger_id=0 (NONE/REST) from the label set for clarity.
        finger_label_ids = [i for i in range(n_fingers) if i != 0]
        finger_cm = confusion_matrix(
            y_finger_test[mask],
            finger_preds[mask],
            labels=finger_label_ids,
        )
        finger_labels = [_safe_label_list(FINGER_NAMES, n_fingers)[i] for i in finger_label_ids]

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

    print(f"\n✅ Saved evaluation plot: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
