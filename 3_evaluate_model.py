"""
STEP 3 — Model Evaluation + Confidence Calibration
ISEF / Paper-Ready
(Deterministic evaluation — Dropout OFF)
"""

# Work around duplicate libomp on macOS (MKL/torch/scipy); must be set before imports.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from typing import Optional
from pathlib import Path

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

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
MIN_TEST_SAMPLES = 30
MAX_SPLIT_ATTEMPTS = 8
DEFAULT_BATCH_SIZE = 256

def _has_len(x) -> bool:
    try:
        _ = len(x)
        return True
    except Exception:
        return False


def _is_maskable_array(val, n: int) -> bool:
    try:
        arr = np.asarray(val)
    except Exception:
        return False
    if arr.ndim == 0:
        return False
    try:
        return len(arr) == n
    except Exception:
        return False


def mask_meta(meta: dict, mask: np.ndarray) -> dict:
    """
    Mask only 1D arrays of length N. Leave scalars/0D arrays/strings/dicts untouched.
    Prevents: TypeError: len() of unsized object
    """
    if not isinstance(meta, dict):
        return meta
    mask = np.asarray(mask, dtype=bool)
    n = int(mask.size)
    out = {}
    for k, v in meta.items():
        if isinstance(v, dict) or isinstance(v, (str, bytes)):
            out[k] = v
            continue
        if _is_maskable_array(v, n):
            out[k] = np.asarray(v)[mask]
        else:
            out[k] = v
    return out


def take_meta(meta: dict, keys, idx: np.ndarray, n_expected: int, dtype=None):
    """
    Safely take meta arrays aligned to dataset length.
    """
    if not isinstance(meta, dict):
        return None
    idx = np.asarray(idx)
    for key in keys:
        if key not in meta:
            continue
        try:
            arr = np.asarray(meta[key])
        except Exception:
            continue
        if arr.ndim == 0:
            continue
        if len(arr) != n_expected:
            continue
        out = arr[idx]
        if dtype is not None:
            out = out.astype(dtype)
        return out
    return None


def _first_meta_scalar(meta: dict, keys, default="UNKNOWN") -> str:
    """
    Returns a reasonable scalar string for keys that might be scalar or length-N arrays.
    """
    if not isinstance(meta, dict):
        return str(default)
    for k in keys:
        if k not in meta:
            continue
        try:
            arr = np.asarray(meta[k])
        except Exception:
            continue
        if arr.ndim == 0:
            s = str(arr)
            if s and s != "UNKNOWN":
                return s
        else:
            if len(arr) > 0:
                s = str(arr.flat[0])
                if s and s != "UNKNOWN":
                    return s
    return str(default)

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


def apply_postprocess_sequence(action_probs, finger_probs, order, settings, trial_ids=None):
    """
    Apply stateful postprocess across a sequence order.
    Reset behavior: if trial_ids provided, RESET PostprocessState when trial_id changes.
    Returns committed_action, committed_finger aligned to `order` output.
    """
    state = PostprocessState()
    committed_action = np.zeros(len(order), dtype=np.int64)
    committed_finger = np.zeros(len(order), dtype=np.int64)

    last_trial = None

    for out_idx, sample_idx in enumerate(order):
        if trial_ids is not None:
            t = int(trial_ids[sample_idx])
            if last_trial is None:
                last_trial = t
            elif t != last_trial:
                state.reset()
                last_trial = t

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
    labels = []
    for i in range(n_classes):
        if i in name_map:
            labels.append(str(name_map[i]))
        else:
            labels.append(f"CLASS_{i}")
    return labels


def _load_predictions_if_present(path: Path):
    if not path.exists():
        return None
    try:
        d = np.load(path, allow_pickle=True)

        # Step 2 writes these keys
        required = {"action_probs", "finger_probs", "y_action", "y_finger"}
        if not required.issubset(set(d.files)):
            return None

        # Accept either legacy 'test_indices' OR step2's 'test_indices_local'
        if "test_indices" in d.files:
            test_idx = d["test_indices"]
        elif "test_indices_local" in d.files:
            test_idx = d["test_indices_local"]
        else:
            return None

        out = {
            "action_probs": d["action_probs"],
            "finger_probs": d["finger_probs"],
            "y_action": d["y_action"],
            "y_finger": d["y_finger"],
            "test_indices": test_idx,
        }

        # Optional metadata aligned to rows of probs/labels
        for k in ["window_start", "window_end", "trial_id", "block_id", "subject_id", "experiment_hash"]:
            if k in d.files:
                out[k] = d[k]

        return out
    except Exception:
        return None


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


def _apply_sample_limit(X, y_action, y_finger, meta, max_samples: Optional[int], seed: int):
    if not max_samples or len(y_action) <= max_samples:
        return X, y_action, y_finger, meta

    indices = np.arange(len(y_action))
    stratify_labels = (y_action.astype(int) * 100) + y_finger.astype(int)
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1, train_size=max_samples, random_state=seed
        )
        keep_idx, _ = next(splitter.split(indices, stratify_labels))
    except ValueError:
        rng = np.random.default_rng(seed)
        keep_idx = rng.choice(indices, size=max_samples, replace=False)

    keep_idx = np.sort(keep_idx)
    X = X[keep_idx]
    y_action = y_action[keep_idx]
    y_finger = y_finger[keep_idx]
    if meta:
        n_before = int(len(indices))
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
                    out[key] = arr[keep_idx]
                else:
                    out[key] = val
            except Exception:
                out[key] = val
        meta = out
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
        finger_train_unique = _unique_non_rest_fingers(y_action[train_idx], y_finger[train_idx])
        finger_test_unique = _unique_non_rest_fingers(y_action[test_idx], y_finger[test_idx])

        action_ok = overall_action_unique < 2 or (action_train_unique >= 2 and action_test_unique >= 2)
        finger_ok = overall_finger_unique < 2 or (finger_train_unique >= 2 and finger_test_unique >= 2)

        if action_ok and finger_ok:
            return train_idx, test_idx

    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, default="eeg_windows.npz", help="Sequence npz file")
    parser.add_argument("--pred-npz", type=str, default="test_predictions.npz", help="Optional cached test predictions")
    parser.add_argument("--model", type=str, default="finger_action_model.pt", help="Model weights path")
    parser.add_argument("--scaler", type=str, default="scaler.save", help="Normalizer/scaler path")
    parser.add_argument("--subject-id", type=str, default="1-M17", help="Filter evaluation to a single subject_id")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples for eval (memory guard)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Inference batch size")
    parser.add_argument("--smooth-action-only", action="store_true",
                    help="Smooth action only; finger stays raw (except forced NONE during REST)")

    parser.add_argument("--smooth", action="store_true", help="Enable postprocess smoothing")
    parser.add_argument("--smooth-method", type=str, default="vote", choices=["vote", "ema"])
    parser.add_argument("--smooth-window", type=int, default=5)
    parser.add_argument("--hysteresis", action="store_true", help="Enable action hysteresis")
    parser.add_argument("--hysteresis-frames", type=int, default=3)
    parser.add_argument("--threshold-action", type=float, default=0.75)
    parser.add_argument("--threshold-finger", type=float, default=0.75)
    parser.add_argument("--adjacency", action="store_true", help="Enable adjacency assist (finger correction)")
    args = parser.parse_args()
    if args.smooth_action_only and not args.smooth:
        args.smooth = True
    npz_path = Path(args.npz)
    pred_npz_path = Path(args.pred_npz)
    model_path = Path(args.model)
    scaler_path = Path(args.scaler)

    # =========================
    # ===== LOAD DATA =========
    # =========================
    X, y_action, y_finger, meta = load_sequence_npz(str(npz_path), mmap_mode="r")
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
        meta = mask_meta(meta, mask)
        subject_filtered = True

    if isinstance(X, np.memmap) and X.dtype != np.float32:
        print(f"ℹ️ X dtype is {X.dtype}; casting to float32 per batch.")

    X, y_action, y_finger, meta = _apply_sample_limit(
        X, y_action, y_finger, meta, args.max_samples, SEED
    )

    _print_label_summary("Filtered", y_action, y_finger)

    subject_ids = meta.get("subject_id", None)
    exp_hash = _first_meta_scalar(meta, ["experiment_hash", "exp_hash"], default="UNKNOWN")

    n_fingers = int(np.max(y_finger)) + 1
    n_actions = int(np.max(y_action)) + 1

    # =========================
    # ===== TEST SPLIT =========
    # =========================
    cached = _load_predictions_if_present(pred_npz_path)
    if subject_filtered and cached is not None:
        print("ℹ️ Ignoring cached predictions because --subject-id filtering is enabled.")
        cached = None
    if args.max_samples and cached is not None:
        print("ℹ️ Ignoring cached predictions because --max-samples is enabled.")
        cached = None

    if cached is not None:
        action_probs = np.asarray(cached["action_probs"])
        finger_probs = np.asarray(cached["finger_probs"])
        y_action_test = np.asarray(cached["y_action"])
        y_finger_test = np.asarray(cached["y_finger"])
        test_idx = np.asarray(cached["test_indices"]).astype(np.int64)

        all_idx = np.arange(len(y_action), dtype=np.int64)
        test_mask = np.zeros(len(y_action), dtype=bool)
        test_mask[test_idx] = True
        train_idx = all_idx[~test_mask]

        if len(action_probs) != len(y_action_test) or len(finger_probs) != len(y_finger_test):
            raise RuntimeError("test_predictions.npz shapes do not align (probs vs labels).")

        print(f"✅ Using cached predictions: {pred_npz_path}")

    else:
        train_idx, test_idx = _split_with_checks(y_action, y_finger, meta=meta, seed=SEED)
        if train_idx is None or test_idx is None:
            print("⚠️ Unable to create a split with multiple classes. Aborting evaluation.")
            return 2

        y_action_test = y_action[test_idx]
        y_finger_test = y_finger[test_idx]

        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler/normalizer file: {scaler_path}")
        normalizer = joblib.load(str(scaler_path))

        if not model_path.exists():
            raise FileNotFoundError(f"Missing model weights: {model_path}")

        model = CNNLSTMFingerActionNet(
            n_channels=X.shape[2],
            n_fingers=n_fingers,
            n_actions=n_actions,
        )
        model.load_state_dict(torch.load(str(model_path), map_location="cpu"))
        model.eval()

        batch_size = max(1, int(args.batch_size))
        action_probs = np.zeros((len(test_idx), n_actions), dtype=np.float32)
        finger_probs = np.zeros((len(test_idx), n_fingers), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, len(test_idx), batch_size):
                end = min(start + batch_size, len(test_idx))
                batch_idx = test_idx[start:end]
                X_batch = np.asarray(X[batch_idx], dtype=np.float32)
                X_batch = apply_channel_normalizer(X_batch, normalizer)
                X_t = torch.tensor(X_batch, dtype=torch.float32)
                finger_logits, action_logits = model(X_t)
                action_probs[start:end] = torch.softmax(action_logits, dim=1).cpu().numpy()
                finger_probs[start:end] = torch.softmax(finger_logits, dim=1).cpu().numpy()

        print("✅ Ran deterministic inference (no cached test_predictions.npz found).")

    if len(test_idx) < MIN_TEST_SAMPLES:
        print(f"⚠️ Test set too small ({len(test_idx)} samples). Aborting evaluation.")
        return 2

    _print_label_summary("Train split", y_action[train_idx], y_finger[train_idx])
    _print_label_summary("Test split", y_action_test, y_finger_test)

    overall_action_unique = len(np.unique(y_action))
    overall_finger_unique = _unique_non_rest_fingers(y_action, y_finger)
    action_train_unique = len(np.unique(y_action[train_idx])) if len(train_idx) else 0
    action_test_unique = len(np.unique(y_action_test)) if len(y_action_test) else 0
    finger_train_unique = _unique_non_rest_fingers(y_action[train_idx], y_finger[train_idx])
    finger_test_unique = _unique_non_rest_fingers(y_action_test, y_finger_test)

    if overall_action_unique < 2:
        print("⚠️ Action labels are single-class overall. Aborting evaluation.")
        return 2
    if action_train_unique < 2 or action_test_unique < 2:
        print("⚠️ Action labels collapsed in train/test split. Aborting evaluation.")
        return 2

    finger_metrics_ok = True
    exit_code = 0
    if overall_finger_unique < 2:
        print("⚠️ Finger labels are single-class overall; skipping finger metrics.")
        finger_metrics_ok = False
        exit_code = 1
    elif finger_train_unique < 2 or finger_test_unique < 2:
        print("⚠️ Finger labels collapsed in train/test split; skipping finger metrics.")
        finger_metrics_ok = False
        exit_code = 1

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
    if not mask.any():
        print("⚠️ No non-REST windows in test set; skipping finger metrics.")
        finger_metrics_ok = False
        exit_code = max(exit_code, 1)

    finger_acc = (
        accuracy_score(y_finger_test[mask], finger_preds[mask]) if (finger_metrics_ok and mask.any()) else None
    )

    print(f"\n🎯 Action Accuracy: {action_acc*100:.2f}%")
    if finger_acc is not None:
        print(f"🎯 Finger Accuracy (non-REST): {finger_acc*100:.2f}%\n")
    else:
        print("🎯 Finger Accuracy (non-REST): skipped\n")

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
            finger_mode="raw" if args.smooth_action_only else "smooth",
        )

        order = np.arange(len(action_probs), dtype=np.int64)
        if "window_start" in meta:
            try:
                starts = np.asarray(meta["window_start"])[test_idx]
                order = np.argsort(starts).astype(np.int64)
            except Exception:
                order = np.arange(len(action_probs), dtype=np.int64)

        # ===== Reset by trial_id (requested) =====
        trial_ids_for_probs = None
        if "trial_id" in meta:
            try:
                trial_ids_for_probs = np.asarray(meta["trial_id"])[test_idx].astype(np.int64)
            except Exception:
                trial_ids_for_probs = None

        smoothed_action, smoothed_finger = apply_postprocess_sequence(
            action_probs, finger_probs, order, settings, trial_ids=trial_ids_for_probs
        )

        # If you want "smooth action only", override finger smoothing here
        if args.smooth_action_only:
            smoothed_finger = finger_preds[order].copy()
            smoothed_finger[smoothed_action == ACTION_REST] = 0  # NONE=0

        action_acc_s = accuracy_score(y_action_test[order], smoothed_action)
        mask_s = y_action_test[order] != ACTION_REST
        finger_acc_s = (
            accuracy_score(y_finger_test[order][mask_s], smoothed_finger[mask_s])
            if (finger_metrics_ok and mask_s.any())
            else None
        )

        print(f"🎯 Smoothed Action Accuracy: {action_acc_s*100:.2f}%")
        if finger_acc_s is not None:
            print(f"🎯 Smoothed Finger Accuracy (non-REST): {finger_acc_s*100:.2f}%\n")
        else:
            print("🎯 Smoothed Finger Accuracy (non-REST): skipped\n")

    # =========================
    # ===== ECE COMPUTATION ===
    # =========================
    action_ece = expected_calibration_error(action_conf, action_preds, y_action_test, N_BINS)
    print(f"📏 Action ECE: {action_ece:.4f}")

    if finger_metrics_ok and mask.any():
        finger_ece = expected_calibration_error(
            finger_conf[mask], finger_preds[mask], y_finger_test[mask], N_BINS
        )
        print(f"📏 Finger ECE (non-REST): {finger_ece:.4f}")

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
                    if finger_metrics_ok and np.any(subj_finger_mask):
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

    if finger_metrics_ok and mask.any():
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
        axs[0, 1].axis("off")
        axs[0, 1].text(0.5, 0.5, "Finger metrics skipped", ha="center", va="center")

    bin_centers, bin_accs = reliability_bins(action_conf, action_preds, y_action_test, N_BINS)
    axs[1, 0].plot([0, 1], [0, 1], "--", color="gray", label="Perfect Calibration")
    axs[1, 0].bar(bin_centers, bin_accs, width=0.08, alpha=0.7)
    axs[1, 0].set_title("Action Reliability Diagram")
    axs[1, 0].set_xlabel("Confidence")
    axs[1, 0].set_ylabel("Accuracy")

    if finger_metrics_ok and mask.any():
        f_bin_centers, f_bin_accs = reliability_bins(
            finger_conf[mask], finger_preds[mask], y_finger_test[mask], N_BINS
        )
        axs[1, 1].plot([0, 1], [0, 1], "--", color="gray")
        axs[1, 1].bar(f_bin_centers, f_bin_accs, width=0.08, alpha=0.7, color="green")
        axs[1, 1].set_title("Finger Reliability (non-REST)")
        axs[1, 1].set_xlabel("Confidence")
        axs[1, 1].set_ylabel("Accuracy")
    else:
        axs[1, 1].axis("off")
        axs[1, 1].text(0.5, 0.5, "Finger metrics skipped", ha="center", va="center")

    plt.tight_layout()

    report_dir = Path("reports/subjects")
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / f"eval_{exp_hash}.png"
    plt.savefig(out_path)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print(f"\n✅ Saved evaluation plot: {out_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())