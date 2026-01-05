"""
STEP 2 — Train Multi-Head EEG Model (CNN + LSTM)
(FIXED: saves global/local indices + test metadata for robust evaluation + smoothing)
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import joblib

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    fit_channel_normalizer,
    apply_channel_normalizer,
)
from utils.experiment_logger import log_experiment, get_latest_experiment_hash
from utils.label_schema import ACTION_REST

SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
LOSS_ACTION_WEIGHT = 1.0
REST_WEIGHT = 0.2

subject_id = "ANON"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EEGWindowDataset(Dataset):
    def __init__(self, X, y_finger, y_action):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y_finger = torch.tensor(y_finger, dtype=torch.long)
        self.y_action = torch.tensor(y_action, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y_finger[idx], self.y_action[idx]


def resolve_experiment_hash():
    exp_hash = "UNKNOWN"
    subject = subject_id
    try:
        exp_hash = get_latest_experiment_hash()
    except Exception:
        meta_path = Path("session_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            subject = meta.get("subject_id", subject)
            exp_hash = meta.get("experiment_hash", exp_hash)
    return subject, exp_hash


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train CNN+LSTM EEG multi-head model")
    p.add_argument("--npz", type=str, default="eeg_windows.npz", help="Path to window dataset")
    p.add_argument("--subject-id", type=str, default="1-M17", help="Filter training data to a single subject_id")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Training batch size")
    p.add_argument("--lr", type=float, default=LR, help="Learning rate")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed")
    p.add_argument("--loss-action-weight", type=float, default=LOSS_ACTION_WEIGHT, help="Action loss weight")
    p.add_argument("--rest-weight", type=float, default=REST_WEIGHT, help="Class weight for REST action (0=ignore)")
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument("--non-rest-only", action="store_true", help="Train only on non-REST windows")
    p.add_argument("--save-model", type=str, default="finger_action_model.pt", help="Model output path")
    p.add_argument("--save-scaler", type=str, default="scaler.save", help="Scaler output path")
    p.add_argument("--save-preds", type=str, default="test_predictions.npz", help="Predictions output path")
    return p


def _safe_meta_take(meta: Dict[str, Any], idx: np.ndarray, key: str, dtype: Optional[Any] = None):
    if not meta or key not in meta:
        return None
    arr = np.asarray(meta[key])
    if len(arr) != len(idx) and len(arr) != len(meta.get("subject_id", arr)):
        # best effort: only take if it matches dataset length
        if len(arr) != len(meta.get("subject_id", arr)):
            return None
    out = arr[idx]
    if dtype is not None:
        out = out.astype(dtype)
    return out


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    subject, exp_hash = resolve_experiment_hash()
    log_experiment(subject, exp_hash, "STEP_2_TRAIN")

    # ===== LOAD FULL DATA =====
    X_full, y_action_full, y_finger_full, meta_full = load_sequence_npz(args.npz)

    # Global index mapping (original dataset positions)
    global_indices = np.arange(len(y_action_full), dtype=np.int64)

    # ===== OPTIONAL SUBJECT FILTER =====
    X = X_full
    y_action = y_action_full
    y_finger = y_finger_full
    meta = meta_full

    if args.subject_id:
        if "subject_id" not in meta:
            print("subject_id not found in dataset metadata; cannot filter.")
            return 2
        subject_ids = np.asarray(meta["subject_id"]).astype(str)
        mask = subject_ids == args.subject_id
        kept = int(mask.sum())
        if kept == 0:
            print(f"No windows found for subject_id={args.subject_id}")
            return 2

        X = X[mask]
        y_action = y_action[mask]
        y_finger = y_finger[mask]
        global_indices = global_indices[mask]  # ✅ critical mapping

        # Filter meta arrays that match dataset length
        meta = {
            key: (np.asarray(val)[mask] if isinstance(val, np.ndarray) and len(val) == len(mask) else val)
            for key, val in meta.items()
        }
        subject = args.subject_id

    # ===== SPLIT =====
    train_idx, test_idx = split_indices(
        y_action,
        y_finger,
        meta=meta,
        test_size=args.test_size,
        random_state=args.seed,
    )

    if args.non_rest_only:
        keep_mask = y_action[train_idx] != ACTION_REST
        kept = int(keep_mask.sum())
        total = int(len(train_idx))
        print(f"Training mode: non-rest-only (kept {kept}/{total})")
        if kept == 0:
            print("No non-REST samples available for training; aborting.")
            return 2
        train_idx = train_idx[keep_mask]

    # ===== SLICE =====
    X_train, X_test = X[train_idx], X[test_idx]
    y_action_train, y_action_test = y_action[train_idx], y_action[test_idx]
    y_finger_train, y_finger_test = y_finger[train_idx], y_finger[test_idx]

    # ===== NORMALIZE =====
    normalizer = fit_channel_normalizer(X_train)
    X_train = apply_channel_normalizer(X_train, normalizer)
    X_test = apply_channel_normalizer(X_test, normalizer)
    joblib.dump(normalizer, args.save_scaler)

    # ===== DATALOADERS =====
    train_loader = DataLoader(
        EEGWindowDataset(X_train, y_finger_train, y_action_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    test_loader = DataLoader(
        EEGWindowDataset(X_test, y_finger_test, y_action_test),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    n_fingers = int(np.max(y_finger)) + 1
    n_actions = int(np.max(y_action)) + 1

    # ===== MODEL =====
    model = CNNLSTMFingerActionNet(n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Finger loss (unweighted by default)
    loss_f = nn.CrossEntropyLoss()

    # Action loss (REST downweighted)
    action_weights = torch.ones(n_actions, dtype=torch.float32)
    if ACTION_REST < n_actions:
        action_weights[ACTION_REST] = max(0.0, float(args.rest_weight))
    loss_a = nn.CrossEntropyLoss(weight=action_weights.to(device))

    # ===== TRAIN =====
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        total_action = 0
        total_finger = 0
        correct_action = 0
        correct_finger = 0

        for Xb, yfb, yab in train_loader:
            Xb = Xb.to(device)
            yfb = yfb.to(device)
            yab = yab.to(device)

            opt.zero_grad()
            f_out, a_out = model(Xb)

            loss_action = loss_a(a_out, yab)
            mask_nr = yab != ACTION_REST
            if mask_nr.any():
                loss_finger = loss_f(f_out[mask_nr], yfb[mask_nr])
            else:
                loss_finger = torch.tensor(0.0, device=device)

            loss = loss_action + float(args.loss_action_weight) * loss_finger
            loss.backward()
            opt.step()

            total_loss += loss.item() * Xb.size(0)

            preds_action = torch.argmax(a_out, dim=1)
            correct_action += (preds_action == yab).sum().item()
            total_action += yab.numel()

            if mask_nr.any():
                preds_finger = torch.argmax(f_out[mask_nr], dim=1)
                correct_finger += (preds_finger == yfb[mask_nr]).sum().item()
                total_finger += yfb[mask_nr].numel()

        avg_loss = total_loss / max(1, len(train_loader.dataset))
        action_acc = correct_action / max(1, total_action)
        finger_acc = correct_finger / max(1, total_finger)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:03d}/{args.epochs} | loss={avg_loss:.4f} "
                f"action_acc={action_acc:.3f} finger_acc={finger_acc:.3f}"
            )

    # ===== SAVE MODEL =====
    model.eval()
    torch.save(model.state_dict(), args.save_model)

    train_config = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "loss_action_weight": args.loss_action_weight,
        "rest_weight": float(args.rest_weight),
        "test_size": args.test_size,
        "non_rest_only": bool(args.non_rest_only),
        "npz_path": str(args.npz),
        "n_fingers": n_fingers,
        "n_actions": n_actions,
        "input_shape": list(X.shape[1:]),
        "normalizer": {"type": normalizer.get("type", "unknown"), "channels": normalizer.get("channels", None)},
        "device": str(device),
        "model": "CNNLSTMFingerActionNet",
        "subject_id_filter": args.subject_id or "",
    }
    config_path = Path("logs") / "experiments" / f"{exp_hash}_train_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(train_config, indent=2))

    # ===== INFERENCE ON TEST SET =====
    all_action_probs = []
    all_finger_probs = []
    with torch.no_grad():
        for Xb, yfb, yab in test_loader:
            Xb = Xb.to(device)
            f_out, a_out = model(Xb)
            all_finger_probs.append(torch.softmax(f_out, dim=1).cpu().numpy())
            all_action_probs.append(torch.softmax(a_out, dim=1).cpu().numpy())

    action_probs = np.concatenate(all_action_probs, axis=0).astype(np.float32)
    finger_probs = np.concatenate(all_finger_probs, axis=0).astype(np.float32)

    # ===== SAVE PREDICTIONS WITH ROBUST INDEXING + META =====
    test_indices_local = test_idx.astype(np.int64)
    test_indices_global = global_indices[test_idx].astype(np.int64)  # ✅ maps back to original NPZ index

    # Save *test-set* meta (aligned to prediction rows)
    # These arrays are crucial for smoothing eval with resets.
    test_window_start = None
    test_window_end = None
    test_trial_id = None
    test_block_id = None
    test_subject_id = None
    test_experiment_hash = None

    if meta:
        try:
            if "window_start" in meta:
                test_window_start = np.asarray(meta["window_start"])[test_idx].astype(np.float32)
            if "window_end" in meta:
                test_window_end = np.asarray(meta["window_end"])[test_idx].astype(np.float32)
            if "trial_id" in meta:
                test_trial_id = np.asarray(meta["trial_id"])[test_idx].astype(np.int64)
            if "block_id" in meta:
                test_block_id = np.asarray(meta["block_id"])[test_idx].astype(np.int64)
            if "subject_id" in meta:
                test_subject_id = np.asarray(meta["subject_id"])[test_idx].astype("U")
            if "experiment_hash" in meta:
                test_experiment_hash = np.asarray(meta["experiment_hash"])[test_idx].astype("U")
        except Exception:
            # best-effort only
            pass

    dataset_info = {
        "npz_path": str(args.npz),
        "subject_id_filter": args.subject_id or "",
        "exp_hash": str(exp_hash),
        "n_samples_used": int(len(y_action)),
        "n_samples_full": int(len(y_action_full)),
        "n_test": int(len(test_idx)),
        "n_actions": int(n_actions),
        "n_fingers": int(n_fingers),
    }

    save_dict = dict(
        action_probs=action_probs,
        finger_probs=finger_probs,
        y_action=y_action_test.astype(np.int64),
        y_finger=y_finger_test.astype(np.int64),
        test_indices_local=test_indices_local,
        test_indices_global=test_indices_global,
        dataset_info=np.array([json.dumps(dataset_info)], dtype="U"),
    )

    # Add optional meta if present
    if test_window_start is not None: save_dict["window_start"] = test_window_start
    if test_window_end is not None: save_dict["window_end"] = test_window_end
    if test_trial_id is not None: save_dict["trial_id"] = test_trial_id
    if test_block_id is not None: save_dict["block_id"] = test_block_id
    if test_subject_id is not None: save_dict["subject_id"] = test_subject_id
    if test_experiment_hash is not None: save_dict["experiment_hash"] = test_experiment_hash

    np.savez_compressed(args.save_preds, **save_dict)

    log_experiment(subject, exp_hash, "STEP_2_COMPLETE", f"loss={avg_loss:.4f}")
    print("✅ Training complete")
    print(f"DECISION: TRAINED (epochs={args.epochs})")
    print(f"✅ Saved: {args.save_model}, {args.save_scaler}, {args.save_preds}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())