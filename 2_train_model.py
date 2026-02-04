"""
STEP 2 — Train Multi-Head EEG Model (CNN + LSTM)
(FIXED: saves global/local indices + test metadata for robust evaluation + smoothing)
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import json
import random
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

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

DEFAULT_NPZ = "eeg_windows.npz"
DEFAULT_MODEL = "finger_action_model.pt"
DEFAULT_SCALER = "scaler.save"
DEFAULT_PREDS = "test_predictions.npz"
MAX_SEARCH_DEPTH = 4

ROOT_DIR = Path(__file__).resolve().parent

subject_id = "ANON"


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return payload.get("settings", payload)


def _apply_config_to_args(args_obj, settings: Dict[str, Any], defaults: Dict[str, Any]):
    for key, default in defaults.items():
        if key in settings and getattr(args_obj, key) == default:
            setattr(args_obj, key, settings[key])


def sha256_file(path: Path) -> Optional[str]:
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def safe_resolve(path: Path) -> str:
    try:
        return str(path.expanduser().resolve())
    except Exception:
        return str(path)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _first_nonempty_meta_value(
    meta: Dict[str, Any], keys, n_expected: int
) -> Optional[str]:
    if not meta:
        return None
    for key in keys:
        if key not in meta:
            continue
        try:
            arr = np.asarray(meta[key])
        except Exception:
            continue
        if arr.ndim == 0:
            val = str(arr)
            if val and val != "UNKNOWN":
                return val
            continue
        if len(arr) != n_expected:
            continue
        arr_u = np.asarray(arr).astype("U")
        unique = [v for v in np.unique(arr_u) if v and v != "UNKNOWN"]
        if unique:
            return unique[0]
    return None


def resolve_experiment_hash(meta: Dict[str, Any], n_expected: int) -> str:
    exp_hash = _first_nonempty_meta_value(
        meta, ["experiment_hash", "exp_hash"], n_expected
    )
    if exp_hash:
        return exp_hash
    meta_path = ROOT_DIR / "session_meta.json"
    if meta_path.exists():
        try:
            root_meta = json.loads(meta_path.read_text())
            root_hash = root_meta.get("experiment_hash")
            if root_hash:
                return str(root_hash)
        except Exception:
            pass
    try:
        return get_latest_experiment_hash()
    except Exception:
        return "UNKNOWN"


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train CNN+LSTM EEG multi-head model")
    p.add_argument("--config", type=str, default=None, help="Path to JSON config")
    p.add_argument(
        "--npz", type=str, default=DEFAULT_NPZ, help="Path to window dataset"
    )
    p.add_argument(
        "--subject-id",
        type=str,
        default="8-M16",
        help="Filter training data to a single subject_id",
    )
    p.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of training epochs"
    )
    p.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Training batch size"
    )
    p.add_argument("--lr", type=float, default=LR, help="Learning rate")
    p.add_argument("--seed", type=int, default=SEED, help="Random seed")
    p.add_argument(
        "--loss-action-weight",
        type=float,
        default=LOSS_ACTION_WEIGHT,
        help="Action loss weight",
    )
    p.add_argument(
        "--rest-weight",
        type=float,
        default=REST_WEIGHT,
        help="Class weight for REST action (0=ignore)",
    )
    p.add_argument("--test-size", type=float, default=0.2, help="Test split fraction")
    p.add_argument(
        "--non-rest-only", action="store_true", help="Train only on non-REST windows"
    )
    p.add_argument(
        "--save-model", type=str, default=DEFAULT_MODEL, help="Model output path"
    )
    p.add_argument(
        "--save-scaler", type=str, default=DEFAULT_SCALER, help="Scaler output path"
    )
    p.add_argument(
        "--save-preds", type=str, default=DEFAULT_PREDS, help="Predictions output path"
    )
    return p


def _latest_by_mtime(paths):
    paths = [p for p in paths if p.exists()]
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _search_eeg_windows(root: Path, max_depth: int):
    matches = []
    for dirpath, dirnames, filenames in os.walk(root):
        rel = Path(dirpath).relative_to(root)
        depth = len(rel.parts)
        if depth >= max_depth:
            dirnames[:] = []
        for name in filenames:
            if name.endswith(".npz") and "eeg_windows" in name:
                matches.append(Path(dirpath) / name)
    return matches


def resolve_npz_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if candidate.exists():
        return candidate
    if not candidate.is_absolute():
        root_candidate = ROOT_DIR / candidate
        if root_candidate.exists():
            return root_candidate
    if path_str not in (DEFAULT_NPZ, f"./{DEFAULT_NPZ}"):
        raise FileNotFoundError(f"NPZ file not found: {path_str}")

    processed_dir = ROOT_DIR / "data/processed"
    if processed_dir.exists():
        latest = _latest_by_mtime(list(processed_dir.glob("*_eeg_windows.npz")))
        if latest:
            return latest
        direct = processed_dir / DEFAULT_NPZ
        if direct.exists():
            return direct

    windows_dir = ROOT_DIR / "data/windows"
    if windows_dir.exists():
        latest = _latest_by_mtime(list(windows_dir.glob("*_eeg_windows.npz")))
        if latest:
            return latest

    deep_matches = _search_eeg_windows(ROOT_DIR, MAX_SEARCH_DEPTH)
    latest = _latest_by_mtime(deep_matches)
    if latest:
        return latest

    raise FileNotFoundError(
        f"NPZ file not found: {path_str}. Searched default locations under {ROOT_DIR}."
    )


def infer_subject_id_from_npz(npz_path: Path) -> Optional[str]:
    stem = npz_path.stem
    if "_eeg_windows" in stem:
        prefix = stem.split("_eeg_windows")[0]
    else:
        prefix = stem
    if prefix in ("eeg_windows", ""):
        return None
    match = re.match(r"^(?P<subject>.+)_\d{8}_\d{6}$", prefix)
    if match:
        return match.group("subject")
    if "_" in prefix:
        return prefix.split("_")[0]
    return prefix


def _subject_ids_from_meta(meta: Dict[str, Any], n_before: int) -> Optional[np.ndarray]:
    if not meta or "subject_id" not in meta:
        return None
    try:
        arr = np.asarray(meta["subject_id"])
    except Exception:
        return None
    if arr.ndim == 0 or len(arr) != n_before:
        return None
    return arr.astype("U")


def infer_subject_id_from_meta(meta: Dict[str, Any], n_before: int) -> Optional[str]:
    subject_ids = _subject_ids_from_meta(meta, n_before)
    if subject_ids is None:
        return None
    unique = np.unique(subject_ids.astype(str))
    if len(unique) == 1:
        return unique[0]
    return None


def mask_meta(
    meta: Dict[str, Any], mask: np.ndarray, n_before: int
) -> Tuple[Dict[str, Any], List[str]]:
    if not meta:
        return {}, []
    mask = np.asarray(mask)
    out: Dict[str, Any] = {}
    masked_keys: List[str] = []
    for key, val in meta.items():
        if isinstance(val, dict):
            out[key] = val
            continue
        if isinstance(val, (str, bytes)):
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
            length = len(arr)
        except Exception:
            out[key] = val
            continue
        if length == n_before:
            out[key] = arr[mask]
            masked_keys.append(key)
        else:
            out[key] = val
    return out, masked_keys


def take_meta(
    meta: Dict[str, Any],
    keys,
    idx: np.ndarray,
    n_expected: int,
    dtype: Optional[Any] = None,
):
    if not meta:
        return None
    idx = np.asarray(idx)
    if idx.size == 0:
        return None
    for key in keys:
        if key not in meta:
            continue
        try:
            arr = np.asarray(meta[key])
            if arr.ndim == 0:
                continue
            if len(arr) != n_expected:
                continue
            out = arr[idx]
            if dtype is not None:
                out = out.astype(dtype)
            return out
        except Exception:
            continue
    return None


def ensure_X_shape(X, meta: Dict[str, Any]):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"Expected X to be 3D (N,T,C), got shape {X.shape}")

    channel_count = None
    if meta and "channel_names" in meta:
        try:
            channel_count = int(len(np.asarray(meta["channel_names"])))
        except Exception:
            channel_count = None

    # If we know the channel count, enforce that channels dimension equals it.
    if channel_count is not None and channel_count > 0:
        if X.shape[2] == channel_count:
            return X  # already (N,T,C)
        if X.shape[1] == channel_count and X.shape[2] != channel_count:
            return np.transpose(X, (0, 2, 1))  # (N,C,T) -> (N,T,C)
        raise ValueError(
            f"Cannot infer X layout: expected channels in dim=2 or dim=1 to equal {channel_count}, got {X.shape}"
        )

    # Fallback: assume (N,T,C) is correct; only transpose if it screams (N,C,T)
    if X.shape[1] <= 16 and X.shape[2] > 16:
        return np.transpose(X, (0, 2, 1))
    return X


def _subject_from_root_meta() -> Optional[str]:
    meta_path = ROOT_DIR / "session_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None
    subject = meta.get("subject_id")
    return str(subject) if subject else None


def resolve_output_paths(args, subject: str, exp_hash: str):
    subject_safe = subject or "UNKNOWN"
    run_dir = ROOT_DIR / "data/models" / subject_safe / exp_hash

    def _resolve(path_str: str, default_name: str) -> Path:
        if path_str == default_name:
            return run_dir / default_name
        return Path(path_str)

    model_path = _resolve(args.save_model, DEFAULT_MODEL)
    scaler_path = _resolve(args.save_scaler, DEFAULT_SCALER)
    preds_path = _resolve(args.save_preds, DEFAULT_PREDS)
    return run_dir, model_path, scaler_path, preds_path


def _validate_indices(idx: np.ndarray, n_samples: int, name: str):
    if idx.size == 0:
        return
    if idx.min() < 0 or idx.max() >= n_samples:
        raise ValueError(
            f"{name} indices out of bounds: min={idx.min()} max={idx.max()} n_samples={n_samples}"
        )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    settings = _load_config(args.config)
    _apply_config_to_args(args, settings, defaults)
    set_seed(args.seed)

    try:
        npz_path = resolve_npz_path(args.npz)
    except FileNotFoundError as exc:
        print(str(exc))
        return 2
    args.npz = str(npz_path)
    print(f"Using NPZ: {npz_path}")

    # ===== LOAD FULL DATA =====
    X_full, y_action_full, y_finger_full, meta_full = load_sequence_npz(str(npz_path))
    meta = meta_full if isinstance(meta_full, dict) else {}
    try:
        X_full = ensure_X_shape(X_full, meta)
    except ValueError as exc:
        print(str(exc))
        return 2

    y_action_full = np.asarray(y_action_full, dtype=np.int64).reshape(-1)
    y_finger_full = np.asarray(y_finger_full, dtype=np.int64).reshape(-1)

    if len(X_full) != len(y_action_full) or len(X_full) != len(y_finger_full):
        print(
            f"Dataset length mismatch: X={len(X_full)} y_action={len(y_action_full)} "
            f"y_finger={len(y_finger_full)}"
        )
        return 2

    meta_keys = sorted(meta.keys()) if meta else []
    print(
        f"Loaded NPZ: X={X_full.shape} y_action={y_action_full.shape} "
        f"y_finger={y_finger_full.shape} meta_keys={meta_keys}"
    )

    # Global index mapping (original dataset positions)
    global_indices = np.arange(len(y_action_full), dtype=np.int64)

    # ===== OPTIONAL SUBJECT FILTER =====
    X = X_full
    y_action = y_action_full
    y_finger = y_finger_full
    n_full = len(y_action_full)
    subject = None

    if args.subject_id:
        if "subject_id" not in meta:
            print("subject_id not found in dataset metadata; cannot filter.")
            print(f"NPZ path: {npz_path}")
            print(f"Available meta keys: {sorted(meta.keys())}")
            return 2
        try:
            subject_ids = np.asarray(meta["subject_id"])
            subject_ids_dtype = subject_ids.dtype
        except Exception:
            print("subject_id metadata could not be read; cannot filter.")
            print(f"NPZ path: {npz_path}")
            return 2
        if subject_ids.ndim == 0 or len(subject_ids) != n_full:
            print("subject_id metadata length mismatch; cannot filter.")
            print(f"NPZ path: {npz_path}")
            print(
                f"subject_id length: {len(subject_ids) if subject_ids.ndim != 0 else 0}, N={n_full}"
            )
            return 2
        subject_ids = subject_ids.astype("U")
        mask = subject_ids == args.subject_id
        kept = int(mask.sum())
        print(f"Subject filter: requested={args.subject_id} kept {kept}/{len(mask)}")
        if kept == 0:
            unique, counts = np.unique(subject_ids, return_counts=True)
            pairs = list(zip(unique.tolist(), counts.tolist()))[:20]
            print(f"No windows found for subject_id={args.subject_id}")
            print(f"NPZ path: {npz_path}")
            print(f"Unique subject_ids (up to 20): {pairs}")
            print(f"meta['subject_id'] dtype: {subject_ids_dtype}")
            print(f"Total N: {len(subject_ids)}")
            print(f"Requested subject_id: {args.subject_id}")
            return 2

        X = X[mask]
        y_action = y_action[mask]
        y_finger = y_finger[mask]
        global_indices = global_indices[mask]  # critical mapping
        meta, masked_keys = mask_meta(meta, mask, n_full)
        kept = len(y_action)
        for key in masked_keys:
            try:
                val = meta.get(key)
                if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) != kept:
                    raise AssertionError(
                        f"Meta length mismatch for {key}: {len(val)} != {kept}"
                    )
                if (
                    not isinstance(val, np.ndarray)
                    and hasattr(val, "__len__")
                    and len(val) != kept
                ):
                    raise AssertionError(
                        f"Meta length mismatch for {key}: {len(val)} != {kept}"
                    )
            except Exception as exc:
                print(str(exc))
                return 2
        subject = args.subject_id
    else:
        inferred_subject = _subject_from_root_meta()
        if not inferred_subject:
            inferred_subject = infer_subject_id_from_meta(meta, n_full)
        if not inferred_subject:
            inferred_subject = infer_subject_id_from_npz(npz_path)
        subject = inferred_subject or subject_id
        print(f"Subject filter: not applied; using subject_id={subject}")

    if len(y_action) == 0 or len(y_finger) == 0:
        print("No windows available after subject filtering.")
        return 2


    # ======================
    # NaN / Inf safety filter
    # ======================
    # We must never feed NaNs/Infs into PyTorch. Drop any samples whose feature tensor contains
    # non-finite values. (Window extraction already tries to avoid this, but we harden here.)
    finite_mask = np.isfinite(X).all(axis=tuple(range(1, X.ndim)))
    n_bad = int((~finite_mask).sum())
    if n_bad > 0:
        print(f"[WARN] Dropping {n_bad}/{len(X)} samples with NaN/Inf in X before training.")
        X = X[finite_mask]
        y_action = y_action[finite_mask]
        y_finger = y_finger[finite_mask]
        global_indices = global_indices[finite_mask]
        meta, _ = mask_meta(meta, finite_mask, len(finite_mask))
        if len(X) == 0:
            raise RuntimeError("All samples were dropped due to NaN/Inf values. Check upstream data collection.")

        def class_counts(y):
            u, c = np.unique(y, return_counts=True)
            return dict(zip(u.tolist(), c.tolist()))

        print(f"Action class counts: {class_counts(y_action)}")
        print(f"Finger class counts: {class_counts(y_finger)}")

        exp_hash = resolve_experiment_hash(meta, len(y_action))
        log_experiment(subject, exp_hash, "STEP_2_TRAIN")

        run_dir, save_model_path, save_scaler_path, save_preds_path = resolve_output_paths(
            args, subject, exp_hash
        )
        for path in [save_model_path, save_scaler_path, save_preds_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Output paths: model={save_model_path}, scaler={save_scaler_path}, preds={save_preds_path}"
        )

        # ===== SPLIT =====
        train_idx, test_idx = split_indices(
            y_action,
            y_finger,
            meta=meta if meta else None,
            test_size=args.test_size,
            random_state=args.seed,
        )

        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        n_samples = len(y_action)
        _validate_indices(train_idx, n_samples, "train")
        _validate_indices(test_idx, n_samples, "test")

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

        n_fingers = int(np.max(y_finger)) + 1
        n_actions = int(np.max(y_action)) + 1

        # ===== NORMALIZE =====
        normalizer = fit_channel_normalizer(X_train)
        X_train = apply_channel_normalizer(X_train, normalizer)
        X_test = apply_channel_normalizer(X_test, normalizer)
        joblib.dump(normalizer, str(save_scaler_path))

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

        # ===== MODEL =====
        model = CNNLSTMFingerActionNet(
            n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions
        )
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
                    f"Epoch {epoch + 1:03d}/{args.epochs} | loss={avg_loss:.4f} "
                    f"action_acc={action_acc:.3f} finger_acc={finger_acc:.3f}"
                )

        # ===== SAVE MODEL =====
        model.eval()
        torch.save(model.state_dict(), str(save_model_path))

        train_config = {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.lr,
            "loss_action_weight": args.loss_action_weight,
            "rest_weight": float(args.rest_weight),
            "test_size": args.test_size,
            "non_rest_only": bool(args.non_rest_only),
            "npz_path": str(npz_path),
            "n_fingers": n_fingers,
            "n_actions": n_actions,
            "input_shape": list(X.shape[1:]),
            "normalizer": {
                "type": normalizer.get("type", "unknown"),
                "channels": normalizer.get("channels", None),
            },
            "device": str(device),
            "model": "CNNLSTMFingerActionNet",
            "subject_id_filter": args.subject_id or "",
            "save_model_path": str(save_model_path),
            "save_scaler_path": str(save_scaler_path),
            "save_preds_path": str(save_preds_path),
        }
        train_config_path = save_model_path.parent / "train_config.json"
        train_config_path.write_text(json.dumps(train_config, indent=2))

        log_config_path = Path("logs") / "experiments" / f"{exp_hash}_train_config.json"
        log_config_path.parent.mkdir(parents=True, exist_ok=True)
        log_config_path.write_text(json.dumps(train_config, indent=2))

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
        test_indices_global = global_indices[test_idx].astype(
            np.int64
        )  # maps back to original NPZ index

        # Save *test-set* meta (aligned to prediction rows)
        n_expected = len(y_action)
        test_window_start = take_meta(
            meta, ["window_start", "start_s", "onset_s"], test_idx, n_expected, np.float32
        )
        test_window_end = take_meta(
            meta, ["window_end", "end_s", "offset_s"], test_idx, n_expected, np.float32
        )
        test_trial_id = take_meta(
            meta, ["trial_id", "trial", "event_trial_id"], test_idx, n_expected, np.int64
        )
        test_block_id = take_meta(
            meta, ["block_id", "block", "event_block_id"], test_idx, n_expected, np.int64
        )
        test_subject_id = take_meta(meta, ["subject_id"], test_idx, n_expected, "U")
        test_experiment_hash = take_meta(
            meta, ["experiment_hash", "exp_hash"], test_idx, n_expected, "U"
        )

        dataset_info = {
            "npz_path": safe_resolve(npz_path),
            "npz_sha256": sha256_file(npz_path) if npz_path.exists() else None,
            "npz_size_bytes": npz_path.stat().st_size if npz_path.exists() else None,
            "experiment_hash": str(exp_hash),
            "n_samples": int(len(y_action)),
            "filters": {
                "subject_id": args.subject_id or "",
                "max_samples": None,
            },
            "created_utc": now_utc_iso(),
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

        if test_window_start is not None:
            save_dict["window_start"] = test_window_start
        if test_window_end is not None:
            save_dict["window_end"] = test_window_end
        if test_trial_id is not None:
            save_dict["trial_id"] = test_trial_id
        if test_block_id is not None:
            save_dict["block_id"] = test_block_id
        if test_subject_id is not None:
            save_dict["subject_id"] = test_subject_id
        if test_experiment_hash is not None:
            save_dict["experiment_hash"] = test_experiment_hash

        np.savez_compressed(str(save_preds_path), **save_dict)

        log_experiment(subject, exp_hash, "STEP_2_COMPLETE", f"loss={avg_loss:.4f}")
        print("✅ Training complete")
        print(f"DECISION: TRAINED (epochs={args.epochs})")
        print(f"✅ Saved: {save_model_path}, {save_scaler_path}, {save_preds_path}")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())