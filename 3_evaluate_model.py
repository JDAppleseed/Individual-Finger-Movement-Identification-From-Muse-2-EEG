"""
STEP 3 — Model Evaluation + Confidence Calibration
ISEF / Paper-Ready
(Deterministic evaluation — Dropout OFF)
"""

# Work around duplicate libomp on macOS (MKL/torch/scipy); must be set before imports.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
from collections import Counter
import hashlib
import json
import platform
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.label_schema import (
    ACTION_REST,
    ACTION_NAMES,
    FINGER_NAMES,
    enforce_prediction_pairs,
)
from utils.eval_utils import (
    resolve_cached_test_indices,
    validate_cached_predictions_with_dataset_info,
)
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    apply_channel_normalizer,
)
from utils.splitting import resolve_auxiliary_rest_sessions
from utils.postprocess import (
    PostprocessSettings,
    PostprocessState,
    postprocess_predictions,
)
from utils.runtime_utils import load_normalizer
from utils.runtime_utils import apply_temperature_to_logits, load_temperature_scaling
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir

# Fixed label ordering for standardized confusion matrices
ACTION_LABELS = sorted(ACTION_NAMES.keys())
ACTION_TICK_LABELS = [ACTION_NAMES[label] for label in ACTION_LABELS]
FINGER_LABELS = sorted(FINGER_NAMES.keys())
FINGER_TICK_LABELS = [FINGER_NAMES[label] for label in FINGER_LABELS]
REPEATED_SPLIT_SEEDS = (7, 21, 42, 84, 168)

# Pipeline handoff: Step 3 reads Step 2 artifacts from one run directory and writes
# calibrated metrics/plots/manifests back into that same run-scoped report path.
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
    """
    Accepts a run dir like:
      .../sessions/<session_id>/processed/models/<run_id>
    Returns the session directory if that structure is present.
    """
    p = resolve_session_dir(run_dir)
    if p.name == "models":
        # models root is .../<session_id>/processed/models
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

N_BINS = 10
SEED = 42
SHOW_PLOTS = os.environ.get("SHOW_PLOTS", "0") == "1"
MIN_TEST_SAMPLES = 30
MAX_SPLIT_ATTEMPTS = 8
DEFAULT_BATCH_SIZE = 256


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


def numpy_sha256(arr: np.ndarray) -> Optional[str]:
    try:
        contiguous = np.ascontiguousarray(arr)
        return hashlib.sha256(contiguous.view(np.uint8)).hexdigest()
    except Exception:
        return None


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_dataset_info(
    *,
    npz_path: Path,
    experiment_hash: str,
    n_samples: int,
    subject_id: str,
    max_samples: Optional[int],
) -> Dict[str, Any]:
    npz_sha = sha256_file(npz_path) if npz_path.exists() else None
    npz_size = npz_path.stat().st_size if npz_path.exists() else None
    return {
        "npz_path": safe_resolve(npz_path),
        "npz_sha256": npz_sha,
        "npz_size_bytes": npz_size,
        "experiment_hash": experiment_hash,
        "n_samples": int(n_samples),
        "filters": {
            "subject_id": subject_id,
            "max_samples": int(max_samples) if max_samples is not None else None,
        },
        "created_utc": now_utc_iso(),
    }


def _parse_dataset_info(payload: dict) -> Optional[Dict[str, Any]]:
    if "dataset_info" not in payload:
        return None
    info = payload.get("dataset_info")
    if info is None:
        return None
    if isinstance(info, dict):
        return info
    if isinstance(info, np.ndarray):
        if info.size == 0:
            return None
        value = info.flat[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            try:
                return json.loads(value)
            except Exception:
                return None
    if isinstance(info, bytes):
        try:
            return json.loads(info.decode("utf-8", errors="ignore"))
        except Exception:
            return None
    if isinstance(info, str):
        try:
            return json.loads(info)
        except Exception:
            return None
    return None


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


def apply_postprocess_sequence(
    action_probs, finger_probs, order, settings, trial_ids=None
):
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
        if "test_indices_local" in d.files:
            out["test_indices_local"] = d["test_indices_local"]

        if "dataset_info" in d.files:
            out["dataset_info"] = d["dataset_info"]

        # Optional metadata aligned to rows of probs/labels
        for k in [
            "window_start",
            "window_end",
            "trial_id",
            "block_id",
            "subject_id",
            "experiment_hash",
        ]:
            if k in d.files:
                out[k] = d[k]

        return out
    except Exception:
        return None


def _load_train_config(run_dir: Path) -> Dict[str, Any]:
    path = run_dir / "train_config.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resolve_temperature_path(run_dir: Path, train_cfg: Dict[str, Any]) -> Path:
    candidate = train_cfg.get("save_temperature_path")
    if candidate:
        return Path(str(candidate)).expanduser()
    return run_dir / "temperature_scaling.json"


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


def _split_with_checks(
    y_action,
    y_finger,
    meta,
    seed: int,
    test_size: float,
    split_mode: str,
    purge_seconds: float,
    hop_seconds: Optional[float],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
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
            return train_idx, test_idx, attempt + 1

    return None, None, MAX_SPLIT_ATTEMPTS


def _subset_meta(meta: Dict[str, Any], idx: np.ndarray, n_expected: int) -> Dict[str, Any]:
    if not meta:
        return {}
    idx = np.asarray(idx, dtype=np.int64)
    out: Dict[str, Any] = {}
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
            if len(arr) == n_expected:
                out[key] = arr[idx]
            else:
                out[key] = val
        except Exception:
            out[key] = val
    return out


def _load_eval_model_and_normalizer(
    *,
    model_path: Path,
    scaler_path: Path,
    temperature_path: Path,
    n_channels: int,
    n_fingers: int,
    n_actions: int,
) -> Tuple[CNNLSTMFingerActionNet, Dict[str, Any], Optional[Any]]:
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler/normalizer file: {scaler_path}")
    normalizer = load_normalizer(scaler_path)
    if normalizer is None:
        raise ValueError(f"Failed to load normalizer: {scaler_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model weights: {model_path}")

    model = CNNLSTMFingerActionNet(
        n_channels=n_channels,
        n_fingers=n_fingers,
        n_actions=n_actions,
    )
    model.load_state_dict(
        torch.load(str(model_path), map_location="cpu", weights_only=True)
    )
    model.eval()
    temperature_state = load_temperature_scaling(temperature_path)
    eval_model = None
    eval_normalizer = None
    return model, normalizer, temperature_state


def _run_deterministic_inference(
    *,
    X: np.ndarray,
    indices: np.ndarray,
    model: CNNLSTMFingerActionNet,
    normalizer: Dict[str, Any],
    temperature_state: Optional[Any],
    n_actions: int,
    n_fingers: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(indices, dtype=np.int64)
    action_probs = np.zeros((len(idx), n_actions), dtype=np.float32)
    finger_probs = np.zeros((len(idx), n_fingers), dtype=np.float32)

    with torch.no_grad():
        for start in range(0, len(idx), batch_size):
            end = min(start + batch_size, len(idx))
            batch_idx = idx[start:end]
            X_batch = np.asarray(X[batch_idx], dtype=np.float32)
            X_batch = apply_channel_normalizer(X_batch, normalizer)
            X_t = torch.tensor(X_batch, dtype=torch.float32)
            finger_logits, action_logits = model(X_t)
            if temperature_state is not None:
                action_logits = apply_temperature_to_logits(
                    action_logits, temperature_state.action_temperature
                )
                finger_logits = apply_temperature_to_logits(
                    finger_logits, temperature_state.finger_temperature
                )
            action_probs[start:end] = torch.softmax(action_logits, dim=1).cpu().numpy()
            finger_probs[start:end] = torch.softmax(finger_logits, dim=1).cpu().numpy()

    return action_probs, finger_probs


def _format_pair_counts(action_ids: np.ndarray, finger_ids: np.ndarray) -> Dict[str, int]:
    action_ids = np.asarray(action_ids, dtype=np.int64).reshape(-1)
    finger_ids = np.asarray(finger_ids, dtype=np.int64).reshape(-1)
    counter = Counter(zip(action_ids.tolist(), finger_ids.tolist()))
    return {
        f"{ACTION_NAMES.get(int(a), str(a))}+{FINGER_NAMES.get(int(f), str(f))}": int(count)
        for (a, f), count in sorted(
            counter.items(),
            key=lambda item: (-item[1], int(item[0][0]), int(item[0][1])),
        )
    }


def _compute_prediction_metrics(
    *,
    action_probs: np.ndarray,
    finger_probs: np.ndarray,
    y_action_true: np.ndarray,
    y_finger_true: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    y_action_true = np.asarray(y_action_true, dtype=np.int64).reshape(-1)
    y_finger_true = np.asarray(y_finger_true, dtype=np.int64).reshape(-1)
    action_probs = np.asarray(action_probs, dtype=np.float32)
    finger_probs = np.asarray(finger_probs, dtype=np.float32)

    action_preds = np.argmax(action_probs, axis=1).astype(np.int64)
    raw_finger_preds = np.argmax(finger_probs, axis=1).astype(np.int64)
    raw_valid_pair_rate = (
        float(np.mean((action_preds != ACTION_REST) | (raw_finger_preds == 0)))
        if len(action_preds)
        else None
    )
    raw_invalid_pair_rate = (
        float(1.0 - raw_valid_pair_rate) if raw_valid_pair_rate is not None else None
    )
    _, finger_preds = enforce_prediction_pairs(action_preds, raw_finger_preds)
    action_conf = action_probs[np.arange(len(action_probs)), action_preds]
    finger_conf = finger_probs[np.arange(len(finger_probs)), finger_preds]

    mask_non_rest = y_action_true != ACTION_REST
    joint_correct = (action_preds == y_action_true) & (finger_preds == y_finger_true)
    rest_mask = y_action_true == ACTION_REST

    rest_tpr = None
    rest_precision = None
    rest_fpr = None
    rest_f1 = None
    if np.any(rest_mask):
        rest_tp = int(np.sum(rest_mask & (action_preds == ACTION_REST)))
        rest_fn = int(np.sum(rest_mask & (action_preds != ACTION_REST)))
        rest_fp = int(np.sum(~rest_mask & (action_preds == ACTION_REST)))
        rest_tn = int(np.sum(~rest_mask & (action_preds != ACTION_REST)))
        rest_tpr = float(rest_tp / (rest_tp + rest_fn)) if (rest_tp + rest_fn) else None
        rest_fpr = float(rest_fp / (rest_fp + rest_tn)) if (rest_fp + rest_tn) else None
        rest_precision = (
            float(rest_tp / (rest_tp + rest_fp)) if (rest_tp + rest_fp) else None
        )
        if rest_tpr is not None and rest_precision is not None:
            denom = rest_tpr + rest_precision
            rest_f1 = float(2.0 * rest_tpr * rest_precision / denom) if denom else None

    metrics = {
        "raw_valid_pair_rate": raw_valid_pair_rate,
        "raw_invalid_pair_rate": raw_invalid_pair_rate,
        "action_acc": float(accuracy_score(y_action_true, action_preds)) if len(y_action_true) else None,
        "action_f1_macro": float(
            f1_score(y_action_true, action_preds, average="macro", zero_division=0)
        )
        if len(y_action_true)
        else None,
        "action_f1_weighted": float(
            f1_score(y_action_true, action_preds, average="weighted", zero_division=0)
        )
        if len(y_action_true)
        else None,
        "joint_acc": float(np.mean(joint_correct)) if len(joint_correct) else None,
        "joint_acc_non_rest": float(np.mean(joint_correct[mask_non_rest])) if np.any(mask_non_rest) else None,
        "finger_acc_non_rest": float(accuracy_score(y_finger_true[mask_non_rest], finger_preds[mask_non_rest]))
        if np.any(mask_non_rest)
        else None,
        "finger_acc_overall": float(accuracy_score(y_finger_true, finger_preds)) if len(y_finger_true) else None,
        "rest_tpr": rest_tpr,
        "rest_fpr": rest_fpr,
        "rest_precision": rest_precision,
        "rest_f1": rest_f1,
        "action_ece": float(expected_calibration_error(action_conf, action_preds, y_action_true, n_bins))
        if len(y_action_true)
        else None,
        "finger_ece_non_rest": float(
            expected_calibration_error(
                finger_conf[mask_non_rest],
                finger_preds[mask_non_rest],
                y_finger_true[mask_non_rest],
                n_bins,
            )
        )
        if np.any(mask_non_rest)
        else None,
    }
    return {
        "metrics": metrics,
        "action_preds": action_preds,
        "finger_preds": finger_preds,
        "raw_finger_preds": raw_finger_preds,
        "action_conf": action_conf,
        "finger_conf": finger_conf,
        "pair_counts": _format_pair_counts(action_preds, finger_preds),
    }


def _build_primary_benchmark(
    *,
    y_action_test: np.ndarray,
    y_finger_test: np.ndarray,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "test_n": int(len(y_action_test)),
        "test_action_counts": _format_label_counts(y_action_test, ACTION_NAMES),
        "test_finger_counts": _format_label_counts(y_finger_test, FINGER_NAMES),
        "metrics": {
            key: (float(value) if isinstance(value, (np.floating, float)) and value is not None else value)
            for key, value in metrics.items()
            if key
            in {
                "action_acc",
                "joint_acc",
                "joint_acc_non_rest",
                "finger_acc_non_rest",
                "rest_tpr",
                "rest_precision",
                "action_ece",
                "raw_invalid_pair_rate",
            }
        },
    }


def _build_rest_event_breakdown(
    *,
    indices: np.ndarray,
    y_action_true: np.ndarray,
    action_probs: np.ndarray,
    action_preds: np.ndarray,
    finger_preds: np.ndarray,
    meta: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    if not meta:
        return [], []
    if "event_id" not in meta or "session_id" not in meta:
        return [], []

    event_id = np.asarray(meta["event_id"])[indices].astype(np.int64)
    session_id = np.asarray(meta["session_id"])[indices].astype("U")
    window_start = (
        np.asarray(meta["window_start"])[indices].astype(np.float32)
        if "window_start" in meta
        else None
    )
    window_end = (
        np.asarray(meta["window_end"])[indices].astype(np.float32)
        if "window_end" in meta
        else None
    )

    rest_mask = np.asarray(y_action_true, dtype=np.int64) == ACTION_REST
    rest_event_ids = np.unique(event_id[rest_mask]) if np.any(rest_mask) else np.array([], dtype=np.int64)
    breakdown: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []
    for eid in rest_event_ids.tolist():
        event_mask = rest_mask & (event_id == int(eid))
        if not np.any(event_mask):
            continue
        event_action_preds = action_preds[event_mask]
        event_finger_preds = finger_preds[event_mask]
        event_probs = action_probs[event_mask]
        rest_tpr = float(np.mean(event_action_preds == ACTION_REST))
        pair_counts = _format_pair_counts(event_action_preds, event_finger_preds)
        non_rest_pair_counts = {
            key: value
            for key, value in pair_counts.items()
            if not key.startswith(f"{ACTION_NAMES.get(ACTION_REST, 'REST')}+")
        }
        dominant_pair = None
        dominant_pair_count = 0
        if non_rest_pair_counts:
            dominant_pair, dominant_pair_count = max(
                non_rest_pair_counts.items(), key=lambda item: item[1]
            )
        dominant_pair_share = (
            float(dominant_pair_count / int(np.sum(event_mask))) if dominant_pair_count else 0.0
        )
        entry = {
            "event_id": int(eid),
            "session_id": str(session_id[event_mask][0]),
            "window_count": int(np.sum(event_mask)),
            "rest_tpr": rest_tpr,
            "pred_action_counts": _format_label_counts(event_action_preds, ACTION_NAMES),
            "pred_pair_counts": pair_counts,
            "median_p_rest": float(np.median(event_probs[:, ACTION_REST])),
            "median_p_open": float(np.median(event_probs[:, 1])) if event_probs.shape[1] > 1 else None,
            "median_p_close": float(np.median(event_probs[:, 2])) if event_probs.shape[1] > 2 else None,
            "window_start": float(np.min(window_start[event_mask])) if window_start is not None else None,
            "window_end": float(np.max(window_end[event_mask])) if window_end is not None else None,
        }
        breakdown.append(entry)
        if rest_tpr < 0.20 or dominant_pair_share >= 0.80:
            flags.append(
                {
                    "event_id": int(eid),
                    "session_id": str(session_id[event_mask][0]),
                    "window_count": int(np.sum(event_mask)),
                    "rest_tpr": rest_tpr,
                    "dominant_non_rest_pair": dominant_pair,
                    "dominant_non_rest_pair_share": dominant_pair_share,
                    "recommended_action": "relabel" if dominant_pair_share >= 0.80 else "prune",
                    "reason": (
                        f"dominant_non_rest_pair={dominant_pair}"
                        if dominant_pair_share >= 0.80
                        else "low_rest_tpr"
                    ),
                }
            )
    breakdown.sort(key=lambda item: item["event_id"])
    flags.sort(key=lambda item: (item["recommended_action"], item["event_id"]))
    return breakdown, flags


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 3: evaluate a trained Step 2 run, optionally apply postprocessing, "
            "and write calibrated metrics, plots, and manifests."
        )
    )
    input_group = parser.add_argument_group("input selection")
    input_group.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Load evaluation settings from a JSON config file.",
    )
    input_group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Specific Step 2 run directory to evaluate (for example: .../processed/models/<run_id>).",
    )
    input_group.add_argument(
        "--project",
        type=str,
        required=False,
        metavar="NAME",
        help="Project identifier used to resolve a session when --run-dir is not provided.",
    )
    input_group.add_argument(
        "--subject",
        type=str,
        required=False,
        metavar="ID",
        help="Subject identifier used to resolve a session when --run-dir is not provided.",
    )
    input_group.add_argument(
        "--session",
        type=str,
        required=False,
        metavar="ID",
        help="Session identifier to evaluate. Defaults to the latest session for the subject.",
    )
    input_group.add_argument(
        "--session-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Legacy session directory override used by the UI.",
    )
    input_group.add_argument(
        "--subject-id",
        type=str,
        default="",
        metavar="ID",
        help="Filter the evaluation dataset to a single subject_id.",
    )
    runtime_group = parser.add_argument_group("runtime and outputs")
    runtime_group.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of windows evaluated. Useful as a memory guard.",
    )
    runtime_group.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Inference batch size.",
    )
    runtime_group.add_argument(
        "--save-manifest",
        type=str,
        default=None,
        metavar="PATH",
        help="Write an evaluation manifest JSON to this path.",
    )
    runtime_group.add_argument(
        "--no-manifest", action="store_true", help="Disable manifest output"
    )
    runtime_group.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Enable deterministic evaluation where possible.",
    )
    runtime_group.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_false",
        help="Allow non-deterministic execution for speed or platform compatibility.",
    )
    split_group = parser.add_argument_group("split overrides")
    split_group.add_argument(
        "--split-seed",
        type=int,
        default=SEED,
        help="Random seed used when rebuilding the evaluation split.",
    )
    split_group.add_argument(
        "--test-size",
        type=float,
        default=None,
        help="Fraction reserved for the test split. Defaults to train_config.json when available.",
    )
    split_group.add_argument(
        "--split-mode",
        type=str,
        default=None,
        choices=["group_trial", "holdout_session"],
        help="Split strategy to use when rebuilding the split. Defaults to train_config.json when available.",
    )
    split_group.add_argument(
        "--purge-seconds",
        type=float,
        default=None,
        help="Drop training windows within this many seconds of any test window from the same session.",
    )
    split_group.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="Window hop size, in seconds, used by leakage-purge heuristics when needed.",
    )
    split_group.add_argument(
        "--export-test-pred",
        action="store_true",
        help="Export cached predictions for the test split.",
    )
    post_group = parser.add_argument_group("postprocessing")
    post_group.add_argument(
        "--smooth-action-only",
        action="store_true",
        help="Smooth only the action head. Finger predictions stay raw except REST forces NONE.",
    )

    post_group.add_argument(
        "--smooth", action="store_true", help="Enable postprocess smoothing"
    )
    post_group.add_argument(
        "--smooth-method",
        type=str,
        default="vote",
        choices=["vote", "ema"],
        help="Smoothing method used when --smooth is enabled.",
    )
    post_group.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Window size used by the smoothing stage.",
    )
    post_group.add_argument(
        "--hysteresis", action="store_true", help="Enable action hysteresis"
    )
    post_group.add_argument(
        "--hysteresis-frames",
        type=int,
        default=3,
        help="Number of consecutive frames required by hysteresis.",
    )
    post_group.add_argument(
        "--threshold-action",
        type=float,
        default=0.75,
        help="Minimum action confidence required after postprocessing.",
    )
    post_group.add_argument(
        "--threshold-finger",
        type=float,
        default=0.75,
        help="Minimum finger confidence required after postprocessing.",
    )
    post_group.add_argument(
        "--adjacency",
        action="store_true",
        help="Enable finger adjacency correction during postprocessing.",
    )
    args = parser.parse_args()
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    settings = _load_config(args.config)
    _apply_config_to_args(args, settings, defaults)
    cli_flags = {
        "split_seed": "--split-seed" in sys.argv,
        "test_size": "--test-size" in sys.argv,
        "split_mode": "--split-mode" in sys.argv,
        "purge_seconds": "--purge-seconds" in sys.argv,
        "hop_seconds": "--hop-seconds" in sys.argv,
    }
    if args.smooth_action_only and not args.smooth:
        args.smooth = True
    split_seed = int(args.split_seed) if args.split_seed is not None else SEED
    if args.deterministic:
        random.seed(split_seed)
        np.random.seed(split_seed)
        torch.manual_seed(split_seed)
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    else:
        random.seed(split_seed)
        np.random.seed(split_seed)

    # PATCHED: session-aware path
    legacy_session_dir: Optional[Path] = None
    run_dir_override: Optional[Path] = None
    if args.run_dir:
        run_dir_override = Path(args.run_dir).expanduser()
        if not run_dir_override.exists():
            print(f"Run dir not found: {run_dir_override}")
            return 2
        legacy_session_dir = _session_dir_from_run_dir(run_dir_override)
        if legacy_session_dir is None:
            print(
                "Could not infer session dir from --run-dir; "
                "pass --project/--subject/--session or use --session-dir."
            )
            return 2
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
        print(
            "Missing --project/--subject (or provide --session-dir under Projects/.../subjects/.../sessions/...)."
        )
        return 2

    if args.session is None:
        args.session = latest_session_for_subject(args.project, args.subject)
    if args.session is None:
        print(
            f"No session found for subject {args.subject} in project {args.project}."
        )
        return 2
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
    pred_npz_path = Path(paths.test_predictions_npz).expanduser()
    model_path = Path(paths.model_file).expanduser()
    scaler_path = Path(paths.scaler_file).expanduser()
    if not npz_path.exists():
        print(f"NPZ file not found: {npz_path}")
        return 2
    train_cfg = _load_train_config(model_path.parent)
    temperature_path = _resolve_temperature_path(model_path.parent, train_cfg)
    split_seed_effective = (
        int(args.split_seed)
        if cli_flags["split_seed"] and args.split_seed is not None
        else int(train_cfg.get("seed", split_seed) or split_seed)
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
    aux_rest_session_policy = str(
        train_cfg.get("aux_rest_session_policy", "none") or "none"
    ).strip()
    if split_seed_effective != split_seed:
        split_seed = int(split_seed_effective)
        if args.deterministic:
            random.seed(split_seed)
            np.random.seed(split_seed)
            torch.manual_seed(split_seed)
        else:
            random.seed(split_seed)
            np.random.seed(split_seed)
    print(
        "Split config: "
        f"test_size={test_size}, seed={split_seed}, mode={split_mode}, "
        f"purge_seconds={purge_seconds}, hop_seconds={hop_seconds}, "
        f"aux_rest_session_policy={aux_rest_session_policy}"
    )
    manifest_enabled = not args.no_manifest
    manifest_name = Path(args.save_manifest).name if args.save_manifest else "eval_manifest.json"
    manifest_path = Path(paths.eval_dir) / manifest_name
    print(f"Saving report/manifest to: {manifest_path}")

    # Manifest ties metrics back to concrete files/hashes so later report/QA steps
    # can verify they are using the same model/scaler/NPZ snapshot.
    def _path_info(path: Path, used: Optional[bool] = None) -> Dict[str, Any]:
        info: Dict[str, Any] = {
            "path": safe_resolve(path),
            "sha256": sha256_file(path) if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        }
        if used is not None:
            info["used"] = used
        return info

    manifest: Dict[str, Any] = {
        "ts_utc": now_utc_iso(),
        "command": sys.argv,
        "paths": {
            "npz": _path_info(npz_path),
            "model": _path_info(model_path),
            "scaler": _path_info(scaler_path),
            "temperature_scaling": _path_info(temperature_path),
            "pred_npz": _path_info(pred_npz_path, used=False),
        },
        "filters": {
            "subject_id": args.subject_id,
            "subject_filtered": False,
            "max_samples": args.max_samples,
            "batch_size": int(args.batch_size),
        },
        "dataset": {
            "n_total": None,
            "n_after_filter": None,
            "n_actions": None,
            "n_fingers": None,
            "exp_hash": None,
            "label_counts": {
                "action": None,
                "finger": None,
                "finger_non_rest": None,
            },
            "meta_keys": None,
        },
        "split": {
            "seed": split_seed,
            "test_size": float(test_size),
            "split_mode": str(split_mode),
            "aux_rest_session_policy": str(aux_rest_session_policy),
            "purge_seconds": float(purge_seconds),
            "hop_seconds": float(hop_seconds) if hop_seconds is not None else None,
            "attempts": None,
            "train_n": None,
            "test_n": None,
            "train_idx_sha256": None,
            "test_idx_sha256": None,
        },
        "postprocess": {
            "smooth": bool(args.smooth),
            "smooth_method": args.smooth_method,
            "smooth_window": int(args.smooth_window),
            "hysteresis": bool(args.hysteresis),
            "hysteresis_frames": int(args.hysteresis_frames),
            "threshold_action": float(args.threshold_action),
            "threshold_finger": float(args.threshold_finger),
            "adjacency": bool(args.adjacency),
            "smooth_action_only": bool(args.smooth_action_only),
        },
        "metrics": {
            "raw_valid_pair_rate": None,
            "raw_invalid_pair_rate": None,
            "action_acc": None,
            "action_f1_macro": None,
            "action_f1_weighted": None,
            "joint_acc": None,
            "joint_acc_non_rest": None,
            "finger_acc_non_rest": None,
            "finger_acc_overall": None,
            "finger_f1_non_rest_macro": None,
            "finger_f1_non_rest_weighted": None,
            "finger_f1_overall_macro": None,
            "finger_f1_overall_weighted": None,
            "rest_tpr": None,
            "rest_fpr": None,
            "rest_precision": None,
            "rest_f1": None,
            "action_ece": None,
            "finger_ece_non_rest": None,
            "smoothed_action_acc": None,
            "smoothed_action_f1_macro": None,
            "smoothed_action_f1_weighted": None,
            "smoothed_joint_acc": None,
            "smoothed_joint_acc_non_rest": None,
            "smoothed_finger_acc_non_rest": None,
            "smoothed_finger_acc_overall": None,
            "smoothed_finger_f1_non_rest_macro": None,
            "smoothed_finger_f1_non_rest_weighted": None,
            "smoothed_finger_f1_overall_macro": None,
            "smoothed_finger_f1_overall_weighted": None,
            "smoothed_rest_tpr": None,
            "smoothed_rest_fpr": None,
            "smoothed_rest_precision": None,
            "smoothed_rest_f1": None,
        },
        "warnings": [],
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }

    def _maybe_write_manifest() -> None:
        if not manifest_enabled:
            return
        try:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except Exception as exc:
            print(f"⚠️ Failed to write manifest: {exc}")

    # =========================
    # ===== LOAD DATA =========
    # =========================
    X, y_action, y_finger, meta = load_sequence_npz(str(npz_path), mmap_mode="r")
    n_total = int(len(y_action))
    manifest["dataset"]["n_total"] = n_total
    if args.subject_id is not None and args.subject_id.strip() != "":
        subject_id_filter = args.subject_id.strip()
        if "subject_id" not in meta:
            print("subject_id not found in dataset metadata; cannot filter.")
            manifest["filters"]["subject_filtered"] = False
            manifest["filters"]["subject_id"] = subject_id_filter
            manifest["abort_reason"] = "subject_id_not_in_meta"
            _maybe_write_manifest()
            return 2
        subject_ids_all = np.asarray(meta["subject_id"]).astype(str)
        mask = subject_ids_all == subject_id_filter
        kept = int(mask.sum())
        if kept == 0:
            print(f"No windows found for subject_id={subject_id_filter}")
            manifest["filters"]["subject_filtered"] = False
            manifest["filters"]["subject_id"] = subject_id_filter
            manifest["abort_reason"] = "subject_id_no_windows"
            _maybe_write_manifest()
            return 2
        X = X[mask]
        y_action = y_action[mask]
        y_finger = y_finger[mask]
        meta = mask_meta(meta, mask)
        manifest["filters"]["subject_filtered"] = True
        manifest["filters"]["subject_id"] = subject_id_filter
    else:
        manifest["filters"]["subject_filtered"] = False

    if isinstance(X, np.memmap) and X.dtype != np.float32:
        print(f"ℹ️ X dtype is {X.dtype}; casting to float32 per batch.")

    X, y_action, y_finger, meta = _apply_sample_limit(
        X, y_action, y_finger, meta, args.max_samples, split_seed
    )
    manifest["dataset"]["n_after_filter"] = int(len(y_action))
    manifest["dataset"]["meta_keys"] = (
        sorted(list(meta.keys())) if isinstance(meta, dict) else []
    )

    _print_label_summary("Filtered", y_action, y_finger)

    if isinstance(meta, dict) and "session_id" in meta:
        try:
            sess = np.asarray(meta["session_id"]).astype("U")
            sess = sess[sess != ""]
            unique_sessions = np.unique(sess)
        except Exception:
            unique_sessions = np.array([])
        if unique_sessions.size <= 1:
            msg = "single_session_eval"
            print(
                "⚠️ Evaluation uses a single session; accuracy may be optimistic for live inference."
            )
            manifest["warnings"].append(msg)

    subject_ids = meta.get("subject_id", None)
    exp_hash = _first_meta_scalar(
        meta, ["experiment_hash", "exp_hash"], default="UNKNOWN"
    )
    subject_id_filter = args.subject_id.strip() if args.subject_id else ""
    dataset_info_current = _build_dataset_info(
        npz_path=npz_path,
        experiment_hash=str(exp_hash),
        n_samples=len(y_action),
        subject_id=subject_id_filter,
        max_samples=args.max_samples,
    )

    n_fingers = int(np.max(y_finger)) + 1
    n_actions = int(np.max(y_action)) + 1
    manifest["dataset"]["n_actions"] = n_actions
    manifest["dataset"]["n_fingers"] = n_fingers
    manifest["dataset"]["exp_hash"] = exp_hash
    manifest["dataset"]["label_counts"]["action"] = _format_label_counts(
        y_action, ACTION_NAMES
    )
    manifest["dataset"]["label_counts"]["finger"] = _format_label_counts(
        y_finger, FINGER_NAMES
    )
    non_rest_mask = y_action != ACTION_REST
    manifest["dataset"]["label_counts"]["finger_non_rest"] = (
        _format_label_counts(y_finger[non_rest_mask], FINGER_NAMES)
        if np.any(non_rest_mask)
        else "none"
    )

    aux_rest_plan = resolve_auxiliary_rest_sessions(
        y_action,
        meta if meta else {},
        policy=aux_rest_session_policy,
    )
    split_idx = np.asarray(aux_rest_plan["core_idx"], dtype=np.int64)
    aux_idx = np.asarray(aux_rest_plan["aux_idx"], dtype=np.int64)
    manifest["split"]["auxiliary_rest_sessions"] = {
        "enabled": bool(aux_rest_plan.get("enabled", False)),
        "reason": str(aux_rest_plan.get("reason", "")),
        "aux_sessions": [str(v) for v in aux_rest_plan.get("aux_sessions", [])],
        "core_sessions": [str(v) for v in aux_rest_plan.get("core_sessions", [])],
        "aux_count": int(len(aux_idx)),
        "core_count": int(len(split_idx)),
    }
    if aux_rest_plan.get("enabled"):
        print(
            "Auxiliary REST-only sessions excluded from main evaluation split: "
            f"{aux_rest_plan.get('aux_sessions', [])}"
        )
    if len(split_idx) == 0:
        print("⚠️ No core windows remain after applying the auxiliary REST session policy.")
        manifest["abort_reason"] = "no_core_windows_after_aux_rest_filter"
        _maybe_write_manifest()
        return 2

    eval_model = None
    eval_normalizer = None

    # =========================
    # ===== TEST SPLIT =========
    # =========================
    cached = _load_predictions_if_present(pred_npz_path)
    cache_rejected_reasons: Optional[List[str]] = None
    temperature_state = load_temperature_scaling(temperature_path)

    cached_used = False
    split_attempts = 0
    if cached is not None:
        action_probs = np.asarray(cached["action_probs"])
        finger_probs = np.asarray(cached["finger_probs"])
        y_action_test = np.asarray(cached["y_action"])
        y_finger_test = np.asarray(cached["y_finger"])
        test_idx = resolve_cached_test_indices(cached)
        if test_idx is None:
            cache_rejected_reasons = ["missing_test_indices"]
            cached = None
        else:
            test_idx = np.asarray(test_idx).astype(np.int64)

        if cached is not None:
            all_idx = np.arange(len(y_action), dtype=np.int64)
            test_mask = np.zeros(len(y_action), dtype=bool)
            test_mask[test_idx] = True
            train_idx = all_idx[~test_mask]

            dataset_info_cache = _parse_dataset_info(cached)
            cache_ok, reject_reasons = validate_cached_predictions_with_dataset_info(
                action_probs=action_probs,
                finger_probs=finger_probs,
                y_action_test=y_action_test,
                y_finger_test=y_finger_test,
                test_idx=test_idx,
                n_actions=n_actions,
                n_fingers=n_fingers,
                n_samples_current=len(y_action),
                dataset_info_cache=dataset_info_cache,
                dataset_info_current=dataset_info_current,
                y_action_current=y_action,
                y_finger_current=y_finger,
                spotcheck_k=10,
                rng_seed=0,
            )

            if cache_ok and temperature_state is not None:
                cached_action_temp = cached.get("action_temperature")
                cached_finger_temp = cached.get("finger_temperature")
                if cached_action_temp is None or cached_finger_temp is None:
                    cache_ok = False
                    reject_reasons.append("cache_missing_temperature_scaling")
                else:
                    try:
                        action_temp_val = float(np.asarray(cached_action_temp).reshape(-1)[0])
                        finger_temp_val = float(np.asarray(cached_finger_temp).reshape(-1)[0])
                    except Exception:
                        cache_ok = False
                        reject_reasons.append("cache_temperature_scaling_invalid")
                    else:
                        if (
                            abs(action_temp_val - float(temperature_state.action_temperature)) > 1e-6
                            or abs(finger_temp_val - float(temperature_state.finger_temperature)) > 1e-6
                        ):
                            cache_ok = False
                            reject_reasons.append("cache_temperature_scaling_mismatch")

            if cache_ok and aux_rest_plan.get("enabled"):
                if np.any(np.isin(test_idx, aux_idx)):
                    cache_ok = False
                    reject_reasons.append("cache_includes_aux_rest_test_indices")

            if not cache_ok:
                print(
                    "⚠️ Cached predictions rejected; recomputing. "
                    f"Reasons: {reject_reasons}"
                )
                cache_rejected_reasons = reject_reasons
                cached = None
            else:
                cached_used = True
                print(f"✅ Using cached predictions: {pred_npz_path}")

    if cached is None:
        split_meta = _subset_meta(meta, split_idx, len(y_action)) if meta else {}
        train_local_idx, test_local_idx, split_attempts = _split_with_checks(
            y_action[split_idx],
            y_finger[split_idx],
            meta=split_meta,
            seed=split_seed,
            test_size=float(test_size),
            split_mode=str(split_mode),
            purge_seconds=float(purge_seconds),
            hop_seconds=hop_seconds,
        )
        if train_local_idx is None or test_local_idx is None:
            print(
                "⚠️ Unable to create a split with multiple classes. Aborting evaluation."
            )
            manifest["abort_reason"] = "split_failed"
            _maybe_write_manifest()
            return 2
        train_idx = split_idx[np.asarray(train_local_idx, dtype=np.int64)]
        test_idx = split_idx[np.asarray(test_local_idx, dtype=np.int64)]

        y_action_test = y_action[test_idx]
        y_finger_test = y_finger[test_idx]

        try:
            eval_model, eval_normalizer, temperature_state = _load_eval_model_and_normalizer(
                model_path=model_path,
                scaler_path=scaler_path,
                temperature_path=temperature_path,
                n_channels=X.shape[2],
                n_fingers=n_fingers,
                n_actions=n_actions,
            )
        except Exception as exc:
            print(str(exc))
            return 2

        batch_size = max(1, int(args.batch_size))
        action_probs, finger_probs = _run_deterministic_inference(
            X=X,
            indices=test_idx,
            model=eval_model,
            normalizer=eval_normalizer,
            temperature_state=temperature_state,
            n_actions=n_actions,
            n_fingers=n_fingers,
            batch_size=batch_size,
        )

        print("✅ Ran deterministic inference (no cached predictions file found).")

    manifest["paths"]["pred_npz"]["used"] = cached_used
    if cached_used:
        manifest["paths"]["pred_npz"]["sha256"] = (
            sha256_file(pred_npz_path) if pred_npz_path.exists() else None
        )
        manifest["paths"]["pred_npz"]["size_bytes"] = (
            pred_npz_path.stat().st_size if pred_npz_path.exists() else None
        )
    manifest["split"]["attempts"] = None if cached_used else split_attempts
    manifest["split"]["train_n"] = int(len(train_idx))
    manifest["split"]["test_n"] = int(len(test_idx))
    manifest["split"]["train_idx_sha256"] = numpy_sha256(
        np.asarray(train_idx, dtype=np.int64)
    )
    manifest["split"]["test_idx_sha256"] = numpy_sha256(
        np.asarray(test_idx, dtype=np.int64)
    )
    if cache_rejected_reasons is not None:
        manifest["split"]["cache_rejected_reasons"] = cache_rejected_reasons

    aux_rest_benchmark = None
    if aux_rest_plan.get("enabled") and len(aux_idx) > 0:
        if eval_model is None or eval_normalizer is None:
            try:
                eval_model, eval_normalizer, temperature_state = _load_eval_model_and_normalizer(
                    model_path=model_path,
                    scaler_path=scaler_path,
                    temperature_path=temperature_path,
                    n_channels=X.shape[2],
                    n_fingers=n_fingers,
                    n_actions=n_actions,
                )
            except Exception as exc:
                print(f"⚠️ Auxiliary REST benchmark skipped: {exc}")
                eval_model = None
                eval_normalizer = None
        if eval_model is not None and eval_normalizer is not None:
            aux_action_probs, aux_finger_probs = _run_deterministic_inference(
                X=X,
                indices=aux_idx,
                model=eval_model,
                normalizer=eval_normalizer,
                temperature_state=temperature_state,
                n_actions=n_actions,
                n_fingers=n_fingers,
                batch_size=max(1, int(args.batch_size)),
            )
            aux_y_action = y_action[aux_idx]
            aux_y_finger = y_finger[aux_idx]
            aux_action_pred = np.argmax(aux_action_probs, axis=1).astype(np.int64)
            aux_raw_finger_pred = np.argmax(aux_finger_probs, axis=1).astype(np.int64)
            _, aux_finger_pred = enforce_prediction_pairs(
                aux_action_pred, aux_raw_finger_pred
            )
            aux_rest_mask = aux_y_action == ACTION_REST
            aux_rest_tpr = (
                float(np.mean(aux_action_pred[aux_rest_mask] == ACTION_REST))
                if np.any(aux_rest_mask)
                else None
            )
            aux_rest_precision = (
                float(np.mean(aux_y_action[aux_action_pred == ACTION_REST] == ACTION_REST))
                if np.any(aux_action_pred == ACTION_REST)
                else None
            )
            aux_rest_f1 = None
            if aux_rest_tpr is not None and aux_rest_precision is not None:
                denom = aux_rest_tpr + aux_rest_precision
                aux_rest_f1 = (
                    float(2.0 * aux_rest_tpr * aux_rest_precision / denom) if denom else None
                )
            aux_session_ids = (
                np.asarray(meta["session_id"])[aux_idx].astype("U")
                if "session_id" in meta
                else np.array([], dtype="U")
            )
            aux_event_ids = (
                np.asarray(meta["event_id"])[aux_idx].astype(np.int64)
                if "event_id" in meta
                else np.array([], dtype=np.int64)
            )
            aux_rest_benchmark = {
                "n_windows": int(len(aux_idx)),
                "n_rest_events": int(len(np.unique(aux_event_ids))) if aux_event_ids.size else None,
                "sessions": sorted(set(aux_session_ids.tolist())) if aux_session_ids.size else [],
                "action_acc": float(np.mean(aux_action_pred == aux_y_action)) if len(aux_y_action) else None,
                "rest_tpr": aux_rest_tpr,
                "rest_precision": aux_rest_precision,
                "rest_f1": aux_rest_f1,
                "pred_action_counts": _format_label_counts(aux_action_pred, ACTION_NAMES),
                "pred_finger_counts": _format_label_counts(aux_finger_pred, FINGER_NAMES),
                "median_rest_prob": float(np.median(aux_action_probs[:, ACTION_REST]))
                if len(aux_action_probs)
                else None,
            }
            print(
                "\n🧪 Auxiliary REST benchmark: "
                f"Action Acc {aux_rest_benchmark['action_acc'] * 100:.2f}% | "
                f"REST TPR {aux_rest_tpr * 100:.2f}% | "
                f"REST Precision {aux_rest_precision * 100:.2f}%"
            )

    if len(test_idx) < MIN_TEST_SAMPLES:
        print(f"⚠️ Test set too small ({len(test_idx)} samples). Aborting evaluation.")
        manifest["abort_reason"] = "test_set_too_small"
        _maybe_write_manifest()
        return 2

    if args.export_test_pred and not cached_used:
        export_payload = {
            "action_probs": action_probs,
            "finger_probs": finger_probs,
            "y_action": y_action_test,
            "y_finger": y_finger_test,
            "test_indices": np.asarray(test_idx, dtype=np.int64),
            "test_indices_local": np.asarray(test_idx, dtype=np.int64),
            "dataset_info": np.array([json.dumps(dataset_info_current)], dtype="U"),
            "action_temperature": np.array(
                [
                    float(temperature_state.action_temperature)
                    if temperature_state is not None
                    else 1.0
                ],
                dtype=np.float32,
            ),
            "finger_temperature": np.array(
                [
                    float(temperature_state.finger_temperature)
                    if temperature_state is not None
                    else 1.0
                ],
                dtype=np.float32,
            ),
        }
        optional_keys = [
            "window_start",
            "window_end",
            "trial_id",
            "block_id",
            "subject_id",
            "experiment_hash",
        ]
        n_expected = int(len(y_action))
        if isinstance(meta, dict):
            for key in optional_keys:
                if key not in meta:
                    continue
                if _is_maskable_array(meta[key], n_expected):
                    export_payload[key] = np.asarray(meta[key])[test_idx]
        export_path = pred_npz_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(export_path, **export_payload)
        print(f"✅ Saved test predictions: {export_path}")

    _print_label_summary("Train split", y_action[train_idx], y_finger[train_idx])
    _print_label_summary("Test split", y_action_test, y_finger_test)

    overall_action_unique = len(np.unique(y_action))
    overall_finger_unique = _unique_non_rest_fingers(y_action, y_finger)
    action_train_unique = len(np.unique(y_action[train_idx])) if len(train_idx) else 0
    action_test_unique = len(np.unique(y_action_test)) if len(y_action_test) else 0
    finger_train_unique = _unique_non_rest_fingers(
        y_action[train_idx], y_finger[train_idx]
    )
    finger_test_unique = _unique_non_rest_fingers(y_action_test, y_finger_test)

    if overall_action_unique < 2:
        print("⚠️ Action labels are single-class overall. Aborting evaluation.")
        manifest["abort_reason"] = "action_single_class_overall"
        _maybe_write_manifest()
        return 2
    if action_train_unique < 2 or action_test_unique < 2:
        print("⚠️ Action labels collapsed in train/test split. Aborting evaluation.")
        manifest["abort_reason"] = "action_split_collapsed"
        _maybe_write_manifest()
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

    raw_finger_preds = np.argmax(finger_probs, axis=1)
    raw_valid_pair_rate = float(
        np.mean((action_preds != ACTION_REST) | (raw_finger_preds == 0))
    ) if len(action_preds) else None
    raw_invalid_pair_rate = (
        float(1.0 - raw_valid_pair_rate) if raw_valid_pair_rate is not None else None
    )
    _, finger_preds = enforce_prediction_pairs(action_preds, raw_finger_preds)
    finger_conf = finger_probs[np.arange(len(finger_probs)), finger_preds]

    # =========================
    # ===== METRICS ===========
    # =========================
    action_acc = accuracy_score(y_action_test, action_preds)
    action_f1_macro = f1_score(
        y_action_test, action_preds, average="macro", zero_division=0
    )
    action_f1_weighted = f1_score(
        y_action_test, action_preds, average="weighted", zero_division=0
    )
    mask = y_action_test != ACTION_REST
    joint_correct = (action_preds == y_action_test) & (finger_preds == y_finger_test)
    joint_acc = float(np.mean(joint_correct)) if len(joint_correct) else None
    joint_acc_non_rest = (
        float(np.mean(joint_correct[mask])) if mask.any() else None
    )
    if not mask.any():
        print("⚠️ No non-REST windows in test set; skipping finger metrics.")
        finger_metrics_ok = False
        exit_code = max(exit_code, 1)

    finger_acc = (
        accuracy_score(y_finger_test[mask], finger_preds[mask])
        if (finger_metrics_ok and mask.any())
        else None
    )
    finger_f1_non_rest_macro = (
        f1_score(y_finger_test[mask], finger_preds[mask], average="macro", zero_division=0)
        if (finger_metrics_ok and mask.any())
        else None
    )
    finger_f1_non_rest_weighted = (
        f1_score(
            y_finger_test[mask], finger_preds[mask], average="weighted", zero_division=0
        )
        if (finger_metrics_ok and mask.any())
        else None
    )

    finger_acc_overall = (
        accuracy_score(y_finger_test, finger_preds)
        if y_finger_test.size
        else None
    )
    finger_f1_overall_macro = (
        f1_score(y_finger_test, finger_preds, average="macro", zero_division=0)
        if y_finger_test.size
        else None
    )
    finger_f1_overall_weighted = (
        f1_score(y_finger_test, finger_preds, average="weighted", zero_division=0)
        if y_finger_test.size
        else None
    )

    rest_mask = y_action_test == ACTION_REST
    rest_tpr = None
    rest_fpr = None
    rest_precision = None
    rest_f1 = None
    if np.any(rest_mask):
        rest_tp = int(np.sum(rest_mask & (action_preds == ACTION_REST)))
        rest_fn = int(np.sum(rest_mask & (action_preds != ACTION_REST)))
        rest_fp = int(np.sum(~rest_mask & (action_preds == ACTION_REST)))
        rest_tn = int(np.sum(~rest_mask & (action_preds != ACTION_REST)))
        rest_tpr = float(rest_tp / (rest_tp + rest_fn)) if (rest_tp + rest_fn) else None
        rest_fpr = float(rest_fp / (rest_fp + rest_tn)) if (rest_fp + rest_tn) else None
        rest_precision = (
            float(rest_tp / (rest_tp + rest_fp)) if (rest_tp + rest_fp) else None
        )
        if rest_precision is not None and rest_tpr is not None:
            denom = rest_precision + rest_tpr
            rest_f1 = float(2 * rest_precision * rest_tpr / denom) if denom else None

    print(f"\n🎯 Action Accuracy: {action_acc * 100:.2f}%")
    print(
        f"🎯 Action F1 (macro/weighted): {action_f1_macro:.3f} / {action_f1_weighted:.3f}"
    )
    if raw_invalid_pair_rate is not None:
        print(
            f"🎯 Raw invalid pair rate: {raw_invalid_pair_rate * 100:.2f}%"
        )
    if joint_acc is not None:
        joint_non_rest_str = (
            f"{joint_acc_non_rest * 100:.2f}%"
            if joint_acc_non_rest is not None
            else "n/a"
        )
        print(
            f"🎯 Joint Accuracy (overall/non-REST): {joint_acc * 100:.2f}% / {joint_non_rest_str}"
        )
    if finger_acc is not None:
        print(f"🎯 Finger Accuracy (non-REST): {finger_acc * 100:.2f}%")
        if finger_f1_non_rest_macro is not None:
            print(
                f"🎯 Finger F1 (non-REST macro/weighted): "
                f"{finger_f1_non_rest_macro:.3f} / {finger_f1_non_rest_weighted:.3f}\n"
            )
        else:
            print()
    else:
        print("🎯 Finger Accuracy (non-REST): skipped\n")
    if rest_tpr is not None:
        rest_prec_str = f"{rest_precision * 100:.2f}%" if rest_precision is not None else "n/a"
        rest_f1_str = f"{rest_f1:.3f}" if rest_f1 is not None else "n/a"
        print(
            f"🎯 REST TPR: {rest_tpr * 100:.2f}% | REST FPR: {rest_fpr * 100:.2f}% "
            f"| REST Precision: {rest_prec_str} | REST F1: {rest_f1_str}"
        )

    action_acc_s = None
    action_f1_macro_s = None
    action_f1_weighted_s = None
    finger_acc_s = None
    joint_acc_s = None
    joint_acc_non_rest_s = None
    finger_acc_overall_s = None
    finger_f1_non_rest_macro_s = None
    finger_f1_non_rest_weighted_s = None
    finger_f1_overall_macro_s = None
    finger_f1_overall_weighted_s = None
    rest_tpr_s = None
    rest_fpr_s = None
    rest_precision_s = None
    rest_f1_s = None
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
                trial_ids_for_probs = np.asarray(meta["trial_id"])[test_idx].astype(
                    np.int64
                )
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
        action_f1_macro_s = f1_score(
            y_action_test[order], smoothed_action, average="macro", zero_division=0
        )
        action_f1_weighted_s = f1_score(
            y_action_test[order], smoothed_action, average="weighted", zero_division=0
        )
        mask_s = y_action_test[order] != ACTION_REST
        joint_correct_s = (
            (smoothed_action == y_action_test[order])
            & (smoothed_finger == y_finger_test[order])
        )
        joint_acc_s = float(np.mean(joint_correct_s)) if len(joint_correct_s) else None
        joint_acc_non_rest_s = (
            float(np.mean(joint_correct_s[mask_s])) if mask_s.any() else None
        )
        finger_acc_s = (
            accuracy_score(y_finger_test[order][mask_s], smoothed_finger[mask_s])
            if (finger_metrics_ok and mask_s.any())
            else None
        )
        finger_f1_non_rest_macro_s = (
            f1_score(
                y_finger_test[order][mask_s],
                smoothed_finger[mask_s],
                average="macro",
                zero_division=0,
            )
            if (finger_metrics_ok and mask_s.any())
            else None
        )
        finger_f1_non_rest_weighted_s = (
            f1_score(
                y_finger_test[order][mask_s],
                smoothed_finger[mask_s],
                average="weighted",
                zero_division=0,
            )
            if (finger_metrics_ok and mask_s.any())
            else None
        )
        finger_acc_overall_s = (
            accuracy_score(y_finger_test[order], smoothed_finger)
            if y_finger_test.size
            else None
        )
        finger_f1_overall_macro_s = (
            f1_score(
                y_finger_test[order], smoothed_finger, average="macro", zero_division=0
            )
            if y_finger_test.size
            else None
        )
        finger_f1_overall_weighted_s = (
            f1_score(
                y_finger_test[order],
                smoothed_finger,
                average="weighted",
                zero_division=0,
            )
            if y_finger_test.size
            else None
        )
        rest_mask_s = y_action_test[order] == ACTION_REST
        rest_precision_s = None
        rest_f1_s = None
        if np.any(rest_mask_s):
            rest_tp_s = int(np.sum(rest_mask_s & (smoothed_action == ACTION_REST)))
            rest_fn_s = int(np.sum(rest_mask_s & (smoothed_action != ACTION_REST)))
            rest_fp_s = int(np.sum(~rest_mask_s & (smoothed_action == ACTION_REST)))
            rest_tn_s = int(np.sum(~rest_mask_s & (smoothed_action != ACTION_REST)))
            rest_tpr_s = (
                float(rest_tp_s / (rest_tp_s + rest_fn_s))
                if (rest_tp_s + rest_fn_s)
                else None
            )
            rest_fpr_s = (
                float(rest_fp_s / (rest_fp_s + rest_tn_s))
                if (rest_fp_s + rest_tn_s)
                else None
            )
            rest_precision_s = (
                float(rest_tp_s / (rest_tp_s + rest_fp_s))
                if (rest_tp_s + rest_fp_s)
                else None
            )
            if rest_precision_s is not None and rest_tpr_s is not None:
                denom = rest_precision_s + rest_tpr_s
                rest_f1_s = (
                    float(2 * rest_precision_s * rest_tpr_s / denom) if denom else None
                )

        print(f"🎯 Smoothed Action Accuracy: {action_acc_s * 100:.2f}%")
        print(
            f"🎯 Smoothed Action F1 (macro/weighted): "
            f"{action_f1_macro_s:.3f} / {action_f1_weighted_s:.3f}"
        )
        if joint_acc_s is not None:
            joint_non_rest_s_str = (
                f"{joint_acc_non_rest_s * 100:.2f}%"
                if joint_acc_non_rest_s is not None
                else "n/a"
            )
            print(
                f"🎯 Smoothed Joint Accuracy (overall/non-REST): "
                f"{joint_acc_s * 100:.2f}% / {joint_non_rest_s_str}"
            )
        if finger_acc_s is not None:
            print(f"🎯 Smoothed Finger Accuracy (non-REST): {finger_acc_s * 100:.2f}%")
            if finger_f1_non_rest_macro_s is not None:
                print(
                    f"🎯 Smoothed Finger F1 (non-REST macro/weighted): "
                    f"{finger_f1_non_rest_macro_s:.3f} / {finger_f1_non_rest_weighted_s:.3f}\n"
                )
            else:
                print()
        else:
            print("🎯 Smoothed Finger Accuracy (non-REST): skipped\n")
        if rest_tpr_s is not None:
            rest_prec_s_str = (
                f"{rest_precision_s * 100:.2f}%" if rest_precision_s is not None else "n/a"
            )
            rest_f1_s_str = f"{rest_f1_s:.3f}" if rest_f1_s is not None else "n/a"
            print(
                f"🎯 Smoothed REST TPR: {rest_tpr_s * 100:.2f}% | "
                f"REST FPR: {rest_fpr_s * 100:.2f}% | "
                f"REST Precision: {rest_prec_s_str} | REST F1: {rest_f1_s_str}"
            )

    # =========================
    # ===== ECE COMPUTATION ===
    # =========================
    finger_ece = None
    action_ece = expected_calibration_error(
        action_conf, action_preds, y_action_test, N_BINS
    )
    print(f"📏 Action ECE: {action_ece:.4f}")

    if finger_metrics_ok and mask.any():
        finger_ece = expected_calibration_error(
            finger_conf[mask], finger_preds[mask], y_finger_test[mask], N_BINS
        )
        print(f"📏 Finger ECE (non-REST): {finger_ece:.4f}")

    manifest["metrics"]["raw_valid_pair_rate"] = (
        float(raw_valid_pair_rate) if raw_valid_pair_rate is not None else None
    )
    manifest["metrics"]["raw_invalid_pair_rate"] = (
        float(raw_invalid_pair_rate) if raw_invalid_pair_rate is not None else None
    )
    manifest["metrics"]["action_acc"] = float(action_acc)
    manifest["metrics"]["action_f1_macro"] = float(action_f1_macro)
    manifest["metrics"]["action_f1_weighted"] = float(action_f1_weighted)
    manifest["metrics"]["joint_acc"] = float(joint_acc) if joint_acc is not None else None
    manifest["metrics"]["joint_acc_non_rest"] = (
        float(joint_acc_non_rest) if joint_acc_non_rest is not None else None
    )
    manifest["metrics"]["finger_acc_non_rest"] = (
        float(finger_acc) if finger_acc is not None else None
    )
    manifest["metrics"]["finger_acc_overall"] = (
        float(finger_acc_overall) if finger_acc_overall is not None else None
    )
    manifest["metrics"]["finger_f1_non_rest_macro"] = (
        float(finger_f1_non_rest_macro)
        if finger_f1_non_rest_macro is not None
        else None
    )
    manifest["metrics"]["finger_f1_non_rest_weighted"] = (
        float(finger_f1_non_rest_weighted)
        if finger_f1_non_rest_weighted is not None
        else None
    )
    manifest["metrics"]["finger_f1_overall_macro"] = (
        float(finger_f1_overall_macro)
        if finger_f1_overall_macro is not None
        else None
    )
    manifest["metrics"]["finger_f1_overall_weighted"] = (
        float(finger_f1_overall_weighted)
        if finger_f1_overall_weighted is not None
        else None
    )
    manifest["metrics"]["rest_tpr"] = float(rest_tpr) if rest_tpr is not None else None
    manifest["metrics"]["rest_fpr"] = float(rest_fpr) if rest_fpr is not None else None
    manifest["metrics"]["rest_precision"] = (
        float(rest_precision) if rest_precision is not None else None
    )
    manifest["metrics"]["rest_f1"] = float(rest_f1) if rest_f1 is not None else None
    manifest["metrics"]["action_ece"] = float(action_ece)
    manifest["metrics"]["finger_ece_non_rest"] = (
        float(finger_ece) if finger_ece is not None else None
    )
    manifest["metrics"]["smoothed_action_acc"] = (
        float(action_acc_s) if action_acc_s is not None else None
    )
    manifest["metrics"]["smoothed_action_f1_macro"] = (
        float(action_f1_macro_s) if action_f1_macro_s is not None else None
    )
    manifest["metrics"]["smoothed_action_f1_weighted"] = (
        float(action_f1_weighted_s) if action_f1_weighted_s is not None else None
    )
    manifest["metrics"]["smoothed_joint_acc"] = (
        float(joint_acc_s) if joint_acc_s is not None else None
    )
    manifest["metrics"]["smoothed_joint_acc_non_rest"] = (
        float(joint_acc_non_rest_s) if joint_acc_non_rest_s is not None else None
    )
    manifest["metrics"]["smoothed_finger_acc_non_rest"] = (
        float(finger_acc_s) if finger_acc_s is not None else None
    )
    manifest["metrics"]["smoothed_finger_acc_overall"] = (
        float(finger_acc_overall_s) if finger_acc_overall_s is not None else None
    )
    manifest["metrics"]["smoothed_finger_f1_non_rest_macro"] = (
        float(finger_f1_non_rest_macro_s)
        if finger_f1_non_rest_macro_s is not None
        else None
    )
    manifest["metrics"]["smoothed_finger_f1_non_rest_weighted"] = (
        float(finger_f1_non_rest_weighted_s)
        if finger_f1_non_rest_weighted_s is not None
        else None
    )
    manifest["metrics"]["smoothed_finger_f1_overall_macro"] = (
        float(finger_f1_overall_macro_s)
        if finger_f1_overall_macro_s is not None
        else None
    )
    manifest["metrics"]["smoothed_finger_f1_overall_weighted"] = (
        float(finger_f1_overall_weighted_s)
        if finger_f1_overall_weighted_s is not None
        else None
    )
    manifest["metrics"]["smoothed_rest_tpr"] = (
        float(rest_tpr_s) if rest_tpr_s is not None else None
    )
    manifest["metrics"]["smoothed_rest_fpr"] = (
        float(rest_fpr_s) if rest_fpr_s is not None else None
    )
    manifest["metrics"]["smoothed_rest_precision"] = (
        float(rest_precision_s) if rest_precision_s is not None else None
    )
    manifest["metrics"]["smoothed_rest_f1"] = (
        float(rest_f1_s) if rest_f1_s is not None else None
    )

    primary_benchmark = _build_primary_benchmark(
        y_action_test=y_action_test,
        y_finger_test=y_finger_test,
        metrics=manifest["metrics"],
    )

    core_full_session_replay = None
    rest_event_breakdown: List[Dict[str, Any]] = []
    candidate_event_flags: List[Dict[str, Any]] = []
    repeated_split_summary = {
        "seeds": list(REPEATED_SPLIT_SEEDS),
        "per_seed": [],
        "mean": {},
        "std": {},
    }

    core_split_meta = _subset_meta(meta, split_idx, len(y_action)) if meta else {}
    if eval_model is None or eval_normalizer is None:
        try:
            eval_model, eval_normalizer, temperature_state = _load_eval_model_and_normalizer(
                model_path=model_path,
                scaler_path=scaler_path,
                temperature_path=temperature_path,
                n_channels=X.shape[2],
                n_fingers=n_fingers,
                n_actions=n_actions,
            )
        except Exception as exc:
            print(f"⚠️ Core-session replay skipped: {exc}")
            eval_model = None
            eval_normalizer = None

    if eval_model is not None and eval_normalizer is not None:
        core_action_probs, core_finger_probs = _run_deterministic_inference(
            X=X,
            indices=split_idx,
            model=eval_model,
            normalizer=eval_normalizer,
            temperature_state=temperature_state,
            n_actions=n_actions,
            n_fingers=n_fingers,
            batch_size=max(1, int(args.batch_size)),
        )
        core_metrics_payload = _compute_prediction_metrics(
            action_probs=core_action_probs,
            finger_probs=core_finger_probs,
            y_action_true=y_action[split_idx],
            y_finger_true=y_finger[split_idx],
            n_bins=N_BINS,
        )
        core_metrics = core_metrics_payload["metrics"]
        core_full_session_replay = {
            "n_windows": int(len(split_idx)),
            "sessions": sorted(
                set(np.asarray(meta["session_id"])[split_idx].astype("U").tolist())
            )
            if "session_id" in meta
            else [],
            "metrics": {
                key: (
                    float(value)
                    if isinstance(value, (float, np.floating)) and value is not None
                    else value
                )
                for key, value in core_metrics.items()
                if key
                in {
                    "action_acc",
                    "joint_acc",
                    "joint_acc_non_rest",
                    "finger_acc_non_rest",
                    "rest_tpr",
                    "rest_precision",
                    "action_ece",
                    "raw_invalid_pair_rate",
                }
            },
            "pred_action_counts": _format_label_counts(
                core_metrics_payload["action_preds"], ACTION_NAMES
            ),
            "pred_pair_counts": core_metrics_payload["pair_counts"],
        }
        rest_event_breakdown, candidate_event_flags = _build_rest_event_breakdown(
            indices=split_idx,
            y_action_true=y_action[split_idx],
            action_probs=core_action_probs,
            action_preds=core_metrics_payload["action_preds"],
            finger_preds=core_metrics_payload["finger_preds"],
            meta=meta,
        )
        if candidate_event_flags:
            print("\n⚠️ Candidate REST events for review:")
            for flag in candidate_event_flags:
                print(
                    f"  event {flag['event_id']} ({flag['session_id']}): "
                    f"REST TPR {flag['rest_tpr'] * 100:.2f}% | "
                    f"recommend {flag['recommended_action']} | "
                    f"reason={flag['reason']}"
                )

        repeated_metric_keys = [
            "action_acc",
            "joint_acc",
            "finger_acc_non_rest",
            "rest_tpr",
            "rest_precision",
        ]
        repeated_values: Dict[str, List[float]] = {key: [] for key in repeated_metric_keys}
        for repeat_seed in REPEATED_SPLIT_SEEDS:
            if int(repeat_seed) == int(split_seed):
                metrics_payload = {
                    "metrics": {
                        key: manifest["metrics"].get(key) for key in repeated_metric_keys
                    }
                }
                test_count = int(len(test_idx))
            else:
                rep_train_idx, rep_test_idx, _ = _split_with_checks(
                    y_action[split_idx],
                    y_finger[split_idx],
                    meta=core_split_meta,
                    seed=int(repeat_seed),
                    test_size=float(test_size),
                    split_mode=str(split_mode),
                    purge_seconds=float(purge_seconds),
                    hop_seconds=hop_seconds,
                )
                if rep_test_idx is None:
                    repeated_split_summary["per_seed"].append(
                        {"seed": int(repeat_seed), "status": "split_failed"}
                    )
                    continue
                rep_test_global = split_idx[np.asarray(rep_test_idx, dtype=np.int64)]
                rep_action_probs, rep_finger_probs = _run_deterministic_inference(
                    X=X,
                    indices=rep_test_global,
                    model=eval_model,
                    normalizer=eval_normalizer,
                    temperature_state=temperature_state,
                    n_actions=n_actions,
                    n_fingers=n_fingers,
                    batch_size=max(1, int(args.batch_size)),
                )
                metrics_payload = _compute_prediction_metrics(
                    action_probs=rep_action_probs,
                    finger_probs=rep_finger_probs,
                    y_action_true=y_action[rep_test_global],
                    y_finger_true=y_finger[rep_test_global],
                    n_bins=N_BINS,
                )
                test_count = int(len(rep_test_global))
            metric_entry = {
                key: (
                    float(metrics_payload["metrics"][key])
                    if metrics_payload["metrics"].get(key) is not None
                    else None
                )
                for key in repeated_metric_keys
            }
            repeated_split_summary["per_seed"].append(
                {
                    "seed": int(repeat_seed),
                    "status": "ok",
                    "test_n": test_count,
                    "metrics": metric_entry,
                }
            )
            for key in repeated_metric_keys:
                value = metric_entry.get(key)
                if value is not None:
                    repeated_values[key].append(float(value))

        repeated_split_summary["mean"] = {
            key: float(np.mean(values)) if values else None
            for key, values in repeated_values.items()
        }
        repeated_split_summary["std"] = {
            key: float(np.std(values)) if values else None
            for key, values in repeated_values.items()
        }
        rest_tpr_mean = repeated_split_summary["mean"].get("rest_tpr")
        rest_tpr_std = repeated_split_summary["std"].get("rest_tpr")
        joint_acc_mean = repeated_split_summary["mean"].get("joint_acc")
        joint_acc_std = repeated_split_summary["std"].get("joint_acc")
        if rest_tpr_mean is not None and joint_acc_mean is not None:
            print(
                "\n🔁 Repeated split summary: "
                f"REST TPR mean/std {rest_tpr_mean * 100:.2f}% / "
                f"{(rest_tpr_std or 0.0) * 100:.2f}% | "
                f"Joint Acc mean/std {joint_acc_mean * 100:.2f}% / "
                f"{(joint_acc_std or 0.0) * 100:.2f}%"
            )

    manifest["benchmarks"] = {
        "primary_mixed_holdout": primary_benchmark,
        "aux_rest_only": aux_rest_benchmark,
        "core_full_session_replay": core_full_session_replay,
    }
    manifest["rest_event_breakdown"] = rest_event_breakdown
    manifest["candidate_event_flags"] = candidate_event_flags
    manifest["repeated_split_summary"] = repeated_split_summary

    if subject_ids is not None:
        try:
            subj_test = np.asarray(subject_ids)[test_idx]
            unique_subjects = sorted(set(subj_test.tolist()))
            if len(unique_subjects) > 1:
                print("\nPer-subject ECE (test set):")
                for subj in unique_subjects:
                    subj_mask = subj_test == subj
                    if not np.any(subj_mask):
                        continue
                    subj_action_ece = expected_calibration_error(
                        action_conf[subj_mask],
                        action_preds[subj_mask],
                        y_action_test[subj_mask],
                        N_BINS,
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

                    print(
                        f"  {subj}: action_ece={subj_action_ece:.4f}, finger_ece={subj_finger_ece:.4f}"
                    )
        except Exception:
            pass

    # =========================
    # ===== PLOTS =============
    # =========================
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

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

    if finger_metrics_ok and mask.any():
        finger_cm = confusion_matrix(
            y_finger_test[mask],
            finger_preds[mask],
            labels=FINGER_LABELS,
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
        axs[0, 1].axis("off")
        axs[0, 1].text(0.5, 0.5, "Finger metrics skipped", ha="center", va="center")

    bin_centers, bin_accs = reliability_bins(
        action_conf, action_preds, y_action_test, N_BINS
    )
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

    plot_dir = Path(paths.eval_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    out_path = plot_dir / f"eval_{exp_hash}.png"
    plt.savefig(out_path)

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close()

    print(f"\n✅ Saved evaluation plot: {out_path}")
    _maybe_write_manifest()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
