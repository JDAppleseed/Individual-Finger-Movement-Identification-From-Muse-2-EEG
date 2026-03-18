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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from utils.sequence_data import (
    load_sequence_npz,
    split_indices,
    fit_channel_normalizer,
    apply_channel_normalizer,
)
from utils.runtime_utils import (
    TemperatureScalingState,
    apply_temperature_to_logits,
    save_normalizer,
    save_temperature_scaling,
)
from utils.experiment_logger import log_experiment, get_latest_experiment_hash
from utils.label_schema import (
    ACTION_NAMES,
    ACTION_REST,
    FINGER_NAMES,
    FINGER_NONE,
    decode_finger_predictions,
    finger_id_to_model_index,
    uses_active_finger_head,
)
from utils.splitting import (
    compose_split_indices,
    infer_groups,
    assert_no_group_overlap,
    resolve_auxiliary_rest_sessions,
)
from utils.session_layout import SessionLayout, resolve_session_dir

# Pipeline handoff: Step 2 consumes Step 1b NPZ windows, then writes model/scaler
# and split metadata that Step 3/3b/3c/4 reuse from the same run directory.
SEED = 42
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
LOSS_ACTION_WEIGHT = 1.0
# Default REST class weight. Historically this repo used smaller values for
# early, REST-imbalanced datasets. Current default uses parity weighting so the
# action head does not systematically under-emphasize REST.
REST_WEIGHT = 1.0
REST_BALANCE_MODE = "core_event_equalized"
WINDOW_PREPROCESS = "center_detrend"
AUX_REST_SESSION_POLICY = "auto_train_only"
REST_FINGER_LOSS_WEIGHT = 0.0
ACTIVE_FINGER_HEAD = True

DEFAULT_NPZ = "eeg_windows.npz"
DEFAULT_MODEL = "finger_action_model.pt"
DEFAULT_SCALER = "scaler.npz"
DEFAULT_PREDS = "test_predictions.npz"
DEFAULT_TEMPERATURE = "temperature_scaling.json"
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


def _parse_finger_weights(value: Any, n_fingers: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() in {"none", "null", "default"}:
            return None
        if raw[0] in "[{":
            try:
                value = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"Invalid JSON for finger weights: {raw}") from exc
        else:
            parts = [p for p in re.split(r"[,\\s]+", raw) if p]
            try:
                values = [float(p) for p in parts]
            except Exception as exc:
                raise ValueError(f"Invalid finger weight list: {raw}") from exc
            value = values

    if isinstance(value, dict):
        name_map = {name.lower(): idx for idx, name in FINGER_NAMES.items()}
        weights = [1.0] * n_fingers
        for key, val in value.items():
            if isinstance(key, str):
                k = key.strip().lower()
                if k.isdigit():
                    raw_idx = int(k)
                elif k in name_map:
                    raw_idx = name_map[k]
                else:
                    raise ValueError(f"Unknown finger key: {key}")
            else:
                raw_idx = int(key)
            idx = (
                finger_id_to_model_index(raw_idx, n_fingers)
                if uses_active_finger_head(n_fingers)
                else raw_idx
            )
            if idx < 0 or idx >= n_fingers:
                raise ValueError(f"Finger weight index out of range: {idx}")
            weights[idx] = float(val)
        return torch.tensor(weights, dtype=torch.float32)

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        weights = [float(v) for v in value]
        if len(weights) != n_fingers:
            raise ValueError(
                f"finger_weights length {len(weights)} does not match n_fingers {n_fingers}"
            )
        return torch.tensor(weights, dtype=torch.float32)

    raise ValueError(f"Unsupported finger_weights type: {type(value)}")


def _parse_action_weights(value: Any, n_actions: int) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw or raw.lower() in {"none", "null", "default"}:
            return None
        if raw[0] in "[{":
            try:
                value = json.loads(raw)
            except Exception as exc:
                raise ValueError(f"Invalid JSON for action weights: {raw}") from exc
        else:
            parts = [p for p in re.split(r"[,\\s]+", raw) if p]
            try:
                value = [float(p) for p in parts]
            except Exception as exc:
                raise ValueError(f"Invalid action weight list: {raw}") from exc

    if isinstance(value, dict):
        name_map = {name.lower(): idx for idx, name in ACTION_NAMES.items()}
        weights = [1.0] * n_actions
        for key, val in value.items():
            if isinstance(key, str):
                k = key.strip().lower()
                if k.isdigit():
                    idx = int(k)
                elif k in name_map:
                    idx = name_map[k]
                else:
                    raise ValueError(f"Unknown action key: {key}")
            else:
                idx = int(key)
            if idx < 0 or idx >= n_actions:
                raise ValueError(f"Action weight index out of range: {idx}")
            weights[idx] = float(val)
        return torch.tensor(weights, dtype=torch.float32)

    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        weights = [float(v) for v in value]
        if len(weights) != n_actions:
            raise ValueError(
                f"action_weights length {len(weights)} does not match n_actions {n_actions}"
            )
        return torch.tensor(weights, dtype=torch.float32)

    raise ValueError(f"Unsupported action_weights type: {type(value)}")


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


def _meta_text_array(
    meta: Dict[str, Any], key: str, n_expected: int
) -> Optional[np.ndarray]:
    if not meta or key not in meta:
        return None
    try:
        arr = np.asarray(meta[key])
    except Exception:
        return None
    if arr.ndim == 0:
        return None
    if len(arr) != n_expected:
        return None
    return arr.astype("U")


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


def _class_counts(values: np.ndarray) -> Dict[int, int]:
    if len(values) == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))


def _subset_meta(meta: Dict[str, Any], idx: np.ndarray, n_expected: int) -> Dict[str, Any]:
    if not meta:
        return {}
    idx = np.asarray(idx, dtype=np.int64)
    out: Dict[str, Any] = {}
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
            if len(arr) == n_expected:
                out[key] = arr[idx]
            else:
                out[key] = val
        except Exception:
            out[key] = val
    return out


def _calibration_stratify_labels(y_action: np.ndarray, y_finger: np.ndarray) -> np.ndarray:
    y_action = np.asarray(y_action, dtype=np.int64).reshape(-1)
    y_finger = np.asarray(y_finger, dtype=np.int64).reshape(-1)
    max_finger = int(np.max(y_finger)) if y_finger.size else 0
    return (y_action * (max_finger + 1)) + y_finger


def _split_calibration_indices(
    train_idx: np.ndarray,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    meta: Dict[str, Any],
    *,
    calibration_size: float,
    random_state: int,
    split_mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    train_idx = np.asarray(train_idx, dtype=np.int64)
    if calibration_size <= 0.0 or len(train_idx) < 8:
        return train_idx, np.array([], dtype=np.int64)

    y_action_train = y_action[train_idx]
    y_finger_train = y_finger[train_idx]
    train_meta = _subset_meta(meta, train_idx, len(y_action)) if meta else {}

    try:
        fit_local, calib_local = split_indices(
            y_action_train,
            y_finger_train,
            meta=train_meta if train_meta else None,
            test_size=float(calibration_size),
            random_state=int(random_state),
            split_mode=str(split_mode),
            purge_seconds=0.0,
            hop_seconds=None,
            allow_fallback=True,
        )
        fit_idx = train_idx[np.asarray(fit_local, dtype=np.int64)]
        calib_idx = train_idx[np.asarray(calib_local, dtype=np.int64)]
        if len(fit_idx) and len(calib_idx):
            return fit_idx, calib_idx
    except Exception:
        pass

    try:
        fit_local, calib_local = train_test_split(
            np.arange(len(train_idx), dtype=np.int64),
            test_size=float(calibration_size),
            random_state=int(random_state),
            stratify=_calibration_stratify_labels(y_action_train, y_finger_train),
        )
    except ValueError:
        fit_local, calib_local = train_test_split(
            np.arange(len(train_idx), dtype=np.int64),
            test_size=float(calibration_size),
            random_state=int(random_state),
            stratify=None,
        )
    return train_idx[np.asarray(fit_local, dtype=np.int64)], train_idx[
        np.asarray(calib_local, dtype=np.int64)
    ]


def _fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
    max_iter: int = 50,
) -> Tuple[float, Dict[str, Optional[float]]]:
    logits = np.asarray(logits, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if logits.ndim != 2 or len(logits) != len(labels) or len(labels) == 0:
        return 1.0, {"nll_before": None, "nll_after": None}
    if len(np.unique(labels)) < 2:
        return 1.0, {"nll_before": None, "nll_after": None}

    fit_device = torch.device("cpu")
    logits_t = torch.tensor(logits, dtype=torch.float32, device=fit_device)
    labels_t = torch.tensor(labels, dtype=torch.long, device=fit_device)
    criterion = nn.CrossEntropyLoss()
    nll_before = float(criterion(logits_t, labels_t).item())

    log_temp = torch.nn.Parameter(torch.zeros((), dtype=torch.float32, device=device))
    opt = torch.optim.LBFGS(
        [log_temp],
        lr=0.1,
        max_iter=int(max_iter),
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp(min=1e-3, max=100.0)
        loss = criterion(logits_t / temp, labels_t)
        loss.backward()
        return loss

    try:
        opt.step(closure)
        temp = float(torch.exp(log_temp).clamp(min=1e-3, max=100.0).item())
        nll_after = float(criterion(logits_t / temp, labels_t).item())
    except Exception:
        temp = 1.0
        nll_after = nll_before
    if not np.isfinite(temp) or temp <= 0.0:
        temp = 1.0
    return temp, {"nll_before": nll_before, "nll_after": nll_after}


def _split_groups_from_meta(
    meta: Dict[str, Any], n_expected: int, split_mode: str
) -> np.ndarray:
    if split_mode == "holdout_session":
        if not meta or "session_id" not in meta:
            raise ValueError("split_mode=holdout_session requires session_id in meta.")
        session = np.asarray(meta["session_id"])
        if session.ndim == 0 or len(session) != n_expected:
            raise ValueError("session_id meta length mismatch for split diagnostics.")
        return session
    return infer_groups(meta, n_expected)


def _log_split_diagnostics(
    groups: np.ndarray,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
):
    groups = np.asarray(groups).reshape(-1)
    unique_groups, group_counts = np.unique(groups, return_counts=True)
    train_groups = np.unique(groups[train_idx])
    test_groups = np.unique(groups[test_idx])
    order = np.argsort(group_counts)[::-1]
    top = [(str(unique_groups[i]), int(group_counts[i])) for i in order[:10]]
    singleton_count = int(np.sum(group_counts == 1))

    print(
        "Split diagnostics: "
        f"train_groups={len(train_groups)} test_groups={len(test_groups)} total_groups={len(unique_groups)}"
    )
    print(f"Top group sizes (up to 10): {top}")
    print(f"Singleton groups: {singleton_count}")

    print(f"Train action counts: {_class_counts(y_action[train_idx])}")
    print(f"Test action counts: {_class_counts(y_action[test_idx])}")
    print(f"Train finger counts: {_class_counts(y_finger[train_idx])}")
    print(f"Test finger counts: {_class_counts(y_finger[test_idx])}")

    train_non_rest = y_action[train_idx] != ACTION_REST
    test_non_rest = y_action[test_idx] != ACTION_REST
    if np.any(train_non_rest):
        print(
            f"Train finger (non-REST) counts: {_class_counts(y_finger[train_idx][train_non_rest])}"
        )
    else:
        print("Train finger (non-REST) counts: none")
    if np.any(test_non_rest):
        print(
            f"Test finger (non-REST) counts: {_class_counts(y_finger[test_idx][test_non_rest])}"
        )
    else:
        print("Test finger (non-REST) counts: none")


def _preprocess_config_from_mode(mode: str) -> Dict[str, bool]:
    mode = str(mode or "none").strip().lower()
    if mode == "none":
        return {"per_window_center": False, "per_window_detrend": False}
    if mode == "center":
        return {"per_window_center": True, "per_window_detrend": False}
    if mode == "center_detrend":
        return {"per_window_center": True, "per_window_detrend": True}
    raise ValueError(f"Unsupported window preprocess mode: {mode}")


def _resolve_auxiliary_rest_sessions(
    y_action: np.ndarray,
    meta: Dict[str, Any],
    *,
    policy: str,
) -> Dict[str, Any]:
    return resolve_auxiliary_rest_sessions(y_action, meta, policy=policy)


def _build_train_sample_weights(
    y_action: np.ndarray,
    meta: Dict[str, Any],
    *,
    balance_mode: str,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    y_action = np.asarray(y_action, dtype=np.int64).reshape(-1)
    summary: Dict[str, Any] = {"mode": str(balance_mode or "none")}
    if len(y_action) == 0 or str(balance_mode or "none") == "none":
        summary["enabled"] = False
        return None, summary

    if str(balance_mode) not in {"session_equalized", "core_event_equalized"}:
        raise ValueError(f"Unsupported rest balance mode: {balance_mode}")

    session_id = _meta_text_array(meta, "session_id", len(y_action))
    if session_id is None:
        summary["enabled"] = False
        summary["reason"] = "missing_session_id"
        return None, summary

    rest_mask = y_action == ACTION_REST
    if not np.any(rest_mask):
        summary["enabled"] = False
        summary["reason"] = "no_rest_samples"
        return None, summary

    rest_sessions = session_id[rest_mask]
    valid_mask = (rest_sessions != "") & (rest_sessions != "UNKNOWN")
    rest_sessions = rest_sessions[valid_mask]
    if len(rest_sessions) == 0:
        summary["enabled"] = False
        summary["reason"] = "unknown_rest_sessions"
        return None, summary

    weights = np.ones(len(y_action), dtype=np.float32)
    unique_sessions, counts = np.unique(rest_sessions, return_counts=True)

    if str(balance_mode) == "session_equalized":
        if len(unique_sessions) < 2:
            summary["enabled"] = False
            summary["reason"] = "single_rest_session"
            return None, summary

        target_per_session = float(np.sum(counts)) / float(len(unique_sessions))
        rest_counts = {sid: int(count) for sid, count in zip(unique_sessions, counts)}
        for sid, count in rest_counts.items():
            sid_mask = rest_mask & (session_id == sid)
            weights[sid_mask] = target_per_session / float(count)

        summary["enabled"] = True
        summary["rest_counts"] = rest_counts
        summary["target_per_session"] = target_per_session
        summary["expected_rest_mass"] = {
            sid: float(np.sum(weights[rest_mask & (session_id == sid)])) for sid in rest_counts
        }
        summary["weight_range"] = [float(weights.min()), float(weights.max())]
        return weights, summary

    event_id = None
    if meta and "event_id" in meta:
        try:
            candidate = np.asarray(meta["event_id"])
            if candidate.ndim != 0 and len(candidate) == len(y_action):
                event_id = candidate.astype(np.int64)
        except Exception:
            event_id = None
    if event_id is None:
        summary["enabled"] = False
        summary["reason"] = "missing_event_id"
        return None, summary

    unique_all_sessions = np.unique(session_id[(session_id != "") & (session_id != "UNKNOWN")])
    aux_sessions: List[str] = []
    core_sessions: List[str] = []
    for sid in unique_all_sessions.tolist():
        sid_mask = session_id == sid
        sid_actions = np.unique(y_action[sid_mask])
        if len(sid_actions) == 1 and int(sid_actions[0]) == int(ACTION_REST):
            aux_sessions.append(str(sid))
        else:
            core_sessions.append(str(sid))

    core_rest_mask = rest_mask & np.isin(session_id, np.asarray(core_sessions, dtype="U"))
    aux_rest_mask = rest_mask & np.isin(session_id, np.asarray(aux_sessions, dtype="U"))
    if not np.any(core_rest_mask):
        summary["enabled"] = False
        summary["reason"] = "no_core_rest_samples"
        return None, summary

    total_rest = float(np.sum(rest_mask))
    if np.any(aux_rest_mask):
        target_core_mass = 0.70 * total_rest
        target_aux_mass = 0.30 * total_rest
    else:
        target_core_mass = total_rest
        target_aux_mass = 0.0

    core_event_ids = np.unique(event_id[core_rest_mask])
    if len(core_event_ids) == 0:
        summary["enabled"] = False
        summary["reason"] = "no_core_rest_events"
        return None, summary

    core_event_counts = {
        int(eid): int(np.sum(core_rest_mask & (event_id == eid)))
        for eid in core_event_ids.tolist()
    }
    target_per_event = float(target_core_mass) / float(len(core_event_ids))
    for eid, count in core_event_counts.items():
        eid_mask = core_rest_mask & (event_id == int(eid))
        weights[eid_mask] = target_per_event / float(count)

    aux_session_counts: Dict[str, int] = {}
    if np.any(aux_rest_mask):
        unique_aux_sessions, aux_counts = np.unique(session_id[aux_rest_mask], return_counts=True)
        target_per_aux_session = float(target_aux_mass) / float(len(unique_aux_sessions))
        aux_session_counts = {
            str(sid): int(count)
            for sid, count in zip(unique_aux_sessions.tolist(), aux_counts.tolist())
        }
        for sid, count in aux_session_counts.items():
            sid_mask = aux_rest_mask & (session_id == sid)
            weights[sid_mask] = target_per_aux_session / float(count)

    summary["enabled"] = True
    summary["core_sessions"] = core_sessions
    summary["aux_sessions"] = aux_sessions
    summary["core_rest_event_counts"] = core_event_counts
    summary["aux_rest_session_counts"] = aux_session_counts
    summary["expected_rest_mass"] = {
        "core_events": {
            str(eid): float(np.sum(weights[core_rest_mask & (event_id == int(eid))]))
            for eid in core_event_ids.tolist()
        },
        "aux_sessions": {
            sid: float(np.sum(weights[aux_rest_mask & (session_id == sid)]))
            for sid in aux_session_counts
        },
        "core_total": float(np.sum(weights[core_rest_mask])),
        "aux_total": float(np.sum(weights[aux_rest_mask])) if np.any(aux_rest_mask) else 0.0,
    }
    summary["weight_range"] = [float(weights.min()), float(weights.max())]
    return weights, summary


def _resolve_action_class_weights(
    *,
    action_weights: Any,
    n_actions: int,
    rest_weight: float,
) -> Tuple[torch.Tensor, bool]:
    parsed = _parse_action_weights(action_weights, n_actions)
    if parsed is not None:
        return parsed, True
    weights = torch.ones(n_actions, dtype=torch.float32)
    if ACTION_REST < n_actions:
        weights[ACTION_REST] = max(0.0, float(rest_weight))
    return weights, False


def _compute_batch_losses(
    *,
    finger_logits: torch.Tensor,
    action_logits: torch.Tensor,
    y_finger: torch.Tensor,
    y_action: torch.Tensor,
    action_loss_fn: nn.Module,
    finger_loss_fn: nn.Module,
    loss_action_weight: float,
    rest_finger_loss_weight: float,
    n_finger_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    loss_action = action_loss_fn(action_logits, y_action)
    mask_nr = y_action != ACTION_REST
    if mask_nr.any():
        nr_targets = y_finger[mask_nr]
        if uses_active_finger_head(n_finger_classes):
            if torch.any(nr_targets <= int(FINGER_NONE)):
                raise ValueError(
                    "Non-REST windows include FINGER_NONE targets, which are invalid for an active finger head."
                )
            nr_targets = nr_targets - 1
        loss_finger_non_rest = finger_loss_fn(finger_logits[mask_nr], nr_targets)
    else:
        loss_finger_non_rest = torch.tensor(0.0, device=action_logits.device)

    mask_rest = y_action == ACTION_REST
    if (
        not uses_active_finger_head(n_finger_classes)
        and rest_finger_loss_weight > 0.0
        and mask_rest.any()
    ):
        rest_targets = torch.full_like(y_finger[mask_rest], int(FINGER_NONE))
        loss_finger_rest = finger_loss_fn(finger_logits[mask_rest], rest_targets)
    else:
        loss_finger_rest = torch.tensor(0.0, device=action_logits.device)

    loss = loss_action + float(loss_action_weight) * (
        loss_finger_non_rest + float(rest_finger_loss_weight) * loss_finger_rest
    )
    return loss, loss_action, loss_finger_non_rest, loss_finger_rest


def _window_idx_leakage_check(
    meta: Dict[str, Any],
    y_action: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    seed: int,
    threshold: float,
    strict: bool,
):
    if not meta:
        return
    if len(np.unique(y_action)) < 2:
        return
    if len(train_idx) == 0 or len(test_idx) == 0:
        return
    window_idx = None
    key_used = None
    for key in ("window_idx", "global_window_idx"):
        if key not in meta:
            continue
        try:
            arr = np.asarray(meta[key])
        except Exception:
            continue
        if arr.ndim == 0 or len(arr) != len(y_action):
            continue
        window_idx = arr.reshape(-1)
        key_used = key
        break
    if window_idx is None:
        return

    if window_idx.dtype.kind in "OUS":
        _, inv = np.unique(window_idx.astype("U"), return_inverse=True)
        X_idx = inv.astype(np.float32).reshape(-1, 1)
    else:
        X_idx = window_idx.astype(np.float32).reshape(-1, 1)

    try:
        from sklearn.tree import DecisionTreeClassifier
    except Exception:
        return

    clf = DecisionTreeClassifier(random_state=seed, max_depth=5)
    clf.fit(X_idx[train_idx], y_action[train_idx])
    acc = float(clf.score(X_idx[test_idx], y_action[test_idx]))
    if acc > float(threshold):
        print(
            "[WARN] window_idx leakage proxy: "
            f"decision tree accuracy={acc:.3f} using {key_used}. "
            "Protocol order may be confounded; consider counterbalancing/randomizing trial order."
        )
        if strict:
            raise RuntimeError(
                f"window_idx leakage accuracy {acc:.3f} exceeded threshold {threshold:.3f}"
            )


def build_arg_parser():
    p = argparse.ArgumentParser(
        description=(
            "Step 2: train the CNN+LSTM finger/action model from Step 1b windows "
            "and write a run directory with model artifacts."
        )
    )
    input_group = p.add_argument_group("input selection")
    input_group.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Load training settings from a JSON config file.",
    )
    input_group.add_argument(
        "--session-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Canonical session directory. Defaults are resolved from this path and outputs are written under processed/models/<run_id>/.",
    )
    input_group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Explicit model run output directory. Overrides the run path derived from --session-dir.",
    )
    input_group.add_argument(
        "--npz",
        type=str,
        default=DEFAULT_NPZ,
        metavar="PATH",
        help="Path to the window dataset NPZ.",
    )
    input_group.add_argument(
        "--subject-id",
        type=str,
        default="2-M16",
        metavar="ID",
        help="Filter the dataset to a single subject_id before splitting.",
    )

    training_group = p.add_argument_group("training")
    training_group.add_argument(
        "--epochs", type=int, default=EPOCHS, help="Number of training epochs"
    )
    training_group.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Training batch size"
    )
    training_group.add_argument("--lr", type=float, default=LR, help="Learning rate.")
    training_group.add_argument("--seed", type=int, default=SEED, help="Random seed.")
    training_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device. 'auto' picks CUDA or MPS when available.",
    )
    training_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (0 = main process).",
    )
    training_group.add_argument(
        "--pin-memory",
        action="store_true",
        help="Pin DataLoader memory (useful for CUDA).",
    )
    training_group.add_argument(
        "--loss-action-weight",
        type=float,
        default=LOSS_ACTION_WEIGHT,
        help="Relative weight applied to the action-head loss term.",
    )
    training_group.add_argument(
        "--rest-weight",
        type=float,
        default=REST_WEIGHT,
        help="Class weight for the REST action. Use 0 to ignore REST.",
    )
    training_group.add_argument(
        "--action-weights",
        type=str,
        default=None,
        help=(
            "Per-action loss weights in REST,OPEN,CLOSE order, or JSON list/dict. "
            "Overrides --rest-weight when provided."
        ),
    )
    training_group.add_argument(
        "--finger-weights",
        type=str,
        default=None,
        help=(
            "Per-finger loss weights. Provide a comma/space list for finger IDs "
            "0..N-1 (e.g. '1,1,1,1,1,0.5' to downweight pinky), or JSON list/dict "
            "(e.g. '[1,1,1,1,1,0.5]' or '{\"pinky\":0.5}')."
        ),
    )
    training_group.add_argument(
        "--active-finger-head",
        dest="active_finger_head",
        action="store_true",
        default=ACTIVE_FINGER_HEAD,
        help=(
            "Train the finger head on active fingers only (THUMB..PINKY). "
            "REST is then handled solely by the action head."
        ),
    )
    training_group.add_argument(
        "--no-active-finger-head",
        dest="active_finger_head",
        action="store_false",
        help="Keep the legacy 6-class finger head with NONE as an explicit class.",
    )
    training_group.add_argument(
        "--rest-finger-loss-weight",
        type=float,
        default=REST_FINGER_LOSS_WEIGHT,
        help="Additional finger-head loss weight applied on REST windows toward NONE.",
    )
    training_group.add_argument(
        "--rest-balance-mode",
        type=str,
        default=REST_BALANCE_MODE,
        choices=["none", "session_equalized", "core_event_equalized"],
        help=(
            "Reweight REST windows within each training epoch. "
            "'session_equalized' keeps total REST mass constant while balancing it across source sessions. "
            "'core_event_equalized' emphasizes core-session REST events while still retaining auxiliary quiet-rest support."
        ),
    )
    training_group.add_argument(
        "--window-preprocess",
        type=str,
        default=WINDOW_PREPROCESS,
        choices=["none", "center", "center_detrend"],
        help=(
            "Per-window preprocessing applied before fitting and applying the channel normalizer. "
            "'center_detrend' removes DC offset and linear drift within each window."
        ),
    )
    split_group = p.add_argument_group("split and leakage controls")
    split_group.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of windows reserved for the evaluation split.",
    )
    split_group.add_argument(
        "--calibration-size",
        type=float,
        default=0.1,
        help="Fraction of the training split held out for post-hoc temperature scaling. Use 0 to disable.",
    )
    split_group.add_argument(
        "--split-mode",
        type=str,
        default="group_trial",
        choices=["group_trial", "holdout_session"],
        help="How train/test data are separated: by trial/event groups or by session_id.",
    )
    split_group.add_argument(
        "--aux-rest-session-policy",
        type=str,
        default=AUX_REST_SESSION_POLICY,
        choices=["none", "auto_train_only", "train_mixed_rest_test_aux_rest"],
        help=(
            "How to use source sessions that contain only REST windows. "
            "'auto_train_only' excludes pure-rest sessions from the main test/calibration split "
            "and appends them back into training as auxiliary REST. "
            "'train_mixed_rest_test_aux_rest' keeps all mixed-session REST in training, "
            "splits non-REST on the mixed session, and holds out quiet-rest events for test."
        ),
    )
    split_group.add_argument(
        "--purge-seconds",
        type=float,
        default=0.0,
        help="Drop training windows within this many seconds of any test window from the same session.",
    )
    split_group.add_argument(
        "--hop-seconds",
        type=float,
        default=None,
        help="Window hop size, in seconds, used by leakage-purge heuristics when needed.",
    )
    split_group.add_argument(
        "--non-rest-only",
        action="store_true",
        help="Train only on non-REST windows.",
    )
    split_group.add_argument(
        "--window-idx-leak-threshold",
        type=float,
        default=0.65,
        help="Warn if a window-index-only leakage probe exceeds this accuracy.",
    )
    split_group.add_argument(
        "--strict-leakage",
        action="store_true",
        help="Fail training if leakage checks exceed their thresholds.",
    )
    output_group = p.add_argument_group("outputs")
    output_group.add_argument(
        "--save-model",
        type=str,
        default=DEFAULT_MODEL,
        metavar="PATH",
        help="Output path for trained model weights.",
    )
    output_group.add_argument(
        "--save-scaler",
        type=str,
        default=DEFAULT_SCALER,
        metavar="PATH",
        help="Output path for the fitted channel normalizer.",
    )
    output_group.add_argument(
        "--save-preds",
        type=str,
        default=DEFAULT_PREDS,
        metavar="PATH",
        help="Output path for cached test predictions.",
    )
    output_group.add_argument(
        "--save-temperature",
        type=str,
        default=DEFAULT_TEMPERATURE,
        metavar="PATH",
        help="Output path for temperature-scaling metadata.",
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


def resolve_npz_path(path_str: str, *, base_dir: Optional[Path] = None) -> Path:
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        base = base_dir if base_dir is not None else Path.cwd()
        candidate = (base / candidate).resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"NPZ file not found: {candidate}")
    return candidate


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


def _next_available_dir(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.name}_{i:02d}")
        if not candidate.exists():
            return candidate
    return path


def resolve_output_paths(
    args, subject: str, exp_hash: str, *, session_dir: Optional[Path] = None
):
    if getattr(args, "run_dir", None):
        run_dir = Path(str(args.run_dir)).expanduser()
    elif session_dir is not None:
        models_root = SessionLayout(session_dir).models_root
        models_root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = _next_available_dir(models_root / run_id)
    else:
        subject_safe = subject or "UNKNOWN"
        run_dir = ROOT_DIR / "data/models" / subject_safe / exp_hash

    run_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(path_str: str, default_name: str) -> Path:
        if path_str == default_name:
            return run_dir / default_name
        candidate = Path(path_str)
        if not candidate.is_absolute() and candidate.parent == Path("."):
            return run_dir / candidate.name
        return candidate

    model_path = _resolve(args.save_model, DEFAULT_MODEL)
    scaler_path = _resolve(args.save_scaler, DEFAULT_SCALER)
    preds_path = _resolve(args.save_preds, DEFAULT_PREDS)
    temperature_path = _resolve(args.save_temperature, DEFAULT_TEMPERATURE)
    return run_dir, model_path, scaler_path, preds_path, temperature_path


def _validate_indices(idx: np.ndarray, n_samples: int, name: str):
    if idx.size == 0:
        return
    if idx.min() < 0 or idx.max() >= n_samples:
        raise ValueError(
            f"{name} indices out of bounds: min={idx.min()} max={idx.max()} n_samples={n_samples}"
        )


def _load_session_meta(session_dir: Path) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for name in ("meta.json", "manifest.json", "session_meta.json"):
        path = session_dir / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if isinstance(payload, dict):
            meta.update(payload)
    return meta


def _infer_subject_id_from_session_dir(session_dir: Path) -> Optional[str]:
    meta = _load_session_meta(session_dir)
    subject = meta.get("subject_id")
    if subject:
        return str(subject)
    name = session_dir.name
    match = re.match(r"^(?P<subject>.+?)_\d{8}_\d{6}$", name)
    if match:
        return match.group("subject")
    return None


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    settings = _load_config(args.config)
    _apply_config_to_args(args, settings, defaults)
    set_seed(args.seed)

    session_dir_path: Optional[Path] = None
    explicit_npz = args.npz not in (DEFAULT_NPZ, f"./{DEFAULT_NPZ}")
    selection_source = "legacy_explicit"
    config_dir = (
        Path(args.config).expanduser().resolve().parent
        if getattr(args, "config", None)
        else None
    )

    if getattr(args, "session_dir", None):
        session_dir_path = resolve_session_dir(str(args.session_dir))
        if not session_dir_path.exists():
            print("Session selection source: session_dir")
            print(f"Session dir not found: {session_dir_path}")
            return 2
        if explicit_npz:
            print(
                "⚠️ Both --session-dir and --npz provided; using explicit --npz path."
            )
            selection_source = "legacy_explicit"
            npz_path = resolve_npz_path(args.npz, base_dir=session_dir_path)
        else:
            selection_source = "session_dir"
            npz_path = SessionLayout(session_dir_path).windows_npz
    else:
        if not explicit_npz:
            print("Session selection source: legacy_explicit")
            print(
                "❌ Missing --session-dir. Provide --session-dir or explicit --npz PATH."
            )
            return 2
        npz_path = resolve_npz_path(args.npz, base_dir=config_dir or Path.cwd())

    if not npz_path.exists():
        print(f"NPZ file not found: {npz_path}")
        return 2
    args.npz = str(npz_path)
    print(f"Session selection source: {selection_source}")
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

    # Keep original sample positions so cached predictions can be validated later
    # against the exact split/data snapshot used in this training run.
    # Global index mapping (original dataset positions)
    global_indices = np.arange(len(y_action_full), dtype=np.int64)

    # ===== OPTIONAL SUBJECT FILTER =====
    X = X_full
    y_action = y_action_full
    y_finger = y_finger_full
    n_full = len(y_action_full)
    subject = None

    if session_dir_path:
        inferred = (
            _infer_subject_id_from_session_dir(session_dir_path)
            or infer_subject_id_from_meta(meta, n_full)
            or infer_subject_id_from_npz(npz_path)
        )
        if inferred and args.subject_id and args.subject_id != inferred:
            print(
                f"[session-dir] ⚠️ subject_id mismatch: requested={args.subject_id!r} inferred={inferred!r}"
            )
        if not args.subject_id and inferred:
            args.subject_id = inferred

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
    # Continue training after optionally dropping bad samples.
    if n_bad >= 0:
        if n_bad > 0:
            print(f"[WARN] Dropping {n_bad}/{len(X)} samples with NaN/Inf in X before training.")
            X = X[finite_mask]
            y_action = y_action[finite_mask]
            y_finger = y_finger[finite_mask]
            global_indices = global_indices[finite_mask]
            meta, _ = mask_meta(meta, finite_mask, len(finite_mask))
            if len(X) == 0:
                raise RuntimeError(
                    "All samples were dropped due to NaN/Inf values. Check upstream data collection."
                )

        def class_counts(y):
            u, c = np.unique(y, return_counts=True)
            return dict(zip(u.tolist(), c.tolist()))

        print(f"Action class counts: {class_counts(y_action)}")
        print(f"Finger class counts: {class_counts(y_finger)}")

        exp_hash = resolve_experiment_hash(meta, len(y_action))
        log_experiment(subject, exp_hash, "STEP_2_TRAIN")

        # Save all training artifacts together; downstream eval/report steps resolve
        # this run folder and read model/scaler/predictions from it.
        run_dir, save_model_path, save_scaler_path, save_preds_path, save_temperature_path = resolve_output_paths(
            args, subject, exp_hash, session_dir=session_dir_path
        )
        for path in [save_model_path, save_scaler_path, save_preds_path, save_temperature_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving outputs to: {run_dir}")
        print(
            f"Output paths: model={save_model_path}, scaler={save_scaler_path}, "
            f"preds={save_preds_path}, temperature={save_temperature_path}"
        )

        aux_rest_plan = _resolve_auxiliary_rest_sessions(
            y_action,
            meta if meta else {},
            policy=args.aux_rest_session_policy,
        )
        if aux_rest_plan.get("enabled"):
            print(
                "Auxiliary REST-only sessions: "
                f"train_only={aux_rest_plan.get('aux_sessions', [])} "
                f"core={aux_rest_plan.get('core_sessions', [])}"
            )
            if args.split_mode == "holdout_session" and len(aux_rest_plan.get("core_sessions", [])) < 2:
                print(
                    "❌ split_mode=holdout_session requires at least two non-rest sessions "
                    "after excluding REST-only auxiliary sessions."
                )
                return 2
        else:
            print(
                "Auxiliary REST-only session policy: "
                f"disabled reason={aux_rest_plan.get('reason', 'not_requested')}"
            )
        try:
            split_plan = compose_split_indices(
                y_action,
                y_finger,
                meta if meta else None,
                test_size=args.test_size,
                random_state=args.seed,
                split_mode=args.split_mode,
                purge_seconds=args.purge_seconds,
                hop_seconds=args.hop_seconds,
                allow_fallback=False,
                aux_rest_session_policy=args.aux_rest_session_policy,
            )
        except ValueError as exc:
            print(f"❌ Split failed: {exc}")
            return 2

        train_idx = np.asarray(split_plan["train_idx"], dtype=np.int64)
        test_idx = np.asarray(split_plan["test_idx"], dtype=np.int64)
        calib_exempt_idx = np.asarray(split_plan["train_locked_idx"], dtype=np.int64)
        split_idx = np.asarray(split_plan["main_split_idx"], dtype=np.int64)
        train_local_idx = np.asarray(split_plan["main_train_local"], dtype=np.int64)
        test_local_idx = np.asarray(split_plan["main_test_local"], dtype=np.int64)
        train_core_idx = np.asarray(split_plan["core_train_idx"], dtype=np.int64)
        aux_train_idx = np.asarray(split_plan["aux_train_idx"], dtype=np.int64)
        aux_test_idx = np.asarray(split_plan["aux_test_idx"], dtype=np.int64)
        n_samples = len(y_action)
        _validate_indices(train_idx, n_samples, "train")
        _validate_indices(test_idx, n_samples, "test")

        if len(split_idx):
            split_meta = _subset_meta(meta, split_idx, len(y_action)) if meta else {}
            try:
                groups = _split_groups_from_meta(split_meta, len(split_idx), args.split_mode)
            except ValueError as exc:
                print(f"❌ Split diagnostics failed: {exc}")
                return 2
            try:
                assert_no_group_overlap(
                    groups,
                    np.asarray(train_local_idx, dtype=np.int64),
                    np.asarray(test_local_idx, dtype=np.int64),
                )
            except RuntimeError as exc:
                print(f"❌ Leakage guard tripped: {exc}")
                return 2

            _log_split_diagnostics(
                groups,
                y_action[split_idx],
                y_finger[split_idx],
                np.asarray(train_local_idx, dtype=np.int64),
                np.asarray(test_local_idx, dtype=np.int64),
            )
            _window_idx_leakage_check(
                split_meta,
                y_action[split_idx],
                np.asarray(train_local_idx, dtype=np.int64),
                np.asarray(test_local_idx, dtype=np.int64),
                seed=args.seed,
                threshold=args.window_idx_leak_threshold,
                strict=args.strict_leakage,
            )
        else:
            groups = np.array([], dtype=np.int64)
            split_meta = {}

        if args.aux_rest_session_policy == "train_mixed_rest_test_aux_rest":
            core_rest_idx = np.asarray(aux_rest_plan.get("core_rest_idx", []), dtype=np.int64)
            print(
                "Mixed REST train-only split: "
                f"core_rest_train={len(core_rest_idx)} "
                f"aux_rest_train={len(aux_train_idx)} "
                f"aux_rest_test={len(aux_test_idx)}"
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

        calib_idx = np.array([], dtype=np.int64)
        train_fit_idx = np.asarray(train_idx, dtype=np.int64)
        calibration_size = max(0.0, float(args.calibration_size))
        if calibration_size > 0.0:
            locked_mask = (
                np.isin(train_fit_idx, calib_exempt_idx)
                if len(calib_exempt_idx)
                else np.zeros(len(train_fit_idx), dtype=bool)
            )
            calib_candidate_idx = train_fit_idx[~locked_mask]
            train_locked_only_idx = train_fit_idx[locked_mask]
            train_fit_idx, calib_idx = _split_calibration_indices(
                calib_candidate_idx,
                y_action,
                y_finger,
                meta if meta else {},
                calibration_size=calibration_size,
                random_state=args.seed,
                split_mode=args.split_mode,
            )
            if len(train_locked_only_idx):
                train_fit_idx = np.concatenate(
                    [np.asarray(train_fit_idx, dtype=np.int64), train_locked_only_idx]
                ).astype(np.int64)
                train_fit_idx = np.unique(train_fit_idx)
            if len(calib_idx) == 0:
                print("Temperature scaling: skipped (no calibration fold available).")
            else:
                print(
                    f"Temperature scaling split: fit={len(train_fit_idx)} calib={len(calib_idx)} "
                    f"(requested={calibration_size:.3f})"
                )

        # ===== SLICE =====
        X_train, X_test = X[train_fit_idx], X[test_idx]
        y_action_train, y_action_test = y_action[train_fit_idx], y_action[test_idx]
        y_finger_train, y_finger_test = y_finger[train_fit_idx], y_finger[test_idx]
        X_calib = X[calib_idx] if len(calib_idx) else None
        y_action_calib = y_action[calib_idx] if len(calib_idx) else None
        y_finger_calib = y_finger[calib_idx] if len(calib_idx) else None
        train_meta = _subset_meta(meta, train_fit_idx, len(y_action)) if meta else {}

        n_fingers = 5 if bool(args.active_finger_head) else int(np.max(y_finger)) + 1
        n_actions = int(np.max(y_action)) + 1

        # ===== NORMALIZE =====
        preprocess_cfg = _preprocess_config_from_mode(args.window_preprocess)
        normalizer = fit_channel_normalizer(X_train, preprocess=preprocess_cfg)
        X_train = apply_channel_normalizer(X_train, normalizer)
        if X_calib is not None:
            X_calib = apply_channel_normalizer(X_calib, normalizer)
        X_test = apply_channel_normalizer(X_test, normalizer)
        save_normalizer(save_scaler_path, normalizer)

        sample_weights, sample_weight_summary = _build_train_sample_weights(
            y_action_train,
            train_meta,
            balance_mode=args.rest_balance_mode,
        )
        train_sampler = None
        train_shuffle = True
        if sample_weights is not None:
            train_sampler = WeightedRandomSampler(
                torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
            train_shuffle = False
            print(
                "REST balancing enabled: "
                f"mode={args.rest_balance_mode} "
                f"rest_counts={sample_weight_summary.get('rest_counts', {})} "
                f"expected_rest_mass={sample_weight_summary.get('expected_rest_mass', {})}"
            )
        else:
            print(
                "REST balancing disabled: "
                f"mode={args.rest_balance_mode} "
                f"reason={sample_weight_summary.get('reason', 'not_requested')}"
            )

        # ===== DATALOADERS =====
        train_loader = DataLoader(
            EEGWindowDataset(X_train, y_finger_train, y_action_train),
            batch_size=args.batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            drop_last=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=bool(args.pin_memory),
        )
        test_loader = DataLoader(
            EEGWindowDataset(X_test, y_finger_test, y_action_test),
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=bool(args.pin_memory),
        )
        calib_loader = None
        if X_calib is not None and y_action_calib is not None and y_finger_calib is not None and len(y_action_calib):
            calib_loader = DataLoader(
                EEGWindowDataset(X_calib, y_finger_calib, y_action_calib),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=max(0, int(args.num_workers)),
                pin_memory=bool(args.pin_memory),
            )

        # ===== MODEL =====
        model = CNNLSTMFingerActionNet(
            n_channels=X.shape[2], n_fingers=n_fingers, n_actions=n_actions
        )
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)
        model.to(device)

        opt = torch.optim.Adam(model.parameters(), lr=args.lr)

        # Finger loss (optionally weighted per finger)
        finger_weights = None
        if getattr(args, "finger_weights", None) is not None:
            try:
                finger_weights = _parse_finger_weights(args.finger_weights, n_fingers)
            except ValueError as exc:
                print(f"❌ Invalid --finger-weights: {exc}")
                return 2
        if finger_weights is not None:
            loss_f = nn.CrossEntropyLoss(weight=finger_weights.to(device))
            print(f"Using finger weights: {finger_weights.tolist()}")
        else:
            loss_f = nn.CrossEntropyLoss()
        if uses_active_finger_head(n_fingers) and float(args.rest_finger_loss_weight) > 0.0:
            print(
                "⚠️ rest_finger_loss_weight is ignored when active_finger_head is enabled."
            )

        # Action loss (explicit action weights override scalar REST weighting)
        try:
            action_weights, action_weights_override = _resolve_action_class_weights(
                action_weights=getattr(args, "action_weights", None),
                n_actions=n_actions,
                rest_weight=float(args.rest_weight),
            )
        except ValueError as exc:
            print(f"❌ Invalid --action-weights: {exc}")
            return 2
        loss_a = nn.CrossEntropyLoss(weight=action_weights.to(device))
        if action_weights_override:
            print(f"Using action weights: {action_weights.tolist()}")
        else:
            print(f"Using scalar rest_weight via action weights: {action_weights.tolist()}")

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

                loss, loss_action, loss_finger_non_rest, loss_finger_rest = _compute_batch_losses(
                    finger_logits=f_out,
                    action_logits=a_out,
                    y_finger=yfb,
                    y_action=yab,
                    action_loss_fn=loss_a,
                    finger_loss_fn=loss_f,
                    loss_action_weight=float(args.loss_action_weight),
                    rest_finger_loss_weight=float(args.rest_finger_loss_weight),
                    n_finger_classes=n_fingers,
                )
                loss.backward()
                opt.step()

                total_loss += loss.item() * Xb.size(0)

                preds_action = torch.argmax(a_out, dim=1)
                correct_action += (preds_action == yab).sum().item()
                total_action += yab.numel()

                mask_nr = yab != ACTION_REST
                if mask_nr.any():
                    preds_finger = torch.argmax(f_out[mask_nr], dim=1)
                    if uses_active_finger_head(n_fingers):
                        preds_finger = preds_finger + 1
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
            "calibration_size": float(args.calibration_size),
            "loss_action_weight": args.loss_action_weight,
            "rest_weight": float(args.rest_weight),
            "action_weights": action_weights.tolist(),
            "action_weights_override": bool(action_weights_override),
            "rest_balance_mode": args.rest_balance_mode,
            "active_finger_head": bool(args.active_finger_head),
            "rest_finger_loss_weight": float(args.rest_finger_loss_weight),
            "finger_weights": finger_weights.tolist() if finger_weights is not None else None,
            "window_preprocess": args.window_preprocess,
            "test_size": args.test_size,
            "split_mode": args.split_mode,
            "aux_rest_session_policy": args.aux_rest_session_policy,
            "purge_seconds": float(args.purge_seconds),
            "hop_seconds": float(args.hop_seconds) if args.hop_seconds is not None else None,
            "window_idx_leak_threshold": float(args.window_idx_leak_threshold),
            "strict_leakage": bool(args.strict_leakage),
            "non_rest_only": bool(args.non_rest_only),
            "npz_path": str(npz_path),
            "session_dir": str(session_dir_path) if session_dir_path else None,
            "run_dir": str(run_dir),
            "n_fingers": n_fingers,
            "n_actions": n_actions,
            "input_shape": list(X.shape[1:]),
            "normalizer": {
                "type": normalizer.get("type", "unknown"),
                "channels": normalizer.get("channels", None),
                "preprocess": normalizer.get("preprocess", {}),
            },
            "auxiliary_rest_sessions": {
                "enabled": bool(aux_rest_plan.get("enabled", False)),
                "reason": str(aux_rest_plan.get("reason", "")),
                "aux_sessions": [str(v) for v in aux_rest_plan.get("aux_sessions", [])],
                "core_sessions": [str(v) for v in aux_rest_plan.get("core_sessions", [])],
                "aux_train_count": int(len(aux_train_idx)),
                "aux_test_count": int(len(aux_test_idx)),
                "core_split_count": int(len(split_idx)),
                "test_count": int(len(test_idx)),
                "core_rest_train_count": int(len(aux_rest_plan.get("core_rest_idx", []))),
                "session_action_counts": aux_rest_plan.get("session_action_counts", {}),
            },
            "train_sampler": sample_weight_summary,
            "device": str(device),
            "model": "CNNLSTMFingerActionNet",
            "subject_id_filter": args.subject_id or "",
            "save_model_path": str(save_model_path),
            "save_scaler_path": str(save_scaler_path),
            "save_preds_path": str(save_preds_path),
            "save_temperature_path": str(save_temperature_path),
        }
        train_config_path = save_model_path.parent / "train_config.json"
        train_config_path.write_text(json.dumps(train_config, indent=2))

        log_config_path = Path("logs") / "experiments" / f"{exp_hash}_train_config.json"
        log_config_path.parent.mkdir(parents=True, exist_ok=True)
        log_config_path.write_text(json.dumps(train_config, indent=2))

        # ===== INFERENCE ON TEST SET =====
        temperature_state = TemperatureScalingState(
            action_temperature=1.0,
            finger_temperature=1.0,
            fit_sample_count=int(len(calib_idx)),
            fit_non_rest_count=int(
                np.sum(np.asarray(y_action_calib, dtype=np.int64) != int(ACTION_REST))
            )
            if y_action_calib is not None
            else 0,
            source="disabled" if calibration_size <= 0.0 else "identity",
            metrics={},
        )

        if calib_loader is not None:
            calib_action_logits = []
            calib_finger_logits = []
            calib_action_labels = []
            calib_finger_labels = []
            with torch.no_grad():
                for Xb, yfb, yab in calib_loader:
                    Xb = Xb.to(device)
                    f_out, a_out = model(Xb)
                    calib_action_logits.append(a_out.detach().cpu().numpy())
                    calib_finger_logits.append(f_out.detach().cpu().numpy())
                    calib_action_labels.append(yab.numpy())
                    calib_finger_labels.append(yfb.numpy())

            action_logits_calib = np.concatenate(calib_action_logits, axis=0).astype(np.float32)
            finger_logits_calib = np.concatenate(calib_finger_logits, axis=0).astype(np.float32)
            y_action_calib_np = np.concatenate(calib_action_labels, axis=0).astype(np.int64)
            y_finger_calib_np = np.concatenate(calib_finger_labels, axis=0).astype(np.int64)
            finger_mask_calib = y_action_calib_np != int(ACTION_REST)

            action_temp, action_metrics = _fit_temperature(
                action_logits_calib,
                y_action_calib_np,
                device=device,
            )
            if np.any(finger_mask_calib):
                finger_targets_calib = y_finger_calib_np[finger_mask_calib]
                if uses_active_finger_head(n_fingers):
                    finger_targets_calib = finger_targets_calib - 1
                finger_temp, finger_metrics = _fit_temperature(
                    finger_logits_calib[finger_mask_calib],
                    finger_targets_calib,
                    device=device,
                )
            else:
                finger_temp, finger_metrics = 1.0, {"nll_before": None, "nll_after": None}

            temperature_state = TemperatureScalingState(
                action_temperature=float(action_temp),
                finger_temperature=float(finger_temp),
                fit_sample_count=int(len(y_action_calib_np)),
                fit_non_rest_count=int(np.sum(finger_mask_calib)),
                source="fit_on_holdout",
                metrics={
                    "action": action_metrics,
                    "finger": finger_metrics,
                },
            )
            print(
                "Temperature scaling: "
                f"action={temperature_state.action_temperature:.4f} "
                f"finger={temperature_state.finger_temperature:.4f}"
            )
        save_temperature_scaling(save_temperature_path, temperature_state)

        all_action_probs = []
        all_finger_probs = []
        with torch.no_grad():
            for Xb, yfb, yab in test_loader:
                Xb = Xb.to(device)
                f_out, a_out = model(Xb)
                a_out = apply_temperature_to_logits(
                    a_out, temperature_state.action_temperature
                )
                f_out = apply_temperature_to_logits(
                    f_out, temperature_state.finger_temperature
                )
                all_finger_probs.append(torch.softmax(f_out, dim=1).cpu().numpy())
                all_action_probs.append(torch.softmax(a_out, dim=1).cpu().numpy())

        action_probs = np.concatenate(all_action_probs, axis=0).astype(np.float32)
        finger_probs = np.concatenate(all_finger_probs, axis=0).astype(np.float32)

        test_action_pred = np.argmax(action_probs, axis=1).astype(np.int64)
        test_action_acc = float(np.mean(test_action_pred == y_action_test.astype(np.int64)))
        test_finger_acc = None
        test_non_rest = (y_action_test.astype(np.int64) != int(ACTION_REST))
        if bool(np.any(test_non_rest)):
            test_finger_pred = decode_finger_predictions(finger_probs[test_non_rest])
            test_finger_acc = float(np.mean(test_finger_pred == y_finger_test[test_non_rest].astype(np.int64)))

        metrics = {
            "schema_version": 1,
            "created_utc": now_utc_iso(),
            "npz_path": safe_resolve(npz_path),
            "run_dir": safe_resolve(run_dir),
            "train": {
                "avg_loss": float(avg_loss),
                "action_acc": float(action_acc),
                "finger_acc": float(finger_acc),
                "epochs": int(args.epochs),
                "batch_size": int(args.batch_size),
                "lr": float(args.lr),
                "seed": int(args.seed),
            },
            "temperature_scaling": {
                "action_temperature": float(temperature_state.action_temperature),
                "finger_temperature": float(temperature_state.finger_temperature),
                "fit_sample_count": int(temperature_state.fit_sample_count),
                "fit_non_rest_count": int(temperature_state.fit_non_rest_count),
                "source": str(temperature_state.source),
                "metrics": temperature_state.metrics or {},
            },
            "test": {
                "action_acc": float(test_action_acc),
                "finger_acc_non_rest": test_finger_acc,
                "n_test": int(len(y_action_test)),
                "n_test_non_rest": int(np.sum(test_non_rest)),
            },
            "artifacts": {
                "model": str(save_model_path.name),
                "scaler": str(save_scaler_path.name),
                "preds": str(save_preds_path.name),
                "temperature_scaling": str(save_temperature_path.name),
            },
        }
        try:
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        except Exception:
            print("⚠️ Failed to write metrics.json")

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
            action_temperature=np.array(
                [float(temperature_state.action_temperature)], dtype=np.float32
            ),
            finger_temperature=np.array(
                [float(temperature_state.finger_temperature)], dtype=np.float32
            ),
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
        print(
            f"✅ Saved: {save_model_path}, {save_scaler_path}, {save_preds_path}, {save_temperature_path}"
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
