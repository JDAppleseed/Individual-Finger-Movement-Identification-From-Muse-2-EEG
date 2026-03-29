from __future__ import annotations

import collections
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from models.finger_action_net import FingerActionNet
from utils.default_recipe import LIVE_INFER_RECIPE_DEFAULTS, PSEUDO_LIVE_RECIPE_DEFAULTS
from utils.eval_utils import (
    resolve_cached_test_indices,
    validate_cached_predictions_with_dataset_info,
)
from utils.label_schema import (
    ACTION_NAMES,
    ACTION_REST,
    FINGER_NAMES,
    decode_finger_predictions,
    decode_finger_predictions_for_actions,
    finger_confidences_for_ids,
    is_valid_action_finger,
)
from utils.live_infer_common import (
    ActuationDecision,
    ReplayRuntimeConfig,
    build_actuation_command_shaper,
    build_actuation_speed_mapper,
    compute_actuation_speed_scalar,
    debounced_should_send,
    is_noop_decision,
    resolve_actuation_candidate,
    uncertainty_gate_passed,
)
from utils.model_outputs import infer_output_dims_from_state_dict
from utils.postprocess import PostprocessSettings, PostprocessState, postprocess_predictions
from utils.runtime_utils import (
    apply_channel_normalizer,
    apply_temperature_to_logits,
    load_normalizer,
    load_temperature_scaling,
    resolve_device,
)
from utils.sequence_data import load_sequence_npz
from utils.session_layout import resolve_latest_run_dir

SUPPORTED_EMBEDDING_SOURCES = ("latent", "logits", "probabilities", "raw")
SUPPORTED_REDUCERS = ("pca", "umap")
SUPPORTED_SAMPLE_STRATEGIES = ("random", "stratified_joint")
SUPPORTED_COLOR_MODES = (
    "true_finger",
    "pred_finger",
    "true_action",
    "pred_action",
    "correctness",
    "applicability_score",
    "action_confidence",
    "finger_confidence",
    "session",
    "subject",
    "split",
)

_CATEGORICAL_COLOR_MODES = {
    "true_finger",
    "pred_finger",
    "true_action",
    "pred_action",
    "correctness",
    "session",
    "subject",
    "split",
}


@dataclass(frozen=True)
class ResolvedArtifacts:
    snapshot_dir: Optional[Path]
    session_dir: Path
    run_dir: Path
    model_path: Path
    scaler_path: Path
    temperature_path: Optional[Path]
    train_config_path: Optional[Path]
    train_config: dict[str, Any]
    dataset_npz: Path
    test_predictions_path: Optional[Path]
    infer_config_path: Optional[Path]
    infer_settings: dict[str, Any]
    replay_manifest_path: Optional[Path]
    replay_manifest: dict[str, Any]


@dataclass(frozen=True)
class InferenceOutputs:
    representation: np.ndarray
    action_logits: np.ndarray
    finger_logits: np.ndarray
    applicability_logits: Optional[np.ndarray]
    action_probs: np.ndarray
    finger_probs: np.ndarray
    applicability_probs: Optional[np.ndarray]


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _safe_resolve(path: Path) -> Path:
    try:
        return path.expanduser().resolve()
    except Exception:
        return path.expanduser()


def _load_json_dict(path: Optional[Path]) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _load_settings_file(path: Optional[Path]) -> dict[str, Any]:
    payload = _load_json_dict(path)
    settings = payload.get("settings")
    if isinstance(settings, dict):
        return dict(settings)
    return payload


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception:
        return None


def _parse_dataset_info(value: Any) -> Optional[dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except Exception:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _unique_nonempty(values: np.ndarray) -> list[str]:
    arr = np.asarray(values).astype("U").reshape(-1)
    keep = (arr != "") & (arr != "UNKNOWN")
    if np.any(keep):
        arr = arr[keep]
    return sorted(np.unique(arr).tolist())


def _candidate_run_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    if root.is_file():
        root = root.parent
    candidates.append(root)
    if root.name == "model_run":
        candidates.append(root)
    candidates.append(root / "model_run")
    if root.name == "models":
        latest = resolve_latest_run_dir(root.parent.parent)
        if latest is not None:
            candidates.append(latest)
    processed_models = root / "processed" / "models"
    if processed_models.exists():
        latest = resolve_latest_run_dir(root)
        if latest is not None:
            candidates.append(latest)
    return [_safe_resolve(candidate) for candidate in candidates]


def _resolve_run_dir(
    run_dir: Optional[str | Path],
    model_path: Optional[str | Path],
) -> tuple[Path, Optional[Path]]:
    if run_dir is not None:
        requested = _safe_resolve(Path(run_dir))
        for candidate in _candidate_run_dirs(requested):
            if (candidate / "finger_action_model.pt").exists():
                snapshot = candidate.parent if candidate.name == "model_run" else None
                return candidate, snapshot
        raise FileNotFoundError(
            f"Could not resolve a run directory from {requested}. "
            "Expected a directory containing finger_action_model.pt or a winning_model snapshot."
        )
    if model_path is not None:
        resolved_model = _safe_resolve(Path(model_path))
        if not resolved_model.exists():
            raise FileNotFoundError(f"Model file not found: {resolved_model}")
        return resolved_model.parent, None
    raise FileNotFoundError("Provide --run-dir or --model-path.")


def _candidate_infer_config(snapshot_dir: Optional[Path], train_config: dict[str, Any]) -> Optional[Path]:
    candidates: list[Path] = []
    if snapshot_dir is not None:
        candidates.append(snapshot_dir / "configs" / "infer.json")
    session_dir_value = train_config.get("session_dir")
    if session_dir_value:
        session_dir = _safe_resolve(Path(str(session_dir_value)))
        candidates.append(session_dir.parents[1] / "config" / "infer.json")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_replay_manifest(
    snapshot_dir: Optional[Path],
    run_dir: Path,
    train_config: dict[str, Any],
) -> Optional[Path]:
    session_dir_value = train_config.get("session_dir")
    session_name = Path(str(session_dir_value)).name if session_dir_value else ""
    run_name = run_dir.name
    candidates: list[Path] = []
    if snapshot_dir is not None:
        if session_name:
            candidates.append(
                snapshot_dir / "pseudo_live" / session_name / "replay_manifest.json"
            )
        candidates.extend(
            sorted((snapshot_dir / "pseudo_live").glob("*/replay_manifest.json"))
            if (snapshot_dir / "pseudo_live").exists()
            else []
        )
    if session_dir_value:
        session_dir = _safe_resolve(Path(str(session_dir_value)))
        candidates.append(
            session_dir / "processed" / "pseudo_live" / run_name / "replay_manifest.json"
        )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_event_space_artifacts(
    *,
    run_dir: Optional[str | Path],
    model_path: Optional[str | Path],
    config_path: Optional[str | Path],
    dataset_npz: Optional[str | Path],
    infer_config_path: Optional[str | Path] = None,
    replay_manifest_path: Optional[str | Path] = None,
) -> ResolvedArtifacts:
    resolved_run_dir, snapshot_dir = _resolve_run_dir(run_dir, model_path)
    snapshot_dir = snapshot_dir if snapshot_dir and snapshot_dir.exists() else None
    model_file = (
        _safe_resolve(Path(model_path))
        if model_path is not None
        else resolved_run_dir / "finger_action_model.pt"
    )
    scaler_file = resolved_run_dir / "scaler.npz"
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")

    train_config_file = (
        _safe_resolve(Path(config_path))
        if config_path is not None
        else resolved_run_dir / "train_config.json"
    )
    train_config = _load_json_dict(train_config_file)
    if not train_config_file.exists():
        train_config_file = None
    session_dir_value = train_config.get("session_dir")
    if session_dir_value:
        session_dir = _safe_resolve(Path(str(session_dir_value)))
    elif (
        resolved_run_dir.parent.name == "models"
        and resolved_run_dir.parent.parent.name == "processed"
    ):
        session_dir = resolved_run_dir.parent.parent.parent
    else:
        session_dir = resolved_run_dir
    if not session_dir.exists():
        session_dir = resolved_run_dir

    if dataset_npz is not None:
        dataset_file = _safe_resolve(Path(dataset_npz))
    else:
        npz_from_train = train_config.get("npz_path")
        if npz_from_train:
            dataset_file = _safe_resolve(Path(str(npz_from_train)))
        elif (
            resolved_run_dir.parent.name == "models"
            and resolved_run_dir.parent.parent.name == "processed"
        ):
            dataset_file = resolved_run_dir.parent.parent / "eeg_windows.npz"
        else:
            raise FileNotFoundError(
                "Could not resolve dataset NPZ. Provide --dataset-npz or use a run directory with train_config.json."
            )
    if not dataset_file.exists():
        raise FileNotFoundError(f"Dataset NPZ not found: {dataset_file}")

    temperature_path = resolved_run_dir / "temperature_scaling.json"
    if not temperature_path.exists():
        temperature_path = None

    test_predictions_path = resolved_run_dir / "test_predictions.npz"
    if not test_predictions_path.exists():
        test_predictions_path = None

    infer_config_file = (
        _safe_resolve(Path(infer_config_path))
        if infer_config_path is not None
        else _candidate_infer_config(snapshot_dir, train_config)
    )
    if infer_config_file is not None and not infer_config_file.exists():
        infer_config_file = None

    replay_manifest_file = (
        _safe_resolve(Path(replay_manifest_path))
        if replay_manifest_path is not None
        else _candidate_replay_manifest(snapshot_dir, resolved_run_dir, train_config)
    )
    if replay_manifest_file is not None and not replay_manifest_file.exists():
        replay_manifest_file = None

    return ResolvedArtifacts(
        snapshot_dir=snapshot_dir,
        session_dir=session_dir,
        run_dir=resolved_run_dir,
        model_path=model_file,
        scaler_path=scaler_file,
        temperature_path=temperature_path,
        train_config_path=train_config_file,
        train_config=train_config,
        dataset_npz=dataset_file,
        test_predictions_path=test_predictions_path,
        infer_config_path=infer_config_file,
        infer_settings=_load_settings_file(infer_config_file),
        replay_manifest_path=replay_manifest_file,
        replay_manifest=_load_json_dict(replay_manifest_file),
    )


def _resolve_postprocess_settings(artifacts: ResolvedArtifacts) -> PostprocessSettings:
    defaults = PostprocessSettings()
    manifest_settings = (
        artifacts.replay_manifest.get("postprocess", {}).get("settings", {})
        if isinstance(artifacts.replay_manifest.get("postprocess"), dict)
        else {}
    )
    merged: dict[str, Any] = {}
    for field in fields(PostprocessSettings):
        value = getattr(defaults, field.name)
        if field.name in artifacts.infer_settings:
            value = artifacts.infer_settings[field.name]
        if field.name in manifest_settings:
            value = manifest_settings[field.name]
        merged[field.name] = value
    return PostprocessSettings(**merged)


def _resolve_runtime_config(artifacts: ResolvedArtifacts) -> ReplayRuntimeConfig:
    defaults = ReplayRuntimeConfig()
    manifest_runtime = artifacts.replay_manifest.get("runtime_config", {})
    merged: dict[str, Any] = {}
    for field in fields(ReplayRuntimeConfig):
        value = getattr(defaults, field.name)
        if field.name in LIVE_INFER_RECIPE_DEFAULTS:
            value = LIVE_INFER_RECIPE_DEFAULTS[field.name]
        if field.name in PSEUDO_LIVE_RECIPE_DEFAULTS:
            value = PSEUDO_LIVE_RECIPE_DEFAULTS[field.name]
        if field.name in artifacts.infer_settings:
            value = artifacts.infer_settings[field.name]
        if field.name in manifest_runtime:
            value = manifest_runtime[field.name]
        merged[field.name] = value
    return ReplayRuntimeConfig(**merged)


def _meta_array(
    meta: dict[str, Any],
    key: str,
    n: int,
    *,
    dtype: str | np.dtype,
    fill_value: Any,
) -> np.ndarray:
    if key not in meta:
        return np.full(n, fill_value, dtype=dtype)
    arr = np.asarray(meta[key])
    if arr.ndim == 0:
        try:
            value = arr.item()
        except Exception:
            value = fill_value
        return np.full(n, value, dtype=dtype)
    arr = arr.reshape(-1)
    if len(arr) != n:
        return np.full(n, fill_value, dtype=dtype)
    try:
        return arr.astype(dtype, copy=False)
    except Exception:
        return np.array([fill_value for _ in range(n)], dtype=dtype)


def _build_current_dataset_info(
    dataset_npz: Path,
    y_action: np.ndarray,
    meta: dict[str, Any],
    train_config: dict[str, Any],
) -> dict[str, Any]:
    experiment_hash = ""
    if "experiment_hash" in meta:
        unique_hashes = _unique_nonempty(np.asarray(meta["experiment_hash"]))
        experiment_hash = unique_hashes[0] if unique_hashes else ""
    subject_id = ""
    if "subject_id" in meta:
        unique_subjects = _unique_nonempty(np.asarray(meta["subject_id"]))
        if len(unique_subjects) > 1 and train_config.get("subject_id_filter") is not None:
            subject_id = str(train_config.get("subject_id_filter") or "")
    return {
        "npz_path": str(dataset_npz),
        "npz_sha256": _sha256_file(dataset_npz),
        "npz_size_bytes": dataset_npz.stat().st_size if dataset_npz.exists() else None,
        "experiment_hash": experiment_hash,
        "n_samples": int(len(y_action)),
        "filters": {
            "subject_id": subject_id,
            "max_samples": None,
        },
    }


def resolve_split_labels(
    *,
    test_predictions_path: Optional[Path],
    dataset_npz: Path,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    meta: dict[str, Any],
    train_config: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    n = int(len(y_action))
    notes: list[str] = []
    split = np.full(n, "unknown", dtype="U16")
    if test_predictions_path is None or not test_predictions_path.exists():
        notes.append("No test_predictions.npz found; split labels marked as unknown.")
        return split, notes

    with np.load(test_predictions_path, allow_pickle=True) as payload:
        test_idx = resolve_cached_test_indices(payload)
        if test_idx is None:
            notes.append("Cached predictions did not include test indices; split labels marked as unknown.")
            return split, notes
        test_idx = np.asarray(test_idx, dtype=np.int64).reshape(-1)
        if test_idx.size == 0:
            split[:] = "train"
            notes.append("Cached test split was empty; all windows marked as train.")
            return split, notes
        if np.any(test_idx < 0) or np.any(test_idx >= n) or len(np.unique(test_idx)) != len(test_idx):
            notes.append("Cached test indices were invalid for the current dataset; split labels marked as unknown.")
            return split, notes

        dataset_info_cache = (
            _parse_dataset_info(payload["dataset_info"])
            if "dataset_info" in payload.files
            else None
        )
        current_info = _build_current_dataset_info(dataset_npz, y_action, meta, train_config)
        cache_ok, reasons = validate_cached_predictions_with_dataset_info(
            action_probs=np.asarray(payload["action_probs"]),
            finger_probs=np.asarray(payload["finger_probs"]),
            y_action_test=np.asarray(payload["y_action"]).astype(np.int64),
            y_finger_test=np.asarray(payload["y_finger"]).astype(np.int64),
            test_idx=test_idx,
            n_actions=int(np.asarray(payload["action_probs"]).shape[1]),
            n_fingers=int(np.asarray(payload["finger_probs"]).shape[1]),
            n_samples_current=n,
            dataset_info_cache=dataset_info_cache,
            dataset_info_current=current_info,
            y_action_current=y_action,
            y_finger_current=y_finger,
        )
        if not cache_ok:
            if np.array_equal(y_action[test_idx], np.asarray(payload["y_action"]).astype(np.int64)) and np.array_equal(
                y_finger[test_idx], np.asarray(payload["y_finger"]).astype(np.int64)
            ):
                notes.append(
                    "Split labels recovered from cached test indices via label alignment "
                    f"(dataset_info mismatch: {', '.join(reasons) or 'unknown'})."
                )
            else:
                notes.append(
                    "Cached test predictions did not validate against the current dataset; "
                    "split labels marked as unknown."
                )
                return split, notes

        split[:] = "train"
        split[test_idx] = "test"
        notes.append(
            f"Recovered split labels from {test_predictions_path.name}: "
            f"{int(np.sum(split == 'train'))} train, {int(np.sum(split == 'test'))} test."
        )
    return split, notes


def build_base_frame(
    *,
    y_action: np.ndarray,
    y_finger: np.ndarray,
    meta: dict[str, Any],
    split_labels: np.ndarray,
) -> pd.DataFrame:
    n = int(len(y_action))
    frame = pd.DataFrame(
        {
            "window_index": np.arange(n, dtype=np.int64),
            "true_action_id": np.asarray(y_action, dtype=np.int64),
            "true_finger_id": np.asarray(y_finger, dtype=np.int64),
            "subject_id": _meta_array(meta, "subject_id", n, dtype="U128", fill_value="UNKNOWN"),
            "session_id": _meta_array(meta, "session_id", n, dtype="U256", fill_value=""),
            "trial_id": _meta_array(meta, "trial_id", n, dtype=np.int64, fill_value=-1),
            "event_id": _meta_array(meta, "event_id", n, dtype=np.int64, fill_value=-1),
            "event_index": _meta_array(meta, "event_index", n, dtype=np.int64, fill_value=-1),
            "block_id": _meta_array(meta, "block_id", n, dtype=np.int64, fill_value=-1),
            "window_start_s": _meta_array(meta, "window_start", n, dtype=np.float32, fill_value=np.nan),
            "window_end_s": _meta_array(meta, "window_end", n, dtype=np.float32, fill_value=np.nan),
            "event_onset_s": _meta_array(meta, "event_onset_s", n, dtype=np.float32, fill_value=np.nan),
            "event_duration_s": _meta_array(meta, "event_duration_s", n, dtype=np.float32, fill_value=np.nan),
            "confidence_hint": _meta_array(meta, "confidence_hint", n, dtype=np.float32, fill_value=np.nan),
            "artifact_flag": _meta_array(meta, "artifact_flag", n, dtype=np.int64, fill_value=0),
            "gap_flag": _meta_array(meta, "gap_flag", n, dtype=np.int64, fill_value=0),
            "gap_fraction": _meta_array(meta, "gap_fraction", n, dtype=np.float32, fill_value=np.nan),
            "overlap_s": _meta_array(meta, "overlap_s", n, dtype=np.float32, fill_value=np.nan),
            "overlap_frac": _meta_array(meta, "overlap_frac", n, dtype=np.float32, fill_value=np.nan),
            "assigned_event_type": _meta_array(meta, "assigned_event_type", n, dtype="U128", fill_value=""),
            "event_source": _meta_array(meta, "event_source", n, dtype="U64", fill_value=""),
            "session_mode": _meta_array(meta, "session_mode", n, dtype="U64", fill_value=""),
            "split": np.asarray(split_labels).astype("U16"),
        }
    )
    frame["window_center_s"] = frame["window_start_s"] + (
        (frame["window_end_s"] - frame["window_start_s"]) / 2.0
    )
    frame["time_offset_s"] = frame["window_start_s"] - frame["event_onset_s"]
    frame["true_action"] = frame["true_action_id"].map(ACTION_NAMES).fillna("UNKNOWN")
    frame["true_finger"] = frame["true_finger_id"].map(FINGER_NAMES).fillna("UNKNOWN")
    frame["trajectory_uid"] = np.where(
        frame["event_id"] >= 0,
        frame["session_id"] + "::event::" + frame["event_id"].astype(str),
        frame["session_id"] + "::trial::" + frame["trial_id"].astype(str),
    )
    return frame


def filter_base_frame(
    frame: pd.DataFrame,
    *,
    split_filter: str,
    subject_filters: Optional[Sequence[str]],
    session_filters: Optional[Sequence[str]],
) -> pd.DataFrame:
    mask = np.ones(len(frame), dtype=bool)
    split_filter = str(split_filter or "all").strip().lower()
    if split_filter != "all":
        split_values = frame["split"].astype(str).to_numpy(dtype="U")
        if not np.any(split_values == split_filter):
            raise ValueError(
                f"Requested --split-filter={split_filter}, but that split is not available."
            )
        mask &= split_values == split_filter
    if subject_filters:
        subjects = np.asarray([str(value) for value in subject_filters], dtype="U")
        mask &= np.isin(frame["subject_id"].astype(str).to_numpy(dtype="U"), subjects)
    if session_filters:
        sessions = np.asarray([str(value) for value in session_filters], dtype="U")
        mask &= np.isin(frame["session_id"].astype(str).to_numpy(dtype="U"), sessions)
    filtered = frame.loc[mask].copy()
    if filtered.empty:
        raise ValueError("Filters removed all rows; nothing left to visualize.")
    filtered.sort_values(
        by=["subject_id", "session_id", "trial_id", "window_start_s", "window_index"],
        inplace=True,
        kind="stable",
    )
    filtered.reset_index(drop=True, inplace=True)
    filtered["window_rank_in_event"] = (
        filtered.groupby("trajectory_uid", sort=False).cumcount().astype(np.int64)
    )
    return filtered


def _build_model(
    *,
    model_path: Path,
    train_config: dict[str, Any],
    n_channels: int,
    window_samples: int,
    device: torch.device,
) -> torch.nn.Module:
    state_dict = torch.load(str(model_path), map_location="cpu", weights_only=True)
    n_fingers, n_actions, has_applicability_head = infer_output_dims_from_state_dict(
        state_dict
    )
    model_name = str(train_config.get("model", "")).strip()
    if not model_name:
        model_name = "CNNLSTMFingerActionNet" if any(
            key.startswith("lstm.") for key in state_dict
        ) else "FingerActionNet"

    if model_name == "FingerActionNet":
        model: torch.nn.Module = FingerActionNet(
            n_channels=int(n_channels),
            window_samples=int(window_samples),
            n_fingers=int(n_fingers),
            n_actions=int(n_actions),
            finger_applicability_head=bool(has_applicability_head),
        )
    else:
        model = CNNLSTMFingerActionNet(
            n_channels=int(n_channels),
            n_fingers=int(n_fingers),
            n_actions=int(n_actions),
            finger_applicability_head=bool(has_applicability_head),
        )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_model_inference(
    *,
    X: np.ndarray,
    indices: np.ndarray,
    artifacts: ResolvedArtifacts,
    embedding_source: str,
    device: torch.device,
    batch_size: int,
) -> InferenceOutputs:
    embedding_source = str(embedding_source).strip().lower()
    if embedding_source not in SUPPORTED_EMBEDDING_SOURCES:
        raise ValueError(
            f"Unsupported embedding source: {embedding_source}. "
            f"Expected one of {SUPPORTED_EMBEDDING_SOURCES}."
        )

    normalizer = load_normalizer(artifacts.scaler_path)
    if normalizer is None:
        raise RuntimeError(f"Failed to load normalizer from {artifacts.scaler_path}")
    temperature_state = (
        load_temperature_scaling(artifacts.temperature_path)
        if artifacts.temperature_path is not None
        else None
    )
    n_channels = int(X.shape[-1])
    window_samples = int(X.shape[1])
    model = _build_model(
        model_path=artifacts.model_path,
        train_config=artifacts.train_config,
        n_channels=n_channels,
        window_samples=window_samples,
        device=device,
    )

    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    n = int(len(indices))
    n_actions = int(model.action_head.out_features)
    n_fingers = int(model.finger_head.out_features)
    action_logits = np.zeros((n, n_actions), dtype=np.float32)
    finger_logits = np.zeros((n, n_fingers), dtype=np.float32)
    action_probs = np.zeros((n, n_actions), dtype=np.float32)
    finger_probs = np.zeros((n, n_fingers), dtype=np.float32)
    applicability_logits = (
        np.zeros(n, dtype=np.float32)
        if getattr(model, "finger_applicability_head", None) is not None
        else None
    )
    applicability_probs = (
        np.zeros(n, dtype=np.float32)
        if getattr(model, "finger_applicability_head", None) is not None
        else None
    )
    representation_batches: list[np.ndarray] = []

    action_temperature = (
        float(temperature_state.action_temperature)
        if temperature_state is not None
        else 1.0
    )
    finger_temperature = (
        float(temperature_state.finger_temperature)
        if temperature_state is not None
        else 1.0
    )
    applicability_temperature = (
        float(temperature_state.applicability_temperature)
        if temperature_state is not None
        else 1.0
    )

    batch_cap = min(max(1, int(batch_size)), max(1, n))
    host_batch = np.empty((batch_cap,) + tuple(X.shape[1:]), dtype=np.float32)
    host_tensor = torch.from_numpy(host_batch)
    device_batch = (
        torch.empty(host_batch.shape, dtype=torch.float32, device=device)
        if device.type != "cpu"
        else None
    )

    with torch.inference_mode():
        for start in range(0, n, batch_cap):
            end = min(start + batch_cap, n)
            batch_idx = indices[start:end]
            current_len = end - start
            host_view = host_batch[:current_len]
            np.copyto(host_view, np.asarray(X[batch_idx], dtype=np.float32), casting="unsafe")
            apply_channel_normalizer(host_view, normalizer, out=host_view)

            if device_batch is not None:
                batch_tensor = device_batch[:current_len]
                batch_tensor.copy_(host_tensor[:current_len])
            else:
                batch_tensor = host_tensor[:current_len]

            if not hasattr(model, "extract_features") or not hasattr(model, "forward_heads"):
                raise RuntimeError(
                    "Model does not expose extract_features/forward_heads; latent extraction is unsupported."
                )

            features_t = model.extract_features(batch_tensor)
            finger_logits_t, action_logits_t, applicability_logits_t = model.forward_heads(
                features_t
            )
            action_logits_t = apply_temperature_to_logits(action_logits_t, action_temperature)
            finger_logits_t = apply_temperature_to_logits(finger_logits_t, finger_temperature)
            if applicability_logits_t is not None:
                applicability_logits_t = apply_temperature_to_logits(
                    applicability_logits_t,
                    applicability_temperature,
                )

            action_prob_t = torch.softmax(action_logits_t, dim=1)
            finger_prob_t = torch.softmax(finger_logits_t, dim=1)
            applicability_prob_t = (
                torch.sigmoid(applicability_logits_t)
                if applicability_logits_t is not None
                else None
            )

            action_logits[start:end] = action_logits_t.float().cpu().numpy()
            finger_logits[start:end] = finger_logits_t.float().cpu().numpy()
            action_probs[start:end] = action_prob_t.float().cpu().numpy()
            finger_probs[start:end] = finger_prob_t.float().cpu().numpy()

            if applicability_logits is not None and applicability_logits_t is not None:
                applicability_logits[start:end] = (
                    applicability_logits_t.float().cpu().numpy().reshape(-1)
                )
                assert applicability_probs is not None
                applicability_probs[start:end] = (
                    applicability_prob_t.float().cpu().numpy().reshape(-1)
                )

            if embedding_source == "latent":
                representation_batches.append(features_t.float().cpu().numpy())
            elif embedding_source == "logits":
                rep = np.concatenate(
                    [
                        finger_logits[start:end],
                        action_logits[start:end],
                        (
                            applicability_logits[start:end].reshape(-1, 1)
                            if applicability_logits is not None
                            else np.zeros((current_len, 0), dtype=np.float32)
                        ),
                    ],
                    axis=1,
                )
                representation_batches.append(rep)
            elif embedding_source == "probabilities":
                rep = np.concatenate(
                    [
                        finger_probs[start:end],
                        action_probs[start:end],
                        (
                            applicability_probs[start:end].reshape(-1, 1)
                            if applicability_probs is not None
                            else np.zeros((current_len, 0), dtype=np.float32)
                        ),
                    ],
                    axis=1,
                )
                representation_batches.append(rep)
            else:
                representation_batches.append(host_view.reshape(current_len, -1).copy())

    representation = np.concatenate(representation_batches, axis=0).astype(np.float32)
    return InferenceOutputs(
        representation=representation,
        action_logits=action_logits,
        finger_logits=finger_logits,
        applicability_logits=applicability_logits,
        action_probs=action_probs,
        finger_probs=finger_probs,
        applicability_probs=applicability_probs,
    )


def _classify_correctness(
    pred_action_id: int,
    pred_finger_id: int,
    true_action_id: int,
    true_finger_id: int,
) -> str:
    action_ok = int(pred_action_id) == int(true_action_id)
    finger_ok = int(pred_finger_id) == int(true_finger_id)
    if action_ok and finger_ok:
        return "correct"
    if not action_ok and not finger_ok:
        return "action+finger_wrong"
    if not action_ok:
        return "action_wrong"
    return "finger_wrong"


def assemble_prediction_frame(
    *,
    base_frame: pd.DataFrame,
    outputs: InferenceOutputs,
    postprocess_settings: PostprocessSettings,
    runtime_config: ReplayRuntimeConfig,
) -> pd.DataFrame:
    frame = base_frame.copy()
    n = len(frame)
    action_probs = np.asarray(outputs.action_probs, dtype=np.float32)
    finger_probs = np.asarray(outputs.finger_probs, dtype=np.float32)
    applicability_probs = (
        np.asarray(outputs.applicability_probs, dtype=np.float32)
        if outputs.applicability_probs is not None
        else None
    )

    raw_action_ids = np.argmax(action_probs, axis=1).astype(np.int64)
    raw_top_finger_ids = decode_finger_predictions(finger_probs)
    pred_finger_ids = decode_finger_predictions_for_actions(raw_action_ids, finger_probs)
    row_idx = np.arange(n, dtype=np.int64)
    raw_action_confidence = action_probs[row_idx, raw_action_ids]
    raw_finger_confidence = finger_confidences_for_ids(finger_probs, pred_finger_ids)

    post_state = PostprocessState()
    actuation_history: collections.deque[ActuationDecision] = collections.deque(
        maxlen=max(3, int(runtime_config.actuation_stability))
    )
    speed_mapper = build_actuation_speed_mapper(
        modulate_actuation_speed=bool(runtime_config.modulate_actuation_speed),
        actuation_speed_gamma=float(runtime_config.actuation_speed_gamma),
    )
    command_shaper = build_actuation_command_shaper(
        actuation_min_prob=float(runtime_config.actuation_min_prob),
        actuation_speed_gamma=float(runtime_config.actuation_speed_gamma),
        hop_sec=float(runtime_config.hop_sec),
        actuation_stability=int(runtime_config.actuation_stability),
        actuation_cooldown_ms=int(runtime_config.actuation_cooldown_ms),
    )

    committed_action = np.zeros(n, dtype=np.int64)
    committed_finger = np.zeros(n, dtype=np.int64)
    committed_action_conf = np.zeros(n, dtype=np.float32)
    committed_finger_conf = np.zeros(n, dtype=np.float32)
    smoothed_action = np.zeros(n, dtype=np.int64)
    smoothed_finger = np.zeros(n, dtype=np.int64)
    finger_gate_ok = np.ones(n, dtype=bool)
    applicability_gate_ok = np.ones(n, dtype=bool)
    uncertainty_gate_ok = np.ones(n, dtype=bool)
    committed_pair_valid = np.ones(n, dtype=bool)
    deployment_gate_ok = np.zeros(n, dtype=bool)
    predicted_applicable = np.zeros(n, dtype=bool)
    actuation_sent = np.zeros(n, dtype=bool)
    actuation_target_action = np.zeros(n, dtype=np.int64)
    actuation_target_finger = np.zeros(n, dtype=np.int64)
    joint_conf = np.zeros(n, dtype=np.float32)
    actuation_speed_scalar = np.zeros(n, dtype=np.float32)
    decision_reason: list[str] = []
    actuation_vote_reason: list[str] = []
    actuation_suppressed_reason: list[str] = []

    last_sent: Optional[tuple[int, int]] = None
    last_send_time_ms: Optional[float] = None
    last_trial: Optional[int] = None
    latency_mode = str(runtime_config.latency_mode or "ignore").strip().lower()

    for idx in range(n):
        trial_id = int(frame.at[idx, "trial_id"])
        if bool(runtime_config.reset_on_trial_change):
            if last_trial is None:
                last_trial = trial_id
            elif trial_id != last_trial:
                post_state.reset()
                actuation_history.clear()
                command_shaper.reset()
                last_sent = None
                last_send_time_ms = None
                last_trial = trial_id

        applicability_score = (
            float(applicability_probs[idx]) if applicability_probs is not None else None
        )
        decision_info = postprocess_predictions(
            action_probs[idx],
            finger_probs[idx],
            postprocess_settings,
            post_state,
            finger_applicable_prob=applicability_score,
        )
        committed_action[idx] = int(decision_info["committed_action_id"])
        committed_finger[idx] = int(decision_info["committed_finger_id"])
        committed_action_conf[idx] = float(decision_info.get("action_conf", 0.0))
        committed_finger_conf[idx] = float(decision_info.get("finger_conf", 0.0))
        smoothed_action[idx] = int(decision_info.get("smoothed_action_id", 0))
        smoothed_finger[idx] = int(decision_info.get("smoothed_finger_id", 0))
        finger_gate_ok[idx] = bool(decision_info.get("finger_gate_ok", True))
        applicability_gate_ok[idx] = bool(
            decision_info.get("applicability_gate_ok", True)
        )
        committed_pair_valid[idx] = bool(
            decision_info.get("committed_pair_valid", True)
        )
        decision_reason.append(str(decision_info.get("decision_reason", "")))

        decision = ActuationDecision(
            finger_id=int(committed_finger[idx]),
            action_id=int(committed_action[idx]),
            prob=float(
                min(
                    float(decision_info.get("action_conf", 0.0)),
                    float(decision_info.get("finger_conf", 0.0)),
                )
            ),
        )
        joint_conf[idx] = float(decision.prob)
        uncertainty_gate_ok[idx] = bool(
            uncertainty_gate_passed(decision_info, {"adaptive_threshold": None})
        )
        actuation_speed_scalar[idx] = float(
            compute_actuation_speed_scalar(
                decision.prob,
                0.0,
                speed_mapper,
                min_speed=float(runtime_config.actuation_min_speed),
            )
        )

        actuation_history.append(decision)
        actuation_vote = resolve_actuation_candidate(
            actuation_history,
            required_finger_stability=int(runtime_config.actuation_stability),
        )
        voted_decision = actuation_vote["decision"]
        actuation_vote_reason.append(str(actuation_vote.get("reason", "")))

        target_finger_id = int(voted_decision.finger_id)
        target_action_id = int(voted_decision.action_id)
        suppressed_reason = ""
        current_time_ms = float(frame.at[idx, "window_center_s"]) * 1000.0
        latency_ok = latency_mode == "ignore"
        if latency_mode != "ignore":
            latency_ok = float(runtime_config.window_sec) * 500.0 <= float(
                runtime_config.latency_threshold_ms
            )
        if not latency_ok:
            suppressed_reason = "latency_gate"
        elif not applicability_gate_ok[idx]:
            suppressed_reason = "applicability_gate"
        elif not finger_gate_ok[idx]:
            suppressed_reason = "finger_gate"
        elif is_noop_decision(voted_decision.finger_id, voted_decision.action_id):
            suppressed_reason = str(actuation_vote.get("reason", "noop"))
        elif not uncertainty_gate_ok[idx]:
            suppressed_reason = "uncertainty_gate"
        else:
            shaped = command_shaper.shape(
                action_id=int(voted_decision.action_id),
                finger_id=int(voted_decision.finger_id),
                action_conf=float(voted_decision.prob),
                speed_scalar_override=float(actuation_speed_scalar[idx]),
                timestamp_stream_ms=int(round(current_time_ms)),
                stability_ok=True,
                timebase_ms=int(round(current_time_ms)),
            )
            target_finger_id = int(shaped.finger_id)
            target_action_id = int(shaped.action_id)
            actuation_speed_scalar[idx] = float(shaped.speed_scalar)
            send_decision = ActuationDecision(
                finger_id=target_finger_id,
                action_id=target_action_id,
                prob=float(voted_decision.prob),
            )
            if is_noop_decision(send_decision.finger_id, send_decision.action_id):
                suppressed_reason = "min_prob"
            elif debounced_should_send(
                send_decision,
                last_sent=last_sent,
                stable_count=1,
                required_stability=1,
                last_send_time_ms=last_send_time_ms,
                current_time_ms=current_time_ms,
                cooldown_ms=int(runtime_config.actuation_cooldown_ms),
                repeat_same_ms=int(runtime_config.actuation_repeat_ms),
            ):
                actuation_sent[idx] = True
                last_sent = (int(send_decision.finger_id), int(send_decision.action_id))
                last_send_time_ms = current_time_ms
            else:
                suppressed_reason = "cooldown_or_duplicate"

        actuation_target_action[idx] = target_action_id
        actuation_target_finger[idx] = target_finger_id
        actuation_suppressed_reason.append(suppressed_reason)
        predicted_applicable[idx] = bool(
            applicability_score is not None
            and float(applicability_score) >= float(postprocess_settings.threshold_applicability)
        )
        deployment_gate_ok[idx] = bool(
            committed_pair_valid[idx]
            and uncertainty_gate_ok[idx]
            and (
                committed_action[idx] == int(ACTION_REST)
                or (finger_gate_ok[idx] and applicability_gate_ok[idx])
            )
        )

    frame["raw_top_action_id"] = raw_action_ids
    frame["raw_top_finger_id"] = raw_top_finger_ids
    frame["pred_action_id"] = raw_action_ids
    frame["pred_finger_id"] = pred_finger_ids
    frame["raw_action_confidence"] = raw_action_confidence.astype(np.float32)
    frame["raw_finger_confidence"] = raw_finger_confidence.astype(np.float32)
    frame["action_confidence"] = raw_action_confidence.astype(np.float32)
    frame["finger_confidence"] = raw_finger_confidence.astype(np.float32)
    frame["applicability_score"] = (
        applicability_probs.astype(np.float32)
        if applicability_probs is not None
        else np.full(n, np.nan, dtype=np.float32)
    )
    frame["predicted_applicable"] = predicted_applicable
    frame["smoothed_action_id"] = smoothed_action
    frame["smoothed_finger_id"] = smoothed_finger
    frame["committed_action_id"] = committed_action
    frame["committed_finger_id"] = committed_finger
    frame["committed_action_confidence"] = committed_action_conf
    frame["committed_finger_confidence"] = committed_finger_conf
    frame["finger_gate_ok"] = finger_gate_ok
    frame["applicability_gate_ok"] = applicability_gate_ok
    frame["uncertainty_gate_ok"] = uncertainty_gate_ok
    frame["committed_pair_valid"] = committed_pair_valid
    frame["deployment_gate_ok"] = deployment_gate_ok
    frame["joint_confidence"] = joint_conf
    frame["decision_reason"] = decision_reason
    frame["actuation_vote_reason"] = actuation_vote_reason
    frame["actuation_suppressed_reason"] = actuation_suppressed_reason
    frame["actuation_sent"] = actuation_sent
    frame["actuation_target_action_id"] = actuation_target_action
    frame["actuation_target_finger_id"] = actuation_target_finger
    frame["actuation_speed_scalar"] = actuation_speed_scalar.astype(np.float32)

    frame["raw_top_action"] = frame["raw_top_action_id"].map(ACTION_NAMES).fillna("UNKNOWN")
    frame["raw_top_finger"] = frame["raw_top_finger_id"].map(FINGER_NAMES).fillna("UNKNOWN")
    frame["pred_action"] = frame["pred_action_id"].map(ACTION_NAMES).fillna("UNKNOWN")
    frame["pred_finger"] = frame["pred_finger_id"].map(FINGER_NAMES).fillna("UNKNOWN")
    frame["committed_action"] = frame["committed_action_id"].map(ACTION_NAMES).fillna("UNKNOWN")
    frame["committed_finger"] = frame["committed_finger_id"].map(FINGER_NAMES).fillna("UNKNOWN")
    frame["actuation_target_action"] = (
        frame["actuation_target_action_id"].map(ACTION_NAMES).fillna("UNKNOWN")
    )
    frame["actuation_target_finger"] = (
        frame["actuation_target_finger_id"].map(FINGER_NAMES).fillna("UNKNOWN")
    )

    frame["pred_action_correct"] = frame["pred_action_id"] == frame["true_action_id"]
    frame["pred_finger_correct"] = frame["pred_finger_id"] == frame["true_finger_id"]
    frame["pred_joint_correct"] = (
        frame["pred_action_correct"] & frame["pred_finger_correct"]
    )
    frame["committed_action_correct"] = (
        frame["committed_action_id"] == frame["true_action_id"]
    )
    frame["committed_finger_correct"] = (
        frame["committed_finger_id"] == frame["true_finger_id"]
    )
    frame["committed_joint_correct"] = (
        frame["committed_action_correct"] & frame["committed_finger_correct"]
    )
    frame["correctness"] = [
        _classify_correctness(
            pred_action_id=int(frame.at[idx, "pred_action_id"]),
            pred_finger_id=int(frame.at[idx, "pred_finger_id"]),
            true_action_id=int(frame.at[idx, "true_action_id"]),
            true_finger_id=int(frame.at[idx, "true_finger_id"]),
        )
        for idx in range(n)
    ]
    frame["deployment_pair_valid"] = [
        bool(
            is_valid_action_finger(
                int(frame.at[idx, "committed_action_id"]),
                int(frame.at[idx, "committed_finger_id"]),
            )
        )
        for idx in range(n)
    ]
    return frame


def reduce_to_3d(
    representation: np.ndarray,
    *,
    reducer: str,
    seed: int,
    umap_n_neighbors: int = 25,
    umap_min_dist: float = 0.1,
) -> np.ndarray:
    representation = np.asarray(representation, dtype=np.float32)
    if representation.ndim != 2:
        raise ValueError(
            f"Representation must be 2-D for reduction, got shape {representation.shape}."
        )
    n_samples, n_features = representation.shape
    if n_samples == 0:
        raise ValueError("No rows available for reduction.")
    if n_samples == 1:
        return np.zeros((1, 3), dtype=np.float32)
    if reducer == "umap" and n_samples >= 4:
        try:
            import umap.umap_ as umap
        except Exception as exc:
            raise ImportError(
                "UMAP requested but umap-learn is not installed. Install `umap-learn` "
                "or use `--reducer pca`."
            ) from exc
        n_neighbors = max(2, min(int(umap_n_neighbors), n_samples - 1))
        reducer_obj = umap.UMAP(
            n_components=3,
            n_neighbors=n_neighbors,
            min_dist=float(umap_min_dist),
            random_state=int(seed),
        )
        coords = reducer_obj.fit_transform(representation)
        return np.asarray(coords, dtype=np.float32)

    n_components = min(3, n_samples, n_features)
    coords = PCA(n_components=n_components, random_state=int(seed)).fit_transform(
        representation
    )
    coords = np.asarray(coords, dtype=np.float32)
    if coords.shape[1] < 3:
        padded = np.zeros((coords.shape[0], 3), dtype=np.float32)
        padded[:, : coords.shape[1]] = coords
        return padded
    return coords


def select_sample_positions(
    frame: pd.DataFrame,
    *,
    max_points: Optional[int],
    sample_strategy: str,
    seed: int,
) -> tuple[np.ndarray, Optional[str]]:
    n = len(frame)
    if max_points is None or max_points <= 0 or n <= int(max_points):
        return np.arange(n, dtype=np.int64), None

    sample_strategy = str(sample_strategy).strip().lower()
    if sample_strategy not in SUPPORTED_SAMPLE_STRATEGIES:
        raise ValueError(
            f"Unsupported sample strategy: {sample_strategy}. "
            f"Expected one of {SUPPORTED_SAMPLE_STRATEGIES}."
        )
    rng = np.random.default_rng(int(seed))
    all_idx = np.arange(n, dtype=np.int64)
    kept: np.ndarray
    detail = ""
    if sample_strategy == "stratified_joint":
        labels = (
            frame["true_action"].astype(str) + "|" + frame["true_finger"].astype(str)
        ).to_numpy()
        try:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=int(max_points),
                random_state=int(seed),
            )
            kept, _ = next(splitter.split(all_idx, labels))
            detail = "stratified_joint"
        except ValueError:
            kept = rng.choice(all_idx, size=int(max_points), replace=False)
            detail = "random_fallback"
    else:
        kept = rng.choice(all_idx, size=int(max_points), replace=False)
        detail = "random"
    kept = np.sort(kept.astype(np.int64))
    note = f"Sampled {len(kept)} of {n} points using {detail} (seed={int(seed)})."
    return kept, note


def _format_optional(value: Any, *, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if bool(value) else "no"
    return str(value)


def build_hover_text(frame: pd.DataFrame) -> np.ndarray:
    rows: list[str] = []
    for _, row in frame.iterrows():
        window_rank_in_event = row.get("window_rank_in_event", -1)
        lines = [
            f"window={int(row['window_index'])} split={row['split']}",
            (
                f"subject={row['subject_id']} session={row['session_id']} "
                f"trial={int(row['trial_id'])} event={int(row['event_id'])}"
            ),
            (
                f"window_rank_in_event={int(window_rank_in_event)} "
                f"time_offset_s={_format_optional(row['time_offset_s'])}"
            ),
            (
                f"true={row['true_action']} / {row['true_finger']} "
                f"pred={row['pred_action']} / {row['pred_finger']}"
            ),
            (
                f"committed={row['committed_action']} / {row['committed_finger']} "
                f"target={row['actuation_target_action']} / {row['actuation_target_finger']}"
            ),
            (
                f"action_conf={_format_optional(row['action_confidence'])} "
                f"finger_conf={_format_optional(row['finger_confidence'])} "
                f"applicability={_format_optional(row['applicability_score'])}"
            ),
            (
                f"pred_applicable={_format_optional(row['predicted_applicable'])} "
                f"gates=finger:{_format_optional(row['finger_gate_ok'])}, "
                f"app:{_format_optional(row['applicability_gate_ok'])}, "
                f"uncertainty:{_format_optional(row['uncertainty_gate_ok'])}"
            ),
            (
                f"deploy_gate_ok={_format_optional(row['deployment_gate_ok'])} "
                f"actuation_sent={_format_optional(row['actuation_sent'])} "
                f"suppressed={row['actuation_suppressed_reason'] or 'none'}"
            ),
            (
                f"pred_joint_correct={_format_optional(row['pred_joint_correct'])} "
                f"committed_joint_correct={_format_optional(row['committed_joint_correct'])} "
                f"correctness={row['correctness']}"
            ),
        ]
        rows.append("<br>".join(lines))
    return np.asarray(rows, dtype="U")


def build_plot_figure(
    frame: pd.DataFrame,
    *,
    color_by: str,
    connect_trajectories: bool,
    title: str,
):
    color_by = str(color_by).strip()
    if color_by not in SUPPORTED_COLOR_MODES:
        raise ValueError(
            f"Unsupported color mode: {color_by}. Expected one of {SUPPORTED_COLOR_MODES}."
        )
    if color_by not in frame.columns:
        raise ValueError(f"Color column not found in frame: {color_by}")

    import plotly.express as px
    import plotly.graph_objects as go

    plot_frame = frame.copy()
    plot_frame["hover_text"] = build_hover_text(plot_frame)

    category_orders = {
        "true_action": [ACTION_NAMES[idx] for idx in sorted(ACTION_NAMES)],
        "pred_action": [ACTION_NAMES[idx] for idx in sorted(ACTION_NAMES)],
        "true_finger": [FINGER_NAMES[idx] for idx in sorted(FINGER_NAMES)],
        "pred_finger": [FINGER_NAMES[idx] for idx in sorted(FINGER_NAMES)],
        "split": ["train", "test", "unknown"],
        "correctness": [
            "correct",
            "finger_wrong",
            "action_wrong",
            "action+finger_wrong",
        ],
    }
    color_is_categorical = color_by in _CATEGORICAL_COLOR_MODES
    fig = px.scatter_3d(
        plot_frame,
        x="emb_x",
        y="emb_y",
        z="emb_z",
        color=color_by,
        category_orders=category_orders,
        custom_data=["hover_text"],
        color_continuous_scale="Viridis",
        opacity=0.82,
        title=title,
    )
    for trace in fig.data:
        trace.update(
            marker={"size": 4.5, "opacity": 0.82},
            hovertemplate="%{customdata[0]}<extra></extra>",
        )
        if color_is_categorical:
            trace.update(legendgroup=trace.name)

    if connect_trajectories:
        traj_x: list[float | None] = []
        traj_y: list[float | None] = []
        traj_z: list[float | None] = []
        for _, group in plot_frame.groupby("trajectory_uid", sort=False):
            if len(group) < 2:
                continue
            ordered = group.sort_values(
                by=["window_start_s", "window_index"], kind="stable"
            )
            traj_x.extend(ordered["emb_x"].tolist())
            traj_x.append(None)
            traj_y.extend(ordered["emb_y"].tolist())
            traj_y.append(None)
            traj_z.extend(ordered["emb_z"].tolist())
            traj_z.append(None)
        if traj_x:
            fig.add_trace(
                go.Scatter3d(
                    x=traj_x,
                    y=traj_y,
                    z=traj_z,
                    mode="lines",
                    line={"color": "rgba(70,70,70,0.25)", "width": 2},
                    name="trajectories",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        legend={"itemsizing": "constant"},
        scene={
            "xaxis_title": "Dim 1",
            "yaxis_title": "Dim 2",
            "zaxis_title": "Dim 3",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 60},
    )
    return fig


def export_frame_to_npz(
    path: Path,
    frame: pd.DataFrame,
    *,
    embedding_source: str,
    reducer: str,
) -> None:
    path = _safe_resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "embedding_source": np.array([str(embedding_source)], dtype="U32"),
        "reducer": np.array([str(reducer)], dtype="U16"),
    }
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            payload[column] = series.to_numpy(dtype=np.int8)
        elif pd.api.types.is_numeric_dtype(series):
            payload[column] = series.to_numpy()
        else:
            payload[column] = series.astype(str).to_numpy(dtype="U")
    np.savez(path, **payload)


def prepare_event_space_dataframe(
    *,
    artifacts: ResolvedArtifacts,
    embedding_source: str,
    reducer: str,
    max_points: Optional[int],
    sample_strategy: str,
    seed: int,
    split_filter: str,
    subject_filters: Optional[Sequence[str]],
    session_filters: Optional[Sequence[str]],
    device_name: str,
    batch_size: int,
    umap_n_neighbors: int,
    umap_min_dist: float,
) -> tuple[pd.DataFrame, dict[str, Any], list[str]]:
    X, y_action, y_finger, meta = load_sequence_npz(artifacts.dataset_npz, mmap_mode="r")
    split_labels, notes = resolve_split_labels(
        test_predictions_path=artifacts.test_predictions_path,
        dataset_npz=artifacts.dataset_npz,
        y_action=y_action,
        y_finger=y_finger,
        meta=meta,
        train_config=artifacts.train_config,
    )
    base_frame = build_base_frame(
        y_action=y_action,
        y_finger=y_finger,
        meta=meta,
        split_labels=split_labels,
    )
    filtered = filter_base_frame(
        base_frame,
        split_filter=split_filter,
        subject_filters=subject_filters,
        session_filters=session_filters,
    )
    device = resolve_device(device_name)
    outputs = run_model_inference(
        X=X,
        indices=filtered["window_index"].to_numpy(dtype=np.int64),
        artifacts=artifacts,
        embedding_source=embedding_source,
        device=device,
        batch_size=batch_size,
    )
    postprocess_settings = _resolve_postprocess_settings(artifacts)
    runtime_config = _resolve_runtime_config(artifacts)
    enriched = assemble_prediction_frame(
        base_frame=filtered,
        outputs=outputs,
        postprocess_settings=postprocess_settings,
        runtime_config=runtime_config,
    )
    coords = reduce_to_3d(
        outputs.representation,
        reducer=reducer,
        seed=seed,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
    )
    enriched["emb_x"] = coords[:, 0]
    enriched["emb_y"] = coords[:, 1]
    enriched["emb_z"] = coords[:, 2]

    sample_idx, sampling_note = select_sample_positions(
        enriched,
        max_points=max_points,
        sample_strategy=sample_strategy,
        seed=seed,
    )
    if sampling_note:
        notes.append(sampling_note)
    sampled = enriched.iloc[sample_idx].copy().reset_index(drop=True)
    summary = {
        "artifact_paths": {
            "dataset_npz": str(artifacts.dataset_npz),
            "session_dir": str(artifacts.session_dir),
            "run_dir": str(artifacts.run_dir),
            "model_path": str(artifacts.model_path),
            "scaler_path": str(artifacts.scaler_path),
            "temperature_path": (
                str(artifacts.temperature_path)
                if artifacts.temperature_path is not None
                else None
            ),
        },
        "counts": {
            "dataset_rows": int(len(base_frame)),
            "filtered_rows": int(len(enriched)),
            "display_rows": int(len(sampled)),
        },
        "embedding_source": str(embedding_source),
        "reducer": str(reducer),
        "device": str(device),
        "postprocess_settings": asdict(postprocess_settings),
        "runtime_config": asdict(runtime_config),
    }
    return sampled, summary, notes
