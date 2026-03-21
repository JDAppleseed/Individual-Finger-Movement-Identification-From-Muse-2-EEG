from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _ensure_cwd_exists() -> None:
    try:
        os.getcwd()
    except FileNotFoundError:
        try:
            os.chdir(Path(__file__).resolve().parents[1])
        except Exception:
            pass


_ensure_cwd_exists()

import torch

logger = logging.getLogger(__name__)

NORMALIZER_VERSION = 2


def normalize_preprocess_config(preprocess: Any) -> dict:
    if preprocess is None:
        preprocess = {}
    if not isinstance(preprocess, dict):
        raise ValueError("preprocess config must be a dict when provided.")
    return {
        "per_window_center": bool(preprocess.get("per_window_center", False)),
        "per_window_detrend": bool(preprocess.get("per_window_detrend", False)),
    }


def _apply_preprocess_inplace(arr: np.ndarray, preprocess: Any) -> np.ndarray:
    cfg = normalize_preprocess_config(preprocess)
    if not cfg["per_window_center"] and not cfg["per_window_detrend"]:
        return arr

    view = arr if arr.ndim == 3 else arr[None, ...]
    if view.ndim != 3:
        raise ValueError(f"Expected 2D or 3D EEG array, got shape {arr.shape}")

    if cfg["per_window_detrend"]:
        view -= view.mean(axis=1, keepdims=True)
        n_time = int(view.shape[1])
        if n_time > 1:
            x = np.arange(n_time, dtype=np.float32)
            x -= x.mean()
            denom = float(np.sum(x * x))
            if denom > 0.0:
                slopes = np.sum(view * x[None, :, None], axis=1, keepdims=True) / denom
                view -= slopes * x[None, :, None]
        return arr

    view -= view.mean(axis=1, keepdims=True)
    return arr


def preprocess_eeg_windows(window_TxC: np.ndarray, preprocess: Any) -> np.ndarray:
    arr = np.array(window_TxC, dtype=np.float32, copy=True)
    return _apply_preprocess_inplace(arr, preprocess)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_device(requested: str) -> torch.device:
    requested = str(requested or "auto").strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if sys.platform == "darwin" and getattr(torch.backends, "mps", None) is not None:
            return torch.device("cpu")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_payload(normalizer: Any) -> Optional[dict]:
    if not isinstance(normalizer, dict):
        return None
    if "mean" not in normalizer or "std" not in normalizer:
        return None
    mean = np.asarray(normalizer["mean"], dtype=np.float32).reshape(-1)
    std = np.asarray(normalizer["std"], dtype=np.float32).reshape(-1)
    if mean.shape != std.shape:
        raise ValueError(f"Normalizer mean/std shape mismatch: {mean.shape} vs {std.shape}")
    channels = int(normalizer.get("channels", mean.shape[0]))
    norm_type = str(normalizer.get("type", "per_channel"))
    preprocess = normalize_preprocess_config(normalizer.get("preprocess"))
    return {
        "version": NORMALIZER_VERSION,
        "type": norm_type,
        "mean": mean,
        "std": std,
        "channels": channels,
        "preprocess": preprocess,
    }


def save_normalizer(path: Path, normalizer: Any) -> None:
    path = Path(path)
    payload = _normalize_payload(normalizer)
    if payload is None:
        raise ValueError("Unsupported normalizer payload; expected dict with mean/std.")
    if path.suffix.lower() != ".npz":
        raise ValueError(f"Unsupported normalizer extension: {path.suffix}. Use .npz.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        version=np.array(payload["version"], dtype=np.int32),
        type=np.array(payload["type"], dtype=str),
        mean=payload["mean"],
        std=payload["std"],
        channels=np.array(payload["channels"], dtype=np.int32),
        preprocess_center=np.array(payload["preprocess"]["per_window_center"], dtype=np.int8),
        preprocess_detrend=np.array(payload["preprocess"]["per_window_detrend"], dtype=np.int8),
    )


def load_normalizer(path: Path) -> Optional[Any]:
    path = Path(path)
    if not path.exists():
        return None
    suffix = path.suffix.lower()
    if suffix != ".npz":
        logger.warning("Unsupported normalizer extension: %s", path.suffix)
        return None
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "mean" not in npz or "std" not in npz:
                raise ValueError("Normalizer NPZ missing mean/std.")
            mean = np.asarray(npz["mean"], dtype=np.float32)
            std = np.asarray(npz["std"], dtype=np.float32)
            if mean.shape != std.shape:
                raise ValueError(f"Normalizer mean/std shape mismatch: {mean.shape} vs {std.shape}")
            channels = int(npz["channels"]) if "channels" in npz else int(mean.shape[-1])
            preprocess = {
                "per_window_center": bool(
                    int(npz["preprocess_center"]) if "preprocess_center" in npz else 0
                ),
                "per_window_detrend": bool(
                    int(npz["preprocess_detrend"]) if "preprocess_detrend" in npz else 0
                ),
            }
            return {
                "type": "per_channel",
                "mean": mean,
                "std": std,
                "channels": channels,
                "preprocess": preprocess,
            }
    except Exception as exc:
        logger.warning("Failed to load normalizer from %s: %s", path, exc)
        return None


def apply_channel_normalizer(
    window_TxC: np.ndarray, normalizer: Any, *, out: Optional[np.ndarray] = None
) -> np.ndarray:
    arr = np.asarray(window_TxC, dtype=np.float32)
    if out is None:
        work = np.array(arr, dtype=np.float32, copy=True)
    else:
        if out.shape != arr.shape:
            raise ValueError(f"out shape {out.shape} does not match input shape {arr.shape}")
        np.copyto(out, arr)
        work = out
    if normalizer is None:
        return work
    if isinstance(normalizer, dict) and "mean" in normalizer and "std" in normalizer:
        _apply_preprocess_inplace(work, normalizer.get("preprocess"))
        mean = np.asarray(normalizer["mean"], dtype=np.float32)
        std = np.asarray(normalizer["std"], dtype=np.float32)
        std = np.where(std == 0, 1.0, std)
        work -= mean
        work /= std
        return work
    if hasattr(normalizer, "mean_") and hasattr(normalizer, "scale_"):
        mean = np.asarray(normalizer.mean_, dtype=np.float32)
        scale = np.asarray(normalizer.scale_, dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        work -= mean
        work /= scale
        return work
    return work


def compute_health_score(window_TxC: np.ndarray) -> float:
    if window_TxC.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(window_TxC**2)))
    saturation = float(np.mean(np.abs(window_TxC) > 2000.0))
    score_rms = (rms - 5.0) / 195.0
    score_rms = float(np.clip(score_rms, 0.0, 1.0))
    score = score_rms * (1.0 - saturation)
    return float(np.clip(score, 0.0, 1.0))


@dataclass
class CalibrationState:
    threshold: float
    config: dict


@dataclass
class TemperatureScalingState:
    action_temperature: float = 1.0
    finger_temperature: float = 1.0
    applicability_temperature: float = 1.0
    fit_sample_count: int = 0
    fit_non_rest_count: int = 0
    has_applicability_temperature: bool = False
    source: str = "unavailable"
    metrics: Optional[dict] = None


@dataclass
class LogitBiasState:
    action_bias: np.ndarray
    finger_bias: np.ndarray
    fit_sample_count: int = 0
    source: str = "unavailable"
    metrics: Optional[dict] = None


def load_calibration_state(path: Path) -> Optional[CalibrationState]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        threshold = float(payload.get("threshold", payload.get("init_threshold", 0.75)))
        config = payload.get("config", {})
        return CalibrationState(threshold=threshold, config=config)
    except Exception:
        return None


def apply_temperature_to_logits(logits: Any, temperature: float) -> Any:
    temp = max(1e-3, float(temperature))
    if torch.is_tensor(logits):
        return logits / temp
    return np.asarray(logits) / temp


def apply_logit_bias(logits: Any, bias: Any) -> Any:
    if bias is None:
        return logits
    if torch.is_tensor(logits):
        bias_t = torch.as_tensor(
            bias,
            dtype=logits.dtype,
            device=logits.device,
        )
        return logits + bias_t
    return np.asarray(logits) + np.asarray(bias, dtype=np.float32)


def save_temperature_scaling(path: Path, state: TemperatureScalingState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "action_temperature": float(state.action_temperature),
        "finger_temperature": float(state.finger_temperature),
        "applicability_temperature": float(state.applicability_temperature),
        "fit_sample_count": int(state.fit_sample_count),
        "fit_non_rest_count": int(state.fit_non_rest_count),
        "has_applicability_temperature": bool(state.has_applicability_temperature),
        "source": str(state.source),
        "metrics": state.metrics or {},
    }
    path.write_text(json.dumps(payload, indent=2))


def load_temperature_scaling(path: Path) -> Optional[TemperatureScalingState]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        has_applicability_temperature = "applicability_temperature" in payload
        return TemperatureScalingState(
            action_temperature=float(payload.get("action_temperature", 1.0)),
            finger_temperature=float(payload.get("finger_temperature", 1.0)),
            applicability_temperature=float(
                payload.get("applicability_temperature", 1.0)
            ),
            fit_sample_count=int(payload.get("fit_sample_count", 0)),
            fit_non_rest_count=int(payload.get("fit_non_rest_count", 0)),
            has_applicability_temperature=bool(
                payload.get(
                    "has_applicability_temperature",
                    has_applicability_temperature,
                )
            ),
            source=str(payload.get("source", "loaded")),
            metrics=payload.get("metrics", {}) or {},
        )
    except Exception as exc:
        logger.warning("Failed to load temperature scaling from %s: %s", path, exc)
        return None


def save_logit_bias_state(path: Path, state: LogitBiasState) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "action_bias": np.asarray(state.action_bias, dtype=np.float32).tolist(),
        "finger_bias": np.asarray(state.finger_bias, dtype=np.float32).tolist(),
        "fit_sample_count": int(state.fit_sample_count),
        "source": str(state.source),
        "metrics": state.metrics or {},
    }
    path.write_text(json.dumps(payload, indent=2))


def load_logit_bias_state(path: Path) -> Optional[LogitBiasState]:
    path = Path(path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        return LogitBiasState(
            action_bias=np.asarray(payload.get("action_bias", []), dtype=np.float32),
            finger_bias=np.asarray(payload.get("finger_bias", []), dtype=np.float32),
            fit_sample_count=int(payload.get("fit_sample_count", 0)),
            source=str(payload.get("source", "loaded")),
            metrics=payload.get("metrics", {}) or {},
        )
    except Exception as exc:
        logger.warning("Failed to load logit bias state from %s: %s", path, exc)
        return None
