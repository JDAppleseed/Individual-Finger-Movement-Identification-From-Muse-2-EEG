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

NORMALIZER_VERSION = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
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
    return {
        "version": NORMALIZER_VERSION,
        "type": norm_type,
        "mean": mean,
        "std": std,
        "channels": channels,
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
            return {
                "type": "per_channel",
                "mean": mean,
                "std": std,
                "channels": channels,
            }
    except Exception as exc:
        logger.warning("Failed to load normalizer from %s: %s", path, exc)
        return None


def apply_channel_normalizer(window_TxC: np.ndarray, normalizer: Any) -> np.ndarray:
    if normalizer is None:
        return window_TxC
    if isinstance(normalizer, dict) and "mean" in normalizer and "std" in normalizer:
        mean = np.asarray(normalizer["mean"], dtype=np.float32)
        std = np.asarray(normalizer["std"], dtype=np.float32)
        std = np.where(std == 0, 1.0, std)
        return (window_TxC - mean) / std
    if hasattr(normalizer, "mean_") and hasattr(normalizer, "scale_"):
        mean = np.asarray(normalizer.mean_, dtype=np.float32)
        scale = np.asarray(normalizer.scale_, dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        return (window_TxC - mean) / scale
    return window_TxC


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
