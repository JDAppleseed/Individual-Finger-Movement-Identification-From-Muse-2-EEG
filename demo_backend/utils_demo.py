from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import joblib


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_repo_on_path() -> None:
    root = repo_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def setup_logger(name: str, log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        file_handler = logging.FileHandler(log_dir / "demo_backend.log")
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested in {"cuda", "auto"} and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_normalizer(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return joblib.load(path)
    except Exception:
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
    rms = float(np.sqrt(np.mean(window_TxC ** 2)))
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
