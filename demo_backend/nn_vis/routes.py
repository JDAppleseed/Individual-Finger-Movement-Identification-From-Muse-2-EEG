from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from demo_backend.nn_vis.extract import (
    extract_activations,
    extract_architecture_manifest,
    extract_weights,
    load_model_and_weights,
    pack_tensor,
)
from demo_backend.utils_demo import load_normalizer, repo_root
from utils.label_schema import ACTION_NAMES, FINGER_NAMES


ROOT = repo_root()
MODEL_PATH = ROOT / "finger_action_model.pt"
NORMALIZER_PATH = ROOT / "scaler.save"
TIMELINE_DIR = ROOT / "exports" / "nnvis_timeline"
logger = logging.getLogger("demo_backend.nn_vis")
logger.setLevel(logging.INFO)


@dataclass
class ModelCache:
    model_path: Optional[Path] = None
    model: Optional[object] = None
    normalizer: Optional[object] = None


cache = ModelCache()
router = APIRouter()


def _get_model() -> object:
    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=404, detail=f"Model weights not found at {MODEL_PATH}"
        )
    if cache.model is None or cache.model_path != MODEL_PATH:
        cache.model = load_model_and_weights(str(MODEL_PATH), device="cpu")
        cache.model_path = MODEL_PATH
    if cache.normalizer is None:
        cache.normalizer = load_normalizer(NORMALIZER_PATH)
    return cache.model


def _resolve_source(path_str: str) -> Path:
    raw = Path(path_str)
    if not raw.is_absolute():
        raw = (ROOT / raw).resolve()
    else:
        raw = raw.resolve()

    if not raw.exists():
        raise HTTPException(status_code=404, detail=f"Source not found: {raw}")

    try:
        raw.relative_to(ROOT)
    except ValueError:
        raise HTTPException(
            status_code=400, detail="Source must be within repository root"
        )

    return raw


def _count_samples(npz_path: Path) -> int:
    data = np.load(npz_path, allow_pickle=True)
    if "X" in data:
        return int(data["X"].shape[0])
    if "action_probs" in data:
        return int(data["action_probs"].shape[0])
    if "test_indices" in data:
        return int(data["test_indices"].shape[0])
    return 0


def _find_sources() -> List[Dict[str, object]]:
    locations = [ROOT, ROOT / "data" / "processed", ROOT / "reports"]
    names = ["test_predictions.npz", "eeg_windows.npz"]
    seen = set()
    sources = []
    for name in names:
        for base in locations:
            path = base / name
            if path.exists():
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                sources.append(
                    {
                        "path": str(resolved),
                        "name": name,
                        "sample_count": _count_samples(resolved),
                    }
                )
    return sources


def _find_fallback_eeg_windows() -> Optional[Path]:
    locations = [ROOT, ROOT / "data" / "processed", ROOT / "reports"]
    for base in locations:
        candidate = base / "eeg_windows.npz"
        if candidate.exists():
            return candidate.resolve()
    return None


def _load_window_from_npz(
    npz_path: Path, index: int
) -> tuple[np.ndarray, Optional[float], int]:
    data = np.load(npz_path, allow_pickle=True)
    if "X" in data:
        X = data["X"]
        if index < 0 or index >= X.shape[0]:
            raise HTTPException(status_code=400, detail="Index out of range")
        window = X[index]
        window_start = data["window_start"][index] if "window_start" in data else None
        window_end = data["window_end"][index] if "window_end" in data else None
        time_s = None
        if window_start is not None and window_end is not None:
            time_s = float((window_start + window_end) / 2.0)
        elif window_end is not None:
            time_s = float(window_end)
        return window, time_s, int(index)

    if "test_indices" in data:
        test_indices = data["test_indices"]
        if index < 0 or index >= test_indices.shape[0]:
            raise HTTPException(status_code=400, detail="Index out of range")
        real_index = int(test_indices[index])
        # Resolve eeg_windows across known locations for test_predictions indices.
        fallback = _find_fallback_eeg_windows()
        if not fallback:
            locations = [
                str(ROOT),
                str(ROOT / "data" / "processed"),
                str(ROOT / "reports"),
            ]
            raise HTTPException(
                status_code=404,
                detail=f"eeg_windows.npz required for test_predictions indices. Searched: {locations}",
            )
        window, time_s, _ = _load_window_from_npz(fallback, real_index)
        return window, time_s, real_index

    raise HTTPException(status_code=400, detail="No windows found in npz")


@router.get("/nnvis/manifest")
async def nnvis_manifest():
    model = _get_model()
    timeline_available = (TIMELINE_DIR / "manifest.json").exists()
    logger.info("nnvis_manifest served")
    return extract_architecture_manifest(model, timeline_available)


@router.get("/nnvis/weights")
async def nnvis_weights(
    quantize: int = Query(1, ge=0, le=1),
    downsample: int = Query(1, ge=1, le=8),
    topk: int = Query(150, ge=10, le=1000),
):
    model = _get_model()
    logger.info(
        "nnvis_weights served quantize=%s downsample=%s topk=%s",
        quantize,
        downsample,
        topk,
    )
    return extract_weights(
        model, quantize=bool(quantize), downsample=downsample, topk=topk
    )


@router.get("/nnvis/offline/sources")
async def nnvis_offline_sources():
    sources = _find_sources()
    return {"sources": sources}


@router.get("/nnvis/offline/sample")
async def nnvis_offline_sample(source: str, index: int = Query(0, ge=0)):
    path = _resolve_source(source)
    model = _get_model()
    window, time_s, true_index = _load_window_from_npz(path, index)
    activations, uncertainty = extract_activations(
        model,
        window,
        deterministic=True,
        normalizer=cache.normalizer,
        mc_passes=None,
    )
    finger_probs = np.asarray(activations["finger_probs"])
    action_probs = np.asarray(activations["action_probs"])
    finger_pred = int(np.argmax(finger_probs))
    action_pred = int(np.argmax(action_probs))

    return {
        "sample": {"source": "offline", "index": int(true_index), "time_s": time_s},
        "input": {"shape": [64, 4], "values": pack_tensor(activations["input"])},
        "conv1": {"shape": [16, 64], "values": pack_tensor(activations["conv1"])},
        "conv2": {"shape": [32, 64], "values": pack_tensor(activations["conv2"])},
        "lstm_out": {"shape": [64, 64], "values": pack_tensor(activations["lstm_out"])},
        "last_features": {
            "shape": [64],
            "values": pack_tensor(activations["last_features"]),
        },
        "probs": {
            "finger": {
                "values": finger_probs.tolist(),
                "pred_id": finger_pred,
                "pred_name": FINGER_NAMES.get(finger_pred, "UNKNOWN"),
            },
            "action": {
                "values": action_probs.tolist(),
                "pred_id": action_pred,
                "pred_name": ACTION_NAMES.get(action_pred, "UNKNOWN"),
            },
        },
        "uncertainty": uncertainty,
    }


@router.get("/nnvis/timeline/manifest")
async def nnvis_timeline_manifest():
    manifest_path = TIMELINE_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Timeline manifest not found")
    return json.loads(manifest_path.read_text())


@router.get("/nnvis/timeline/weights")
async def nnvis_timeline_weights(file: str):
    manifest_path = TIMELINE_DIR / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail="Timeline manifest not found")
    weights_path = (TIMELINE_DIR / file).resolve()
    if not weights_path.exists():
        raise HTTPException(status_code=404, detail="Timeline weights not found")
    try:
        weights_path.relative_to(TIMELINE_DIR.resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid weights path")

    payload = np.load(weights_path, allow_pickle=True)
    if "weights" not in payload:
        raise HTTPException(
            status_code=400, detail="Timeline weights file missing 'weights'"
        )
    return json.loads(payload["weights"].item())
