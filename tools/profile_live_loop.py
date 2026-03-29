#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.runtime_utils import load_temperature_scaling, repo_root, resolve_device


def _load_live_module():
    module_path = repo_root() / "7_live_infer_and_actuate.py"
    spec = importlib.util.spec_from_file_location("live_infer_profile", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Step 7 module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ensure_window_ntc(window: np.ndarray) -> np.ndarray:
    arr = np.asarray(window, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected a single EEG window to be 2D, got {arr.shape}")
    if arr.shape[0] <= 16 and arr.shape[1] > 16:
        arr = arr.T
    return np.ascontiguousarray(arr)


def _load_window(npz_path: Path, window_index: int) -> np.ndarray:
    with np.load(npz_path) as payload:
        X = np.asarray(payload["X"])
        if X.ndim != 3:
            raise ValueError(f"Expected X to be 3D, got {X.shape}")
        idx = int(window_index)
        if idx < 0:
            idx += int(X.shape[0])
        if idx < 0 or idx >= int(X.shape[0]):
            raise IndexError(f"window_index={window_index} out of range for {X.shape[0]} windows")
        return _ensure_window_ntc(X[idx])


def _resolve_paths(run_dir_arg: Optional[str], npz_arg: Optional[str]) -> tuple[Path, Path]:
    root = repo_root()
    if run_dir_arg:
        run_dir = Path(run_dir_arg).expanduser().resolve()
    else:
        run_dir = root
    if npz_arg:
        npz_path = Path(npz_arg).expanduser().resolve()
    elif run_dir_arg:
        npz_path = run_dir.parents[1] / "eeg_windows.npz"
    else:
        npz_path = root / "eeg_windows.npz"
    return run_dir, npz_path


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def _build_args(mc_passes: int) -> Any:
    from utils.live_infer_common import ReplayRuntimeConfig
    from utils.inference import InferenceConfig

    runtime_defaults = ReplayRuntimeConfig()
    infer_defaults = InferenceConfig()
    return SimpleNamespace(
        use_inference_engine=True,
        uncertainty_base_threshold=float(infer_defaults.base_threshold),
        uncertainty_weight=float(infer_defaults.uncertainty_weight),
        actuation_stability=int(runtime_defaults.actuation_stability),
        mc_passes=max(1, int(mc_passes)),
    )


def _run_profile(
    *,
    backend: str,
    emit_viz: bool,
    window: np.ndarray,
    device: torch.device,
    iterations: int,
    warmup: int,
    mc_passes: int,
    seed: int,
    model: Any,
    scaler: Any,
    temperature_state: Any,
    live_mod: Any,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    direct_engine = None
    inference_engine = None
    if backend == "direct":
        direct_engine = live_mod._build_direct_inference_engine(
            model, scaler, device, temperature_state
        )
    elif backend == "mc":
        inference_engine = live_mod._build_inference_engine(
            model,
            scaler,
            device,
            _build_args(mc_passes),
            temperature_state,
        )
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    def _infer_once() -> Any:
        return live_mod._predict_window(
            window,
            scaler=scaler,
            model=model,
            device=device,
            inference_engine=inference_engine,
            direct_engine=direct_engine,
            temperature_state=temperature_state,
            emit_viz=emit_viz,
        )

    for _ in range(max(0, int(warmup))):
        _infer_once()
    _synchronize_device(device)

    timings_ms: list[float] = []
    last_result: Any = None
    for _ in range(max(1, int(iterations))):
        t0 = time.perf_counter()
        last_result = _infer_once()
        _synchronize_device(device)
        timings_ms.append((time.perf_counter() - t0) * 1000.0)

    ordered = sorted(timings_ms)
    p50 = ordered[len(ordered) // 2]
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "backend": backend,
        "emit_viz": bool(emit_viz),
        "iterations": int(iterations),
        "warmup": int(warmup),
        "mc_passes": int(mc_passes),
        "seed": int(seed),
        "device": str(device),
        "window_shape": list(window.shape),
        "mean_ms": float(mean(timings_ms)),
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "result_backend": (
            str(last_result.get("backend"))
            if isinstance(last_result, dict) and "backend" in last_result
            else None
        ),
        "has_viz_payload": bool(
            isinstance(last_result, dict) and last_result.get("live_viz_payload") is not None
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Repeatable Step 7 single-window benchmark for direct/MC inference and live-viz."
    )
    parser.add_argument("--run-dir", type=str, default=None, help="Model run directory. Defaults to repo root artifact files.")
    parser.add_argument("--npz", type=str, default=None, help="Window NPZ path. Defaults to <run_dir>/../eeg_windows.npz.")
    parser.add_argument(
        "--backend",
        type=str,
        default="direct",
        choices=["direct", "mc", "both"],
        help="Inference backend to profile.",
    )
    parser.add_argument(
        "--emit-viz",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include Step 7 live-viz payload generation in the benchmark.",
    )
    parser.add_argument("--mc-passes", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--window-index", type=int, default=0, help="Window index from the NPZ to profile.")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_dir, npz_path = _resolve_paths(args.run_dir, args.npz)
    if args.run_dir:
        model_path = run_dir / "finger_action_model.pt"
        scaler_path = run_dir / "scaler.npz"
        temperature_path = run_dir / "temperature_scaling.json"
    else:
        root = repo_root()
        model_path = root / "finger_action_model.pt"
        scaler_path = root / "scaler.npz"
        temperature_path = root / "temperature_scaling.json"
    if not model_path.exists():
        raise SystemExit(f"Model weights not found: {model_path}")
    if not scaler_path.exists():
        raise SystemExit(f"Scaler not found: {scaler_path}")
    if not npz_path.exists():
        raise SystemExit(f"Window NPZ not found: {npz_path}")

    live_mod = _load_live_module()
    device = resolve_device(args.device)
    model, scaler = live_mod.load_model_and_scaler(
        model_path,
        scaler_path,
        device=device,
    )
    temperature_state = (
        load_temperature_scaling(temperature_path)
        if temperature_path.exists()
        else None
    )
    window = _load_window(npz_path, int(args.window_index))

    backends = ["direct", "mc"] if args.backend == "both" else [str(args.backend)]
    results = [
        _run_profile(
            backend=backend,
            emit_viz=bool(args.emit_viz),
            window=window,
            device=device,
            iterations=int(args.iterations),
            warmup=int(args.warmup),
            mc_passes=int(args.mc_passes),
            seed=int(args.seed),
            model=model,
            scaler=scaler,
            temperature_state=temperature_state,
            live_mod=live_mod,
        )
        for backend in backends
    ]

    payload = {
        "run_dir": str(run_dir) if args.run_dir else None,
        "npz": str(npz_path),
        "temperature_path": str(temperature_path) if temperature_path.exists() else None,
        "results": results,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
