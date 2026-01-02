from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from demo_backend.inference import InferenceConfig, InferenceEngine
from demo_backend.live_lsl import LiveLSLSource
from demo_backend.nn_vis.extract import extract_activations, pack_tensor
from demo_backend.nn_vis.routes import router as nnvis_router
from demo_backend.replay import ReplaySource
from demo_backend.schemas import schema_bundle
from demo_backend.postprocess import PostprocessSettings, PostprocessState, postprocess_predictions
from demo_backend.utils_demo import (
    ensure_repo_on_path,
    load_calibration_state,
    load_normalizer,
    now_utc_iso,
    repo_root,
    resolve_device,
    setup_logger,
)

ensure_repo_on_path()

from utils.label_schema import ACTION_NAMES, FINGER_NAMES  # noqa: E402
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet  # noqa: E402


ROOT = repo_root()
LOG_DIR = ROOT / "demo_backend" / "logs"
RECORDINGS_DIR = ROOT / "demo_backend" / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

logger = setup_logger("demo_backend", LOG_DIR)


@dataclass
class RuntimeState:
    mode: str = "idle"
    replay_path: Path = ROOT / "eeg_windows.npz"
    fps: float = 20.0
    device: str = "cpu"
    mc_passes: int = 10
    smoothing_enabled: bool = True
    smoothing_method: str = "vote"
    smoothing_window: int = 5
    hysteresis_enabled: bool = True
    hysteresis_frames: int = 3
    threshold_action: float = 0.75
    threshold_finger: float = 0.75
    adjacency_enabled: bool = True
    user_set_hysteresis: bool = False


class ControlPayload(BaseModel):
    mode: str
    replay_path: Optional[str] = None
    fps: Optional[float] = None
    device: Optional[str] = None
    mc_passes: Optional[int] = None
    smoothing_enabled: Optional[bool] = None
    smoothing_method: Optional[str] = None
    smoothing_window: Optional[int] = None
    hysteresis_enabled: Optional[bool] = None
    hysteresis_frames: Optional[int] = None
    threshold_action: Optional[float] = None
    threshold_finger: Optional[float] = None
    adjacency_enabled: Optional[bool] = None


@dataclass
class NnvisSubscription:
    enabled: bool = False
    rate_hz: float = 5.0
    last_sent: float = 0.0


class BackendState:
    def __init__(self) -> None:
        self.runtime = RuntimeState()
        self.engine: Optional[InferenceEngine] = None
        self.model_loaded = False
        self.normalizer_loaded = False
        self.calibration_loaded = False
        self.replay: Optional[ReplaySource] = None
        self.live = LiveLSLSource()
        self.lsl_connected = False
        self.live_status_sent = False
        self.last_status_sent = 0.0
        self.postprocess = PostprocessState()
        self.lock = asyncio.Lock()

    def load_assets(self) -> None:
        model_path = ROOT / "finger_action_model.pt"
        normalizer_path = ROOT / "scaler.save"
        calib_path = ROOT / "logs" / "calibration"

        normalizer = load_normalizer(normalizer_path)
        self.normalizer_loaded = normalizer is not None

        calibration_state = None
        if calib_path.exists():
            for state_file in calib_path.glob("calibration_state_*.json"):
                calibration_state = load_calibration_state(state_file)
                if calibration_state:
                    self.calibration_loaded = True
                    break

        if not model_path.exists():
            logger.warning("Model weights not found at %s", model_path)
            self.engine = InferenceEngine(
                model=None,
                normalizer=normalizer,
                device=resolve_device(self.runtime.device),
                action_names=ACTION_NAMES,
                finger_names=FINGER_NAMES,
                config=InferenceConfig(mc_passes=self.runtime.mc_passes),
            )
            return

        state = torch.load(model_path, map_location="cpu")
        # Infer head sizes from checkpoint to avoid mismatch with label schema
        n_fingers = int(state["finger_head.weight"].shape[0])
        n_actions = int(state["action_head.weight"].shape[0])
        model = CNNLSTMFingerActionNet(n_channels=4, n_fingers=n_fingers, n_actions=n_actions)
        model.load_state_dict(state)
        self.model_loaded = True
        base_threshold = calibration_state.threshold if calibration_state else 0.75
        config = InferenceConfig(
            base_threshold=base_threshold,
            mc_passes=self.runtime.mc_passes,
        )
        self.engine = InferenceEngine(
            model=model,
            normalizer=normalizer,
            device=resolve_device(self.runtime.device),
            action_names=ACTION_NAMES,
            finger_names=FINGER_NAMES,
            config=config,
        )


state = BackendState()
state.load_assets()

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(nnvis_router)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": state.model_loaded,
        "normalizer_loaded": state.normalizer_loaded,
        "mode": state.runtime.mode,
    }


@app.get("/schema")
async def schema():
    return schema_bundle()


@app.post("/control")
async def control(payload: ControlPayload):
    async with state.lock:
        previous_mode = state.runtime.mode
        state.runtime.mode = payload.mode
        if payload.mode != previous_mode:
            state.postprocess.reset()
            if not state.runtime.user_set_hysteresis:
                state.runtime.hysteresis_enabled = payload.mode == "live"
        if payload.mode != "live":
            state.live_status_sent = False
        if payload.replay_path:
            state.runtime.replay_path = Path(payload.replay_path)
            if state.replay and state.replay.path != state.runtime.replay_path:
                state.postprocess.reset()
        if payload.fps:
            state.runtime.fps = float(payload.fps)
        if payload.device:
            state.runtime.device = payload.device
        if payload.mc_passes:
            state.runtime.mc_passes = int(payload.mc_passes)
            if state.engine:
                state.engine.config.mc_passes = state.runtime.mc_passes
        if payload.smoothing_enabled is not None:
            state.runtime.smoothing_enabled = bool(payload.smoothing_enabled)
        if payload.smoothing_method:
            state.runtime.smoothing_method = payload.smoothing_method
        if payload.smoothing_window:
            state.runtime.smoothing_window = int(payload.smoothing_window)
        if payload.hysteresis_enabled is not None:
            state.runtime.hysteresis_enabled = bool(payload.hysteresis_enabled)
            state.runtime.user_set_hysteresis = True
        if payload.hysteresis_frames:
            state.runtime.hysteresis_frames = int(payload.hysteresis_frames)
        if payload.threshold_action is not None:
            state.runtime.threshold_action = float(payload.threshold_action)
        if payload.threshold_finger is not None:
            state.runtime.threshold_finger = float(payload.threshold_finger)
        if payload.adjacency_enabled is not None:
            state.runtime.adjacency_enabled = bool(payload.adjacency_enabled)
        if state.engine:
            state.engine.set_device(resolve_device(state.runtime.device))
    return {"status": "ok", "mode": state.runtime.mode}


def _build_postprocess_settings():
    return PostprocessSettings(
        smoothing_enabled=state.runtime.smoothing_enabled,
        smoothing_method=state.runtime.smoothing_method,
        smoothing_window=state.runtime.smoothing_window,
        hysteresis_enabled=state.runtime.hysteresis_enabled,
        hysteresis_frames=state.runtime.hysteresis_frames,
        threshold_action=state.runtime.threshold_action,
        threshold_finger=state.runtime.threshold_finger,
        adjacency_enabled=state.runtime.adjacency_enabled,
    )


async def _send_status(ws: WebSocket, level: str, message: str, details: Optional[dict] = None):
    await ws.send_json({
        "type": "status",
        "ts_utc": now_utc_iso(),
        "level": level,
        "message": message,
        "details": details or {},
    })


def _build_nnvis_payload(window: np.ndarray, source: str, index: Optional[int], time_s: Optional[float], passes: int) -> Optional[dict]:
    if state.engine is None or state.engine.model is None:
        return None
    activations, uncertainty = extract_activations(
        state.engine.model,
        window,
        deterministic=True,
        normalizer=state.engine.normalizer,
        mc_passes=passes if passes > 0 else None,
    )

    finger_probs = activations["finger_probs"]
    action_probs = activations["action_probs"]
    finger_pred = int(np.argmax(finger_probs))
    action_pred = int(np.argmax(action_probs))

    # Pack activations for websocket efficiency; keep probabilities as lists.
    return {
        "sample": {"source": source, "index": index, "time_s": time_s},
        "input": {"shape": [64, 4], "values": pack_tensor(activations["input"])},
        "conv1": {"shape": [16, 64], "values": pack_tensor(activations["conv1"])},
        "conv2": {"shape": [32, 64], "values": pack_tensor(activations["conv2"])},
        "lstm_out": {"shape": [64, 64], "values": pack_tensor(activations["lstm_out"])},
        "last_features": {"shape": [64], "values": pack_tensor(activations["last_features"])},
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


async def _stream_replay(ws: WebSocket, nnvis: NnvisSubscription):
    if state.replay is None or state.replay.path != state.runtime.replay_path:
        if not state.runtime.replay_path.exists():
            await _send_status(ws, "error", f"Replay file not found: {state.runtime.replay_path}")
            state.runtime.mode = "idle"
            return
        state.replay = ReplaySource(state.runtime.replay_path)

    start_time = time.perf_counter()
    tick_count = 0
    for window, meta in state.replay.iter_windows():
        if state.runtime.mode != "replay":
            break
        post = None
        t0 = time.perf_counter()
        action_probs, finger_probs, action_unc, finger_unc, diag = state.engine.predict_proba(window)
        if action_probs is None or finger_probs is None:
            prediction, safety, diag2 = state.engine.predict(window)
            diag.update(diag2)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        else:
            settings = _build_postprocess_settings()
            post = postprocess_predictions(action_probs, finger_probs, settings, state.postprocess)
            committed_action = int(post["committed_action_id"])
            committed_finger = int(post["committed_finger_id"])
            action_conf = float(post["action_conf"])
            finger_conf = float(post["finger_conf"])

            prediction = {
                "action_id": committed_action,
                "action_name": ACTION_NAMES.get(committed_action, "UNKNOWN"),
                "finger_id": committed_finger,
                "finger_name": FINGER_NAMES.get(committed_finger, "UNKNOWN"),
                "action_confidence": action_conf,
                "action_uncertainty": float(action_unc),
                "finger_confidence": finger_conf,
                "finger_uncertainty": float(finger_unc),
            }

            allow_actuation = committed_action != 0 and action_conf >= settings.threshold_action
            safety = {
                "base_threshold": settings.threshold_action,
                "adaptive_threshold": settings.threshold_action,
                "allow_actuation": allow_actuation,
                "stability_frames": settings.hysteresis_frames,
                "stability_ok": post["decision_reason"] not in {"hysteresis_hold", "below_threshold"},
                "velocity": action_conf * (1.0 - float(action_unc)) if committed_action != 0 else 0.0,
            }
            latency_ms = (time.perf_counter() - t0) * 1000.0
        tick_count += 1
        elapsed = time.perf_counter() - start_time
        fps_actual = tick_count / elapsed if elapsed > 0 else state.runtime.fps

        tick = {
            "type": "tick",
            "ts_utc": now_utc_iso(),
            "mode": "replay",
            "session": {
                "subject_id": meta.subject_id,
                "experiment_hash": meta.experiment_hash,
                "window_index": meta.index,
                "window_start_s": meta.window_start_s,
                "window_end_s": meta.window_end_s,
            },
            "prediction": prediction,
            "safety": safety,
            "diagnostics": {
                "latency_ms": latency_ms,
                "fps_target": state.runtime.fps,
                "fps_actual": fps_actual,
                "health_score": diag["health_score"],
                "lsl_connected": False,
                "artifact_suppression": None,
                "notes": "calibration_loaded=true" if state.calibration_loaded else "",
                "smoothing_enabled": state.runtime.smoothing_enabled,
                "smoothing_method": state.runtime.smoothing_method,
                "smoothing_window": state.runtime.smoothing_window,
                "hysteresis_enabled": state.runtime.hysteresis_enabled if state.runtime.mode == "live" else False,
                "hysteresis_frames": state.runtime.hysteresis_frames,
                "threshold_action": state.runtime.threshold_action,
                "threshold_finger": state.runtime.threshold_finger,
                "adjacency_enabled": state.runtime.adjacency_enabled,
                "decision_reason": post["decision_reason"] if post else "",
                "raw_top_action_id": post["raw_top_action_id"] if post else -1,
                "raw_top_finger_id": post["raw_top_finger_id"] if post else -1,
                "committed_action_id": post["committed_action_id"] if post else prediction.get("action_id", -1),
                "committed_finger_id": post["committed_finger_id"] if post else prediction.get("finger_id", -1),
                "smoothed_action_id": post["smoothed_action_id"] if post else prediction.get("action_id", -1),
                "smoothed_finger_id": post["smoothed_finger_id"] if post else prediction.get("finger_id", -1),
                "frames_in_state": post["frames_in_state"] if post else 0,
            },
        }

        if nnvis.enabled:
            now = time.perf_counter()
            interval = 1.0 / max(nnvis.rate_hz, 0.1)
            if now - nnvis.last_sent >= interval:
                nnvis_passes = state.runtime.mc_passes
                if nnvis.rate_hz > 2.0 and nnvis_passes > 3:
                    # Clamp MC passes to keep high-rate nnvis updates responsive.
                    nnvis_passes = 3
                tick["nnvis"] = _build_nnvis_payload(
                    window,
                    source="replay",
                    index=meta.index,
                    time_s=meta.window_end_s,
                    passes=nnvis_passes,
                )
                nnvis.last_sent = now

        await ws.send_json(tick)
        await asyncio.sleep(max(0.0, (1.0 / state.runtime.fps)))

    if state.runtime.mode == "replay":
        state.runtime.mode = "idle"
        await _send_status(ws, "info", "Replay complete", {})


async def _stream_live(ws: WebSocket, nnvis: NnvisSubscription):
    if not state.lsl_connected:
        ok, msg = state.live.connect()
        state.lsl_connected = ok
        level = "info" if ok else "warning"
        await _send_status(ws, level, msg, {})
        state.live_status_sent = ok
        if not ok:
            return
    elif not state.live_status_sent and state.live.status_message:
        await _send_status(ws, "info", state.live.status_message, {})
        state.live_status_sent = True

    start_time = time.perf_counter()
    tick_count = 0
    while state.runtime.mode == "live":
        window_data = state.live.pull_window()
        if window_data is None:
            await asyncio.sleep(0.01)
            continue

        t0 = time.perf_counter()
        post = None
        action_probs, finger_probs, action_unc, finger_unc, diag = state.engine.predict_proba(window_data.window)
        if action_probs is None or finger_probs is None:
            prediction, safety, diag2 = state.engine.predict(window_data.window)
            diag.update(diag2)
            latency_ms = (time.perf_counter() - t0) * 1000.0
        else:
            settings = _build_postprocess_settings()
            post = postprocess_predictions(action_probs, finger_probs, settings, state.postprocess)
            committed_action = int(post["committed_action_id"])
            committed_finger = int(post["committed_finger_id"])
            action_conf = float(post["action_conf"])
            finger_conf = float(post["finger_conf"])

            prediction = {
                "action_id": committed_action,
                "action_name": ACTION_NAMES.get(committed_action, "UNKNOWN"),
                "finger_id": committed_finger,
                "finger_name": FINGER_NAMES.get(committed_finger, "UNKNOWN"),
                "action_confidence": action_conf,
                "action_uncertainty": float(action_unc),
                "finger_confidence": finger_conf,
                "finger_uncertainty": float(finger_unc),
            }

            allow_actuation = committed_action != 0 and action_conf >= settings.threshold_action
            safety = {
                "base_threshold": settings.threshold_action,
                "adaptive_threshold": settings.threshold_action,
                "allow_actuation": allow_actuation,
                "stability_frames": settings.hysteresis_frames,
                "stability_ok": post["decision_reason"] not in {"hysteresis_hold", "below_threshold"},
                "velocity": action_conf * (1.0 - float(action_unc)) if committed_action != 0 else 0.0,
            }
            latency_ms = (time.perf_counter() - t0) * 1000.0
        tick_count += 1
        elapsed = time.perf_counter() - start_time
        fps_actual = tick_count / elapsed if elapsed > 0 else state.runtime.fps

        tick = {
            "type": "tick",
            "ts_utc": now_utc_iso(),
            "mode": "live",
            "session": {
                "subject_id": "LIVE",
                "experiment_hash": "LIVE",
                "window_index": tick_count,
                "window_start_s": window_data.window_start_s,
                "window_end_s": window_data.window_end_s,
            },
            "prediction": prediction,
            "safety": safety,
            "diagnostics": {
                "latency_ms": latency_ms,
                "fps_target": state.runtime.fps,
                "fps_actual": fps_actual,
                "health_score": diag["health_score"],
                "lsl_connected": True,
                "artifact_suppression": False,
                "notes": "calibration_loaded=true" if state.calibration_loaded else "",
                "smoothing_enabled": state.runtime.smoothing_enabled,
                "smoothing_method": state.runtime.smoothing_method,
                "smoothing_window": state.runtime.smoothing_window,
                "hysteresis_enabled": state.runtime.hysteresis_enabled if state.runtime.mode == "live" else False,
                "hysteresis_frames": state.runtime.hysteresis_frames,
                "threshold_action": state.runtime.threshold_action,
                "threshold_finger": state.runtime.threshold_finger,
                "adjacency_enabled": state.runtime.adjacency_enabled,
                "decision_reason": post["decision_reason"] if post else "",
                "raw_top_action_id": post["raw_top_action_id"] if post else -1,
                "raw_top_finger_id": post["raw_top_finger_id"] if post else -1,
                "committed_action_id": post["committed_action_id"] if post else prediction.get("action_id", -1),
                "committed_finger_id": post["committed_finger_id"] if post else prediction.get("finger_id", -1),
                "smoothed_action_id": post["smoothed_action_id"] if post else prediction.get("action_id", -1),
                "smoothed_finger_id": post["smoothed_finger_id"] if post else prediction.get("finger_id", -1),
                "frames_in_state": post["frames_in_state"] if post else 0,
            },
        }

        if nnvis.enabled:
            now = time.perf_counter()
            interval = 1.0 / max(nnvis.rate_hz, 0.1)
            if now - nnvis.last_sent >= interval:
                nnvis_passes = state.runtime.mc_passes
                if nnvis.rate_hz > 2.0 and nnvis_passes > 3:
                    # Clamp MC passes to keep high-rate nnvis updates responsive.
                    nnvis_passes = 3
                tick["nnvis"] = _build_nnvis_payload(
                    window_data.window,
                    source="live",
                    index=tick_count,
                    time_s=window_data.window_end_s,
                    passes=nnvis_passes,
                )
                nnvis.last_sent = now

        await ws.send_json(tick)
        await asyncio.sleep(max(0.0, (1.0 / state.runtime.fps)))


async def _listen_ws(ws: WebSocket, nnvis: NnvisSubscription):
    while True:
        try:
            message = await ws.receive_json()
        except WebSocketDisconnect:
            return
        if not isinstance(message, dict):
            continue
        if message.get("type") == "nnvis_subscribe":
            # Guard parsing to keep websocket resilient to malformed inputs.
            enabled = message.get("enabled", nnvis.enabled)
            if isinstance(enabled, bool):
                nnvis.enabled = enabled
            elif isinstance(enabled, (int, float, str)):
                nnvis.enabled = str(enabled).lower() not in {"0", "false", "none", ""}

            rate_hz = message.get("rate_hz", nnvis.rate_hz)
            try:
                rate_value = float(rate_hz)
            except (TypeError, ValueError):
                rate_value = None
            if rate_value is not None:
                nnvis.rate_hz = max(0.5, min(rate_value, 30.0))


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    state.postprocess.reset()
    await _send_status(ws, "info", "Connected", {"mode": state.runtime.mode})
    nnvis = NnvisSubscription()
    listener = asyncio.create_task(_listen_ws(ws, nnvis))

    while True:
        if state.runtime.mode == "idle":
            now = time.time()
            if now - state.last_status_sent > 2.0:
                await _send_status(ws, "info", "Idle", {"mode": "idle"})
                state.last_status_sent = now
            await asyncio.sleep(0.5)
            continue

        try:
            if state.runtime.mode == "replay":
                await _stream_replay(ws, nnvis)
            elif state.runtime.mode == "live":
                await _stream_live(ws, nnvis)
            else:
                await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            break

    listener.cancel()
    try:
        await listener
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
