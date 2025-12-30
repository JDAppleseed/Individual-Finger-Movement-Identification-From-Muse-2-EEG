from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from demo_backend.inference import InferenceConfig, InferenceEngine
from demo_backend.live_lsl import LiveLSLSource
from demo_backend.replay import ReplaySource
from demo_backend.schemas import schema_bundle
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


class ControlPayload(BaseModel):
    mode: str
    replay_path: Optional[str] = None
    fps: Optional[float] = None
    device: Optional[str] = None
    mc_passes: Optional[int] = None


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
        self.last_status_sent = 0.0
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
        state.runtime.mode = payload.mode
        if payload.replay_path:
            state.runtime.replay_path = Path(payload.replay_path)
        if payload.fps:
            state.runtime.fps = float(payload.fps)
        if payload.device:
            state.runtime.device = payload.device
        if payload.mc_passes:
            state.runtime.mc_passes = int(payload.mc_passes)
            if state.engine:
                state.engine.config.mc_passes = state.runtime.mc_passes
        if state.engine:
            state.engine.set_device(resolve_device(state.runtime.device))
    return {"status": "ok", "mode": state.runtime.mode}


async def _send_status(ws: WebSocket, level: str, message: str, details: Optional[dict] = None):
    await ws.send_json({
        "type": "status",
        "ts_utc": now_utc_iso(),
        "level": level,
        "message": message,
        "details": details or {},
    })


async def _stream_replay(ws: WebSocket):
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
        t0 = time.perf_counter()
        prediction, safety, diag = state.engine.predict(window)
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
            },
        }

        await ws.send_json(tick)
        await asyncio.sleep(max(0.0, (1.0 / state.runtime.fps)))

    if state.runtime.mode == "replay":
        state.runtime.mode = "idle"
        await _send_status(ws, "info", "Replay complete", {})


async def _stream_live(ws: WebSocket):
    if not state.lsl_connected:
        ok, msg = state.live.connect()
        state.lsl_connected = ok
        level = "info" if ok else "warning"
        await _send_status(ws, level, msg, {})
        if not ok:
            return

    start_time = time.perf_counter()
    tick_count = 0
    while state.runtime.mode == "live":
        window_data = state.live.pull_window()
        if window_data is None:
            await asyncio.sleep(0.01)
            continue

        t0 = time.perf_counter()
        prediction, safety, diag = state.engine.predict(window_data.window)
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
            },
        }

        await ws.send_json(tick)
        await asyncio.sleep(max(0.0, (1.0 / state.runtime.fps)))


@app.websocket("/stream")
async def stream(ws: WebSocket):
    await ws.accept()
    await _send_status(ws, "info", "Connected", {"mode": state.runtime.mode})

    while True:
        if state.runtime.mode == "idle":
            now = time.time()
            if now - state.last_status_sent > 2.0:
                await _send_status(ws, "info", "Idle", {"mode": "idle"})
                state.last_status_sent = now
            await asyncio.sleep(0.5)
            continue

        if state.runtime.mode == "replay":
            await _stream_replay(ws)
        elif state.runtime.mode == "live":
            await _stream_live(ws)
        else:
            await asyncio.sleep(0.5)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8008)
