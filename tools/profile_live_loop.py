#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from collections import deque
from statistics import mean
from typing import Deque, List, Optional, Tuple

import numpy as np
import torch

from demo_backend.inference import InferenceConfig, InferenceEngine
from demo_backend.timebase import clamp_monotonic_window
from demo_backend.utils_demo import load_normalizer, repo_root, resolve_device

from utils.label_schema import ACTION_NAMES, FINGER_NAMES
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet


class SlidingBuffer:
    def __init__(self, fs: int, window_sec: float, step_sec: float) -> None:
        self.fs = fs
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.window_samples = int(fs * window_sec)
        self.step_samples = max(1, int(fs * step_sec))
        self.buffer: Deque[np.ndarray] = deque(maxlen=self.window_samples)
        self.sample_times: Deque[float] = deque(maxlen=self.window_samples)
        self.stream_start: Optional[float] = None
        self.sample_count = 0
        self.last_emit = 0
        self.last_window_end_s: Optional[float] = None

    def push(
        self, sample: np.ndarray, lsl_ts: float
    ) -> Optional[Tuple[np.ndarray, float, float, float]]:
        if self.stream_start is None:
            self.stream_start = lsl_ts
        self.buffer.append(sample)
        self.sample_times.append(lsl_ts)
        if len(self.buffer) < self.window_samples:
            return None
        self.sample_count += 1
        if (self.sample_count - self.last_emit) < self.step_samples:
            return None
        self.last_emit = self.sample_count
        window = np.array(self.buffer, dtype=np.float32)
        start_s = float(self.sample_times[0] - self.stream_start)
        end_s = float(self.sample_times[-1] - self.stream_start)
        start_s, end_s, _ = clamp_monotonic_window(
            self.last_window_end_s, start_s, end_s
        )
        self.last_window_end_s = end_s
        emitted_perf = time.perf_counter()
        return window, start_s, end_s, emitted_perf


def _load_engine(device: str, mc_passes: int) -> InferenceEngine:
    root = repo_root()
    model_path = root / "finger_action_model.pt"
    normalizer = load_normalizer(root / "scaler.save")
    model: Optional[CNNLSTMFingerActionNet] = None
    if model_path.exists():
        state = torch.load(model_path, map_location="cpu")
        n_fingers = int(state["finger_head.weight"].shape[0])
        n_actions = int(state["action_head.weight"].shape[0])
        model = CNNLSTMFingerActionNet(
            n_channels=4, n_fingers=n_fingers, n_actions=n_actions
        )
        model.load_state_dict(state)
        model.eval()
    return InferenceEngine(
        model=model,
        normalizer=normalizer,
        device=resolve_device(device),
        action_names=ACTION_NAMES,
        finger_names=FINGER_NAMES,
        config=InferenceConfig(mc_passes=mc_passes),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fs", type=int, default=256)
    parser.add_argument("--window-sec", type=float, default=0.25)
    parser.add_argument("--step-sec", type=float, default=0.05)
    parser.add_argument("--duration-sec", type=float, default=3.0)
    parser.add_argument("--mc-passes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    buffer = SlidingBuffer(args.fs, args.window_sec, args.step_sec)
    engine = _load_engine(args.device, args.mc_passes)

    hop_ms: List[float] = []
    infer_ms: List[float] = []
    publish_age_ms: List[float] = []
    last_emit_perf: Optional[float] = None

    start = time.perf_counter()
    samples = int(args.duration_sec * args.fs)
    for i in range(samples):
        lsl_ts = i / float(args.fs)
        sample = np.random.randn(4).astype(np.float32)
        result = buffer.push(sample, lsl_ts)
        if result is None:
            continue
        window, _, _, emitted_perf = result
        t0 = time.perf_counter()
        engine.predict_proba(window)
        infer_ms.append((time.perf_counter() - t0) * 1000.0)
        if last_emit_perf is not None:
            hop_ms.append((emitted_perf - last_emit_perf) * 1000.0)
        last_emit_perf = emitted_perf
        publish_age_ms.append((time.perf_counter() - emitted_perf) * 1000.0)

    elapsed = time.perf_counter() - start
    total_windows = len(infer_ms)
    hz = total_windows / elapsed if elapsed > 0 else 0.0

    print(f"windows: {total_windows} in {elapsed:.2f}s ({hz:.2f} Hz)")
    if infer_ms:
        print(
            f"infer_ms: mean={mean(infer_ms):.2f} p50={sorted(infer_ms)[len(infer_ms) // 2]:.2f}"
        )
    if hop_ms:
        print(
            f"hop_ms: mean={mean(hop_ms):.2f} p50={sorted(hop_ms)[len(hop_ms) // 2]:.2f}"
        )
    if publish_age_ms:
        print(
            f"publish_age_ms: mean={mean(publish_age_ms):.2f} p50={sorted(publish_age_ms)[len(publish_age_ms) // 2]:.2f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
