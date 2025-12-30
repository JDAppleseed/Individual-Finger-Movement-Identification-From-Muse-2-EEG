from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

try:
    from pylsl import StreamInlet, resolve_streams, local_clock
    LSL_AVAILABLE = True
except Exception:
    StreamInlet = None
    resolve_streams = None
    local_clock = None
    LSL_AVAILABLE = False


@dataclass
class LiveWindow:
    window: np.ndarray
    window_start_s: float
    window_end_s: float


class LiveLSLSource:
    def __init__(self, fs: int = 256, window_sec: float = 0.25, step_sec: float = 0.05) -> None:
        self.fs = fs
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.window_samples = int(fs * window_sec)
        self.step_samples = max(1, int(fs * step_sec))
        self.inlet = None
        self.buffer = deque(maxlen=self.window_samples)
        self.sample_times = deque(maxlen=self.window_samples)
        self._last_emit_idx = 0
        self._stream_start = None

    def connect(self) -> Tuple[bool, str]:
        if not LSL_AVAILABLE:
            return False, "pylsl not installed"
        streams = resolve_streams()
        if not streams:
            return False, "No LSL streams found"
        self.inlet = StreamInlet(streams[0])
        return True, f"Connected to LSL stream: {streams[0].name()}"

    def pull_window(self) -> Optional[LiveWindow]:
        if self.inlet is None:
            return None

        sample, ts = self.inlet.pull_sample(timeout=0.0)
        if sample is None:
            return None

        if self._stream_start is None:
            self._stream_start = ts

        self.buffer.append(sample[:4])
        self.sample_times.append(ts)

        if len(self.buffer) < self.window_samples:
            return None

        # Emit every step samples
        if (self._last_emit_idx + self.step_samples) > len(self.sample_times):
            return None
        self._last_emit_idx = len(self.sample_times)

        window = np.array(self.buffer, dtype=np.float32)
        start_s = float(self.sample_times[0] - self._stream_start)
        end_s = float(self.sample_times[-1] - self._stream_start)

        return LiveWindow(window=window, window_start_s=start_s, window_end_s=end_s)
