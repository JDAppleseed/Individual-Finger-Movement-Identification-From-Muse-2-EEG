from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Sequence, Tuple, List

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
    def __init__(
        self, fs: int = 256, window_sec: float = 0.25, step_sec: float = 0.05
    ) -> None:
        self.fs = fs
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.window_samples = int(fs * window_sec)
        self.step_samples = max(1, int(fs * step_sec))
        self.inlet: Optional[StreamInlet] = None
        self.channel_indices: Optional[List[int]] = None
        self.status_message = ""
        self.buffer: Deque[Sequence[float]] = deque(maxlen=self.window_samples)
        self.sample_times: Deque[float] = deque(maxlen=self.window_samples)
        self._last_emit_idx = 0
        self._sample_count = 0
        self._stream_start: Optional[float] = None

    def connect(self) -> Tuple[bool, str]:
        if not LSL_AVAILABLE:
            return False, "pylsl not installed"
        streams = resolve_streams()
        if not streams:
            return False, "No LSL streams found"
        eeg_stream = None
        for stream in streams:
            name = stream.name().lower()
            if "eeg" in name:
                eeg_stream = stream
                break

        if eeg_stream is None:
            return False, "no eeg stream found"

        stream = eeg_stream
        if stream.channel_count() < 4:
            return (
                False,
                f"EEG stream has {stream.channel_count()} channels; expected at least 4",
            )

        self.inlet = StreamInlet(stream)
        self.channel_indices = None

        channel_note = "channels=first4"
        try:
            info = stream.info()
            desc = info.desc()
            ch = desc.child("channels").child("channel")
            labels = []
            while ch and ch.name():
                label = ch.child_value("label")
                if label:
                    labels.append(label.strip())
                ch = ch.next_sibling()
            if labels:
                labels_lower = [label.lower() for label in labels]
                wanted = ["tp9", "af7", "af8", "tp10"]
                indices = []
                for name in wanted:
                    if name in labels_lower:
                        indices.append(labels_lower.index(name))
                if len(indices) == 4:
                    self.channel_indices = indices
                    channel_note = "channels=TP9,AF7,AF8,TP10"
        except Exception:
            self.channel_indices = None

        self.status_message = (
            f"Connected to LSL stream: {stream.name()} ({channel_note})"
        )
        return True, self.status_message

    def pull_window(self) -> Optional[LiveWindow]:
        if self.inlet is None:
            return None
        last_window = None
        while True:
            sample, ts = self.inlet.pull_sample(timeout=0.0)
            if sample is None:
                break
            if self._stream_start is None:
                self._stream_start = ts

            if len(sample) < 4:
                continue
            if self.channel_indices is not None:
                sample = [sample[i] for i in self.channel_indices]
            else:
                sample = sample[:4]
            self.buffer.append(sample)
            self.sample_times.append(ts)

            if len(self.buffer) < self.window_samples:
                continue

            self._sample_count += 1

            # Emit every step samples
            if (self._sample_count - self._last_emit_idx) < self.step_samples:
                continue
            self._last_emit_idx = self._sample_count

            window = np.array(self.buffer, dtype=np.float32)
            start_s = float(self.sample_times[0] - self._stream_start)
            end_s = float(self.sample_times[-1] - self._stream_start)

            last_window = LiveWindow(
                window=window,
                window_start_s=start_s,
                window_end_s=end_s,
            )
        return last_window
