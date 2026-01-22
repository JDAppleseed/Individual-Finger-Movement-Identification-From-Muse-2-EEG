from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional, Tuple


@dataclass
class StreamClock:
    # stream timebase, in ms, derived from window_end_s
    last_stream_ms: Optional[int] = None
    # perf clock at the moment last_stream_ms was observed
    last_perf_s: Optional[float] = None

    def reset(self) -> None:
        self.last_stream_ms = None
        self.last_perf_s = None

    def update_from_window_end_s(
        self, window_end_s: float, perf_s: Optional[float] = None
    ) -> int:
        perf_s = time.perf_counter() if perf_s is None else perf_s
        stream_ms = int(window_end_s * 1000.0)
        self.last_stream_ms = stream_ms
        self.last_perf_s = perf_s
        return stream_ms

    def estimate_stream_ms_now(self, perf_s: Optional[float] = None) -> Optional[int]:
        # Used only when LSL/windows stall; extrapolate from last known stream_ms using perf delta.
        if self.last_stream_ms is None or self.last_perf_s is None:
            return None
        perf_s = time.perf_counter() if perf_s is None else perf_s
        delta_ms = int((perf_s - self.last_perf_s) * 1000.0)
        return self.last_stream_ms + max(0, delta_ms)


def clamp_monotonic_window(
    prev_end_s: Optional[float],
    start_s: float,
    end_s: float,
    eps: float = 1e-6,
) -> Tuple[float, float, bool]:
    """
    Ensure window time is monotonic. If end_s goes backwards, shift the window forward.
    Returns (start_s, end_s, clamped).
    """
    if prev_end_s is None:
        return start_s, end_s, False
    if end_s >= prev_end_s - eps:
        return start_s, end_s, False
    delta = (prev_end_s + eps) - end_s
    return start_s + delta, end_s + delta, True


def clamp_monotonic_time(
    prev_s: Optional[float],
    current_s: Optional[float],
    epsilon_s: float = 1e-6,
) -> Tuple[Optional[float], bool]:
    if current_s is None:
        return None, False
    if prev_s is None:
        return current_s, False
    if current_s + epsilon_s < prev_s:
        return prev_s, True
    return current_s, False
