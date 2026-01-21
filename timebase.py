"""Monotonic timebase mapping for LSL timestamps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TimebaseSnapshot:
    offset: float = 0.0
    has_anchor: bool = False
    prev_continuous_mono: Optional[float] = None
    last_discontinuity_mono: Optional[float] = None
    stream_start_continuous_mono: Optional[float] = None
    segment_id: int = 0
    eps: float = 1e-6


@dataclass
class TimebaseMapper:
    offset: float = 0.0
    has_anchor: bool = False
    prev_lsl_raw: Optional[float] = None
    prev_continuous_mono: Optional[float] = None
    last_discontinuity_mono: Optional[float] = None
    segment_id: int = 0

    BACKWARD_JUMP_S: float = 0.050
    FORWARD_JUMP_S: float = 1.000
    EPS: float = 1e-6

    def map(self, lsl_raw: float, now_mono: float) -> Tuple[float, float, bool, bool]:
        lsl_raw = float(lsl_raw)
        now_mono = float(now_mono)
        discontinuity = False
        if not self.has_anchor:
            self.offset = now_mono - lsl_raw
            self.has_anchor = True
        if self.prev_lsl_raw is not None:
            delta = lsl_raw - self.prev_lsl_raw
            if delta < -self.BACKWARD_JUMP_S or delta > self.FORWARD_JUMP_S:
                discontinuity = True
                self.offset = now_mono - lsl_raw
                self.segment_id += 1
                self.last_discontinuity_mono = now_mono
        mapped_mono = lsl_raw + self.offset
        if self.prev_continuous_mono is None:
            continuous_mono = mapped_mono
            clamped = False
        else:
            continuous_mono = max(self.prev_continuous_mono + self.EPS, mapped_mono)
            clamped = continuous_mono > mapped_mono + (self.EPS / 2.0)
        self.prev_lsl_raw = lsl_raw
        self.prev_continuous_mono = continuous_mono
        return continuous_mono, mapped_mono, discontinuity, clamped


def map_lsl_to_mono_snapshot(
    snapshot: TimebaseSnapshot, lsl_ts: Optional[float]
) -> Optional[float]:
    if lsl_ts is None:
        return None
    if not snapshot.has_anchor:
        return None
    mapped = float(lsl_ts) + float(snapshot.offset)
    if snapshot.prev_continuous_mono is None:
        return mapped
    return max(float(snapshot.prev_continuous_mono) + snapshot.eps, mapped)
