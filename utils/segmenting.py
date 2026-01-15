from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SegmentBreak:
    should_break: bool
    reason: Optional[str]


class SegmentBreaker:
    def __init__(self, gap_break_s: float) -> None:
        self.gap_break_s = float(gap_break_s)
        self.last_lsl_ts: Optional[float] = None

    def check(self, lsl_ts: float) -> SegmentBreak:
        if self.last_lsl_ts is None:
            self.last_lsl_ts = float(lsl_ts)
            return SegmentBreak(False, None)
        if lsl_ts <= self.last_lsl_ts:
            self.last_lsl_ts = float(lsl_ts)
            return SegmentBreak(True, "backwards")
        gap = float(lsl_ts - self.last_lsl_ts)
        if gap > self.gap_break_s:
            self.last_lsl_ts = float(lsl_ts)
            return SegmentBreak(True, "gap")
        self.last_lsl_ts = float(lsl_ts)
        return SegmentBreak(False, None)

    def reset(self) -> None:
        self.last_lsl_ts = None
