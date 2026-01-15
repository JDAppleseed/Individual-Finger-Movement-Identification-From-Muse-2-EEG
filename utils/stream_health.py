from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class StreamHealthStatus:
    active: bool
    stalled_reason: Optional[str]
    last_write_utc: Optional[str]


class StreamHealthMonitor:
    def __init__(self, timeout_s: float, *, reason: str = "no_csv_updates") -> None:
        self.timeout_s = float(timeout_s)
        self.reason = reason
        self.last_write_monotonic: Optional[float] = None
        self.last_write_utc: Optional[str] = None

    def mark_write(self, now_monotonic: float, now_utc: str) -> None:
        self.last_write_monotonic = float(now_monotonic)
        self.last_write_utc = now_utc

    def check(self, now_monotonic: float) -> StreamHealthStatus:
        if self.last_write_monotonic is None:
            return StreamHealthStatus(False, self.reason, self.last_write_utc)
        elapsed = float(now_monotonic - self.last_write_monotonic)
        if elapsed > self.timeout_s:
            return StreamHealthStatus(False, self.reason, self.last_write_utc)
        return StreamHealthStatus(True, None, self.last_write_utc)
