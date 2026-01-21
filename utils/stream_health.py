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


@dataclass
class RollingHealthDecision:
    healthy: bool
    reason: Optional[str]
    measured_fs: Optional[float]
    write_rate: float
    event_allowed: bool
    queue_size: int
    queue_label: Optional[str]
    backwards_count: int
    last_received_lsl_ts: Optional[float]
    last_written_lsl_ts: Optional[float]


class RollingStreamHealthGate:
    def __init__(
        self,
        *,
        expected_fs: float,
        health_window_s: float,
        stall_s: float,
        max_queue: int,
        backlog_grace_s: float,
        recovery_s: float,
        backwards_threshold: int,
        backwards_window_s: float,
        gap_threshold_s: float,
        gap_count_threshold: int,
        gap_window_s: float,
    ) -> None:
        self.expected_fs = float(expected_fs)
        self.health_window_s = float(health_window_s)
        self.stall_s = float(stall_s)
        self.max_queue = int(max_queue)
        self.backlog_grace_s = float(backlog_grace_s)
        self.recovery_s = float(recovery_s)
        self.backwards_threshold = int(backwards_threshold)
        self.backwards_window_s = float(backwards_window_s)
        self.gap_threshold_s = float(gap_threshold_s)
        self.gap_count_threshold = int(gap_count_threshold)
        self.gap_window_s = float(gap_window_s)

        self._received = []
        self._written = []
        self._backwards = []
        self._gaps = []
        self._queue_size = 0
        self._queue_label: Optional[str] = None
        self._last_received_mono: Optional[float] = None
        self._last_received_lsl_ts: Optional[float] = None
        self._last_written_lsl_ts: Optional[float] = None
        self._event_allowed = False
        self._healthy_since: Optional[float] = None
        self._backlog_since: Optional[float] = None

    def _trim(self, now_monotonic: float, window_s: float, items: list) -> None:
        cutoff = float(now_monotonic - window_s)
        while items and items[0][0] < cutoff:
            items.pop(0)

    def record_received(self, lsl_ts: float, now_monotonic: float) -> None:
        """Record a received sample time using the monotonic domain."""
        if self._last_received_lsl_ts is not None and lsl_ts <= self._last_received_lsl_ts:
            self._backwards.append((float(now_monotonic), float(lsl_ts)))
        if self._last_received_lsl_ts is not None:
            dt = float(lsl_ts - self._last_received_lsl_ts)
            if dt > self.gap_threshold_s:
                self._gaps.append((float(now_monotonic), float(dt)))
        self._last_received_mono = float(now_monotonic)
        self._last_received_lsl_ts = float(lsl_ts)
        self._received.append((float(now_monotonic), float(lsl_ts)))

    def record_written(self, lsl_ts: float, now_monotonic: float) -> None:
        """Record a written sample time using the monotonic domain."""
        self._last_written_lsl_ts = float(lsl_ts)
        self._written.append((float(now_monotonic), float(lsl_ts)))

    def set_queue_size(self, size: int, *, label: Optional[str] = None) -> None:
        self._queue_size = int(size)
        if label is not None:
            self._queue_label = str(label)

    def mark_backlog_overflow(self, now_monotonic: float) -> None:
        if self._backlog_since is None:
            self._backlog_since = float(now_monotonic) - float(self.backlog_grace_s)

    def evaluate(self, now_monotonic: float) -> RollingHealthDecision:
        now = float(now_monotonic)
        self._trim(now, self.health_window_s, self._received)
        self._trim(now, self.health_window_s, self._written)
        self._trim(now, self.backwards_window_s, self._backwards)
        self._trim(now, self.gap_window_s, self._gaps)

        if self._last_received_mono is None or (now - self._last_received_mono) > self.stall_s:
            healthy = False
            reason = "lsl_starvation"
            measured_fs = None
            write_rate = 0.0
        else:
            measured_fs = self._compute_measured_fs()
            write_rate = (
                len(self._written) / self.health_window_s
                if self.health_window_s > 0
                else 0.0
            )
            reason = None
            healthy = True
            warn_queue = max(1, int(self.max_queue * 0.8))
            fail_queue = max(1, int(self.max_queue))
            if self._queue_size > warn_queue:
                if self._backlog_since is None:
                    self._backlog_since = now
                if (now - self._backlog_since) >= self.backlog_grace_s:
                    if self._queue_size > fail_queue:
                        healthy = False
                        reason = "raw_backlog_high"
                    else:
                        reason = "raw_backlog_warn"
                else:
                    reason = None
            else:
                self._backlog_since = None
            if healthy and reason == "raw_backlog_warn":
                healthy = True
            if healthy and len(self._gaps) >= self.gap_count_threshold:
                healthy = False
                reason = "lsl_timestamp_gaps"
            if healthy and len(self._backwards) >= self.backwards_threshold:
                healthy = False
                reason = "lsl_timestamp_backwards"
        if healthy:
            if self._healthy_since is None:
                self._healthy_since = now
            if now - self._healthy_since >= self.recovery_s:
                self._event_allowed = True
        else:
            self._event_allowed = False
            self._healthy_since = None

        return RollingHealthDecision(
            healthy=healthy,
            reason=reason,
            measured_fs=measured_fs,
            write_rate=float(write_rate),
            event_allowed=self._event_allowed,
            queue_size=self._queue_size,
            queue_label=self._queue_label,
            backwards_count=len(self._backwards),
            last_received_lsl_ts=self._last_received_lsl_ts,
            last_written_lsl_ts=self._last_written_lsl_ts,
        )

    def _compute_measured_fs(self) -> Optional[float]:
        if len(self._received) < 2:
            return None
        diffs = []
        for idx in range(1, len(self._received)):
            dt = self._received[idx][1] - self._received[idx - 1][1]
            if dt > 0:
                diffs.append(dt)
        if not diffs:
            return None
        diffs.sort()
        mid = diffs[len(diffs) // 2]
        if mid <= 0:
            return None
        return float(1.0 / mid)
