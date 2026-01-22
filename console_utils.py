"""Console logging helpers with character capping and rate limiting."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

CONSOLE_CAP_CHARS = 200_000
CONSOLE_RING_CHARS = 50_000

_CAP_WARNING = (
    "[console] output capped; further console output suppressed (cap={cap})"
)


@dataclass
class ConsoleBudget:
    cap_chars: int
    ring_chars: int
    emitted_chars: int = 0
    warned: bool = False
    ring: Deque[str] = field(default_factory=deque)
    _ring_chars: int = 0

    def _add_to_ring(self, text: str) -> None:
        if self.ring_chars <= 0:
            return
        if not text:
            return
        self.ring.append(text)
        self._ring_chars += len(text)
        while self._ring_chars > self.ring_chars and self.ring:
            removed = self.ring.popleft()
            self._ring_chars -= len(removed)

    def allow(self, text: str) -> bool:
        self._add_to_ring(text)
        if self.cap_chars <= 0:
            return True
        next_total = self.emitted_chars + len(text)
        if next_total > self.cap_chars:
            self.emitted_chars = self.cap_chars
            return False
        self.emitted_chars = next_total
        return True


class CappedStreamHandler(logging.StreamHandler):
    def __init__(self, budget: ConsoleBudget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.budget = budget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            msg = msg + self.terminator
            if self.budget.allow(msg):
                self.stream.write(msg)
                self.flush()
                return
            if not self.budget.warned:
                self.budget.warned = True
                warning = _CAP_WARNING.format(cap=self.budget.cap_chars)
                self.budget._add_to_ring(warning + self.terminator)
                self.stream.write(warning + self.terminator)
                self.flush()
        except Exception:
            self.handleError(record)


_console_budget: Optional[ConsoleBudget] = None
_rate_lock = threading.Lock()
_last_logged: Dict[str, float] = {}
_burst_windows: Dict[str, Deque[float]] = {}


def reset_console_budget() -> None:
    global _console_budget
    if _console_budget is None:
        return
    _console_budget.emitted_chars = 0
    _console_budget.warned = False
    _console_budget.ring.clear()
    _console_budget._ring_chars = 0


def install_console_logging(
    cap_chars: int = CONSOLE_CAP_CHARS,
    *,
    ring_chars: int = CONSOLE_RING_CHARS,
    level: int = logging.INFO,
) -> logging.Logger:
    global _console_budget
    budget_changed = False
    if (
        _console_budget is None
        or _console_budget.cap_chars != cap_chars
        or _console_budget.ring_chars != ring_chars
    ):
        _console_budget = ConsoleBudget(
            cap_chars=int(cap_chars),
            ring_chars=int(ring_chars),
        )
        budget_changed = True
    logger = logging.getLogger("muse_stream")
    logger.setLevel(level)
    handlers = [h for h in logger.handlers if isinstance(h, CappedStreamHandler)]
    if handlers and budget_changed and _console_budget is not None:
        for handler in handlers:
            handler.budget = _console_budget
    if not handlers:
        handler = CappedStreamHandler(_console_budget)
        formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def log_every(key: str, interval_s: float, level: int, msg: str, *args) -> None:
    now = time.monotonic()
    with _rate_lock:
        last = _last_logged.get(key)
        if last is not None and (now - last) < float(interval_s):
            return
        _last_logged[key] = now
    logging.getLogger("muse_stream").log(level, msg, *args)


def log_burst(
    key: str,
    max_per_window: int,
    window_s: float,
    level: int,
    msg: str,
    *args,
) -> None:
    now = time.monotonic()
    with _rate_lock:
        window = _burst_windows.setdefault(key, deque())
        cutoff = now - float(window_s)
        while window and window[0] < cutoff:
            window.popleft()
        if len(window) >= int(max_per_window):
            return
        window.append(now)
    logging.getLogger("muse_stream").log(level, msg, *args)


def get_console_tail(max_chars: int) -> str:
    if _console_budget is None:
        return ""
    if max_chars <= 0:
        return ""
    text = "".join(_console_budget.ring)
    if len(text) <= max_chars:
        return text
    return text[-int(max_chars) :]
