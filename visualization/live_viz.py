from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - optional dependency
    pg = None

from PySide6.QtWidgets import QLabel


class LiveHiddenMagnitudePlot:
    def __init__(self) -> None:
        self.timestamps = deque(maxlen=512)
        self.values = deque(maxlen=512)
        if pg is None:
            self.widget = QLabel("pyqtgraph not available")
            self._curve = None
            return
        self.widget = pg.PlotWidget()
        self.widget.setTitle("LSTM Hidden Magnitude")
        self._curve = self.widget.plot(pen=pg.mkPen("c", width=2))

    def update(self, payload: Dict[str, Any]) -> None:
        if self._curve is None:
            return
        t = payload.get("t")
        hidden_mag = payload.get("hidden_mag")
        if t is None or hidden_mag is None:
            return
        self.timestamps.append(float(t))
        self.values.append(float(hidden_mag))
        self._curve.setData(list(self.timestamps), list(self.values))


def parse_viz_line(line: str) -> Optional[Dict[str, Any]]:
    if not line.startswith("VIZ "):
        return None
    try:
        parts = line.replace("VIZ ", "").strip().split()
        payload: Dict[str, Any] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, val = part.split("=", 1)
            payload[key] = float(val)
        return payload
    except Exception:
        return None
