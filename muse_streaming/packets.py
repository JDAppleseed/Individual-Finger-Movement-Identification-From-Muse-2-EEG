from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass(frozen=True, slots=True)
class SamplePacket:
    seq: int
    lsl_ts_raw: float
    lsl_ts_mono: float
    local_ts: float
    sample: np.ndarray
    flags: int
    segment_id: int
    clamped: bool
    raw_path: Optional[Path] = None
    segment_break_reason: Optional[str] = None
