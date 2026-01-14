from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional, Tuple

import numpy as np

from demo_backend.utils_demo import ensure_repo_on_path

ensure_repo_on_path()

try:
    from utils.sequence_data import load_sequence_npz
except Exception:  # fallback
    load_sequence_npz = None

load_sequence_npz: Optional[Callable[[str | Path, str | None], object]]


@dataclass
class ReplayMeta:
    subject_id: str
    experiment_hash: str
    window_start_s: float
    window_end_s: float
    index: int
    timebase_version: str


class ReplaySource:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._load()

    def _load(self) -> None:
        if load_sequence_npz is not None:
            X, y_action, y_finger, meta = load_sequence_npz(self.path)
        else:
            data = np.load(self.path, allow_pickle=True)
            X = data["X"].astype(np.float32)
            y_action = data["y_action"].astype(np.int64)
            y_finger = data["y_finger"].astype(np.int64)
            meta = {
                k: data[k] for k in data.files if k not in {"X", "y_action", "y_finger"}
            }

        self.X = X
        self.y_action = y_action
        self.y_finger = y_finger
        self.meta = meta

        self.subject_ids = meta.get("subject_id")
        self.experiment_hashes = meta.get("experiment_hash")
        self.window_start = meta.get("window_start")
        self.window_end = meta.get("window_end")
        self.timebase_version = meta.get("timebase_version")

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def iter_windows(self) -> Iterator[Tuple[np.ndarray, ReplayMeta]]:
        n = self.X.shape[0]
        for idx in range(n):
            subject_id = "UNKNOWN"
            exp_hash = "UNKNOWN"
            if self.subject_ids is not None:
                subject_id = str(self.subject_ids[idx])
            if self.experiment_hashes is not None:
                exp_hash = str(self.experiment_hashes[idx])

            window_start = (
                float(self.window_start[idx])
                if self.window_start is not None
                else float(idx)
            )
            window_end = (
                float(self.window_end[idx])
                if self.window_end is not None
                else float(idx)
            )
            timebase_version = "absolute_v1"
            if self.timebase_version is not None:
                try:
                    timebase_version = str(self.timebase_version[idx])
                except Exception:
                    timebase_version = str(self.timebase_version)

            yield (
                self.X[idx],
                ReplayMeta(
                    subject_id=subject_id,
                    experiment_hash=exp_hash,
                    window_start_s=window_start,
                    window_end_s=window_end,
                    index=idx,
                    timebase_version=timebase_version,
                ),
            )
