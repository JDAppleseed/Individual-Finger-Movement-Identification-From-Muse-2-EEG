from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class StreamRequirements:
    required_labels: list[str]
    require_exact_channels: bool
    expected_channels: int


@dataclass
class HardStopPolicy:
    hard_stop_after_unhealthy_s: float
    failed_write_window_s: float
    failed_dir: str
    hard_stop_exit_code: int


@dataclass
class FailedWriters:
    features_file: Optional[object] = None
    features_writer: Optional[object] = None
    raw_file: Optional[object] = None
    raw_writer: Optional[object] = None
    preds_file: Optional[object] = None
    preds_writer: Optional[object] = None
    events_path: Optional[Path] = None

    def is_open(self) -> bool:
        return self.features_writer is not None or self.raw_writer is not None

    def open_failed_files(
        self,
        prefix: str,
        *,
        headers: list[str],
        save_raw: bool,
        save_preds: bool,
        failed_dir: str,
        prediction_header: list[str],
        raw_header: list[str],
    ) -> None:
        root = Path(failed_dir)
        root.mkdir(parents=True, exist_ok=True)
        features_path = root / f"{prefix}_eeg_features.csv"
        preds_path = root / f"{prefix}_predictions.csv"
        raw_path = root / f"{prefix}_raw.csv"
        self.events_path = root / f"{prefix}_events.csv"

        self.features_file = features_path.open("w", newline="")
        import csv

        self.features_writer = csv.writer(self.features_file)
        self.features_writer.writerow(headers)

        if save_preds:
            self.preds_file = preds_path.open("w", newline="")
            self.preds_writer = csv.writer(self.preds_file)
            self.preds_writer.writerow(prediction_header)
        if save_raw:
            self.raw_file = raw_path.open("w", newline="")
            self.raw_writer = csv.writer(self.raw_file)
            self.raw_writer.writerow(raw_header)

    def close_failed_files(self) -> None:
        for file_obj in (self.features_file, self.raw_file, self.preds_file):
            if file_obj:
                try:
                    file_obj.flush()
                    file_obj.close()
                except Exception:
                    pass
        self.features_file = None
        self.features_writer = None
        self.raw_file = None
        self.raw_writer = None
        self.preds_file = None
        self.preds_writer = None
        self.events_path = None


@dataclass
class HealthStopState:
    unhealthy_since_mono: Optional[float] = None
    failed_write_until_mono: Optional[float] = None
    hard_stop_triggered: bool = False
    hard_stop_report_path: Optional[Path] = None
    label_check_status: Optional[dict] = None
    has_health_decision: bool = False
    last_health_reason: Optional[str] = None
