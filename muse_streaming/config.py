from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional

DEFAULT_STREAM_NAME = "Muse2-EEG"
DEFAULT_STREAM_TYPE = "EEG"
DEFAULT_NOMINAL_SRATE = 256.0
DEFAULT_LABELS = ["TP9", "AF7", "AF8", "TP10"]
DEFAULT_TARGET_FS = 256.0
DEFAULT_WINDOW_SEC = 0.25
DEFAULT_WINDOW_HOP_SEC = 0.05
DEFAULT_OUTPUT_ROOT = Path("data")


@dataclass(frozen=True)
class StreamSettings:
    name: str = DEFAULT_STREAM_NAME
    stype: str = DEFAULT_STREAM_TYPE
    nominal_srate: float = DEFAULT_NOMINAL_SRATE
    labels: List[str] = field(default_factory=lambda: list(DEFAULT_LABELS))


@dataclass(frozen=True)
class RecorderSettings:
    output_root: Path = DEFAULT_OUTPUT_ROOT
    subject_id: str = "unknown"
    session_id: Optional[str] = None
    resume: bool = False
    target_fs: float = DEFAULT_TARGET_FS
    window_sec: float = DEFAULT_WINDOW_SEC
    window_hop_sec: float = DEFAULT_WINDOW_HOP_SEC
    events_enabled: bool = True


@dataclass(frozen=True)
class LoggingSettings:
    log_level: str = "INFO"
    session_id: Optional[str] = None


@dataclass(frozen=True)
class PipelineSettings:
    stream: StreamSettings = field(default_factory=StreamSettings)
    recorder: RecorderSettings = field(default_factory=RecorderSettings)
    logging: LoggingSettings = field(default_factory=LoggingSettings)
    sim: bool = False


def parse_labels(label_csv: str) -> List[str]:
    return [label.strip() for label in label_csv.split(",") if label.strip()]


class SessionLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        session_id = self.extra.get("session_id")
        prefix = f"session_id={session_id} " if session_id else ""
        return prefix + msg, kwargs


def configure_logging(settings: LoggingSettings) -> logging.LoggerAdapter:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    base_logger = logging.getLogger("muse_streaming")
    return SessionLoggerAdapter(base_logger, {"session_id": settings.session_id})


def ensure_output_root(path: Path) -> Path:
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
