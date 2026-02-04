from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


@dataclass(frozen=True)
class SessionLayout:
    session_dir: Path

    @property
    def raw_dir(self) -> Path:
        return self.session_dir / "raw"

    @property
    def events_dir(self) -> Path:
        return self.session_dir / "events"

    @property
    def processed_dir(self) -> Path:
        return self.session_dir / "processed"

    @property
    def logs_dir(self) -> Path:
        return self.session_dir / "logs"

    @property
    def windows_npz(self) -> Path:
        return self.processed_dir / "eeg_windows.npz"

    @property
    def windows_csv(self) -> Path:
        return self.processed_dir / "eeg_windows.csv"

    @property
    def models_root(self) -> Path:
        return self.processed_dir / "models"

    @property
    def reports_root(self) -> Path:
        return self.processed_dir / "reports"


def _latest_dir_by_mtime(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def resolve_session_dir(session_dir: Union[str, Path]) -> Path:
    p = Path(session_dir).expanduser()
    return p.resolve() if p.exists() else p


def resolve_latest_run_dir(session_dir: Path) -> Optional[Path]:
    return _latest_dir_by_mtime(SessionLayout(session_dir).models_root)
