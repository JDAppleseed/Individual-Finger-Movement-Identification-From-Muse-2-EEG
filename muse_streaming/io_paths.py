from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class SessionPaths:
    session_id: str
    raw_path: Path
    features_path: Path
    events_path: Path


@dataclass(frozen=True)
class SessionDirPaths:
    session_id: str
    session_dir: Path
    raw_dir: Path
    events_dir: Path
    meta_path: Path
    manifest_path: Path
    timebase_report_path: Path


def default_processed_dir() -> Path:
    return Path(os.environ.get("MUSE_PROCESSED_DIR", "data/processed"))


def default_raw_dir() -> Path:
    return Path(os.environ.get("MUSE_RAW_DIR", "data/raw"))


def _now_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _csv_has_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            row = next(reader, None)
            return row is not None
    except Exception:
        return False


def _paths_exist(paths: Iterable[Path]) -> bool:
    return any(path.exists() for path in paths)


def _ensure_unique_session_id(output_root: Path, subject_id: str, session_id: str) -> str:
    suffix = 0
    candidate = session_id
    while _paths_exist(
        [
            output_root / f"{subject_id}_{candidate}_raw.csv",
            output_root / f"{subject_id}_{candidate}_features.csv",
            output_root / f"{subject_id}_{candidate}_events.csv",
        ]
    ):
        suffix += 1
        candidate = f"{session_id}_{suffix:02d}"
    return candidate


def _ensure_unique_session_dir(output_root: Path, subject_id: str, session_id: str) -> str:
    suffix = 0
    candidate = session_id
    while (output_root / f"{subject_id}_{candidate}").exists():
        suffix += 1
        candidate = f"{session_id}_{suffix:02d}"
    return candidate


def build_session_dir_paths(output_root: Path, subject_id: str, session_id: str) -> SessionDirPaths:
    session_dir = output_root / f"{subject_id}_{session_id}"
    raw_dir = session_dir / "raw"
    events_dir = session_dir / "events"
    return SessionDirPaths(
        session_id=session_id,
        session_dir=session_dir,
        raw_dir=raw_dir,
        events_dir=events_dir,
        meta_path=session_dir / "meta.json",
        manifest_path=session_dir / "manifest.json",
        timebase_report_path=session_dir / "timebase_report.json",
    )


def build_session_paths(output_root: Path, subject_id: str, session_id: str) -> SessionPaths:
    return SessionPaths(
        session_id=session_id,
        raw_path=output_root / f"{subject_id}_{session_id}_raw.csv",
        features_path=output_root / f"{subject_id}_{session_id}_features.csv",
        events_path=output_root / f"{subject_id}_{session_id}_events.csv",
    )


def prepare_session_paths(
    *,
    output_root: Path,
    subject_id: str,
    session_id: Optional[str],
    resume: bool,
) -> tuple[SessionPaths, bool, str]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reason = "new_session"
    if session_id is None:
        session_id = _now_session_id()
    candidate = build_session_paths(output_root, subject_id, session_id)

    if resume:
        if _csv_has_rows(candidate.features_path):
            reason = "resume"
            return candidate, True, reason
        reason = "resume_blocked_missing_features"

    unique_session_id = _ensure_unique_session_id(output_root, subject_id, session_id)
    if unique_session_id != session_id:
        reason = f"new_session_collision({session_id})"
    return build_session_paths(output_root, subject_id, unique_session_id), False, reason


def prepare_session_dir_paths(
    *,
    output_root: Path,
    subject_id: str,
    session_id: Optional[str],
    resume: bool,
) -> tuple[SessionDirPaths, bool, str]:
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    reason = "new_session"
    if session_id is None:
        session_id = _now_session_id()
    candidate = build_session_dir_paths(output_root, subject_id, session_id)

    if resume and candidate.session_dir.exists():
        reason = "resume"
        return candidate, True, reason

    unique_session_id = _ensure_unique_session_dir(output_root, subject_id, session_id)
    if unique_session_id != session_id:
        reason = f"new_session_collision({session_id})"
    return build_session_dir_paths(output_root, subject_id, unique_session_id), False, reason
