from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

PROJECTS_DIR = Path("Projects")


@dataclass
class SubjectInfo:
    subject_id: str
    handedness: str = "Unknown"
    age: Optional[int] = None
    notes: str = ""


def list_projects(root: Path = PROJECTS_DIR) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def project_root(project_name: str) -> Path:
    return PROJECTS_DIR / project_name


def ensure_project(project_name: str) -> Path:
    root = project_root(project_name)
    (root / "subjects").mkdir(parents=True, exist_ok=True)
    return root


def list_subjects(project_name: str) -> List[str]:
    root = project_root(project_name) / "subjects"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def subject_root(project_name: str, subject_id: str) -> Path:
    return project_root(project_name) / "subjects" / subject_id


def ensure_subject_dirs(subject_dir: Path) -> None:
    for name in [
        "config",
        "raw",
        "features",
        "events",
        "windows",
        "models",
        "exports",
        "logs",
        "sessions",
    ]:
        (subject_dir / name).mkdir(parents=True, exist_ok=True)


def subject_meta_path(subject_dir: Path) -> Path:
    return subject_dir / "subject.json"


def session_backend_id(timestamp: Optional[datetime] = None) -> str:
    ts = timestamp or datetime.now()
    return ts.strftime("%Y%m%d_%H%M%S")


def ui_session_id(subject_id: str, backend_session_id: str) -> str:
    return f"{subject_id}_{backend_session_id}"


def session_root(subject_dir: Path, ui_session_id_value: str) -> Path:
    return subject_dir / "sessions" / ui_session_id_value


def safe_child(parent: Path, name: str) -> Path:
    """Return a child path without creating it."""
    return parent / name


def ensure_session_dirs(session_dir: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    for name in ["events", "windows", "logs", "exports", "config", "features", "raw", "models"]:
        (session_dir / name).mkdir(parents=True, exist_ok=True)


def next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for i in range(2, 1000):
        candidate = path.with_name(f"{stem}_v{i}{suffix}")
        if not candidate.exists():
            return candidate
    return path
