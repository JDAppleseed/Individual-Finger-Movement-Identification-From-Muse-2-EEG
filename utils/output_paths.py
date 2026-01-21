from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple


def _normalize_path(value: Optional[object]) -> Optional[Path]:
    if value is None:
        return None
    if isinstance(value, Path):
        path = value
    else:
        text = str(value).strip()
        if not text:
            return None
        path = Path(text)
    return path


def _resolve_repo_path(path: Path, repo_root: Path) -> Path:
    base = path.expanduser()
    if base.is_absolute():
        return base.resolve()
    return (repo_root / base).resolve()


def _first_config_path(
    settings: Mapping[str, object], keys: Iterable[str]
) -> Optional[Path]:
    for key in keys:
        if key not in settings:
            continue
        candidate = _normalize_path(settings.get(key))
        if candidate is not None:
            return candidate
    return None


def _derive_project_subject(
    config_payload: Mapping[str, object], config_path: Optional[Path]
) -> Tuple[Optional[str], Optional[str]]:
    project_name = config_payload.get("project_name")
    subject_id = config_payload.get("subject_id")
    if project_name and subject_id:
        return str(project_name), str(subject_id)
    if config_path is None:
        return None, None
    parts = config_path.resolve().parts
    for idx in range(len(parts) - 3):
        if parts[idx] == "Projects" and parts[idx + 2] == "subjects":
            return parts[idx + 1], parts[idx + 3]
    return None, None


def _derive_session_ui(
    subject_id: Optional[str],
    session_id_hint: Optional[str],
    config_payload: Mapping[str, object],
) -> Optional[str]:
    session_ui = config_payload.get("session_id")
    if session_ui:
        return str(session_ui)
    if not subject_id or not session_id_hint:
        return None
    subject_id = str(subject_id)
    session_id_hint = str(session_id_hint)
    if session_id_hint.startswith(f"{subject_id}_"):
        return session_id_hint
    return f"{subject_id}_{session_id_hint}"


def resolve_output_dir(
    *,
    kind: str,
    cli_value: Optional[object],
    config_settings: Mapping[str, object],
    config_payload: Optional[Mapping[str, object]],
    config_path: Optional[Path],
    repo_root: Path,
    subject_id: Optional[str],
    session_id_hint: Optional[str],
    default_base: Path,
    session_id_seed: Optional[str],
    resume_path: Optional[Path] = None,
    create: bool = True,
) -> Path:
    if kind not in {"processed", "raw"}:
        raise ValueError(f"Unsupported output kind: {kind}")

    if resume_path is not None:
        resolved = resume_path.expanduser().resolve().parent
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    cli_path = _normalize_path(cli_value)
    if cli_path is not None:
        resolved = _resolve_repo_path(cli_path, repo_root)
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    config_payload = config_payload or {}
    config_keys = ["processed_dir", "processed_path", "output_dir"]
    if kind == "raw":
        config_keys = ["raw_dir", "raw_path", "raw_output_dir"]
    config_path_value = _first_config_path(config_settings, config_keys)
    if config_path_value is not None:
        resolved = _resolve_repo_path(config_path_value, repo_root)
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    project_name, config_subject_id = _derive_project_subject(
        config_payload, config_path
    )
    session_ui = _derive_session_ui(config_subject_id, session_id_hint, config_payload)
    if project_name and config_subject_id and session_ui:
        session_root = (
            repo_root
            / "Projects"
            / project_name
            / "subjects"
            / config_subject_id
            / "sessions"
            / session_ui
        )
        resolved = session_root / ("processed" if kind == "processed" else "raw")
        if create:
            resolved.mkdir(parents=True, exist_ok=True)
        return resolved

    base = _resolve_repo_path(default_base, repo_root)
    subject = str(subject_id or "unknown")
    session = str(session_id_hint or session_id_seed or "session")
    resolved = base / subject / session
    if create:
        resolved.mkdir(parents=True, exist_ok=True)
    return resolved
