from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.session_layout import SessionLayout, resolve_latest_run_dir


def _normalize_existing_path(value: str) -> Optional[Path]:
    text = value.strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.exists():
        return None
    try:
        return path.resolve()
    except Exception:
        return path


def _infer_session_dir_from_artifact(path: Optional[Path]) -> Optional[Path]:
    if path is None:
        return None
    parts = path.parts
    if path.name == "eeg_windows.npz" and path.parent.name == "processed":
        return path.parent.parent
    if (
        path.name == "eeg_windows.npz"
        and path.parent.name == "windows"
        and path.parent.parent.exists()
    ):
        return path.parent.parent
    if (
        path.parent.name == "models"
        and path.name == "models"
        and path.exists()
    ):
        return path.parent.parent
    if path.parent.parent.name == "models":
        processed_dir = path.parent.parent.parent
        if processed_dir.name == "processed":
            return processed_dir.parent
    return None


def _session_has_replay_artifacts(session_dir: Path) -> bool:
    layout = SessionLayout(session_dir)
    has_npz = layout.windows_npz.exists() or (session_dir / "windows" / "eeg_windows.npz").exists()
    if not has_npz:
        return False
    run_dir = resolve_latest_run_dir(session_dir)
    if run_dir is None:
        return False
    return (run_dir / "finger_action_model.pt").exists() and (run_dir / "scaler.npz").exists()


def _latest_replay_ready_session(sessions_root: Optional[Path]) -> Optional[Path]:
    if sessions_root is None or not sessions_root.exists():
        return None
    candidates = [p for p in sessions_root.iterdir() if p.is_dir() and _session_has_replay_artifacts(p)]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_replay_artifact_paths(
    *,
    session_dir: Optional[Path],
    sessions_root: Optional[Path],
    npz_text: str,
    model_text: str,
    scaler_text: str,
) -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    npz_path = _normalize_existing_path(npz_text)
    model_path = _normalize_existing_path(model_text)
    scaler_path = _normalize_existing_path(scaler_text)
    replay_ready_fallback = _latest_replay_ready_session(sessions_root)
    selected_session_ready = (
        session_dir is not None and session_dir.exists() and _session_has_replay_artifacts(session_dir)
    )

    candidate_session_dirs = []
    candidates = [
        _infer_session_dir_from_artifact(npz_path),
        _infer_session_dir_from_artifact(model_path),
        _infer_session_dir_from_artifact(scaler_path),
    ]
    candidates.append(session_dir)
    if session_dir is None and replay_ready_fallback is not None:
        candidates.append(replay_ready_fallback)

    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        try:
            candidate = candidate.resolve()
        except Exception:
            pass
        if candidate not in candidate_session_dirs:
            candidate_session_dirs.append(candidate)

    if not candidate_session_dirs:
        return npz_path, model_path, scaler_path

    primary_session_dir = candidate_session_dirs[0]
    layout = SessionLayout(primary_session_dir)
    if npz_path is None:
        if layout.windows_npz.exists():
            npz_path = layout.windows_npz
        else:
            legacy_npz = primary_session_dir / "windows" / "eeg_windows.npz"
            if legacy_npz.exists():
                npz_path = legacy_npz

    if model_path is None or scaler_path is None:
        for candidate_session_dir in candidate_session_dirs:
            run_dir = resolve_latest_run_dir(candidate_session_dir)
            if run_dir is None:
                continue
            if model_path is None:
                candidate_model = run_dir / "finger_action_model.pt"
                if candidate_model.exists():
                    model_path = candidate_model
            if scaler_path is None:
                candidate_scaler = run_dir / "scaler.npz"
                if candidate_scaler.exists():
                    scaler_path = candidate_scaler
            if model_path is not None and scaler_path is not None:
                break

    return npz_path, model_path, scaler_path
