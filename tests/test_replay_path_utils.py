from pathlib import Path

from app.replay_path_utils import resolve_replay_artifact_paths


def test_resolve_replay_artifact_paths_prefers_session_dir(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    run_dir = session_dir / "processed" / "models" / "run1"
    run_dir.mkdir(parents=True)
    npz_path = session_dir / "processed" / "eeg_windows.npz"
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.write_bytes(b"npz")
    model_path = run_dir / "finger_action_model.pt"
    scaler_path = run_dir / "scaler.npz"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")

    resolved = resolve_replay_artifact_paths(
        session_dir=session_dir,
        sessions_root=None,
        npz_text="",
        model_text="",
        scaler_text="",
    )

    assert resolved == (npz_path.resolve(), model_path.resolve(), scaler_path.resolve())


def test_resolve_replay_artifact_paths_keeps_existing_explicit_paths(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    npz_path = tmp_path / "custom.npz"
    model_path = tmp_path / "custom.pt"
    scaler_path = tmp_path / "custom_scaler.npz"
    npz_path.write_bytes(b"npz")
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")

    resolved = resolve_replay_artifact_paths(
        session_dir=session_dir,
        sessions_root=None,
        npz_text=str(npz_path),
        model_text=str(model_path),
        scaler_text=str(scaler_path),
    )

    assert resolved == (npz_path.resolve(), model_path.resolve(), scaler_path.resolve())


def test_resolve_replay_artifact_paths_prefers_npz_implied_session_for_models(
    tmp_path: Path,
) -> None:
    stale_session = tmp_path / "stale_session"
    stale_processed = stale_session / "processed"
    stale_processed.mkdir(parents=True)
    (stale_processed / "eeg_windows.npz").write_bytes(b"stale")
    (stale_processed / "models").mkdir()

    combined_session = tmp_path / "combined_session"
    combined_run = combined_session / "processed" / "models" / "run1"
    combined_run.mkdir(parents=True)
    combined_npz = combined_session / "processed" / "eeg_windows.npz"
    combined_npz.parent.mkdir(parents=True, exist_ok=True)
    combined_npz.write_bytes(b"combined")
    combined_model = combined_run / "finger_action_model.pt"
    combined_scaler = combined_run / "scaler.npz"
    combined_model.write_bytes(b"model")
    combined_scaler.write_bytes(b"scaler")

    resolved = resolve_replay_artifact_paths(
        session_dir=stale_session,
        sessions_root=None,
        npz_text=str(combined_npz),
        model_text="",
        scaler_text="",
    )

    assert resolved == (
        combined_npz.resolve(),
        combined_model.resolve(),
        combined_scaler.resolve(),
    )


def test_resolve_replay_artifact_paths_falls_back_to_latest_replay_ready_session(
    tmp_path: Path,
) -> None:
    stale_session = tmp_path / "stale_session"
    stale_processed = stale_session / "processed"
    stale_processed.mkdir(parents=True)
    (stale_processed / "eeg_windows.npz").write_bytes(b"stale")
    (stale_processed / "models").mkdir()

    older_ready = tmp_path / "older_ready"
    older_run = older_ready / "processed" / "models" / "run1"
    older_run.mkdir(parents=True)
    older_npz = older_ready / "processed" / "eeg_windows.npz"
    older_npz.parent.mkdir(parents=True, exist_ok=True)
    older_npz.write_bytes(b"older")
    (older_run / "finger_action_model.pt").write_bytes(b"model")
    (older_run / "scaler.npz").write_bytes(b"scaler")

    latest_ready = tmp_path / "latest_ready"
    latest_run = latest_ready / "processed" / "models" / "run2"
    latest_run.mkdir(parents=True)
    latest_npz = latest_ready / "processed" / "eeg_windows.npz"
    latest_npz.parent.mkdir(parents=True, exist_ok=True)
    latest_npz.write_bytes(b"latest")
    latest_model = latest_run / "finger_action_model.pt"
    latest_scaler = latest_run / "scaler.npz"
    latest_model.write_bytes(b"model")
    latest_scaler.write_bytes(b"scaler")

    resolved = resolve_replay_artifact_paths(
        session_dir=None,
        sessions_root=tmp_path,
        npz_text="",
        model_text="",
        scaler_text="",
    )

    assert resolved == (
        latest_npz.resolve(),
        latest_model.resolve(),
        latest_scaler.resolve(),
    )


def test_resolve_replay_artifact_paths_does_not_drift_when_session_is_pinned(
    tmp_path: Path,
) -> None:
    stale_session = tmp_path / "stale_session"
    stale_processed = stale_session / "processed"
    stale_processed.mkdir(parents=True)
    (stale_processed / "eeg_windows.npz").write_bytes(b"stale")
    (stale_processed / "models").mkdir()

    latest_ready = tmp_path / "latest_ready"
    latest_run = latest_ready / "processed" / "models" / "run2"
    latest_run.mkdir(parents=True)
    latest_npz = latest_ready / "processed" / "eeg_windows.npz"
    latest_npz.parent.mkdir(parents=True, exist_ok=True)
    latest_npz.write_bytes(b"latest")
    (latest_run / "finger_action_model.pt").write_bytes(b"model")
    (latest_run / "scaler.npz").write_bytes(b"scaler")

    resolved = resolve_replay_artifact_paths(
        session_dir=stale_session,
        sessions_root=tmp_path,
        npz_text="",
        model_text="",
        scaler_text="",
    )

    assert resolved == (
        (stale_session / "processed" / "eeg_windows.npz").resolve(),
        None,
        None,
    )
