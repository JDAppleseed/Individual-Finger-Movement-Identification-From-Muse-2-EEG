from __future__ import annotations

from pathlib import Path

from muse_streaming import io_paths


def test_default_dirs_no_env(monkeypatch):
    monkeypatch.delenv("MUSE_PROCESSED_DIR", raising=False)
    monkeypatch.delenv("MUSE_RAW_DIR", raising=False)
    assert str(io_paths.default_processed_dir()) == "data/processed"
    assert str(io_paths.default_raw_dir()) == "data/raw"


def test_default_dirs_env_override(monkeypatch):
    monkeypatch.setenv("MUSE_PROCESSED_DIR", "custom/processed")
    monkeypatch.setenv("MUSE_RAW_DIR", "custom/raw")
    assert str(io_paths.default_processed_dir()) == "custom/processed"
    assert str(io_paths.default_raw_dir()) == "custom/raw"


def test_build_session_dir_paths_prefixes_subject_id():
    paths = io_paths.build_session_dir_paths(
        Path("/tmp/output"),
        "S1",
        "20240101_010101",
    )
    assert paths.session_dir == Path("/tmp/output/S1_20240101_010101")
    assert paths.raw_dir == Path("/tmp/output/S1_20240101_010101/raw")
    assert paths.events_dir == Path("/tmp/output/S1_20240101_010101/events")


def test_prepare_session_dir_paths_resumes_existing_session(tmp_path):
    output_root = tmp_path / "sessions"
    existing = output_root / "S1_20240101_010101"
    (existing / "raw").mkdir(parents=True)

    paths, resumed, reason = io_paths.prepare_session_dir_paths(
        output_root=output_root,
        subject_id="S1",
        session_id="20240101_010101",
        resume=True,
    )

    assert resumed is True
    assert reason == "resume"
    assert paths.session_dir == existing


def test_prepare_session_dir_paths_avoids_collisions_with_existing_files(tmp_path):
    output_root = tmp_path / "sessions"
    existing = output_root / "S1_20240101_010101"
    existing.mkdir(parents=True)
    (existing / "meta.json").write_text("{}", encoding="utf-8")

    paths, resumed, reason = io_paths.prepare_session_dir_paths(
        output_root=output_root,
        subject_id="S1",
        session_id="20240101_010101",
        resume=False,
    )

    assert resumed is False
    assert reason == "new_session_collision(20240101_010101)"
    assert paths.session_id == "20240101_010101_01"
    assert paths.session_dir == output_root / "S1_20240101_010101_01"
