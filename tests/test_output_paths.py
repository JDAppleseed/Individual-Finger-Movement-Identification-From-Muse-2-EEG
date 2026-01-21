from __future__ import annotations

from pathlib import Path

from utils.output_paths import resolve_output_dir


def test_resolve_output_dir_cli_overrides_config(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    result = resolve_output_dir(
        kind="processed",
        cli_value=tmp_path / "cli_out",
        config_settings={"processed_dir": "cfg_out"},
        config_payload={},
        config_path=None,
        repo_root=repo_root,
        subject_id="S1",
        session_id_hint="20240101_010101",
        default_base=Path("data/processed"),
        session_id_seed="seed",
        create=False,
    )
    assert result == (tmp_path / "cli_out").resolve()


def test_resolve_output_dir_config_overrides_default(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    result = resolve_output_dir(
        kind="processed",
        cli_value=None,
        config_settings={"processed_dir": "cfg_out"},
        config_payload={},
        config_path=None,
        repo_root=repo_root,
        subject_id="S1",
        session_id_hint="20240101_010101",
        default_base=Path("data/processed"),
        session_id_seed="seed",
        create=False,
    )
    assert result == (repo_root / "cfg_out").resolve()


def test_resolve_output_dir_project_session_default(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    config_payload = {
        "project_name": "Test1",
        "subject_id": "Har",
        "session_id": "Har_20260120_213753",
    }
    result = resolve_output_dir(
        kind="processed",
        cli_value=None,
        config_settings={},
        config_payload=config_payload,
        config_path=None,
        repo_root=repo_root,
        subject_id="8-M16",
        session_id_hint="20260120_213753",
        default_base=Path("data/processed"),
        session_id_seed="seed",
        create=True,
    )
    expected = (
        repo_root
        / "Projects"
        / "Test1"
        / "subjects"
        / "Har"
        / "sessions"
        / "Har_20260120_213753"
        / "processed"
    ).resolve()
    assert result == expected
    assert expected.exists()


def test_resolve_output_dir_fallback_creates_dir(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    result = resolve_output_dir(
        kind="processed",
        cli_value=None,
        config_settings={},
        config_payload={},
        config_path=None,
        repo_root=repo_root,
        subject_id="Har",
        session_id_hint="20260120_213753",
        default_base=Path("data/processed"),
        session_id_seed="seed",
        create=True,
    )
    expected = (
        repo_root
        / "data"
        / "processed"
        / "Har"
        / "20260120_213753"
    ).resolve()
    assert result == expected
    assert expected.exists()


def test_resolve_output_dir_resume_overrides_all(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    resume_path = tmp_path / "resume" / "file.csv"
    resume_path.parent.mkdir(parents=True, exist_ok=True)
    result = resolve_output_dir(
        kind="processed",
        cli_value="cli_out",
        config_settings={"processed_dir": "cfg_out"},
        config_payload={},
        config_path=None,
        repo_root=repo_root,
        subject_id="S1",
        session_id_hint="20240101_010101",
        default_base=Path("data/processed"),
        session_id_seed="seed",
        resume_path=resume_path,
        create=True,
    )
    assert result == resume_path.parent.resolve()
