from __future__ import annotations

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
