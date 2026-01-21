from __future__ import annotations

from pathlib import Path

from muse_streaming.io_paths import prepare_session_paths


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        for row in rows:
            handle.write(",".join(row) + "\n")


def test_resume_blocked_when_features_missing(tmp_path: Path):
    paths, resumed, reason = prepare_session_paths(
        output_root=tmp_path,
        subject_id="subj",
        session_id="session",
        resume=True,
    )
    assert not resumed
    assert reason == "resume_blocked_missing_features"


def test_resume_allowed_with_nonempty_features(tmp_path: Path):
    features = tmp_path / "subj_session_features.csv"
    _write_csv(features, [["time_s", "lsl_ts"], ["0.0", "1.0"]])

    paths, resumed, reason = prepare_session_paths(
        output_root=tmp_path,
        subject_id="subj",
        session_id="session",
        resume=True,
    )
    assert resumed
    assert reason == "resume"
    assert paths.session_id == "session"


def test_new_session_avoids_overwrite(tmp_path: Path):
    existing = tmp_path / "subj_session_raw.csv"
    _write_csv(existing, [["time_s"], ["0.0"]])

    paths, resumed, reason = prepare_session_paths(
        output_root=tmp_path,
        subject_id="subj",
        session_id="session",
        resume=False,
    )
    assert not resumed
    assert paths.session_id != "session"
    assert "collision" in reason
