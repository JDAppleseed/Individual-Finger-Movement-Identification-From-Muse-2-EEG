import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

_real_version_info = sys.version_info


class _FakeVersionInfo(tuple):
    major = 3
    minor = 11
    micro = 0
    releaselevel = "final"
    serial = 0

    def __new__(cls):
        return super().__new__(cls, (3, 11, 0, "final", 0))


sys.version_info = _FakeVersionInfo()
import eeglab_wrapper_ui as ui_mod
sys.version_info = _real_version_info


@pytest.fixture
def app():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def window(app, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    window = ui_mod.MainWindow()
    yield window
    window._auto_scan_timer.stop()
    window._live_viz_status_timer.stop()
    window._replay_auto_timer.stop()
    window.close()


def test_infer_step_arg_specs_do_not_expose_project_name_override(window):
    infer_specs = window._build_step_arg_specs()["infer"]

    assert all(spec.name != "project_name" for spec in infer_specs)


def test_prepare_live_infer_launch_freezes_source_and_session_artifacts(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    project = "Demo"
    subject = "S01"
    backend_session = "20250101_000000"
    ui_session = f"{subject}_{backend_session}"
    subject_dir = Path("Projects") / project / "subjects" / subject
    session_dir = subject_dir / "sessions" / ui_session
    run_dir = session_dir / "processed" / "models" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "finger_action_model.pt").write_text("model")
    (run_dir / "scaler.npz").write_text("scaler")
    (run_dir / "temperature_scaling.json").write_text("{}")

    window.current_project = project
    window.current_subject = subject
    window.current_session_backend = backend_session
    window.current_session_ui = ui_session
    window.session_dir_input.setText(str(session_dir))
    window.live_stream_name = "Muse2-EEG"
    window.live_stream_type = "EEG"
    window.live_lsl_source_id = "captured-source-123"

    infer_fields = window.fields["infer"]
    infer_fields["stream_name"].setText("Muse2-EEG")
    infer_fields["stream_type"].setText("EEG")
    infer_fields["lsl_source_id"].setText("stale-ui-source")
    infer_fields["deployment_session_dir"].setText(str(session_dir))
    infer_fields["out_dir"].setText("")

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250101_000010")

    launch = window._prepare_live_infer_launch(window.scripts["live_infer"])

    assert launch is not None
    assert launch.settings["lsl_source_id"] == "captured-source-123"
    assert launch.settings["deployment_session_dir"] == str(session_dir)
    assert launch.settings["model_path"].endswith("run_001/finger_action_model.pt")
    assert launch.settings["scaler_path"].endswith("run_001/scaler.npz")
    assert launch.settings["out_dir"].endswith("processed/live_infer_20250101_000010")
    assert launch.config_path.exists()
    assert "--config" in launch.args
    assert "--session-dir" in launch.args
    assert str(session_dir.resolve()) in launch.args
    assert launch.config_payload["settings"]["out_dir"].endswith(
        "processed/live_infer_20250101_000010"
    )


def test_autofill_dependent_paths_sets_timestamped_step7_out_dir(
    window, monkeypatch: pytest.MonkeyPatch
):
    project = "Demo"
    subject = "S01"
    ui_session = f"{subject}_20250101_000000"
    subject_dir = Path("Projects") / project / "subjects" / subject
    session_dir = subject_dir / "sessions" / ui_session
    run_dir = session_dir / "processed" / "models" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "finger_action_model.pt").write_text("model")
    (run_dir / "scaler.npz").write_text("scaler")

    window.current_project = project
    window.current_subject = subject
    window.current_session_ui = ui_session
    window.session_dir_input.setText(str(session_dir))

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250101_000011")

    window._autofill_dependent_paths_from_session_dir()

    out_dir_widget = window.fields["infer"]["out_dir"]
    assert out_dir_widget.text().endswith("processed/live_infer_20250101_000011")


def test_live_ready_gate_uses_frozen_launch_settings(window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict[str, object] = {}

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={
            "stream_name": "FrozenMuse",
            "stream_type": "EEG",
            "REQUIRED_LSL_LABELS": ["TP9", "AF7", "AF8", "TP10"],
            "REQUIRE_EXACTLY_4_CHANNELS": True,
            "SAMPLING_RATE": 256,
        },
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
    )

    def fake_healthcheck(*, interactive, timeout_s, settings_override=None):
        captured["interactive"] = interactive
        captured["timeout_s"] = timeout_s
        captured["settings_override"] = settings_override
        return SimpleNamespace(ok=True)

    monkeypatch.setattr(window, "_run_stream_healthcheck", fake_healthcheck)
    monkeypatch.setattr(window, "_launch_live_infer_after_ready_gate", lambda: None)

    window._live_ready_gate_active = True
    window._pending_live_launch = launch
    window._run_live_ready_gate_attempt()

    assert captured["interactive"] is False
    assert captured["timeout_s"] == 1.0
    assert captured["settings_override"] == launch.settings


def test_prepare_live_infer_launch_derives_effective_expected_labels_from_training_npz(
    window, monkeypatch: pytest.MonkeyPatch
):
    project = "Demo"
    subject = "S01"
    backend_session = "20250101_000001"
    ui_session = f"{subject}_{backend_session}"
    subject_dir = Path("Projects") / project / "subjects" / subject
    session_dir = subject_dir / "sessions" / ui_session
    processed_dir = session_dir / "processed"
    run_dir = processed_dir / "models" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "finger_action_model.pt").write_text("model")
    (run_dir / "scaler.npz").write_text("scaler")
    (run_dir / "temperature_scaling.json").write_text("{}")
    np.savez(
        processed_dir / "eeg_windows.npz",
        channel_names=np.asarray(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(
        ui_mod.QMessageBox,
        "warning",
        lambda _parent, title, message: pytest.fail(
            f"Unexpected warning dialog: {title}: {message}"
        ),
    )

    window.current_project = project
    window.current_subject = subject
    window.current_session_backend = backend_session
    window.current_session_ui = ui_session
    window.session_dir_input.setText(str(session_dir))
    window.live_stream_name = "Muse2-EEG"
    window.live_stream_type = "EEG"
    window.live_lsl_source_id = "captured-source-123"

    infer_fields = window.fields["infer"]
    infer_fields["stream_name"].setText("Muse2-EEG")
    infer_fields["stream_type"].setText("EEG")
    infer_fields["lsl_source_id"].setText("")
    infer_fields["REQUIRED_LSL_LABELS"].setText("")
    infer_fields["deployment_session_dir"].setText(str(session_dir))
    infer_fields["out_dir"].setText("")

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250101_000020")

    launch = window._prepare_live_infer_launch(window.scripts["live_infer"])

    assert launch is not None
    assert launch.settings["LABEL_CHECK_EXPECTED_LABELS"] == [
        "TP9",
        "AF7",
        "AF8",
        "TP10",
    ]
    assert (
        launch.settings["LABEL_CHECK_EXPECTED_LABELS_SOURCE"]
        == "training_npz.channel_names"
    )
    assert launch.settings["out_dir"].endswith("processed/live_infer_20250101_000020")
    preview = window._format_live_launch_preview(launch)
    assert "Configured REQUIRED_LSL_LABELS: []" in preview
    assert "training_npz.channel_names" in preview


def test_prepare_live_infer_launch_blocks_when_expected_labels_are_unavailable(
    window, monkeypatch: pytest.MonkeyPatch
):
    project = "Demo"
    subject = "S01"
    backend_session = "20250101_000002"
    ui_session = f"{subject}_{backend_session}"
    subject_dir = Path("Projects") / project / "subjects" / subject
    session_dir = subject_dir / "sessions" / ui_session
    run_dir = session_dir / "processed" / "models" / "run_001"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "finger_action_model.pt").write_text("model")
    (run_dir / "scaler.npz").write_text("scaler")
    (run_dir / "temperature_scaling.json").write_text("{}")

    captured: dict[str, str] = {}

    def fake_warning(_parent, title, message):
        captured["title"] = title
        captured["message"] = message
        return 0

    monkeypatch.setattr(ui_mod.QMessageBox, "warning", fake_warning)

    window.current_project = project
    window.current_subject = subject
    window.current_session_backend = backend_session
    window.current_session_ui = ui_session
    window.session_dir_input.setText(str(session_dir))
    window.live_stream_name = "Muse2-EEG"
    window.live_stream_type = "EEG"
    window.live_lsl_source_id = "captured-source-123"

    infer_fields = window.fields["infer"]
    infer_fields["stream_name"].setText("Muse2-EEG")
    infer_fields["stream_type"].setText("EEG")
    infer_fields["lsl_source_id"].setText("")
    infer_fields["REQUIRED_LSL_LABELS"].setText("")
    infer_fields["deployment_session_dir"].setText(str(session_dir))

    launch = window._prepare_live_infer_launch(window.scripts["live_infer"])

    assert launch is None
    assert captured["title"] == "Expected Channel Labels Missing"
    assert "cannot prove model-order channel mapping" in captured["message"]
