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


def test_format_prediction_text_decodes_active_finger_head_with_action_conditioning():
    finger_probs = np.asarray(
        [
            [0.91, 0.03, 0.02, 0.02, 0.02],
        ],
        dtype=float,
    )
    action_probs = np.asarray(
        [
            [0.02, 0.90, 0.08],
        ],
        dtype=float,
    )

    finger_text, action_text = ui_mod._format_prediction_text(
        finger_probs, action_probs
    )

    assert finger_text.startswith("THUMB")
    assert "NONE" not in finger_text
    assert action_text.startswith("OPEN")


def test_finger_label_map_for_active_finger_head_skips_none_label():
    finger_probs = np.zeros((4, 5), dtype=float)

    label_map = ui_mod._finger_label_map_for_probs(finger_probs)

    assert label_map[0] == "THUMB"
    assert "NONE" not in set(label_map.values())


def test_prediction_label_payload_marks_correct_replay_predictions_green():
    finger_probs = np.asarray([[0.91, 0.03, 0.02, 0.02, 0.02]], dtype=float)
    action_probs = np.asarray([[0.02, 0.90, 0.08]], dtype=float)

    label_text, label_style = ui_mod._prediction_label_payload(
        finger_probs,
        action_probs,
        truth_action_id=1,
        truth_finger_id=1,
    )

    assert "Truth: Finger THUMB, Action OPEN" in label_text
    assert "130, 255, 130" in label_style


def test_prediction_label_payload_marks_incorrect_replay_predictions_red():
    finger_probs = np.asarray([[0.91, 0.03, 0.02, 0.02, 0.02]], dtype=float)
    action_probs = np.asarray([[0.02, 0.90, 0.08]], dtype=float)

    label_text, label_style = ui_mod._prediction_label_payload(
        finger_probs,
        action_probs,
        truth_action_id=2,
        truth_finger_id=2,
    )

    assert "Truth: Finger INDEX, Action CLOSE" in label_text
    assert "255, 120, 120" in label_style


def test_replay_preview_actuation_from_record_uses_step7_target_state():
    assert ui_mod._replay_preview_actuation_from_record(
        {
            "actuation_sent": False,
            "actuation_target_finger_id": 1,
            "actuation_target_action_id": 2,
            "actuation_speed_scalar": 0.75,
        }
    ) == (1, 2, 0.75)

    assert ui_mod._replay_preview_actuation_from_record(
        {
            "actuation_target_finger_id": 0,
            "actuation_target_action_id": 0,
            "actuation_speed_scalar": 0.75,
        }
    ) is None


def test_scale_replay_preview_speed_halves_step7_speed_for_replay_only():
    assert ui_mod._scale_replay_preview_speed(0.8) == pytest.approx(0.4)


def test_build_replay_runtime_config_maps_warn_latency_policy_to_ignore():
    runtime_config = ui_mod._build_replay_runtime_config(
        {
            "window_sec": 0.25,
            "hop_sec": 0.05,
            "latency_threshold_ms": 750.0,
            "latency_policy": "warn",
            "actuation_min_prob": 0.2,
            "actuation_stability": 2,
            "actuation_cooldown_ms": 0,
            "actuation_repeat_ms": 100,
            "actuation_min_speed": 0.5,
            "modulate_actuation_speed": False,
            "actuation_speed_gamma": 1.0,
            "use_inference_engine": False,
            "mc_passes": 10,
            "uncertainty_base_threshold": 0.75,
            "uncertainty_weight": 0.5,
            "live_quality_enabled": True,
            "input_clip_abs_z": 6.0,
            "bad_channel_rms_z": 4.0,
            "bad_channel_abs_p95_z": 6.0,
            "bad_channel_clipped_frac": 0.05,
            "bad_window_clipped_frac": 0.10,
            "bad_window_max_masked_channels": 1,
        }
    )

    assert runtime_config.latency_mode == "ignore"


def test_estimate_replay_auto_interval_ms_uses_window_hop():
    interval_ms = ui_mod._estimate_replay_auto_interval_ms(
        np.asarray([0.00, 0.05, 0.10, 0.15], dtype=float)
    )

    assert interval_ms == 50


def test_build_scrambled_replay_order_preserves_event_blocks_and_anchor():
    order = ui_mod._build_scrambled_replay_order(
        6,
        event_ids=np.asarray([10, 10, 11, 11, 12, 12], dtype=np.int64),
        trial_ids=np.asarray([1, 1, 2, 2, 3, 3], dtype=np.int64),
        anchor_idx=2,
        rng=np.random.default_rng(0),
    )

    assert order[:2] == [2, 3]
    assert sorted(order) == [0, 1, 2, 3, 4, 5]
    assert order[2:4] in ([0, 1], [4, 5])
    assert order[4:6] in ([0, 1], [4, 5])


def test_robot_hand_preview_enforces_auto_advance_and_realistic_default(window):
    model_views = window._build_model_views_widget()
    window.replay_viz = SimpleNamespace(window_start=np.asarray([0.0, 0.05, 0.10]))
    window._refresh_replay_views = lambda: None

    window._toggle_replay_hand_preview(True)

    assert window.replay_auto_checkbox.isChecked()
    assert not window.replay_auto_checkbox.isEnabled()
    assert window.replay_auto_interval.value() == 50
    assert model_views is not None


def test_replay_preview_does_not_resend_same_active_target(window):
    model_views = window._build_model_views_widget()
    window.replay_viz = SimpleNamespace(window_start=np.asarray([0.0, 0.05, 0.10]))
    window._refresh_replay_views = lambda: None
    window._toggle_replay_hand_preview(True)
    window.replay_hand_preview_checkbox.setChecked(True)

    class _FakeActuator:
        def __init__(self) -> None:
            self.calls = []

        def send(self, finger_id, action_id, speed_scalar=None):
            self.calls.append((finger_id, action_id, speed_scalar))

    actuator = _FakeActuator()
    window._ensure_replay_hand_actuator = lambda: actuator
    window._ensure_replay_runtime_records = lambda: [
        {
            "actuation_target_finger_id": 2,
            "actuation_target_action_id": 1,
            "actuation_speed_scalar": 0.8,
        },
        {
            "actuation_target_finger_id": 2,
            "actuation_target_action_id": 1,
            "actuation_speed_scalar": 0.8,
        },
    ]

    window._maybe_send_replay_preview(0)
    window._maybe_send_replay_preview(1)

    assert actuator.calls == [(2, 1, pytest.approx(0.4))]
    assert model_views is not None


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


def test_prepare_live_infer_launch_prefers_step7_session_field_over_global_session(
    window, monkeypatch: pytest.MonkeyPatch
):
    project = "Demo"
    subject = "S01"
    deployment_ui_session = f"{subject}_20250101_000000"
    global_ui_session = f"{subject}_20250102_000000"
    subject_dir = Path("Projects") / project / "subjects" / subject
    deployment_session_dir = subject_dir / "sessions" / deployment_ui_session
    global_session_dir = subject_dir / "sessions" / global_ui_session
    deployment_run_dir = deployment_session_dir / "processed" / "models" / "run_001"
    deployment_run_dir.mkdir(parents=True, exist_ok=True)
    (deployment_run_dir / "finger_action_model.pt").write_text("model")
    (deployment_run_dir / "scaler.npz").write_text("scaler")
    (deployment_run_dir / "temperature_scaling.json").write_text("{}")
    global_session_dir.mkdir(parents=True, exist_ok=True)

    window.current_project = project
    window.current_subject = subject
    window.current_session_ui = global_ui_session
    window.session_dir_input.setText(str(global_session_dir))
    window.live_stream_name = "Muse2-EEG"
    window.live_stream_type = "EEG"
    window.live_lsl_source_id = "captured-source-123"

    infer_fields = window.fields["infer"]
    infer_fields["session_dir"].setText(str(deployment_session_dir))
    infer_fields["deployment_session_dir"].setText(str(deployment_session_dir))
    infer_fields["out_dir"].setText("")

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250103_000000")

    launch = window._prepare_live_infer_launch(window.scripts["live_infer"])

    assert launch is not None
    assert launch.session_dir == deployment_session_dir.resolve()
    assert Path(launch.settings["session_dir"]).resolve() == deployment_session_dir.resolve()
    assert Path(launch.settings["deployment_session_dir"]).resolve() == deployment_session_dir.resolve()
    assert launch.settings["model_path"].endswith("run_001/finger_action_model.pt")
    assert launch.settings["scaler_path"].endswith("run_001/scaler.npz")
    assert launch.settings["out_dir"].endswith("processed/live_infer_20250103_000000")
    assert str(deployment_session_dir.resolve()) in launch.args
    assert str(global_session_dir.resolve()) not in launch.args


def test_autofill_step7_artifacts_uses_step7_session_not_global_session(
    window, monkeypatch: pytest.MonkeyPatch
):
    project = "Demo"
    subject = "S01"
    deployment_ui_session = f"{subject}_20250101_000000"
    global_ui_session = f"{subject}_20250102_000000"
    subject_dir = Path("Projects") / project / "subjects" / subject
    deployment_session_dir = subject_dir / "sessions" / deployment_ui_session
    global_session_dir = subject_dir / "sessions" / global_ui_session
    deployment_run_dir = deployment_session_dir / "processed" / "models" / "run_001"
    deployment_run_dir.mkdir(parents=True, exist_ok=True)
    (deployment_run_dir / "finger_action_model.pt").write_text("model")
    (deployment_run_dir / "scaler.npz").write_text("scaler")
    global_session_dir.mkdir(parents=True, exist_ok=True)
    live_dir = deployment_session_dir / "processed" / "live_infer_20250101_101010"
    legacy_live_dir = deployment_session_dir / "processed" / "live_infer_v9"
    live_dir.mkdir(parents=True, exist_ok=True)
    legacy_live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "predictions.jsonl").write_text('{"committed_action_id":1}\n')
    (legacy_live_dir / "predictions.jsonl").write_text('{"committed_action_id":0}\n')

    window.current_project = project
    window.current_subject = subject
    window.current_session_ui = global_ui_session
    window.session_dir_input.setText(str(global_session_dir))
    infer_fields = window.fields["infer"]
    infer_fields["session_dir"].setText(str(deployment_session_dir))
    infer_fields["deployment_session_dir"].setText(str(deployment_session_dir))

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250103_000001")

    window._autofill_dependent_paths_from_session_dir()

    assert window.fields["infer"]["model_path"].text().endswith(
        "run_001/finger_action_model.pt"
    )
    assert window.fields["infer"]["scaler_path"].text().endswith("run_001/scaler.npz")
    assert window.fields["infer"]["out_dir"].text().endswith(
        "processed/live_infer_20250103_000001"
    )
    assert window.fields["live_review"]["pred_log"].text().endswith(
        "processed/live_infer_20250101_101010/predictions.jsonl"
    )


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
