import json
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
from utils.step7_config import default_step7_settings


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


def test_infer_transport_isolation_settings_are_ui_visible(window):
    expected = {
        "force_no_serial",
        "serial_write_timeout_s",
        "serial_max_hz",
        "serial_settle_s",
        "serial_movement_warmup_enabled",
        "lsl_acquirer_queue_max_chunks",
    }
    assert expected.issubset(set(window.fields["infer"]))
    assert expected.issubset(
        {spec.name for spec in window._build_step_arg_specs()["infer"]}
    )


def test_ui_label_field_parsing_matches_canonical_step7_defaults(window):
    parsed = window._parse_label_field("['tp9', \"af7\", 'AF8', 'tp10']")

    assert parsed == default_step7_settings()["REQUIRED_LSL_LABELS"]


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


def test_replay_preview_different_active_target_does_not_force_global_rest(window):
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
            "actuation_target_finger_id": 3,
            "actuation_target_action_id": 2,
            "actuation_speed_scalar": 0.6,
        },
    ]

    window._maybe_send_replay_preview(0)
    window._maybe_send_replay_preview(1)

    assert actuator.calls == [
        (2, 1, pytest.approx(0.4)),
        (3, 2, pytest.approx(0.3)),
    ]
    assert model_views is not None


def test_replay_hand_preview_blocks_serial_while_live_transport_active(
    window, monkeypatch: pytest.MonkeyPatch
):
    def fail_load(*args, **kwargs):
        raise AssertionError("live module must not load or touch serial")

    monkeypatch.setattr(ui_mod, "_load_live_infer_ui_module", fail_load)

    scenarios = [
        ("ready_gate", lambda: setattr(window, "_live_ready_gate_active", True)),
        (
            "preflight",
            lambda: monkeypatch.setattr(
                window.live_preflight_runner, "is_running", lambda: True
            ),
        ),
        (
            "step7",
            lambda: (
                setattr(window, "active_step", "infer"),
                monkeypatch.setattr(window.runner, "is_running", lambda: True),
            ),
        ),
        (
            "muse_connector",
            lambda: monkeypatch.setattr(window.muse_connector, "is_running", lambda: True),
        ),
    ]

    for _label, activate in scenarios:
        window._replay_hand_actuator = None
        window._live_ready_gate_active = False
        window.active_step = None
        monkeypatch.setattr(window.runner, "is_running", lambda: False)
        monkeypatch.setattr(window.live_preflight_runner, "is_running", lambda: False)
        monkeypatch.setattr(window.muse_connector, "is_running", lambda: False)
        activate()
        with pytest.raises(RuntimeError, match="Replay hand preview serial access is blocked"):
            window._ensure_replay_hand_actuator()


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
    canonical_config_path = subject_dir / "config" / "infer.json"
    canonical_config_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_config_path.write_text(
        json.dumps({"settings": {"actuation_min_prob": 0.2}}, indent=2)
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
    infer_fields["lsl_source_id"].setText("stale-ui-source")
    infer_fields["deployment_session_dir"].setText(str(session_dir))
    infer_fields["out_dir"].setText("")

    monkeypatch.setattr(
        ui_mod, "session_backend_id", lambda timestamp=None: "20250101_000010"
    )

    launch = window._prepare_live_infer_launch(window.scripts["live_infer"])

    assert launch is not None
    assert launch.settings["lsl_source_id"] == "captured-source-123"
    assert launch.settings["deployment_session_dir"] == str(session_dir.resolve())
    assert launch.settings["model_path"].endswith("run_001/finger_action_model.pt")
    assert launch.settings["scaler_path"].endswith("run_001/scaler.npz")
    assert launch.settings["out_dir"].endswith("processed/live_infer_20250101_000010")
    assert launch.config_path.exists()
    assert launch.config_path == Path(launch.settings["out_dir"]) / "step7_launch_config.json"
    assert launch.base_config_path == canonical_config_path.resolve()
    assert "--config" in launch.args
    assert "--session-dir" in launch.args
    assert str(session_dir.resolve()) in launch.args
    assert launch.config_payload["settings"]["out_dir"].endswith(
        "processed/live_infer_20250101_000010"
    )
    persisted = json.loads(canonical_config_path.read_text())
    assert persisted["settings"]["actuation_min_prob"] == pytest.approx(0.2)
    assert "--model-path" in launch.args
    assert "--scaler-path" in launch.args
    assert "--out-dir" in launch.args
    assert "captured-source-123" in launch.args


def test_prepare_live_infer_launch_warns_when_ui_disables_canonical_actuation(
    window, monkeypatch: pytest.MonkeyPatch
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
    canonical_config_path = subject_dir / "config" / "infer.json"
    canonical_config_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_config_path.write_text(
        json.dumps({"settings": {"enable_actuation": True}}, indent=2)
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
    infer_fields["deployment_session_dir"].setText(str(session_dir))
    infer_fields["out_dir"].setText("")
    infer_fields["enable_actuation"].setChecked(False)

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250101_000010")

    launch = window._prepare_live_infer_launch(window.scripts["live_infer"])

    assert launch is not None
    assert launch.settings["enable_actuation"] is False
    assert (
        "canonical_enable_actuation_overridden_false"
        in launch.config_payload["config_resolution"]["warnings"]
    )
    assert any(
        "canonical config but disabled in the UI runtime state" in line
        for line in window.log_entries
    )
    assert "canonical config is enabled" in window._format_live_launch_preview(launch)


def test_prepare_live_infer_launch_refreshes_autofilled_out_dir_for_new_launch_action(
    window, monkeypatch: pytest.MonkeyPatch
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

    stale_out_dir = session_dir / "processed" / "live_infer_20250101_000010"
    stale_out_dir.mkdir(parents=True, exist_ok=True)
    (stale_out_dir / "stale.txt").write_text("stale")

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
    infer_fields["deployment_session_dir"].setText(str(session_dir))
    infer_fields["out_dir"].setText(str(stale_out_dir))
    window._auto_field_values["infer.out_dir"] = str(stale_out_dir)

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250101_000011")

    launch = window._prepare_live_infer_launch(window.scripts["live_infer"])

    assert launch is not None
    assert launch.settings["out_dir"].endswith("processed/live_infer_20250101_000011")
    assert "20250101_000010" not in launch.settings["out_dir"]


def test_select_subject_prefers_canonical_step7_config_over_subject_mirror(
    window, monkeypatch: pytest.MonkeyPatch
):
    project = "Demo"
    subject = "S01"
    subject_dir = Path("Projects") / project / "subjects" / subject
    winning_config = subject_dir / "winning_model" / "configs" / "infer.json"
    mirror_config = subject_dir / "config" / "infer.json"
    winning_config.parent.mkdir(parents=True, exist_ok=True)
    mirror_config.parent.mkdir(parents=True, exist_ok=True)
    winning_config.write_text(
        json.dumps({"settings": {"actuation_min_prob": 0.0}}, indent=2)
    )
    mirror_config.write_text(
        json.dumps({"settings": {"actuation_min_prob": 0.9}}, indent=2)
    )

    window.current_project = project
    window._refresh_subjects = lambda: None
    window._auto_select_latest_session_for_subject = lambda: None
    window._auto_fill_paths = lambda: None
    window._refresh_export_controls = lambda: None
    window._seed_stream_name_input = lambda: None

    window._select_subject(subject)

    widget = window.fields["infer"]["actuation_min_prob"]
    assert widget.value() == pytest.approx(0.0)


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


def test_autofill_step7_artifacts_prefers_winning_model_snapshot(
    window, monkeypatch: pytest.MonkeyPatch
):
    project = "Demo"
    subject = "S01"
    ui_session = f"{subject}_20250101_000000"
    subject_dir = Path("Projects") / project / "subjects" / subject
    session_dir = subject_dir / "sessions" / ui_session
    winning_run = session_dir / "processed" / "models" / "20260319_075520"
    newer_local_run = session_dir / "processed" / "models" / "20260403_grouptrial_rest050"
    for run_dir in (winning_run, newer_local_run):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "finger_action_model.pt").write_text("model")
        (run_dir / "scaler.npz").write_text("scaler")

    winning_config = subject_dir / "winning_model" / "configs" / "infer.json"
    winning_config.parent.mkdir(parents=True, exist_ok=True)
    winning_config.write_text(
        json.dumps(
            {
                "settings": {
                    "session_dir": str(session_dir),
                    "deployment_session_dir": str(session_dir),
                    "model_path": str(winning_run / "finger_action_model.pt"),
                    "scaler_path": str(winning_run / "scaler.npz"),
                }
            },
            indent=2,
        )
    )
    os.utime(newer_local_run, (2_000_000_000, 2_000_000_000))

    window.current_project = project
    window.current_subject = subject
    window.current_session_ui = ui_session
    window.session_dir_input.setText(str(session_dir))

    monkeypatch.setattr(ui_mod, "session_backend_id", lambda timestamp=None: "20250101_000011")

    window._autofill_dependent_paths_from_session_dir()

    infer_fields = window.fields["infer"]
    assert infer_fields["model_path"].text().endswith(
        "20260319_075520/finger_action_model.pt"
    )
    assert infer_fields["scaler_path"].text().endswith("20260319_075520/scaler.npz")
    assert "20260403_grouptrial_rest050" not in infer_fields["model_path"].text()


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


def test_build_live_preflight_args_uses_explicit_report_artifact(window, tmp_path: Path):
    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=tmp_path / "live_preflight_report.json",
    )

    args = window._build_live_preflight_args(launch)

    assert "--report-path" in args
    assert str(launch.preflight_report_path) in args
    assert "--json" not in args


def test_build_live_preflight_args_uses_frozen_launch_plan_paths(window, tmp_path: Path):
    resolved_plan = SimpleNamespace(
        selected_session_dir=tmp_path / "session",
        model_path=tmp_path / "artifacts" / "finger_action_model.pt",
        scaler_path=tmp_path / "artifacts" / "scaler.npz",
        out_dir=tmp_path / "live_infer_001",
    )
    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={"LSL_SOURCE_ID": "source-123"},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={"lsl_source_id": "source-123"},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=tmp_path / "live_preflight_report.json",
        resolved_launch_plan=resolved_plan,
    )

    args = window._build_live_preflight_args(launch)

    assert ["--session-dir", str(resolved_plan.selected_session_dir)] == args[5:7]
    assert "--model-path" in args
    assert str(resolved_plan.model_path) in args
    assert "--scaler-path" in args
    assert str(resolved_plan.scaler_path) in args
    assert "--out-dir" in args
    assert str(resolved_plan.out_dir) in args
    assert "--lsl-source-id" in args
    assert "source-123" in args


def test_load_live_preflight_report_reports_missing_empty_and_malformed(window, tmp_path: Path):
    report_path = tmp_path / "live_preflight_report.json"

    report, diagnostics = window._load_live_preflight_report(
        report_path=report_path,
        raw_output="[stderr] boom",
    )
    assert report == {}
    assert diagnostics["reason"] == "preflight_report_missing"

    report_path.write_text("")
    report, diagnostics = window._load_live_preflight_report(
        report_path=report_path,
        raw_output="",
    )
    assert report == {}
    assert diagnostics["reason"] == "preflight_report_empty"

    report_path.write_text("{not-json")
    report, diagnostics = window._load_live_preflight_report(
        report_path=report_path,
        raw_output="stdout-noise",
    )
    assert report == {}
    assert diagnostics["reason"] == "preflight_report_malformed"
    assert "report_preview" in diagnostics


def test_live_preflight_finish_prefers_report_file_over_noisy_output(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    payload = {
        "ready": True,
        "warnings": [],
        "errors": [],
        "launch_plan": {"out_dir": str(tmp_path / "live_infer")},
        "effective_contract": {},
    }
    report_path.write_text(json.dumps(payload))

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = ["[stderr] log noise", "not json"]

    monkeypatch.setattr(window, "_confirm_live_launch_after_preflight", lambda report: False)
    notices = []
    statuses = []
    monkeypatch.setattr(window, "_show_blocking_notice", lambda title, message: notices.append((title, message)))
    monkeypatch.setattr(window, "_set_step_status", lambda step_id, status: statuses.append((step_id, status)))

    window._on_live_preflight_finished(0, 0)

    assert window._last_live_preflight_report["ready"] is True
    assert notices == []
    assert ("infer", "Cancelled") in statuses


def test_live_preflight_finish_surfaces_specific_contract_failure(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = ["[stderr] traceback line 1"]

    notices = []
    monkeypatch.setattr(window, "_show_blocking_notice", lambda title, message: notices.append((title, message)))

    window._on_live_preflight_finished(1, 0)

    assert window._last_live_preflight_report == {}
    assert notices
    assert "preflight_subprocess_failed" in notices[0][1]
    assert "report_exists=False" in notices[0][1]


def test_live_preflight_finish_blocks_on_launch_plan_mismatch(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    payload = {
        "ready": True,
        "warnings": [],
        "errors": [],
        "launch_plan": {
            "selected_session_dir": str(tmp_path / "other_session"),
            "model_path": str(tmp_path / "other_model.pt"),
            "scaler_path": str(tmp_path / "other_scaler.npz"),
            "out_dir": str(tmp_path / "other_live_infer"),
        },
        "effective_contract": {},
    }
    report_path.write_text(json.dumps(payload))

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
        resolved_launch_plan=SimpleNamespace(
            selected_session_dir=tmp_path / "session",
            model_path=tmp_path / "finger_action_model.pt",
            scaler_path=tmp_path / "scaler.npz",
            out_dir=tmp_path / "live_infer",
        ),
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = []

    notices = []
    monkeypatch.setattr(window, "_show_blocking_notice", lambda title, message: notices.append((title, message)))

    window._on_live_preflight_finished(0, 0)

    assert notices
    assert "preflight_runtime_resolution_mismatch" in notices[0][1]


def test_live_preflight_finish_uses_matching_launch_plan_on_blocked_preflight(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    payload = {
        "ready": False,
        "warnings": [],
        "errors": [
            "preflight_out_dir_not_fresh: Output dir already exists and is not empty: /tmp/live_infer. Choose a fresh --out-dir for an unambiguous live run."
        ],
        "launch_plan": {
            "selected_session_dir": str(tmp_path / "session"),
            "model_path": str(tmp_path / "finger_action_model.pt"),
            "scaler_path": str(tmp_path / "scaler.npz"),
            "out_dir": str(tmp_path / "live_infer"),
        },
        "launch_plan_resolution_succeeded": True,
        "launch_plan_resolved_before_validation": True,
        "launch_plan_contract_status": "ok",
        "launch_plan_validation_errors": [
            "preflight_out_dir_not_fresh: Output dir already exists and is not empty: /tmp/live_infer. Choose a fresh --out-dir for an unambiguous live run."
        ],
        "effective_contract": {},
    }
    report_path.write_text(json.dumps(payload))

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
        resolved_launch_plan=SimpleNamespace(
            selected_session_dir=tmp_path / "session",
            model_path=tmp_path / "finger_action_model.pt",
            scaler_path=tmp_path / "scaler.npz",
            out_dir=tmp_path / "live_infer",
        ),
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = []

    notices = []
    monkeypatch.setattr(
        window,
        "_show_blocking_notice",
        lambda title, message: notices.append((title, message)),
    )

    window._on_live_preflight_finished(0, 0)

    assert notices
    assert notices[0][0] == "Step 7 Not Ready"
    assert "preflight_out_dir_not_fresh" in notices[0][1]
    assert "preflight_runtime_resolution_mismatch" not in notices[0][1]
    assert "preflight_launch_plan_empty" not in notices[0][1]


def test_live_preflight_finish_reports_missing_launch_plan_distinctly(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    report_path.write_text(
        json.dumps(
            {
                "ready": True,
                "warnings": [],
                "errors": [],
                "effective_contract": {},
            }
        )
    )

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
        resolved_launch_plan=SimpleNamespace(
            selected_session_dir=tmp_path / "session",
            model_path=tmp_path / "finger_action_model.pt",
            scaler_path=tmp_path / "scaler.npz",
            out_dir=tmp_path / "live_infer",
        ),
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = []

    notices = []
    monkeypatch.setattr(
        window,
        "_show_blocking_notice",
        lambda title, message: notices.append((title, message)),
    )

    window._on_live_preflight_finished(0, 0)

    assert notices
    assert "preflight_launch_plan_missing" in notices[0][1]
    assert "launch_plan_present=False" in notices[0][1]


def test_live_preflight_finish_reports_null_launch_plan_as_empty(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    report_path.write_text(
        json.dumps(
            {
                "ready": True,
                "warnings": [],
                "errors": [],
                "launch_plan": None,
                "effective_contract": {},
            }
        )
    )

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
        resolved_launch_plan=SimpleNamespace(
            selected_session_dir=tmp_path / "session",
            model_path=tmp_path / "finger_action_model.pt",
            scaler_path=tmp_path / "scaler.npz",
            out_dir=tmp_path / "live_infer",
        ),
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = []

    notices = []
    monkeypatch.setattr(
        window,
        "_show_blocking_notice",
        lambda title, message: notices.append((title, message)),
    )

    window._on_live_preflight_finished(0, 0)

    assert notices
    assert "preflight_launch_plan_empty" in notices[0][1]
    assert "launch_plan_key_exists=True" in notices[0][1]
    assert "launch_plan_is_empty=True" in notices[0][1]


def test_live_preflight_finish_reports_empty_launch_plan_as_empty(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    report_path.write_text(
        json.dumps(
            {
                "ready": True,
                "warnings": [],
                "errors": [],
                "launch_plan": {},
                "effective_contract": {},
            }
        )
    )

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
        resolved_launch_plan=SimpleNamespace(
            selected_session_dir=tmp_path / "session",
            model_path=tmp_path / "finger_action_model.pt",
            scaler_path=tmp_path / "scaler.npz",
            out_dir=tmp_path / "live_infer",
        ),
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = []

    notices = []
    monkeypatch.setattr(
        window,
        "_show_blocking_notice",
        lambda title, message: notices.append((title, message)),
    )

    window._on_live_preflight_finished(0, 0)

    assert notices
    assert "preflight_launch_plan_empty" in notices[0][1]
    assert "launch_plan_key_exists=True" in notices[0][1]
    assert "launch_plan_is_empty=True" in notices[0][1]


def test_live_preflight_finish_reports_launch_plan_schema_mismatch_distinctly(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "live_preflight_report.json"
    report_path.write_text(
        json.dumps(
            {
                "ready": True,
                "warnings": [],
                "errors": [],
                "launch_plan": {"out_dir": ""},
                "effective_contract": {},
            }
        )
    )

    launch = ui_mod.PreparedLiveInferLaunch(
        args=[],
        cwd=str(tmp_path),
        env={},
        config_path=tmp_path / "infer.json",
        config_payload={},
        settings={},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
        preflight_report_path=report_path,
        resolved_launch_plan=SimpleNamespace(
            selected_session_dir=tmp_path / "session",
            model_path=tmp_path / "finger_action_model.pt",
            scaler_path=tmp_path / "scaler.npz",
            out_dir=tmp_path / "live_infer",
        ),
    )
    window._pending_live_launch = launch
    window._live_preflight_lines = []

    notices = []
    monkeypatch.setattr(
        window,
        "_show_blocking_notice",
        lambda title, message: notices.append((title, message)),
    )

    window._on_live_preflight_finished(0, 0)

    assert notices
    assert "preflight_launch_plan_schema_mismatch" in notices[0][1]
    assert "missing_keys" in notices[0][1]
    assert "empty_keys" in notices[0][1]


def test_execute_prepared_live_infer_launch_uses_frozen_args_not_current_widgets(
    window, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    captured = {}
    runner = SimpleNamespace(
        start=lambda exe, args, cwd=None, env=None: captured.update(
            {"exe": exe, "args": list(args), "cwd": cwd, "env": dict(env or {})}
        )
    )
    monkeypatch.setattr(window, "runner", runner)
    monkeypatch.setattr(window, "_write_session_snapshot", lambda *args, **kwargs: None)
    monkeypatch.setattr(window, "_set_step_status", lambda *args, **kwargs: None)
    monkeypatch.setattr(window, "_append_log", lambda *args, **kwargs: None)

    window.fields["infer"]["model_path"].setText(str(tmp_path / "ui_model.pt"))
    launch = ui_mod.PreparedLiveInferLaunch(
        args=["/tmp/7_live_infer_and_actuate.py", "--config", str(tmp_path / "frozen.json"), "--model-path", str(tmp_path / "frozen_model.pt")],
        cwd=str(tmp_path),
        env={"LSL_SOURCE_ID": "source-123"},
        config_path=tmp_path / "frozen.json",
        config_payload={},
        settings={"model_path": str(tmp_path / "frozen_model.pt")},
        subject_dir=tmp_path,
        session_dir=tmp_path / "session",
    )

    window._execute_prepared_live_infer_launch(launch)

    assert str(tmp_path / "frozen_model.pt") in captured["args"]
    assert str(tmp_path / "ui_model.pt") not in captured["args"]


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
        window,
        "_show_warning_message",
        lambda title, message: pytest.fail(
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

    def fake_warning(title, message):
        captured["title"] = title
        captured["message"] = message
        return 0

    monkeypatch.setattr(window, "_show_warning_message", fake_warning)

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
