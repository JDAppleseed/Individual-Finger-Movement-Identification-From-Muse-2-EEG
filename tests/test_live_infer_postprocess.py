import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from utils.inference import InferenceConfig, InferenceEngine
from utils.postprocess import PostprocessSettings, PostprocessState
from visualization.live_viz import parse_viz_line


def _load_live_module():
    module_path = Path(__file__).resolve().parents[1] / "7_live_infer_and_actuate.py"
    spec = importlib.util.spec_from_file_location("live_infer", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_postprocess_decision_deterministic():
    mod = _load_live_module()
    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.5,
        threshold_finger=0.5,
        adjacency_enabled=False,
    )
    state = PostprocessState()
    action_probs = np.array([0.1, 0.8, 0.1], dtype=float)
    finger_probs = np.array([0.05, 0.05, 0.85, 0.02, 0.02, 0.01], dtype=float)

    out = mod._postprocess_decision(
        action_probs,
        finger_probs,
        enabled=True,
        settings=settings,
        state=state,
    )

    assert out["committed_action_id"] == 1
    assert out["committed_finger_id"] == 2


class _DummyMCModel(torch.nn.Module):
    def forward(self, x):
        finger_logits = torch.tensor(
            [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        action_logits = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
        return finger_logits, action_logits

    def mc_forward(self, x, passes=20):
        return {
            "finger_mean": torch.tensor(
                [[0.05, 0.05, 0.80, 0.04, 0.03, 0.03]], dtype=torch.float32
            ),
            "action_mean": torch.tensor([[0.10, 0.75, 0.15]], dtype=torch.float32),
            "finger_std": torch.tensor(
                [[0.01, 0.01, 0.05, 0.01, 0.01, 0.01]], dtype=torch.float32
            ),
            "action_std": torch.tensor([[0.02, 0.10, 0.03]], dtype=torch.float32),
        }


def test_predict_window_uses_inference_engine_backend():
    mod = _load_live_module()
    model = _DummyMCModel()
    engine = InferenceEngine(
        model=model,
        normalizer=None,
        device=torch.device("cpu"),
        action_names={},
        finger_names={},
        config=InferenceConfig(
            base_threshold=0.7,
            uncertainty_weight=0.5,
            stability_frames=2,
            mc_passes=5,
        ),
    )

    out = mod._predict_window(
        np.zeros((64, 4), dtype=np.float32),
        scaler=None,
        model=model,
        device=torch.device("cpu"),
        inference_engine=engine,
        emit_viz=False,
    )

    assert out["backend"] == "inference_engine"
    assert np.isclose(out["action_probs"][1], 0.75)
    assert np.isclose(out["finger_probs"][2], 0.80)
    assert np.isclose(out["action_uncertainty"], np.mean([0.02, 0.10, 0.03]))
    assert np.isclose(
        out["finger_uncertainty"], np.mean([0.01, 0.01, 0.05, 0.01, 0.01, 0.01])
    )
    assert out["adaptive_threshold"] > 0.7


def test_compute_actuation_speed_scalar_uses_uncertainty():
    mod = _load_live_module()
    mapper = mod._build_actuation_speed_mapper(
        type(
            "Args",
            (),
            {"modulate_actuation_speed": True, "actuation_speed_gamma": 1.0},
        )()
    )

    speed = mod._compute_actuation_speed_scalar(
        decision_prob=0.8,
        action_uncertainty=0.25,
        speed_mapper=mapper,
    )

    assert np.isclose(speed, 0.6)


def test_parser_accepts_ui_hyphenated_flags():
    mod = _load_live_module()
    parser, _ = mod._build_arg_parser()

    args = parser.parse_args(
        [
            "--config",
            "infer.json",
            "--enable-actuation",
            "--window-sec",
            "0.5",
            "--allow-drop",
            "--latency-threshold-ms",
            "333",
        ]
    )

    assert args.enable_actuation is True
    assert np.isclose(args.window_sec, 0.5)
    assert args.allow_drop is True
    assert np.isclose(args.latency_threshold_ms, 333.0)


def test_resolve_lsl_inlet_retries_and_prefers_source_id(monkeypatch):
    mod = _load_live_module()

    class _FakeStream:
        def __init__(self, name: str, stream_type: str, source_id: str):
            self._name = name
            self._type = stream_type
            self._source_id = source_id

        def name(self):
            return self._name

        def type(self):
            return self._type

        def source_id(self):
            return self._source_id

        def uid(self):
            return f"uid-{self._source_id}"

        def channel_count(self):
            return 4

        def nominal_srate(self):
            return 256.0

    class _FakeInlet:
        def __init__(self, info, max_chunklen=64):
            self.info_obj = info
            self.max_chunklen = max_chunklen

        def pull_sample(self, timeout=0.0):
            return [0.0, 0.0, 0.0, 0.0], 1.0

    calls = {"count": 0}

    def _fake_resolve_streams(wait_time=0.0):
        calls["count"] += 1
        if calls["count"] < 3:
            return []
        return [
            _FakeStream("Muse2-EEG-2-M16", "EEG", "wrong"),
            _FakeStream("Muse2-EEG-2-M16", "EEG", "wanted"),
        ]

    monkeypatch.setattr(mod, "LSL_AVAILABLE", True)
    monkeypatch.setattr(mod, "resolve_streams", _fake_resolve_streams)
    monkeypatch.setattr(mod, "resolve_byprop", None)
    monkeypatch.setattr(mod, "StreamInlet", _FakeInlet)
    monkeypatch.setattr(mod.time, "sleep", lambda *_args, **_kwargs: None)

    inlet = mod._resolve_lsl_inlet(
        "Muse2-EEG-2-M16",
        "EEG",
        timeout_s=1.0,
        source_id="wanted",
    )

    assert isinstance(inlet, _FakeInlet)
    assert inlet.info_obj.source_id() == "wanted"
    assert calls["count"] >= 3


def test_choose_auto_serial_port_prefers_usb_modem():
    mod = _load_live_module()

    class _Port:
        def __init__(self, device: str, description: str = "", manufacturer: str = ""):
            self.device = device
            self.description = description
            self.manufacturer = manufacturer
            self.product = ""
            self.interface = ""
            self.name = Path(device).name
            self.vid = 0x2341
            self.pid = 0x0043

    chosen = mod._choose_auto_serial_port(
        [
            _Port("/dev/tty.Bluetooth-Incoming-Port", description="Bluetooth"),
            _Port("/dev/cu.usbmodem1101", description="Arduino Uno", manufacturer="Arduino"),
        ]
    )

    assert chosen == "/dev/cu.usbmodem1101"


def test_main_uses_config_model_override_with_session_dir(tmp_path, monkeypatch):
    mod = _load_live_module()
    session_dir = tmp_path / "live_session"
    (session_dir / "processed").mkdir(parents=True)
    model_path = tmp_path / "trained" / "finger_action_model.pt"
    scaler_path = tmp_path / "trained" / "scaler.npz"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    config_path = tmp_path / "infer.json"
    config_path.write_text(
        json.dumps(
            {
                "project_name": "P",
                "subject_id": "S",
                "session_id": "sess",
                "settings": {
                    "session_dir": str(session_dir),
                    "model_path": str(model_path),
                    "scaler_path": str(scaler_path),
                    "stream_name": "Muse2-EEG-2-M16",
                    "stream_type": "EEG",
                    "no_file_io": True,
                },
            }
        )
    )

    captured = {}

    class _DummyModel:
        def eval(self):
            return self

    def _fake_load_model_and_scaler(model_arg, scaler_arg, device=None):
        captured["model_path"] = model_arg
        captured["scaler_path"] = scaler_arg
        return _DummyModel(), object()

    def _stop_after_model_load(*_args, **_kwargs):
        raise RuntimeError("stop after model load")

    monkeypatch.setattr(mod, "load_model_and_scaler", _fake_load_model_and_scaler)
    monkeypatch.setattr(mod, "_resolve_lsl_inlet", _stop_after_model_load)
    monkeypatch.setattr(mod, "setup_logger", lambda **_kwargs: None)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["7_live_infer_and_actuate.py", "--config", str(config_path)],
    )

    with pytest.raises(RuntimeError, match="stop after model load"):
        mod.main()

    assert captured["model_path"] == str(model_path)
    assert captured["scaler_path"] == str(scaler_path)


def test_main_falls_back_to_latest_trained_sibling_session(tmp_path, monkeypatch):
    mod = _load_live_module()
    repo_root = tmp_path
    selected_session = repo_root / "Projects" / "P" / "subjects" / "S" / "sessions" / "live_only"
    trained_run = (
        repo_root
        / "Projects"
        / "P"
        / "subjects"
        / "S"
        / "sessions"
        / "combined_20260302_120000"
        / "processed"
        / "models"
        / "20260303_042421"
    )
    (selected_session / "processed").mkdir(parents=True)
    trained_run.mkdir(parents=True)
    model_path = trained_run / "finger_action_model.pt"
    scaler_path = trained_run / "scaler.npz"
    model_path.write_bytes(b"model")
    scaler_path.write_bytes(b"scaler")
    config_path = repo_root / "Projects" / "P" / "subjects" / "S" / "config" / "infer.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        json.dumps(
            {
                "project_name": "P",
                "subject_id": "S",
                "session_id": "live_only",
                "settings": {
                    "session_dir": str(selected_session),
                    "stream_name": "Muse2-EEG-S",
                    "stream_type": "EEG",
                    "no_file_io": True,
                },
            }
        )
    )

    captured = {}

    class _DummyModel:
        def eval(self):
            return self

    def _fake_load_model_and_scaler(model_arg, scaler_arg, device=None):
        captured["model_path"] = model_arg
        captured["scaler_path"] = scaler_arg
        return _DummyModel(), object()

    def _stop_after_model_load(*_args, **_kwargs):
        raise RuntimeError("stop after model load")

    monkeypatch.setattr(mod, "load_model_and_scaler", _fake_load_model_and_scaler)
    monkeypatch.setattr(mod, "_resolve_lsl_inlet", _stop_after_model_load)
    monkeypatch.setattr(mod, "setup_logger", lambda **_kwargs: None)
    monkeypatch.setattr(
        mod.sys,
        "argv",
        ["7_live_infer_and_actuate.py", "--config", str(config_path), "--session-dir", str(selected_session)],
    )

    with pytest.raises(RuntimeError, match="stop after model load"):
        mod.main()

    assert captured["model_path"] == str(model_path)
    assert captured["scaler_path"] == str(scaler_path)


def test_parse_viz_line_accepts_vizjson():
    payload = parse_viz_line(
        'VIZJSON {"t":1.25,"hidden_mag":0.42,"finger_probs":[[0.1,0.9]],"action_probs":[[0.7,0.3]]}'
    )

    assert payload is not None
    assert payload["t"] == 1.25
    assert payload["hidden_mag"] == 0.42
    assert payload["finger_probs"][0][1] == 0.9
