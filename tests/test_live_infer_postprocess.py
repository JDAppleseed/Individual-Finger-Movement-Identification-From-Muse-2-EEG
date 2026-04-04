import importlib.util
import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest
import torch

from app.config_model import default_infer_settings
from utils.default_recipe import LIVE_INFER_RECIPE_DEFAULTS
from utils.inference import InferenceConfig, InferenceEngine
from utils.postprocess import PostprocessSettings, PostprocessState
from utils.runtime_utils import TemperatureScalingState
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
    assert out["finger_gate_ok"] is True
    assert out["committed_pair_valid"] is True


def test_postprocess_decision_maps_active_only_finger_head():
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
    finger_probs = np.array([0.05, 0.85, 0.02, 0.02, 0.06], dtype=float)

    out = mod._postprocess_decision(
        action_probs,
        finger_probs,
        enabled=True,
        settings=settings,
        state=state,
    )

    assert out["committed_action_id"] == 1
    assert out["committed_finger_id"] == 2
    assert out["raw_top_finger_id"] == 2
    assert out["committed_pair_valid"] is True


def test_postprocess_decision_raw_argmax_gates_rest_to_none():
    mod = _load_live_module()
    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.0,
        adjacency_enabled=False,
    )
    state = PostprocessState()
    action_probs = np.array([0.9, 0.05, 0.05], dtype=float)
    finger_probs = np.array([0.01, 0.02, 0.90, 0.03, 0.02, 0.02], dtype=float)

    out = mod._postprocess_decision(
        action_probs,
        finger_probs,
        enabled=False,
        settings=settings,
        state=state,
    )

    assert out["raw_top_action_id"] == 0
    assert out["raw_top_finger_id"] == 2
    assert out["committed_action_id"] == 0
    assert out["committed_finger_id"] == 0
    assert np.isclose(out["finger_conf"], 0.01)
    assert out["finger_gate_ok"] is True
    assert out["committed_pair_valid"] is True
    assert out["decision_reason"] == "raw_argmax_gated"


def test_postprocess_decision_low_finger_conf_blocks_actuation_without_rewriting_label():
    mod = _load_live_module()
    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.60,
        adjacency_enabled=False,
    )
    state = PostprocessState()
    action_probs = np.array([0.05, 0.90, 0.05], dtype=float)
    finger_probs = np.array([0.40, 0.30, 0.20, 0.05, 0.03, 0.02], dtype=float)

    out = mod._postprocess_decision(
        action_probs,
        finger_probs,
        enabled=True,
        settings=settings,
        state=state,
    )

    assert out["committed_action_id"] == 1
    assert out["committed_finger_id"] == 1
    assert np.isclose(out["finger_conf"], 0.30)
    assert out["finger_gate_ok"] is False
    assert out["committed_pair_valid"] is True
    assert out["decision_reason"] == "commit"


def test_postprocess_decision_low_applicability_blocks_actuation_without_rewriting_label():
    mod = _load_live_module()
    settings = PostprocessSettings(
        smoothing_enabled=False,
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.20,
        threshold_applicability=0.75,
        adjacency_enabled=False,
    )
    state = PostprocessState()
    action_probs = np.array([0.05, 0.90, 0.05], dtype=float)
    finger_probs = np.array([0.10, 0.72, 0.10, 0.04, 0.02, 0.02], dtype=float)

    out = mod._postprocess_decision(
        action_probs,
        finger_probs,
        enabled=True,
        settings=settings,
        state=state,
        finger_applicable_prob=0.40,
    )

    assert out["committed_action_id"] == 1
    assert out["committed_finger_id"] == 1
    assert out["finger_gate_ok"] is True
    assert out["applicability_gate_ok"] is False
    assert out["committed_pair_valid"] is True
    assert out["decision_reason"] == "commit"


def test_postprocess_decision_nonfinite_probs_reset_state_and_recover():
    mod = _load_live_module()
    settings = PostprocessSettings(
        smoothing_enabled=True,
        smoothing_method="ema",
        hysteresis_enabled=False,
        threshold_action=0.0,
        threshold_finger=0.0,
        adjacency_enabled=False,
    )
    state = PostprocessState()

    healthy = mod._postprocess_decision(
        np.array([0.05, 0.90, 0.05], dtype=float),
        np.array([0.10, 0.72, 0.10, 0.04, 0.02], dtype=float),
        enabled=True,
        settings=settings,
        state=state,
        finger_applicable_prob=0.80,
    )
    assert healthy["committed_action_id"] == 1
    assert healthy["committed_finger_id"] == 2
    assert state.ema_action is not None

    poisoned = mod._postprocess_decision(
        np.array([np.nan, np.nan, np.nan], dtype=float),
        np.array([np.nan, np.nan, np.nan, np.nan, np.nan], dtype=float),
        enabled=True,
        settings=settings,
        state=state,
        finger_applicable_prob=float("nan"),
    )
    assert poisoned["committed_action_id"] == 0
    assert poisoned["committed_finger_id"] == 0
    assert poisoned["decision_reason"] == "nonfinite_probs"
    assert state.ema_action is None
    assert state.ema_finger is None
    assert state.frames_in_state == 0

    recovered = mod._postprocess_decision(
        np.array([0.05, 0.88, 0.07], dtype=float),
        np.array([0.08, 0.76, 0.08, 0.05, 0.03], dtype=float),
        enabled=True,
        settings=settings,
        state=state,
        finger_applicable_prob=0.85,
    )
    assert recovered["committed_action_id"] == 1
    assert recovered["committed_finger_id"] == 2
    assert recovered["decision_reason"] == "commit"


def test_resample_window_rejects_nonfinite_values():
    mod = _load_live_module()
    times = np.array([0.00, 0.01, 0.02, 0.03], dtype=float)
    values = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.1, np.nan, 3.1, 4.1],
            [1.2, 2.2, 3.2, 4.2],
            [1.3, 2.3, 3.3, 4.3],
        ],
        dtype=float,
    )

    window = mod._resample_window(
        times,
        values,
        start_s=0.00,
        end_s=0.03,
        target_fs=100.0,
    )

    assert window is None


def test_rest_finger_bias_correction_reduces_pinky_bias_after_rest_warmup():
    mod = _load_live_module()
    correction = mod.RestFingerBiasCorrection(
        enabled=True,
        min_rest_windows=3,
        strength=1.5,
    )
    rest_action_probs = np.array([0.90, 0.05, 0.05], dtype=float)

    for _ in range(3):
        became_ready = correction.update(
            rest_action_probs,
            np.array([0.02, 0.10, 0.12, 0.06, 0.70], dtype=float),
        )

    assert became_ready is True
    assert correction.ready is True

    raw = np.array([0.05, 0.11, 0.14, 0.10, 0.60], dtype=float)
    corrected = correction.apply(raw)

    assert np.isclose(np.sum(corrected), 1.0)
    assert corrected[-1] < raw[-1]
    assert corrected[1] > raw[1]


def test_sanitize_live_window_masks_single_bad_channel_without_quality_gate():
    mod = _load_live_module()
    window = np.zeros((64, 4), dtype=np.float32)
    window[:, 1] = 5.0

    out = mod._sanitize_live_window(
        window,
        scaler=None,
        enabled=True,
        input_clip_abs_z=6.0,
        bad_channel_rms_z=4.0,
        bad_channel_abs_p95_z=6.0,
        bad_channel_clipped_frac=0.05,
        bad_window_clipped_frac=0.10,
        bad_window_max_masked_channels=1,
    )

    assert out.window_quality_bad is False
    assert out.quality_bad_reason is None
    assert out.bad_channel_ids == (1,)
    assert out.masked_channel_ids == (1,)
    assert np.allclose(out.prepared_window[:, 1], 0.0)


def test_sanitize_live_window_quality_gates_two_bad_channels():
    mod = _load_live_module()
    window = np.zeros((64, 4), dtype=np.float32)
    window[:, 1] = 5.0
    window[:, 2] = 5.0

    out = mod._sanitize_live_window(
        window,
        scaler=None,
        enabled=True,
        input_clip_abs_z=6.0,
        bad_channel_rms_z=4.0,
        bad_channel_abs_p95_z=6.0,
        bad_channel_clipped_frac=0.05,
        bad_window_clipped_frac=0.10,
        bad_window_max_masked_channels=1,
    )

    assert out.window_quality_bad is True
    assert out.quality_bad_reason == "too_many_bad_channels"
    assert out.bad_channel_ids == (1, 2)
    assert out.masked_channel_ids == ()


def test_live_infer_defaults_match_best_live_profile():
    mod = _load_live_module()
    _, defaults = mod._build_arg_parser()
    config_defaults = default_infer_settings()

    assert PostprocessSettings().finger_mode == "raw"
    assert PostprocessSettings().threshold_action == pytest.approx(0.0)
    assert PostprocessSettings().threshold_finger == pytest.approx(0.0)
    assert PostprocessSettings().smoothing_enabled is False
    assert defaults["postprocess"] is False
    assert defaults["finger_mode"] == "raw"
    assert defaults["window_sec"] == pytest.approx(0.25)
    assert defaults["hop_sec"] == pytest.approx(0.05)
    assert defaults["latency_threshold_ms"] == pytest.approx(750.0)
    assert defaults["smoothing_method"] == "ema"
    assert defaults["smoothing_enabled"] is False
    assert defaults["smoothing_window"] == 1
    assert defaults["hysteresis_enabled"] is False
    assert defaults["hysteresis_frames"] == 3
    assert defaults["threshold_action"] == pytest.approx(0.0)
    assert defaults["threshold_finger"] == pytest.approx(0.0)
    assert defaults["threshold_applicability"] == pytest.approx(
        LIVE_INFER_RECIPE_DEFAULTS["threshold_applicability"]
    )
    assert defaults["adjacency_enabled"] is False
    assert defaults["hysteresis_margin"] == pytest.approx(0.05)
    assert defaults["finger_delta"] == pytest.approx(0.05)
    assert defaults["rest_bias_correction_enabled"] is False
    assert defaults["rest_bias_strength"] == pytest.approx(1.5)
    assert defaults["rest_bias_min_windows"] == 10
    assert defaults["live_quality_enabled"] is True
    assert defaults["input_clip_abs_z"] == pytest.approx(6.0)
    assert defaults["bad_channel_rms_z"] == pytest.approx(4.0)
    assert defaults["bad_channel_abs_p95_z"] == pytest.approx(6.0)
    assert defaults["bad_channel_clipped_frac"] == pytest.approx(0.05)
    assert defaults["bad_window_clipped_frac"] == pytest.approx(0.10)
    assert defaults["bad_window_max_masked_channels"] == 1
    assert defaults["serial_port"] is None
    assert defaults["serial_baud"] == 9600
    assert defaults["actuation_min_prob"] == pytest.approx(0.2)
    assert defaults["actuation_stability"] == 2
    assert defaults["actuation_cooldown_ms"] == 0
    assert defaults["actuation_repeat_ms"] == 100
    assert defaults["actuation_min_speed"] == pytest.approx(0.5)
    assert defaults["modulate_actuation_speed"] is False
    assert defaults["actuation_speed_gamma"] == pytest.approx(1.0)
    assert config_defaults["postprocess"] is False
    assert config_defaults["finger_mode"] == "raw"
    assert config_defaults["window_sec"] == pytest.approx(0.25)
    assert config_defaults["hop_sec"] == pytest.approx(0.05)
    assert config_defaults["latency_threshold_ms"] == pytest.approx(750.0)
    assert config_defaults["smoothing_method"] == "ema"
    assert config_defaults["smoothing_enabled"] is False
    assert config_defaults["smoothing_window"] == 1
    assert config_defaults["hysteresis_enabled"] is False
    assert config_defaults["hysteresis_frames"] == 3
    assert config_defaults["threshold_action"] == pytest.approx(0.0)
    assert config_defaults["threshold_finger"] == pytest.approx(0.0)
    assert config_defaults["threshold_applicability"] == pytest.approx(
        LIVE_INFER_RECIPE_DEFAULTS["threshold_applicability"]
    )
    assert config_defaults["adjacency_enabled"] is False
    assert config_defaults["hysteresis_margin"] == pytest.approx(0.05)
    assert config_defaults["finger_delta"] == pytest.approx(0.05)
    assert config_defaults["rest_bias_correction_enabled"] is False
    assert config_defaults["rest_bias_strength"] == pytest.approx(1.5)
    assert config_defaults["rest_bias_min_windows"] == 10
    assert config_defaults["live_quality_enabled"] is True
    assert config_defaults["input_clip_abs_z"] == pytest.approx(6.0)
    assert config_defaults["bad_channel_rms_z"] == pytest.approx(4.0)
    assert config_defaults["bad_channel_abs_p95_z"] == pytest.approx(6.0)
    assert config_defaults["bad_channel_clipped_frac"] == pytest.approx(0.05)
    assert config_defaults["bad_window_clipped_frac"] == pytest.approx(0.10)
    assert config_defaults["bad_window_max_masked_channels"] == 1
    assert config_defaults["serial_port"] is None
    assert config_defaults["serial_baud"] == 9600
    assert config_defaults["actuation_min_prob"] == pytest.approx(0.2)
    assert config_defaults["actuation_stability"] == 2
    assert config_defaults["actuation_cooldown_ms"] == 0
    assert config_defaults["actuation_repeat_ms"] == 100
    assert config_defaults["actuation_min_speed"] == pytest.approx(0.5)
    assert config_defaults["modulate_actuation_speed"] is False
    assert config_defaults["actuation_speed_gamma"] == pytest.approx(1.0)


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


class _DummyRestMCModel(torch.nn.Module):
    def forward(self, x):
        finger_logits = torch.tensor(
            [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        action_logits = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        return finger_logits, action_logits

    def mc_forward(self, x, passes=20):
        return {
            "finger_mean": torch.tensor(
                [[0.05, 0.05, 0.80, 0.04, 0.03, 0.03]], dtype=torch.float32
            ),
            "action_mean": torch.tensor([[0.75, 0.10, 0.15]], dtype=torch.float32),
            "finger_std": torch.tensor(
                [[0.01, 0.01, 0.05, 0.01, 0.01, 0.01]], dtype=torch.float32
            ),
            "action_std": torch.tensor([[0.10, 0.02, 0.03]], dtype=torch.float32),
        }


class _DummyRestMCModelActiveHead(torch.nn.Module):
    def forward(self, x):
        finger_logits = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
        action_logits = torch.tensor([[2.0, 0.0, 0.0]], dtype=torch.float32)
        return finger_logits, action_logits

    def mc_forward(self, x, passes=20):
        return {
            "finger_mean": torch.tensor(
                [[0.05, 0.80, 0.04, 0.03, 0.08]], dtype=torch.float32
            ),
            "action_mean": torch.tensor([[0.75, 0.10, 0.15]], dtype=torch.float32),
            "finger_std": torch.tensor(
                [[0.01, 0.05, 0.01, 0.01, 0.01]], dtype=torch.float32
            ),
            "action_std": torch.tensor([[0.10, 0.02, 0.03]], dtype=torch.float32),
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


class _DirectLogitModel(torch.nn.Module):
    def forward(self, x):
        finger_logits = torch.tensor(
            [[0.0, 0.0, 2.0, 0.0, 0.0, 0.0]], dtype=torch.float32
        )
        action_logits = torch.tensor([[0.0, 2.0, 0.0]], dtype=torch.float32)
        return finger_logits, action_logits


def test_predict_window_direct_applies_temperature_scaling():
    mod = _load_live_module()
    model = _DirectLogitModel()

    out_raw = mod._predict_window(
        np.zeros((64, 4), dtype=np.float32),
        scaler=None,
        model=model,
        device=torch.device("cpu"),
        inference_engine=None,
        temperature_state=None,
        emit_viz=False,
    )
    out_temp = mod._predict_window(
        np.zeros((64, 4), dtype=np.float32),
        scaler=None,
        model=model,
        device=torch.device("cpu"),
        inference_engine=None,
        temperature_state=TemperatureScalingState(
            action_temperature=2.0,
            finger_temperature=2.0,
            source="test",
        ),
        emit_viz=False,
    )

    assert out_raw["action_probs"][1] > out_temp["action_probs"][1]
    assert out_raw["finger_probs"][2] > out_temp["finger_probs"][2]


def test_inference_engine_predict_gates_rest_to_none():
    model = _DummyRestMCModel()
    engine = InferenceEngine(
        model=model,
        normalizer=None,
        device=torch.device("cpu"),
        action_names={0: "REST"},
        finger_names={0: "NONE", 2: "INDEX"},
        config=InferenceConfig(
            base_threshold=0.7,
            uncertainty_weight=0.5,
            stability_frames=1,
            mc_passes=5,
        ),
    )

    prediction, safety, diagnostics = engine.predict(
        np.zeros((64, 4), dtype=np.float32)
    )

    assert prediction["action_id"] == 0
    assert prediction["finger_id"] == 0
    assert np.isclose(prediction["finger_confidence"], 0.05)
    assert safety["allow_actuation"] is True
    assert diagnostics["health_score"] >= 0.0


def test_inference_engine_predict_gates_rest_to_none_with_active_finger_head():
    model = _DummyRestMCModelActiveHead()
    engine = InferenceEngine(
        model=model,
        normalizer=None,
        device=torch.device("cpu"),
        action_names={0: "REST"},
        finger_names={0: "NONE", 2: "INDEX"},
        config=InferenceConfig(
            base_threshold=0.7,
            uncertainty_weight=0.5,
            stability_frames=1,
            mc_passes=5,
        ),
    )

    prediction, safety, diagnostics = engine.predict(
        np.zeros((64, 4), dtype=np.float32)
    )

    assert prediction["action_id"] == 0
    assert prediction["finger_id"] == 0
    assert np.isclose(prediction["finger_confidence"], 0.0)
    assert safety["allow_actuation"] is True
    assert diagnostics["health_score"] >= 0.0


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


def test_compute_actuation_speed_scalar_applies_min_speed_floor():
    mod = _load_live_module()
    mapper = mod._build_actuation_speed_mapper(
        type(
            "Args",
            (),
            {"modulate_actuation_speed": True, "actuation_speed_gamma": 1.0},
        )()
    )

    speed = mod._compute_actuation_speed_scalar(
        decision_prob=0.2,
        action_uncertainty=0.0,
        speed_mapper=mapper,
        min_speed=0.45,
    )

    assert np.isclose(speed, 0.45)


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


def test_require_deployable_run_rejects_legacy_finger_head(tmp_path: Path):
    mod = _load_live_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_config.json").write_text(
        json.dumps({"active_finger_head": False})
    )
    torch.save(
        {
            "finger_head.weight": torch.zeros((6, 4), dtype=torch.float32),
            "action_head.weight": torch.zeros((3, 4), dtype=torch.float32),
        },
        run_dir / "finger_action_model.pt",
    )

    with pytest.raises(RuntimeError, match="Deployment model requirement failed"):
        mod._require_deployable_run(run_dir)


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
        def __init__(
            self,
            device: str,
            description: str = "",
            manufacturer: str = "",
            *,
            vid=None,
            pid=None,
        ):
            self.device = device
            self.description = description
            self.manufacturer = manufacturer
            self.product = ""
            self.interface = ""
            self.name = Path(device).name
            self.vid = vid
            self.pid = pid

    chosen = mod._choose_auto_serial_port(
        [
            _Port("/dev/tty.Bluetooth-Incoming-Port", description="Bluetooth"),
            _Port(
                "/dev/cu.usbmodem1101",
                description="Arduino Uno",
                manufacturer="Arduino",
                vid=0x2341,
                pid=0x0043,
            ),
        ]
    )

    assert chosen == "/dev/cu.usbmodem1101"


def test_choose_auto_serial_port_rejects_system_console_ports():
    mod = _load_live_module()

    class _Port:
        def __init__(self, device: str, description: str = ""):
            self.device = device
            self.description = description
            self.manufacturer = ""
            self.product = ""
            self.interface = ""
            self.name = Path(device).name
            self.vid = None
            self.pid = None

    chosen = mod._choose_auto_serial_port(
        [
            _Port("/dev/cu.debug-console"),
            _Port("/dev/tty.Bluetooth-Incoming-Port", description="Bluetooth"),
        ]
    )

    assert chosen is None


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


def test_debounced_should_send_allows_repeat_for_same_command(monkeypatch):
    mod = _load_live_module()
    decision = mod.ActuationDecision(finger_id=1, action_id=1, prob=0.9)
    monkeypatch.setattr(mod.time, "monotonic", lambda: 10.0)

    assert (
        mod._debounced_should_send(
            decision=decision,
            last_sent=(1, 1),
            stable_count=1,
            required_stability=1,
            last_send_ts=9.0,
            cooldown_ms=250,
            repeat_same_ms=500,
        )
        is True
    )


def test_resolve_live_sample_time_prefers_lsl_spacing_over_chunk_arrival_time():
    mod = _load_live_module()

    stream_origin_mono = None
    stream_origin_lsl = None
    prev_lsl_mono = None
    resolved = []
    for lsl_ts in [100.0, 100.00390625, 100.0078125]:
        out = mod._resolve_live_sample_time(
            lsl_ts=lsl_ts,
            sample_mono=1.0,
            stream_origin_mono=stream_origin_mono,
            stream_origin_lsl=stream_origin_lsl,
            prev_lsl_mono=prev_lsl_mono,
        )
        time_s, _, _, stream_origin_mono, stream_origin_lsl, prev_lsl_mono = out
        resolved.append(time_s)

    assert np.allclose(resolved, [0.0, 0.00390625, 0.0078125])


def test_resolve_live_sample_time_falls_back_when_lsl_missing():
    mod = _load_live_module()

    out0 = mod._resolve_live_sample_time(
        lsl_ts=float("nan"),
        sample_mono=10.0,
        stream_origin_mono=None,
        stream_origin_lsl=None,
        prev_lsl_mono=None,
    )
    out1 = mod._resolve_live_sample_time(
        lsl_ts=float("nan"),
        sample_mono=10.25,
        stream_origin_mono=out0[3],
        stream_origin_lsl=out0[4],
        prev_lsl_mono=out0[5],
    )

    assert np.isclose(out0[0], 0.0)
    assert np.isclose(out1[0], 0.25)


def test_estimate_window_center_mono_tracks_latest_sample_clock():
    mod = _load_live_module()

    center = mod._estimate_window_center_mono(
        latest_sample_mono=12.0,
        latest_stream_time_s=5.0,
        window_center_stream_s=4.75,
    )

    assert np.isclose(center, 11.75)


def test_latency_gate_rejects_negative_and_stale_predictions():
    mod = _load_live_module()

    assert mod._latency_gate_passed(35.0, 750.0) is True
    assert mod._latency_gate_passed(-75.0, 750.0) is False
    assert mod._latency_gate_passed(900.0, 750.0) is False


def test_resolve_actuation_candidate_requires_exact_pair_match():
    mod = _load_live_module()

    history = [
        mod.ActuationDecision(finger_id=1, action_id=1, prob=0.9),
        mod.ActuationDecision(finger_id=1, action_id=2, prob=0.9),
    ]

    out = mod._resolve_actuation_candidate(history, required_finger_stability=2)

    assert out["decision"].finger_id == 0
    assert out["decision"].action_id == 0
    assert out["reason"] == "pair_stability"


def test_resolve_actuation_candidate_returns_exact_pair_when_tail_matches():
    mod = _load_live_module()

    history = [
        mod.ActuationDecision(finger_id=2, action_id=1, prob=0.8),
        mod.ActuationDecision(finger_id=2, action_id=1, prob=0.9),
    ]

    out = mod._resolve_actuation_candidate(history, required_finger_stability=2)

    assert out["decision"].finger_id == 2
    assert out["decision"].action_id == 1
    assert out["reason"] == "exact_pair_stability"


def test_resolve_live_actuation_vote_ignores_quality_window_without_advancing_history():
    mod = _load_live_module()
    history = deque(maxlen=2)

    first = mod._resolve_live_actuation_vote(
        history,
        mod.ActuationDecision(finger_id=3, action_id=2, prob=0.82),
        required_pair_stability=2,
        ignore_window=False,
    )
    skipped = mod._resolve_live_actuation_vote(
        history,
        mod.ActuationDecision(finger_id=4, action_id=2, prob=0.97),
        required_pair_stability=2,
        ignore_window=True,
    )
    second = mod._resolve_live_actuation_vote(
        history,
        mod.ActuationDecision(finger_id=3, action_id=2, prob=0.88),
        required_pair_stability=2,
        ignore_window=False,
    )

    assert first["reason"] == "pair_stability"
    assert skipped["reason"] == "quality_gate"
    assert len(history) == 2
    assert second["decision"].finger_id == 3
    assert second["decision"].action_id == 2
    assert second["reason"] == "exact_pair_stability"


def test_top_counter_snapshot_returns_most_common_items():
    mod = _load_live_module()

    snapshot = mod._top_counter_snapshot(
        mod.collections.Counter({1: 7, 2: 3, 5: 11}), top_k=2
    )

    assert snapshot == {"5": 11, "1": 7}


def test_actuation_command_shaper_holds_unstable_changes():
    mod = _load_live_module()
    shaper = mod._build_actuation_command_shaper(
        type(
            "Args",
            (),
            {
                "actuation_min_prob": 0.75,
                "actuation_speed_gamma": 1.0,
                "actuation_cooldown_ms": 250,
                "hop_sec": 0.05,
                "actuation_stability": 3,
            },
        )()
    )

    steady = shaper.shape(
        action_id=1,
        finger_id=2,
        action_conf=0.9,
        timestamp_stream_ms=1000,
        stability_ok=True,
        timebase_ms=1000,
    )
    held = shaper.shape(
        action_id=1,
        finger_id=4,
        action_conf=0.92,
        timestamp_stream_ms=1050,
        stability_ok=False,
        timebase_ms=1050,
    )

    assert steady.action_id == 1
    assert steady.finger_id == 2
    assert held.action_id == 1
    assert held.finger_id == 2


def test_actuation_command_shaper_recovers_after_initial_unstable_noops():
    mod = _load_live_module()
    shaper = mod._build_actuation_command_shaper(
        type(
            "Args",
            (),
            {
                "actuation_min_prob": 0.75,
                "actuation_speed_gamma": 1.0,
                "actuation_cooldown_ms": 250,
                "hop_sec": 0.05,
                "actuation_stability": 3,
            },
        )()
    )

    first = shaper.shape(
        action_id=1,
        finger_id=3,
        action_conf=0.95,
        timestamp_stream_ms=1000,
        stability_ok=False,
        timebase_ms=1000,
    )
    second = shaper.shape(
        action_id=1,
        finger_id=3,
        action_conf=0.96,
        timestamp_stream_ms=1200,
        stability_ok=True,
        timebase_ms=1200,
    )

    assert first.action_id == 0
    assert first.finger_id == 0
    assert second.action_id == 1
    assert second.finger_id == 3


def test_actuation_command_shaper_immediate_profile_does_not_hold_changed_command():
    mod = _load_live_module()
    shaper = mod._build_actuation_command_shaper(
        type(
            "Args",
            (),
            {
                "actuation_min_prob": 0.0,
                "actuation_speed_gamma": 1.0,
                "actuation_cooldown_ms": 0,
                "hop_sec": 0.05,
                "actuation_stability": 1,
            },
        )()
    )

    first = shaper.shape(
        action_id=1,
        finger_id=2,
        action_conf=0.20,
        timestamp_stream_ms=1000,
        stability_ok=True,
        timebase_ms=1000,
    )
    second = shaper.shape(
        action_id=2,
        finger_id=4,
        action_conf=0.15,
        timestamp_stream_ms=1050,
        stability_ok=True,
        timebase_ms=1050,
    )

    assert first.action_id == 1
    assert first.finger_id == 2
    assert second.action_id == 2
    assert second.finger_id == 4
