import importlib.util
import sys
from pathlib import Path

from app.config_model import (
    default_evaluate_settings,
    default_infer_settings,
    default_step1b_settings,
    default_train_settings,
)
from utils.default_recipe import (
    ARTIFACT_DEFAULTS,
    CANONICAL_DEPLOYMENT_APPLICABILITY_THRESHOLD,
    EVAL_RECIPE_DEFAULTS,
    HISTORICAL_TRAIN_ARTIFACT_APPLICABILITY_THRESHOLD,
    LIVE_INFER_RECIPE_DEFAULTS,
    TRAIN_RECIPE_DEFAULTS,
    WINDOW_EXTRACTION_DEFAULTS,
)
from utils.inference import InferenceConfig
from utils.live_infer_common import ReplayRuntimeConfig
from utils.postprocess import PostprocessSettings


def _load_module(filename: str, module_name: str):
    module_path = Path(__file__).resolve().parents[1] / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_config_templates_match_canonical_recipe():
    step1b = default_step1b_settings()
    assert step1b["subject_id"] is None
    assert step1b["WINDOW_SEC"] == WINDOW_EXTRACTION_DEFAULTS["window_sec"]
    assert step1b["STEP_SEC"] == WINDOW_EXTRACTION_DEFAULTS["step_sec"]
    assert step1b["PAD_SEC"] == WINDOW_EXTRACTION_DEFAULTS["pad_sec"]
    assert step1b["GAP_THRESHOLD_SEC"] == WINDOW_EXTRACTION_DEFAULTS["gap_threshold_sec"]
    assert step1b["REST_POLICY"] == WINDOW_EXTRACTION_DEFAULTS["rest_policy"]

    train = default_train_settings()
    assert train["subject_id"] is None
    assert train["npz"] == ARTIFACT_DEFAULTS["windows_npz"]
    assert train["seed"] == TRAIN_RECIPE_DEFAULTS["seed"]
    assert train["rest_balance_mode"] == TRAIN_RECIPE_DEFAULTS["rest_balance_mode"]
    assert train["aux_rest_session_policy"] == TRAIN_RECIPE_DEFAULTS["aux_rest_session_policy"]
    assert train["threshold_applicability"] == TRAIN_RECIPE_DEFAULTS["threshold_applicability"]
    assert train["save_temperature"] == ARTIFACT_DEFAULTS["temperature"]

    evaluate = default_evaluate_settings()
    assert evaluate["split_seed"] == EVAL_RECIPE_DEFAULTS["split_seed"]
    assert evaluate["threshold_action"] == EVAL_RECIPE_DEFAULTS["threshold_action"]
    assert evaluate["threshold_finger"] == EVAL_RECIPE_DEFAULTS["threshold_finger"]
    assert (
        evaluate["threshold_applicability"]
        == EVAL_RECIPE_DEFAULTS["threshold_applicability"]
    )

    infer = default_infer_settings()
    assert infer["deployment_session_dir"] is None
    assert infer["window_sec"] == LIVE_INFER_RECIPE_DEFAULTS["window_sec"]
    assert infer["hop_sec"] == LIVE_INFER_RECIPE_DEFAULTS["hop_sec"]
    assert infer["threshold_action"] == LIVE_INFER_RECIPE_DEFAULTS["threshold_action"]
    assert infer["threshold_finger"] == LIVE_INFER_RECIPE_DEFAULTS["threshold_finger"]
    assert (
        infer["threshold_applicability"]
        == LIVE_INFER_RECIPE_DEFAULTS["threshold_applicability"]
    )
    assert (
        TRAIN_RECIPE_DEFAULTS["threshold_applicability"]
        == HISTORICAL_TRAIN_ARTIFACT_APPLICABILITY_THRESHOLD
    )
    assert (
        EVAL_RECIPE_DEFAULTS["threshold_applicability"]
        == CANONICAL_DEPLOYMENT_APPLICABILITY_THRESHOLD
    )
    assert LIVE_INFER_RECIPE_DEFAULTS["threshold_applicability"] == 0.0


def test_runtime_defaults_match_canonical_recipe():
    post = PostprocessSettings()
    assert post.threshold_action == LIVE_INFER_RECIPE_DEFAULTS["threshold_action"]
    assert post.threshold_finger == LIVE_INFER_RECIPE_DEFAULTS["threshold_finger"]
    assert (
        post.threshold_applicability
        == LIVE_INFER_RECIPE_DEFAULTS["threshold_applicability"]
    )

    infer_cfg = InferenceConfig()
    assert infer_cfg.base_threshold == LIVE_INFER_RECIPE_DEFAULTS["uncertainty_base_threshold"]
    assert infer_cfg.uncertainty_weight == LIVE_INFER_RECIPE_DEFAULTS["uncertainty_weight"]
    assert infer_cfg.stability_frames == LIVE_INFER_RECIPE_DEFAULTS["actuation_stability"]
    assert infer_cfg.mc_passes == LIVE_INFER_RECIPE_DEFAULTS["mc_passes"]

    replay = ReplayRuntimeConfig()
    assert replay.window_sec == LIVE_INFER_RECIPE_DEFAULTS["window_sec"]
    assert replay.hop_sec == LIVE_INFER_RECIPE_DEFAULTS["hop_sec"]
    assert replay.actuation_min_prob == LIVE_INFER_RECIPE_DEFAULTS["actuation_min_prob"]
    assert replay.actuation_stability == LIVE_INFER_RECIPE_DEFAULTS["actuation_stability"]
    assert replay.live_quality_enabled == LIVE_INFER_RECIPE_DEFAULTS["live_quality_enabled"]
    assert replay.input_clip_abs_z == LIVE_INFER_RECIPE_DEFAULTS["input_clip_abs_z"]


def test_step_entrypoints_consume_canonical_defaults():
    extract_mod = _load_module("1b_extract_windows.py", "test_extract_defaults")
    assert extract_mod.DEFAULT_SUBJECT_ID == ""

    train_mod = _load_module("2_train_model.py", "test_train_defaults")
    train_args = train_mod.build_arg_parser().parse_args([])
    assert train_args.subject_id == ""
    assert train_args.seed == TRAIN_RECIPE_DEFAULTS["seed"]
    assert train_args.test_size == TRAIN_RECIPE_DEFAULTS["test_size"]
    assert train_args.calibration_size == TRAIN_RECIPE_DEFAULTS["calibration_size"]
    assert train_args.aux_rest_session_policy == TRAIN_RECIPE_DEFAULTS["aux_rest_session_policy"]
    assert train_args.threshold_applicability == TRAIN_RECIPE_DEFAULTS["threshold_applicability"]

    eval_mod = _load_module("3_evaluate_model.py", "test_eval_defaults")
    assert (
        eval_mod._compute_prediction_metrics.__kwdefaults__["threshold_applicability"]
        == EVAL_RECIPE_DEFAULTS["threshold_applicability"]
    )

    live_mod = _load_module("7_live_infer_and_actuate.py", "test_live_defaults")
    _, live_defaults = live_mod._build_arg_parser()
    assert live_defaults["window_sec"] == LIVE_INFER_RECIPE_DEFAULTS["window_sec"]
    assert live_defaults["hop_sec"] == LIVE_INFER_RECIPE_DEFAULTS["hop_sec"]
    assert (
        live_defaults["threshold_applicability"]
        == LIVE_INFER_RECIPE_DEFAULTS["threshold_applicability"]
    )
