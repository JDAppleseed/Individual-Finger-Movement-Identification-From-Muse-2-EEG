from app.config_model import default_infer_settings, default_pseudo_live_settings
from app.ui_config_validation import validate_step_settings


def test_train_record_disallows_drop():
    settings = {"MODE": "train_record", "ALLOW_DROP": True, "SAVE_RAW": True}
    result = validate_step_settings("step1", settings)
    assert result.ok is False


def test_live_infer_warns_on_drop():
    settings = {"MODE": "live_infer", "ALLOW_DROP": True}
    result = validate_step_settings("infer", settings)
    assert result.ok is True
    assert result.warnings


def test_live_infer_allows_actuation():
    result = validate_step_settings("infer", {"enable_actuation": True})
    assert result.ok is True
    assert not result.errors


def test_live_infer_best_defaults_validate():
    result = validate_step_settings("infer", default_infer_settings())
    assert result.ok is True
    assert not result.errors


def test_pseudo_live_validation_covers_latency_mode():
    settings = default_pseudo_live_settings()
    settings["session_dir"] = "sessions/example"
    settings["target_session_dirs"] = ["sessions/example"]
    result = validate_step_settings("evaluate_pseudo_live", settings)
    assert result.ok is True
    assert not result.errors

    settings["latency_mode"] = "fixed"
    settings["fixed_latency_ms"] = None
    result = validate_step_settings("evaluate_pseudo_live", settings)
    assert any("fixed_latency_ms is required" in err for err in result.errors)


def test_train_record_disallows_actuation():
    result = validate_step_settings("step1", {"enable_actuation": True})
    assert any("only be true for the live_infer step" in err for err in result.errors)


def test_legacy_actuation_key_errors():
    result = validate_step_settings("infer", {"ENABLE_ACTUATION": True})
    assert any("Legacy key" in err for err in result.errors)


def test_actuation_type_validation():
    result = validate_step_settings("infer", {"enable_actuation": "yes"})
    assert any("must be a boolean" in err for err in result.errors)


def test_inference_engine_validation():
    result = validate_step_settings(
        "infer",
        {
            "use_inference_engine": "yes",
            "mc_passes": 0,
            "uncertainty_base_threshold": "high",
            "modulate_actuation_speed": "yes",
            "actuation_speed_gamma": "fast",
        },
    )
    assert any("use_inference_engine must be a boolean." in err for err in result.errors)
    assert any("mc_passes must be >= 1." in err for err in result.errors)
    assert any(
        "modulate_actuation_speed must be a boolean." in err
        for err in result.errors
    )
    assert any(
        "uncertainty_base_threshold must be numeric." in err
        for err in result.errors
    )
    assert any("actuation_speed_gamma must be numeric." in err for err in result.errors)


def test_live_infer_rejects_bad_actuation_and_postprocess_values():
    result = validate_step_settings(
        "infer",
        {
            "serial_baud": "fast",
            "actuation_min_prob": 1.2,
            "actuation_stability": 0,
            "actuation_cooldown_ms": -1,
            "actuation_repeat_ms": -1,
            "actuation_min_speed": 1.1,
            "threshold_action": "high",
            "threshold_finger": -0.1,
            "threshold_applicability": 1.1,
            "smoothing_method": "median",
            "finger_mode": "vote",
        },
    )
    assert any("serial_baud must be an integer." in err for err in result.errors)
    assert any("actuation_min_prob must be in [0.0, 1.0]." in err for err in result.errors)
    assert any("actuation_stability must be >= 1." in err for err in result.errors)
    assert any("actuation_cooldown_ms must be >= 0." in err for err in result.errors)
    assert any("actuation_repeat_ms must be >= 0." in err for err in result.errors)
    assert any("actuation_min_speed must be in [0.0, 1.0]." in err for err in result.errors)
    assert any("threshold_action must be numeric." in err for err in result.errors)
    assert any("threshold_finger must be in [0.0, 1.0]." in err for err in result.errors)
    assert any("threshold_applicability must be in [0.0, 1.0]." in err for err in result.errors)
    assert any("smoothing_method must be 'vote' or 'ema'." in err for err in result.errors)
    assert any("finger_mode must be 'raw' or 'smooth'." in err for err in result.errors)


def test_live_infer_warns_when_actuation_serial_port_is_implicit():
    settings = default_infer_settings()
    settings["enable_actuation"] = True
    settings["serial_port"] = None
    result = validate_step_settings("infer", settings)
    assert result.ok is True
    assert any("serial_port is blank" in warning for warning in result.warnings)


def test_live_infer_rejects_bad_live_quality_settings():
    result = validate_step_settings(
        "infer",
        {
            "live_quality_enabled": "yes",
            "input_clip_abs_z": -1.0,
            "bad_channel_rms_z": -2.0,
            "bad_channel_abs_p95_z": "high",
            "bad_channel_clipped_frac": 1.2,
            "bad_window_clipped_frac": -0.1,
            "bad_window_max_masked_channels": -1,
            "rest_bias_correction_enabled": "yes",
            "rest_bias_strength": -0.1,
            "rest_bias_min_windows": 0,
            "latency_policy": "slow",
        },
    )
    assert any("live_quality_enabled must be a boolean." in err for err in result.errors)
    assert any("input_clip_abs_z must be >= 0.0." in err for err in result.errors)
    assert any("bad_channel_rms_z must be >= 0.0." in err for err in result.errors)
    assert any("bad_channel_abs_p95_z must be numeric." in err for err in result.errors)
    assert any("bad_channel_clipped_frac must be in [0.0, 1.0]." in err for err in result.errors)
    assert any("bad_window_clipped_frac must be in [0.0, 1.0]." in err for err in result.errors)
    assert any("bad_window_max_masked_channels must be >= 0." in err for err in result.errors)
    assert any("rest_bias_correction_enabled must be a boolean." in err for err in result.errors)
    assert any("rest_bias_strength must be >= 0.0." in err for err in result.errors)
    assert any("rest_bias_min_windows must be >= 1." in err for err in result.errors)
    assert any("latency_policy must be 'warn', 'drop', or 'degrade'." in err for err in result.errors)


def test_live_infer_maps_legacy_latency_policy_enforce_to_drop():
    settings = {"latency_policy": "enforce"}
    result = validate_step_settings("infer", settings)

    assert result.ok is True
    assert settings["latency_policy"] == "drop"
    assert any("legacy" in warning and "drop" in warning for warning in result.warnings)


def test_live_infer_rejects_parity_capture_when_no_file_io_is_enabled():
    result = validate_step_settings(
        "infer",
        {
            "no_file_io": True,
            "parity_capture_enabled": True,
        },
    )

    assert any(
        "parity_capture_enabled cannot be true when no_file_io is enabled."
        in err
        for err in result.errors
    )


def test_live_infer_warns_when_stream_contract_is_relaxed():
    settings = default_infer_settings()
    settings["REQUIRE_EXACTLY_4_CHANNELS"] = False
    settings["REQUIRED_LSL_LABELS"] = ""
    settings["lsl_source_id"] = ""

    result = validate_step_settings("infer", settings)

    assert result.ok is True
    assert any("non-4-channel streams" in warning for warning in result.warnings)
    assert any("REQUIRED_LSL_LABELS is blank" in warning for warning in result.warnings)
    assert any("lsl_source_id is blank" in warning for warning in result.warnings)


def test_train_calibration_size_validation():
    result = validate_step_settings("train", {"calibration_size": 1.0})
    assert any("calibration_size must be in [0.0, 1.0)." in err for err in result.errors)


def test_train_new_mode_validation():
    result = validate_step_settings(
        "train",
        {
            "rest_balance_mode": "bad",
            "window_preprocess": "weird",
        },
    )
    assert any("rest_balance_mode" in err for err in result.errors)
    assert any("window_preprocess" in err for err in result.errors)


def test_train_accepts_new_rest_balance_mode_and_action_weights():
    result = validate_step_settings(
        "train",
        {
            "rest_balance_mode": "core_event_equalized",
            "action_weights": "1.25,0.9,1.0",
            "rest_finger_loss_weight": 0.1,
            "finger_applicability_head": True,
            "applicability_loss_weight": 0.5,
            "threshold_applicability": 0.5,
        },
    )
    assert result.ok is True


def test_train_rejects_bad_action_weights_and_rest_finger_loss_weight():
    result = validate_step_settings(
        "train",
        {
            "action_weights": "1.0,0.5",
            "rest_finger_loss_weight": -0.1,
            "finger_applicability_head": "yes",
            "applicability_loss_weight": -0.2,
            "threshold_applicability": "high",
        },
    )
    assert any("action_weights" in err for err in result.errors)
    assert any("rest_finger_loss_weight" in err for err in result.errors)
    assert any("finger_applicability_head must be a boolean." in err for err in result.errors)
    assert any("applicability_loss_weight must be >= 0." in err for err in result.errors)
    assert any("threshold_applicability must be numeric." in err for err in result.errors)


def test_topomap_band_validation():
    result = validate_step_settings(
        "topomaps",
        {
            "band_low": 12.0,
            "band_high": 8.0,
        },
    )
    assert any("band_low must be strictly less than band_high." in err for err in result.errors)


def test_topomap_quantile_and_blur_validation():
    result = validate_step_settings(
        "topomaps",
        {
            "blur_sigma": -0.1,
            "robust_quantile": 0.5,
        },
    )
    assert any("blur_sigma must be >= 0." in err for err in result.errors)
    assert any("robust_quantile must be in [0.0, 0.5)." in err for err in result.errors)
