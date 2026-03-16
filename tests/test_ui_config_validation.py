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
        },
    )
    assert result.ok is True


def test_train_rejects_bad_action_weights_and_rest_finger_loss_weight():
    result = validate_step_settings(
        "train",
        {
            "action_weights": "1.0,0.5",
            "rest_finger_loss_weight": -0.1,
        },
    )
    assert any("action_weights" in err for err in result.errors)
    assert any("rest_finger_loss_weight" in err for err in result.errors)
