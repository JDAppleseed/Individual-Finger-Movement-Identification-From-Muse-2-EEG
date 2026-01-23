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
