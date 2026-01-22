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
