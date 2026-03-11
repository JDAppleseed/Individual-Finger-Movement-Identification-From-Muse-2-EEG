from app.autofill_utils import should_replace_autofilled_text


def test_replace_autofilled_text_when_blank() -> None:
    assert should_replace_autofilled_text("", None, set()) is True


def test_replace_autofilled_text_for_legacy_placeholder() -> None:
    assert should_replace_autofilled_text(
        "finger_action_model.pt",
        None,
        {"finger_action_model.pt"},
    ) is True


def test_replace_autofilled_text_for_previous_auto_value() -> None:
    assert should_replace_autofilled_text(
        "/tmp/old-session/model.pt",
        "/tmp/old-session/model.pt",
        set(),
    ) is True


def test_preserve_manual_override_text() -> None:
    assert should_replace_autofilled_text(
        "/tmp/custom/model.pt",
        "/tmp/old-session/model.pt",
        {"finger_action_model.pt"},
    ) is False
