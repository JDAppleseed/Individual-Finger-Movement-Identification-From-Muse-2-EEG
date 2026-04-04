import pytest

from utils.lsl_stream_select import (
    MultipleStreamsMatchedError,
    NoStreamMatchedError,
    StreamSelector,
    resolve_source_id_preference,
    select_stream_by_source_id,
    select_stream_candidate,
)


def test_select_stream_by_name_contains():
    candidates = [
        {
            "name": "Muse EEG",
            "type": "EEG",
            "channel_count": 4,
            "source_id": "a",
            "uid": "u1",
        },
        {
            "name": "Other",
            "type": "Markers",
            "channel_count": 1,
            "source_id": "b",
            "uid": "u2",
        },
    ]
    selector = StreamSelector(name_contains="muse", type_equals=None, min_channels=4)
    chosen = select_stream_candidate(candidates, selector)
    assert chosen["name"] == "Muse EEG"


def test_select_stream_rejects_ambiguous():
    candidates = [
        {
            "name": "EEG One",
            "type": "EEG",
            "channel_count": 4,
            "source_id": "a",
            "uid": "u1",
        },
        {
            "name": "EEG Two",
            "type": "EEG",
            "channel_count": 8,
            "source_id": "b",
            "uid": "u2",
        },
    ]
    selector = StreamSelector(name_contains=None, type_equals="EEG", min_channels=4)
    with pytest.raises(MultipleStreamsMatchedError, match="Multiple LSL streams matched"):
        select_stream_candidate(candidates, selector)


def test_select_stream_no_match():
    candidates = [
        {
            "name": "Markers",
            "type": "Markers",
            "channel_count": 1,
            "source_id": "a",
            "uid": "u1",
        }
    ]
    selector = StreamSelector(name_contains=None, type_equals="EEG", min_channels=4)
    with pytest.raises(NoStreamMatchedError, match="No LSL streams matched"):
        select_stream_candidate(candidates, selector)


def test_select_stream_by_source_id_exact_match():
    candidates = [
        {"name": "Muse EEG", "type": "EEG", "channel_count": 4, "source_id": "old"},
        {"name": "Muse EEG", "type": "EEG", "channel_count": 4, "source_id": "fresh"},
    ]

    selection = select_stream_by_source_id(
        candidates,
        requested_source_id="fresh",
    )

    assert selection.selected["source_id"] == "fresh"
    assert selection.recovery_used is False


def test_select_stream_by_source_id_recovers_single_live_candidate():
    candidates = [
        {"name": "Muse EEG", "type": "EEG", "channel_count": 4, "source_id": "fresh"}
    ]

    selection = select_stream_by_source_id(
        candidates,
        requested_source_id="stale",
    )

    assert selection.selected["source_id"] == "fresh"
    assert selection.requested_source_id == "stale"
    assert selection.recovery_used is True


def test_select_stream_by_source_id_refuses_ambiguous_recovery():
    candidates = [
        {"name": "Muse EEG A", "type": "EEG", "channel_count": 4, "source_id": "fresh-a"},
        {"name": "Muse EEG B", "type": "EEG", "channel_count": 4, "source_id": "fresh-b"},
    ]

    with pytest.raises(NoStreamMatchedError, match="refusing ambiguous recovery"):
        select_stream_by_source_id(
            candidates,
            requested_source_id="stale",
        )


def test_select_stream_by_source_id_succeeds_without_source_id_when_unique():
    candidates = [
        {"name": "Muse EEG", "type": "EEG", "channel_count": 4, "source_id": "fresh"}
    ]

    selection = select_stream_by_source_id(
        candidates,
        requested_source_id=None,
    )

    assert selection.selected["source_id"] == "fresh"
    assert selection.recovery_used is False


def test_resolve_source_id_preference_prefers_cli_then_env_then_config():
    cli = resolve_source_id_preference("cli-id", "env-id", "config-id")
    env = resolve_source_id_preference(None, "env-id", "config-id")
    config = resolve_source_id_preference(None, None, "config-id")
    auto = resolve_source_id_preference("auto", "env-id", "config-id")

    assert cli.requested_source_id == "cli-id"
    assert cli.source == "cli"
    assert env.requested_source_id == "env-id"
    assert env.source == "env"
    assert config.requested_source_id == "config-id"
    assert config.source == "config"
    assert auto.requested_source_id == "env-id"
    assert auto.source == "env"
