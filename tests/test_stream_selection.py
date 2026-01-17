import pytest

from utils.lsl_stream_select import (
    MultipleStreamsMatchedError,
    NoStreamMatchedError,
    StreamSelector,
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
