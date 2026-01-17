import pytest

from utils.lsl_stream_select import (
    MultipleStreamsMatchedError,
    NoStreamFoundError,
    NoStreamMatchedError,
    StreamSelector,
    select_stream_candidate,
)


def test_no_streams_found_raises():
    selector = StreamSelector(name_contains=None, type_equals="EEG", min_channels=4)
    with pytest.raises(NoStreamFoundError, match="No LSL streams found"):
        select_stream_candidate([], selector)


def test_no_streams_matched_raises():
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
    with pytest.raises(NoStreamMatchedError, match="Set LSL_STREAM_NAME"):
        select_stream_candidate(candidates, selector)


def test_multiple_streams_matched_raises():
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
    with pytest.raises(MultipleStreamsMatchedError, match="Set LSL_STREAM_NAME"):
        select_stream_candidate(candidates, selector)
