from __future__ import annotations

import pytest

from muse_streaming.config import DEFAULT_LABELS
from muse_streaming.muse_lsl_streamer import LSL_AVAILABLE, MuseLslStreamer


def test_streamer_defaults_empty_labels() -> None:
    streamer = MuseLslStreamer(labels=[])
    streamer._normalize_labels()
    assert streamer.config.labels == DEFAULT_LABELS


def test_streamer_refuses_empty_outlet_labels() -> None:
    if not LSL_AVAILABLE:
        pytest.skip("pylsl not available")
    streamer = MuseLslStreamer(labels=[])
    streamer.config.labels = []
    with pytest.raises(RuntimeError, match="No EEG labels available"):
        streamer._build_outlet(streamer.config)


def test_streamer_build_outlet_publishes_canonical_channel_metadata(monkeypatch) -> None:
    class _FakeChannelNode:
        def __init__(self) -> None:
            self.values = {}

        def append_child_value(self, key, value):
            self.values[str(key)] = value
            return self

    class _FakeChannelsNode:
        def __init__(self) -> None:
            self.children = []

        def append_child(self, name):
            assert name == "channel"
            node = _FakeChannelNode()
            self.children.append(node)
            return node

    class _FakeDescNode:
        def __init__(self) -> None:
            self.channels = None

        def append_child(self, name):
            assert name == "channels"
            self.channels = _FakeChannelsNode()
            return self.channels

    class _FakeStreamInfo:
        def __init__(self, name, stype, channel_count, rate, fmt, source_id):
            self.desc_node = _FakeDescNode()
            self.name = name
            self.stype = stype
            self.channel_count = channel_count
            self.rate = rate
            self.fmt = fmt
            self.source_id = source_id

        def desc(self):
            return self.desc_node

    class _FakeOutlet:
        def __init__(self, info, chunk_size):
            self.info = info
            self.chunk_size = chunk_size

    monkeypatch.setattr("muse_streaming.muse_lsl_streamer.StreamInfo", _FakeStreamInfo)
    monkeypatch.setattr("muse_streaming.muse_lsl_streamer.StreamOutlet", _FakeOutlet)

    streamer = MuseLslStreamer(labels=["'tp9'", " AF7 ", '"af8"', "‘TP10’"])
    streamer._normalize_labels()
    outlet = streamer._build_outlet(streamer.config)

    published = [node.values["label"] for node in outlet.info.desc_node.channels.children]
    assert published == ["TP9", "AF7", "AF8", "TP10"]
    assert all("'" not in label and '"' not in label for label in published)
