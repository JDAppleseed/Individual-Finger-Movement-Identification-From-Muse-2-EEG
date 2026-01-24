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
