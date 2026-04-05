from __future__ import annotations

from collections import deque

import numpy as np

from muse_streaming.muse_lsl_streamer import MuseLslStreamer


class DummyOutlet:
    def __init__(self) -> None:
        self.chunks = []

    def push_chunk(self, samples, timestamps) -> None:
        self.chunks.append((samples, timestamps))


def _make_samples(offset: float) -> np.ndarray:
    return (np.arange(12, dtype=np.float32) + np.float32(offset)).astype(np.float32)


def test_stale_partial_packet_is_dropped_without_nan_push() -> None:
    streamer = MuseLslStreamer()
    outlet = DummyOutlet()
    streamer._outlet = outlet

    packet_index = 7
    streamer._packet_buffer[packet_index] = {
        "TP9": _make_samples(0.0),
        "AF7": _make_samples(100.0),
        "AF8": _make_samples(200.0),
    }
    streamer._packet_first_seen[packet_index] = 0.0
    streamer._packet_arrival_order = deque([packet_index])

    streamer._flush_stale_packets(now_time=streamer._packet_deadline_s + 1.0)

    assert outlet.chunks == []
    assert streamer._packets_dropped_partial == 1
    assert streamer._sample_index == 12
    assert packet_index not in streamer._packet_buffer
    assert packet_index not in streamer._packet_first_seen


def test_complete_packet_still_pushes_chunk() -> None:
    streamer = MuseLslStreamer()
    outlet = DummyOutlet()
    streamer._outlet = outlet

    packet_index = 8
    slot = {
        "TP9": _make_samples(0.0),
        "AF7": _make_samples(100.0),
        "AF8": _make_samples(200.0),
        "TP10": _make_samples(300.0),
    }
    streamer._packet_buffer[packet_index] = dict(slot)
    streamer._packet_first_seen[packet_index] = 0.0
    streamer._packet_arrival_order = deque([packet_index])

    streamer._flush_packet(packet_index, slot, partial=False)

    assert len(outlet.chunks) == 1
    pushed_samples, pushed_timestamps = outlet.chunks[0]
    assert len(pushed_samples) == 12
    assert len(pushed_timestamps) == 12
    assert streamer._packets_dropped_partial == 0
    assert streamer._sample_index == 12
    assert packet_index not in streamer._packet_buffer
