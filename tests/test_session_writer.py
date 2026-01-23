from dataclasses import dataclass
from pathlib import Path

import numpy as np

from muse_streaming.session_writer import SessionWriter


@dataclass(frozen=True)
class DummyPacket:
    seq: int
    lsl_ts_raw: float
    lsl_ts_mono: float
    local_ts: float
    sample: np.ndarray
    flags: int
    segment_id: int
    raw_path: Path
    clamped: bool


def _make_packets(count: int, start_seq: int = 0) -> list[DummyPacket]:
    packets = []
    for i in range(count):
        seq = start_seq + i
        packets.append(
            DummyPacket(
                seq=seq,
                lsl_ts_raw=1.0 + 0.01 * seq,
                lsl_ts_mono=1.0 + 0.01 * seq,
                local_ts=1.0 + 0.01 * seq,
                sample=np.ones(4, dtype=float) * seq,
                flags=0,
                segment_id=0,
                raw_path=Path("."),
                clamped=False,
            )
        )
    return packets


def test_session_writer_lossless(tmp_path):
    writer = SessionWriter(
        output_root=tmp_path,
        subject_id="test",
        session_id="sess",
        channel_labels=["ch1", "ch2", "ch3", "ch4"],
        sampling_rate=256.0,
        timebase_version="absolute_v1",
        shard_size_samples=5,
    )
    packets = _make_packets(10)
    writer.append_packets(packets[:5])
    writer.append_packets(packets[5:])
    writer.finalize("normal")

    manifest = (tmp_path / "test_sess" / "manifest.json").read_text()
    payload = __import__("json").loads(manifest)
    assert payload["missing_seq_count"] == 0
    assert payload["expected_sample_count"] == 10
    assert payload["actual_sample_count"] == 10
    assert payload["seq_min"] == 0
    assert payload["seq_max"] == 9


def test_session_writer_gap_counts_missing(tmp_path):
    writer = SessionWriter(
        output_root=tmp_path,
        subject_id="test",
        session_id="sess2",
        channel_labels=["ch1", "ch2", "ch3", "ch4"],
        sampling_rate=256.0,
        timebase_version="absolute_v1",
        shard_size_samples=10,
    )
    packets = _make_packets(4)
    packets += _make_packets(4, start_seq=5)
    writer.append_packets(packets)
    writer.finalize("normal")
    manifest = (tmp_path / "test_sess2" / "manifest.json").read_text()
    payload = __import__("json").loads(manifest)
    assert payload["missing_seq_count"] == 1
