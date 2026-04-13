from __future__ import annotations

from dataclasses import dataclass

import muse_streaming.healthcheck as healthcheck
from utils.channel_labels import parse_channel_label_list


def _info_xml(labels: list[str]) -> str:
    channels = "".join(
        f"<channel><label>{label}</label><unit>microvolts</unit><type>EEG</type></channel>"
        for label in labels
    )
    return f"<info><desc><channels>{channels}</channels></desc></info>"


@dataclass
class _FakeStreamInfo:
    _name: str
    _type: str
    _channel_count: int
    _rate: float
    _source_id: str
    _xml: str

    def name(self):
        return self._name

    def type(self):
        return self._type

    def channel_count(self):
        return self._channel_count

    def nominal_srate(self):
        return self._rate

    def source_id(self):
        return self._source_id

    def uid(self):
        return f"uid-{self._source_id}"

    def as_xml(self):
        return self._xml


class _FakeInlet:
    def __init__(self, info, hydrated_info=None):
        self._candidate = info
        self._hydrated_info = hydrated_info or info
        self._sample_index = 0

    def info(self, timeout=0.5):
        return self._hydrated_info

    def pull_sample(self, timeout=0.2):
        self._sample_index += 1
        return [0.0, 0.0, 0.0, 0.0], self._sample_index / 256.0


def test_parse_channel_label_list_strips_quotes_consistently():
    assert parse_channel_label_list(["'TP9'", " ‘AF7’ ", '"af8"', "`tp10`"]) == [
        "TP9",
        "AF7",
        "AF8",
        "TP10",
    ]


def test_run_healthcheck_passes_with_hydrated_metadata_labels(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml([]),
    )
    hydrated = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml(["'TP9'", "'AF7'", "'AF8'", "'TP10'"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(
        healthcheck,
        "StreamInlet",
        lambda info: _FakeInlet(info, hydrated_info=hydrated),
    )
    monkeypatch.setattr(healthcheck, "local_clock", lambda: 10.0)

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.2,
        min_sample_window_s=0.01,
    )

    assert result.ok is True
    assert result.reason == "ok"
    assert result.raw_metadata_labels == ["'TP9'", "'AF7'", "'AF8'", "'TP10'"]
    assert result.normalized_metadata_labels == ["TP9", "AF7", "AF8", "TP10"]
    assert result.labels == ["TP9", "AF7", "AF8", "TP10"]


def test_run_healthcheck_reports_labels_missing_when_stream_metadata_absent(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml([]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(healthcheck, "StreamInlet", lambda info: _FakeInlet(info))

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.2,
    )

    assert result.ok is False
    assert result.reason == "stream_found_labels_missing"
    assert result.raw_metadata_labels == []
    assert result.normalized_metadata_labels == []


def test_run_healthcheck_reports_label_mismatch_with_normalized_found_labels(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml(["'TP9'", "'AF7'", "'AF8'", "'AUX'"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(healthcheck, "StreamInlet", lambda info: _FakeInlet(info))

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.2,
    )

    assert result.ok is False
    assert result.reason == "stream_found_label_mismatch"
    assert result.raw_metadata_labels == ["'TP9'", "'AF7'", "'AF8'", "'AUX'"]
    assert result.normalized_metadata_labels == ["TP9", "AF7", "AF8", "AUX"]


def test_run_healthcheck_reports_channel_count_mismatch_distinctly(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        5,
        256.0,
        "stream-1",
        _info_xml(["TP9", "AF7", "AF8", "TP10", "AUX"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(healthcheck, "StreamInlet", lambda info: _FakeInlet(info))

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        require_exact_channels=True,
        timeout_s=0.2,
    )

    assert result.ok is False
    assert result.reason == "stream_found_channel_count_mismatch"


def test_run_healthcheck_reports_source_id_mismatch_distinctly(monkeypatch):
    wrong = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "wrong-source",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [wrong])
    monkeypatch.setattr(healthcheck, "StreamInlet", lambda info: _FakeInlet(info))

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        source_id="wanted-source",
        timeout_s=0.2,
    )

    assert result.ok is False
    assert result.reason == "stream_found_source_id_mismatch"
