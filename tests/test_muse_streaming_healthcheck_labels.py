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
    def __init__(
        self,
        info,
        hydrated_info=None,
        samples=None,
        open_error=None,
        pull_error=None,
    ):
        self._candidate = info
        self._hydrated_info = hydrated_info or info
        self._sample_index = 0
        self._samples = list(samples) if samples is not None else None
        self._open_error = open_error
        self._pull_error = pull_error
        self.opened = False

    def open_stream(self, timeout=0.5):
        if self._open_error is not None:
            raise self._open_error
        self.opened = True

    def info(self, timeout=0.5):
        return self._hydrated_info

    def pull_sample(self, timeout=0.2):
        if self._pull_error is not None:
            raise self._pull_error
        if self._samples is not None:
            if not self._samples:
                return None, None
            return self._samples.pop(0)
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
    assert result.inlet_created is True
    assert result.inlet_opened is True
    assert result.samples_received >= 1
    assert result.pull_attempts >= 1


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


def test_run_healthcheck_reports_stream_selection_ambiguous_without_source_id(monkeypatch):
    first = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "source-a",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )
    second = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "source-b",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [first, second])

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.05,
    )

    assert result.ok is False
    assert result.reason == "stream_selection_ambiguous"
    assert result.matching_candidate_count == 2


def test_run_healthcheck_reports_no_samples_with_resolved_stream_identity(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(
        healthcheck,
        "StreamInlet",
        lambda info: _FakeInlet(info, samples=[]),
    )

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.01,
    )

    assert result.ok is False
    assert result.reason == "stream_resolved_but_no_samples_pulled"
    assert result.inlet_created is True
    assert result.inlet_opened is True
    assert result.pull_attempts >= 1
    assert result.source_id == "stream-1"


def test_run_healthcheck_accepts_verified_sample_flow_without_full_window(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(
        healthcheck,
        "StreamInlet",
        lambda info: _FakeInlet(info, samples=[([0.0, 0.0, 0.0, 0.0], 1 / 256.0)]),
    )

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.01,
        min_sample_window_s=1.0,
    )

    assert result.ok is True
    assert result.reason == "ok"
    assert result.samples_received == 1
    assert result.measured_sps is None


def test_run_healthcheck_reports_inlet_open_failure(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(
        healthcheck,
        "StreamInlet",
        lambda info: _FakeInlet(info, open_error=RuntimeError("open failed")),
    )

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.05,
    )

    assert result.ok is False
    assert result.reason == "stream_resolved_but_inlet_open_failed"
    assert result.inlet_created is True
    assert result.inlet_opened is False


def test_run_healthcheck_reports_sample_shape_invalid(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(
        healthcheck,
        "StreamInlet",
        lambda info: _FakeInlet(info, samples=[([0.0, 0.0, 0.0], 1 / 256.0)]),
    )

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.05,
    )

    assert result.ok is False
    assert result.reason == "healthcheck_sample_shape_invalid"
    assert result.first_sample_length == 3
    assert result.sample_validation_error == "expected 4 channels but received 3"


def test_run_healthcheck_reports_pull_contract_violation(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "stream-1",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(
        healthcheck,
        "StreamInlet",
        lambda info: _FakeInlet(info, pull_error=RuntimeError("pull failed")),
    )

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        timeout_s=0.05,
    )

    assert result.ok is False
    assert result.reason == "healthcheck_stream_pull_contract_violation"
    assert "pull failed" in (result.sample_validation_error or "")


def test_run_healthcheck_reports_resolved_source_id_mismatch_after_inlet_hydration(monkeypatch):
    candidate = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "wanted-source",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )
    hydrated = _FakeStreamInfo(
        "Muse2-EEG",
        "EEG",
        4,
        256.0,
        "other-source",
        _info_xml(["TP9", "AF7", "AF8", "TP10"]),
    )

    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda timeout=0.1: [candidate])
    monkeypatch.setattr(
        healthcheck,
        "StreamInlet",
        lambda info: _FakeInlet(info, hydrated_info=hydrated),
    )

    result = healthcheck.run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        source_id="wanted-source",
        timeout_s=0.05,
    )

    assert result.ok is False
    assert result.reason == "source_id_resolved_mismatch"
    assert result.source_id == "other-source"
