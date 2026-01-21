from __future__ import annotations

import logging
import sys

import pytest

import muse_streaming.cli as cli
import muse_streaming.healthcheck as healthcheck


class DummyHealthResult:
    def __init__(self, ok: bool = True) -> None:
        self.ok = ok

    def to_dict(self) -> dict:
        return {"ok": self.ok}


def test_run_healthcheck_accepts_legacy_name(monkeypatch):
    monkeypatch.setattr(healthcheck, "LSL_AVAILABLE", True)
    monkeypatch.setattr(healthcheck, "resolve_streams", lambda: [])
    with pytest.warns(DeprecationWarning):
        result = healthcheck.run_healthcheck(
            name="Muse2-EEG",
            stype="EEG",
            required_labels=["TP9", "AF7", "AF8", "TP10"],
        )
    assert result.reason == "stream_not_found"


@pytest.mark.parametrize("flag", ["--stream-name", "--name"])
def test_cli_healthcheck_accepts_stream_name_alias(monkeypatch, flag):
    called = {}

    def fake_run_healthcheck(*, stream, require_exact_channels, check_timebase):
        called["stream"] = stream
        called["require_exact_channels"] = require_exact_channels
        called["check_timebase"] = check_timebase
        return DummyHealthResult(ok=True)

    monkeypatch.setattr(cli, "run_healthcheck", fake_run_healthcheck)
    monkeypatch.setattr(cli, "configure_logging", lambda settings: logging.getLogger("test"))
    monkeypatch.setattr(cli, "LSL_AVAILABLE", True)
    monkeypatch.setattr(cli, "resolve_streams", lambda *args, **kwargs: [])
    monkeypatch.setattr(sys, "argv", ["cli", "healthcheck", flag, "Muse2-EEG"])
    exit_code = cli.main()
    assert exit_code == 0
    assert called["stream"].name == "Muse2-EEG"


def test_cli_list_streams_output_stable(monkeypatch, capsys):
    class DummyInfo:
        def name(self):
            return '"Muse2-EEG"'

        def type(self):
            return "'EEG'"

        def channel_count(self):
            return 4

        def nominal_srate(self):
            return 256

    monkeypatch.setattr(cli, "LSL_AVAILABLE", True)
    monkeypatch.setattr(cli, "resolve_streams", lambda: [DummyInfo()])
    monkeypatch.setattr(cli, "configure_logging", lambda settings: logging.getLogger("test"))
    monkeypatch.setattr(sys, "argv", ["cli", "list-streams"])
    exit_code = cli.main()
    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    assert out == "Muse2-EEG | EEG | ch=4 | rate=256"
