from __future__ import annotations

from muse_streaming.cli import format_stream_info


class DummyInfo:
    def name(self):
        return "Muse2-EEG"

    def type(self):
        return "EEG"

    def channel_count(self):
        return 4

    def nominal_srate(self):
        return 256


def test_format_stream_info_no_quotes():
    info = DummyInfo()
    output = format_stream_info(info)
    assert output == "Muse2-EEG | EEG | ch=4 | rate=256"
