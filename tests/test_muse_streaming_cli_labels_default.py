from __future__ import annotations

import logging

from muse_streaming.cli import _build_stream_settings
from muse_streaming.config import DEFAULT_LABELS


def test_cli_labels_empty_uses_defaults() -> None:
    logger = logging.getLogger("test_cli_labels")
    stream = _build_stream_settings(
        name="Muse2-EEG",
        stype="EEG",
        nominal_srate=256.0,
        labels_arg="",
        logger=logger,
    )
    assert stream.labels == DEFAULT_LABELS
