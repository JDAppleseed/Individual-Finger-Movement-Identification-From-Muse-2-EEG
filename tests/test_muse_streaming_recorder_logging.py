from __future__ import annotations

import logging

from muse_streaming.recorder import wrap_logger_with_session


def test_wrap_logger_with_session_logger():
    base_logger = logging.getLogger("muse_streaming.test")
    wrapped = wrap_logger_with_session(base_logger, "session123")
    assert wrapped is not None
    assert wrapped.extra.get("session_id") == "session123"


def test_wrap_logger_with_session_adapter():
    base_logger = logging.getLogger("muse_streaming.test.adapter")
    adapter = logging.LoggerAdapter(base_logger, {"session_id": "old"})
    wrapped = wrap_logger_with_session(adapter, "session456")
    assert wrapped is not None
    assert wrapped.extra.get("session_id") == "session456"
