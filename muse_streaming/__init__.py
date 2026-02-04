"""Internal Muse 2 streaming utilities.

Keep this module import-light: pipeline steps import submodules like
`muse_streaming.session_writer` and should not require BLE streamer deps unless
explicitly used.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MuseLslStreamer", "run_healthcheck"]


def __getattr__(name: str) -> Any:
    if name == "MuseLslStreamer":
        from muse_streaming.muse_lsl_streamer import MuseLslStreamer

        return MuseLslStreamer
    if name == "run_healthcheck":
        from muse_streaming.healthcheck import run_healthcheck

        return run_healthcheck
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
