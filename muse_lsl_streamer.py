from __future__ import annotations

"""Compatibility shim for the Muse 2 BLE -> LSL streamer CLI + imports."""

from muse_streaming.muse_lsl_streamer import (
    BLEAK_AVAILABLE,
    LSL_AVAILABLE,
    BleakClient,
    BleakScanner,
    MuseLslStreamer,
    install_signal_handlers,
    main,
)

__all__ = [
    "BLEAK_AVAILABLE",
    "LSL_AVAILABLE",
    "BleakClient",
    "BleakScanner",
    "MuseLslStreamer",
    "install_signal_handlers",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
