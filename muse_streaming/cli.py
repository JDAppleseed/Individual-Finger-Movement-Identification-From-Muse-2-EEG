from __future__ import annotations

import argparse
import asyncio
import sys

from muse_streaming.muse_lsl_streamer import (
    DEFAULT_LABELS,
    MuseLslStreamer,
    install_signal_handlers,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Muse 2 BLE -> LSL streamer")
    parser.add_argument("--name", type=str, default="Muse2-EEG", help="LSL stream name")
    parser.add_argument("--type", type=str, default="EEG", help="LSL stream type")
    parser.add_argument("--rate", type=float, default=256.0, help="Nominal sampling rate")
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated channel labels",
    )
    parser.add_argument("--device-name", type=str, default=None, help="BLE device name")
    parser.add_argument("--mac-address", type=str, default=None, help="BLE MAC address")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    streamer = MuseLslStreamer(
        name=args.name,
        stype=args.type,
        rate=args.rate,
        labels=labels,
        device_name=args.device_name,
        mac_address=args.mac_address,
    )
    install_signal_handlers(streamer)

    try:
        asyncio.run(streamer.run())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"❌ Muse 2 streamer failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
