from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path

from muse_streaming.config import (
    DEFAULT_LABELS,
    DEFAULT_NOMINAL_SRATE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_STREAM_NAME,
    DEFAULT_STREAM_TYPE,
    DEFAULT_TARGET_FS,
    DEFAULT_WINDOW_HOP_SEC,
    DEFAULT_WINDOW_SEC,
    LoggingSettings,
    RecorderSettings,
    StreamSettings,
    configure_logging,
    parse_labels,
)
from muse_streaming.healthcheck import run_healthcheck
from muse_streaming.muse_lsl_streamer import MuseLslStreamer, install_signal_handlers
from muse_streaming.recorder import record

try:
    from pylsl import resolve_streams

    LSL_AVAILABLE = True
except Exception:  # pragma: no cover
    resolve_streams = None
    LSL_AVAILABLE = False


def _start_sim_streamer(stream: StreamSettings, logger) -> MuseLslStreamer:
    streamer = MuseLslStreamer(
        name=stream.name,
        stype=stream.stype,
        rate=stream.nominal_srate,
        labels=stream.labels,
        simulate=True,
        logger=logger,
    )

    def _run():
        asyncio.run(streamer.run())

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return streamer


def _parse_common_stream_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=str, default=DEFAULT_STREAM_NAME)
    parser.add_argument("--type", type=str, default=DEFAULT_STREAM_TYPE)
    parser.add_argument("--rate", type=float, default=DEFAULT_NOMINAL_SRATE)
    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_LABELS),
        help="Comma-separated channel labels",
    )
    parser.add_argument("--log-level", type=str, default="INFO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Muse 2 streaming CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    streamer_parser = sub.add_parser("start-streamer", help="Start BLE -> LSL streamer")
    _parse_common_stream_args(streamer_parser)
    streamer_parser.add_argument("--device-name", type=str, default=None)
    streamer_parser.add_argument("--mac-address", type=str, default=None)
    streamer_parser.add_argument("--sim", action="store_true", help="Use simulator")

    record_parser = sub.add_parser("record", help="Record LSL stream to CSV")
    _parse_common_stream_args(record_parser)
    record_parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT)
    record_parser.add_argument("--subject-id", type=str, default="unknown")
    record_parser.add_argument("--session-id", type=str, default=None)
    record_parser.add_argument("--resume", action="store_true")
    record_parser.add_argument("--target-fs", type=float, default=DEFAULT_TARGET_FS)
    record_parser.add_argument("--window-sec", type=float, default=DEFAULT_WINDOW_SEC)
    record_parser.add_argument("--window-hop-sec", type=float, default=DEFAULT_WINDOW_HOP_SEC)
    record_parser.add_argument("--duration-s", type=float, default=None)
    record_parser.add_argument("--no-events", action="store_true")
    record_parser.add_argument("--sim", action="store_true", help="Use simulator")

    health_parser = sub.add_parser("healthcheck", help="Validate LSL stream")
    _parse_common_stream_args(health_parser)
    health_parser.add_argument("--exact", action="store_true")
    health_parser.add_argument("--check-timebase", action="store_true")
    health_parser.add_argument("--sim", action="store_true", help="Use simulator")

    list_parser = sub.add_parser("list-streams", help="List available LSL streams")
    list_parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logging(LoggingSettings(log_level=args.log_level))

    if args.command == "list-streams":
        if not LSL_AVAILABLE:
            print("❌ pylsl not available", file=sys.stderr)
            return 2
        streams = resolve_streams()
        for info in streams:
            print(f\"{info.name()} | {info.type()} | ch={info.channel_count()} | rate={info.nominal_srate()}\")
        return 0

    stream = StreamSettings(
        name=args.name,
        stype=args.type,
        nominal_srate=args.rate,
        labels=parse_labels(args.labels),
    )

    sim_streamer = None

    try:
        if args.command == "start-streamer":
            streamer = MuseLslStreamer(
                name=stream.name,
                stype=stream.stype,
                rate=stream.nominal_srate,
                labels=stream.labels,
                device_name=args.device_name,
                mac_address=args.mac_address,
                simulate=args.sim,
                logger=logger,
            )
            install_signal_handlers(streamer)
            asyncio.run(streamer.run())
            return 0

        if args.command == "healthcheck":
            if args.sim:
                sim_streamer = _start_sim_streamer(stream, logger)
                time.sleep(0.2)
            result = run_healthcheck(
                stream=stream,
                require_exact_channels=args.exact,
                check_timebase=args.check_timebase,
            )
            print(result.to_dict())
            return 0 if result.ok else 2

        if args.command == "record":
            if args.sim:
                sim_streamer = _start_sim_streamer(stream, logger)
                time.sleep(0.2)
            recorder = RecorderSettings(
                output_root=args.output_dir,
                subject_id=args.subject_id,
                session_id=args.session_id,
                resume=args.resume,
                target_fs=args.target_fs,
                window_sec=args.window_sec,
                window_hop_sec=args.window_hop_sec,
                events_enabled=not args.no_events,
            )
            logger = configure_logging(
                LoggingSettings(log_level=args.log_level, session_id=args.session_id)
            )
            record(
                stream=stream,
                recorder=recorder,
                logger=logger,
                duration_s=args.duration_s,
            )
            return 0

    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"❌ CLI failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if sim_streamer is not None:
            sim_streamer.request_stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
