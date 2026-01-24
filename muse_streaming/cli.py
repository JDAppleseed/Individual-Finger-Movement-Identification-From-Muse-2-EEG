from __future__ import annotations

import argparse
import asyncio
import sys
import threading
import time
from pathlib import Path
from typing import Optional

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


def _clean_stream_value(value: str) -> str:
    s = str(value).strip()
    while len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s


def format_stream_info(info) -> str:
    name = _clean_stream_value(info.name())
    stype = _clean_stream_value(info.type())
    return (
        f"{name} | {stype} | ch={info.channel_count()} | "
        f"rate={info.nominal_srate()}"
    )


def _resolve_healthcheck_stream_name(
    stream_name: Optional[str],
    stream_type: Optional[str],
) -> Optional[str]:
    if stream_name:
        return stream_name
    if not LSL_AVAILABLE or resolve_streams is None:
        raise RuntimeError("pylsl not available")
    candidates = [
        info
        for info in resolve_streams()
        if (not stream_type or info.type() == stream_type)
    ]
    if len(candidates) > 1:
        names = sorted({_clean_stream_value(info.name()) for info in candidates})
        hint = f" Candidates: {', '.join(names)}" if names else ""
        raise RuntimeError(
            "Multiple LSL streams found. Use --stream-name to disambiguate." + hint
        )
    if len(candidates) == 1:
        return candidates[0].name()
    return None


def _parse_common_stream_args(
    parser: argparse.ArgumentParser, *, default_name: Optional[str] = DEFAULT_STREAM_NAME
) -> None:
    parser.add_argument(
        "--stream-name",
        "--name",
        dest="stream_name",
        type=str,
        default=default_name,
        help="LSL stream name (deprecated alias: --name)",
    )
    parser.add_argument("--type", type=str, default=DEFAULT_STREAM_TYPE)
    parser.add_argument("--rate", type=float, default=DEFAULT_NOMINAL_SRATE)
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help=f"Comma-separated channel labels (default: {','.join(DEFAULT_LABELS)})",
    )
    parser.add_argument("--log-level", type=str, default="INFO")


def _resolve_labels_arg(
    labels_arg: Optional[object], logger
) -> Optional[list[str]]:
    if labels_arg is None:
        logger.info(
            "Labels not provided (or empty); using DEFAULT_LABELS: %s",
            DEFAULT_LABELS,
        )
        return None
    if isinstance(labels_arg, (list, tuple)):
        labels = [str(label).strip() for label in labels_arg if str(label).strip()]
    else:
        text = str(labels_arg).strip()
        labels = parse_labels(text) if text else []
    if labels:
        return labels
    logger.info(
        "Labels not provided (or empty); using DEFAULT_LABELS: %s",
        DEFAULT_LABELS,
    )
    return None


def _build_stream_settings(
    *,
    name: str,
    stype: str,
    nominal_srate: float,
    labels_arg: Optional[object],
    logger,
) -> StreamSettings:
    labels = _resolve_labels_arg(labels_arg, logger)
    kwargs = {
        "name": name,
        "stype": stype,
        "nominal_srate": nominal_srate,
    }
    if labels is not None:
        kwargs["labels"] = labels
    return StreamSettings(**kwargs)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
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
    _parse_common_stream_args(health_parser, default_name=None)
    health_parser.add_argument("--exact", action="store_true")
    health_parser.add_argument("--check-timebase", action="store_true")
    health_parser.add_argument("--sim", action="store_true", help="Use simulator")

    list_parser = sub.add_parser("list-streams", help="List available LSL streams")
    list_parser.add_argument("--log-level", type=str, default="INFO")

    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    logger = configure_logging(LoggingSettings(log_level=args.log_level))

    if args.command == "list-streams":
        if not LSL_AVAILABLE or resolve_streams is None:
            print("❌ pylsl not available", file=sys.stderr)
            return 2
        streams = resolve_streams()
        for info in streams:
            print(format_stream_info(info))
        return 0

    sim_streamer = None

    try:
        if args.command == "start-streamer":
            stream = _build_stream_settings(
                name=args.stream_name,
                stype=args.type,
                nominal_srate=args.rate,
                labels_arg=args.labels,
                logger=logger,
            )
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
            stream_name = args.stream_name
            if stream_name is None:
                if args.sim:
                    stream_name = DEFAULT_STREAM_NAME
                else:
                    try:
                        stream_name = _resolve_healthcheck_stream_name(stream_name, args.type)
                    except RuntimeError as exc:
                        print(f"❌ {exc}", file=sys.stderr)
                        return 2
            stream = _build_stream_settings(
                name=stream_name,
                stype=args.type,
                nominal_srate=args.rate,
                labels_arg=args.labels,
                logger=logger,
            )
            if args.sim:
                sim_streamer = _start_sim_streamer(stream, logger)
                # Wait for the simulator stream to become available
                if not LSL_AVAILABLE or resolve_streams is None:
                    raise RuntimeError("pylsl not available")
                stream_resolved = False
                start_wait = time.monotonic()
                while time.monotonic() - start_wait < 5.0:  # 5s timeout
                    if any(s.name() == stream.name for s in resolve_streams(timeout=0.1)):
                        stream_resolved = True
                        break
                if not stream_resolved:
                    raise RuntimeError(
                        f"Simulator stream '{stream.name}' failed to start in time."
                    )
            result = run_healthcheck(
                stream=stream,
                require_exact_channels=args.exact,
                check_timebase=args.check_timebase,
            )
            print(result.to_dict())
            return 0 if result.ok else 2

        if args.command == "record":
            stream = _build_stream_settings(
                name=args.stream_name,
                stype=args.type,
                nominal_srate=args.rate,
                labels_arg=args.labels,
                logger=logger,
            )
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
