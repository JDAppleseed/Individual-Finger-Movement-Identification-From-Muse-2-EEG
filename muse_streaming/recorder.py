from __future__ import annotations

import csv
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Iterable, List, Optional

import numpy as np

from muse_streaming.config import RecorderSettings, SessionLoggerAdapter, StreamSettings
from muse_streaming.io_paths import SessionPaths, prepare_session_paths
from muse_streaming.resample import resample_window, verify_alignment
from muse_streaming.timebase import (
    TimebaseCheck,
    absolute_v1_time_s,
    check_timebase_invariants,
    clamp_monotonic,
    latency_ms,
)

try:
    from pylsl import StreamInlet, resolve_streams, local_clock

    LSL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StreamInlet = None
    resolve_streams = None
    local_clock = None
    LSL_AVAILABLE = False


@dataclass
class RecordState:
    stop_requested: bool = False
    stream_start_lsl_ts: Optional[float] = None
    last_lsl_ts: Optional[float] = None
    last_time_s: Optional[float] = None


@dataclass
class RecordArtifacts:
    paths: SessionPaths
    resumed: bool
    reason: str


def _write_csv_header(writer: csv.writer, fields: Iterable[str]) -> None:
    writer.writerow(list(fields))


def _open_writer(path: Path) -> tuple[csv.writer, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", newline="")
    return csv.writer(handle), handle


def _resolve_stream(stream: StreamSettings):
    if not LSL_AVAILABLE or resolve_streams is None:
        raise RuntimeError("pylsl is required for recording.")
    matches = [
        info
        for info in resolve_streams()
        if (not stream.name or info.name() == stream.name)
        and (not stream.stype or info.type() == stream.stype)
    ]
    if not matches:
        raise RuntimeError("No matching LSL stream found.")
    return matches[0]


def _make_features(window: np.ndarray) -> List[float]:
    means = window.mean(axis=0)
    stds = window.std(axis=0)
    return [float(x) for x in np.concatenate([means, stds])]


def _timebase_check(timestamps: Deque[float], max_gap_s: float) -> TimebaseCheck:
    return check_timebase_invariants(list(timestamps), max_gap_s=max_gap_s)


def _register_signals(state: RecordState) -> None:
    def _handler(_sig, _frame):
        state.stop_requested = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handler)
        except Exception:
            continue


def record(
    *,
    stream: StreamSettings,
    recorder: RecorderSettings,
    logger,
    duration_s: Optional[float] = None,
) -> RecordArtifacts:
    if not LSL_AVAILABLE:
        raise RuntimeError("pylsl is required for recording.")

    paths, resumed, reason = prepare_session_paths(
        output_root=recorder.output_root,
        subject_id=recorder.subject_id,
        session_id=recorder.session_id,
        resume=recorder.resume,
    )
    if logger is not None:
        try:
            logger = SessionLoggerAdapter(logger.logger, {"session_id": paths.session_id})
        except Exception:
            pass
    state = RecordState()
    state.stop_requested = False
    _register_signals(state)

    info = _resolve_stream(stream)
    inlet = StreamInlet(info)

    raw_writer, raw_handle = _open_writer(paths.raw_path)
    features_writer, features_handle = _open_writer(paths.features_path)
    events_writer = None
    events_handle = None
    if recorder.events_enabled:
        events_writer, events_handle = _open_writer(paths.events_path)

    if not resumed or paths.raw_path.stat().st_size == 0:
        _write_csv_header(
            raw_writer,
            ["time_s", "lsl_ts", "latency_ms", *stream.labels],
        )
    if not resumed or paths.features_path.stat().st_size == 0:
        feature_fields = [
            "time_s",
            "lsl_ts",
            "window_start_s",
            "window_end_s",
            "latency_ms",
        ]
        feature_fields += [f"mean_{label.lower()}" for label in stream.labels]
        feature_fields += [f"std_{label.lower()}" for label in stream.labels]
        _write_csv_header(features_writer, feature_fields)
    if events_writer and (not resumed or paths.events_path.stat().st_size == 0):
        _write_csv_header(events_writer, ["event_time_s", "event_lsl_ts", "event_type"])

    sample_buffer: Deque[tuple[float, np.ndarray]] = deque(maxlen=int(stream.nominal_srate * 5))
    window_start_s: Optional[float] = None
    last_feature_time_s: Optional[float] = None
    sample_timestamps: Deque[float] = deque(maxlen=256)

    if events_writer:
        events_writer.writerow([0.0, "", "session_start"])

    start_wall = time.monotonic()
    logger.info("Recording started (%s)", reason)

    try:
        while not state.stop_requested:
            if duration_s is not None and (time.monotonic() - start_wall) >= duration_s:
                break
            sample, lsl_ts = inlet.pull_sample(timeout=0.2)
            if sample is None:
                continue

            lsl_ts = float(lsl_ts)
            lsl_ts, clamped = clamp_monotonic(lsl_ts, state.last_lsl_ts)
            state.last_lsl_ts = lsl_ts

            if state.stream_start_lsl_ts is None:
                state.stream_start_lsl_ts = lsl_ts
                window_start_s = 0.0

            time_s = absolute_v1_time_s(lsl_ts, state.stream_start_lsl_ts)
            state.last_time_s = time_s
            sample_arr = np.asarray(sample, dtype=float)

            sample_buffer.append((time_s, sample_arr))
            sample_timestamps.append(lsl_ts)

            current_latency_ms = latency_ms(float(local_clock()), lsl_ts)
            raw_writer.writerow([time_s, lsl_ts, current_latency_ms, *sample_arr.tolist()])

            if window_start_s is None:
                continue

            while (window_start_s + recorder.window_sec) <= time_s:
                window_end_s = window_start_s + recorder.window_sec
                times = np.array([t for t, _ in sample_buffer], dtype=float)
                values = np.array([v for _, v in sample_buffer], dtype=float)
                mask = (times >= window_start_s) & (times < window_end_s)
                if not np.any(mask):
                    window_start_s += recorder.window_hop_sec
                    continue

                window_times = times[mask]
                window_values = values[mask]

                alignment = verify_alignment(
                    window_times,
                    start_s=window_start_s,
                    end_s=window_end_s,
                    target_fs=recorder.target_fs,
                    max_gap_s=1.0 / recorder.target_fs * 4,
                )
                if not alignment.ok:
                    logger.warning(
                        "Window alignment warning: %s", alignment.reason
                    )
                    window_start_s += recorder.window_hop_sec
                    continue

                grid, window = resample_window(
                    window_times,
                    window_values,
                    start_s=window_start_s,
                    end_s=window_end_s,
                    target_fs=recorder.target_fs,
                )
                if window.shape[0] != alignment.window_size:
                    logger.warning("Window size mismatch: %s", window.shape[0])
                    window_start_s += recorder.window_hop_sec
                    continue

                center_s = window_start_s + recorder.window_sec / 2.0
                feature_lsl_ts = state.stream_start_lsl_ts + center_s
                feature_latency = latency_ms(float(local_clock()), feature_lsl_ts)
                features = _make_features(window)
                features_writer.writerow(
                    [center_s, feature_lsl_ts, window_start_s, window_end_s, feature_latency, *features]
                )
                last_feature_time_s = center_s
                window_start_s += recorder.window_hop_sec

            if len(sample_timestamps) >= 2:
                check = _timebase_check(sample_timestamps, max_gap_s=1.0)
                if not check.ok:
                    logger.warning("Timebase invariant warning: %s", ",".join(check.warnings))
    finally:
        if events_writer and state.last_time_s is not None and state.stream_start_lsl_ts is not None:
            end_lsl = state.stream_start_lsl_ts + state.last_time_s
            events_writer.writerow([state.last_time_s, end_lsl, "session_end"])

        raw_handle.close()
        features_handle.close()
        if events_handle:
            events_handle.close()

    logger.info("Recording complete (last_feature_time_s=%s)", last_feature_time_s)
    return RecordArtifacts(paths=paths, resumed=resumed, reason=reason)
