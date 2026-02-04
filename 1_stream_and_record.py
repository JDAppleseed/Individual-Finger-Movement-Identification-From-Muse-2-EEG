#!/usr/bin/env python3
"""
Step 1 (record-only): connect to an existing LSL stream, show a live plot for
visual inspection, allow live event marking, and save raw.csv + events.csv.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import queue
import shutil
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from muse_streaming.io_paths import default_raw_dir
from muse_streaming.packets import SamplePacket
from muse_streaming.session_writer import SessionWriter
from muse_streaming.timebase import clamp_monotonic
from utils.stream_runtime import FailedWriters, HardStopPolicy, HealthStopState, StreamRequirements
from utils.label_schema import (
    ACTION_CLOSE,
    ACTION_OPEN,
    ACTION_REST,
    FINGER_INDEX,
    FINGER_MIDDLE,
    FINGER_NONE,
    FINGER_PINKY,
    FINGER_RING,
    FINGER_THUMB,
    event_type_for,
    is_valid_action_finger,
)

DEFAULT_SUBJECT_ID = "8-M16"
GENDER = "M"
AGE = 16
TIMEBASE_VERSION = "absolute_v1"

SAMPLING_RATE = 256
CHANNELS = 4

PLOT_FPS = 20.0
PLOT_DISPLAY_FS = 64.0
PLOT_QUEUE_MAXSIZE = 512
PLOT_SCALE_MODE = "fixed"
PLOT_FIXED_YLIM = (-200.0, 200.0)
PLOT_ROBUST_WINDOW_SEC = 5.0
PLOT_ROBUST_EMA = 0.2
PLOT_REFERENCE_OVERLAY = False
PLOT_WINDOW_SEC = 5.0

EVENT_MARKING_ENABLED = True
DEFAULT_EVENT_KEYMAP = "space:mark,1:thumb,2:index,3:middle,4:ring,5:pinky,o:open,c:close,r:rest"

LSL_RESOLVE_TIMEOUT = 2.0
LSL_INLET_MAX_BUFLEN_SEC = 2
LSL_INLET_MAX_CHUNKLEN = 1

RAW_QUEUE_MAXSIZE = 4096
RAW_SHARD_SAMPLES = 2048
MAX_BACKPRESSURE_S = 3.0
QUEUE_PUT_TIMEOUT_S = 0.1

HEARTBEAT_INTERVAL_S = 5.0
NO_SAMPLE_TIMEOUT_S = 5.0
WRITE_STALL_TIMEOUT_S = 5.0
WARMUP_SAMPLE_COUNT = 3
WARMUP_TIMEOUT_S = 3.0
EVENT_FLUSH_INTERVAL_S = 1.0

MODE = "train_record"
ALLOW_DROP = False

RAW_FLAG_NONFINITE = 1
INTEGRITY_GAP_TOLERANCE_MULT = 1.5

logger = logging.getLogger("step1_record")

stop_event = threading.Event()
termination_reason = "normal"


@dataclass
class StreamState:
    gap_count: int = 0


state = StreamState()

last_report_time = 0.0
last_report_samples_written = 0
timebase_report_initialized = False
last_live_viz_emit = 0.0

stream_requirements = StreamRequirements(
    required_labels=["TP9", "AF7", "AF8", "TP10"],
    require_exact_channels=False,
    expected_channels=4,
)

hard_stop_policy = HardStopPolicy(
    hard_stop_after_unhealthy_s=2.0,
    failed_write_window_s=5.0,
    failed_dir="data/failed",
    hard_stop_exit_code=2,
)

failed_writers = FailedWriters()
health_state = HealthStopState()


@dataclass
class EventRecord:
    event_time_s: float
    lsl_ts_mono: float
    local_ts: float
    label: str
    metadata: Dict[str, Any]


class EventRecorder:
    def __init__(self, writer: csv.writer, lock: threading.Lock) -> None:
        self._writer = writer
        self._lock = lock

    def record(self, event: EventRecord) -> None:
        """
        Write one row to the legacy events.csv schema (primary inspection artifact).
        """
        md = dict(event.metadata or {})
        onset_s = float(event.event_time_s)
        duration_s = float(md.get("duration_s") or md.get("duration") or 0.0)
        ev_type = str(event.label)

        channel = md.get("channel", "n/a")
        confidence = md.get("confidence", "")
        notes = md.get("notes", "")

        finger_id = md.get("finger_id", md.get("finger", ""))
        action_id = md.get("action_id", md.get("action", ""))
        trial_id = md.get("trial_id", md.get("trial", ""))
        block_id = md.get("block_id", md.get("block", ""))
        source = md.get("source", "keyboard")

        # Keep the CSV human-friendly; avoid dumping huge JSON blobs here.
        with self._lock:
            self._writer.writerow(
                [
                    onset_s,
                    duration_s,
                    ev_type,
                    channel,
                    confidence,
                    notes,
                    finger_id,
                    action_id,
                    trial_id,
                    block_id,
                    source,
                ]
            )



# NOTE: Legacy writer retained for historical reference; Step 1 now uses SessionWriter
# as the single source of truth for lossless session artifacts.
class SidecarNewFormatWriter:
    """
    Writes the "new-format" session artifacts expected by validate_session.py and downstream steps:
      - raw/eeg_raw_shard_*.npy  (dtype includes seq, lsl_ts_mono, sample)
      - events/events.jsonl
      - meta.json
      - manifest.json (written on finalize)

    This writer is intentionally independent of the CSV inspection artifacts.
    """

    def __init__(
        self,
        *,
        session_dir: Path,
        subject_id: str,
        session_id: str,
        channel_labels: List[str],
        sampling_rate: float,
        timebase_version: str,
        shard_size_samples: int,
    ) -> None:
        self.session_dir = session_dir
        self.subject_id = subject_id
        self.session_id = session_id
        self.channel_labels = list(channel_labels)
        self.sampling_rate = float(sampling_rate)
        self.timebase_version = str(timebase_version)
        self.shard_size_samples = int(shard_size_samples)

        self.raw_dir = self.session_dir / "raw"
        self.events_dir = self.session_dir / "events"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.events_dir.mkdir(parents=True, exist_ok=True)

        self.events_path = self.events_dir / "events.jsonl"
        self._events_f = self.events_path.open("a", encoding="utf-8")

        self._dtype = np.dtype(
            [
                ("seq", "<i8"),
                ("lsl_ts_mono", "<f8"),
                ("sample", "<f4", (len(self.channel_labels),)),
            ]
        )
        self._buf = np.empty(self.shard_size_samples, dtype=self._dtype)
        self._buf_n = 0
        self._seq_out = 0
        self._shard_start_seq = 0
        self._shard_paths: List[Path] = []

        self._t0_mono: Optional[float] = None
        self._tN_mono: Optional[float] = None
        self._created_utc = datetime.now(timezone.utc).isoformat()

        self._write_meta(initial=True)

    def _write_meta(self, *, initial: bool) -> None:
        meta = {
            "schema_version": 1,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "timebase_version": self.timebase_version,
            "channel_labels": self.channel_labels,
            "sampling_rate_hz": self.sampling_rate,
            "created_utc": self._created_utc,
            "complete": False if initial else True,
            "sample_count": int(self._seq_out),
            "lsl_ts_mono_start": self._t0_mono,
            "lsl_ts_mono_end": self._tN_mono,
        }
        (self.session_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def append_sample(self, *, lsl_ts_mono: float, sample: Iterable[float]) -> None:
        if self._t0_mono is None:
            self._t0_mono = float(lsl_ts_mono)
        self._tN_mono = float(lsl_ts_mono)

        i = self._buf_n
        self._buf["seq"][i] = int(self._seq_out)
        self._buf["lsl_ts_mono"][i] = float(lsl_ts_mono)
        self._buf["sample"][i] = np.asarray(list(sample), dtype=np.float32)
        self._buf_n += 1
        self._seq_out += 1

        if self._buf_n >= self.shard_size_samples:
            self._flush_shard()

    def append_event(self, event: Dict[str, Any]) -> None:
        # ensure JSON-serializable; write one JSON per line
        self._events_f.write(json.dumps(event, sort_keys=False) + "\n")
        self._events_f.flush()

    def _flush_shard(self) -> None:
        if self._buf_n <= 0:
            return
        shard_path = self.raw_dir / f"eeg_raw_shard_{self._shard_start_seq:06d}.npy"
        np.save(shard_path, self._buf[: self._buf_n].copy())
        self._shard_paths.append(shard_path)
        self._shard_start_seq = int(self._seq_out)
        self._buf_n = 0

    def finalize(self, *, termination_reason: str, missing_seq_count: int = 0) -> None:
        self._flush_shard()
        self._events_f.close()

        manifest = {
            "schema_version": 1,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "timebase_version": self.timebase_version,
            "termination_reason": str(termination_reason),
            "missing_seq_count": int(missing_seq_count),
            "shard_list": [
                {"path": str(p.relative_to(self.session_dir))} for p in self._shard_paths
            ],
        }
        (self.session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        self._write_meta(initial=False)

# === Test helpers / legacy hooks === / legacy hooks ===

def _build_session_state_payload(state_obj: StreamState) -> Dict[str, Any]:
    return {
        "gap_count": int(state_obj.gap_count),
    }


def _apply_channel_indices(
    sample: Iterable[float], indices: List[int], channel_count: int
) -> List[float]:
    values = list(sample)
    # If the stream publishes more channels than we need, allow it.
    # We only require that the requested indices are within bounds.
    if not indices:
        return [float(v) for v in values]
    if max(indices) >= len(values):
        return [float(v) for v in values]
    return [float(values[idx]) for idx in indices]


def _should_accept_ica_result(result_segment_id: int, current_segment_id: int) -> bool:
    return int(result_segment_id) == int(current_segment_id)


def _reset_segment_state(_state: StreamState) -> None:
    global last_report_time, last_report_samples_written
    global timebase_report_initialized, last_live_viz_emit
    last_report_time = 0.0
    last_report_samples_written = 0
    timebase_report_initialized = False
    last_live_viz_emit = 0.0


def _evaluate_label_check(
    found_labels: Optional[List[str]], channel_count: int
) -> Dict[str, Any]:
    expected_n = int(stream_requirements.expected_channels)
    required_labels = list(stream_requirements.required_labels)
    found_labels = [str(x) for x in (found_labels or [])]

    errors: list[str] = []
    if stream_requirements.require_exact_channels and channel_count != expected_n:
        errors.append("channel_count_mismatch")
    if stream_requirements.require_exact_channels:
        if found_labels[:expected_n] != required_labels:
            errors.append("label_mismatch")
    else:
        missing = [lab for lab in required_labels if lab not in found_labels]
        if missing:
            errors.append("missing_labels")

    return {
        "ok": not errors,
        "reason": "ok" if not errors else ",".join(errors),
        "expected_labels": required_labels,
        "found_labels": found_labels[:expected_n]
        if stream_requirements.require_exact_channels
        else found_labels,
        "channel_count": int(channel_count),
        "expected_channel_count": expected_n,
        "require_exact_channels": bool(stream_requirements.require_exact_channels),
    }


def _route_writers_for_health(_now_mono: float, decision) -> None:
    if not health_state.has_health_decision:
        return
    if decision is None:
        return
    if decision.healthy:
        return
    if failed_writers.is_open():
        return
    failed_writers.open_failed_files(
        prefix="failed",
        headers=["time_s"],
        save_raw=True,
        save_preds=False,
        failed_dir=hard_stop_policy.failed_dir,
        prediction_header=[],
        raw_header=_raw_header(CHANNELS),
    )


def _enqueue_with_overflow(
    target_queue: queue.Queue,
    packet: SamplePacket,
    *,
    label: str,
    timeout_s: float,
    allow_drop: bool,
) -> bool:
    global termination_reason
    try:
        target_queue.put(packet, timeout=float(timeout_s))
        return True
    except queue.Full:
        if not allow_drop:
            termination_reason = "backpressure_abort"
            stop_event.set()
            logger.error("Backpressure abort on %s queue.", label)
        return False


def _raw_header(channel_count: int) -> List[str]:
    return [
        "seq",
        "lsl_ts_raw",
        "lsl_ts_mono",
        "local_ts",
        *[f"ch{idx + 1}" for idx in range(channel_count)],
        "clamped",
        "segment_id",
        "flags",
    ]


def _open_raw_csv(path: Path, channel_count: int = CHANNELS) -> tuple[Any, csv.writer]:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    file_obj = path.open("a", newline="", buffering=1024 * 1024)
    writer = csv.writer(file_obj)
    if not exists:
        writer.writerow(_raw_header(channel_count))
        file_obj.flush()  # persist header immediately
    return file_obj, writer


def _open_events_csv(path: Path) -> tuple[Any, csv.writer]:
    """
    Legacy human-readable events.csv (primary inspection artifact).

    Schema (matches legacy sessions):
      onset_s,duration_s,type,channel,confidence,notes,finger_id,action_id,trial_id,block_id,source

    NOTE: The new-format events/events.jsonl is written separately for pipeline steps.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    file_obj = path.open("a", newline="", buffering=1)
    writer = csv.writer(file_obj)
    if not exists:
        writer.writerow(
            [
                "onset_s",
                "duration_s",
                "type",
                "channel",
                "confidence",
                "notes",
                "finger_id",
                "action_id",
                "trial_id",
                "block_id",
                "source",
            ]
        )
    return file_obj, writer


def _raw_flags_for_sample(sample: Iterable[float]) -> int:
    try:
        if not np.all(np.isfinite(sample)):
            return RAW_FLAG_NONFINITE
    except Exception:
        return RAW_FLAG_NONFINITE
    return 0


def _build_raw_row(
    lsl_ts_raw: float,
    lsl_ts_mono: float,
    local_ts: float,
    sample: Iterable[float],
    flags: int,
    seq: Optional[int] = None,
    clamped: bool = False,
    segment_id: int = 0,
) -> List[Any]:
    seq_val = int(seq) if seq is not None else -1
    return [
        seq_val,
        float(lsl_ts_raw),
        float(lsl_ts_mono),
        float(local_ts),
        *[float(x) for x in sample],
        int(bool(clamped)),
        int(segment_id),
        int(flags),
    ]


def _gap_tolerance_s(nominal_dt_s: float) -> float:
    return float(nominal_dt_s) * float(INTEGRITY_GAP_TOLERANCE_MULT)


def _estimate_missing_samples(
    dt_s: float, nominal_dt_s: float, gap_tolerance_s: float
) -> int:
    if nominal_dt_s <= 0 or dt_s <= gap_tolerance_s:
        return 0
    return max(0, int(round(float(dt_s) / float(nominal_dt_s))) - 1)


def analyze_lsl_timestamp_gaps(
    timestamps: Iterable[float], nominal_fs: float
) -> Dict[str, Any]:
    finite_ts = [float(ts) for ts in timestamps if ts is not None and np.isfinite(ts)]
    if len(finite_ts) < 2 or nominal_fs <= 0:
        return {
            "duration_s": 0.0,
            "expected_samples": 0,
            "gap_count": 0,
            "estimated_missing": 0,
            "max_gap_s": None,
        }
    nominal_dt = 1.0 / float(nominal_fs)
    tolerance = _gap_tolerance_s(nominal_dt)
    gap_count = 0
    missing_total = 0
    max_gap_s = 0.0
    prev = finite_ts[0]
    for ts in finite_ts[1:]:
        dt = float(ts - prev)
        if dt > tolerance:
            gap_count += 1
            missing_total += _estimate_missing_samples(dt, nominal_dt, tolerance)
            max_gap_s = max(max_gap_s, dt)
        prev = ts
    duration_s = float(finite_ts[-1] - finite_ts[0])
    expected_samples = int(round(float(nominal_fs) * duration_s))
    return {
        "duration_s": duration_s,
        "expected_samples": expected_samples,
        "gap_count": gap_count,
        "estimated_missing": missing_total,
        "max_gap_s": max_gap_s if gap_count else None,
    }


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse_keymap(text: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not text:
        return mapping
    parts = [part.strip() for part in text.split(",") if part.strip()]
    for part in parts:
        if ":" not in part:
            continue
        key, label = part.split(":", 1)
        key = key.strip().lower()
        label = label.strip()
        if key and label:
            mapping[key] = label
    return mapping


def _normalize_label_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        return [part for part in parts if part]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            item_str = str(item).strip()
            if item_str:
                out.append(item_str)
        return out
    return []


def _format_stream_info(info_obj) -> str:
    parts = []
    try:
        parts.append(f"name={info_obj.name()}")
    except Exception:
        parts.append("name=?")
    try:
        parts.append(f"type={info_obj.type()}")
    except Exception:
        parts.append("type=?")
    try:
        parts.append(f"ch={info_obj.channel_count()}")
    except Exception:
        parts.append("ch=?")
    try:
        parts.append(f"rate={info_obj.nominal_srate()}")
    except Exception:
        parts.append("rate=?")
    try:
        parts.append(f"source_id={info_obj.source_id()}")
    except Exception:
        pass
    return ", ".join(parts)


def _key_to_name(key) -> Optional[str]:
    try:
        if hasattr(key, "char") and key.char:
            return str(key.char).lower()
    except Exception:
        pass
    try:
        import pynput

        if key == pynput.keyboard.Key.space:
            return "space"
        if key == pynput.keyboard.Key.esc:
            return "esc"
    except Exception:
        pass
    return None


def _should_stop_key(name: str) -> bool:
    return name in {"esc", "q"}


def _write_run_meta(session_dir: Path, payload: Dict[str, Any]) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "run_meta.json").write_text(json.dumps(payload, indent=2))


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def _write_session_state(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path),
        ],
    )


def _attach_session_log(session_log_path: Path, previous_log_path: Optional[Path] = None) -> None:
    session_log_path.parent.mkdir(parents=True, exist_ok=True)
    if previous_log_path and previous_log_path.exists() and not session_log_path.exists():
        try:
            shutil.copy2(previous_log_path, session_log_path)
        except Exception:
            pass

    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            existing = Path(getattr(handler, "baseFilename", "")).resolve()
            if existing == session_log_path.resolve():
                return
            if previous_log_path and existing == previous_log_path.resolve():
                root_logger.removeHandler(handler)
                try:
                    handler.close()
                except Exception:
                    pass

    session_handler = logging.FileHandler(session_log_path)
    session_handler.setLevel(logging.INFO)
    session_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    root_logger.addHandler(session_handler)


def _force_interactive_matplotlib_backend(logger: logging.Logger) -> None:
    """Force a GUI-capable Matplotlib backend.

    On macOS, the default MacOSX backend can show a blank/transparent window when this
    script is launched as a subprocess from a Qt app. QtAgg is typically the most robust.

    Behavior:
      - Prefer QtAgg when available.
      - Fall back to TkAgg, then MacOSX.
      - Respect an explicit MPLBACKEND unless it is clearly non-interactive.
    """
    try:
        import matplotlib  # noqa: WPS433
    except Exception as e:
        logger.warning("[plot] Matplotlib not available: %s", e)
        return

    env_backend = os.environ.get("MPLBACKEND")
    logger.info("[plot] env MPLBACKEND=%s", env_backend)

    def _has_qt() -> bool:
        try:
            import PyQt5  # noqa: F401
            return True
        except Exception:
            pass
        try:
            import PySide6  # noqa: F401
            return True
        except Exception:
            return False

    def _has_tk() -> bool:
        try:
            import tkinter  # noqa: F401
            return True
        except Exception:
            return False

    non_interactive = {"agg", "pdf", "ps", "svg", "cairo", "template"}
    if env_backend and env_backend.strip().lower() not in non_interactive:
        try:
            matplotlib.interactive(True)
        except Exception:
            pass
        try:
            logger.info("[plot] Using matplotlib backend=%s", matplotlib.get_backend())
        except Exception:
            logger.info("[plot] Using matplotlib backend=(unknown)")
        return

    chosen = None
    if sys.platform == "darwin":
        if _has_qt():
            chosen = "QtAgg"
        elif _has_tk():
            chosen = "TkAgg"
        else:
            chosen = "MacOSX"

    if chosen:
        try:
            matplotlib.use(chosen, force=True)
        except Exception as e:
            logger.warning("[plot] Failed to set backend=%s (%s).", chosen, e)

    try:
        matplotlib.interactive(True)
    except Exception:
        pass

    try:
        logger.info("[plot] Using matplotlib backend=%s", matplotlib.get_backend())
    except Exception:
        logger.info("[plot] Using matplotlib backend=(unknown)")


def _plot_process_main(
    *,
    sample_queue: mp.Queue,
    stop_flag: mp.Event,
    channel_labels: list[str],
    plot_window_sec: float,
    plot_fps: float,
    plot_fixed_ylim: tuple[float, float],
    plot_scale: str,
    plot_robust_ema: float,
    plot_reference_overlay: bool,
    title: str,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    plot_logger = logging.getLogger("step1_plot")
    try:
        _force_interactive_matplotlib_backend(plot_logger)
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception as exc:
        plot_logger.error("[plot] Plot process failed to import matplotlib: %s", exc)
        return

    channel_count = len(channel_labels)
    plot_scale = _normalize_scale_mode(plot_scale)
    plot_window_sec = float(plot_window_sec)
    plot_fps = float(plot_fps)
    plot_robust_ema = float(plot_robust_ema)
    plot_fixed_ylim = _resolve_plot_fixed_ylim(list(plot_fixed_ylim))

    plt.ion()
    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    lines = [ax.plot([], [])[0] for _ in range(channel_count)]
    plot_stack_step_uv = 250.0
    plot_offsets = np.arange(channel_count, dtype=float) * plot_stack_step_uv
    try:
        ax.set_yticks(plot_offsets.tolist())
        ax.set_yticklabels([str(l) for l in channel_labels])
    except Exception:
        pass
    ax.set_title("EEG (uV)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")

    overlay_lines = []
    if plot_reference_overlay:
        for off in plot_offsets:
            overlay_lines.append(
                ax.axhline(float(off), color="#888888", alpha=0.2, linewidth=0.6)
            )

    base_half = max(abs(plot_fixed_ylim[0]), abs(plot_fixed_ylim[1]))
    ax.set_ylim(float(plot_offsets[0] - base_half), float(plot_offsets[-1] + base_half))
    ax.set_xlim(-plot_window_sec, 0.0)

    times: deque[float] = deque()
    values: deque[np.ndarray] = deque()
    plot_ylim_ema: Optional[Tuple[float, float]] = None
    last_draw = 0.0

    def _trim(now_s: float) -> None:
        while times and (now_s - float(times[0])) > plot_window_sec:
            times.popleft()
            values.popleft()

    def _draw(now_s: float) -> None:
        nonlocal plot_ylim_ema, last_draw
        if plot_fps > 0 and (now_s - last_draw) < (1.0 / plot_fps):
            return
        last_draw = now_s
        if not times:
            return
        t_arr = np.asarray(times, dtype=float)
        v_arr = np.stack(values, axis=0) if len(values) > 1 else np.asarray(values[0:1])
        t0 = float(t_arr[-1])
        x = t_arr - t0

        for idx in range(v_arr.shape[1]):
            y = v_arr[:, idx] + plot_offsets[idx]
            lines[idx].set_data(x, y)
        ax.set_xlim(-plot_window_sec, 0.0)

        if plot_scale == "robust":
            flat = v_arr.reshape(-1)
            if flat.size > 0:
                low, high = np.percentile(flat, [5, 95])
                if low == high:
                    low -= 1.0
                    high += 1.0
                target_low, target_high = float(low), float(high)
                if plot_ylim_ema is None:
                    plot_ylim_ema = (target_low, target_high)
                else:
                    alpha = max(0.0, min(1.0, plot_robust_ema))
                    plot_ylim_ema = (
                        (1.0 - alpha) * plot_ylim_ema[0] + alpha * target_low,
                        (1.0 - alpha) * plot_ylim_ema[1] + alpha * target_high,
                    )
            if plot_ylim_ema is not None:
                half = max(abs(plot_ylim_ema[0]), abs(plot_ylim_ema[1]))
                half = float(max(50.0, min(400.0, half)))
                ax.set_ylim(
                    float(plot_offsets[0] - half), float(plot_offsets[-1] + half)
                )
        else:
            half = max(abs(plot_fixed_ylim[0]), abs(plot_fixed_ylim[1]))
            half = float(max(50.0, min(400.0, half)))
            ax.set_ylim(float(plot_offsets[0] - half), float(plot_offsets[-1] + half))

        if overlay_lines:
            try:
                for line in overlay_lines:
                    line.set_alpha(0.2)
            except Exception:
                pass

        fig.canvas.draw_idle()
        plt.pause(0.001)

    while not stop_flag.is_set():
        try:
            item = sample_queue.get(timeout=0.1)
        except queue.Empty:
            item = None
        if item is None:
            if not plt.fignum_exists(fig.number):
                break
            continue
        try:
            now_s, sample = item
            now_s = float(now_s)
            sample_arr = np.asarray(sample, dtype=float)
            if sample_arr.ndim != 1 or sample_arr.size != channel_count:
                continue
            times.append(now_s)
            values.append(sample_arr)
            _trim(now_s)
            _draw(now_s)
        except Exception:
            continue


class _PlotProcess:
    def __init__(
        self,
        *,
        enabled: bool,
        channel_labels: list[str],
        plot_window_sec: float,
        plot_fps: float,
        plot_fixed_ylim: tuple[float, float],
        plot_scale: str,
        plot_robust_ema: float,
        plot_reference_overlay: bool,
        title: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.dropped = 0
        self._queue: Optional[mp.Queue] = None
        self._stop: Optional[mp.Event] = None
        self._proc: Optional[mp.Process] = None
        if not self.enabled:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=int(PLOT_QUEUE_MAXSIZE))
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=_plot_process_main,
            kwargs={
                "sample_queue": self._queue,
                "stop_flag": self._stop,
                "channel_labels": list(channel_labels),
                "plot_window_sec": float(plot_window_sec),
                "plot_fps": float(plot_fps),
                "plot_fixed_ylim": tuple(plot_fixed_ylim),
                "plot_scale": str(plot_scale),
                "plot_robust_ema": float(plot_robust_ema),
                "plot_reference_overlay": bool(plot_reference_overlay),
                "title": str(title),
            },
            daemon=True,
        )
        self._proc.start()

    def push(self, *, now_s: float, sample: np.ndarray) -> None:
        if not self._queue or not self._proc or not self._proc.is_alive():
            return
        try:
            self._queue.put_nowait((float(now_s), np.asarray(sample, dtype=np.float32)))
        except queue.Full:
            self.dropped += 1
        except Exception:
            return

    def stop(self) -> None:
        if not self._stop or not self._proc:
            return
        try:
            self._stop.set()
        except Exception:
            pass
        try:
            self._proc.join(timeout=1.0)
        except Exception:
            pass
        if self._proc.is_alive():
            try:
                self._proc.terminate()
            except Exception:
                pass


def _resolve_plot_fixed_ylim(value: Optional[List[float]]) -> Tuple[float, float]:
    if not value or len(value) != 2:
        return float(PLOT_FIXED_YLIM[0]), float(PLOT_FIXED_YLIM[1])
    low = float(value[0])
    high = float(value[1])
    if low == high:
        if low == 0:
            return -200.0, 200.0
        return low - abs(low), low + abs(low)
    return (min(low, high), max(low, high))


def _normalize_scale_mode(value: str) -> str:
    val = (value or "").strip().lower()
    if val in {"robust", "robust_auto", "auto"}:
        return "robust"
    return "fixed"


def _should_enable_plot(value: Optional[bool]) -> bool:
    return bool(value)


def _load_config_file(path: Optional[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not path:
        return {}, {}
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}, {}
    if not isinstance(payload, dict):
        return {}, {}
    settings = payload.get("settings")
    if isinstance(settings, dict):
        return payload, settings
    return payload, payload


def _apply_config_to_args(
    args_obj, settings: Dict[str, Any], defaults: Dict[str, Any]
) -> List[str]:
    """
    Merge JSON config settings into argparse args with strict precedence:
      CLI args > config JSON > defaults

    This intentionally prefers canonical UPPER_CASE config keys (written by the UI) over
    legacy "shadow" lower-case keys that may exist in older configs (e.g. ENABLE_PLOT
    vs enable_plot). Conflicts are reported as warnings.
    """

    # alias key -> (arg dest, priority). Higher priority wins when multiple keys map to same dest.
    alias_specs: Dict[str, tuple[str, int]] = {
        "ENABLE_PLOT": ("enable_plot", 100),
        "EVENT_MARKING_ENABLED": ("event_marking_enabled", 100),
        "EVENT_KEYMAP": ("event_keymap", 100),
        "LSL_STREAM_NAME": ("stream_name", 100),
        "LSL_STREAM_TYPE": ("stream_type", 100),
        "LSL_SOURCE_ID": ("lsl_source_id", 100),
        "RAW_QUEUE_MAXSIZE": ("raw_queue_maxsize", 100),
        "RAW_SHARD_SAMPLES": ("raw_shard_samples", 100),
        "SESSION_ID_OVERRIDE": ("session_id", 100),
        "SESSION_NAME": ("session_id", 90),
        "MODE": ("mode", 100),
        "OUTPUT_DIR": ("output_dir", 100),
        "OUT_DIR": ("output_dir", 90),
        "RAW_DIR": ("output_dir", 90),
        "raw_dir": ("output_dir", 10),
        "DURATION_S": ("duration_s", 100),
        "DRY_RUN": ("dry_run", 100),
        "STREAM_CH": ("stream_ch", 100),
        "STREAM_RATE": ("stream_rate", 100),
        # Plot options (prefer UPPER_CASE keys from UI configs)
        "PLOT_SCALE_MODE": ("plot_scale", 100),
        "PLOT_FIXED_YLIM": ("plot_fixed_ylim", 100),
        "PLOT_FIXED_UV": ("plot_fixed_uv", 100),
        "PLOT_FIXED_YLIM_MIN": ("plot_fixed_ylim_min", 100),
        "PLOT_FIXED_YLIM_MAX": ("plot_fixed_ylim_max", 100),
        "PLOT_ROBUST_WINDOW_SEC": ("plot_robust_window_sec", 100),
        "PLOT_ROBUST_EMA": ("plot_robust_ema", 100),
        "PLOT_REFERENCE_OVERLAY": ("plot_reference_overlay", 100),
        "PLOT_REFERENCE_LINES": ("plot_reference_overlay", 90),
        "PLOT_WINDOW_SEC": ("plot_window_sec", 100),
        "plot-window-sec": ("plot_window_sec", 5),
        "plotWindowSec": ("plot_window_sec", 5),
    }

    chosen: Dict[str, tuple[int, str, Any]] = {}
    warnings: List[str] = []
    normalized: Dict[str, Any] = {}

    for key, val in (settings or {}).items():
        dest, priority = alias_specs.get(key, (key, 0))
        # Only apply to known argparse destinations.
        if dest not in defaults:
            continue
        existing = chosen.get(dest)
        if existing is None:
            chosen[dest] = (priority, str(key), val)
        else:
            existing_priority, existing_key, existing_val = existing
            if priority > existing_priority:
                if existing_val != val:
                    warnings.append(
                        f"Config key conflict for {dest}: {existing_key}={existing_val!r} overridden by {key}={val!r}"
                    )
                chosen[dest] = (priority, str(key), val)
            elif priority == existing_priority and existing_val != val:
                warnings.append(
                    f"Config key conflict for {dest}: {existing_key}={existing_val!r} and {key}={val!r} (keeping {existing_key})"
                )

    for dest, (_priority, _key, val) in chosen.items():
        normalized[dest] = val

    # Derived plot_fixed_ylim convenience.
    if "plot_fixed_uv" in normalized and "plot_fixed_ylim" not in normalized:
        try:
            uv = float(normalized["plot_fixed_uv"])
            normalized["plot_fixed_ylim"] = [-uv, uv]
        except Exception:
            pass
    if (
        "plot_fixed_ylim_min" in normalized
        and "plot_fixed_ylim_max" in normalized
        and "plot_fixed_ylim" not in normalized
    ):
        normalized["plot_fixed_ylim"] = [
            normalized["plot_fixed_ylim_min"],
            normalized["plot_fixed_ylim_max"],
        ]

    for dest, default in defaults.items():
        if dest not in normalized:
            continue
        current = getattr(args_obj, dest)
        if current != default:
            # CLI overrides config.
            continue
        val = normalized[dest]
        if val is None and default is not None:
            # Avoid clobbering non-optional CLI defaults with nulls from older configs.
            continue
        setattr(args_obj, dest, val)

    return warnings


def _run_recording(
    args: argparse.Namespace,
    *,
    config_payload: Dict[str, Any],
    config_settings: Dict[str, Any],
    config_warnings: Optional[List[str]] = None,
) -> int:
    global termination_reason
    stop_event.clear()
    termination_reason = "normal"
    _reset_segment_state(state)
    try:
        from pylsl import StreamInlet, local_clock, resolve_streams
        try:
            from pylsl import resolve_byprop
        except Exception:
            resolve_byprop = None
    except Exception:
        logger.error("pylsl is required for recording.")
        return 2

    cfg = dict(config_settings or {})
    repo_root = Path(__file__).resolve().parent
    project_name = (
        config_payload.get("project_name") if isinstance(config_payload, dict) else None
    )
    subject_id = args.subject_id or (
        (config_payload.get("subject_id") if isinstance(config_payload, dict) else None)
    )
    subject_id = str(subject_id or DEFAULT_SUBJECT_ID)

    session_dir_arg = getattr(args, "session_dir", None)
    session_dir = Path(session_dir_arg).expanduser().resolve() if session_dir_arg else None
    if session_dir is None:
        session_id_cfg = (
            config_payload.get("session_id") if isinstance(config_payload, dict) else None
        )
        session_id_cfg = str(session_id_cfg) if session_id_cfg and session_id_cfg != "PENDING" else None
        if project_name and session_id_cfg:
            session_dir = (
                repo_root
                / "Projects"
                / str(project_name)
                / "subjects"
                / subject_id
                / "sessions"
                / session_id_cfg
            ).resolve()
        else:
            session_root = (
                repo_root
                / "Projects"
                / str(project_name or "DEFAULT")
                / "subjects"
                / subject_id
                / "sessions"
            ).resolve()
            session_root.mkdir(parents=True, exist_ok=True)
            backend_id = args.session_id or time.strftime("%Y%m%d_%H%M%S")
            backend_id = str(backend_id)
            session_ui = (
                backend_id
                if backend_id.startswith(f"{subject_id}_")
                else f"{subject_id}_{backend_id}"
            )
            session_dir = (session_root / session_ui).resolve()

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "logs").mkdir(parents=True, exist_ok=True)
    bootstrap_log_path = session_dir / "logs" / "step1.log"
    _configure_logging(bootstrap_log_path)
    for w in config_warnings or []:
        logger.warning("[config] %s", w)
    logger.info("Project: %s", project_name or "DEFAULT")
    logger.info("Subject: %s", subject_id)
    logger.info("Session dir (requested): %s", session_dir)

    def _handle_signal(sig, _frame) -> None:
        global termination_reason
        if stop_event.is_set():
            return
        name = getattr(sig, "name", str(sig))
        termination_reason = f"signal_{name}"
        logger.info("Received %s; stopping recording.", name)
        stop_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            pass

    stream_name = args.stream_name
    stream_type = args.stream_type
    source_id = args.lsl_source_id

    resolve_timeout_s = float(cfg.get("LSL_RESOLVE_TIMEOUT", LSL_RESOLVE_TIMEOUT))
    inlet_max_buflen_sec = int(cfg.get("LSL_INLET_MAX_BUFLEN_SEC", LSL_INLET_MAX_BUFLEN_SEC))
    inlet_max_chunklen = int(cfg.get("LSL_INLET_MAX_CHUNKLEN", LSL_INLET_MAX_CHUNKLEN))
    heartbeat_interval_s = float(cfg.get("HEARTBEAT_INTERVAL_S", HEARTBEAT_INTERVAL_S))
    no_sample_timeout_s = float(cfg.get("NO_SAMPLE_TIMEOUT_S", NO_SAMPLE_TIMEOUT_S))
    write_stall_timeout_s = float(cfg.get("WRITE_STALL_TIMEOUT_S", WRITE_STALL_TIMEOUT_S))
    warmup_sample_count = int(cfg.get("WARMUP_SAMPLE_COUNT", WARMUP_SAMPLE_COUNT))
    warmup_timeout_s = float(cfg.get("WARMUP_TIMEOUT_S", WARMUP_TIMEOUT_S))
    event_flush_interval_s = float(cfg.get("EVENT_FLUSH_INTERVAL_S", EVENT_FLUSH_INTERVAL_S))
    duration_s = cfg.get("DURATION_S", None)
    if duration_s is None:
        duration_s = getattr(args, "duration_s", None)
    if duration_s is not None:
        try:
            duration_s = float(duration_s)
        except Exception:
            duration_s = None
    if duration_s is not None and duration_s <= 0.0:
        duration_s = None
    if duration_s is not None:
        logger.info("Max duration enabled: %.2fs", duration_s)

    # Guard against unsafe config/UI values that can silently break runtime.
    try:
        args.raw_queue_maxsize = int(args.raw_queue_maxsize)
    except Exception:
        args.raw_queue_maxsize = int(RAW_QUEUE_MAXSIZE)
    if args.raw_queue_maxsize <= 0:
        logger.warning(
            "[config] raw_queue_maxsize=%r invalid; using default=%d",
            getattr(args, "raw_queue_maxsize", None),
            RAW_QUEUE_MAXSIZE,
        )
        args.raw_queue_maxsize = int(RAW_QUEUE_MAXSIZE)

    try:
        args.raw_shard_samples = int(args.raw_shard_samples)
    except Exception:
        args.raw_shard_samples = int(RAW_SHARD_SAMPLES)
    if args.raw_shard_samples <= 0:
        logger.warning(
            "[config] raw_shard_samples=%r invalid; using default=%d",
            getattr(args, "raw_shard_samples", None),
            RAW_SHARD_SAMPLES,
        )
        args.raw_shard_samples = int(RAW_SHARD_SAMPLES)

    queue_put_timeout_s = float(cfg.get("QUEUE_PUT_TIMEOUT_S", QUEUE_PUT_TIMEOUT_S))
    if (not np.isfinite(queue_put_timeout_s)) or queue_put_timeout_s <= 0:
        logger.warning(
            "[config] QUEUE_PUT_TIMEOUT_S=%r invalid; using default=%.3fs",
            cfg.get("QUEUE_PUT_TIMEOUT_S", None),
            QUEUE_PUT_TIMEOUT_S,
        )
        queue_put_timeout_s = float(QUEUE_PUT_TIMEOUT_S)

    allow_drop = bool(cfg.get("ALLOW_DROP", ALLOW_DROP))

    hard_stop_after_unhealthy_s = float(
        cfg.get("HARD_STOP_AFTER_UNHEALTHY_S", hard_stop_policy.hard_stop_after_unhealthy_s)
    )
    if (not np.isfinite(hard_stop_after_unhealthy_s)) or hard_stop_after_unhealthy_s <= 0:
        logger.warning(
            "[config] HARD_STOP_AFTER_UNHEALTHY_S=%r invalid; using default=%.2fs",
            cfg.get("HARD_STOP_AFTER_UNHEALTHY_S", None),
            hard_stop_policy.hard_stop_after_unhealthy_s,
        )
        hard_stop_after_unhealthy_s = float(hard_stop_policy.hard_stop_after_unhealthy_s)

    required_labels = _normalize_label_list(cfg.get("REQUIRED_LSL_LABELS"))
    if not required_labels:
        required_labels = list(stream_requirements.required_labels)
    require_exact = bool(cfg.get("REQUIRE_EXACTLY_4_CHANNELS", stream_requirements.require_exact_channels))
    expected_channels = int(cfg.get("EXPECTED_CHANNELS", args.stream_ch or stream_requirements.expected_channels))
    expected_srate = float(cfg.get("EXPECTED_SAMPLING_RATE", args.stream_rate or SAMPLING_RATE))
    requirements = StreamRequirements(
        required_labels=required_labels,
        require_exact_channels=require_exact,
        expected_channels=int(expected_channels),
    )

    logger.info(
        "Stream selector: name=%s type=%s source_id=%s expected_ch=%s expected_rate=%s required_labels=%s require_exact=%s",
        stream_name,
        stream_type,
        source_id,
        expected_channels,
        expected_srate,
        required_labels,
        require_exact,
    )
    logger.info(
        "Timing: resolve_timeout=%.2fs inlet_buflen=%ss inlet_chunklen=%s heartbeat=%.2fs no_sample_timeout=%.2fs write_stall_timeout=%.2fs",
        resolve_timeout_s,
        inlet_max_buflen_sec,
        inlet_max_chunklen,
        heartbeat_interval_s,
        no_sample_timeout_s,
        write_stall_timeout_s,
    )

    # --------------------------
    # Resolve LSL stream robustly
    # --------------------------
    desired_source_id = args.lsl_source_id
    env_source_id = os.environ.get("LSL_SOURCE_ID")

    # UI historically passed a placeholder like "muse2_internal".
    # Treat these as "auto" and prefer the connector-provided env var if present.
    # If no env var is available, DO NOT filter by source_id (pick the best matching EEG stream).
    AUTO_TOKENS = {None, "", "auto", "muse2_internal", "internal"}
    desired_is_auto = (desired_source_id in AUTO_TOKENS)
    if desired_is_auto:
        desired_source_id = env_source_id or None

    def _score_stream(info_obj) -> float:
        sid = ""
        try:
            sid = info_obj.source_id()
        except Exception:
            sid = ""
        score = 0.0
        if env_source_id and sid == env_source_id:
            score += 1e12
        if desired_source_id and sid == desired_source_id:
            score += 5e11
        # Parse trailing epoch-ms if present: muse2-<rand>-<epochms>
        try:
            tail = sid.split("-")[-1]
            score += float(int(tail))
        except Exception:
            pass
        try:
            score += 1e4 * float(info_obj.channel_count())
        except Exception:
            pass
        try:
            score += 10.0 * float(info_obj.nominal_srate())
        except Exception:
            pass
        return score

    def _list_candidates():
        # Resolve ALL streams, then filter. This is more robust than resolve_byprop/resolve_stream
        # (which require exact matches and can miss streams during startup or when stale streams exist).
        all_streams = resolve_streams(wait_time=resolve_timeout_s)
        def norm(s: str) -> str:
            return (s or "").strip()
        want_name = norm(args.stream_name)
        want_type = norm(args.stream_type)
        want_source_id = norm(desired_source_id)  # may be '', exact match preferred when user provided it
        env_sid = norm(env_source_id)
        def stream_ok(info: StreamInfo) -> bool:
            # Basic sanity
            try:
                ch = int(info.channel_count())
            except Exception:
                ch = 0
            if ch <= 0:
                return False

            if want_name and norm(info.name()) != want_name:
                return False
            if want_type and norm(info.type()) != want_type:
                return False
            if requirements.expected_channels and ch < int(requirements.expected_channels):
                return False

            if expected_srate:
                try:
                    sr = float(info.nominal_srate())
                except Exception:
                    return False
                if sr <= 0:
                    return False
                tol = max(0.5, float(expected_srate) * 0.05)
                if abs(sr - float(expected_srate)) > tol:
                    return False
            return True
        # Candidate pre-filter
        candidates = [s for s in all_streams if stream_ok(s)]
        # If user/environment requested a specific source_id and it exists, prefer it.
        # Otherwise, prefer any stream whose source_id contains "muse2" (case-insensitive),
        # which matches Muse connector-generated IDs like "muse2-....".
        muse2_substr = "muse2"
        def sid(info: StreamInfo) -> str:

            try:
                return norm(info.source_id())
            except Exception:
                return ""
        # If no candidates by name/type, fall back to "any Muse2 EEG-like stream"
        if not candidates:
            loose = []
            for s in all_streams:
                try:
                    if norm(s.type()) != want_type and want_type:
                        continue
                    if int(s.channel_count()) < int(requirements.expected_channels or 4):
                        continue
                    if muse2_substr not in sid(s).lower():
                        continue
                    loose.append(s)
                except Exception:
                    continue
            candidates = loose

        # Emit a helpful debug dump when desired
        if args.verbose:
            print("[lsl] resolve_streams found:", len(all_streams))
            for s in all_streams:
                try:
                    print(f"  - name={s.name()} type={s.type()} ch={s.channel_count()} sr={s.nominal_srate()} source_id={sid(s)}")
                except Exception:
                    pass
            print("[lsl] candidates:", len(candidates))
            for s in candidates[:10]:
                print(f"  * name={s.name()} type={s.type()} ch={s.channel_count()} sr={s.nominal_srate()} source_id={sid(s)}")

        streams = candidates

        # Optional label check (only enforced when required_labels provided).
        # When the stream has >4 channels, we still only select the required labels or the first 4.
        if requirements.required_labels:
            def get_stream_labels(stream):
                labels = []
                try:
                    ch = stream.desc().child("channels").child("channel")
                    for _ in range(stream.channel_count()):
                        labels.append(ch.child_value("label") or "")
                        ch = ch.next_sibling()
                except Exception:
                    labels = []
                return [lab.strip() for lab in labels if lab is not None]

            labeled = []
            for s in streams:
                labels = get_stream_labels(s)
                if labels and all(lab in labels for lab in requirements.required_labels):
                    labeled.append(s)
            if labeled:
                streams = labeled

        return streams, all_streams

    candidates, all_streams = _list_candidates()
    if not candidates:
        logger.error(
            "No LSL streams found for name=%s type=%s ch=%s rate=%s (requested source_id=%s, env LSL_SOURCE_ID=%s).",
            args.stream_name,
            args.stream_type,
            args.stream_ch,
            args.stream_rate,
            args.lsl_source_id,
            env_source_id,
        )
        if all_streams:
            logger.error("Available LSL streams (%d):", len(all_streams))
            for s in all_streams:
                logger.error("  - %s", _format_stream_info(s))
        logger.error(
            "Next steps: start the Muse LSL streamer or verify stream name/type. "
            "Try: `python -m muse_streaming.cli list-streams` or `python -m muse_streaming.cli start-streamer`."
        )
        return 1

    # Log all candidates for debugging / operator disambiguation.
    candidates_sorted = list(candidates)
    logger.info("[lsl] Found %d candidate stream(s):", len(candidates_sorted))
    for i, s in enumerate(candidates_sorted):
        try:
            logger.info(
                "[lsl]  #%d name=%s type=%s ch=%s rate=%s source_id=%s uid=%s",
                i,
                s.name(),
                s.type(),
                s.channel_count(),
                s.nominal_srate(),
                s.source_id(),
                s.uid(),
            )
        except Exception:
            logger.info("[lsl]  #%d (unable to print full stream info)", i)

    explicit_source_id = str(args.lsl_source_id).strip() if args.lsl_source_id else ""
    AUTO_TOKENS = {"", "auto", "muse2_internal", "internal"}
    if explicit_source_id.lower() in AUTO_TOKENS:
        explicit_source_id = ""
    preferred_source_id = explicit_source_id or (str(env_source_id).strip() if env_source_id else "")

    def _sid(stream) -> str:
        try:
            return str(stream.source_id()).strip()
        except Exception:
            return ""

    filtered = candidates_sorted
    if preferred_source_id:
        filtered = [s for s in filtered if _sid(s) == preferred_source_id]
        if not filtered:
            logger.error(
                "[lsl] No candidate matched source_id=%s. Set --lsl-source-id or LSL_SOURCE_ID to one of:",
                preferred_source_id,
            )
            for s in candidates_sorted:
                logger.error("  - %s", _format_stream_info(s))
            return 1

    if not preferred_source_id and len(filtered) > 1:
        logger.error(
            "[lsl] Multiple streams match name/type/ch/rate. Provide --lsl-source-id or set LSL_SOURCE_ID."
        )
        for i, s in enumerate(filtered):
            logger.error("  #%d %s", i, _format_stream_info(s))
        return 1

    if len(filtered) > 1:
        logger.error(
            "[lsl] Multiple streams matched source_id=%s. Stop duplicates or refine stream-name/type.",
            preferred_source_id,
        )
        for i, s in enumerate(filtered):
            logger.error("  #%d %s", i, _format_stream_info(s))
        return 1

    info = filtered[0]
    try:
        inlet = StreamInlet(
            info, max_buflen=inlet_max_buflen_sec, max_chunklen=inlet_max_chunklen
        )
        sample, ts = inlet.pull_sample(timeout=1.0)
        if sample is None or ts is None:
            raise RuntimeError("no sample within 1s (stale stream?)")
    except Exception as e:
        logger.error("[lsl] Selected stream produced no samples: %s", e)
        return 1

    logger.info(
        "Connected to LSL stream: name=%s type=%s ch=%s rate=%s source_id=%s",
        info.name(),
        info.type(),
        info.channel_count(),
        info.nominal_srate(),
        info.source_id(),
    )

    if inlet is None:
        inlet = StreamInlet(
            info, max_buflen=inlet_max_buflen_sec, max_chunklen=inlet_max_chunklen
        )
    logger.info(
        "LSL inlet configured: max_buflen=%ss max_chunklen=%s",
        inlet_max_buflen_sec,
        inlet_max_chunklen,
    )

    if getattr(args, "dry_run", False):
        logger.info("Dry-run requested; stream resolved successfully. Exiting without recording.")
        return 0

    def _extract_stream_labels(info_obj) -> List[str]:
        labels: List[str] = []
        try:
            ch = info_obj.desc().child("channels").child("channel")
            for _ in range(info_obj.channel_count()):
                labels.append(ch.child_value("label") or "")
                ch = ch.next_sibling()
        except Exception:
            labels = []
        return [lab.strip() for lab in labels if lab is not None and str(lab).strip()]
    channel_count = int(info.channel_count())
    nominal_srate = float(info.nominal_srate() or SAMPLING_RATE)
    stream_labels = _extract_stream_labels(info)
    if stream_labels and len(stream_labels) >= channel_count:
        channel_labels = stream_labels[:channel_count]
    elif requirements.required_labels and len(requirements.required_labels) == channel_count:
        channel_labels = list(requirements.required_labels)
    else:
        channel_labels = [f"ch{i + 1}" for i in range(channel_count)]
    logger.info("Channel labels: %s", channel_labels)

    output_root = session_dir.parent
    session_id = session_dir.name
    session_writer = SessionWriter(
        output_root=output_root,
        subject_id=subject_id,
        session_id=session_id,
        channel_labels=channel_labels,
        sampling_rate=nominal_srate,
        timebase_version=TIMEBASE_VERSION,
        shard_size_samples=int(args.raw_shard_samples),
        resume=False,
        mode="train_record",
    )
    session_id = session_writer.session_id
    session_dir = session_writer.paths.session_dir
    _attach_session_log(session_dir / "logs" / "step1.log", previous_log_path=bootstrap_log_path)
    if session_dir != Path(getattr(args, "session_dir", session_dir)).expanduser().resolve():
        logger.info("Session dir updated: %s", session_dir)

    raw_csv_path = session_dir / "raw" / "raw.csv"
    events_csv_path = session_dir / "events" / "events.csv"
    raw_file, raw_writer = _open_raw_csv(raw_csv_path, channel_count=channel_count)
    raw_flush_interval_s = float(cfg.get("RAW_FLUSH_INTERVAL_S", cfg.get("raw_flush_interval_s", 1.0)))
    last_raw_flush_t = time.time()
    events_file, events_writer = _open_events_csv(events_csv_path)
    events_lock = threading.Lock()
    event_recorder = EventRecorder(events_writer, events_lock)

    session_writer.update_meta(
        {
            "project_name": str(project_name or "DEFAULT"),
            "raw_csv_path": "raw/raw.csv",
            "events_csv_path": "events/events.csv",
            "session_dir": str(session_dir),
        }
    )
    try:
        session_writer.update_manifest_files(
            {
                "raw_csv": "raw/raw.csv",
                "events_csv": "events/events.csv",
                "events_jsonl": "events/events.jsonl",
            }
        )
    except Exception:
        logger.exception("Failed to update manifest files mapping.")

    logger.info("Raw CSV (legacy inspection): %s", raw_csv_path)
    logger.info("Events CSV (legacy inspection): %s", events_csv_path)

    if args.init_only:
        raw_file.close()
        events_file.close()
        session_writer.finalize(
            "init_only",
            extra_manifest={
                "files": {
                    "raw_csv": "raw/raw.csv",
                    "events_csv": "events/events.csv",
                    "events_jsonl": "events/events.jsonl",
                }
            },
        )
        return 0

    plot_scale = _normalize_scale_mode(args.plot_scale)
    plot_fixed_ylim = _resolve_plot_fixed_ylim(args.plot_fixed_ylim)
    plot_robust_window_sec = float(args.plot_robust_window_sec)
    plot_robust_ema = float(args.plot_robust_ema)
    plot_reference_overlay = bool(args.plot_reference_overlay)
    # B2: Add a final safety clamp to prevent non-positive/invalid plot window duration.
    # This guards against config/UI edge cases that can set 0.0 or NaN.
    plot_window_sec = float(getattr(args, 'plot_window_sec', PLOT_WINDOW_SEC) or PLOT_WINDOW_SEC)
    if (not np.isfinite(plot_window_sec)) or plot_window_sec <= 0.0:
        logger.warning("[plot] Invalid plot_window_sec=%r; defaulting to 5.0", getattr(args, 'plot_window_sec', None))
        plot_window_sec = 5.0

    enable_plot = _should_enable_plot(args.enable_plot)
    if args.event_marking_enabled is None:
        event_marking_config_enabled = bool(EVENT_MARKING_ENABLED)
    else:
        event_marking_config_enabled = bool(args.event_marking_enabled)
    event_keymap = _parse_keymap(args.event_keymap)
    if event_marking_config_enabled:
        logger.info("Event marking enabled (keypresses ignored until first sample).")
    else:
        logger.info("Event marking disabled; events.jsonl will remain empty.")

    raw_queue: queue.Queue[SamplePacket] = queue.Queue(maxsize=int(args.raw_queue_maxsize))
    writer_exc: Optional[BaseException] = None
    counter_lock = threading.Lock()
    samples_received = 0
    samples_written = 0
    events_received = 0
    events_written = 0
    last_sample_wall: Optional[float] = None
    last_write_wall: Optional[float] = None
    queue_max_depth = 0
    start_wall = time.monotonic()
    last_heartbeat = start_wall
    last_event_flush_t = time.time()

    def _writer_worker() -> None:
        nonlocal writer_exc, last_raw_flush_t, samples_written, last_write_wall
        try:
            batch: List[SamplePacket] = []
            while not stop_event.is_set() or not raw_queue.empty():
                try:
                    packet = raw_queue.get(timeout=0.1)
                except queue.Empty:
                    packet = None
                if packet is not None:
                    batch.append(packet)

                # A1.1: Guard against empty batch before costly processing.
                if not batch:
                    continue

                if len(batch) >= 128 or (packet is None and batch):
                    # NOTE: session_writer is now robust, but we still avoid empty calls.
                    session_writer.append_packets(batch)
                    for pkt in batch:
                        row = _build_raw_row(
                            pkt.lsl_ts_raw,
                            pkt.lsl_ts_mono,
                            pkt.local_ts,
                            pkt.sample,
                            pkt.flags,
                            seq=pkt.seq,
                            clamped=pkt.clamped,
                            segment_id=pkt.segment_id,
                        )
                        raw_writer.writerow(row)
                        with counter_lock:
                            samples_written += 1
                            last_write_wall = time.monotonic()
                        now_t = time.time()
                        if now_t - last_raw_flush_t >= raw_flush_interval_s:
                            raw_file.flush()
                            last_raw_flush_t = now_t
                    batch = []
            if batch:
                session_writer.append_packets(batch)
                for pkt in batch:
                    row = _build_raw_row(
                        pkt.lsl_ts_raw,
                        pkt.lsl_ts_mono,
                        pkt.local_ts,
                        pkt.sample,
                        pkt.flags,
                        seq=pkt.seq,
                        clamped=pkt.clamped,
                        segment_id=pkt.segment_id,
                    )
                    raw_writer.writerow(row)
                    with counter_lock:
                        samples_written += 1
                        last_write_wall = time.monotonic()
                    now_t = time.time()
                    if now_t - last_raw_flush_t >= raw_flush_interval_s:
                        raw_file.flush()
                        last_raw_flush_t = now_t
        # A2: Make writer-thread failures visible and fatal.
        except BaseException as e:
            logger.exception("Writer thread crashed")
            writer_exc = e

    writer_thread = threading.Thread(target=_writer_worker, daemon=True)
    writer_thread.start()
    plot_display_fs = float(cfg.get("PLOT_DISPLAY_FS", PLOT_DISPLAY_FS))
    plot_fps = float(cfg.get("PLOT_FPS", PLOT_FPS))
    plotter = _PlotProcess(
        enabled=enable_plot,
        channel_labels=channel_labels,
        plot_window_sec=plot_window_sec,
        plot_fps=plot_fps,
        plot_fixed_ylim=plot_fixed_ylim,
        plot_scale=plot_scale,
        plot_robust_ema=plot_robust_ema,
        plot_reference_overlay=plot_reference_overlay,
        title=f"Step 1: Recording {subject_id} - {session_id}",
    )
    plot_decim = 1
    if enable_plot and plot_display_fs > 0:
        try:
            plot_decim = max(1, int(round(float(nominal_srate) / float(plot_display_fs))))
        except Exception:
            plot_decim = 1

    resolved_settings = {
        "project_name": str(project_name or "DEFAULT"),
        "subject_id": str(subject_id),
        "session_id": str(session_id),
        "session_dir": str(session_dir),
        "stream_name": str(stream_name),
        "stream_type": str(stream_type),
        "lsl_source_id": str(args.lsl_source_id) if args.lsl_source_id is not None else None,
        "expected_channels": int(expected_channels),
        "expected_sampling_rate": float(expected_srate),
        "enable_plot": bool(enable_plot),
        "plot_fps": float(plot_fps),
        "plot_display_fs": float(plot_display_fs),
        "plot_window_sec": float(plot_window_sec),
        "plot_decim": int(plot_decim),
        "plot_scale": str(plot_scale),
        "plot_fixed_ylim": [float(plot_fixed_ylim[0]), float(plot_fixed_ylim[1])],
        "plot_reference_overlay": bool(plot_reference_overlay),
        "event_marking_enabled": bool(event_marking_config_enabled),
        "event_keymap": str(args.event_keymap),
        "raw_queue_maxsize": int(args.raw_queue_maxsize),
        "raw_shard_samples": int(args.raw_shard_samples),
        "queue_put_timeout_s": float(queue_put_timeout_s),
        "allow_drop": bool(allow_drop),
        "duration_s": float(duration_s) if duration_s is not None else None,
    }
    try:
        (session_dir / "logs" / "resolved_settings.json").write_text(
            json.dumps(resolved_settings, indent=2, sort_keys=True)
        )
    except Exception:
        logger.exception("Failed to write resolved_settings.json")
    try:
        session_writer.update_meta({"resolved_settings": resolved_settings})
    except Exception:
        logger.exception("Failed to persist resolved_settings into meta.json")
    logger.info("Resolved settings: %s", json.dumps(resolved_settings, sort_keys=True))

    last_state_update = 0.0
    stream_start_lsl_ts: Optional[float] = None
    last_lsl_ts_mono: Optional[float] = None
    seq = 0
    segment_id = 0
    timestamps_recent: deque[float]
    timestamps_recent = deque(maxlen=512)
    event_marking_active = False

    action_map = {
        "rest": ACTION_REST,
        "open": ACTION_OPEN,
        "close": ACTION_CLOSE,
    }
    finger_map = {
        "none": FINGER_NONE,
        "thumb": FINGER_THUMB,
        "index": FINGER_INDEX,
        "middle": FINGER_MIDDLE,
        "ring": FINGER_RING,
        "pinky": FINGER_PINKY,
    }

    current_action_id = int(ACTION_REST)
    current_finger_id = int(FINGER_NONE)
    pending_override_type: Optional[str] = None
    active_mark: Optional[dict[str, object]] = None

    def _now_event_time_s() -> Optional[tuple[float, float, bool]]:
        nonlocal last_lsl_ts_mono
        if stream_start_lsl_ts is None:
            return None
        lsl_ts_raw = float(local_clock())
        lsl_ts_mono, clamped = clamp_monotonic(lsl_ts_raw, last_lsl_ts_mono)
        last_lsl_ts_mono = lsl_ts_mono
        return float(lsl_ts_mono - stream_start_lsl_ts), float(lsl_ts_mono), bool(clamped)

    def _emit_event(
        *,
        onset_s: float,
        duration_s: float,
        action_id: int,
        finger_id: int,
        event_type: str,
        source_key: str,
        clamped: bool,
        lsl_ts_mono: float,
    ) -> None:
        nonlocal events_received, events_written, last_event_flush_t
        duration_s = max(0.0, float(duration_s))
        if not is_valid_action_finger(int(action_id), int(finger_id)):
            # Fail safe: coerce invalid combos to REST/NONE.
            action_id = int(ACTION_REST)
            finger_id = int(FINGER_NONE)
            event_type = "rest"
        md = {
            "duration_s": float(duration_s),
            "action_id": int(action_id),
            "finger_id": int(finger_id),
            "source": "keyboard",
            "key": str(source_key),
            "clamped": bool(clamped),
        }
        payload = EventRecord(
            event_time_s=float(onset_s),
            lsl_ts_mono=float(lsl_ts_mono),
            local_ts=time.time(),
            label=str(event_type),
            metadata=md,
        )
        event_recorder.record(payload)
        with counter_lock:
            events_received += 1
            events_written += 1
        session_writer.append_event(
            {
                "onset_s": float(onset_s),
                "event_time_s": float(onset_s),
                "duration_s": float(duration_s),
                "end_s": float(onset_s + duration_s),
                "lsl_ts_mono": float(lsl_ts_mono),
                "local_ts": float(payload.local_ts),
                "type": str(event_type),
                "action_id": int(action_id),
                "finger_id": int(finger_id),
                "source": "keyboard",
            }
        )
        if (time.time() - last_event_flush_t) >= event_flush_interval_s:
            events_file.flush()
            last_event_flush_t = time.time()

    def _start_mark(source_key: str) -> None:
        nonlocal active_mark
        if active_mark is not None:
            return
        now = _now_event_time_s()
        if now is None:
            return
        onset_s, lsl_ts_mono, clamped = now
        action_id = int(current_action_id)
        finger_id = int(current_finger_id)
        if action_id == int(ACTION_REST):
            finger_id = int(FINGER_NONE)
        override = pending_override_type
        event_type = (
            str(override) if override else event_type_for(int(action_id), int(finger_id))
        )
        active_mark = {
            "onset_s": float(onset_s),
            "lsl_ts_mono": float(lsl_ts_mono),
            "clamped": bool(clamped),
            "action_id": int(action_id),
            "finger_id": int(finger_id),
            "type": str(event_type),
            "key": str(source_key),
        }
        logger.info(
            "[event] mark_start onset=%.3fs type=%s action_id=%s finger_id=%s",
            float(onset_s),
            str(event_type),
            int(action_id),
            int(finger_id),
        )

    def _end_mark(source_key: str) -> None:
        nonlocal active_mark, pending_override_type
        if active_mark is None:
            return
        now = _now_event_time_s()
        if now is None:
            active_mark = None
            pending_override_type = None
            return
        end_s, lsl_ts_mono, clamped = now
        onset_s = float(active_mark.get("onset_s", end_s))
        duration_s = max(0.0, float(end_s - onset_s))
        action_id = int(active_mark.get("action_id", ACTION_REST))
        finger_id = int(active_mark.get("finger_id", FINGER_NONE))
        event_type = str(active_mark.get("type", ""))
        _emit_event(
            onset_s=onset_s,
            duration_s=duration_s,
            action_id=action_id,
            finger_id=finger_id,
            event_type=event_type,
            source_key=source_key,
            clamped=bool(clamped),
            lsl_ts_mono=float(lsl_ts_mono),
        )
        logger.info(
            "[event] mark_end onset=%.3fs dur=%.3fs type=%s",
            float(onset_s),
            float(duration_s),
            str(event_type),
        )
        active_mark = None
        pending_override_type = None

    listener = None
    if event_marking_config_enabled:
        try:
            from pynput import keyboard

            def _on_press(key) -> None:
                nonlocal current_action_id, current_finger_id, pending_override_type
                global termination_reason
                name = _key_to_name(key)
                if not name:
                    return
                if _should_stop_key(name):
                    termination_reason = "user_stop"
                    _end_mark(name)
                    stop_event.set()
                    return
                label = event_keymap.get(name)
                if not label:
                    return
                label = str(label).strip().lower()
                if label in action_map:
                    current_action_id = int(action_map[label])
                    if current_action_id == int(ACTION_REST):
                        current_finger_id = int(FINGER_NONE)
                    logger.info("[event] mode action=%s (action_id=%s)", label, current_action_id)
                    return
                if label in finger_map:
                    current_finger_id = int(finger_map[label])
                    logger.info("[event] mode finger=%s (finger_id=%s)", label, current_finger_id)
                    return
                if label in {"artifact", "calibration"}:
                    pending_override_type = label
                    logger.info("[event] override=%s (next mark)", label)
                    return
                if label == "mark":
                    if event_marking_active:
                        _start_mark(name)
                    return

            def _on_release(key) -> None:
                name = _key_to_name(key)
                if not name:
                    return
                label = event_keymap.get(name)
                if not label:
                    return
                if str(label).strip().lower() == "mark":
                    _end_mark(name)

            listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
            listener.start()
        except Exception:
            logger.warning("pynput not available; event marking disabled.")
            event_marking_config_enabled = False

    session_state_paths = [
        session_dir / "logs" / "session_state.json",
        repo_root / "logs" / f"session_state_{subject_id}.json",
    ]
    run_error: Optional[Exception] = None

    try:
        while not stop_event.is_set():
            if writer_exc:
                raise RuntimeError("Writer thread crashed") from writer_exc
            if not writer_thread.is_alive():
                raise RuntimeError("Writer thread stopped unexpectedly")

            sample, lsl_ts = inlet.pull_sample(timeout=0.1)
            if sample is None:
                continue
            with counter_lock:
                samples_received += 1
                last_sample_wall = time.monotonic()
            lsl_ts_raw = float(lsl_ts)
            lsl_ts_mono, clamped = clamp_monotonic(lsl_ts_raw, last_lsl_ts_mono)
            last_lsl_ts_mono = lsl_ts_mono
            if stream_start_lsl_ts is None:
                stream_start_lsl_ts = lsl_ts_mono

            local_ts = time.time()
            sample_arr = np.asarray(sample, dtype=float)
            # Enforce exactly CHANNELS samples written downstream (1b_extract_windows expects 4 EEG channels).
            if sample_arr.ndim == 0:
                sample_arr = np.asarray([float(sample_arr)], dtype=float)
            if sample_arr.size >= CHANNELS:
                sample_arr = sample_arr[:CHANNELS]
            else:
                # pad missing channels with NaN to keep shape stable
                sample_arr = np.concatenate([sample_arr, np.full(CHANNELS - sample_arr.size, np.nan, dtype=float)])
            flags = _raw_flags_for_sample(sample_arr)
            packet = SamplePacket(
                seq=seq,
                lsl_ts_raw=lsl_ts_raw,
                lsl_ts_mono=lsl_ts_mono,
                local_ts=local_ts,
                sample=sample_arr,
                flags=flags,
                segment_id=segment_id,
                clamped=clamped,
                raw_path=None,
                segment_break_reason=None,
            )
            seq += 1
            timestamps_recent.append(lsl_ts_mono)
            if enable_plot and plot_decim > 0 and (packet.seq % plot_decim == 0):
                plotter.push(now_s=float(lsl_ts_mono - stream_start_lsl_ts), sample=sample_arr)

            if not _enqueue_with_overflow(
                raw_queue,
                packet,
                label="raw",
                timeout_s=queue_put_timeout_s,
                allow_drop=allow_drop,
            ):
                break
            queue_depth = raw_queue.qsize()
            if queue_depth > queue_max_depth:
                queue_max_depth = queue_depth

            now = time.monotonic()
            with counter_lock:
                samples_received_snapshot = samples_received
                samples_written_snapshot = samples_written
                events_written_snapshot = events_written
                last_sample_wall_snapshot = last_sample_wall
                last_write_wall_snapshot = last_write_wall

            if duration_s is not None and (now - start_wall) >= float(duration_s):
                termination_reason = "duration_elapsed"
                logger.info("Max duration reached (%.2fs); stopping.", float(duration_s))
                stop_event.set()
                break

            if samples_received_snapshot == 0 and (now - start_wall) > no_sample_timeout_s:
                termination_reason = "no_samples"
                logger.error(
                    "No samples received within %.2fs after start; aborting.",
                    no_sample_timeout_s,
                )
                logger.error(
                    "Next steps: confirm the Muse LSL streamer is running and producing EEG. "
                    "Try: `python -m muse_streaming.cli list-streams` or `python -m muse_streaming.cli healthcheck`."
                )
                stop_event.set()
                break

            if warmup_sample_count > 0 and samples_received_snapshot < warmup_sample_count and (now - start_wall) > warmup_timeout_s:
                termination_reason = "no_samples"
                logger.error(
                    "LSL stream did not produce required samples within %.2fs "
                    "(received %d < %d).",
                    warmup_timeout_s,
                    samples_received_snapshot,
                    warmup_sample_count,
                )
                logger.error(
                    "Next steps: confirm the Muse LSL streamer is running and producing EEG. "
                    "Try: `python -m muse_streaming.cli healthcheck`."
                )
                stop_event.set()
                break

            if samples_received_snapshot > 0 and samples_written_snapshot == 0:
                if (now - start_wall) > write_stall_timeout_s:
                    termination_reason = "write_stall"
                    logger.error(
                        "Samples received but none written after %.2fs; "
                        "writer_alive=%s queue_depth=%s.",
                        write_stall_timeout_s,
                        writer_thread.is_alive(),
                        raw_queue.qsize(),
                    )
                    stop_event.set()
                    break

            if now - last_heartbeat >= heartbeat_interval_s:
                last_sample_age = (
                    None
                    if last_sample_wall_snapshot is None
                    else float(now - last_sample_wall_snapshot)
                )
                last_write_age = (
                    None
                    if last_write_wall_snapshot is None
                    else float(now - last_write_wall_snapshot)
                )
                logger.info(
                    "[alive] recv=%d wrote=%d events=%d queue=%d plot_dropped=%d last_sample_age_s=%s last_write_age_s=%s",
                    samples_received_snapshot,
                    samples_written_snapshot,
                    events_written_snapshot,
                    raw_queue.qsize(),
                    int(getattr(plotter, "dropped", 0)),
                    "n/a" if last_sample_age is None else f"{last_sample_age:.2f}",
                    "n/a" if last_write_age is None else f"{last_write_age:.2f}",
                )
                last_heartbeat = now

            if now - last_state_update >= 1.0:
                is_healthy = True
                unhealthy_reason = "healthy"
                packet_rate = None
                if len(timestamps_recent) >= 2:
                    diffs = np.diff(np.array(timestamps_recent, dtype=float))
                    diffs = diffs[diffs > 0]
                    if diffs.size:
                        packet_rate = float(1.0 / np.median(diffs))

                last_age = None
                if last_lsl_ts_mono is not None:
                    last_age = float(local_clock() - last_lsl_ts_mono)
                    if last_age > hard_stop_after_unhealthy_s:
                        is_healthy = False
                        unhealthy_reason = f"stale LSL data (age: {last_age:.2f}s)"

                # B2/B4: Update event marking status and overlay
                current_event_marking_active = event_marking_config_enabled and is_healthy
                if current_event_marking_active != event_marking_active:
                    reason = "enabled by config and stream is healthy" if current_event_marking_active else unhealthy_reason
                    if not event_marking_config_enabled:
                        reason = "disabled by user config"
                    logger.info("Event marking %s: %s", "enabled" if current_event_marking_active else "disabled", reason)
                event_marking_active = current_event_marking_active

                backend_session_id = session_id
                try:
                    m = re.search(r"(\\d{8}_\\d{6})(?:_\\d{2})?$", str(session_id))
                    if m:
                        backend_session_id = m.group(1)
                except Exception:
                    backend_session_id = session_id
                state_payload = {
                    "subject_id": subject_id,
                    "session_id": backend_session_id,
                    "session_ui_id": session_id,
                    "session_dir": str(session_dir),
                    "updated_utc": _now_utc_iso(),
                    "data_stream_active": is_healthy,
                    "last_lsl_ts_raw": lsl_ts_raw,
                    "last_lsl_ts_mono": last_lsl_ts_mono,
                    "last_local_ts": local_ts,
                    "packet_rate_hz": packet_rate,
                    "last_sample_age_s": last_age,
                    "samples_received": samples_received_snapshot,
                    "samples_written": samples_written_snapshot,
                    "events_written": events_written_snapshot,
                    "queue_depth": raw_queue.qsize(),
                    "queue_max_depth": queue_max_depth,
                    "writer_alive": writer_thread.is_alive(),
                    "last_write_age_s": None
                    if last_write_wall_snapshot is None
                    else float(now - last_write_wall_snapshot),
                    "termination_reason": termination_reason,
                    "hard_stop_triggered": termination_reason == "backpressure_abort",
                    "event_marking_allowed": event_marking_active,
                }
                for state_path in session_state_paths:
                    _write_session_state(state_path, state_payload)
                last_state_update = now

    except KeyboardInterrupt:
        logger.info("Stopping recording.")
    except Exception as exc:
        run_error = exc
        logger.error("Recording error: %s", exc, exc_info=True)
    finally:
        stop_event.set()
        plotter.stop()
        if listener is not None:
            try:
                listener.stop()
            except Exception:
                pass
        writer_thread.join(timeout=2.0)
        raw_file.flush()
        raw_file.close()
        events_file.flush()
        events_file.close()

        if run_error is not None and termination_reason == "normal":
            termination_reason_final = "error"
        else:
            termination_reason_final = termination_reason
        try:
            session_writer.finalize(
                termination_reason_final,
                extra_manifest={
                    "files": {
                        "raw_csv": "raw/raw.csv",
                        "events_csv": "events/events.csv",
                        "events_jsonl": "events/events.jsonl",
                    }
                },
            )
        except Exception:
            logger.exception("Failed to finalize session writer.")

        try:
            _write_run_meta(
                session_dir,
                {
                    "subject_id": subject_id,
                    "session_id": session_id,
                    "termination_reason": termination_reason_final,
                    "created_utc": _now_utc_iso(),
                    "raw_csv": str(raw_csv_path),
                    "events_csv": str(events_csv_path),
                    "samples_received": samples_received,
                    "samples_written": samples_written,
                    "events_written": events_written,
                    "queue_max_depth": queue_max_depth,
                    "stream": {
                        "name": getattr(info, "name", lambda: None)(),
                        "type": getattr(info, "type", lambda: None)(),
                        "source_id": getattr(info, "source_id", lambda: None)(),
                        "channels": channel_count,
                        "nominal_srate": nominal_srate,
                    },
                },
            )
        except Exception:
            logger.exception("Failed to write run_meta.json.")

        try:
            backend_session_id = session_id
            try:
                m = re.search(r"(\\d{8}_\\d{6})(?:_\\d{2})?$", str(session_id))
                if m:
                    backend_session_id = m.group(1)
            except Exception:
                backend_session_id = session_id
            final_state = {
                "subject_id": subject_id,
                "session_id": backend_session_id,
                "session_ui_id": session_id,
                "session_dir": str(session_dir),
                "updated_utc": _now_utc_iso(),
                "data_stream_active": False,
                "termination_reason": termination_reason_final,
                "hard_stop_triggered": termination_reason_final == "backpressure_abort",
                "samples_received": samples_received,
                "samples_written": samples_written,
                "events_written": events_written,
            }
            for state_path in session_state_paths:
                _write_session_state(state_path, final_state)
        except Exception:
            logger.exception("Failed to write final session_state.")

    if writer_exc:
        logger.error("Writer thread failed, returning exit code 1.")
        return 1
    if run_error is not None:
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--subject-id", type=str, default=None)
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--session-name", dest="session_id", type=str, default=None)
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Canonical session directory (Projects/<project>/subjects/<subject>/sessions/<session_id>/...)",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--out-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--stream-name", type=str, default="Muse2-EEG")
    parser.add_argument("--stream-type", type=str, default="EEG")

    parser.add_argument("--stream-ch", dest="stream_ch", type=int, default=4,
                        help="Optional: expected channel count for the input LSL stream (used for filtering when auto-selecting).")
    parser.add_argument("--stream-rate", dest="stream_rate", type=float, default=256.0,
                        help="Optional: expected nominal sampling rate for the input LSL stream (used for filtering when auto-selecting).")
    parser.add_argument("--lsl-source-id", type=str, default=None)
    # CLI default: no plot (protect ingestion latency). UI/config can enable.
    parser.add_argument(
        "--enable-plot",
        dest="enable_plot",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--no-plot",
        dest="enable_plot",
        action="store_const",
        const=False,
    )
    parser.add_argument(
        "--plot-scale",
        type=str,
        choices=["fixed", "robust", "robust_auto"],
        default="fixed",
    )
    parser.add_argument("--plot-fixed-ylim", type=float, nargs=2, default=None)
    parser.add_argument("--plot-robust-window-sec", type=float, default=PLOT_ROBUST_WINDOW_SEC)
    parser.add_argument("--plot-robust-ema", type=float, default=PLOT_ROBUST_EMA)
    parser.add_argument("--plot-reference-overlay", action="store_true", default=False)
    parser.add_argument("--plot-window-sec", type=float, default=5.0, help="Seconds of EEG to display in the live plot.")
    parser.add_argument(
        "--event-marking-enabled",
        dest="event_marking_enabled",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--no-event-marking",
        dest="event_marking_enabled",
        action="store_const",
        const=False,
    )
    parser.add_argument("--event-keymap", type=str, default=DEFAULT_EVENT_KEYMAP)
    parser.add_argument("--raw-queue-maxsize", type=int, default=RAW_QUEUE_MAXSIZE)
    parser.add_argument("--raw-shard-samples", type=int, default=RAW_SHARD_SAMPLES)
    parser.add_argument("--mode", type=str, default="train_record")
    parser.add_argument("--init-only", action="store_true", default=False)
    parser.add_argument("--duration-s", type=float, default=None, help="Stop recording after N seconds.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Resolve LSL stream and exit without writing data.",
    )
    parser.add_argument("--force-new-session", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging.")

    args = parser.parse_args()
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    config_payload, config_settings = _load_config_file(args.config)
    config_warnings = _apply_config_to_args(args, config_settings, defaults)

    args.plot_scale = _normalize_scale_mode(args.plot_scale)

    return _run_recording(
        args, config_payload=config_payload, config_settings=config_settings, config_warnings=config_warnings
    )


if __name__ == "__main__":
    raise SystemExit(main())
