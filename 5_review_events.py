"""
STEP 5 — Interactive Event Review & Edit

Keyboard controls:
  Left/Right: move cursor by 0.1s
  Up/Down: move cursor by 1.0s
  n/p: next/previous event
  [/]: shrink/grow visible window
  e: edit selected event (terminal prompts)
  d: delete selected event
  s: save events.jsonl
  q: quit

Mouse controls:
  Drag main EEG view: pan visible window
  Click/drag timeline: jump quickly through the full session
  Scroll wheel: zoom visible window in/out
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Slider

from utils.events_audit import log_event_edit
from utils.label_schema import (
    ACTION_REST,
    FINGER_NONE,
    event_type_for,
)
from utils.session_event_io import (
    event_rows_to_flat_records,
    event_rows_to_payloads,
    load_events_dataframe,
    resolve_raw_shard_paths,
)

EVENTS_PATH = Path("events.jsonl")
DEFAULT_FS = 256.0
PLOT_FIXED_YLIM = (-200.0, 200.0)
PLOT_CHANNEL_SPACING_UV = 120.0
DEFAULT_WINDOW_SEC = 5.0
MIN_WINDOW_SEC = 0.5
MAX_WINDOW_SEC = 10.0
MAX_OVERVIEW_POINTS = 3000


def latest_subject_file(subject_id, suffix, base_dir):
    base = Path(base_dir)
    pattern = f"{subject_id}_*_{suffix}"
    candidates = sorted(base.glob(pattern), key=lambda p: p.name)
    return candidates[-1] if candidates else None


def _resolve_events_path(
    session_dir: Optional[Path], explicit_events: Optional[str]
) -> Path:
    if explicit_events:
        return Path(explicit_events)
    if session_dir:
        for name in ("events.jsonl", "events.json", "events.csv"):
            candidate = session_dir / "events" / name
            if candidate.exists():
                return candidate
        return session_dir / "events" / "events.jsonl"
    return EVENTS_PATH


def _load_raw_shards(session_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    shard_paths = resolve_raw_shard_paths(session_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No raw shards found in {session_dir / 'raw'}")
    records = [np.load(path) for path in shard_paths]
    raw = np.concatenate(records) if len(records) > 1 else records[0]
    if raw.size < 1 or "lsl_ts_mono" not in raw.dtype.names:
        raise ValueError("Invalid raw shard format (missing lsl_ts_mono)")
    time_s = raw["lsl_ts_mono"].astype(float)
    time_s = time_s - float(time_s[0])
    signal = raw["sample"].astype(float)
    return time_s, signal


def _channel_labels_from_session_meta(session_dir: Optional[Path], count: int) -> list[str]:
    if session_dir:
        meta_path = session_dir / "meta.json"
        if meta_path.exists():
            try:
                payload = json.loads(meta_path.read_text())
                labels = payload.get("channel_labels")
                if isinstance(labels, list) and len(labels) >= count:
                    return [str(v) for v in labels[:count]]
            except Exception:
                pass
    fallback = ["TP9", "AF7", "AF8", "TP10"]
    if count <= len(fallback):
        return fallback[:count]
    return [f"ch{i + 1}" for i in range(count)]


def _load_features_csv(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    if "time_s" in df.columns:
        times = df["time_s"].to_numpy(dtype=float)
    else:
        times = np.arange(len(df), dtype=float) / DEFAULT_FS
    channel_cols = [c for c in ["ch1", "ch2", "ch3", "ch4"] if c in df.columns]
    if not channel_cols:
        channel_cols = [c for c in ["TP9", "AF7", "AF8", "TP10"] if c in df.columns]
    if not channel_cols:
        raise ValueError(f"No channel columns found in {path}")
    signal = df[channel_cols].values
    return times, signal, [str(c) for c in channel_cols]


def _resolve_plot_fixed_ylim(value: Optional[list[float]]) -> tuple[float, float]:
    if not value or len(value) != 2:
        return float(PLOT_FIXED_YLIM[0]), float(PLOT_FIXED_YLIM[1])
    low = float(value[0])
    high = float(value[1])
    if low == high:
        if low == 0:
            return (-200.0, 200.0)
        return (low - abs(low), low + abs(low))
    return (min(low, high), max(low, high))


def _normalize_scale_mode(value: str) -> str:
    val = (value or "").strip().lower()
    if val in {"robust", "robust_auto", "auto"}:
        return "robust"
    return "fixed"


def _apply_plot_lines(
    lines: list[Any],
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    offsets: np.ndarray,
    plot_channels: int,
) -> None:
    for idx in range(plot_channels):
        lines[idx].set_data(t_arr, y_arr[:, idx] + offsets[idx])
    for idx in range(plot_channels, len(lines)):
        lines[idx].set_data([], [])


def _load_signal(
    session_dir: Optional[Path], features_override: Optional[str]
) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    if features_override:
        times, signal, labels = _load_features_csv(Path(features_override))
        return times, signal, "legacy_features_csv", labels
    if session_dir:
        try:
            times, signal = _load_raw_shards(session_dir)
            labels = _channel_labels_from_session_meta(session_dir, signal.shape[1])
            return times, signal, "raw_shards", labels
        except Exception:
            pass
        raw_dir = session_dir / "raw"
        raw_csv = raw_dir / "raw.csv"
        if raw_csv.exists():
            times, signal, labels = _load_features_csv(raw_csv)
            return times, signal, "legacy_raw_csv", labels
        features_csv = session_dir / "features" / "eeg_features.csv"
        if features_csv.exists():
            times, signal, labels = _load_features_csv(features_csv)
            return times, signal, "legacy_features_csv", labels
    raise FileNotFoundError("No raw shards or legacy features CSV found.")


parser = argparse.ArgumentParser(
    description=(
        "Step 5: open an interactive event-review UI for a recorded session "
        "and save edited event annotations."
    ),
    epilog=(
        "Keyboard shortcuts:\n"
        "  Left/Right  move cursor by 0.1 s\n"
        "  Up/Down     move cursor by 1.0 s\n"
        "  n / p       next or previous event\n"
        "  e / d       edit or delete selected event\n"
        "  s / q       save or quit"
    ),
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
input_group = parser.add_argument_group("input selection")
input_group.add_argument(
    "--session-dir",
    type=str,
    default=None,
    metavar="PATH",
    help="Canonical session directory. Defaults to events/events.jsonl plus raw/eeg_raw_shard_*.npy.",
)
input_group.add_argument(
    "--subject-id",
    type=str,
    default="8-M16",
    metavar="ID",
    help="Deprecated. Subject lookup is not supported without explicit paths.",
)
input_group.add_argument(
    "--events",
    type=str,
    default=None,
    metavar="PATH",
    help="Override the event file path (events.jsonl, events.json, or legacy events.csv).",
)
input_group.add_argument(
    "--features",
    type=str,
    default=None,
    metavar="PATH",
    help="Optional raw/features path override used for plotting alignment.",
)
plot_group = parser.add_argument_group("plot options")
plot_group.add_argument(
    "--plot-scale",
    type=str,
    default="fixed",
    choices=["fixed", "robust"],
    help="Plot scaling mode matching Step 1 semantics.",
)
plot_group.add_argument(
    "--plot-fixed-ylim",
    type=float,
    nargs=2,
    metavar=("MIN_UV", "MAX_UV"),
    default=list(PLOT_FIXED_YLIM),
    help="Y-axis limits in microvolts when --plot-scale=fixed.",
)
plot_group.add_argument(
    "--plot-reference-overlay",
    action="store_true",
    help="Overlay per-channel reference guides like Step 1.",
)
plot_group.add_argument(
    "--plot-channel-spacing-uv",
    type=float,
    default=PLOT_CHANNEL_SPACING_UV,
    help="Vertical spacing between plotted EEG channels.",
)
plot_group.add_argument(
    "--window-sec",
    type=float,
    default=DEFAULT_WINDOW_SEC,
    metavar="SECONDS",
    help="Initial visible window width in seconds.",
)
args = parser.parse_args()
subject_id_provided = "--subject-id" in sys.argv

explicit_events = bool(args.events)
explicit_features = bool(args.features)
selection_source = "legacy_explicit"
session_dir = Path(args.session_dir).expanduser().resolve() if args.session_dir else None

# Primary path: review/edit events emitted by Step 1 in a session directory.
if session_dir:
    if not session_dir.exists():
        print("Session selection source: session_dir")
        print(f"Session dir not found: {session_dir}")
        raise SystemExit(2)
    if explicit_events or explicit_features:
        print(
            "⚠️ Explicit --events/--features provided with --session-dir; using explicit paths."
        )
        selection_source = "legacy_explicit"
    else:
        selection_source = "session_dir"
else:
    if subject_id_provided:
        print("Session selection source: legacy_explicit")
        print(
            "❌ subject-id lookup is not supported without --session-dir. Provide explicit --events/--features."
        )
        raise SystemExit(2)
    if not explicit_features:
        print("Session selection source: legacy_explicit")
        print(
            "❌ Missing --session-dir. Provide --session-dir or explicit --features."
        )
        raise SystemExit(2)

EVENTS_PATH = _resolve_events_path(session_dir, args.events)

print(f"Session selection source: {selection_source}")
print(f"Using events file: {EVENTS_PATH}")

if not EVENTS_PATH.exists():
    print(f"events file not found: {EVENTS_PATH}")
    raise SystemExit(2)

try:
    times, signal, signal_source, channel_labels = _load_signal(session_dir, args.features)
except FileNotFoundError as exc:
    print(str(exc))
    raise SystemExit(2)

print(f"Using signal source: {signal_source}")
print(f"Saving edits to: {EVENTS_PATH}")

events_df = load_events_dataframe(EVENTS_PATH)

# If timestamps are flat/invalid, fall back to sample-derived time.
if np.nanmax(times) - np.nanmin(times) < 1e-6:
    times = np.arange(len(signal), dtype=float) / DEFAULT_FS

events = events_df.to_dict(orient="records")

# Align signal timeline to event timeline if ranges differ significantly.
if events:
    events_start = min(e["onset_s"] for e in events)
    events_end = max(e["onset_s"] + e["duration_s"] for e in events)
else:
    events_start = None
    events_end = None

times_start = float(times[0])
times_end = float(times[-1])
times_range = times_end - times_start
if events and times_range > 0:
    events_range = events_end - events_start
    # Rescale if events extend beyond the signal or are much longer.
    if (
        events_end > times_end
        or events_start < times_start
        or events_range > times_range * 1.25
    ):
        scale = events_range / times_range
        times = (times - times_start) * scale + events_start
        times_start = float(times[0])
        times_end = float(times[-1])

cursor_t = times[0]
selected_idx = 0 if events else None

# Align the plotted signal timeline to the event timeline if they don't overlap.
if events:
    events_start = min(e["onset_s"] for e in events)
    events_end = max(e["onset_s"] + e["duration_s"] for e in events)
    times_start = float(times[0])
    times_end = float(times[-1])
    if events_start > times_end or events_end < times_start:
        times = times - times_start + events_start

time_min = float(times[0])
time_max = float(times[-1])
time_span = max(0.0, float(time_max - time_min))
window_sec = float(args.window_sec) if np.isfinite(args.window_sec) else DEFAULT_WINDOW_SEC
window_sec = float(max(MIN_WINDOW_SEC, window_sec))
if time_span > 0:
    window_sec = float(min(window_sec, min(MAX_WINDOW_SEC, max(MIN_WINDOW_SEC, time_span))))
else:
    window_sec = float(min(window_sec, MAX_WINDOW_SEC))

fig = plt.figure(figsize=(12, 6))
gs = fig.add_gridspec(3, 1, height_ratios=[8.0, 1.35, 0.6], hspace=0.18)
ax = fig.add_subplot(gs[0])
timeline_ax = fig.add_subplot(gs[1], sharex=None)
slider_ax = fig.add_subplot(gs[2])
slider_ax.set_facecolor("#f2f2f2")
plot_scale = _normalize_scale_mode(args.plot_scale)
plot_fixed_ylim = _resolve_plot_fixed_ylim(list(args.plot_fixed_ylim))
channel_count = int(signal.shape[1]) if signal.ndim == 2 else 1
plot_channels = min(channel_count, len(channel_labels))
spacing_uv = float(args.plot_channel_spacing_uv)
if (not np.isfinite(spacing_uv)) or spacing_uv <= 0.0:
    stds = np.nanstd(signal[:, :plot_channels], axis=0)
    median_std = float(np.nanmedian(stds)) if stds.size else 0.0
    spacing_uv = float(max(80.0, min(400.0, max(120.0, 6.0 * median_std))))
plot_offsets = np.arange(channel_count, dtype=float) * spacing_uv
lines: list[Any] = []
for _ in range(channel_count):
    line, = ax.plot([], [], lw=1.0, color="black")
    lines.append(line)
overlay_lines = []
if args.plot_reference_overlay:
    for off in plot_offsets[:plot_channels]:
        overlay_lines.append(
            ax.axhline(float(off), color="#888888", alpha=0.2, linewidth=0.6)
        )
try:
    ax.set_yticks(plot_offsets[:plot_channels].tolist())
    ax.set_yticklabels([str(v) for v in channel_labels[:plot_channels]])
except Exception:
    pass
base_half = max(abs(plot_fixed_ylim[0]), abs(plot_fixed_ylim[1]))
if plot_scale == "robust":
    lows = np.nanpercentile(signal[:, :plot_channels], 5, axis=0)
    highs = np.nanpercentile(signal[:, :plot_channels], 95, axis=0)
    low_off = lows + plot_offsets[:plot_channels]
    high_off = highs + plot_offsets[:plot_channels]
    ax.set_ylim(float(np.min(low_off)), float(np.max(high_off)))
else:
    ax.set_ylim(float(plot_offsets[0] - base_half), float(plot_offsets[-1] + base_half))
cursor_line = ax.axvline(cursor_t, color="red", linestyle="--")
span_patches: list[Any] = []
timeline_span_patches: list[Any] = []
timeline_cursor_line = timeline_ax.axvline(cursor_t, color="red", linestyle="--", linewidth=1.0)
timeline_selected_line = timeline_ax.axvline(
    cursor_t, color="#ff9800", linestyle="-", linewidth=0.8, alpha=0.0
)
timeline_view_patch = None

overview_stride = max(1, int(np.ceil(len(times) / float(MAX_OVERVIEW_POINTS))))
overview_times = times[::overview_stride]
if plot_channels > 0:
    overview_signal = np.nanmean(np.abs(signal[:, :plot_channels]), axis=1)
else:
    overview_signal = np.nanmean(np.abs(np.atleast_2d(signal)), axis=1)
overview_values = overview_signal[::overview_stride]
if overview_values.size == 0:
    overview_values = np.zeros(1, dtype=float)
    overview_times = np.array([time_min], dtype=float)
timeline_ax.plot(overview_times, overview_values, color="#606060", linewidth=0.7, alpha=0.9)
timeline_ax.set_xlim(time_min, time_max if time_max > time_min else time_min + 1.0)
timeline_ax.set_yticks([])
timeline_ax.set_ylabel("Nav")
timeline_ax.set_xlabel("Timeline (s)")

window_slider_max = MAX_WINDOW_SEC
window_slider = Slider(
    ax=slider_ax,
    label="Window (s)",
    valmin=MIN_WINDOW_SEC,
    valmax=window_slider_max,
    valinit=window_sec,
)
interaction_state = {
    "mode": None,
    "anchor_x": None,
    "cursor_start": cursor_t,
    "suspend_slider": False,
}


def _clamp_window_sec(value: float) -> float:
    max_window = min(
        MAX_WINDOW_SEC,
        max(MIN_WINDOW_SEC, time_span) if time_span > 0 else DEFAULT_WINDOW_SEC,
    )
    return float(min(max_window, max(MIN_WINDOW_SEC, float(value))))


def _view_bounds(center_t: float) -> tuple[float, float]:
    if time_span <= 0.0:
        return time_min, max(time_min + window_sec, time_min)
    width = _clamp_window_sec(window_sec)
    half = width / 2.0
    start = max(time_min, float(center_t) - half)
    end = start + width
    if end > time_max:
        end = time_max
        start = max(time_min, end - width)
    return float(start), float(end)


def _nearest_sample_time(value: float) -> float:
    if len(times) == 0:
        return float(value)
    idx = int(np.searchsorted(times, value, side="left"))
    idx = max(0, min(idx, len(times) - 1))
    return float(times[idx])


def _visible_slice(start_t: float, end_t: float) -> slice:
    if len(times) == 0:
        return slice(0, 0)
    left = max(0, int(np.searchsorted(times, start_t, side="left")) - 1)
    right = min(len(times), int(np.searchsorted(times, end_t, side="right")) + 1)
    if right <= left:
        right = min(len(times), left + 1)
    return slice(left, right)


def redraw_spans():
    global span_patches, timeline_span_patches, timeline_view_patch
    for patch in span_patches:
        patch.remove()
    span_patches = []
    view_start, view_end = _view_bounds(cursor_t)
    for idx, e in enumerate(events):
        start = e["onset_s"]
        end = start + e["duration_s"]
        if end < view_start or start > view_end:
            continue
        color = "orange" if idx == selected_idx else "blue"
        patch = ax.axvspan(start, end, alpha=0.15, color=color)
        span_patches.append(patch)
    for patch in timeline_span_patches:
        patch.remove()
    timeline_span_patches = []
    for idx, e in enumerate(events):
        start = e["onset_s"]
        end = start + e["duration_s"]
        color = "#ffb74d" if idx == selected_idx else "#7f7fff"
        alpha = 0.35 if idx == selected_idx else 0.18
        timeline_span_patches.append(
            timeline_ax.axvspan(start, end, alpha=alpha, color=color)
        )
    if timeline_view_patch is not None:
        timeline_view_patch.remove()
    timeline_view_patch = timeline_ax.axvspan(
        view_start, view_end, alpha=0.2, color="#ffcc80"
    )
    timeline_cursor_line.set_xdata([cursor_t, cursor_t])
    if selected_idx is not None and events:
        selected_time = float(events[selected_idx]["onset_s"])
        timeline_selected_line.set_xdata([selected_time, selected_time])
        timeline_selected_line.set_alpha(0.9)
    else:
        timeline_selected_line.set_alpha(0.0)
    fig.canvas.draw_idle()


def update_cursor(new_t):
    global cursor_t
    cursor_t = max(time_min, min(time_max, float(new_t)))
    view_start, view_end = _view_bounds(cursor_t)
    visible = _visible_slice(view_start, view_end)
    t_view = times[visible]
    y_view = signal[visible]
    _apply_plot_lines(lines, t_view, y_view, plot_offsets, plot_channels)
    ax.set_xlim(view_start, view_end if view_end > view_start else view_start + 1.0)
    cursor_line.set_xdata([cursor_t, cursor_t])
    if plot_scale == "robust" and len(t_view) > 0:
        visible_signal = y_view[:, :plot_channels]
        lows = np.nanpercentile(visible_signal, 5, axis=0)
        highs = np.nanpercentile(visible_signal, 95, axis=0)
        low_off = lows + plot_offsets[:plot_channels]
        high_off = highs + plot_offsets[:plot_channels]
        ax.set_ylim(float(np.min(low_off)), float(np.max(high_off)))
    elif plot_scale != "robust":
        ax.set_ylim(float(plot_offsets[0] - base_half), float(plot_offsets[-1] + base_half))
    title_suffix = f"window={window_sec:.1f}s cursor={cursor_t:.2f}s"
    ax.set_title(f"EEG (uV) [{title_suffix}]")
    redraw_spans()
    fig.canvas.draw_idle()


def set_window_sec(new_window_sec: float):
    global window_sec
    window_sec = _clamp_window_sec(new_window_sec)
    if not interaction_state["suspend_slider"]:
        interaction_state["suspend_slider"] = True
        try:
            window_slider.set_val(window_sec)
        finally:
            interaction_state["suspend_slider"] = False
    update_cursor(cursor_t)


def select_event(idx):
    global selected_idx
    if not events:
        selected_idx = None
        return
    selected_idx = max(0, min(idx, len(events) - 1))
    update_cursor(events[selected_idx]["onset_s"])
    redraw_spans()


def save_events():
    rows = [dict(event) for event in events]
    payload_records = event_rows_to_payloads(rows)
    flat_records = event_rows_to_flat_records(rows)
    if session_dir:
        events_jsonl_path = session_dir / "events" / "events.jsonl"
    else:
        events_jsonl_path = EVENTS_PATH.with_suffix(".jsonl")
    events_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with events_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in payload_records:
            handle.write(json.dumps(record) + "\n")

    if EVENTS_PATH.suffix.lower() == ".json":
        events_json_path = EVENTS_PATH
        events_json_path.parent.mkdir(parents=True, exist_ok=True)
        events_json_path.write_text(json.dumps(payload_records, indent=2))
    else:
        events_json_path = None

    legacy_csv_path = None
    if EVENTS_PATH.suffix.lower() == ".csv":
        legacy_csv_path = EVENTS_PATH
    elif session_dir:
        candidate = session_dir / "events" / "events.csv"
        if candidate.exists():
            legacy_csv_path = candidate
    if legacy_csv_path is not None:
        pd.DataFrame(flat_records).to_csv(legacy_csv_path, index=False)

    print(f"✅ Saved {len(payload_records)} events to {events_jsonl_path}")
    if events_json_path is not None:
        print(f"↪︎ Updated legacy JSON: {events_json_path}")
    if legacy_csv_path is not None:
        print(f"↪︎ Updated legacy CSV: {legacy_csv_path}")


def normalize_event(event):
    # Keep type/action/finger internally consistent with label_schema so Step 1b
    # labeling and downstream training do not see contradictory labels.
    override = event.get("type")
    if override in {"artifact", "calibration", "rest"}:
        event["action_id"] = ACTION_REST
        event["finger_id"] = FINGER_NONE
    else:
        event["type"] = event_type_for(
            int(event["action_id"]), int(event["finger_id"]), None
        )
    return event


def edit_event():
    if selected_idx is None:
        print("No events to edit")
        return
    event = events[selected_idx]
    before = dict(event)
    try:
        onset = input(f"onset_s [{event['onset_s']}]: ").strip()
        duration = input(f"duration_s [{event['duration_s']}]: ").strip()
        action_id = input(f"action_id [{event['action_id']}]: ").strip()
        finger_id = input(f"finger_id [{event['finger_id']}]: ").strip()
        notes = input(f"notes [{event['notes']}]: ").strip()
        override_type = input(f"type [{event['type']}]: ").strip()

        if onset:
            event["onset_s"] = float(onset)
        if duration:
            event["duration_s"] = float(duration)
        if action_id:
            event["action_id"] = int(action_id)
        if finger_id:
            event["finger_id"] = int(finger_id)
        if notes:
            event["notes"] = notes
        if override_type:
            event["type"] = override_type

        event = normalize_event(event)
        events[selected_idx] = event
        log_event_edit("edit", before, dict(event))
        redraw_spans()
    except Exception as e:
        print(f"⚠️ Edit failed: {e}")


def delete_event():
    global selected_idx
    if selected_idx is None:
        return
    before = events[selected_idx]
    events.pop(selected_idx)
    log_event_edit("delete", before, {})
    if events:
        selected_idx = min(selected_idx, len(events) - 1)
    else:
        selected_idx = None
    redraw_spans()


def on_key(event):
    global selected_idx
    if event.key == "left":
        update_cursor(cursor_t - 0.1)
    elif event.key == "right":
        update_cursor(cursor_t + 0.1)
    elif event.key == "up":
        update_cursor(cursor_t + 1.0)
    elif event.key == "down":
        update_cursor(cursor_t - 1.0)
    elif event.key == "n":
        if selected_idx is not None:
            select_event(selected_idx + 1)
    elif event.key == "p":
        if selected_idx is not None:
            select_event(selected_idx - 1)
    elif event.key == "[":
        set_window_sec(window_sec * 0.8)
    elif event.key == "]":
        set_window_sec(window_sec * 1.25)
    elif event.key == "e":
        edit_event()
    elif event.key == "d":
        delete_event()
    elif event.key == "s":
        save_events()
    elif event.key == "q":
        save_events()
        plt.close(fig)


def on_scroll(event):
    if event.inaxes not in {ax, timeline_ax}:
        return
    factor = 0.8 if event.button == "up" else 1.25
    if event.xdata is not None and np.isfinite(event.xdata):
        update_cursor(event.xdata)
    set_window_sec(window_sec * factor)


def on_button_press(event):
    if event.button != 1:
        return
    if event.inaxes == timeline_ax and event.xdata is not None:
        interaction_state["mode"] = "timeline"
        update_cursor(event.xdata)
    elif event.inaxes == ax and event.xdata is not None:
        interaction_state["mode"] = "main"
        interaction_state["anchor_x"] = float(event.xdata)
        interaction_state["cursor_start"] = float(cursor_t)


def on_motion(event):
    mode = interaction_state.get("mode")
    if mode == "timeline":
        if event.xdata is not None and np.isfinite(event.xdata):
            update_cursor(event.xdata)
    elif mode == "main":
        if event.xdata is None or not np.isfinite(event.xdata):
            return
        anchor_x = interaction_state.get("anchor_x")
        cursor_start = interaction_state.get("cursor_start", cursor_t)
        if anchor_x is None:
            return
        delta = float(event.xdata - anchor_x)
        update_cursor(cursor_start - delta)


def on_button_release(_event):
    interaction_state["mode"] = None
    interaction_state["anchor_x"] = None


def on_slider_change(value):
    if interaction_state["suspend_slider"]:
        return
    set_window_sec(float(value))


fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("button_press_event", on_button_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_button_release)
window_slider.on_changed(on_slider_change)

if events:
    select_event(0)
else:
    update_cursor(_nearest_sample_time(cursor_t))

fig.suptitle("EEG Event Review")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (uV)")

print(
    "Event review started. Focus the plot window for keyboard controls. "
    "Drag the main plot to pan, drag/click the timeline to jump, and use the wheel or [/] to resize the window."
)
plt.show()
