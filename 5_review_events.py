"""
STEP 5 — Interactive Event Review & Edit

Keyboard controls:
  Left/Right: move cursor by 0.1s
  Up/Down: move cursor by 1.0s
  n/p: next/previous event
  e: edit selected event (terminal prompts)
  d: delete selected event
  s: save events.csv
  q: quit
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.events_audit import log_event_edit
from utils.label_schema import (
    ACTION_REST,
    FINGER_NONE,
    event_type_for,
)

EVENTS_PATH = Path("events.csv")
FEATURES_PATH = Path("eeg_features.csv")
DEFAULT_FS = 256.0


def latest_subject_file(subject_id, suffix, base_dir):
    base = Path(base_dir)
    pattern = f"{subject_id}_*_{suffix}"
    candidates = sorted(base.glob(pattern), key=lambda p: p.name)
    return candidates[-1] if candidates else None


parser = argparse.ArgumentParser()
parser.add_argument("--subject-id", type=str, default="1-M17", help="Subject ID to select latest session files")
parser.add_argument("--events", type=str, default=None, help="Override events path")
parser.add_argument("--features", type=str, default=None, help="Override features path")
args = parser.parse_args()

meta_path = Path("session_meta.json")
events_candidate = None
features_candidate = None

if args.subject_id:
    events_candidate = latest_subject_file(args.subject_id, "events.csv", "data/processed")
    features_candidate = latest_subject_file(args.subject_id, "eeg_features.csv", "data/processed")
    if events_candidate is None or features_candidate is None:
        print(f"No session files found for subject_id={args.subject_id} in data/processed.")
        raise SystemExit(2)

if args.events:
    events_candidate = Path(args.events)
if args.features:
    features_candidate = Path(args.features)

if events_candidate is None or features_candidate is None:
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if events_candidate is None:
            candidate = Path(meta.get("events_path", str(EVENTS_PATH)))
            if candidate.exists():
                events_candidate = candidate
        if features_candidate is None:
            candidate = Path(meta.get("features_path", str(FEATURES_PATH)))
            if candidate.exists():
                features_candidate = candidate

if events_candidate is not None:
    EVENTS_PATH = events_candidate
if features_candidate is not None:
    FEATURES_PATH = features_candidate

if not EVENTS_PATH.exists():
    raise FileNotFoundError(f"events.csv not found: {EVENTS_PATH}")
if not FEATURES_PATH.exists():
    raise FileNotFoundError(f"eeg_features.csv not found: {FEATURES_PATH}")

events_df = pd.read_csv(EVENTS_PATH)

required_cols = [
    "onset_s", "duration_s", "type", "channel",
    "confidence", "notes", "finger_id", "action_id", "source"
]
for col in required_cols:
    if col not in events_df.columns:
        raise ValueError(f"Missing column: {col}")

features = pd.read_csv(FEATURES_PATH)
if "time_s" in features.columns:
    times = features["time_s"].to_numpy(dtype=float)
else:
    times = np.arange(len(features), dtype=float) / DEFAULT_FS

# If timestamps are flat/invalid, fall back to sample-derived time.
if np.nanmax(times) - np.nanmin(times) < 1e-6:
    times = np.arange(len(features), dtype=float) / DEFAULT_FS

signal = features[["ch1", "ch2", "ch3", "ch4"]].values
signal_mean = signal.mean(axis=1)

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
    if events_end > times_end or events_start < times_start or events_range > times_range * 1.25:
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

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, signal_mean, linewidth=0.5, color="black")
cursor_line = ax.axvline(cursor_t, color="red", linestyle="--")
span_patches = []


def redraw_spans():
    global span_patches
    for patch in span_patches:
        patch.remove()
    span_patches = []
    for idx, e in enumerate(events):
        start = e["onset_s"]
        end = start + e["duration_s"]
        color = "orange" if idx == selected_idx else "blue"
        patch = ax.axvspan(start, end, alpha=0.15, color=color)
        span_patches.append(patch)
    fig.canvas.draw_idle()


def update_cursor(new_t):
    global cursor_t
    cursor_t = max(times[0], min(times[-1], new_t))
    cursor_line.set_xdata([cursor_t, cursor_t])
    fig.canvas.draw_idle()


def select_event(idx):
    global selected_idx
    if not events:
        selected_idx = None
        return
    selected_idx = max(0, min(idx, len(events) - 1))
    update_cursor(events[selected_idx]["onset_s"])
    redraw_spans()


def save_events():
    pd.DataFrame(events).to_csv(EVENTS_PATH, index=False)
    print(f"✅ Saved {len(events)} events to {EVENTS_PATH}")


def normalize_event(event):
    override = event.get("type")
    if override in {"artifact", "calibration", "rest"}:
        event["action_id"] = ACTION_REST
        event["finger_id"] = FINGER_NONE
    else:
        event["type"] = event_type_for(
            int(event["action_id"]),
            int(event["finger_id"]),
            None
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
    elif event.key == "e":
        edit_event()
    elif event.key == "d":
        delete_event()
    elif event.key == "s":
        save_events()
    elif event.key == "q":
        save_events()
        plt.close(fig)


fig.canvas.mpl_connect("key_press_event", on_key)

if events:
    select_event(0)
else:
    redraw_spans()

ax.set_title("EEG Event Review (mean channel)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Mean amplitude")
if events:
    ax.set_xlim(min(times[0], events_start), max(times[-1], events_end))

print("Event review started. Focus the plot window for keyboard controls.")
plt.show()
