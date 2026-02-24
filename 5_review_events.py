"""
STEP 5 — Interactive Event Review & Edit

Keyboard controls:
  Left/Right: move cursor by 0.1s
  Up/Down: move cursor by 1.0s
  n/p: next/previous event
  e: edit selected event (terminal prompts)
  d: delete selected event
  s: save events.jsonl
  q: quit
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.events_audit import log_event_edit
from utils.label_schema import (
    ACTION_REST,
    FINGER_NONE,
    event_type_for,
)

EVENTS_PATH = Path("events.jsonl")
DEFAULT_FS = 256.0


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


def _load_events_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix == ".json":
        try:
            payload = json.loads(path.read_text())
        except Exception:
            return []
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            payload = payload.get("events")
        if not isinstance(payload, list):
            return []
        return [dict(ev) for ev in payload if isinstance(ev, dict)]
    if suffix == ".jsonl":
        events = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events
    if suffix == ".csv":
        df = pd.read_csv(path)
        return df.to_dict(orient="records")
    return []


def _normalize_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    if "onset_s" not in events_df.columns and "event_time_s" in events_df.columns:
        events_df["onset_s"] = events_df["event_time_s"]
    if "type" not in events_df.columns and "label" in events_df.columns:
        events_df["type"] = events_df["label"]
    defaults = {
        "duration_s": 0.0,
        "type": "",
        "channel": "n/a",
        "confidence": "",
        "notes": "",
        "finger_id": 0,
        "action_id": 0,
        "trial_id": 0,
        "block_id": 0,
        "source": "unknown",
    }
    for col, value in defaults.items():
        if col not in events_df.columns:
            events_df[col] = value
    for col in ("onset_s", "duration_s", "finger_id", "action_id"):
        events_df[col] = pd.to_numeric(events_df[col], errors="coerce")
        if col in ("finger_id", "action_id"):
            events_df[col] = events_df[col].fillna(0).astype(int)
        else:
            events_df[col] = events_df[col].fillna(0.0).astype(float)
    return events_df


def _sorted_raw_shards(raw_dir: Path) -> list[Path]:
    shard_paths = list(raw_dir.glob("eeg_raw_shard_*.npy"))
    if not shard_paths:
        return []
    def _key(path: Path) -> tuple[int, str]:
        match = re.search(r"eeg_raw_shard_(\\d+)\\.npy$", path.name)
        if match:
            return int(match.group(1)), path.name
        return (10**12, path.name)
    return sorted(shard_paths, key=_key)


def _load_raw_shards(raw_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    shard_paths = _sorted_raw_shards(raw_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No raw shards found in {raw_dir}")
    records = [np.load(path) for path in shard_paths]
    raw = np.concatenate(records) if len(records) > 1 else records[0]
    if raw.size < 1 or "lsl_ts_mono" not in raw.dtype.names:
        raise ValueError("Invalid raw shard format (missing lsl_ts_mono)")
    time_s = raw["lsl_ts_mono"].astype(float)
    time_s = time_s - float(time_s[0])
    signal = raw["sample"].astype(float)
    return time_s, signal


def _load_features_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    return times, signal


def _load_signal(
    session_dir: Optional[Path], features_override: Optional[str]
) -> tuple[np.ndarray, np.ndarray, str]:
    if features_override:
        times, signal = _load_features_csv(Path(features_override))
        return times, signal, "legacy_features_csv"
    if session_dir:
        raw_dir = session_dir / "raw"
        try:
            times, signal = _load_raw_shards(raw_dir)
            return times, signal, "raw_shards"
        except Exception:
            pass
        raw_csv = raw_dir / "raw.csv"
        if raw_csv.exists():
            times, signal = _load_features_csv(raw_csv)
            return times, signal, "legacy_raw_csv"
        features_csv = session_dir / "features" / "eeg_features.csv"
        if features_csv.exists():
            times, signal = _load_features_csv(features_csv)
            return times, signal, "legacy_features_csv"
    raise FileNotFoundError("No raw shards or legacy features CSV found.")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _events_to_records(events_df: pd.DataFrame) -> list[dict]:
    records = []
    for row in events_df.to_dict(orient="records"):
        record = {k: _json_safe(v) for k, v in row.items()}
        onset = float(record.get("onset_s", record.get("event_time_s", 0.0)) or 0.0)
        duration = float(record.get("duration_s", 0.0) or 0.0)
        record["onset_s"] = onset
        record["event_time_s"] = onset
        record["duration_s"] = max(0.0, duration)
        record["end_s"] = onset + max(0.0, duration)
        if "type" not in record and "label" in record:
            record["type"] = record.get("label", "")
        records.append(record)
    return records


parser = argparse.ArgumentParser()
parser.add_argument(
    "--session-dir",
    type=str,
    default=None,
    help="Session directory (defaults to events/events.jsonl + raw/eeg_raw_shard_*.npy).",
)
parser.add_argument(
    "--subject-id",
    type=str,
    default="8-M16",
    help="(Deprecated) Subject ID lookup is no longer supported without explicit paths.",
)
parser.add_argument(
    "--events",
    type=str,
    default=None,
    help="Override events path (events.jsonl / legacy events.csv).",
)
parser.add_argument(
    "--features",
    type=str,
    default=None,
    help="Override legacy features CSV path (optional).",
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
    times, signal, signal_source = _load_signal(session_dir, args.features)
except FileNotFoundError as exc:
    print(str(exc))
    raise SystemExit(2)

print(f"Using signal source: {signal_source}")
print(f"Saving edits to: {EVENTS_PATH}")

events_records = _load_events_records(EVENTS_PATH)
events_df = _normalize_events_df(pd.DataFrame(events_records))

# If timestamps are flat/invalid, fall back to sample-derived time.
if np.nanmax(times) - np.nanmin(times) < 1e-6:
    times = np.arange(len(signal), dtype=float) / DEFAULT_FS

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

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(times, signal_mean, linewidth=0.5, color="black")
cursor_line = ax.axvline(cursor_t, color="red", linestyle="--")
span_patches: list[Any] = []


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
    events_df = pd.DataFrame(events)
    records = _events_to_records(events_df)
    if session_dir:
        events_jsonl_path = session_dir / "events" / "events.jsonl"
    else:
        events_jsonl_path = EVENTS_PATH.with_suffix(".jsonl")
    events_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with events_jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    if EVENTS_PATH.suffix.lower() == ".json":
        events_json_path = EVENTS_PATH
        events_json_path.parent.mkdir(parents=True, exist_ok=True)
        events_json_path.write_text(json.dumps(records, indent=2))
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
        pd.DataFrame(records).to_csv(legacy_csv_path, index=False)

    print(f"✅ Saved {len(records)} events to {events_jsonl_path}")
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
