"""
STEP 1b — Window Extraction (BCI-standard)

Converts continuous EEG + event markers → windowed feature dataset
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from utils.label_schema import (
    ACTION_REST,
    FINGER_NONE,
    is_valid_action_finger,
)

FS = 256
WINDOW_SEC = 0.25
STEP_SEC = 0.05
WINDOW_SAMPLES = int(FS * WINDOW_SEC)
STEP_SAMPLES = max(1, int(FS * STEP_SEC))

RAW_FILE = "eeg_features.csv"
EVENT_FILE = "events.csv"
OUT_FILE = "eeg_windows.csv"
OUT_NPZ = "eeg_windows.npz"

session_meta = {}
meta_path = Path("session_meta.json")
if meta_path.exists():
    session_meta = json.loads(meta_path.read_text())
    FS = int(session_meta.get("sampling_rate", FS))
    WINDOW_SEC = float(session_meta.get("window_sec", WINDOW_SEC))
    WINDOW_SAMPLES = int(FS * WINDOW_SEC)
    STEP_SAMPLES = max(1, int(FS * STEP_SEC))

    feature_path = Path(session_meta.get("features_path", RAW_FILE))
    if feature_path.exists():
        RAW_FILE = str(feature_path)

    events_path = Path(session_meta.get("events_path", EVENT_FILE))
    if events_path.exists():
        EVENT_FILE = str(events_path)

df = pd.read_csv(RAW_FILE)
events_df = pd.read_csv(EVENT_FILE)

if "time_s" in df.columns:
    times = df["time_s"].values
else:
    times = np.arange(len(df)) / FS

signal = df[["ch1", "ch2", "ch3", "ch4"]].values

events = []
for _, row in events_df.iterrows():
    events.append({
        "onset_s": float(row["onset_s"]),
        "duration_s": float(row["duration_s"]),
        "type": str(row.get("type", "")),
        "finger_id": int(row.get("finger_id", 0)),
        "action_id": int(row.get("action_id", 0)),
        "confidence": row.get("confidence", np.nan),
    })

windows = []
sequence_windows = []
action_labels = []
finger_labels = []
subject_ids = []
experiment_hashes = []
window_starts = []
window_ends = []
confidence_hints = []
artifact_flags = []
max_idx = len(signal) - WINDOW_SAMPLES

def overlap(a_start, a_end, b_start, b_end):
    if b_end == b_start:
        return 1.0 if a_start <= b_start <= a_end else 0.0
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

for start_idx in range(0, max_idx + 1, STEP_SAMPLES):
    end_idx = start_idx + WINDOW_SAMPLES
    window_start = times[start_idx]
    window_end = times[end_idx - 1]

    overlapping = []
    for e in events:
        e_start = e["onset_s"]
        e_end = e_start + e["duration_s"]
        ov = overlap(window_start, window_end, e_start, e_end)
        if ov > 0:
            overlapping.append((ov, e))

    artifact_flag = 0
    action_id = ACTION_REST
    finger_id = FINGER_NONE
    confidence_hint = np.nan

    if overlapping:
        overlapping.sort(key=lambda x: x[0], reverse=True)
        _, best = overlapping[0]
        if best["type"] == "artifact":
            artifact_flag = 1
        else:
            action_id = int(best["action_id"])
            finger_id = int(best["finger_id"])
            confidence_hint = best.get("confidence", np.nan)

    if artifact_flag:
        continue

    if not is_valid_action_finger(action_id, finger_id):
        continue

    segment = signal[start_idx:end_idx]
    if len(segment) != WINDOW_SAMPLES:
        continue

    features = segment.mean(axis=0)

    sequence_windows.append(segment.astype(np.float32))
    action_labels.append(action_id)
    finger_labels.append(finger_id)
    subject_ids.append(session_meta.get("subject_id", "UNKNOWN"))
    experiment_hashes.append(session_meta.get("experiment_hash", "UNKNOWN"))
    window_starts.append(float(window_start))
    window_ends.append(float(window_end))
    confidence_hints.append(confidence_hint)
    artifact_flags.append(artifact_flag)

    windows.append({
        "ch1": features[0],
        "ch2": features[1],
        "ch3": features[2],
        "ch4": features[3],
        "action_id": action_id,
        "finger_id": finger_id,
        "subject_id": session_meta.get("subject_id", "UNKNOWN"),
        "experiment_hash": session_meta.get("experiment_hash", "UNKNOWN"),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "confidence_hint": confidence_hint,
        "artifact_flag": artifact_flag,
    })

pd.DataFrame(windows).to_csv(OUT_FILE, index=False)
print(f"✅ Saved {len(windows)} windows → {OUT_FILE}")

if sequence_windows:
    X = np.stack(sequence_windows).astype(np.float32)
    np.savez_compressed(
        OUT_NPZ,
        X=X,
        y_action=np.array(action_labels, dtype=np.int64),
        y_finger=np.array(finger_labels, dtype=np.int64),
        subject_id=np.array(subject_ids, dtype="U"),
        experiment_hash=np.array(experiment_hashes, dtype="U"),
        window_start=np.array(window_starts, dtype=np.float32),
        window_end=np.array(window_ends, dtype=np.float32),
        confidence_hint=np.array(confidence_hints, dtype=np.float32),
        artifact_flag=np.array(artifact_flags, dtype=np.int64),
        fs=np.array(FS, dtype=np.int64),
        window_sec=np.array(WINDOW_SEC, dtype=np.float32),
        step_sec=np.array(STEP_SEC, dtype=np.float32),
        channel_names=np.array(["ch1", "ch2", "ch3", "ch4"], dtype="U"),
    )
    print(f"✅ Saved sequence windows → {OUT_NPZ} with shape {X.shape}")
