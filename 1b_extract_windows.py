"""
STEP 1b — Window Extraction (BCI-standard, audited)

Converts continuous EEG + event markers → windowed dataset (tabular + sequence npz)

Key upgrades:
- Minimum overlap gating to reduce boundary label noise
- Guard band around event boundaries to avoid mixed windows
- Artifact priority: if artifact overlaps enough, skip window even if not "best"
- Deterministic tie-breaking + metadata for QA
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

# =========================
# ===== CONFIG ============
# =========================
FS = 256
WINDOW_SEC = 0.25
STEP_SEC = 0.05

# Label assignment robustness
MIN_OVERLAP_RATIO = 0.20   # fraction of WINDOW_SEC required for non-REST labels
GUARD_BAND_SEC = 0.15     # skip windows within ± this time of any movement event boundary
ARTIFACT_MIN_OVERLAP_FRAC = 0.20  # if artifact overlaps >=20% of window, drop window

MIN_OVERLAP_SEC = MIN_OVERLAP_RATIO * WINDOW_SEC
WINDOW_SAMPLES = int(FS * WINDOW_SEC)
STEP_SAMPLES = max(1, int(FS * STEP_SEC))

RAW_FILE = "eeg_features.csv"
EVENT_FILE = "events.csv"
OUT_FILE = "eeg_windows.csv"
OUT_NPZ = "eeg_windows.npz"

# =========================
# ===== SESSION META ======
# =========================
session_meta = {}
meta_path = Path("session_meta.json")
if meta_path.exists():
    session_meta = json.loads(meta_path.read_text())
    FS = int(session_meta.get("sampling_rate", FS))
    WINDOW_SEC = float(session_meta.get("window_sec", WINDOW_SEC))
    WINDOW_SAMPLES = int(FS * WINDOW_SEC)
    MIN_OVERLAP_SEC = MIN_OVERLAP_RATIO * WINDOW_SEC
    STEP_SAMPLES = max(1, int(FS * STEP_SEC))

    feature_path = Path(session_meta.get("features_path", RAW_FILE))
    if feature_path.exists():
        RAW_FILE = str(feature_path)

    events_path = Path(session_meta.get("events_path", EVENT_FILE))
    if events_path.exists():
        EVENT_FILE = str(events_path)

# =========================
# ===== LOAD DATA =========
# =========================
df = pd.read_csv(RAW_FILE)
events_df = pd.read_csv(EVENT_FILE)

if "time_s" in df.columns:
    times = df["time_s"].values.astype(float)
else:
    times = (np.arange(len(df), dtype=float) / float(FS))

# Expect exactly 4 channels for now
signal = df[["ch1", "ch2", "ch3", "ch4"]].values.astype(float)

# =========================
# ===== LOAD EVENTS =======
# =========================
events = []
for _, row in events_df.iterrows():
    onset_s = float(row["onset_s"])
    duration_s = float(row["duration_s"])
    if duration_s < 0:
        duration_s = 0.0

    e = {
        "onset_s": onset_s,
        "duration_s": duration_s,
        "end_s": onset_s + duration_s,
        "type": str(row.get("type", "")).strip(),
        "finger_id": int(row.get("finger_id", 0)),
        "action_id": int(row.get("action_id", 0)),
        "confidence": row.get("confidence", np.nan),
        "source": str(row.get("source", "")).strip() or "unknown",
        "notes": str(row.get("notes", "")).strip() if "notes" in row else "",
        "session_mode": str(row.get("session_mode", "")).strip() if "session_mode" in row else "",
        "trial_id": int(row.get("trial_id", 0)) if "trial_id" in row else 0,
        "block_id": int(row.get("block_id", 0)) if "block_id" in row else 0,
    }
    events.append(e)

# Precompute boundaries for guard band (movement events only)
# We treat any non-rest action as "movement" for guard band purposes.
movement_boundaries = []
for e in events:
    if e["type"] == "artifact":
        continue
    # If the schema uses action_id to represent rest, treat action_id != REST as movement.
    if int(e["action_id"]) != int(ACTION_REST):
        movement_boundaries.append(float(e["onset_s"]))
        movement_boundaries.append(float(e["end_s"]))
movement_boundaries = np.array(sorted(set(movement_boundaries)), dtype=float)

# =========================
# ===== HELPERS ===========
# =========================
def overlap_s(a_start, a_end, b_start, b_end) -> float:
    """Returns overlap duration in seconds between [a_start,a_end] and [b_start,b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))

def in_guard_band(window_start, window_end) -> bool:
    """Skip windows close to movement boundaries (reduces mixed windows)."""
    if movement_boundaries.size == 0:
        return False
    # window midpoint is typically a good reference for boundary proximity
    mid = 0.5 * (window_start + window_end)
    return np.any(np.abs(movement_boundaries - mid) <= GUARD_BAND_SEC)

def event_priority(e: dict) -> int:
    """
    Lower number = higher priority when overlap ties.
    Priority:
      0: artifact
      1: calibration (if present)
      2: movement (action != REST)
      3: rest
      4: unknown/other
    """
    if e["type"] == "artifact":
        return 0
    if e["type"] == "calibration":
        return 1
    if int(e["action_id"]) != int(ACTION_REST):
        return 2
    if int(e["action_id"]) == int(ACTION_REST):
        return 3
    return 4

# =========================
# ===== WINDOW LOOP =======
# =========================
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

# New metadata arrays for QA
assigned_event_types = []
overlap_seconds = []
overlap_fracs = []
event_onsets = []
event_durations = []
event_sources = []
session_modes = []
trial_ids = []
block_ids = []

max_idx = len(signal) - WINDOW_SAMPLES

if max_idx <= 0:
    raise RuntimeError(
        f"Not enough samples in {RAW_FILE}: got {len(signal)} rows, need at least {WINDOW_SAMPLES}."
    )

for start_idx in range(0, max_idx + 1, STEP_SAMPLES):
    end_idx = start_idx + WINDOW_SAMPLES

    window_start = float(times[start_idx])
    window_end = float(times[end_idx - 1])
    window_len_s = max(1e-9, window_end - window_start)

    # Guard band skip (optional but recommended)
    if GUARD_BAND_SEC > 0 and in_guard_band(window_start, window_end):
        continue

    # Compute overlaps with events
    overlapping = []
    any_artifact_flag = 0

    for e in events:
        ov = overlap_s(window_start, window_end, e["onset_s"], e["end_s"])
        if ov <= 0:
            continue

        ov_frac = ov / window_len_s

        # Artifact veto: if artifact overlap is meaningful, drop window
        if e["type"] == "artifact" and ov_frac >= ARTIFACT_MIN_OVERLAP_FRAC:
            any_artifact_flag = 1
            break

        overlapping.append((ov, ov_frac, e))

    if any_artifact_flag:
        continue

    # Default label if no overlap: REST/NONE
    artifact_flag = 0
    action_id = int(ACTION_REST)
    finger_id = int(FINGER_NONE)
    confidence_hint = np.nan

    assigned_type = "rest"
    best_ov = 0.0
    best_ov_frac = 0.0
    best_onset = np.nan
    best_dur = np.nan
    best_source = ""
    best_session_mode = ""
    best_trial_id = 0
    best_block_id = 0

    if overlapping:
        # Sort by: overlap desc, priority asc, later onset desc (helps boundary alignment), longer duration desc
        overlapping.sort(
            key=lambda x: (
                -x[0],                      # overlap seconds descending
                event_priority(x[2]),       # priority (artifact/calibration/movement/rest)
                -float(x[2]["onset_s"]),    # later onset preferred
                -float(x[2]["duration_s"]), # longer event preferred
            )
        )
        best_ov, best_ov_frac, best = overlapping[0]

        if best["type"] == "artifact":
            artifact_flag = 1
        else:
            # Minimum overlap gating:
            # If overlap is weak, treat as REST by exclusion.
            if best_ov >= MIN_OVERLAP_SEC or int(best["action_id"]) == int(ACTION_REST):
                action_id = int(best["action_id"])
                finger_id = int(best["finger_id"])
                confidence_hint = best.get("confidence", np.nan)
                assigned_type = best.get("type", "") or "event"
                best_onset = float(best["onset_s"])
                best_dur = float(best["duration_s"])
                best_source = str(best.get("source", ""))
                best_session_mode = str(best.get("session_mode", ""))
                best_trial_id = int(best.get("trial_id", 0))
                best_block_id = int(best.get("block_id", 0))

            else:
                # Not enough overlap: leave REST/NONE
                action_id = int(ACTION_REST)
                finger_id = int(FINGER_NONE)
                assigned_type = "rest_by_low_overlap"
                best_session_mode = str(best.get("session_mode", ""))
                best_trial_id = int(best.get("trial_id", 0))
                best_block_id = int(best.get("block_id", 0))

    if artifact_flag:
        continue

    # Enforce label validity (multi-head invariants)
    if not is_valid_action_finger(action_id, finger_id):
        continue

    segment = signal[start_idx:end_idx]
    if segment.shape[0] != WINDOW_SAMPLES:
        continue

    # Tabular features (kept consistent with your original schema)
    features = segment.mean(axis=0)

    # Accumulate outputs
    sequence_windows.append(segment.astype(np.float32))
    action_labels.append(int(action_id))
    finger_labels.append(int(finger_id))

    subject_ids.append(session_meta.get("subject_id", "UNKNOWN"))
    experiment_hashes.append(session_meta.get("experiment_hash", "UNKNOWN"))

    window_starts.append(window_start)
    window_ends.append(window_end)
    confidence_hints.append(float(confidence_hint) if pd.notna(confidence_hint) else np.nan)
    artifact_flags.append(int(artifact_flag))

    # New QA fields
    assigned_event_types.append(str(assigned_type))
    overlap_seconds.append(float(best_ov))
    overlap_fracs.append(float(best_ov_frac))
    event_onsets.append(float(best_onset) if np.isfinite(best_onset) else np.nan)
    event_durations.append(float(best_dur) if np.isfinite(best_dur) else np.nan)
    event_sources.append(str(best_source) if best_source is not None else "")
    session_modes.append(str(best_session_mode) if best_session_mode is not None else "")
    trial_ids.append(int(best_trial_id))
    block_ids.append(int(best_block_id))

    windows.append({
        "ch1": float(features[0]),
        "ch2": float(features[1]),
        "ch3": float(features[2]),
        "ch4": float(features[3]),
        "action_id": int(action_id),
        "finger_id": int(finger_id),
        "subject_id": session_meta.get("subject_id", "UNKNOWN"),
        "experiment_hash": session_meta.get("experiment_hash", "UNKNOWN"),
        "window_start": float(window_start),
        "window_end": float(window_end),
        "confidence_hint": float(confidence_hint) if pd.notna(confidence_hint) else np.nan,
        "artifact_flag": int(artifact_flag),

        # QA metadata
        "assigned_event_type": str(assigned_type),
        "overlap_s": float(best_ov),
        "overlap_frac": float(best_ov_frac),
        "event_onset_s": float(best_onset) if np.isfinite(best_onset) else np.nan,
        "event_duration_s": float(best_dur) if np.isfinite(best_dur) else np.nan,
        "event_source": str(best_source),
        "session_mode": str(best_session_mode),
        "trial_id": int(best_trial_id),
        "block_id": int(best_block_id),
    })

# =========================
# ===== SAVE CSV ==========
# =========================
pd.DataFrame(windows).to_csv(OUT_FILE, index=False)
print(f"✅ Saved {len(windows)} windows → {OUT_FILE}")

# =========================
# ===== SAVE NPZ ==========
# =========================
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

        # New QA fields
        assigned_event_type=np.array(assigned_event_types, dtype="U"),
        overlap_s=np.array(overlap_seconds, dtype=np.float32),
        overlap_frac=np.array(overlap_fracs, dtype=np.float32),
        event_onset_s=np.array(event_onsets, dtype=np.float32),
        event_duration_s=np.array(event_durations, dtype=np.float32),
        event_source=np.array(event_sources, dtype="U"),
        session_mode=np.array(session_modes, dtype="U"),
        trial_id=np.array(trial_ids, dtype=np.int64),
        block_id=np.array(block_ids, dtype=np.int64),

        fs=np.array(FS, dtype=np.int64),
        window_sec=np.array(WINDOW_SEC, dtype=np.float32),
        step_sec=np.array(STEP_SEC, dtype=np.float32),
        channel_names=np.array(["ch1", "ch2", "ch3", "ch4"], dtype="U"),
        config=np.array([json.dumps({
            "min_overlap_ratio": MIN_OVERLAP_RATIO,
            "guard_band_sec": GUARD_BAND_SEC,
            "artifact_min_overlap_frac": ARTIFACT_MIN_OVERLAP_FRAC,
            "window_sec": WINDOW_SEC,
            "step_sec": STEP_SEC,
            "fs": FS,
        })], dtype="U"),
    )
    print(f"✅ Saved sequence windows → {OUT_NPZ} with shape {X.shape}")
else:
    print("⚠ No sequence windows produced; check RAW_FILE/EVENT_FILE alignment.")