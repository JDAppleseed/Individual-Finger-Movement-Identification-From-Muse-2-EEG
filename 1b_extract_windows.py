"""
STEP 1b — Window Extraction (time-based resampling, audited)

Converts continuous EEG + event markers → windowed dataset (tabular + sequence npz)

Key upgrades:
- Time-based windows using features.time_s (absolute_v1)
- Resampling to fixed shape via interpolation (handles irregular sampling ~50–90 Hz)
- Gap detection with strict drop by default, optional allow-gaps
- Deterministic session auto-pick with completed-session preference
"""

import json
import argparse
import csv
import sys
from typing import Optional
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
SOURCE_FS_DEFAULT = 256
TARGET_FS_DEFAULT = 256.0
WINDOW_SEC_DEFAULT = 0.25
STEP_SEC = 0.05
PAD_SEC = 0.05
GAP_THRESHOLD_SEC = 0.10
DEDUP_POLICY = "keep_last"
INTERPOLATION_POLICY = "np.interp.linear"

# Label assignment robustness
LABEL_GATED = True  # If True, drop unlabeled windows instead of REST-by-exclusion
KEEP_BASELINE_REST_EVENTS = 2  # Keep REST only if overlapping first N rest events
MIN_OVERLAP_RATIO = 0.20   # fraction of WINDOW_SEC required for non-REST labels
GUARD_BAND_SEC = 0.00     # skip windows within ± this time of any movement event boundary
ARTIFACT_MIN_OVERLAP_FRAC = 0.20  # if artifact overlaps >=20% of window, drop window

RAW_FILE = "eeg_features.csv"
EVENT_FILE = "events.csv"
OUT_FILE = "eeg_windows.csv"
OUT_NPZ = "eeg_windows.npz"
DEFAULT_SUBJECT_ID = "1-M17"


# =========================
# ===== HELPERS ===========
# =========================

def _read_json(path: Path):
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _csv_has_data_rows(path: Path) -> bool:
    if not path or not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if any(str(cell).strip() for cell in row):
                    return True
    except Exception:
        return False
    return False


def _session_id_from_filename(filename: str, subject_id: str):
    prefix = f"{subject_id}_"
    if not filename.startswith(prefix):
        return None
    if filename.endswith("_eeg_features.csv"):
        return filename[len(prefix):-len("_eeg_features.csv")]
    if filename.endswith("_events.csv"):
        return filename[len(prefix):-len("_events.csv")]
    return None


def _session_sort_key(entry: dict):
    meta = entry.get("meta", {})
    session_id = str(meta.get("session_id") or "")
    updated = str(meta.get("updated_utc") or "")
    meta_name = entry.get("meta_path").name if entry.get("meta_path") else ""
    return (session_id, updated, meta_name)


def _collect_session_meta(base_dir: Path, subject_id: Optional[str]):
    candidates = []
    if not base_dir.exists():
        return candidates
    for meta_path in sorted(base_dir.glob("*_session_meta.json")):
        meta = _read_json(meta_path)
        if not meta:
            continue
        if subject_id and meta.get("subject_id") != subject_id:
            continue
        features_path = Path(meta.get("features_path", "")) if meta.get("features_path") else None
        events_path = Path(meta.get("events_path", "")) if meta.get("events_path") else None
        if not features_path or not events_path:
            continue
        if not (_csv_has_data_rows(features_path) and _csv_has_data_rows(events_path)):
            continue
        candidates.append({
            "meta": meta,
            "meta_path": meta_path,
            "features_path": features_path,
            "events_path": events_path,
            "complete": bool(meta.get("complete")),
        })
    return candidates


def _find_latest_pair_by_subject(subject_id: str, base_dir: Path):
    if not base_dir.exists():
        return None
    features_files = sorted(base_dir.glob(f"{subject_id}_*_eeg_features.csv"), key=lambda p: p.name)
    candidates = []
    for feat in features_files:
        session_id = _session_id_from_filename(feat.name, subject_id)
        if not session_id:
            continue
        events_path = base_dir / f"{subject_id}_{session_id}_events.csv"
        if not events_path.exists():
            continue
        if not (_csv_has_data_rows(feat) and _csv_has_data_rows(events_path)):
            continue
        candidates.append((session_id, feat, events_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def _infer_events_from_features(features_path: Path):
    if not features_path:
        return None
    name = features_path.name
    if name.endswith("_eeg_features.csv"):
        return features_path.with_name(name.replace("_eeg_features.csv", "_events.csv"))
    return None


def _infer_features_from_events(events_path: Path):
    if not events_path:
        return None
    name = events_path.name
    if name.endswith("_events.csv"):
        return events_path.with_name(name.replace("_events.csv", "_eeg_features.csv"))
    return None


def _select_session_paths(args):
    session_meta = {}
    features_path = Path(args.features) if args.features else None
    events_path = Path(args.events) if args.events else None

    if features_path or events_path:
        if features_path and not events_path:
            inferred = _infer_events_from_features(features_path)
            if inferred and inferred.exists():
                events_path = inferred
        if events_path and not features_path:
            inferred = _infer_features_from_events(events_path)
            if inferred and inferred.exists():
                features_path = inferred
        if not features_path or not events_path:
            root_meta = _read_json(Path("session_meta.json"))
            if root_meta:
                session_meta = root_meta
                if not features_path:
                    features_path = Path(root_meta.get("features_path", RAW_FILE))
                if not events_path:
                    events_path = Path(root_meta.get("events_path", EVENT_FILE))
        return features_path, events_path, session_meta, "overrides"

    base_dir = Path("data/processed")
    candidates = _collect_session_meta(base_dir, args.subject_id)
    selected = None
    if candidates:
        complete = [c for c in candidates if c.get("complete")]
        pool = complete if complete else candidates
        pool.sort(key=_session_sort_key)
        selected = pool[-1]

    if selected:
        session_meta = selected.get("meta", {})
        features_path = selected.get("features_path")
        events_path = selected.get("events_path")
        source = f"session_meta:{selected.get('meta_path').name}"
        return features_path, events_path, session_meta, source

    root_meta = _read_json(Path("session_meta.json"))
    if root_meta:
        features_path = Path(root_meta.get("features_path", RAW_FILE))
        events_path = Path(root_meta.get("events_path", EVENT_FILE))
        session_meta = root_meta
        return features_path, events_path, session_meta, "session_meta.json"

    if args.subject_id:
        pair = _find_latest_pair_by_subject(args.subject_id, base_dir)
        if pair:
            _, features_path, events_path = pair
            return features_path, events_path, session_meta, "latest_subject_files"

    return None, None, session_meta, "none"


def _dedupe_times_keep_last(times: np.ndarray, signal: np.ndarray):
    order = np.argsort(times)
    times_sorted = times[order]
    signal_sorted = signal[order]
    if times_sorted.size == 0:
        return times_sorted, signal_sorted
    rev_idx = np.unique(times_sorted[::-1], return_index=True)[1]
    keep_idx = times_sorted.size - 1 - rev_idx
    keep_idx.sort()
    return times_sorted[keep_idx], signal_sorted[keep_idx]


def _load_features(path: Path):
    df = pd.read_csv(path)
    if "time_s" not in df.columns:
        raise RuntimeError(f"time_s column missing in features file: {path}")
    times = df["time_s"].astype(float).to_numpy()

    channel_cols = [c for c in ["ch1", "ch2", "ch3", "ch4"] if c in df.columns]
    if len(channel_cols) < 1:
        raise RuntimeError(f"No EEG channel columns found in {path} (expected ch1..ch4)")
    signal = df[channel_cols].astype(float).to_numpy()

    valid_mask = np.isfinite(times)
    times = times[valid_mask]
    signal = signal[valid_mask]

    if times.size < 2:
        raise RuntimeError(f"Not enough valid time_s samples in {path}.")

    times, signal = _dedupe_times_keep_last(times, signal)
    if times.size < 2:
        raise RuntimeError(f"Not enough unique time_s samples in {path} after dedupe.")

    if np.any(np.diff(times) <= 0):
        raise RuntimeError(f"time_s must be strictly increasing after dedupe in {path}.")

    return times, signal, channel_cols


def _load_events(path: Path):
    events_df = pd.read_csv(path)
    events = []
    for _, row in events_df.iterrows():
        try:
            onset_s = float(row.get("onset_s", np.nan))
        except Exception:
            onset_s = np.nan
        if not np.isfinite(onset_s):
            continue
        try:
            duration_s = float(row.get("duration_s", 0.0))
        except Exception:
            duration_s = 0.0
        if duration_s < 0:
            duration_s = 0.0

        end_s = row.get("end_s", np.nan)
        try:
            end_s = float(end_s)
        except Exception:
            end_s = np.nan
        if not np.isfinite(end_s):
            end_s = onset_s + duration_s

        e = {
            "onset_s": onset_s,
            "duration_s": duration_s,
            "end_s": end_s,
            "type": str(row.get("type", "")).strip(),
            "finger_id": int(row.get("finger_id", 0)) if "finger_id" in row else 0,
            "action_id": int(row.get("action_id", 0)) if "action_id" in row else 0,
            "confidence": row.get("confidence", np.nan),
            "source": str(row.get("source", "")).strip() or "unknown",
            "notes": str(row.get("notes", "")).strip() if "notes" in row else "",
            "session_mode": str(row.get("session_mode", "")).strip() if "session_mode" in row else "",
            "trial_id": int(row.get("trial_id", 0)) if "trial_id" in row else 0,
            "block_id": int(row.get("block_id", 0)) if "block_id" in row else 0,
        }
        events.append(e)
    return events


def overlap_s(a_start, a_end, b_start, b_end) -> float:
    """Returns overlap duration in seconds between [a_start,a_end] and [b_start,b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


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


def is_baseline_rest_window(window_start, window_end, baseline_rest_events, min_overlap_sec) -> bool:
    if not baseline_rest_events:
        return False
    for e in baseline_rest_events:
        ov = overlap_s(window_start, window_end, e["onset_s"], e["end_s"])
        if ov >= min_overlap_sec:
            return True
    return False


# =========================
# ===== MAIN ==============
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=None, help="Override features path")
    parser.add_argument("--events", type=str, default=None, help="Override events path")
    parser.add_argument("--subject-id", type=str, default=DEFAULT_SUBJECT_ID, help="Subject ID to select latest session files")
    parser.add_argument("--target-fs", type=float, default=None, help="Target sampling rate (Hz) for resampling windows")
    parser.add_argument("--allow-gaps", action="store_true", help="Keep windows with large gaps and mark them")
    parser.add_argument("--ignore-misalignment", action="store_true", help="Warn but continue if events are outside feature range")
    args = parser.parse_args()

    features_path, events_path, session_meta, source = _select_session_paths(args)

    if not features_path:
        print("No features file found. Run: python 1_stream_and_record.py to create a new session, then re-run 1b_extract_windows.py.")
        raise SystemExit(2)
    if not features_path.exists():
        print(f"No features file found at {features_path}")
        raise SystemExit(2)

    if LABEL_GATED and (not events_path or not events_path.exists()):
        print("No events file found. Provide --events PATH or run Step 1 to create events before extraction.")
        raise SystemExit(2)

    target_fs = float(args.target_fs) if args.target_fs is not None else float(session_meta.get("sampling_rate", TARGET_FS_DEFAULT))
    window_sec = float(session_meta.get("window_sec", WINDOW_SEC_DEFAULT)) if session_meta else WINDOW_SEC_DEFAULT
    window_samples = int(round(window_sec * target_fs))
    if window_samples <= 0:
        print(f"Invalid window_samples={window_samples}; check window_sec={window_sec} and target_fs={target_fs}.")
        raise SystemExit(2)

    min_overlap_sec = MIN_OVERLAP_RATIO * window_sec

    timebase_version = session_meta.get("timebase_version") or session_meta.get("timebase") or "unknown"

    print(f"Session selection source: {source}")
    print(f"Using features file: {features_path}")
    print(f"Using events file: {events_path}")
    print(f"Target window rate: {target_fs} Hz ({window_samples} samples/window)")
    print(f"Interpolation policy: {INTERPOLATION_POLICY}, dedupe: {DEDUP_POLICY}")

    # =========================
    # ===== LOAD DATA =========
    # =========================
    times, signal, channel_cols = _load_features(features_path)

    events = _load_events(events_path)

    # Alignment strictness
    if events:
        features_min = float(times[0])
        features_max = float(times[-1])
        event_onsets = np.array([e["onset_s"] for e in events], dtype=float)
        outside_mask = (event_onsets < features_min) | (event_onsets > features_max)
        outside_count = int(outside_mask.sum())
        if outside_count > 0:
            msg = (
                f"Event onsets outside feature time range: {outside_count}/{len(event_onsets)} "
                f"(features range {features_min:.4f}..{features_max:.4f})."
            )
            if args.ignore_misalignment:
                print(f"⚠️ {msg} Proceeding due to --ignore-misalignment.")
            else:
                print(f"❌ {msg}")
                raise SystemExit(2)

    # Precompute boundaries for guard band (movement events only)
    movement_boundaries = []
    for e in events:
        if e["type"] == "artifact":
            continue
        if int(e["action_id"]) != int(ACTION_REST):
            movement_boundaries.append(float(e["onset_s"]))
            movement_boundaries.append(float(e["end_s"]))
    movement_boundaries = np.array(sorted(set(movement_boundaries)), dtype=float)

    # Baseline REST allow-list (first N rest events by onset)
    baseline_rest_events = []
    if KEEP_BASELINE_REST_EVENTS > 0:
        rest_events = [
            e for e in events
            if int(e["action_id"]) == int(ACTION_REST) and e["type"] == "rest"
        ]
        rest_events.sort(key=lambda x: float(x["onset_s"]))
        baseline_rest_events = rest_events[:KEEP_BASELINE_REST_EVENTS]

    def in_guard_band(window_start, window_end) -> bool:
        if movement_boundaries.size == 0:
            return False
        mid = 0.5 * (window_start + window_end)
        return np.any(np.abs(movement_boundaries - mid) <= GUARD_BAND_SEC)

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
    gap_flags = []

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

    start_time = float(times[0])
    end_time = float(times[-1])
    last_start = end_time - window_sec
    if last_start < start_time:
        raise RuntimeError(
            f"Not enough time coverage in {features_path}: "
            f"{end_time - start_time:.3f}s available, need {window_sec:.3f}s."
        )

    window_starts_grid = np.arange(start_time, last_start + 1e-9, STEP_SEC, dtype=float)
    if window_starts_grid.size == 0:
        raise RuntimeError(
            f"No windows available with step {STEP_SEC:.3f}s over {end_time - start_time:.3f}s span."
        )

    total_windows = 0
    kept_windows = 0
    drop_no_overlap = 0
    drop_artifact = 0
    drop_guard_band = 0
    drop_invalid_label = 0
    drop_short_segment = 0
    drop_gap = 0

    for window_start in window_starts_grid:
        total_windows += 1
        window_end = float(window_start + window_sec)

        if GUARD_BAND_SEC > 0 and in_guard_band(window_start, window_end):
            drop_guard_band += 1
            continue

        mask_pad = (times >= (window_start - PAD_SEC)) & (times <= (window_end + PAD_SEC))
        if not np.any(mask_pad):
            drop_short_segment += 1
            continue

        times_pad = times[mask_pad]
        signal_pad = signal[mask_pad]
        core_count = np.sum((times_pad >= window_start) & (times_pad < window_end))
        if core_count < 2 or times_pad.size < 2:
            drop_short_segment += 1
            continue

        max_dt = float(np.max(np.diff(times_pad))) if times_pad.size >= 2 else np.inf
        gap_flag = int(max_dt > GAP_THRESHOLD_SEC)
        if gap_flag and not args.allow_gaps:
            drop_gap += 1
            continue

        # Compute overlaps with events
        overlapping = []
        any_artifact_flag = 0

        for e in events:
            ov = overlap_s(window_start, window_end, e["onset_s"], e["end_s"])
            if ov <= 0:
                continue

            ov_frac = ov / window_sec

            if e["type"] == "artifact" and ov_frac >= ARTIFACT_MIN_OVERLAP_FRAC:
                any_artifact_flag = 1
                break

            overlapping.append((ov, ov_frac, e))

        if any_artifact_flag:
            drop_artifact += 1
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

        if LABEL_GATED and not overlapping:
            if is_baseline_rest_window(window_start, window_end, baseline_rest_events, min_overlap_sec):
                assigned_type = "baseline_rest"
            else:
                drop_no_overlap += 1
                continue

        if overlapping:
            overlapping.sort(
                key=lambda x: (
                    -x[0],
                    event_priority(x[2]),
                    -float(x[2]["onset_s"]),
                    -float(x[2]["duration_s"]),
                )
            )
            best_ov, best_ov_frac, best = overlapping[0]

            if best["type"] == "artifact":
                artifact_flag = 1
            else:
                if best_ov >= min_overlap_sec:
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
                elif int(best["action_id"]) == int(ACTION_REST):
                    if LABEL_GATED:
                        if is_baseline_rest_window(window_start, window_end, baseline_rest_events, min_overlap_sec):
                            action_id = int(ACTION_REST)
                            finger_id = int(FINGER_NONE)
                            assigned_type = "baseline_rest"
                            best_session_mode = str(best.get("session_mode", ""))
                            best_trial_id = int(best.get("trial_id", 0))
                            best_block_id = int(best.get("block_id", 0))
                        else:
                            drop_no_overlap += 1
                            continue
                    else:
                        action_id = int(ACTION_REST)
                        finger_id = int(FINGER_NONE)
                        assigned_type = "rest_by_exclusion"
                        best_session_mode = str(best.get("session_mode", ""))
                        best_trial_id = int(best.get("trial_id", 0))
                        best_block_id = int(best.get("block_id", 0))
                else:
                    if LABEL_GATED:
                        drop_no_overlap += 1
                        continue
                    action_id = int(ACTION_REST)
                    finger_id = int(FINGER_NONE)
                    assigned_type = "rest_by_low_overlap"
                    best_session_mode = str(best.get("session_mode", ""))
                    best_trial_id = int(best.get("trial_id", 0))
                    best_block_id = int(best.get("block_id", 0))

        if artifact_flag:
            drop_artifact += 1
            continue

        if not is_valid_action_finger(action_id, finger_id):
            drop_invalid_label += 1
            continue

        grid = np.linspace(window_start, window_end, window_samples, endpoint=False)
        segment = np.empty((window_samples, signal.shape[1]), dtype=float)
        for ch_idx in range(signal.shape[1]):
            segment[:, ch_idx] = np.interp(grid, times_pad, signal_pad[:, ch_idx])

        features = segment.mean(axis=0)

        sequence_windows.append(segment.astype(np.float32))
        action_labels.append(int(action_id))
        finger_labels.append(int(finger_id))

        subject_ids.append(session_meta.get("subject_id", "UNKNOWN"))
        experiment_hashes.append(session_meta.get("experiment_hash", "UNKNOWN"))

        window_starts.append(window_start)
        window_ends.append(window_end)
        confidence_hints.append(float(confidence_hint) if pd.notna(confidence_hint) else np.nan)
        artifact_flags.append(int(artifact_flag))
        gap_flags.append(int(gap_flag))

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
            "ch1": float(features[0]) if signal.shape[1] > 0 else 0.0,
            "ch2": float(features[1]) if signal.shape[1] > 1 else 0.0,
            "ch3": float(features[2]) if signal.shape[1] > 2 else 0.0,
            "ch4": float(features[3]) if signal.shape[1] > 3 else 0.0,
            "action_id": int(action_id),
            "finger_id": int(finger_id),
            "subject_id": session_meta.get("subject_id", "UNKNOWN"),
            "experiment_hash": session_meta.get("experiment_hash", "UNKNOWN"),
            "window_start": float(window_start),
            "window_end": float(window_end),
            "confidence_hint": float(confidence_hint) if pd.notna(confidence_hint) else np.nan,
            "artifact_flag": int(artifact_flag),
            "gap_flag": int(gap_flag),

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
        kept_windows += 1

    # =========================
    # ===== SAVE CSV ==========
    # =========================
    pd.DataFrame(windows).to_csv(OUT_FILE, index=False)
    print(f"✅ Saved {len(windows)} windows → {OUT_FILE}")

    action_dist = pd.Series(action_labels).value_counts().sort_index().to_dict()
    finger_dist = pd.Series(finger_labels).value_counts().sort_index().to_dict()
    print("---- Window Extraction Summary ----")
    print(f"Total windows considered: {total_windows}")
    print(f"Windows kept: {kept_windows}")
    print(
        "Dropped windows (no-overlap): "
        f"{drop_no_overlap}, artifact-overlap: {drop_artifact}, guard-band: {drop_guard_band}, "
        f"invalid label: {drop_invalid_label}, short segment: {drop_short_segment}, gap: {drop_gap}"
    )
    print(f"Kept class distribution (action_id): {action_dist}")
    print(f"Kept class distribution (finger_id): {finger_dist}")
    if KEEP_BASELINE_REST_EVENTS == 0:
        print("Sanity: KEEP_BASELINE_REST_EVENTS=0 → no REST windows are kept.")

    # =========================
    # ===== SAVE NPZ ==========
    # =========================
    if sequence_windows:
        X = np.stack(sequence_windows).astype(np.float32)
        gap_policy = "allow_gaps" if args.allow_gaps else "strict_drop"
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
            gap_flag=np.array(gap_flags, dtype=np.int64),

            assigned_event_type=np.array(assigned_event_types, dtype="U"),
            overlap_s=np.array(overlap_seconds, dtype=np.float32),
            overlap_frac=np.array(overlap_fracs, dtype=np.float32),
            event_onset_s=np.array(event_onsets, dtype=np.float32),
            event_duration_s=np.array(event_durations, dtype=np.float32),
            event_source=np.array(event_sources, dtype="U"),
            session_mode=np.array(session_modes, dtype="U"),
            trial_id=np.array(trial_ids, dtype=np.int64),
            block_id=np.array(block_ids, dtype=np.int64),

            fs=np.array(int(round(target_fs)), dtype=np.int64),
            target_fs=np.array(target_fs, dtype=np.float32),
            window_sec=np.array(window_sec, dtype=np.float32),
            step_sec=np.array(STEP_SEC, dtype=np.float32),
            channel_names=np.array(channel_cols, dtype="U"),
            timebase_version=np.array(str(timebase_version), dtype="U"),
            interpolation_policy=np.array(INTERPOLATION_POLICY, dtype="U"),
            gap_policy=np.array(gap_policy, dtype="U"),
            features_path=np.array(str(features_path), dtype="U"),
            events_path=np.array(str(events_path), dtype="U"),
            config=np.array([json.dumps({
                "min_overlap_ratio": MIN_OVERLAP_RATIO,
                "guard_band_sec": GUARD_BAND_SEC,
                "artifact_min_overlap_frac": ARTIFACT_MIN_OVERLAP_FRAC,
                "window_sec": window_sec,
                "step_sec": STEP_SEC,
                "fs": float(target_fs),
                "source_fs": SOURCE_FS_DEFAULT,
                "target_fs": float(target_fs),
                "pad_sec": PAD_SEC,
                "gap_threshold_sec": GAP_THRESHOLD_SEC,
                "gap_policy": gap_policy,
                "interpolation_policy": INTERPOLATION_POLICY,
                "dedupe_policy": DEDUP_POLICY,
            })], dtype="U"),
        )
        print(f"✅ Saved sequence windows → {OUT_NPZ} with shape {X.shape}")
    else:
        print("⚠ No sequence windows produced; check features/events alignment.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
