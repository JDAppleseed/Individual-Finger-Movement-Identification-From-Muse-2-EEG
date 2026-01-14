#!/usr/bin/env python3
"""
STEP 1b — Window Extraction (time-based resampling, audited)

Converts continuous EEG + event markers → windowed dataset (CSV + sequence NPZ).

Key upgrades:
- Time-based windows using features.time_s (absolute_v1)
- Resampling to fixed shape via interpolation (handles irregular sampling ~50–90 Hz)
- Gap detection with strict drop by default, optional allow-gaps
- Deterministic session auto-pick with completed-session preference
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
WINDOW_SEC = WINDOW_SEC_DEFAULT
STEP_SEC = 0.05
PAD_SEC = 0.05

GAP_THRESHOLD_SEC = 0.10
DEDUP_POLICY = "keep_last"
INTERPOLATION_POLICY = "np.interp.linear"

# Label assignment robustness
LABEL_GATED = True  # If True, drop unlabeled windows instead of REST-by-exclusion
KEEP_BASELINE_REST_EVENTS = 2  # Keep REST only if overlapping first N rest events
MIN_OVERLAP_RATIO = 0.20  # fraction of WINDOW_SEC required for non-REST labels
GUARD_BAND_SEC = (
    0.00  # skip windows within ± this time of any movement boundary (midpoint-based)
)
ARTIFACT_MIN_OVERLAP_FRAC = 0.20  # if artifact overlaps >=20% of window, drop window
SEED = 42
REST_SUBSAMPLE_PROB = 1.0
REST_SUBSAMPLE_SEED = 1337
REST_MAX_WINDOWS: Optional[int] = None

RAW_FILE = "eeg_features.csv"
EVENT_FILE = "events.csv"
OUT_FILE = "eeg_windows.csv"
OUT_NPZ = "eeg_windows.npz"
DEFAULT_SUBJECT_ID = "1-M60"
ROOT_DIR = Path(__file__).resolve().parent


# =========================
# ===== UTILITIES =========
# =========================


def _load_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return payload.get("settings", payload)


def _apply_config(settings: Dict[str, Any]):
    for key, val in settings.items():
        if key in globals():
            globals()[key] = val


def _apply_config_to_args(args_obj, settings: Dict[str, Any], defaults: Dict[str, Any]):
    for key, default in defaults.items():
        if key in settings and getattr(args_obj, key) == default:
            setattr(args_obj, key, settings[key])


def _read_json(path: Path) -> Dict[str, Any]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def load_root_session_meta() -> Dict[str, Any]:
    return _read_json(ROOT_DIR / "session_meta.json")


def _resolve_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    return ROOT_DIR / p


def _csv_has_data_rows(path: Optional[Path]) -> bool:
    if not path or not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)  # header
            for row in reader:
                if any(str(cell).strip() for cell in row):
                    return True
    except Exception:
        return False
    return False


def _session_id_from_filename(filename: str, subject_id: str) -> Optional[str]:
    """
    Expected:
      {subject_id}_{session_id}_eeg_features.csv
      {subject_id}_{session_id}_events*.csv
    """
    prefix = f"{subject_id}_"
    if not filename.startswith(prefix):
        return None
    if filename.endswith("_eeg_features.csv"):
        return filename[len(prefix) : -len("_eeg_features.csv")]
    events_idx = filename.find("_events")
    if events_idx != -1:
        return filename[len(prefix) : events_idx]
    return None


def _events_prefix_from_features(features_path: Path) -> Optional[str]:
    if not features_path:
        return None
    name = features_path.name
    suffix = "_eeg_features.csv"
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]  # includes "{subject}_{session}"


def _features_prefix_from_events(events_path: Path) -> Optional[str]:
    if not events_path:
        return None
    name = events_path.name
    idx = name.find("_events")
    if idx < 0:
        return None
    return name[:idx]  # includes "{subject}_{session}"


def _select_events_for_prefix(prefix: str, base_dir: Path) -> Optional[Path]:
    """
    prefix example: "1-M60_20260103_153353"
    Preference order:
      1) shifted (highest name sort)
      2) exact "{prefix}_events.csv"
      3) latest non-autosave
      4) latest anything
    """
    if not prefix:
        return None

    candidates = sorted(base_dir.glob(f"{prefix}_events*.csv"), key=lambda p: p.name)
    if not candidates:
        return None

    shifted = [p for p in candidates if "_events_shifted_" in p.name]
    if shifted:
        return sorted(shifted, key=lambda p: p.name)[-1]

    exact = base_dir / f"{prefix}_events.csv"
    if exact.exists():
        return exact

    non_autosave = [p for p in candidates if "_events_autosave" not in p.name]
    if non_autosave:
        return sorted(non_autosave, key=lambda p: p.name)[-1]

    return candidates[-1]


def infer_subject_session_from_features_path(path: Path):
    """
    Expected filename:
      <subject_id>_<YYYYMMDD>_<HHMMSS>_eeg_features.csv

    Example:
      1-M60_20260103_153353_eeg_features.csv
      subject_id = "1-M60"
      session_id = "20260103_153353"
    """
    if not path:
        return None, None

    name = path.name

    m = re.match(r"^(?P<subject>.+?)_(?P<session>\d{8}_\d{6})_eeg_features\.csv$", name)
    if m:
        return m.group("subject"), m.group("session")

    # fallback: if it ends with _eeg_features.csv but doesn't match strict pattern
    suffix = "_eeg_features.csv"
    if name.endswith(suffix):
        prefix = name[: -len(suffix)]
        # Try split on last 2 underscores to recover session like YYYYMMDD_HHMMSS
        parts = prefix.split("_")
        if len(parts) >= 3:
            session = "_".join(parts[-2:])
            subject = "_".join(parts[:-2])
            return subject, session
        return prefix, None

    return None, None


def _infer_events_shift_s(events_path: Optional[Path]) -> float:
    """
    Parse ..._events_shifted_49.5s.csv or ..._events_shifted_-2.0s.csv
    """
    if not events_path:
        return 0.0
    m = re.search(
        r"_events_shifted_(?P<shift>-?\d+(?:\.\d+)?)s\.csv$", events_path.name
    )
    if m:
        try:
            return float(m.group("shift"))
        except ValueError:
            return 0.0
    return 0.0


def _load_latest_session_meta(base_dir: Path) -> Dict[str, Any]:
    if not base_dir.exists():
        return {}
    candidates = list(base_dir.glob("*_session_meta.json"))
    if not candidates:
        return {}
    try:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
    except Exception:
        return {}
    return _read_json(latest)


def _session_sort_key(entry: dict):
    meta = entry.get("meta", {})
    session_id = str(meta.get("session_id") or "")
    updated = str(meta.get("updated_utc") or "")
    meta_path = entry.get("meta_path")
    meta_name = meta_path.name if meta_path else ""
    return (session_id, updated, meta_name)


def _infer_events_from_features(features_path: Optional[Path]) -> Optional[Path]:
    if not features_path:
        return None
    prefix = _events_prefix_from_features(features_path)
    if not prefix:
        return None
    exact = features_path.with_name(f"{prefix}_events.csv")
    if exact.exists():
        return exact
    return _select_events_for_prefix(prefix, features_path.parent)


def _infer_features_from_events(events_path: Optional[Path]) -> Optional[Path]:
    if not events_path:
        return None
    prefix = _features_prefix_from_events(events_path)
    if not prefix:
        return None
    return events_path.with_name(f"{prefix}_eeg_features.csv")


def _collect_session_meta(base_dir: Path, subject_id: Optional[str]):
    candidates: List[Dict[str, Any]] = []
    if not base_dir.exists():
        return candidates

    for meta_path in sorted(base_dir.glob("*_session_meta.json")):
        meta = _read_json(meta_path)
        if not meta:
            continue
        if subject_id and str(meta.get("subject_id", "")) != str(subject_id):
            continue

        features_path = (
            _resolve_path(meta.get("features_path", ""))
            if meta.get("features_path")
            else None
        )
        events_path = (
            _resolve_path(meta.get("events_path", ""))
            if meta.get("events_path")
            else None
        )

        if features_path and not events_path:
            inferred = _infer_events_from_features(features_path)
            if inferred and inferred.exists():
                events_path = inferred
        if events_path and not features_path:
            inferred = _infer_features_from_events(events_path)
            if inferred and inferred.exists():
                features_path = inferred

        if not features_path or not events_path:
            continue
        if not (_csv_has_data_rows(features_path) and _csv_has_data_rows(events_path)):
            continue

        candidates.append(
            {
                "meta": meta,
                "meta_path": meta_path,
                "features_path": features_path,
                "events_path": events_path,
                "complete": bool(meta.get("complete")),
            }
        )

    return candidates


def _find_latest_pair_by_subject(subject_id: str, base_dir: Path):
    if not base_dir.exists():
        return None
    features_files = sorted(
        base_dir.glob(f"{subject_id}_*_eeg_features.csv"), key=lambda p: p.name
    )
    candidates = []
    for feat in features_files:
        session_id = _session_id_from_filename(feat.name, subject_id)
        if not session_id:
            continue
        prefix = f"{subject_id}_{session_id}"
        events_path = _select_events_for_prefix(prefix, base_dir)
        if not events_path:
            continue
        if not (_csv_has_data_rows(feat) and _csv_has_data_rows(events_path)):
            continue
        candidates.append((session_id, feat, events_path))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1]


def _select_session_paths(args):
    """
    Selection order:
      1) overrides (--features/--events), with inference pairing
      2) latest completed session_meta in data/processed
      3) root session_meta.json
      4) latest subject files in data/processed
    """
    session_meta: Dict[str, Any] = {}
    features_path = _resolve_path(args.features) if args.features else None
    events_path = _resolve_path(args.events) if args.events else None

    # (1) overrides
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
            root_meta = load_root_session_meta()
            if root_meta:
                session_meta = root_meta
                if not features_path:
                    features_path = _resolve_path(
                        root_meta.get("features_path", RAW_FILE)
                    )
                if not events_path:
                    events_path = _resolve_path(
                        root_meta.get("events_path", EVENT_FILE)
                    )
        return features_path, events_path, session_meta, "overrides"

    base_dir = ROOT_DIR / "data/processed"

    # (2) session_meta files
    candidates = _collect_session_meta(base_dir, args.subject_id)
    if candidates:
        complete = [c for c in candidates if c.get("complete")]
        pool = complete if complete else candidates
        pool.sort(key=_session_sort_key)
        selected = pool[-1]
        session_meta = selected.get("meta", {})
        features_path = selected.get("features_path")
        events_path = selected.get("events_path")
        return (
            features_path,
            events_path,
            session_meta,
            f"session_meta:{selected.get('meta_path').name}",
        )

    # (3) root session_meta.json
    root_meta = load_root_session_meta()
    if root_meta:
        fp = _resolve_path(root_meta.get("features_path", RAW_FILE))
        ep = _resolve_path(root_meta.get("events_path", EVENT_FILE))
        if _csv_has_data_rows(fp) and _csv_has_data_rows(ep):
            return fp, ep, root_meta, "session_meta.json"

    # (4) latest subject files
    if args.subject_id:
        pair = _find_latest_pair_by_subject(args.subject_id, base_dir)
        if pair:
            _, fp, ep = pair
            return fp, ep, session_meta, "latest_subject_files"

    return None, None, session_meta, "none"


def _dedupe_times_keep_last(times: np.ndarray, signal: np.ndarray):
    order = np.argsort(times)
    times_sorted = times[order]
    signal_sorted = signal[order]
    if times_sorted.size == 0:
        return times_sorted, signal_sorted

    # keep last sample for each duplicated time
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
        raise RuntimeError(
            f"No EEG channel columns found in {path} (expected ch1..ch4)"
        )
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
        raise RuntimeError(
            f"time_s must be strictly increasing after dedupe in {path}."
        )

    return times, signal, channel_cols


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _event_id_from_row(row: pd.Series, fallback_id: int) -> Optional[int]:
    raw_id = row.get("event_id", row.get("id", fallback_id))
    try:
        if pd.isna(raw_id):
            return fallback_id
    except Exception:
        pass
    return _safe_int(raw_id, fallback_id)


def _event_key_from_fields(
    onset_s: float,
    duration_s: float,
    action_id: int,
    finger_id: int,
    event_type: str,
    precision: int = 6,
) -> Tuple[Any, ...]:
    return (
        round(float(onset_s), precision),
        round(float(duration_s), precision),
        int(action_id),
        int(finger_id),
        str(event_type),
    )


def _event_key(event: Dict[str, Any]) -> Any:
    if event.get("event_id") is not None:
        return ("event_id", int(event["event_id"]))
    return ("fields",) + _event_key_from_fields(
        event.get("onset_s", np.nan),
        event.get("duration_s", np.nan),
        event.get("action_id", ACTION_REST),
        event.get("finger_id", FINGER_NONE),
        event.get("type", ""),
    )


def _event_id_sort_key(event: Dict[str, Any]) -> Any:
    event_index = event.get("event_index")
    if event_index is not None:
        return int(event_index)
    event_id = event.get("event_id")
    if event_id is None:
        return str(_event_key(event))
    return int(event_id)


def _overlap_sort_key(overlap: float, event: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        -float(overlap),
        event_priority(event),
        -float(event.get("duration_s", 0.0)),
        float(event.get("onset_s", 0.0)),
        _event_id_sort_key(event),
    )


def _overlap_tie_key(overlap: float, event: Dict[str, Any]) -> Tuple[Any, ...]:
    return (
        float(overlap),
        event_priority(event),
        float(event.get("duration_s", 0.0)),
        float(event.get("onset_s", 0.0)),
    )


def _select_best_overlap(
    overlaps: List[Tuple[float, float, Dict[str, Any]]],
) -> Tuple[Optional[Tuple[float, float, Dict[str, Any]]], bool]:
    if not overlaps:
        return None, False
    overlaps.sort(key=lambda item: _overlap_sort_key(item[0], item[2]))
    best = overlaps[0]
    best_key = _overlap_tie_key(best[0], best[2])
    ambiguous = any(_overlap_tie_key(ov, ev) == best_key for ov, _, ev in overlaps[1:])
    return best, ambiguous


def _load_events(path: Path, session_meta: Optional[Dict[str, Any]] = None):
    events_df = pd.read_csv(path)
    events: List[Dict[str, Any]] = []
    stream_start_lsl = np.nan
    if session_meta:
        stream_start_lsl = _safe_float(
            session_meta.get("stream_start_lsl_ts"), default=np.nan
        )

    for row_idx, row in events_df.iterrows():
        onset_s = _safe_float(row.get("event_time_s", np.nan))
        if not np.isfinite(onset_s):
            onset_s = _safe_float(row.get("onset_s", np.nan))
        if not np.isfinite(onset_s):
            onset_lsl = _safe_float(row.get("onset_lsl", np.nan))
            if np.isfinite(onset_lsl) and np.isfinite(stream_start_lsl):
                onset_s = float(onset_lsl - stream_start_lsl)
        if not np.isfinite(onset_s):
            continue

        duration_s = _safe_float(row.get("duration_s", 0.0))
        if duration_s < 0:
            duration_s = 0.0

        end_s = _safe_float(row.get("end_s", np.nan))
        if not np.isfinite(end_s):
            end_s = onset_s + duration_s

        e = {
            "onset_s": float(onset_s),
            "duration_s": float(duration_s),
            "end_s": float(end_s),
            "type": str(row.get("type", "")).strip(),
            "finger_id": _safe_int(row.get("finger_id", 0), 0),
            "action_id": _safe_int(row.get("action_id", 0), 0),
            "confidence": row.get("confidence", np.nan),
            "source": str(row.get("source", "")).strip() or "unknown",
            "notes": str(row.get("notes", "")).strip() if "notes" in row else "",
            "session_mode": str(row.get("session_mode", "")).strip()
            if "session_mode" in row
            else "",
            "trial_id": _safe_int(row.get("trial_id", 0), 0),
            "block_id": _safe_int(row.get("block_id", 0), 0),
        }
        e["event_id"] = _event_id_from_row(row, int(row_idx))
        e["event_index"] = int(row_idx)
        events.append(e)

    return events


def overlap_s(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Overlap duration (seconds) between [a_start,a_end] and [b_start,b_end]."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def event_priority(e: Dict[str, Any]) -> int:
    """
    Lower number = higher priority when overlap ties.
      0: artifact
      1: calibration (if present)
      2: movement (action != REST)
      3: rest
      4: other
    """
    if e.get("type") == "artifact":
        return 0
    if e.get("type") == "calibration":
        return 1
    if int(e.get("action_id", ACTION_REST)) != int(ACTION_REST):
        return 2
    if int(e.get("action_id", ACTION_REST)) == int(ACTION_REST):
        return 3
    return 4


def is_baseline_rest_window(
    window_start: float,
    window_end: float,
    baseline_rest_events: List[Dict[str, Any]],
    min_overlap_sec: float,
) -> bool:
    if not baseline_rest_events:
        return False
    for e in baseline_rest_events:
        if (
            overlap_s(window_start, window_end, e["onset_s"], e["end_s"])
            >= min_overlap_sec
        ):
            return True
    return False


def _select_rest_keep_indices(
    rest_indices: List[int],
    non_rest_count: int,
    subsample_prob: float,
    seed: int,
    rest_cap: Optional[int],
) -> Tuple[List[int], int, int]:
    if not rest_indices:
        return [], 0, 0
    if non_rest_count == 0:
        return list(rest_indices), 0, 0
    rng = np.random.default_rng(int(seed))
    shuffled = rng.permutation(np.array(rest_indices, dtype=int))

    target = int(len(shuffled))
    if subsample_prob < 1.0:
        target = int(np.floor(subsample_prob * len(shuffled)))
    target = max(0, min(len(shuffled), target))
    subsample_dropped = len(shuffled) - target

    cap_dropped = 0
    if rest_cap is not None and rest_cap > 0 and target > rest_cap:
        cap_dropped = target - int(rest_cap)
        target = int(rest_cap)

    keep = shuffled[:target].tolist()
    return keep, subsample_dropped, cap_dropped


def _filter_by_mask(values: List[Any], keep_mask: np.ndarray) -> List[Any]:
    return [val for val, keep in zip(values, keep_mask) if keep]


def _next_available_path(base_path: Path) -> Path:
    if not base_path.exists():
        return base_path
    stem = base_path.stem
    suffix = base_path.suffix
    for i in range(2, 1000):
        candidate = base_path.with_name(f"{stem}_v{i}{suffix}")
        if not candidate.exists():
            return candidate
    return base_path


def _u_dtype_for(s: str, min_len: int = 1) -> np.dtype:
    """Return a NumPy unicode dtype long enough to hold string s without truncation."""
    s = "" if s is None else str(s)
    n = max(min_len, len(s))
    return np.dtype(f"<U{n}")


def _uarr_fill(n: int, value: str) -> np.ndarray:
    """np.full that cannot truncate unicode strings."""
    v = "" if value is None else str(value)
    return np.full((n,), v, dtype=_u_dtype_for(v))


def _uarr_from_list(values: List[str]) -> np.ndarray:
    """np.array for unicode lists with max-length dtype (no truncation)."""
    vals = ["" if v is None else str(v) for v in values]
    max_len = max([1] + [len(v) for v in vals])
    return np.array(vals, dtype=np.dtype(f"<U{max_len}"))


# =========================
# ===== MAIN ==============
# =========================


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument(
        "--features", type=str, default=None, help="Override features path"
    )
    parser.add_argument("--events", type=str, default=None, help="Override events path")
    parser.add_argument(
        "--subject-id",
        type=str,
        default=DEFAULT_SUBJECT_ID,
        help="Subject ID to select latest session files",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=None,
        help="Target sampling rate (Hz) for resampling windows",
    )
    parser.add_argument(
        "--allow-gaps",
        action="store_true",
        help="Keep windows with large gaps and mark them",
    )
    parser.add_argument(
        "--ignore-misalignment",
        action="store_true",
        help="Warn but continue if events are outside feature range",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for deterministic REST subsampling"
    )
    args = parser.parse_args()
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    settings = _load_config(args.config)
    _apply_config(settings)
    _apply_config_to_args(args, settings, defaults)

    if isinstance(settings, dict):
        config_seed = settings.get(
            "seed", settings.get("SEED", settings.get("REST_SUBSAMPLE_SEED"))
        )
    else:
        config_seed = None
    if config_seed is not None:
        seed_value = int(config_seed)
    elif args.seed is not None:
        seed_value = int(args.seed)
    else:
        seed_value = int(SEED)

    features_path, events_path, session_meta, source = _select_session_paths(args)

    if not features_path:
        print(
            "No features file found. Run Step 1 to create a new session, then re-run 1b_extract_windows.py."
        )
        raise SystemExit(2)
    if not features_path.exists():
        print(f"No features file found at {features_path}")
        raise SystemExit(2)

    if LABEL_GATED and (not events_path or not events_path.exists()):
        print(
            "No events file found. Provide --events PATH or run Step 1 to create events before extraction."
        )
        raise SystemExit(2)

    inferred_subject, inferred_session = infer_subject_session_from_features_path(
        features_path
    )

    # Start with filename-derived values (most reliable)
    subject_id_value = inferred_subject
    session_id_value = inferred_session

    experiment_hash_value = None
    if session_meta:
        experiment_hash_value = (
            session_meta.get("experiment_hash") or experiment_hash_value
        )

    # Root meta is only used if it matches the inferred subject/session
    root_meta = load_root_session_meta()
    if root_meta:
        root_subject = root_meta.get("subject_id")
        root_session = root_meta.get("session_id")
        root_exp = root_meta.get("experiment_hash")

        root_matches = True
        if inferred_subject and root_subject and root_subject != inferred_subject:
            root_matches = False
        if inferred_session and root_session and root_session != inferred_session:
            root_matches = False

        if root_matches:
            if not experiment_hash_value and root_exp:
                experiment_hash_value = root_exp

    # If still missing, use session_meta hash (already attempted) or UNKNOWN
    if not experiment_hash_value:
        experiment_hash_value = "UNKNOWN"

    if not subject_id_value:
        subject_id_value = "UNKNOWN"
    if session_id_value is None:
        session_id_value = ""

    events_time_shift_s = _infer_events_shift_s(events_path)

    target_fs = (
        float(args.target_fs)
        if args.target_fs is not None
        else float(session_meta.get("sampling_rate", TARGET_FS_DEFAULT))
    )
    window_sec = (
        float(session_meta.get("window_sec", WINDOW_SEC))
        if session_meta
        else WINDOW_SEC
    )
    window_samples = int(round(window_sec * target_fs))
    if window_samples <= 0:
        print(
            f"Invalid window_samples={window_samples}; check window_sec={window_sec} and target_fs={target_fs}."
        )
        raise SystemExit(2)

    min_overlap_sec = float(MIN_OVERLAP_RATIO) * float(window_sec)
    timebase_version = (
        session_meta.get("timebase_version")
        or session_meta.get("timebase")
        or "unknown"
    )

    print(f"Session selection source: {source}")
    print(f"Using features file: {features_path}")
    print(f"Using events file: {events_path}")
    print(f"Derived subject_id: {subject_id_value}")
    print(f"Derived experiment_hash: {experiment_hash_value}")
    print(f"Derived session_id: {session_id_value}")
    print(f"Events time shift (s): {events_time_shift_s}")
    print(f"Target window rate: {target_fs} Hz ({window_samples} samples/window)")
    print(f"Interpolation policy: {INTERPOLATION_POLICY}, dedupe: {DEDUP_POLICY}")
    print(f"Timebase version: {timebase_version}")

    # ===== LOAD DATA =====
    times, signal, channel_cols = _load_features(features_path)
    events = _load_events(events_path, session_meta=session_meta)

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

    # Guard band boundaries (movement events only)
    movement_boundaries: List[float] = []
    for e in events:
        if e.get("type") == "artifact":
            continue
        if int(e.get("action_id", ACTION_REST)) != int(ACTION_REST):
            movement_boundaries.append(float(e["onset_s"]))
            movement_boundaries.append(float(e["end_s"]))
    movement_boundaries_arr = np.array(sorted(set(movement_boundaries)), dtype=float)

    def in_guard_band(window_start: float, window_end: float) -> bool:
        if movement_boundaries_arr.size == 0:
            return False
        mid = 0.5 * (window_start + window_end)
        return bool(np.any(np.abs(movement_boundaries_arr - mid) <= GUARD_BAND_SEC))

    # Baseline REST allow-list (first N rest events)
    baseline_rest_events: List[Dict[str, Any]] = []
    baseline_rest_event_indices: set = set()
    if KEEP_BASELINE_REST_EVENTS > 0:
        rest_events = [
            e
            for e in events
            if int(e.get("action_id", ACTION_REST)) == int(ACTION_REST)
            and e.get("type") == "rest"
        ]
        rest_events.sort(key=lambda x: float(x["onset_s"]))
        baseline_rest_events = rest_events[: int(KEEP_BASELINE_REST_EVENTS)]
        baseline_rest_event_indices = {
            e.get("event_index") for e in baseline_rest_events
        }

    # ===== WINDOW LOOP =====
    windows: List[Dict[str, Any]] = []
    sequence_windows: List[np.ndarray] = []
    action_labels: List[int] = []
    finger_labels: List[int] = []
    window_starts: List[float] = []
    window_ends: List[float] = []
    confidence_hints: List[float] = []
    artifact_flags: List[int] = []
    gap_flags: List[int] = []

    # QA/meta arrays
    assigned_event_types: List[str] = []
    overlap_seconds: List[float] = []
    overlap_fracs: List[float] = []
    event_onsets_out: List[float] = []
    event_durations_out: List[float] = []
    event_sources: List[str] = []
    session_modes: List[str] = []
    trial_ids: List[int] = []
    block_ids: List[int] = []

    start_time = float(times[0])
    end_time = float(times[-1])
    last_start = end_time - float(window_sec)
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
    drop_ambiguous = 0
    drop_rest_subsample = 0
    drop_rest_cap = 0

    for window_start in window_starts_grid:
        total_windows += 1
        window_end = float(window_start + float(window_sec))

        if GUARD_BAND_SEC > 0 and in_guard_band(window_start, window_end):
            drop_guard_band += 1
            continue

        mask_pad = (times >= (window_start - PAD_SEC)) & (
            times <= (window_end + PAD_SEC)
        )
        if not np.any(mask_pad):
            drop_short_segment += 1
            continue

        times_pad = times[mask_pad]
        signal_pad = signal[mask_pad]

        core_count = int(np.sum((times_pad >= window_start) & (times_pad < window_end)))
        if core_count < 2 or times_pad.size < 2:
            drop_short_segment += 1
            continue

        max_dt = (
            float(np.max(np.diff(times_pad))) if times_pad.size >= 2 else float("inf")
        )
        gap_flag = int(max_dt > GAP_THRESHOLD_SEC)
        if gap_flag and not args.allow_gaps:
            drop_gap += 1
            continue

        # Overlaps with events
        overlapping = []
        any_artifact_flag = 0

        for e in events:
            ov = overlap_s(
                window_start, window_end, float(e["onset_s"]), float(e["end_s"])
            )
            if ov <= 0:
                continue

            ov_frac = ov / float(window_sec)

            if e.get("type") == "artifact" and ov_frac >= ARTIFACT_MIN_OVERLAP_FRAC:
                any_artifact_flag = 1
                break

            overlapping.append((ov, ov_frac, e))

        if any_artifact_flag:
            drop_artifact += 1
            continue

        # Default labels
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
            if is_baseline_rest_window(
                window_start, window_end, baseline_rest_events, min_overlap_sec
            ):
                assigned_type = "baseline_rest"
            else:
                drop_no_overlap += 1
                continue

        if overlapping:
            best_selection, ambiguous = _select_best_overlap(overlapping)
            if ambiguous:
                drop_ambiguous += 1
                continue
            if best_selection is None:
                drop_no_overlap += 1
                continue
            best_ov, best_ov_frac, best = best_selection
            best_event_index = best.get("event_index")

            if best.get("type") == "artifact":
                artifact_flag = 1
            else:
                if best_ov >= min_overlap_sec:
                    is_rest = int(best.get("action_id", ACTION_REST)) == int(
                        ACTION_REST
                    )
                    if is_rest and best_event_index not in baseline_rest_event_indices:
                        drop_no_overlap += 1
                        continue
                    action_id = int(best.get("action_id", ACTION_REST))
                    finger_id = int(best.get("finger_id", FINGER_NONE))
                    confidence_hint = best.get("confidence", np.nan)
                    if is_rest and best_event_index in baseline_rest_event_indices:
                        assigned_type = "baseline_rest"
                    else:
                        assigned_type = best.get("type", "") or "event"
                    best_onset = float(best.get("onset_s", np.nan))
                    best_dur = float(best.get("duration_s", np.nan))
                    best_source = str(best.get("source", ""))
                    best_session_mode = str(best.get("session_mode", ""))
                    best_trial_id = int(best.get("trial_id", 0))
                    best_block_id = int(best.get("block_id", 0))
                elif int(best.get("action_id", ACTION_REST)) == int(ACTION_REST):
                    if LABEL_GATED:
                        if (
                            best_event_index in baseline_rest_event_indices
                            and is_baseline_rest_window(
                                window_start,
                                window_end,
                                baseline_rest_events,
                                min_overlap_sec,
                            )
                        ):
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

        # Resample to fixed window_samples
        grid = np.linspace(window_start, window_end, window_samples, endpoint=False)
        segment = np.empty((window_samples, signal.shape[1]), dtype=float)
        for ch_idx in range(signal.shape[1]):
            segment[:, ch_idx] = np.interp(grid, times_pad, signal_pad[:, ch_idx])

        features = segment.mean(axis=0)

        # collect
        sequence_windows.append(segment.astype(np.float32))
        action_labels.append(int(action_id))
        finger_labels.append(int(finger_id))

        window_starts.append(float(window_start))
        window_ends.append(float(window_end))
        confidence_hints.append(
            float(confidence_hint) if pd.notna(confidence_hint) else np.nan
        )
        artifact_flags.append(int(artifact_flag))
        gap_flags.append(int(gap_flag))

        assigned_event_types.append(str(assigned_type))
        overlap_seconds.append(float(best_ov))
        overlap_fracs.append(float(best_ov_frac))
        event_onsets_out.append(
            float(best_onset) if np.isfinite(best_onset) else np.nan
        )
        event_durations_out.append(float(best_dur) if np.isfinite(best_dur) else np.nan)
        event_sources.append(str(best_source) if best_source is not None else "")
        session_modes.append(
            str(best_session_mode) if best_session_mode is not None else ""
        )
        trial_ids.append(int(best_trial_id))
        block_ids.append(int(best_block_id))

        windows.append(
            {
                "ch1": float(features[0]) if signal.shape[1] > 0 else 0.0,
                "ch2": float(features[1]) if signal.shape[1] > 1 else 0.0,
                "ch3": float(features[2]) if signal.shape[1] > 2 else 0.0,
                "ch4": float(features[3]) if signal.shape[1] > 3 else 0.0,
                "action_id": int(action_id),
                "finger_id": int(finger_id),
                "subject_id": str(subject_id_value),
                "experiment_hash": str(experiment_hash_value),
                "session_id": str(session_id_value),
                "window_start": float(window_start),
                "window_end": float(window_end),
                "confidence_hint": float(confidence_hint)
                if pd.notna(confidence_hint)
                else np.nan,
                "artifact_flag": int(artifact_flag),
                "gap_flag": int(gap_flag),
                "assigned_event_type": str(assigned_type),
                "overlap_s": float(best_ov),
                "overlap_frac": float(best_ov_frac),
                "event_onset_s": float(best_onset)
                if np.isfinite(best_onset)
                else np.nan,
                "event_duration_s": float(best_dur)
                if np.isfinite(best_dur)
                else np.nan,
                "event_source": str(best_source),
                "session_mode": str(best_session_mode),
                "trial_id": int(best_trial_id),
                "block_id": int(best_block_id),
            }
        )
        kept_windows += 1

    # ===== REST SUBSAMPLE/CAP =====
    rest_indices = [
        i for i, a in enumerate(action_labels) if int(a) == int(ACTION_REST)
    ]
    non_rest_count = int(len(action_labels) - len(rest_indices))
    rest_keep, drop_rest_subsample, drop_rest_cap = _select_rest_keep_indices(
        rest_indices,
        non_rest_count,
        float(REST_SUBSAMPLE_PROB),
        int(seed_value),
        REST_MAX_WINDOWS,
    )
    rest_kept = len(rest_indices)
    rest_dropped = 0

    if rest_indices and (len(rest_keep) != len(rest_indices)):
        rest_kept = len(rest_keep)
        rest_dropped = len(rest_indices) - rest_kept
        keep_mask = np.ones(len(action_labels), dtype=bool)
        keep_mask[rest_indices] = False
        if rest_keep:
            keep_mask[np.array(rest_keep, dtype=int)] = True

        sequence_windows = _filter_by_mask(sequence_windows, keep_mask)
        action_labels = _filter_by_mask(action_labels, keep_mask)
        finger_labels = _filter_by_mask(finger_labels, keep_mask)
        window_starts = _filter_by_mask(window_starts, keep_mask)
        window_ends = _filter_by_mask(window_ends, keep_mask)
        confidence_hints = _filter_by_mask(confidence_hints, keep_mask)
        artifact_flags = _filter_by_mask(artifact_flags, keep_mask)
        gap_flags = _filter_by_mask(gap_flags, keep_mask)

        assigned_event_types = _filter_by_mask(assigned_event_types, keep_mask)
        overlap_seconds = _filter_by_mask(overlap_seconds, keep_mask)
        overlap_fracs = _filter_by_mask(overlap_fracs, keep_mask)
        event_onsets_out = _filter_by_mask(event_onsets_out, keep_mask)
        event_durations_out = _filter_by_mask(event_durations_out, keep_mask)
        event_sources = _filter_by_mask(event_sources, keep_mask)
        session_modes = _filter_by_mask(session_modes, keep_mask)
        trial_ids = _filter_by_mask(trial_ids, keep_mask)
        block_ids = _filter_by_mask(block_ids, keep_mask)

        windows = _filter_by_mask(windows, keep_mask)
        kept_windows = len(action_labels)

    # ===== SAVE CSV =====
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
        f"invalid label: {drop_invalid_label}, short segment: {drop_short_segment}, gap: {drop_gap}, "
        f"ambiguous: {drop_ambiguous}, rest-subsample: {drop_rest_subsample}, rest-cap: {drop_rest_cap}"
    )
    print(f"Kept class distribution (action_id): {action_dist}")
    print(f"Kept class distribution (finger_id): {finger_dist}")
    if KEEP_BASELINE_REST_EVENTS == 0:
        print("Sanity: KEEP_BASELINE_REST_EVENTS=0 → no REST windows are kept.")

    report_payload = {
        "total_windows": int(total_windows),
        "kept_windows": int(kept_windows),
        "drop_counts": {
            "no_overlap": int(drop_no_overlap),
            "artifact": int(drop_artifact),
            "guard_band": int(drop_guard_band),
            "invalid_label": int(drop_invalid_label),
            "short_segment": int(drop_short_segment),
            "gap": int(drop_gap),
            "ambiguous": int(drop_ambiguous),
            "rest_subsample": int(drop_rest_subsample),
            "rest_cap": int(drop_rest_cap),
        },
        "rest_policy": {
            "keep_baseline_events": int(KEEP_BASELINE_REST_EVENTS),
            "subsample_prob": float(REST_SUBSAMPLE_PROB),
            "seed": int(seed_value),
            "rest_max_windows": int(REST_MAX_WINDOWS) if REST_MAX_WINDOWS else None,
            "baseline_rest_event_indices_count": int(len(baseline_rest_event_indices)),
            "rest_kept": int(rest_kept),
            "rest_dropped": int(rest_dropped),
        },
        "timebase_version": str(timebase_version),
        "subject_id": str(subject_id_value),
        "session_id": str(session_id_value),
        "events_path": str(events_path),
        "features_path": str(features_path),
    }
    report_path = Path("extraction_report.json")
    report_path.write_text(json.dumps(report_payload, indent=2))
    print(f"✅ Saved extraction report → {report_path}")

    # ===== SAVE NPZ =====
    if not sequence_windows:
        print("⚠ No sequence windows produced; check features/events alignment.")
        return 0

    X = np.stack(sequence_windows).astype(np.float32)
    n_kept = X.shape[0]
    gap_policy = "allow_gaps" if args.allow_gaps else "strict_drop"

    # Contract-critical arrays for Step 2 filtering (NO TRUNCATION)
    print(f"[SANITY] subject_id_value used for NPZ: {subject_id_value!r}")

    subject_id_arr = _uarr_fill(n_kept, str(subject_id_value))
    experiment_hash_arr = _uarr_fill(n_kept, str(experiment_hash_value))
    session_id_arr = _uarr_fill(n_kept, str(session_id_value or ""))
    source_features_path_arr = _uarr_fill(n_kept, str(features_path))
    source_events_path_arr = _uarr_fill(n_kept, str(events_path))

    # Non-fixed-length strings (safe max-length dtype)
    channel_names_arr = _uarr_from_list([str(c) for c in channel_cols])

    # config JSON can be long → do NOT store as dtype="U" without sizing
    config_json = json.dumps(
        {
            "min_overlap_ratio": float(MIN_OVERLAP_RATIO),
            "guard_band_sec": float(GUARD_BAND_SEC),
            "artifact_min_overlap_frac": float(ARTIFACT_MIN_OVERLAP_FRAC),
            "window_sec": float(window_sec),
            "step_sec": float(STEP_SEC),
            "fs": float(target_fs),
            "source_fs": int(SOURCE_FS_DEFAULT),
            "target_fs": float(target_fs),
            "pad_sec": float(PAD_SEC),
            "gap_threshold_sec": float(GAP_THRESHOLD_SEC),
            "gap_policy": str(gap_policy),
            "interpolation_policy": str(INTERPOLATION_POLICY),
            "dedupe_policy": str(DEDUP_POLICY),
            "rest_subsample_prob": float(REST_SUBSAMPLE_PROB),
            "rest_subsample_seed": int(seed_value),
            "rest_max_windows": int(REST_MAX_WINDOWS) if REST_MAX_WINDOWS else None,
        }
    )
    config_arr = np.array([config_json], dtype=_u_dtype_for(config_json))

    npz_payload = dict(
        X=X,
        y_action=np.array(action_labels, dtype=np.int64),
        y_finger=np.array(finger_labels, dtype=np.int64),
        subject_id=subject_id_arr,
        experiment_hash=experiment_hash_arr,
        session_id=session_id_arr,
        window_start=np.array(window_starts, dtype=np.float32),
        window_end=np.array(window_ends, dtype=np.float32),
        trial_id=np.array(trial_ids, dtype=np.int64),
        block_id=np.array(block_ids, dtype=np.int64),
        source_features_path=source_features_path_arr,
        source_events_path=source_events_path_arr,
        events_time_shift_s=np.array([events_time_shift_s], dtype=np.float32),
        confidence_hint=np.array(confidence_hints, dtype=np.float32),
        artifact_flag=np.array(artifact_flags, dtype=np.int64),
        gap_flag=np.array(gap_flags, dtype=np.int64),
        assigned_event_type=_uarr_from_list(assigned_event_types),
        overlap_s=np.array(overlap_seconds, dtype=np.float32),
        overlap_frac=np.array(overlap_fracs, dtype=np.float32),
        event_onset_s=np.array(event_onsets_out, dtype=np.float32),
        event_duration_s=np.array(event_durations_out, dtype=np.float32),
        event_source=_uarr_from_list(event_sources),
        session_mode=_uarr_from_list(session_modes),
        fs=np.array(int(round(target_fs)), dtype=np.int64),
        target_fs=np.array(float(target_fs), dtype=np.float32),
        window_sec=np.array(float(window_sec), dtype=np.float32),
        step_sec=np.array(float(STEP_SEC), dtype=np.float32),
        channel_names=channel_names_arr,
        timebase_version=np.array(
            str(timebase_version), dtype=_u_dtype_for(str(timebase_version))
        ),
        interpolation_policy=np.array(
            INTERPOLATION_POLICY, dtype=_u_dtype_for(INTERPOLATION_POLICY)
        ),
        gap_policy=np.array(gap_policy, dtype=_u_dtype_for(gap_policy)),
        features_path=np.array(
            str(features_path), dtype=_u_dtype_for(str(features_path))
        ),
        events_path=np.array(str(events_path), dtype=_u_dtype_for(str(events_path))),
        config=config_arr,
    )

    np.savez_compressed(OUT_NPZ, **npz_payload)
    print(f"✅ Saved sequence windows → {OUT_NPZ} with shape {X.shape}")

    u = np.unique(subject_id_arr.astype("U"))
    print(f"[SANITY] unique subject_id saved in NPZ: {u.tolist()}")
    print(f"N windows saved: {n_kept}")

    # Also save per-session NPZ in data/processed
    if subject_id_value and subject_id_value != "UNKNOWN" and session_id_value:
        session_dir = ROOT_DIR / "data/processed"
        session_dir.mkdir(parents=True, exist_ok=True)
        session_name = f"{subject_id_value}_{session_id_value}_eeg_windows.npz"
        session_path = _next_available_path(session_dir / session_name)
        np.savez_compressed(session_path, **npz_payload)
        print(f"✅ Saved session windows → {session_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(1)
