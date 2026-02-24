#!/usr/bin/env python
"""
Legacy CSV alignment check between features/events on the absolute_v1 timebase.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

TIMEBASE_VERSION = "absolute_v1"


def _read_json(path: Path):
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_session_meta(
    session_meta_path: Optional[Path], features_path: Optional[Path]
):
    if session_meta_path:
        return _read_json(session_meta_path)
    if features_path and str(features_path).endswith("_eeg_features.csv"):
        meta_guess = features_path.with_name(
            features_path.name.replace("_eeg_features.csv", "_session_meta.json")
        )
        if meta_guess.exists():
            return _read_json(meta_guess)
    return {}


def _fs_stats(times: np.ndarray):
    times = times[np.isfinite(times)]
    if times.size < 2:
        return None
    times_sorted = np.sort(times)
    dt = np.diff(times_sorted)
    dt = dt[dt > 0]
    if dt.size == 0:
        return None
    median_dt = float(np.median(dt))
    mean_dt = float(np.mean(dt))
    p95_dt = float(np.percentile(dt, 95))
    stats = {
        "median_dt": median_dt,
        "mean_dt": mean_dt,
        "p95_dt": p95_dt,
        "median_fs": float(1.0 / median_dt) if median_dt > 0 else np.nan,
        "mean_fs": float(1.0 / mean_dt) if mean_dt > 0 else np.nan,
        "p95_fs": float(1.0 / p95_dt) if p95_dt > 0 else np.nan,
    }
    return stats


def _check_alignment(
    features_path: Path,
    events_path: Path,
    target_fs: float,
    session_meta_path: Optional[Path],
):
    try:
        features_df = pd.read_csv(features_path)
    except Exception as exc:
        print(f"ERROR: Failed to read features file: {features_path} ({exc})")
        return 1

    if "time_s" not in features_df.columns or (
        "lsl_timestamp" not in features_df.columns
        and "lsl_timestamp_mono" not in features_df.columns
    ):
        print(
            "ERROR: Features file missing required columns: time_s and lsl_timestamp(_mono)"
        )
        return 1

    times = features_df["time_s"].astype(float).to_numpy()
    lsl_col = (
        "lsl_timestamp_mono"
        if "lsl_timestamp_mono" in features_df.columns
        else "lsl_timestamp"
    )
    lsl_ts = features_df[lsl_col].astype(float).to_numpy()
    finite_mask = np.isfinite(times)
    times = times[finite_mask]
    lsl_ts = lsl_ts[finite_mask]

    if times.size < 2:
        print("ERROR: Not enough valid time_s samples in features file.")
        return 1

    non_increasing = int(np.sum(np.diff(times) <= 0))
    if non_increasing > 0:
        print(
            f"WARN: time_s is not strictly increasing ({non_increasing} non-increasing steps)."
        )

    fs_stats = _fs_stats(times)

    features_min = float(np.min(times))
    features_max = float(np.max(times))
    lsl_min = float(np.nanmin(lsl_ts)) if lsl_ts.size else np.nan
    lsl_max = float(np.nanmax(lsl_ts)) if lsl_ts.size else np.nan

    try:
        events_df = pd.read_csv(events_path)
    except Exception as exc:
        print(f"ERROR: Failed to read events file: {events_path} ({exc})")
        return 1

    if "onset_s" not in events_df.columns or "duration_s" not in events_df.columns:
        print("ERROR: Events file missing required columns: onset_s, duration_s")
        return 1

    onset_s = events_df["onset_s"].astype(float).to_numpy()
    duration_s = events_df["duration_s"].astype(float).to_numpy()
    end_s = (
        events_df["end_s"].astype(float).to_numpy()
        if "end_s" in events_df.columns
        else onset_s + duration_s
    )

    onset_lsl = (
        events_df["onset_lsl"].astype(float).to_numpy()
        if "onset_lsl" in events_df.columns
        else np.array([])
    )
    end_lsl = (
        events_df["end_lsl"].astype(float).to_numpy()
        if "end_lsl" in events_df.columns
        else np.array([])
    )

    negative_durations = int(np.sum(duration_s < 0))
    end_before_onset = int(np.sum(end_s < onset_s))

    events_min = float(np.nanmin(onset_s)) if onset_s.size else np.nan
    events_max = float(np.nanmax(end_s)) if end_s.size else np.nan
    events_lsl_min = float(np.nanmin(onset_lsl)) if onset_lsl.size else np.nan
    events_lsl_max = float(np.nanmax(end_lsl)) if end_lsl.size else np.nan

    if onset_s.size:
        inside_mask = (onset_s >= features_min) & (onset_s <= features_max)
        inside_ratio = float(np.mean(inside_mask))
    else:
        inside_ratio = 1.0

    warnings = []
    major = False

    meta = _load_session_meta(session_meta_path, features_path)
    timebase_version = meta.get("timebase_version") or meta.get("timebase")
    if not timebase_version:
        warnings.append("timebase_version missing")
    elif timebase_version != TIMEBASE_VERSION:
        warnings.append(f"timebase_version mismatch: {timebase_version}")

    outside_ratio = 1.0 - inside_ratio
    if outside_ratio >= 0.10 and onset_s.size:
        warnings.append(f"{outside_ratio:.1%} events outside feature range")
        major = True

    if negative_durations > 0 or end_before_onset > 0:
        major = True

    if non_increasing > 0:
        major = True

    if fs_stats and fs_stats["median_fs"] < 0.5 * float(target_fs):
        warnings.append(
            f"effective fs {fs_stats['median_fs']:.2f} Hz < 0.5 * target_fs ({target_fs:.2f})"
        )

    print("---- Features ----")
    print(f"time_s min/max: {features_min:.6f} .. {features_max:.6f}")
    print(f"lsl_timestamp min/max: {lsl_min:.6f} .. {lsl_max:.6f}")
    if fs_stats:
        print(
            "dt median/mean/p95: "
            f"{fs_stats['median_dt']:.6f} / {fs_stats['mean_dt']:.6f} / {fs_stats['p95_dt']:.6f} s"
        )
        print(
            "fs median/mean/p95: "
            f"{fs_stats['median_fs']:.2f} / {fs_stats['mean_fs']:.2f} / {fs_stats['p95_fs']:.2f} Hz"
        )

    print("---- Events ----")
    print(f"onset_s/end_s min/max: {events_min:.6f} .. {events_max:.6f}")
    if onset_lsl.size and end_lsl.size:
        print(
            f"onset_lsl/end_lsl min/max: {events_lsl_min:.6f} .. {events_lsl_max:.6f}"
        )
    else:
        print("onset_lsl/end_lsl min/max: n/a")

    print("---- Overlap ----")
    print(f"events inside feature range: {inside_ratio:.1%}")

    if warnings:
        print("---- Warnings ----")
        for w in warnings:
            print(f"WARN: {w}")

    if negative_durations > 0:
        print(f"ERROR: Negative durations: {negative_durations}")
    if end_before_onset > 0:
        print(f"ERROR: end_s before onset_s: {end_before_onset}")

    if major:
        return 2
    return 0


def _write_csv(path: Path, header: List[str], rows: List[List[object]]):
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(path, index=False)


def _run_self_test(target_fs: float):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        features_path = tmp_dir / "features.csv"
        events_path = tmp_dir / "events.csv"

        times = np.array([0.0, 0.1, 0.2], dtype=float)
        lsl = 1000.0 + times
        features_rows = [
            [lsl[i], times[i], 0.1, 0.2, 0.3, 0.4] for i in range(len(times))
        ]
        _write_csv(
            features_path,
            ["lsl_timestamp", "time_s", "ch1", "ch2", "ch3", "ch4"],
            features_rows,
        )

        events_rows = [
            [1001.0, 1.0, 0.1, 1001.1, 1.1, "open", "n/a", "", "", 1, 1, 1, 0, "manual"]
        ]
        _write_csv(
            events_path,
            [
                "onset_lsl",
                "onset_s",
                "duration_s",
                "end_lsl",
                "end_s",
                "type",
                "channel",
                "confidence",
                "notes",
                "finger_id",
                "action_id",
                "trial_id",
                "block_id",
                "source",
            ],
            events_rows,
        )

        code = _check_alignment(features_path, events_path, target_fs, None)
        if code == 2:
            print("SELF-TEST OK (exit=2)")
            return 2
        print(f"SELF-TEST FAILED: expected exit=2, got {code}")
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Check time alignment between features/events"
    )
    parser.add_argument(
        "--features",
        type=str,
        default=None,
        help="Features CSV path (legacy)",
    )
    parser.add_argument(
        "--events",
        type=str,
        default=None,
        help="Events CSV path (legacy)",
    )
    parser.add_argument(
        "--session-meta",
        type=str,
        default=None,
        help="Session metadata JSON (optional)",
    )
    parser.add_argument(
        "--target-fs",
        type=float,
        default=256.0,
        help="Target sampling rate for sanity checks",
    )
    parser.add_argument(
        "--self-test", action="store_true", help="Run a built-in misalignment self-test"
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_run_self_test(float(args.target_fs)))

    if not args.features or not args.events:
        print("ERROR: --features and --events are required (or use --self-test).")
        sys.exit(1)

    features_path = Path(args.features)
    events_path = Path(args.events)
    if not features_path.exists() or not events_path.exists():
        print("ERROR: Features/events paths do not exist.")
        sys.exit(1)

    session_meta_path = Path(args.session_meta) if args.session_meta else None
    code = _check_alignment(
        features_path, events_path, float(args.target_fs), session_meta_path
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
