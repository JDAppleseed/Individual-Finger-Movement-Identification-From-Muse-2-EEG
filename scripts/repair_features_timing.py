#!/usr/bin/env python3
"""
Repair eeg_features.csv timing so it can align with events.csv and be used for training.

What it does:
1) Loads eeg_features.csv even if it has "extra" columns vs header.
2) Builds a monotonic "time_s_fixed" by stitching segments when time_s resets backwards.
3) Loads events.csv and finds which stitched segment best overlaps event onsets.
4) Writes a cleaned features CSV that contains ONLY the best segment, with corrected time_s.

Outputs:
- <features_in>.__repaired__.csv  (default)
- Prints diagnostics about detected segments and chosen segment.

Usage:
  python scripts/repair_features_timing.py \
    --features eeg_features.csv \
    --events events.csv \
    --out eeg_features_repaired.csv

If your session_meta.json exists, you can then point features_path to the repaired file.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "lsl_timestamp",
    "time_s",
    "ch1",
    "ch2",
    "ch3",
    "ch4",
]


def robust_read_features_csv(path: Path) -> pd.DataFrame:
    """
    Read CSV that may have:
      - rows with more values than header columns
      - mixed-format appends
    Strategy:
      - read with python engine
      - keep only required columns that exist
      - coerce numeric types
      - drop rows missing critical numeric values
    """
    df = pd.read_csv(path, engine="python")

    # Keep only what we need for alignment/extraction
    keep = [c for c in REQUIRED_COLS if c in df.columns]
    if not keep:
        raise ValueError(
            f"No expected columns found in {path}. Columns: {list(df.columns)[:20]}"
        )

    df = df[keep].copy()

    # Coerce numeric
    for c in keep:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Must have at least lsl_timestamp + 4 channels
    must = [c for c in ["lsl_timestamp", "ch1", "ch2", "ch3", "ch4"] if c in df.columns]
    df = df.dropna(subset=must)

    # If time_s missing, reconstruct a naive one from sample index
    if "time_s" not in df.columns:
        # NOTE: If you have FS known you can replace this; but we prefer to use existing time_s when present.
        df["time_s"] = np.arange(len(df), dtype=float)

    # Sort by lsl_timestamp to preserve order if file got shuffled
    df = df.sort_values("lsl_timestamp").reset_index(drop=True)
    return df


def stitch_time_s(
    time_s: np.ndarray, eps: float = 1e-6
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    """
    Create a monotonic time base by detecting backward jumps (resets) and adding an offset.

    Returns:
      time_fixed: monotonic stitched time array
      segments: list of (start_idx, end_idx_exclusive, seg_min, seg_max) in the ORIGINAL index space
    """
    t = np.asarray(time_s, dtype=float).copy()
    if t.size == 0:
        return t, []

    # Identify reset points: where time decreases significantly
    dt = np.diff(t)
    reset_points = np.where(dt < -0.05)[0]  # 50ms backward jump -> new segment
    # Segment boundaries in index space
    boundaries = [0] + (reset_points + 1).tolist() + [len(t)]

    segments = []
    for i in range(len(boundaries) - 1):
        s = boundaries[i]
        e = boundaries[i + 1]
        seg = t[s:e]
        if seg.size == 0:
            continue
        segments.append((s, e, float(np.nanmin(seg)), float(np.nanmax(seg))))

    # Stitch: cumulative offset so each segment starts after previous ends
    t_fixed = np.empty_like(t)
    offset = 0.0
    prev_end = None

    for s, e, seg_min, seg_max in segments:
        seg = t[s:e].copy()

        # Normalize to start at 0 within segment (handles weird absolute values)
        seg = seg - seg[0]

        if prev_end is None:
            offset = 0.0
        else:
            # Place this segment after prev_end with a tiny gap
            offset = prev_end + eps

        seg_fixed = seg + offset

        # Enforce monotonic within segment (clamp tiny reversals)
        for k in range(1, len(seg_fixed)):
            if seg_fixed[k] < seg_fixed[k - 1]:
                seg_fixed[k] = seg_fixed[k - 1] + eps

        t_fixed[s:e] = seg_fixed
        prev_end = float(seg_fixed[-1])

    # Update segment ranges to reflect stitched coordinates
    stitched_segments = []
    for s, e, _, _ in segments:
        stitched_segments.append((s, e, float(t_fixed[s]), float(t_fixed[e - 1])))

    return t_fixed, stitched_segments


def choose_best_segment(
    segments: list[tuple[int, int, float, float]], event_onsets: np.ndarray
) -> int:
    """
    Pick segment whose stitched time range best covers event onset range.
    Returns index into `segments`.
    """
    if len(segments) == 0:
        raise ValueError("No segments detected.")
    if event_onsets.size == 0:
        # No events: choose the longest segment
        lengths = [seg[1] - seg[0] for seg in segments]
        return int(np.argmax(lengths))

    ev_min = float(np.nanmin(event_onsets))
    ev_max = float(np.nanmax(event_onsets))

    best_i = 0
    best_score = -1e18

    for i, (s, e, seg_min, seg_max) in enumerate(segments):
        # coverage overlap in time
        overlap = max(0.0, min(seg_max, ev_max) - max(seg_min, ev_min))
        seg_span = max(1e-9, seg_max - seg_min)
        ev_span = max(1e-9, ev_max - ev_min)

        # Score: maximize overlap fraction + prefer segments that fully cover events
        overlap_frac_ev = overlap / ev_span
        overlap_frac_seg = overlap / seg_span
        covers_all = 1.0 if (seg_min <= ev_min and seg_max >= ev_max) else 0.0

        score = 3.0 * overlap_frac_ev + 1.0 * overlap_frac_seg + 2.0 * covers_all
        # Slight preference for more samples
        score += 0.000001 * (e - s)

        if score > best_score:
            best_score = score
            best_i = i

    return best_i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=str, default="eeg_features.csv")
    ap.add_argument("--events", type=str, default="events.csv")
    ap.add_argument("--out", type=str, default="eeg_features_repaired.csv")
    args = ap.parse_args()

    features_path = Path(args.features)
    events_path = Path(args.events)
    out_path = Path(args.out)

    if not features_path.exists():
        raise FileNotFoundError(features_path)
    if not events_path.exists():
        raise FileNotFoundError(events_path)

    df = robust_read_features_csv(features_path)

    # Load events
    events_df = pd.read_csv(events_path)
    if "onset_s" not in events_df.columns:
        raise ValueError(f"{events_path} missing onset_s column.")
    event_onsets = (
        pd.to_numeric(events_df["onset_s"], errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )

    # Stitch timing
    t_fixed, segments = stitch_time_s(df["time_s"].to_numpy(dtype=float))
    df["time_s_fixed"] = t_fixed

    print("\n=== Detected stitched segments (index, time range) ===")
    for i, (s, e, t0, t1) in enumerate(segments):
        print(
            f"  seg[{i}] idx [{s}:{e}]  time_s_fixed: {t0:.3f} -> {t1:.3f}  (N={e - s})"
        )

    if event_onsets.size:
        print(
            f"\nEvents onset range: {float(event_onsets.min()):.3f} -> {float(event_onsets.max()):.3f} s"
        )
    else:
        print("\nNo events found in events.csv (onset_s empty).")

    best_i = choose_best_segment(segments, event_onsets)
    s, e, t0, t1 = segments[best_i]
    print(f"\n✅ Selected seg[{best_i}] idx [{s}:{e}] time_s_fixed {t0:.3f}->{t1:.3f}")

    # Slice to best segment only (this is the “pairing” fix)
    df_seg = df.iloc[s:e].copy()

    # Replace time_s with fixed monotonic time (and drop the helper column)
    df_seg["time_s"] = df_seg["time_s_fixed"]
    df_seg = df_seg.drop(columns=["time_s_fixed"])

    # Keep only columns Step 1b expects + whatever else is present but harmless
    # (If your Step 1b strictly reads ch1..ch4 + time_s, leaving extras is OK.)
    df_seg.to_csv(out_path, index=False)
    print(f"\n📝 Wrote repaired features to: {out_path}")
    print("\nNext step:")
    print(
        f"  python 1b_extract_windows.py   (make sure session_meta.json points features_path to {out_path} OR set RAW_FILE accordingly)\n"
    )


if __name__ == "__main__":
    main()
