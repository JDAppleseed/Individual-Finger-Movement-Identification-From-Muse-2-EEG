#!/usr/bin/env python3
"""
Trim an EEG features CSV to the time span covered by an events CSV (+/- margin).

Designed for this repo's outputs:

Features CSV (from 1_stream_and_record.py) includes:
  - time_s (float seconds, session-relative)
  - lsl_timestamp (float seconds, LSL clock)
  - ch1..ch4 (EEG channels after cleaning)
  - optional prediction/diagnostic columns (kept as-is)

Events CSV includes:
  - onset_s (float seconds, session-relative)
  - duration_s (float seconds)
  - action_id, finger_id, etc. (not required for trimming)

Trimming rule:
  keep features where time_s is within:
    [min(onset_s) - margin_s, max(onset_s + duration_s) + margin_s]

Usage:
  python scripts/trim_features_to_events.py --features path --events path --margin-s 2 --inplace
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


def _require_cols(df: pd.DataFrame, path: str, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"ERROR: {path} missing required columns: {missing}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--features", required=True, type=str, help="Features CSV (eeg_features.csv)"
    )
    p.add_argument("--events", required=True, type=str, help="Events CSV (events.csv)")
    p.add_argument(
        "--margin-s",
        type=float,
        default=0.0,
        help="Seconds to expand trim window on both sides",
    )
    p.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite features file (creates .bak backup)",
    )
    p.add_argument(
        "--out",
        type=str,
        default="",
        help="Optional output path (ignored if --inplace)",
    )
    args = p.parse_args()

    features_path = Path(args.features)
    events_path = Path(args.events)

    if not features_path.exists():
        raise SystemExit(f"ERROR: features file not found: {features_path}")
    if not events_path.exists():
        raise SystemExit(f"ERROR: events file not found: {events_path}")

    df_f = pd.read_csv(features_path)
    df_e = pd.read_csv(events_path)

    _require_cols(df_f, str(features_path), ["time_s"])
    _require_cols(df_e, str(events_path), ["onset_s", "duration_s"])

    # Drop NaNs defensively
    onset = pd.to_numeric(df_e["onset_s"], errors="coerce")
    dur = pd.to_numeric(df_e["duration_s"], errors="coerce").fillna(0.0)

    valid = onset.notna()
    if valid.sum() == 0:
        raise SystemExit(
            "ERROR: events file has no valid onset_s values to trim against"
        )

    onset = onset[valid]
    dur = dur[valid]

    t_start = float((onset.min()) - float(args.margin_s))
    t_end = float((onset + dur).max() + float(args.margin_s))

    # Ensure sane ordering
    if t_end <= t_start:
        raise SystemExit(
            f"ERROR: computed trim window invalid: start={t_start:.3f}, end={t_end:.3f}"
        )

    time_s = pd.to_numeric(df_f["time_s"], errors="coerce")
    keep = time_s.notna() & (time_s >= t_start) & (time_s <= t_end)

    n_before = len(df_f)
    df_out = df_f.loc[keep].reset_index(drop=True)
    n_after = len(df_out)

    # Decide output target
    if args.inplace:
        backup = features_path.with_suffix(features_path.suffix + ".bak")
        # Only create backup if it doesn't already exist to avoid overwriting evidence
        if not backup.exists():
            features_path.replace(backup)
        else:
            # If backup exists, write a second backup with increment
            i = 2
            while True:
                b2 = features_path.with_suffix(features_path.suffix + f".bak{i}")
                if not b2.exists():
                    features_path.replace(b2)
                    backup = b2
                    break
                i += 1
        out_path = features_path
        df_out.to_csv(out_path, index=False)
        print("✅ Trim complete (inplace).")
        print(f"   Backup: {backup}")
    else:
        out_path = (
            Path(args.out)
            if args.out
            else features_path.with_name(features_path.stem + "_TRIMMED.csv")
        )
        df_out.to_csv(out_path, index=False)
        print("✅ Trim complete.")
        print(f"   Output: {out_path}")

    print(
        f"   Window: [{t_start:.3f}, {t_end:.3f}] seconds (margin={float(args.margin_s):.3f}s)"
    )
    print(f"   Rows:   {n_before} -> {n_after}  (dropped {n_before - n_after})")

    # Quick sanity info
    if n_after == 0:
        print(
            "⚠️ WARNING: No feature rows remain after trimming. Check time bases / margin.",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
