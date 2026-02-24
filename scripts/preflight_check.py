#!/usr/bin/env python3
"""Preflight checks for EEG acquisition readiness."""

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_EVENT_COLS = {"onset_s", "duration_s", "action_id", "finger_id"}
OPTIONAL_EVENT_COLS = {
    "type",
    "channel",
    "confidence",
    "notes",
    "source",
    "session_mode",
    "trial_id",
    "block_id",
}
REQUIRED_FEATURE_COLS = {"ch1", "ch2", "ch3", "ch4"}
ALT_FEATURE_COLS = {"TP9", "AF7", "AF8", "TP10"}


def load_session_meta() -> dict:
    meta_path = Path("session_meta.json")
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text())
    except Exception as exc:
        print(f"⚠️ Failed to read session_meta.json: {exc}")
        return {}


def load_events_df(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def resolve_paths():
    meta = load_session_meta()
    features_path = Path(meta.get("features_path", "eeg_features.csv"))
    events_path = Path(meta.get("events_path", "events.jsonl"))
    raw_path = Path(meta.get("raw_path", "raw"))
    return meta, features_path, events_path, raw_path


def check_required_files(mode: str) -> bool:
    ok = True
    _, features_path, events_path, raw_path = resolve_paths()
    model_path = Path("finger_action_model.pt")
    scaler_path = Path("scaler.npz")

    if not events_path.exists():
        print(f"❌ Missing events file: {events_path}")
        ok = False
    if not features_path.exists():
        print(f"❌ Missing features file: {features_path}")
        ok = False

    if raw_path.is_dir():
        if not any(raw_path.glob("eeg_raw_shard_*.npy")):
            print(f"❌ Missing raw shards in: {raw_path}")
            ok = False
    elif not raw_path.exists():
        print(f"❌ Missing raw file: {raw_path}")
        ok = False

    if mode in {"demo", "eval"}:
        if not model_path.exists():
            print("⚠ Model weights missing: finger_action_model.pt")
        if not scaler_path.exists():
            print("⚠ Normalizer missing: scaler.npz")

    return ok


def check_events_schema() -> bool:
    _, _, events_path, _ = resolve_paths()
    if not events_path.exists():
        return False
    df = load_events_df(events_path)
    cols = set(df.columns)
    missing = REQUIRED_EVENT_COLS - cols
    if missing:
        print(f"❌ Events file missing required columns: {sorted(missing)}")
        return False
    optional_missing = OPTIONAL_EVENT_COLS - cols
    if optional_missing:
        print(f"ℹ Optional event columns not present: {sorted(optional_missing)}")
    return True


def check_features_schema() -> bool:
    _, features_path, _, _ = resolve_paths()
    if not features_path.exists():
        return False
    df = pd.read_csv(features_path)
    cols = set(df.columns)
    missing = REQUIRED_FEATURE_COLS - cols
    alt_missing = ALT_FEATURE_COLS - cols
    if missing and alt_missing:
        print(f"❌ Features file missing required columns: {sorted(missing)}")
        return False
    if "time_s" not in cols:
        print("ℹ Features file missing time_s; will infer from sampling rate")
    return True


def check_time_alignment() -> bool:
    _, features_path, events_path, _ = resolve_paths()
    if not features_path.exists() or not events_path.exists():
        return False

    df_feat = pd.read_csv(features_path)
    if "time_s" not in df_feat.columns:
        print("⚠️ time_s not found; skipping alignment checks")
        return True

    time_s = df_feat["time_s"].astype(float).to_numpy()
    if time_s.size == 0:
        print("⚠️ Features file is empty")
        return False

    diffs = time_s[1:] - time_s[:-1]
    if (diffs < -1e-6).any():
        print("❌ time_s is not monotonic increasing")
        return False

    df_events = load_events_df(events_path)
    if df_events.empty:
        print("ℹ No events recorded; skipping event alignment check")
        return True

    event_end = (
        df_events["onset_s"].astype(float) + df_events["duration_s"].astype(float)
    ).max()
    if event_end > time_s.max() + 1.0:
        print(
            f"❌ Event end ({event_end:.3f}s) exceeds feature time range ({time_s.max():.3f}s)"
        )
        return False

    return True


def check_feature_row_lengths() -> bool:
    _, features_path, _, _ = resolve_paths()
    if not features_path.exists():
        return False
    with features_path.open() as f:
        header = f.readline().strip().split(",")
        expected = len(header)
        for i, line in enumerate(f, start=2):
            if not line.strip():
                continue
            cols = line.strip().split(",")
            if len(cols) != expected:
                print(
                    f"❌ Row {i} in {features_path} has {len(cols)} columns, expected {expected}"
                )
                return False
            if i > 2000:
                break
    return True


def print_balance_stats() -> None:
    path = Path("eeg_windows.csv")
    if not path.exists():
        print(
            "ℹ eeg_windows.csv not found; run 1b_extract_windows.py to generate balance stats"
        )
        return
    df = pd.read_csv(path)
    if "action_id" in df.columns:
        print("\nAction balance:")
        print(df["action_id"].value_counts().sort_index())
    if "finger_id" in df.columns:
        print("\nFinger balance:")
        print(df["finger_id"].value_counts().sort_index())
    if "session_mode" in df.columns:
        print("\nSession mode balance:")
        print(df["session_mode"].fillna("UNKNOWN").value_counts())


def main():
    parser = argparse.ArgumentParser(description="Preflight data readiness check")
    parser.add_argument(
        "--mode", choices=["acquire", "demo", "eval"], default="acquire"
    )
    args = parser.parse_args()

    ok = True
    ok &= check_required_files(args.mode)
    ok &= check_events_schema()
    ok &= check_features_schema()
    ok &= check_feature_row_lengths()
    ok &= check_time_alignment()
    print_balance_stats()

    if not ok:
        raise SystemExit(1)
    print("\n✅ Preflight checks passed")


if __name__ == "__main__":
    main()
