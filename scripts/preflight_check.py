#!/usr/bin/env python3
"""Preflight checks for EEG acquisition readiness."""

import argparse
from pathlib import Path

import pandas as pd

REQUIRED_EVENT_COLS = {"onset_s", "duration_s", "action_id", "finger_id"}
OPTIONAL_EVENT_COLS = {"type", "channel", "confidence", "notes", "source", "session_mode", "trial_id", "block_id"}
REQUIRED_FEATURE_COLS = {"ch1", "ch2", "ch3", "ch4"}


def check_required_files(mode: str) -> bool:
    ok = True
    events_path = Path("events.csv")
    features_path = Path("eeg_features.csv")
    model_path = Path("finger_action_model.pt")
    scaler_path = Path("scaler.save")

    if not events_path.exists():
        print("❌ Missing events.csv")
        ok = False
    if not features_path.exists():
        print("❌ Missing eeg_features.csv")
        ok = False

    if mode in {"demo", "eval"}:
        if not model_path.exists():
            print("⚠ Model weights missing: finger_action_model.pt")
        if not scaler_path.exists():
            print("⚠ Normalizer missing: scaler.save")

    return ok


def check_events_schema() -> bool:
    path = Path("events.csv")
    if not path.exists():
        return False
    df = pd.read_csv(path)
    cols = set(df.columns)
    missing = REQUIRED_EVENT_COLS - cols
    if missing:
        print(f"❌ events.csv missing required columns: {sorted(missing)}")
        return False
    optional_missing = OPTIONAL_EVENT_COLS - cols
    if optional_missing:
        print(f"ℹ Optional event columns not present: {sorted(optional_missing)}")
    return True


def check_features_schema() -> bool:
    path = Path("eeg_features.csv")
    if not path.exists():
        return False
    df = pd.read_csv(path)
    cols = set(df.columns)
    missing = REQUIRED_FEATURE_COLS - cols
    if missing:
        print(f"❌ eeg_features.csv missing required columns: {sorted(missing)}")
        return False
    if "time_s" not in cols:
        print("ℹ eeg_features.csv missing time_s; will infer from sampling rate")
    return True


def print_balance_stats() -> None:
    path = Path("eeg_windows.csv")
    if not path.exists():
        print("ℹ eeg_windows.csv not found; run 1b_extract_windows.py to generate balance stats")
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
    parser.add_argument("--mode", choices=["acquire", "demo", "eval"], default="acquire")
    args = parser.parse_args()

    ok = True
    ok &= check_required_files(args.mode)
    ok &= check_events_schema()
    ok &= check_features_schema()
    print_balance_stats()

    if not ok:
        raise SystemExit(1)
    print("\n✅ Preflight checks passed")


if __name__ == "__main__":
    main()
