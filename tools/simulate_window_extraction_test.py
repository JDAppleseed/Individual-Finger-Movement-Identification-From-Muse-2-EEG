#!/usr/bin/env python
"""
Deterministic simulation for time-based window extraction.
"""

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
from numpy.typing import NDArray


def _write_features(path: Path, times: NDArray[np.floating], lsl_start: float):
    lsl_ts = lsl_start + times
    ch1 = np.sin(2 * np.pi * 8.0 * times)
    ch2 = np.cos(2 * np.pi * 6.0 * times)
    ch3 = np.sin(2 * np.pi * 4.0 * times)
    ch4 = np.cos(2 * np.pi * 10.0 * times)
    noise = np.random.default_rng(123).normal(0, 0.05, size=times.shape)

    df = pd.DataFrame(
        {
            "lsl_timestamp": lsl_ts,
            "time_s": times,
            "ch1": ch1 + noise,
            "ch2": ch2 + noise,
            "ch3": ch3 + noise,
            "ch4": ch4 + noise,
        }
    )
    df.to_csv(path, index=False)


def _write_events(path: Path, lsl_start: float):
    events = [
        {
            "onset_s": 1.0,
            "duration_s": 0.4,
            "action_id": 1,
            "finger_id": 2,
            "type": "open",
            "trial_id": 1,
            "block_id": 0,
        },
        {
            "onset_s": 3.0,
            "duration_s": 0.4,
            "action_id": 2,
            "finger_id": 3,
            "type": "close",
            "trial_id": 2,
            "block_id": 0,
        },
    ]

    rows = []
    for e in events:
        onset_s = float(e["onset_s"])
        duration_s = float(e["duration_s"])
        onset_lsl = lsl_start + onset_s
        end_s = onset_s + duration_s
        end_lsl = lsl_start + end_s
        rows.append(
            [
                onset_lsl,
                onset_s,
                duration_s,
                end_lsl,
                end_s,
                e["type"],
                "n/a",
                "",
                "",
                e["finger_id"],
                e["action_id"],
                e["trial_id"],
                e["block_id"],
                "manual",
            ]
        )

    df = pd.DataFrame(
        rows,
        columns=[
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
    )
    df.to_csv(path, index=False)


def _generate_times(duration_s: float):
    rng = np.random.default_rng(42)
    times = [0.0]
    t = 0.0
    while t < duration_s:
        dt = (1.0 / 60.0) + rng.normal(0.0, 0.002)
        if rng.random() < 0.02:
            dt += 0.12
        dt = max(0.005, dt)
        t += dt
        times.append(t)
    return np.array(times, dtype=float)


def main():
    start = time.time()
    repo_root = Path(__file__).resolve().parents[1]
    extractor = repo_root / "1b_extract_windows.py"

    if not extractor.exists():
        print("ERROR: 1b_extract_windows.py not found")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        features_path = tmp_dir / "features.csv"
        events_path = tmp_dir / "events.csv"

        times = _generate_times(6.0)
        lsl_start = 1000.0
        _write_features(features_path, times, lsl_start)
        _write_events(events_path, lsl_start)

        session_meta = {
            "subject_id": "SIM",
            "session_id": "SIM001",
            "sampling_rate": 256,
            "window_sec": 0.25,
            "features_path": str(features_path),
            "events_path": str(events_path),
            "timebase_version": "absolute_v1",
            "complete": True,
        }
        (tmp_dir / "session_meta.json").write_text(json.dumps(session_meta, indent=2))

        cmd = [
            sys.executable,
            str(extractor),
            "--target-fs",
            "256",
        ]
        result = subprocess.run(cmd, cwd=tmp_dir, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stdout)
            print(result.stderr)
            print(f"ERROR: Extraction failed with code {result.returncode}")
            sys.exit(1)

        npz_path = tmp_dir / "eeg_windows.npz"
        if not npz_path.exists():
            print("ERROR: eeg_windows.npz not found")
            sys.exit(1)

        data = np.load(npz_path, allow_pickle=True)
        X = data["X"]
        if X.ndim != 3:
            print(f"ERROR: Unexpected X shape: {X.shape}")
            sys.exit(1)

        window_samples = int(round(0.25 * 256))
        if X.shape[1] != window_samples or X.shape[2] != 4:
            print(f"ERROR: Unexpected window shape: {X.shape}")
            sys.exit(1)

        if "timebase_version" not in data or "target_fs" not in data:
            print("ERROR: Missing timebase_version or target_fs metadata")
            sys.exit(1)

        elapsed = time.time() - start
        if elapsed > 5.0:
            print(f"ERROR: Simulation exceeded time limit: {elapsed:.2f}s")
            sys.exit(1)

        print(f"OK windows shape={X.shape}")


if __name__ == "__main__":
    main()
