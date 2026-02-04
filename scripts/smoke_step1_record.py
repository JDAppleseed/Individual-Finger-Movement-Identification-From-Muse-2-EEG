from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pylsl import StreamOutlet


def build_outlet(name: str = "Muse2-EEG", stype: str = "EEG") -> "StreamOutlet":
    from pylsl import StreamInfo, StreamOutlet

    info = StreamInfo(name, stype, 4, 256, "float32", "smoke_step1")
    desc = info.desc()
    channels = desc.append_child("channels")
    for label in ["TP9", "AF7", "AF8", "TP10"]:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    return StreamOutlet(info)


def push_samples(outlet: Any, stop_event: threading.Event, duration_s: float) -> None:
    start = time.monotonic()
    while time.monotonic() - start < duration_s and not stop_event.is_set():
        sample = np.random.randn(4).astype(np.float32).tolist()
        outlet.push_sample(sample)
        time.sleep(1.0 / 256.0)


def _latest_session_dir(sessions_root: Path) -> Path:
    candidates = [p for p in sessions_root.iterdir() if p.is_dir()]
    if not candidates:
        raise SystemExit(f"No session dirs found under {sessions_root}")
    candidates.sort(key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def _append_synthetic_event(session_dir: Path, *, onset_s: float, duration_s: float) -> None:
    events_dir = session_dir / "events"
    events_jsonl = events_dir / "events.jsonl"
    events_csv = events_dir / "events.csv"
    end_s = float(onset_s + duration_s)
    event = {
        "event_id": 0,
        "event_index": 0,
        "onset_s": float(onset_s),
        "duration_s": float(duration_s),
        "end_s": float(end_s),
        "type": "open",
        "finger_id": 0,
        "action_id": 1,
        "confidence": 1.0,
        "source": "smoke",
        "notes": "synthetic event (non-interactive smoke test)",
        "session_mode": "train_record",
        "trial_id": 0,
        "block_id": 0,
    }
    events_dir.mkdir(parents=True, exist_ok=True)
    events_jsonl.touch(exist_ok=True)
    with events_jsonl.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")

    # Keep inspection CSVs in sync (best-effort).
    if events_csv.exists():
        with events_csv.open("a", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    f"{onset_s:.6f}",
                    f"{duration_s:.6f}",
                    event["type"],
                    "n/a",
                    f"{event['confidence']:.3f}",
                    event["notes"],
                    int(event["finger_id"]),
                    int(event["action_id"]),
                    int(event["trial_id"]),
                    int(event["block_id"]),
                    event["source"],
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="SMOKE", help="Project name under Projects/")
    parser.add_argument("--subject", type=str, default="SMOKE", help="Subject ID under Projects/<project>/subjects/")
    parser.add_argument("--duration-s", type=float, default=2.0, help="Step 1 capture duration")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Also run Step 2 training + Step 3 evaluation + Step 4 reports (requires torch/matplotlib/etc).",
    )
    args = parser.parse_args()

    # Ensure pylsl can locate liblsl on macOS Homebrew installs.
    if not os.environ.get("PYLSL_LIB"):
        candidate = Path("/opt/homebrew/Frameworks/lsl.framework/lsl")
        if candidate.exists():
            os.environ["PYLSL_LIB"] = str(candidate)

    repo_root = Path(__file__).resolve().parents[1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"{args.subject}_{ts}"
    sessions_root = (
        repo_root / "Projects" / args.project / "subjects" / args.subject / "sessions"
    )
    sessions_root.mkdir(parents=True, exist_ok=True)
    requested_session_dir = sessions_root / session_id

    outlet = build_outlet()
    stop_event = threading.Event()
    producer = threading.Thread(
        target=push_samples,
        args=(outlet, stop_event, float(args.duration_s) + 4.0),
        daemon=True,
    )
    producer.start()

    time.sleep(0.5)

    config_path = sessions_root / f"{session_id}_step1_smoke_config.json"
    config_path.write_text(
        json.dumps(
            {
                "project_name": args.project,
                "subject_id": args.subject,
                "session_id": session_id,
                "settings": {
                    "LSL_STREAM_NAME": "Muse2-EEG",
                    "LSL_STREAM_TYPE": "EEG",
                    "REQUIRED_LSL_LABELS": ["TP9", "AF7", "AF8", "TP10"],
                    "ENABLE_PLOT": False,
                    "EVENT_MARKING_ENABLED": False,
                    "HEARTBEAT_INTERVAL_S": 0.5,
                    "NO_SAMPLE_TIMEOUT_S": 2.0,
                    "WARMUP_SAMPLE_COUNT": 1,
                    "WARMUP_TIMEOUT_S": 2.0,
                },
            },
            indent=2,
        )
        + "\n"
    )

    cmd = [
        sys.executable,
        str(repo_root / "1_stream_and_record.py"),
        "--config",
        str(config_path),
        "--session-dir",
        str(requested_session_dir),
        "--stream-name",
        "Muse2-EEG",
        "--stream-type",
        "EEG",
        "--lsl-source-id",
        "smoke_step1",
        "--subject-id",
        str(args.subject),
        "--no-plot",
        "--no-event-marking",
        "--duration-s",
        str(float(args.duration_s)),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(repo_root))
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"Recording failed (exit {proc.returncode})")

    # Use the newest session dir under the expected root (handles collision suffixes).
    session_dir = _latest_session_dir(sessions_root)
    print(f"[smoke] session_dir={session_dir}")

    expected = [
        session_dir / "manifest.json",
        session_dir / "meta.json",
        session_dir / "raw" / "raw.csv",
        session_dir / "events" / "events.csv",
        session_dir / "events" / "events.jsonl",
        session_dir / "logs" / "step1.log",
        session_dir / "logs" / "resolved_settings.json",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise SystemExit("Missing expected artifacts:\n" + "\n".join(str(p) for p in missing))

    shard_candidates = sorted((session_dir / "raw").glob("eeg_raw_shard_*.npy"))
    if not shard_candidates:
        raise SystemExit("No raw shards found under session_dir/raw/")

    # Confirm raw CSV has data rows.
    raw_csv = session_dir / "raw" / "raw.csv"
    with raw_csv.open("r", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) < 2:
        raise SystemExit("Raw CSV has header only; no samples written")

    step1_log = (session_dir / "logs" / "step1.log").read_text(errors="ignore")
    if "[alive]" not in step1_log:
        raise SystemExit("Heartbeat lines missing from logs/step1.log")

    # Validate session manifest/meta/shards.
    validate_cmd = [
        sys.executable,
        "-m",
        "muse_streaming.validate_session",
        "--session",
        str(session_dir),
    ]
    validate_proc = subprocess.run(
        validate_cmd, capture_output=True, text=True, cwd=str(repo_root)
    )
    print(validate_proc.stdout)
    print(validate_proc.stderr, file=sys.stderr)
    if validate_proc.returncode != 0:
        raise SystemExit(f"validate_session failed (exit {validate_proc.returncode})")

    # Step 1 is non-interactive here; add a synthetic labeled event for Step 1b.
    _append_synthetic_event(session_dir, onset_s=0.2, duration_s=min(1.0, float(args.duration_s) * 0.8))

    step1b_cmd = [
        sys.executable,
        str(repo_root / "1b_extract_windows.py"),
        "--session-dir",
        str(session_dir),
    ]
    step1b_proc = subprocess.run(step1b_cmd, capture_output=True, text=True, cwd=str(repo_root))
    print(step1b_proc.stdout)
    print(step1b_proc.stderr, file=sys.stderr)
    if step1b_proc.returncode != 0:
        raise SystemExit(f"1b_extract_windows failed (exit {step1b_proc.returncode})")

    if not (session_dir / "processed" / "eeg_windows.npz").exists():
        raise SystemExit("Missing processed/eeg_windows.npz")

    if args.full:
        # Step 2 (train)
        train_cmd = [
            sys.executable,
            str(repo_root / "2_train_model.py"),
            "--session-dir",
            str(session_dir),
            "--subject-id",
            str(args.subject),
            "--epochs",
            "1",
            "--batch-size",
            "8",
        ]
        train_proc = subprocess.run(train_cmd, capture_output=True, text=True, cwd=str(repo_root))
        print(train_proc.stdout)
        print(train_proc.stderr, file=sys.stderr)
        if train_proc.returncode != 0:
            raise SystemExit(f"2_train_model failed (exit {train_proc.returncode})")

        models_root = session_dir / "processed" / "models"
        run_dirs = [p for p in models_root.iterdir() if p.is_dir()] if models_root.exists() else []
        if not run_dirs:
            raise SystemExit("No run dirs created under processed/models/")
        run_dirs.sort(key=lambda p: p.stat().st_mtime)
        run_dir = run_dirs[-1]
        if not (run_dir / "finger_action_model.pt").exists():
            raise SystemExit("Missing trained model: processed/models/<run_id>/finger_action_model.pt")
        if not (run_dir / "scaler.save").exists():
            raise SystemExit("Missing scaler: processed/models/<run_id>/scaler.save")

        # Step 3 (evaluate)
        eval_cmd = [sys.executable, str(repo_root / "3_evaluate_model.py"), "--session-dir", str(session_dir)]
        eval_proc = subprocess.run(eval_cmd, capture_output=True, text=True, cwd=str(repo_root))
        print(eval_proc.stdout)
        print(eval_proc.stderr, file=sys.stderr)
        if eval_proc.returncode != 0:
            raise SystemExit(f"3_evaluate_model failed (exit {eval_proc.returncode})")

        # Step 4 (reports)
        report_cmd = [sys.executable, str(repo_root / "4_generate_reports.py"), "--session-dir", str(session_dir)]
        report_proc = subprocess.run(report_cmd, capture_output=True, text=True, cwd=str(repo_root))
        print(report_proc.stdout)
        print(report_proc.stderr, file=sys.stderr)
        if report_proc.returncode != 0:
            raise SystemExit(f"4_generate_reports failed (exit {report_proc.returncode})")

    stop_event.set()
    producer.join(timeout=1.0)
    print("✅ Smoke pipeline check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
