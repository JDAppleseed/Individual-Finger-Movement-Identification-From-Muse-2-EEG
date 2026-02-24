from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from typing import Any, Optional, TYPE_CHECKING

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


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _run_cmd(cmd: list[str], *, cwd: Path, label: str) -> int:
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd))
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        print(f"[smoke] {label} failed (exit {proc.returncode})")
    return int(proc.returncode)


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
                    "WARMUP_TIMEOUT_S": 4.0,
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
    core_pass = True
    core_errors: list[str] = []
    full_status = "SKIP"
    session_dir: Optional[Path] = None

    if _run_cmd(cmd, cwd=repo_root, label="step1_record") != 0:
        core_pass = False
        core_errors.append("step1_record")

    if core_pass:
        try:
            session_dir = _latest_session_dir(sessions_root)
        except Exception as exc:
            core_pass = False
            core_errors.append(f"session_dir_resolve: {exc}")

    if session_dir:
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
            core_pass = False
            core_errors.append("missing_artifacts")
            print("Missing expected artifacts:\n" + "\n".join(str(p) for p in missing))

    if core_pass and session_dir:
        shard_candidates = sorted((session_dir / "raw").glob("eeg_raw_shard_*.npy"))
        if not shard_candidates:
            core_pass = False
            core_errors.append("no_raw_shards")
            print("No raw shards found under session_dir/raw/")

    if core_pass and session_dir:
        raw_csv = session_dir / "raw" / "raw.csv"
        with raw_csv.open("r", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if len(rows) < 2:
            core_pass = False
            core_errors.append("raw_csv_empty")
            print("Raw CSV has header only; no samples written")

    if core_pass and session_dir:
        step1_log = (session_dir / "logs" / "step1.log").read_text(errors="ignore")
        if "[alive]" not in step1_log:
            core_pass = False
            core_errors.append("heartbeat_missing")
            print("Heartbeat lines missing from logs/step1.log")

    if core_pass and session_dir:
        validate_cmd = [
            sys.executable,
            "-m",
            "muse_streaming.validate_session",
            "--session",
            str(session_dir),
        ]
        if _run_cmd(validate_cmd, cwd=repo_root, label="validate_session") != 0:
            core_pass = False
            core_errors.append("validate_session")

    if core_pass and session_dir:
        _append_synthetic_event(
            session_dir, onset_s=0.2, duration_s=min(1.0, float(args.duration_s) * 0.8)
        )
        step1b_cmd = [
            sys.executable,
            str(repo_root / "1b_extract_windows.py"),
            "--session-dir",
            str(session_dir),
        ]
        if _run_cmd(step1b_cmd, cwd=repo_root, label="1b_extract_windows") != 0:
            core_pass = False
            core_errors.append("1b_extract_windows")

    if core_pass and session_dir:
        if not (session_dir / "processed" / "eeg_windows.npz").exists():
            core_pass = False
            core_errors.append("missing_windows_npz")
            print("Missing processed/eeg_windows.npz")

    if core_pass and args.full and session_dir:
        full_status = "PASS"
        if not _module_available("torch"):
            full_status = "PARTIAL"
            print("[smoke] FULL SKIP: torch not available")
        else:
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
            if _run_cmd(train_cmd, cwd=repo_root, label="2_train_model") != 0:
                full_status = "FAIL"
            else:
                models_root = session_dir / "processed" / "models"
                run_dirs = (
                    [p for p in models_root.iterdir() if p.is_dir()] if models_root.exists() else []
                )
                if not run_dirs:
                    full_status = "FAIL"
                    print("No run dirs created under processed/models/")
                else:
                    run_dirs.sort(key=lambda p: p.stat().st_mtime)
                    run_dir = run_dirs[-1]
                    if not (run_dir / "finger_action_model.pt").exists():
                        full_status = "FAIL"
                        print("Missing trained model: processed/models/<run_id>/finger_action_model.pt")
                    if not (run_dir / "scaler.npz").exists():
                        full_status = "FAIL"
                        print("Missing scaler: processed/models/<run_id>/scaler.npz")

            if full_status == "PASS":
                eval_cmd = [
                    sys.executable,
                    str(repo_root / "3_evaluate_model.py"),
                    "--session-dir",
                    str(session_dir),
                ]
                if _run_cmd(eval_cmd, cwd=repo_root, label="3_evaluate_model") != 0:
                    full_status = "FAIL"

            if full_status == "PASS":
                if not _module_available("matplotlib"):
                    full_status = "PARTIAL"
                    print("[smoke] FULL SKIP: matplotlib not available (reports)")
                else:
                    report_cmd = [
                        sys.executable,
                        str(repo_root / "4_generate_reports.py"),
                        "--session-dir",
                        str(session_dir),
                    ]
                    if _run_cmd(report_cmd, cwd=repo_root, label="4_generate_reports") != 0:
                        full_status = "FAIL"

    stop_event.set()
    producer.join(timeout=1.0)

    print("\n[smoke] SUMMARY")
    print(f"CORE={'PASS' if core_pass else 'FAIL'}")
    print(f"FULL={full_status}")
    if core_errors:
        print("[smoke] core_errors=" + ", ".join(core_errors))
    if session_dir:
        key_paths = [
            session_dir,
            session_dir / "manifest.json",
            session_dir / "meta.json",
            session_dir / "raw" / "raw.csv",
            session_dir / "events" / "events.jsonl",
            session_dir / "processed" / "eeg_windows.npz",
            session_dir / "processed" / "models",
            session_dir / "processed" / "reports",
        ]
        for p in key_paths:
            if p.exists():
                print(f"[smoke] artifact={p}")

    if not core_pass:
        return 2
    if args.full and full_status == "FAIL":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
