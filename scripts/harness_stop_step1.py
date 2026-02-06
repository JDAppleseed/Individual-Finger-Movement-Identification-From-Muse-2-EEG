from __future__ import annotations

import argparse
import json
import os
import re
import select
import signal
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

    info = StreamInfo(name, stype, 4, 256, "float32", "stop_harness")
    desc = info.desc()
    channels = desc.append_child("channels")
    for label in ["TP9", "AF7", "AF8", "TP10"]:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    return StreamOutlet(info)


def push_samples(outlet: Any, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        sample = np.random.randn(4).astype(np.float32).tolist()
        outlet.push_sample(sample)
        time.sleep(1.0 / 256.0)


def staged_stop(proc: subprocess.Popen[str], name: str) -> None:
    if proc.poll() is not None:
        return
    try:
        os.kill(proc.pid, signal.SIGINT)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.kill(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.05)
    try:
        os.kill(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _tail_step1_stderr(proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> list[str]:
    lines: list[str] = []
    if proc.stderr is None:
        return lines
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        try:
            ready, _, _ = select.select([proc.stderr], [], [], 0.05)
        except Exception:
            ready = []
        if not ready:
            continue
        try:
            line = proc.stderr.readline()
        except Exception:
            line = ""
        if not line:
            continue
        clean = line.rstrip("\r\n")
        if clean:
            lines.append(clean)
    return lines


def _stderr_has_ready_signals(lines: list[str]) -> bool:
    saw_connected = any("Connected to LSL stream" in line for line in lines)
    alive_re = re.compile(r"\\[alive\\].*recv=(\\d+).*wrote=(\\d+)")
    saw_alive = False
    for line in lines:
        m = alive_re.search(line)
        if not m:
            continue
        recv = int(m.group(1))
        wrote = int(m.group(2))
        if recv > 0 and wrote > 0:
            saw_alive = True
            break
    return saw_connected and saw_alive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="Enable plot during stop test.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    project = "STOP_HARNESS"
    subject = "STOP"
    session_id = f"{subject}_{ts}"
    sessions_root = (
        repo_root / "Projects" / project / "subjects" / subject / "sessions"
    )
    sessions_root.mkdir(parents=True, exist_ok=True)
    requested_session_dir = sessions_root / session_id

    outlet = build_outlet()
    stop_event = threading.Event()
    producer = threading.Thread(target=push_samples, args=(outlet, stop_event), daemon=True)
    producer.start()

    config_path = sessions_root / f"{session_id}_stop_config.json"
    config_path.write_text(
        json.dumps(
            {
                "project_name": project,
                "subject_id": subject,
                "session_id": session_id,
                "settings": {
                    "LSL_STREAM_NAME": "Muse2-EEG",
                    "LSL_STREAM_TYPE": "EEG",
                    "ENABLE_PLOT": bool(args.plot),
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

    step1_cmd = [
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
        "stop_harness",
        "--subject-id",
        subject,
        "--no-event-marking",
    ]
    if not args.plot:
        step1_cmd.append("--no-plot")
    step1_proc = subprocess.Popen(
        step1_cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    stderr_lines = _tail_step1_stderr(step1_proc, timeout_s=2.0)
    if not _stderr_has_ready_signals(stderr_lines):
        print("[harness] Step1 stderr:")
        for line in stderr_lines:
            print(line, file=sys.stderr)
        raise SystemExit("Step 1 did not report connected + alive within timeout")

    # Dummy connector process (simulates a long-running streamer).
    connector_cmd = [
        sys.executable,
        "-u",
        "-c",
        "import time\nprint('connector heartbeat')\ntime.sleep(9999)\n",
    ]
    connector_proc = subprocess.Popen(
        connector_cmd, cwd=str(repo_root), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    time.sleep(1.0)
    staged_stop(step1_proc, "step1")
    staged_stop(connector_proc, "connector")

    step1_deadline = time.monotonic() + 4.0
    while time.monotonic() < step1_deadline:
        if step1_proc.poll() is not None:
            break
        time.sleep(0.05)
    if step1_proc.poll() is None:
        raise SystemExit("Step 1 did not exit within timeout")

    connector_deadline = time.monotonic() + 4.0
    while time.monotonic() < connector_deadline:
        if connector_proc.poll() is not None:
            break
        time.sleep(0.05)
    if connector_proc.poll() is None:
        raise SystemExit("Connector did not exit within timeout")

    stop_event.set()
    producer.join(timeout=1.0)

    expected = [
        requested_session_dir / "manifest.json",
        requested_session_dir / "meta.json",
        requested_session_dir / "events" / "events.jsonl",
    ]
    missing = [p for p in expected if not p.exists()]
    if missing:
        raise SystemExit("Missing expected artifacts:\n" + "\n".join(str(p) for p in missing))
    shard_candidates = sorted((requested_session_dir / "raw").glob("eeg_raw_shard_*.npy"))
    if not shard_candidates:
        raise SystemExit("No raw shards found under session_dir/raw/")

    print("[harness] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
