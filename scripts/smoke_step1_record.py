from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamOutlet


def build_outlet(name: str = "Muse2-EEG", stype: str = "EEG") -> StreamOutlet:
    info = StreamInfo(name, stype, 4, 256, "float32", "smoke_step1")
    desc = info.desc()
    channels = desc.append_child("channels")
    for label in ["TP9", "AF7", "AF8", "TP10"]:
        ch = channels.append_child("channel")
        ch.append_child_value("label", label)
        ch.append_child_value("unit", "microvolts")
        ch.append_child_value("type", "EEG")
    return StreamOutlet(info)


def push_samples(outlet: StreamOutlet, stop_event: threading.Event, duration_s: float) -> None:
    start = time.monotonic()
    while time.monotonic() - start < duration_s and not stop_event.is_set():
        sample = np.random.randn(4).astype(np.float32).tolist()
        outlet.push_sample(sample)
        time.sleep(1.0 / 256.0)


def main() -> int:
    outlet = build_outlet()
    stop_event = threading.Event()
    producer = threading.Thread(
        target=push_samples, args=(outlet, stop_event, 6.0), daemon=True
    )
    producer.start()

    time.sleep(0.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        config_path = out_dir / "step1_smoke.json"
        config_path.write_text(
            """
{
  "settings": {
    "ENABLE_PLOT": false,
    "EVENT_MARKING_ENABLED": false,
    "HEARTBEAT_INTERVAL_S": 0.5,
    "NO_SAMPLE_TIMEOUT_S": 2.0,
    "WARMUP_SAMPLE_COUNT": 1,
    "WARMUP_TIMEOUT_S": 2.0
  }
}
""".strip()
        )
        cmd = [
            sys.executable,
            "1_stream_and_record.py",
            "--config",
            str(config_path),
            "--stream-name",
            "Muse2-EEG",
            "--stream-type",
            "EEG",
            "--subject-id",
            "SMOKE",
            "--session-id",
            "smoke",
            "--output-dir",
            str(out_dir),
            "--no-plot",
            "--no-event-marking",
            "--duration-s",
            "2.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        if proc.returncode != 0:
            raise SystemExit(f"Recording failed (exit {proc.returncode})")

        raw_candidates = sorted(out_dir.glob("SMOKE_smoke*_raw.csv"))
        if not raw_candidates:
            raise SystemExit("Raw CSV not found")
        raw_csv = raw_candidates[-1]
        with raw_csv.open("r", newline="") as handle:
            reader = csv.reader(handle)
            rows = list(reader)
        if len(rows) < 2:
            raise SystemExit("Raw CSV has header only; no samples written")

        session_id = raw_csv.name.replace("SMOKE_", "").replace("_raw.csv", "")
        session_dir = out_dir / f"SMOKE_{session_id}"
        if not session_dir.exists():
            raise SystemExit("Session dir not created")
        manifest = session_dir / "manifest.json"
        if not manifest.exists():
            raise SystemExit("manifest.json not written")
        run_log = session_dir / "run.log"
        if not run_log.exists():
            raise SystemExit("run.log not written")
        if "[alive]" not in run_log.read_text():
            raise SystemExit("Heartbeat lines missing from run.log")

    stop_event.set()
    producer.join(timeout=1.0)
    print("✅ Step 1 smoke test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
