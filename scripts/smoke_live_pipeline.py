from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
from pylsl import StreamInfo, StreamOutlet

from muse_streaming.healthcheck import run_healthcheck


def build_outlet(name: str = "Muse2-EEG") -> StreamOutlet:
    info = StreamInfo(name, "EEG", 4, 256, "float32", "smoke_test")
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
        target=push_samples, args=(outlet, stop_event, 1.0), daemon=True
    )
    producer.start()

    result = run_healthcheck(
        stream_name="Muse2-EEG",
        stype="EEG",
        required_labels=["TP9", "AF7", "AF8", "TP10"],
        require_exact_channels=True,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.ok:
        raise SystemExit(2)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "step1.json"
        payload = {
            "settings": {
                "LSL_STREAM_NAME": "Muse2-EEG",
                "LSL_STREAM_TYPE": "EEG",
                "REQUIRED_LSL_LABELS": ["TP9", "AF7", "AF8", "TP10"],
                "REQUIRE_EXACTLY_4_CHANNELS": True,
                "HARD_STOP_AFTER_UNHEALTHY_S": 2.0,
                "FAILED_WRITE_WINDOW_S": 5.0,
                "FAILED_DIR": "data/failed",
                "SAVE_TO_DISK": True,
                "SAVE_RAW": True,
                "ENABLE_PLOT": False,
                "EVENT_MARKING_ENABLED": False,
                "LABEL_CHECK_ACKNOWLEDGED": True,
            }
        }
        config_path.write_text(json.dumps(payload, indent=2))

        proc = subprocess.Popen(
            [
                sys.executable,
                "1_stream_and_record.py",
                "--config",
                str(config_path),
                "--force-new-session",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        # stop samples after 1s to trigger hard stop
        producer.join()
        stop_event.set()

        output, _ = proc.communicate(timeout=15)
        print(output)
        if proc.returncode != 73:
            raise SystemExit(f"Expected hard stop exit code 73, got {proc.returncode}")

    failed_dir = Path("data/failed")
    if not failed_dir.exists() or not list(failed_dir.glob("*_UNHEALTHY*.csv")):
        raise SystemExit("Failed debug files not found in data/failed")

    if not list(Path("logs").glob("hard_stop_*.json")):
        raise SystemExit("Hard stop report not found in logs")

    print("✅ Smoke live pipeline check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
