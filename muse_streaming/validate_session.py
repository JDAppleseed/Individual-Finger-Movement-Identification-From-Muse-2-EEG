from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _load_events(path: Path) -> Tuple[int, List[str]]:
    if not path.exists():
        return 0, ["events.jsonl missing"]
    errors: List[str] = []
    count = 0
    for idx, line in enumerate(path.read_text().splitlines()):
        if not line.strip():
            continue
        try:
            json.loads(line)
            count += 1
        except Exception:
            errors.append(f"events.jsonl parse error at line {idx + 1}")
    return count, errors


def validate_session_dir(session_dir: Path, *, allow_partial: bool = False) -> Dict[str, Any]:
    session_dir = session_dir.expanduser().resolve()
    manifest_path = session_dir / "manifest.json"
    meta_path = session_dir / "meta.json"
    raw_dir = session_dir / "raw"
    events_path = session_dir / "events" / "events.jsonl"

    report: Dict[str, Any] = {
        "session_dir": str(session_dir),
        "ok": True,
        "issues": [],
        "manifest": {},
    }

    manifest = _load_json(manifest_path)
    meta = _load_json(meta_path)
    report["manifest"] = manifest
    if not manifest:
        report["ok"] = False
        report["issues"].append("manifest_missing_or_invalid")
        return report
    if not meta:
        report["ok"] = False
        report["issues"].append("meta_missing_or_invalid")

    termination_reason = manifest.get("termination_reason")
    missing_seq_count = int(manifest.get("missing_seq_count") or 0)
    if not allow_partial:
        allowed_termination = {
            "normal",
            "duration_elapsed",
            "user_stop",
            "init_only",
            "signal_SIGINT",
            "signal_SIGTERM",
        }
        if termination_reason not in allowed_termination:
            report["ok"] = False
            report["issues"].append(f"termination_reason={termination_reason}")
        if missing_seq_count != 0:
            report["ok"] = False
            report["issues"].append(f"missing_seq_count={missing_seq_count}")

    shard_list = manifest.get("shard_list") or []
    shard_paths = []
    for item in shard_list:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path")
        if not raw_path:
            continue
        p = Path(str(raw_path))
        if not p.is_absolute():
            p = (session_dir / p).resolve()
        shard_paths.append(p)
    if not shard_paths:
        shard_paths = sorted(raw_dir.glob("eeg_raw_shard_*.npy"))
    if not shard_paths:
        report["ok"] = False
        report["issues"].append("no_raw_shards_found")
        return report

    seq_prev = None
    lsl_prev = None
    total_samples = 0
    seq_gaps = 0
    for shard_path in shard_paths:
        if not shard_path.exists():
            report["ok"] = False
            report["issues"].append(f"missing_shard:{shard_path}")
            continue
        data = np.load(shard_path)
        if "seq" not in data.dtype.names or "lsl_ts_mono" not in data.dtype.names:
            report["ok"] = False
            report["issues"].append(f"invalid_shard_format:{shard_path}")
            continue
        seq = data["seq"].astype(np.int64)
        lsl = data["lsl_ts_mono"].astype(float)
        total_samples += int(seq.shape[0])
        if seq_prev is not None:
            gap = int(seq[0] - seq_prev - 1)
            if gap > 0:
                seq_gaps += gap
        seq_prev = int(seq[-1])
        if np.any(np.diff(seq) != 1):
            seq_gaps += int(np.sum(np.maximum(np.diff(seq) - 1, 0)))
        if lsl_prev is not None and lsl[0] <= lsl_prev:
            report["ok"] = False
            report["issues"].append("non_monotonic_lsl_ts_mono")
        if np.any(np.diff(lsl) <= 0):
            report["ok"] = False
            report["issues"].append("non_monotonic_lsl_ts_mono")
        lsl_prev = float(lsl[-1])

    if not allow_partial and seq_gaps > 0:
        report["ok"] = False
        report["issues"].append(f"seq_gaps_detected={seq_gaps}")

    event_count, event_errors = _load_events(events_path)
    if event_errors:
        report["ok"] = False
        report["issues"].extend(event_errors)
    report["event_count"] = event_count
    report["subject_id"] = meta.get("subject_id")
    report["session_id"] = meta.get("session_id")
    report["total_samples"] = total_samples
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True, help="Session directory path")
    parser.add_argument("--allow-partial", action="store_true", help="Allow partial sessions")
    args = parser.parse_args()

    report = validate_session_dir(Path(args.session), allow_partial=args.allow_partial)
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
