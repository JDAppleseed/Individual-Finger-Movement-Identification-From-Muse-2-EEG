#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils.segmenting import SegmentBreaker


def _parse_float(value: str) -> Optional[float]:
    try:
        if value == "" or value is None:
            return None
        return float(value)
    except Exception:
        return None


def _segment_features(
    features_path: Path, gap_break_s: float
) -> Tuple[List[Dict[str, float]], List[Path]]:
    segments: List[Dict[str, float]] = []
    outputs: List[Path] = []
    prefix = features_path.name.replace("_eeg_features.csv", "")
    out_dir = features_path.parent

    with features_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if not header:
            return segments, outputs
        try:
            lsl_idx = header.index("lsl_timestamp")
        except ValueError as exc:
            raise RuntimeError("features.csv missing lsl_timestamp column") from exc

        breaker = SegmentBreaker(gap_break_s=gap_break_s)
        seg_idx = 0
        writer = None
        out_handle = None
        seg_min = None
        seg_max = None

        def _open_new_segment() -> None:
            nonlocal writer, out_handle, seg_min, seg_max, seg_idx
            out_path = out_dir / f"{prefix}_SEG{seg_idx:02d}_eeg_features.csv"
            out_handle = out_path.open("w", newline="")
            writer = csv.writer(out_handle)
            writer.writerow(header)
            outputs.append(out_path)
            seg_min = None
            seg_max = None

        for row in reader:
            if not row:
                continue
            lsl_ts = _parse_float(row[lsl_idx] if lsl_idx < len(row) else "")
            if lsl_ts is None or not math.isfinite(lsl_ts):
                continue
            break_info = breaker.check(lsl_ts)
            if writer is None:
                _open_new_segment()
            elif break_info.should_break:
                if seg_min is not None and seg_max is not None:
                    segments.append(
                        {"index": float(seg_idx), "min_lsl": seg_min, "max_lsl": seg_max}
                    )
                if out_handle:
                    out_handle.close()
                seg_idx += 1
                _open_new_segment()
            if writer is None:
                continue
            writer.writerow(row)
            seg_min = lsl_ts if seg_min is None else min(seg_min, lsl_ts)
            seg_max = lsl_ts if seg_max is None else max(seg_max, lsl_ts)

        if writer is not None:
            if seg_min is not None and seg_max is not None:
                segments.append(
                    {"index": float(seg_idx), "min_lsl": seg_min, "max_lsl": seg_max}
                )
            if out_handle:
                out_handle.close()

    return segments, outputs


def _trim_events(
    events_path: Path, segments: List[Dict[str, float]], prefix: str
) -> List[Path]:
    if not segments:
        return []
    outputs: List[Path] = []
    out_dir = events_path.parent
    rows: List[Dict[str, object]] = []
    suffix = events_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(events_path.read_text())
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            payload = payload.get("events")
        if isinstance(payload, list):
            rows = [dict(ev) for ev in payload if isinstance(ev, dict)]
    elif suffix == ".jsonl":
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    else:
        with events_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)

    for segment in segments:
        idx = int(segment["index"])
        seg_min = segment["min_lsl"]
        seg_max = segment["max_lsl"]
        out_path = out_dir / f"{prefix}_SEG{idx:02d}_events.jsonl"
        seg_rows: List[Dict[str, object]] = []
        for row in rows:
            onset_lsl = _parse_float(str(row.get("onset_lsl", "")))
            end_lsl = _parse_float(str(row.get("end_lsl", "")))
            if onset_lsl is None:
                onset_lsl = _parse_float(str(row.get("lsl_ts_mono", "")))
            if onset_lsl is None:
                onset_lsl = _parse_float(str(row.get("event_lsl_ts", "")))
            if onset_lsl is None:
                continue
            if end_lsl is None:
                duration = _parse_float(str(row.get("duration_s", ""))) or 0.0
                end_lsl = onset_lsl + duration
            if end_lsl < seg_min or onset_lsl > seg_max:
                continue
            seg_rows.append(row)
        with out_path.open("w", encoding="utf-8") as handle:
            for row in seg_rows:
                handle.write(json.dumps(row) + "\n")
        outputs.append(out_path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Salvage legacy session files into segmented outputs."
    )
    parser.add_argument("--features", required=True, help="Path to features.csv")
    parser.add_argument(
        "--events",
        required=True,
        help="Path to events.jsonl or legacy CSV",
    )
    parser.add_argument(
        "--gap-break-s",
        type=float,
        default=1.0,
        help="Gap in LSL seconds that triggers a new segment",
    )
    args = parser.parse_args()

    features_path = Path(args.features)
    events_path = Path(args.events)
    if not features_path.exists():
        raise FileNotFoundError(f"features file not found: {features_path}")
    if not events_path.exists():
        raise FileNotFoundError(f"events file not found: {events_path}")

    segments, feature_outputs = _segment_features(features_path, args.gap_break_s)
    prefix = features_path.name.replace("_eeg_features.csv", "")
    event_outputs = _trim_events(events_path, segments, prefix)

    print(f"Segments written: {len(feature_outputs)} features, {len(event_outputs)} events")
    for path in feature_outputs:
        print(f"  features: {path}")
    for path in event_outputs:
        print(f"  events:   {path}")


if __name__ == "__main__":
    main()
