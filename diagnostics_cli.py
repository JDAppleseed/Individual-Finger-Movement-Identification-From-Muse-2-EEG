#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from utils.stream_timebase import gap_threshold_s, is_gap, summarize_gaps
from utils.timebase_selfcheck import (
    evaluate_event_time_consistency,
    evaluate_timebase_consistency,
)


def _read_float_column(path: Path, column: str) -> List[float]:
    if not path or not path.exists():
        return []
    values: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row or column not in row:
                continue
            try:
                value = float(row[column])
            except Exception:
                continue
            if np.isfinite(value):
                values.append(value)
    return values


def _load_session_meta(path: Optional[Path]) -> Dict[str, object]:
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _resolve_path_from_meta(meta: Dict[str, object], key: str) -> Optional[Path]:
    value = meta.get(key) if meta else None
    if not value:
        return None
    return Path(str(value))


def _count_in_range(values: Iterable[float], min_s: float, max_s: float) -> Tuple[int, int]:
    in_range = 0
    out_range = 0
    for value in values:
        if min_s <= value <= max_s:
            in_range += 1
        else:
            out_range += 1
    return in_range, out_range


def _prediction_monotonic(pred_times: List[float]) -> bool:
    if len(pred_times) < 2:
        return True
    diffs = np.diff(np.asarray(pred_times, dtype=float))
    return bool(np.all(diffs >= 0))


def diagnostics(
    features_path: Path,
    events_path: Optional[Path],
    predictions_path: Optional[Path],
    session_meta: Dict[str, object],
) -> Dict[str, object]:
    time_s = _read_float_column(features_path, "time_s")
    lsl_raw = _read_float_column(features_path, "lsl_timestamp")
    lsl_mono = _read_float_column(features_path, "lsl_timestamp_mono")

    if len(time_s) < 2:
        raise SystemExit("Not enough feature samples to compute diagnostics.")

    time_s_arr = np.asarray(time_s, dtype=float)
    dt = np.diff(time_s_arr)
    rows_per_sec = float(len(time_s_arr) / max(1e-9, time_s_arr[-1] - time_s_arr[0]))

    sampling_rate = float(session_meta.get("sampling_rate", 256.0))
    nominal_dt = 1.0 / sampling_rate
    gap_thresh = gap_threshold_s(nominal_dt)
    gap_durations = [float(d) for d in dt if is_gap(float(d), nominal_dt)]
    gap_summary = summarize_gaps(gap_durations)

    backwards_count = 0
    if lsl_raw and lsl_mono and len(lsl_raw) == len(lsl_mono):
        prev_mono = None
        for raw_ts, mono_ts in zip(lsl_raw, lsl_mono):
            if prev_mono is not None and raw_ts < prev_mono:
                backwards_count += 1
            prev_mono = mono_ts

    event_onset_s = _read_float_column(events_path, "onset_s") if events_path else []
    event_onset_lsl = _read_float_column(events_path, "onset_lsl") if events_path else []
    pred_times = (
        _read_float_column(predictions_path, "prediction_time_s")
        if predictions_path
        else []
    )
    pred_lsl = (
        _read_float_column(predictions_path, "prediction_lsl_ts")
        if predictions_path
        else []
    )

    features_min = float(time_s_arr[0])
    features_max = float(time_s_arr[-1])

    event_in_range, event_out_range = _count_in_range(
        event_onset_s, features_min, features_max
    )
    pred_in_range, pred_out_range = _count_in_range(
        pred_times, features_min, features_max
    )

    stream_start_lsl_ts = session_meta.get("stream_start_lsl_ts")
    consistency = None
    event_consistency = None
    if stream_start_lsl_ts is not None and lsl_mono:
        consistency = evaluate_timebase_consistency(
            time_s_arr, lsl_mono, float(stream_start_lsl_ts)
        )
        if event_onset_s and event_onset_lsl:
            event_consistency = evaluate_event_time_consistency(
                event_onset_s, event_onset_lsl, float(stream_start_lsl_ts)
            )

    pred_monotonic = _prediction_monotonic(pred_times)
    verdict = "OK"
    if event_out_range or pred_out_range:
        verdict = "INVALID"
    if consistency and consistency.error:
        verdict = "INVALID"
    if not pred_monotonic and pred_times:
        verdict = "INVALID"

    return {
        "rows_per_sec": rows_per_sec,
        "dt_p50_s": float(np.percentile(dt, 50)),
        "dt_p95_s": float(np.percentile(dt, 95)),
        "dt_p99_s": float(np.percentile(dt, 99)),
        "gap_threshold_s": gap_thresh,
        "gap_count": gap_summary.count,
        "gap_max_s": gap_summary.max_gap_s,
        "gap_p95_s": gap_summary.p95_gap_s,
        "gap_p99_s": gap_summary.p99_gap_s,
        "backwards_count": backwards_count,
        "event_in_range": event_in_range,
        "event_out_of_range": event_out_range,
        "pred_in_range": pred_in_range,
        "pred_out_of_range": pred_out_range,
        "pred_monotonic": pred_monotonic,
        "timebase_consistency": consistency,
        "event_consistency": event_consistency,
        "verdict": verdict,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=str, default=None)
    parser.add_argument("--events", type=str, default=None)
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--session-meta", type=str, default="session_meta.json")
    args = parser.parse_args()

    session_meta_path = Path(args.session_meta) if args.session_meta else None
    session_meta = _load_session_meta(session_meta_path)

    features_path = (
        Path(args.features)
        if args.features
        else _resolve_path_from_meta(session_meta, "features_path")
    )
    events_path = (
        Path(args.events)
        if args.events
        else _resolve_path_from_meta(session_meta, "events_path")
    )
    predictions_path = (
        Path(args.predictions)
        if args.predictions
        else _resolve_path_from_meta(session_meta, "predictions_path")
    )

    if not features_path or not features_path.exists():
        raise SystemExit("features path not found; provide --features or session_meta.json")

    report = diagnostics(features_path, events_path, predictions_path, session_meta)

    print("==== Diagnostics ====")
    print(f"Rows/sec: {report['rows_per_sec']:.2f}")
    print(
        "dt percentiles (s): "
        f"p50={report['dt_p50_s']:.6f}, p95={report['dt_p95_s']:.6f}, p99={report['dt_p99_s']:.6f}"
    )
    print(
        "gaps: "
        f"count={report['gap_count']}, max={report['gap_max_s']}, "
        f"p95={report['gap_p95_s']}, p99={report['gap_p99_s']}"
    )
    print(f"backward count: {report['backwards_count']}")
    print(
        f"events in/out-of-range: {report['event_in_range']}/{report['event_out_of_range']}"
    )
    print(
        f"predictions in/out-of-range: {report['pred_in_range']}/{report['pred_out_of_range']}"
    )
    print(f"prediction_time_s monotonic: {report['pred_monotonic']}")
    print(f"session usability verdict: {report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
