#!/usr/bin/env python3
"""
Summarize Step 7 prediction logs and export predicted state segments for review.

The main use case is shadow-mode validation:
- summarize latency, stability, and actuation behavior from predictions.jsonl
- export predicted segments with relative offsets for side-by-side video review
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.label_schema import ACTION_NAMES, FINGER_NAMES


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return None
        return out
    except Exception:
        return None


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _pair_label(action_id: int, finger_id: int) -> str:
    return f"{ACTION_NAMES.get(int(action_id), str(action_id))}+{FINGER_NAMES.get(int(finger_id), str(finger_id))}"


def _load_config(path: str | Path | None) -> Tuple[Dict[str, Any], Path | None]:
    if not path:
        return {}, None
    config_path = Path(path).expanduser()
    if config_path.exists():
        try:
            config_path = config_path.resolve()
        except Exception:
            pass
    payload = json.loads(config_path.read_text())
    settings = payload.get("settings", payload)
    return dict(settings or {}), config_path.parent


def _resolve_input_path(value: str | None, *, base_dir: Path | None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if path.exists():
        try:
            return path.resolve()
        except Exception:
            return path
    return path


def _latest_dir_by_mtime(root: Path) -> Path | None:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def resolve_latest_live_infer_dir(session_dir: str | Path) -> Path | None:
    session_path = Path(session_dir).expanduser()
    if session_path.exists():
        try:
            session_path = session_path.resolve()
        except Exception:
            pass
    processed_dir = session_path / "processed"
    if not processed_dir.exists():
        return None
    candidates = [
        p
        for p in processed_dir.iterdir()
        if p.is_dir()
        and (
            p.name == "live_infer"
            or p.name.startswith("live_infer_")
            or p.name.startswith("live_infer_v")
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def resolve_prediction_log_path(
    *,
    pred_log: str | None,
    session_dir: str | Path | None,
    config_dir: Path | None,
) -> Path:
    direct = _resolve_input_path(pred_log, base_dir=config_dir)
    if direct is not None:
        if not direct.exists():
            raise FileNotFoundError(f"Prediction log not found: {direct}")
        return direct
    if session_dir is None:
        raise FileNotFoundError(
            "No prediction log provided. Pass --pred-log or provide session_dir in --config."
        )
    live_dir = resolve_latest_live_infer_dir(session_dir)
    if live_dir is None:
        raise FileNotFoundError(
            f"No Step 7 live_infer* output directory found under session: {Path(session_dir).expanduser()}"
        )
    candidate = live_dir / "predictions.jsonl"
    if not candidate.exists():
        raise FileNotFoundError(f"Prediction log not found: {candidate}")
    return candidate.resolve()


def _resolve_output_path(
    value: str | None,
    *,
    default_name: str | None,
    pred_log_path: Path,
) -> Path | None:
    text = str(value or "").strip()
    if not text:
        if not default_name:
            return None
        text = str(default_name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = pred_log_path.parent / path
    return path


def load_prediction_log(path: str | Path) -> List[Dict[str, Any]]:
    pred_path = Path(path)
    records: List[Dict[str, Any]] = []
    with pred_path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {pred_path}: {exc}") from exc
            payload["_line_no"] = int(line_no)
            records.append(payload)
    records.sort(
        key=lambda row: (
            _safe_float(row.get("window_start_s"))
            if _safe_float(row.get("window_start_s")) is not None
            else float("inf"),
            _safe_float(row.get("ts_utc"))
            if _safe_float(row.get("ts_utc")) is not None
            else float("inf"),
            int(row.get("_line_no", 0)),
        )
    )
    return records


def build_segments(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = list(records)
    if not rows:
        return []

    valid_starts = [
        _safe_float(row.get("window_start_s"))
        for row in rows
        if _safe_float(row.get("window_start_s")) is not None
    ]
    origin = float(valid_starts[0]) if valid_starts else 0.0

    segments: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    for row in rows:
        action_id = _safe_int(row.get("committed_action_id"), 0)
        finger_id = _safe_int(row.get("committed_finger_id"), 0)
        pair = (action_id, finger_id)
        start_s = _safe_float(row.get("window_start_s"))
        end_s = _safe_float(row.get("window_end_s"))
        ts_utc = _safe_float(row.get("ts_utc"))
        joint_conf = _safe_float(row.get("joint_conf"))
        action_conf = _safe_float(row.get("action_conf"))
        finger_conf = _safe_float(row.get("finger_conf"))
        applicability_prob = _safe_float(row.get("finger_applicable_prob"))
        actuation_sent = bool(row.get("actuation_sent", False))

        if current is None or tuple(current["pair"]) != pair:
            if current is not None:
                _finalize_segment(current, origin)
                segments.append(current)
            current = {
                "pair": [int(action_id), int(finger_id)],
                "pair_label": _pair_label(action_id, finger_id),
                "action_id": int(action_id),
                "finger_id": int(finger_id),
                "action_name": ACTION_NAMES.get(int(action_id), str(action_id)),
                "finger_name": FINGER_NAMES.get(int(finger_id), str(finger_id)),
                "start_s": start_s,
                "end_s": end_s,
                "start_ts_utc": ts_utc,
                "end_ts_utc": ts_utc,
                "window_count": 0,
                "joint_conf_values": [],
                "action_conf_values": [],
                "finger_conf_values": [],
                "applicability_prob_values": [],
                "actuation_sent_count": 0,
                "any_actuation_sent": False,
                "decision_reasons": Counter(),
                "alignment_fail_count": 0,
            }
        else:
            if current["start_s"] is None and start_s is not None:
                current["start_s"] = start_s
            if current["start_ts_utc"] is None and ts_utc is not None:
                current["start_ts_utc"] = ts_utc

        current["window_count"] += 1
        current["end_s"] = end_s if end_s is not None else current["end_s"]
        current["end_ts_utc"] = ts_utc if ts_utc is not None else current["end_ts_utc"]
        current["decision_reasons"][str(row.get("decision_reason", ""))] += 1
        current["alignment_fail_count"] += 0 if bool(row.get("alignment_ok", True)) else 1
        if joint_conf is not None:
            current["joint_conf_values"].append(joint_conf)
        if action_conf is not None:
            current["action_conf_values"].append(action_conf)
        if finger_conf is not None:
            current["finger_conf_values"].append(finger_conf)
        if applicability_prob is not None:
            current["applicability_prob_values"].append(applicability_prob)
        if actuation_sent:
            current["actuation_sent_count"] += 1
            current["any_actuation_sent"] = True

    if current is not None:
        _finalize_segment(current, origin)
        segments.append(current)
    return segments


def _finalize_segment(segment: Dict[str, Any], origin: float) -> None:
    start_s = _safe_float(segment.get("start_s"))
    end_s = _safe_float(segment.get("end_s"))
    duration_s = None
    if start_s is not None and end_s is not None and end_s >= start_s:
        duration_s = float(end_s - start_s)
    segment["duration_s"] = duration_s
    segment["start_offset_s"] = None if start_s is None else float(start_s - origin)
    segment["end_offset_s"] = None if end_s is None else float(end_s - origin)
    joint_conf_values = segment.pop("joint_conf_values", [])
    action_conf_values = segment.pop("action_conf_values", [])
    finger_conf_values = segment.pop("finger_conf_values", [])
    applicability_prob_values = segment.pop("applicability_prob_values", [])
    segment["mean_joint_conf"] = _mean_or_none(joint_conf_values)
    segment["max_joint_conf"] = _max_or_none(joint_conf_values)
    segment["mean_action_conf"] = _mean_or_none(action_conf_values)
    segment["mean_finger_conf"] = _mean_or_none(finger_conf_values)
    segment["mean_applicability_prob"] = _mean_or_none(applicability_prob_values)
    segment["dominant_decision_reason"] = _counter_top(segment["decision_reasons"])
    segment["decision_reasons"] = dict(segment["decision_reasons"])


def _mean_or_none(values: List[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=float)))


def _max_or_none(values: List[float]) -> float | None:
    if not values:
        return None
    return float(np.max(np.asarray(values, dtype=float)))


def _counter_top(counter: Counter) -> str | None:
    if not counter:
        return None
    return str(counter.most_common(1)[0][0])


def summarize_records(
    records: Iterable[Dict[str, Any]],
    *,
    short_segment_sec: float = 0.25,
) -> Dict[str, Any]:
    rows = list(records)
    segments = build_segments(rows)
    valid_rows = [row for row in rows if bool(row.get("alignment_ok", True))]

    latencies = np.asarray(
        [_safe_float(row.get("latency_ms")) for row in valid_rows if _safe_float(row.get("latency_ms")) is not None],
        dtype=float,
    )
    joint_conf = np.asarray(
        [_safe_float(row.get("joint_conf")) for row in valid_rows if _safe_float(row.get("joint_conf")) is not None],
        dtype=float,
    )
    action_unc = np.asarray(
        [
            _safe_float(row.get("action_uncertainty"))
            for row in valid_rows
            if _safe_float(row.get("action_uncertainty")) is not None
        ],
        dtype=float,
    )

    start_vals = [_safe_float(row.get("window_start_s")) for row in rows]
    end_vals = [_safe_float(row.get("window_end_s")) for row in rows]
    valid_starts = [v for v in start_vals if v is not None]
    valid_ends = [v for v in end_vals if v is not None]
    duration_s = None
    if valid_starts and valid_ends:
        duration_s = float(max(valid_ends) - min(valid_starts))

    hop_candidates = np.diff(np.asarray(sorted(valid_starts), dtype=float)) if len(valid_starts) >= 2 else np.asarray([], dtype=float)
    hop_s = float(np.median(hop_candidates)) if hop_candidates.size else None

    pair_counts = Counter(
        _pair_label(_safe_int(row.get("committed_action_id")), _safe_int(row.get("committed_finger_id")))
        for row in rows
    )
    decision_reason_counts = Counter(str(row.get("decision_reason", "")) for row in rows)
    suppress_counts = Counter(
        str(row.get("actuation_suppressed_reason"))
        for row in rows
        if row.get("actuation_suppressed_reason") not in (None, "")
    )
    quality_bad_reason_counts = Counter(
        str(row.get("quality_bad_reason") or "none") for row in rows
    )
    masked_channel_counts: Counter[int] = Counter()
    for row in rows:
        for channel_id in row.get("masked_channel_ids", []) or []:
            masked_channel_counts[int(channel_id)] += 1

    transition_count = 0
    prev_pair: tuple[int, int] | None = None
    for row in rows:
        pair = (
            _safe_int(row.get("committed_action_id")),
            _safe_int(row.get("committed_finger_id")),
        )
        if prev_pair is not None and pair != prev_pair:
            transition_count += 1
        prev_pair = pair

    actuatable_segments = [
        seg for seg in segments if int(seg.get("action_id", 0)) != 0 and int(seg.get("finger_id", 0)) != 0
    ]
    short_actuatable = [
        seg
        for seg in actuatable_segments
        if seg.get("duration_s") is not None and float(seg["duration_s"]) < float(short_segment_sec)
    ]

    actuation_sent_count = int(sum(bool(row.get("actuation_sent", False)) for row in rows))
    committed_non_rest_none_count = int(
        sum(
            _safe_int(row.get("committed_action_id")) != 0
            and _safe_int(row.get("committed_finger_id")) == 0
            for row in rows
        )
    )
    committed_rest_non_none_count = int(
        sum(
            _safe_int(row.get("committed_action_id")) == 0
            and _safe_int(row.get("committed_finger_id")) != 0
            for row in rows
        )
    )
    sent_non_rest_none_count = int(
        sum(
            bool(row.get("actuation_sent", False))
            and _safe_int(row.get("actuation_target_action_id")) != 0
            and _safe_int(row.get("actuation_target_finger_id")) == 0
            for row in rows
        )
    )
    sent_rest_non_none_count = int(
        sum(
            bool(row.get("actuation_sent", False))
            and _safe_int(row.get("actuation_target_action_id")) == 0
            and _safe_int(row.get("actuation_target_finger_id")) != 0
            for row in rows
        )
    )
    committed_pair_valid_all = all(
        bool(row.get("committed_pair_valid", True)) for row in valid_rows
    )
    uncertainty_gate_fail_count = int(
        sum(not bool(row.get("uncertainty_gate_ok", True)) for row in valid_rows)
    )
    applicability_gate_fail_count = int(
        sum(not bool(row.get("applicability_gate_ok", True)) for row in valid_rows)
    )

    summary = {
        "record_count": int(len(rows)),
        "valid_window_count": int(len(valid_rows)),
        "alignment_fail_count": int(len(rows) - len(valid_rows)),
        "duration_s": duration_s,
        "median_hop_s": hop_s,
        "prediction_rate_hz": (
            float(len(valid_rows) / duration_s) if duration_s and duration_s > 0 else None
        ),
        "latency_ms": _series_summary(latencies),
        "joint_conf": _series_summary(joint_conf),
        "action_uncertainty": _series_summary(action_unc),
        "transition_count": int(transition_count),
        "pair_transition_rate": (
            float(transition_count / max(1, len(rows) - 1)) if rows else None
        ),
        "transition_rate_per_min": (
            float(transition_count / (duration_s / 60.0))
            if duration_s and duration_s > 0
            else None
        ),
        "segment_count": int(len(segments)),
        "actuatable_segment_count": int(len(actuatable_segments)),
        "short_actuatable_segment_count": int(len(short_actuatable)),
        "short_actuatable_segment_rate": (
            float(len(short_actuatable) / len(actuatable_segments))
            if actuatable_segments
            else None
        ),
        "median_actuatable_segment_s": _median_duration(actuatable_segments),
        "pair_counts": dict(sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))),
        "decision_reason_counts": dict(decision_reason_counts),
        "actuation_suppressed_reason_counts": dict(suppress_counts),
        "window_quality_bad_count": int(
            sum(bool(row.get("window_quality_bad")) for row in rows)
        ),
        "quality_bad_reason_counts": dict(quality_bad_reason_counts),
        "alignment_interpolated_count": int(
            sum(bool(row.get("alignment_interpolated", False)) for row in valid_rows)
        ),
        "masked_window_count": int(
            sum(bool(row.get("masked_channel_ids")) for row in rows)
        ),
        "masked_channel_counts": {
            str(key): int(value)
            for key, value in sorted(masked_channel_counts.items(), key=lambda item: item[0])
        },
        "committed_non_rest_none_count": committed_non_rest_none_count,
        "committed_non_rest_none_rate": (
            float(committed_non_rest_none_count / len(rows)) if rows else None
        ),
        "committed_rest_non_none_count": committed_rest_non_none_count,
        "committed_rest_non_none_rate": (
            float(committed_rest_non_none_count / len(rows)) if rows else None
        ),
        "actuation_sent_count": actuation_sent_count,
        "sent_non_rest_none_count": sent_non_rest_none_count,
        "sent_non_rest_none_rate": (
            float(sent_non_rest_none_count / len(rows)) if rows else None
        ),
        "sent_rest_non_none_count": sent_rest_non_none_count,
        "sent_rest_non_none_rate": (
            float(sent_rest_non_none_count / len(rows)) if rows else None
        ),
        "deployment_pair_invariant_ok": bool(
            committed_non_rest_none_count == 0
            and committed_rest_non_none_count == 0
            and sent_non_rest_none_count == 0
            and sent_rest_non_none_count == 0
            and committed_pair_valid_all
        ),
        "actuation_sent_per_min": (
            float(actuation_sent_count / (duration_s / 60.0))
            if duration_s and duration_s > 0
            else None
        ),
        "uncertainty_gate_fail_count": int(uncertainty_gate_fail_count),
        "uncertainty_gate_fail_rate": (
            float(uncertainty_gate_fail_count / len(valid_rows))
            if valid_rows
            else None
        ),
        "applicability_gate_fail_count": int(applicability_gate_fail_count),
        "applicability_gate_fail_rate": (
            float(applicability_gate_fail_count / len(valid_rows))
            if valid_rows
            else None
        ),
    }
    return {"summary": summary, "segments": segments}


def _series_summary(values: np.ndarray) -> Dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _median_duration(segments: List[Dict[str, Any]]) -> float | None:
    durations = [float(seg["duration_s"]) for seg in segments if seg.get("duration_s") is not None]
    if not durations:
        return None
    return float(np.median(np.asarray(durations, dtype=float)))


def write_segments_csv(
    path: str | Path,
    segments: Iterable[Dict[str, Any]],
    *,
    video_offset_s: float = 0.0,
    include_review_columns: bool = False,
) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        start_offset = _safe_float(seg.get("start_offset_s"))
        end_offset = _safe_float(seg.get("end_offset_s"))
        row = {
            "segment_index": int(idx),
            "pair_label": seg.get("pair_label"),
            "action_id": int(seg.get("action_id", 0)),
            "action_name": seg.get("action_name"),
            "finger_id": int(seg.get("finger_id", 0)),
            "finger_name": seg.get("finger_name"),
            "start_offset_s": start_offset,
            "end_offset_s": end_offset,
            "video_start_s": None if start_offset is None else float(start_offset + video_offset_s),
            "video_end_s": None if end_offset is None else float(end_offset + video_offset_s),
            "duration_s": _safe_float(seg.get("duration_s")),
            "window_count": int(seg.get("window_count", 0)),
            "mean_joint_conf": _safe_float(seg.get("mean_joint_conf")),
            "max_joint_conf": _safe_float(seg.get("max_joint_conf")),
            "mean_action_conf": _safe_float(seg.get("mean_action_conf")),
            "mean_finger_conf": _safe_float(seg.get("mean_finger_conf")),
            "mean_applicability_prob": _safe_float(seg.get("mean_applicability_prob")),
            "any_actuation_sent": bool(seg.get("any_actuation_sent", False)),
            "actuation_sent_count": int(seg.get("actuation_sent_count", 0)),
            "dominant_decision_reason": seg.get("dominant_decision_reason"),
            "alignment_fail_count": int(seg.get("alignment_fail_count", 0)),
        }
        if include_review_columns:
            row.update(
                {
                    "observed_action_name": "",
                    "observed_finger_name": "",
                    "observed_pair_label": "",
                    "match": "",
                    "notes": "",
                }
            )
        rows.append(row)

    if not rows:
        rows = [
            {
                "segment_index": "",
                "pair_label": "",
                "action_id": "",
                "action_name": "",
                "finger_id": "",
                "finger_name": "",
                "start_offset_s": "",
                "end_offset_s": "",
                "video_start_s": "",
                "video_end_s": "",
                "duration_s": "",
                "window_count": "",
                "mean_joint_conf": "",
                "max_joint_conf": "",
                "mean_action_conf": "",
                "mean_finger_conf": "",
                "any_actuation_sent": "",
                "actuation_sent_count": "",
                "dominant_decision_reason": "",
                "alignment_fail_count": "",
            }
        ]
        if include_review_columns:
            rows[0].update(
                {
                    "observed_action_name": "",
                    "observed_finger_name": "",
                    "observed_pair_label": "",
                    "match": "",
                    "notes": "",
                }
            )

    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Step 7 predictions.jsonl logs.")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional config JSON produced by the UI.",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Optional session directory used to auto-resolve the latest processed/live_infer*/predictions.jsonl, including fresh live_infer_<run_tag> directories.",
    )
    parser.add_argument("--pred-log", type=str, default=None, help="Path to predictions.jsonl.")
    parser.add_argument("--out-json", type=str, default=None, help="Optional JSON summary output.")
    parser.add_argument(
        "--segments-csv",
        type=str,
        default=None,
        help="Optional CSV export of predicted state segments.",
    )
    parser.add_argument(
        "--review-csv",
        type=str,
        default=None,
        help="Optional CSV export with blank reviewer columns for video comparison.",
    )
    parser.add_argument(
        "--video-offset-s",
        type=float,
        default=0.0,
        help="Offset applied when exporting video-aligned segment timestamps.",
    )
    parser.add_argument(
        "--short-segment-sec",
        type=float,
        default=0.25,
        help="Duration threshold used to flag short actuatable segments.",
    )
    args = parser.parse_args()

    config_settings, config_dir = _load_config(args.config)

    session_dir_value = (
        args.session_dir if args.session_dir is not None else config_settings.get("session_dir")
    )
    pred_log_value = (
        args.pred_log if args.pred_log is not None else config_settings.get("pred_log")
    )
    pred_log_path = resolve_prediction_log_path(
        pred_log=pred_log_value,
        session_dir=session_dir_value,
        config_dir=config_dir,
    )

    out_json_value = (
        args.out_json if args.out_json is not None else config_settings.get("out_json")
    )
    segments_csv_value = (
        args.segments_csv
        if args.segments_csv is not None
        else config_settings.get("segments_csv")
    )
    review_csv_value = (
        args.review_csv if args.review_csv is not None else config_settings.get("review_csv")
    )
    video_offset_s = (
        float(args.video_offset_s)
        if "--video-offset-s" in sys.argv
        else float(config_settings.get("video_offset_s", args.video_offset_s))
    )
    short_segment_sec = (
        float(args.short_segment_sec)
        if "--short-segment-sec" in sys.argv
        else float(config_settings.get("short_segment_sec", args.short_segment_sec))
    )

    records = load_prediction_log(pred_log_path)
    result = summarize_records(records, short_segment_sec=float(short_segment_sec))
    result["summary"]["pred_log_path"] = str(pred_log_path)
    if session_dir_value:
        session_dir_path = _resolve_input_path(str(session_dir_value), base_dir=config_dir)
        if session_dir_path is not None:
            result["summary"]["session_dir"] = str(session_dir_path)

    out_json = _resolve_output_path(
        out_json_value,
        default_name="live_prediction_summary.json",
        pred_log_path=pred_log_path,
    )
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result["summary"], indent=2, sort_keys=True))

    segments_csv = _resolve_output_path(
        segments_csv_value,
        default_name="predicted_segments.csv",
        pred_log_path=pred_log_path,
    )
    if segments_csv is not None:
        write_segments_csv(
            segments_csv,
            result["segments"],
            video_offset_s=float(video_offset_s),
            include_review_columns=False,
        )

    review_csv = _resolve_output_path(
        review_csv_value,
        default_name="predicted_segments_review.csv",
        pred_log_path=pred_log_path,
    )
    if review_csv is not None:
        write_segments_csv(
            review_csv,
            result["segments"],
            video_offset_s=float(video_offset_s),
            include_review_columns=True,
        )

    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
