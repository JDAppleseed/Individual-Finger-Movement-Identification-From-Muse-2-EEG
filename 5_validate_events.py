"""
STEP 5b — Event Validation & Repair
Flags invalid action/finger combinations and basic timing issues.
Use --apply to fix issues in-place and write edit logs.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from utils.events_audit import log_event_edit
from utils.label_schema import (
    ACTION_REST,
    FINGER_NONE,
    is_valid_action_finger,
    event_type_for,
)

DEFAULT_FS = 256
OVERLAP_EPS = 0.02


def latest_subject_file(subject_id, suffix, base_dir):
    base = Path(base_dir)
    pattern = f"{subject_id}_*_{suffix}"
    candidates = sorted(base.glob(pattern), key=lambda p: p.name)
    return candidates[-1] if candidates else None


def resolve_paths(session_dir=None, events_override=None, features_override=None):
    session_used = False
    events_path = Path(events_override) if events_override else None
    features_path = Path(features_override) if features_override else None

    if session_dir:
        session_used = True
        session_path = Path(session_dir).expanduser().resolve()
        if not session_path.exists():
            return False, None, None
        if events_path is None:
            events_path = session_path / "events" / "events.csv"
        if features_path is None:
            features_path = session_path / "raw" / "raw.csv"
            if not features_path.exists():
                alt = session_path / "features" / "eeg_features.csv"
                if alt.exists():
                    features_path = alt

    return session_used, events_path, features_path


def validate_events(events):
    # This validator is a guardrail before Step 1b: enforce schema + timing sanity
    # so window labels are derived from coherent event records.
    issues = []
    warnings = []

    required_cols = {"onset_s", "duration_s", "type", "finger_id", "action_id"}
    optional_cols = {
        "trial_id",
        "block_id",
        "session_mode",
        "channel",
        "confidence",
        "notes",
        "source",
    }

    missing_required = sorted(required_cols - set(events.columns))
    missing_optional = sorted(optional_cols - set(events.columns))

    if missing_optional:
        warnings.append(f"Missing optional columns: {missing_optional}")

    if missing_required:
        issues.append(
            {
                "row": -1,
                "type": "missing_required_columns",
                "detail": missing_required,
            }
        )
        return issues, warnings, missing_required, missing_optional

    for idx, row in events.iterrows():
        try:
            onset = float(row.get("onset_s", 0))
            duration = float(row.get("duration_s", 0))
        except Exception:
            issues.append({"row": int(idx), "type": "non_numeric_time"})
            continue

        try:
            action_id = int(row.get("action_id", 0))
            finger_id = int(row.get("finger_id", 0))
        except Exception:
            issues.append({"row": int(idx), "type": "non_integer_label"})
            continue

        if onset < 0:
            issues.append({"row": int(idx), "type": "negative_onset"})
        if duration < 0:
            issues.append({"row": int(idx), "type": "negative_duration"})
        if not is_valid_action_finger(action_id, finger_id):
            issues.append({"row": int(idx), "type": "invalid_action_finger"})

    if "onset_s" in events.columns:
        onset_series = pd.to_numeric(events["onset_s"], errors="coerce")
        if onset_series.isna().any():
            warnings.append("Found NaNs in onset_s after coercion")
        else:
            if (onset_series.diff().fillna(0) < 0).any():
                warnings.append("onset_s is non-monotonic (decreases at least once)")

    duplicate_cols = [
        c
        for c in ["onset_s", "duration_s", "action_id", "finger_id", "type"]
        if c in events.columns
    ]
    if duplicate_cols:
        dup_mask = events.duplicated(subset=duplicate_cols, keep=False)
        if dup_mask.any():
            warnings.append(f"Duplicate exact events found (cols={duplicate_cols})")

    if {"onset_s", "duration_s"}.issubset(events.columns):
        events_sorted = events.copy()
        events_sorted["onset_s"] = pd.to_numeric(
            events_sorted["onset_s"], errors="coerce"
        )
        events_sorted["duration_s"] = pd.to_numeric(
            events_sorted["duration_s"], errors="coerce"
        )
        events_sorted = events_sorted.dropna(
            subset=["onset_s", "duration_s"]
        ).sort_values("onset_s")
        prev_end = None
        for _, row in events_sorted.iterrows():
            onset = float(row["onset_s"])
            duration = float(row["duration_s"])
            end = onset + max(0.0, duration)
            action_id = int(row.get("action_id", 0))
            event_type = str(row.get("type", ""))
            if action_id == ACTION_REST or event_type == "artifact":
                prev_end = max(prev_end or 0.0, end)
                continue
            if prev_end is not None and onset < (prev_end - OVERLAP_EPS):
                warnings.append("Overlapping events detected (non-REST overlap)")
                break
            prev_end = max(prev_end or 0.0, end)

    return issues, warnings, missing_required, missing_optional


def repair_event(row):
    # Repair logic mirrors label_schema rules used by extraction/training code.
    before = dict(row)
    action_id = int(row["action_id"])
    finger_id = int(row["finger_id"])

    if action_id == ACTION_REST and finger_id != FINGER_NONE:
        row["finger_id"] = FINGER_NONE
        row["type"] = "rest"
    elif action_id != ACTION_REST and finger_id == FINGER_NONE:
        row["type"] = event_type_for(int(row["action_id"]), int(row["finger_id"]))
    else:
        row["type"] = event_type_for(int(row["action_id"]), int(row["finger_id"]))

    return before, row


def alignment_check(features_path, events):
    if features_path is None or not Path(features_path).exists():
        return None, ["features file missing"]

    df = pd.read_csv(features_path)
    warnings = []
    if "time_s" in df.columns:
        time_s = pd.to_numeric(df["time_s"], errors="coerce")
        time_s = time_s.dropna()
        if time_s.empty:
            warnings.append("features time_s is empty after coercion")
            return None, warnings
        feat_start, feat_end = float(time_s.min()), float(time_s.max())
    else:
        feat_start = 0.0
        feat_end = float(len(df)) / float(DEFAULT_FS)

    if events.empty:
        return {
            "features_start": feat_start,
            "features_end": feat_end,
            "events_start": 0.0,
            "events_end": 0.0,
            "overlap_seconds": 0.0,
            "coverage_pct": 0.0,
        }, warnings

    t = events[["onset_s", "duration_s"]].apply(pd.to_numeric, errors="coerce").dropna()
    if t.empty:
        warnings.append("events timing columns not numeric")
        return None, warnings

    ev_start = float(t["onset_s"].min())
    ev_end = float((t["onset_s"] + t["duration_s"]).max())

    overlap_seconds = max(0.0, min(feat_end, ev_end) - max(feat_start, ev_start))
    span = max(0.0, ev_end - ev_start)
    coverage_pct = overlap_seconds / span if span > 0 else 0.0

    metrics = {
        "features_start": feat_start,
        "features_end": feat_end,
        "events_start": ev_start,
        "events_end": ev_end,
        "overlap_seconds": overlap_seconds,
        "coverage_pct": coverage_pct,
    }

    if coverage_pct < 0.6 and span > 0:
        warnings.append(
            "Low event/feature overlap (<60%); possible mis-paired features/events. "
            "Consider running scripts/repair_features_timebase.py."
        )

    return metrics, warnings


def json_safe_dict(data):
    safe = {}
    for key, value in data.items():
        if hasattr(value, "item"):
            try:
                safe[key] = value.item()
                continue
            except Exception:
                pass
        safe[key] = value
    return safe


def summary_counts(events):
    summary = {
        "total_events": int(len(events)),
        "counts_by_action_id": events["action_id"].value_counts().sort_index().to_dict()
        if "action_id" in events.columns
        else {},
        "counts_by_finger_id": events["finger_id"].value_counts().sort_index().to_dict()
        if "finger_id" in events.columns
        else {},
        "counts_by_type": events["type"].value_counts().head(10).to_dict()
        if "type" in events.columns
        else {},
    }
    if "action_id" in events.columns:
        summary["non_rest_count"] = int((events["action_id"] != ACTION_REST).sum())
    else:
        summary["non_rest_count"] = 0
    return summary


def print_decision(exit_code, warnings):
    if exit_code == 0 and not warnings:
        print("DECISION: PASS (no action needed)")
    elif exit_code == 0 and warnings:
        print("DECISION: PASS WITH WARNINGS (consider review_events.py)")
    else:
        print("DECISION: FAIL (run review_events.py or fix issues)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply fixes in-place")
    parser.add_argument("--events", type=str, default=None, help="Override events path")
    parser.add_argument(
        "--features", type=str, default=None, help="Override features path"
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Session directory (defaults to <session_dir>/events/events.csv and <session_dir>/raw/raw.csv).",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default="2-M16",
        help="(Deprecated) Subject ID lookup is no longer supported without explicit paths.",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Exit with code 1 on warnings"
    )
    parser.add_argument(
        "--json-report", type=str, default=None, help="Write JSON report to path"
    )
    args = parser.parse_args()
    subject_id_provided = "--subject-id" in sys.argv

    explicit_events = bool(args.events)
    explicit_features = bool(args.features)
    if subject_id_provided and not args.session_dir:
        print("Session selection source: legacy_explicit")
        print(
            "❌ subject-id lookup is not supported without --session-dir. Provide explicit --events/--features."
        )
        raise SystemExit(2)
    if not args.session_dir and (not explicit_events or not explicit_features):
        print("Session selection source: legacy_explicit")
        print(
            "❌ Missing --session-dir. Provide --session-dir or explicit --events/--features."
        )
        raise SystemExit(2)

    session_used, events_path, features_path = resolve_paths(
        args.session_dir, args.events, args.features
    )
    if args.session_dir and not session_used:
        print("Session selection source: session_dir")
        print(f"Session dir not found: {args.session_dir}")
        raise SystemExit(2)

    selection_source = "session_dir"
    if explicit_events or explicit_features:
        selection_source = "legacy_explicit"
        if args.session_dir:
            print(
                "⚠️ Explicit --events/--features provided with --session-dir; using explicit paths."
            )

    print(f"Session selection source: {selection_source}")
    if args.json_report:
        print(f"Saving report to: {args.json_report}")
    if events_path is None or not Path(events_path).exists():
        print("No events file found.")
        raise SystemExit(2)

    print(f"Validating events file: {events_path}")
    if features_path is not None and Path(features_path).exists():
        print(f"Using features file: {features_path}")

    events = pd.read_csv(events_path)
    if events.empty:
        warnings = ["events file is empty; nothing to validate"]
        summary = summary_counts(events)
        report = {
            "events_path": str(events_path),
            "features_path": str(features_path) if features_path else None,
            "session_meta_used": session_used,
            "issues": [],
            "warnings": warnings,
            "summary": summary,
            "alignment": None,
        }
        print("⚠ events file is empty; nothing to validate.")
        print("\nSummary:")
        print(f"  total_events: {summary['total_events']}")
        print(f"  non_rest_count: {summary['non_rest_count']}")
        print(f"  counts_by_action_id: {summary['counts_by_action_id']}")
        print(f"  counts_by_finger_id: {summary['counts_by_finger_id']}")
        print(f"  counts_by_type (top 10): {summary['counts_by_type']}")
        if args.json_report:
            report_path = Path(args.json_report)
            report_path.write_text(json.dumps(report, indent=2))
            print(f"✅ JSON report written: {report_path}")
        exit_code = 1 if args.strict else 0
        print_decision(exit_code, warnings)
        raise SystemExit(exit_code)

    issues, warnings, missing_required, missing_optional = validate_events(events)

    if missing_required:
        missing = issues[0].get("detail", []) if issues else missing_required
        print(f"❌ Missing required columns: {missing}")
        if warnings:
            print("⚠ Warnings:")
            for warn in warnings:
                print(f"- {warn}")
        summary = summary_counts(events)
        report = {
            "events_path": str(events_path),
            "features_path": str(features_path) if features_path else None,
            "session_meta_used": session_used,
            "issues": issues,
            "warnings": warnings,
            "summary": summary,
            "alignment": None,
        }
        if args.json_report:
            report_path = Path(args.json_report)
            report_path.write_text(json.dumps(report, indent=2))
            print(f"✅ JSON report written: {report_path}")
        print_decision(1, warnings)
        raise SystemExit(1)

    alignment_metrics = None
    alignment_warnings = []
    if features_path is not None and Path(features_path).exists():
        alignment_metrics, alignment_warnings = alignment_check(features_path, events)

    warnings.extend(alignment_warnings)

    if issues:
        print("⚠ Issues detected:")
        for issue in issues:
            print(f"- row {issue.get('row')}: {issue.get('type')}")

    if warnings:
        print("⚠ Warnings:")
        for warn in warnings:
            print(f"- {warn}")

    if args.apply:
        if not issues:
            print("No issues; nothing to apply.")
        else:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = Path(f"{events_path}.bak.{timestamp}")
            backup_path.write_text(Path(events_path).read_text())

            for issue in issues:
                row_idx = issue.get("row", -1)
                if row_idx < 0:
                    continue
                row = events.loc[row_idx].copy()
                issue_type = issue.get("type")
                if issue_type in {"invalid_action_finger"}:
                    before, after = repair_event(row)
                    events.loc[row_idx] = after
                    log_event_edit(
                        "repair",
                        json_safe_dict(before),
                        json_safe_dict(dict(after)),
                        note=issue_type,
                    )
                if issue_type == "negative_onset":
                    before = dict(row)
                    events.at[row_idx, "onset_s"] = 0.0
                    after = dict(events.loc[row_idx])
                    log_event_edit(
                        "repair",
                        json_safe_dict(before),
                        json_safe_dict(after),
                        note=issue_type,
                    )
                if issue_type == "negative_duration":
                    before = dict(row)
                    events.at[row_idx, "duration_s"] = 0.0
                    after = dict(events.loc[row_idx])
                    log_event_edit(
                        "repair",
                        json_safe_dict(before),
                        json_safe_dict(after),
                        note=issue_type,
                    )

            events.to_csv(events_path, index=False)
            print(f"✅ Applied fixes and saved {events_path}")
            print(f"✅ Backup saved to {backup_path}")

            events = pd.read_csv(events_path)
            issues, warnings, missing_required, missing_optional = validate_events(
                events
            )
            if missing_required:
                print("❌ Missing required columns after apply; validation failed.")
                issues = issues or [{"row": -1, "type": "missing_required_columns"}]
            alignment_metrics = None
            alignment_warnings = []
            if features_path is not None and Path(features_path).exists():
                alignment_metrics, alignment_warnings = alignment_check(
                    features_path, events
                )
            warnings.extend(alignment_warnings)

    summary = summary_counts(events)
    print("\nSummary:")
    print(f"  total_events: {summary['total_events']}")
    print(f"  non_rest_count: {summary['non_rest_count']}")
    print(f"  counts_by_action_id: {summary['counts_by_action_id']}")
    print(f"  counts_by_finger_id: {summary['counts_by_finger_id']}")
    print(f"  counts_by_type (top 10): {summary['counts_by_type']}")

    report = {
        "events_path": str(events_path),
        "features_path": str(features_path) if features_path else None,
        "session_meta_used": session_used,
        "issues": issues,
        "warnings": warnings,
        "summary": summary,
        "alignment": alignment_metrics,
    }

    if args.json_report:
        report_path = Path(args.json_report)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"✅ JSON report written: {report_path}")

    exit_code = 0
    if issues:
        exit_code = 1
    if args.strict and warnings:
        exit_code = 1

    print_decision(exit_code, warnings)

    if not args.apply and issues:
        print("Run with --apply to fix safe issues.")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
