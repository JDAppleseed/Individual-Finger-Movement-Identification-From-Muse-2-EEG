#!/usr/bin/env python3
"""Repair features time base to align with events and recover correct windows."""

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

MIN_ROWS = 64
SEG_BREAK_NEG = -0.5
SEG_BREAK_POS = 10.0
MIN_OVERLAP_SECONDS = 30.0


def load_session_meta():
    meta_path = Path("session_meta.json")
    if not meta_path.exists():
        raise FileNotFoundError("session_meta.json not found")
    return meta_path, json.loads(meta_path.read_text())


def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def abs_path(path):
    return str(path.resolve())


def count_columns_in_lines(path, max_lines=100):
    header_cols = None
    row_counts = []
    with path.open() as f:
        header = f.readline().strip("\n")
        header_cols = len(header.split(",")) if header else 0
        for i, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row_counts.append(len(line.strip("\n").split(",")))
            if i >= max_lines:
                break
    return header_cols, row_counts


def read_header(path):
    with path.open() as f:
        header = f.readline().strip("\n")
    return header.split(",") if header else []


def normalize_columns(df, header_cols):
    if not header_cols:
        return df, ["⚠️ Empty header; keeping columns as-is"]
    notes = []
    if len(df.columns) > len(header_cols):
        df = df.iloc[:, : len(header_cols)]
        notes.append("⚠️ Trimmed extra columns to match header length")
    if len(df.columns) < len(header_cols):
        raise RuntimeError("Data has fewer columns than header")
    df.columns = header_cols
    return df, notes


def load_features(path):
    try:
        df = pd.read_csv(path)
        return df, []
    except Exception:
        pass

    df = pd.read_csv(path, engine="python")
    notes = ["⚠️ pandas default read failed; used python engine"]
    unnamed = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed:
        df = df.drop(columns=unnamed)
        notes.append(f"⚠️ Dropped unnamed columns: {unnamed}")
    return df, notes


def segment_indices(lsl_ts):
    diffs = np.diff(lsl_ts)
    breaks = np.where((diffs < SEG_BREAK_NEG) | (diffs > SEG_BREAK_POS))[0]
    starts = [0]
    ends = []
    for idx in breaks:
        ends.append(idx + 1)
        starts.append(idx + 1)
    ends.append(len(lsl_ts))
    segments = [(s, e) for s, e in zip(starts, ends)]
    return segments


def score_segments(segments, lsl_ts, ev_min, ev_max):
    scored = []
    for s, e in segments:
        if e - s < MIN_ROWS:
            continue
        lsl0 = lsl_ts[s]
        seg_end = lsl_ts[e - 1] - lsl0
        overlap = max(0.0, min(seg_end, ev_max) - max(0.0, ev_min))
        score = overlap - 0.01 * ev_min
        scored.append(
            {
                "start": s,
                "end": e,
                "rows": e - s,
                "seg_end": seg_end,
                "overlap": overlap,
                "score": score,
            }
        )
    if not scored:
        return []
    scored.sort(key=lambda x: (x["score"], x["seg_end"], x["rows"]), reverse=True)
    return scored


def repair_time_s(df, lsl_col, time_col, seg_start, seg_end, ev_min):
    df_seg = df.iloc[seg_start:seg_end].copy()
    lsl = df_seg[lsl_col].astype(float).to_numpy()
    lsl0 = lsl[0]
    time_s = lsl - lsl0
    if ev_min > 0:
        time_s = time_s + ev_min
    for i in range(1, len(time_s)):
        if time_s[i] < time_s[i - 1]:
            time_s[i] = time_s[i - 1]
    df_seg[time_col] = time_s
    return df_seg, float(ev_min)


def run_extract_windows():
    return subprocess.run(
        ["python", "1b_extract_windows.py"], capture_output=True, text=True
    )


def load_window_stats():
    path = Path("eeg_windows.csv")
    if not path.exists():
        return None
    df = pd.read_csv(path)
    stats = {
        "rows": len(df),
        "action_counts": df["action_id"].value_counts().sort_index().to_dict()
        if "action_id" in df.columns
        else {},
        "finger_counts": df["finger_id"].value_counts().sort_index().to_dict()
        if "finger_id" in df.columns
        else {},
    }
    non_rest = df[df["action_id"] != 0] if "action_id" in df.columns else df.iloc[0:0]
    stats["non_rest_rows"] = len(non_rest)
    return stats


def list_feature_candidates(subject_id):
    processed = Path("data/processed")
    if not processed.exists():
        return []
    patterns = [
        f"{subject_id}_*_eeg_features.csv",
        f"{subject_id}_*_eeg_features_repaired.csv",
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(processed.glob(pat))
    return sorted(set(candidates))


def analyze_features(path, ev_min, ev_max):
    header_cols, row_counts = count_columns_in_lines(path)
    mismatch = [c for c in row_counts[:100] if c != header_cols]
    df_feat, feat_notes = load_features(path)
    header = read_header(path)
    norm_notes = []
    if header:
        df_feat, norm_notes = normalize_columns(df_feat, header)
    if "lsl_timestamp" not in df_feat.columns:
        raise RuntimeError(f"lsl_timestamp column missing in {path}")
    lsl = df_feat["lsl_timestamp"].astype(float).to_numpy()
    time_s = (
        df_feat["time_s"].astype(float).to_numpy()
        if "time_s" in df_feat.columns
        else None
    )
    segments = segment_indices(lsl)
    scored = score_segments(segments, lsl, ev_min, ev_max)
    return {
        "path": path,
        "header_cols": header_cols,
        "mismatch": mismatch,
        "notes": feat_notes + norm_notes,
        "lsl_range": (float(lsl.min()), float(lsl.max())),
        "time_range": (float(time_s.min()), float(time_s.max()))
        if time_s is not None
        else None,
        "time_nonmono": int((np.diff(time_s) < -1e-6).sum())
        if time_s is not None
        else None,
        "segments": segments,
        "scored": scored,
        "df": df_feat,
        "header": header,
    }


def action_finger_table(df):
    if df.empty:
        return {}
    grouped = df.groupby(["action_id", "finger_id"]).size().sort_index()
    return {f"{a}:{f}": int(c) for (a, f), c in grouped.items()}


def coverage_report(events_df, window_df):
    events_pairs = set(zip(events_df["action_id"], events_df["finger_id"]))
    window_pairs = (
        set(zip(window_df["action_id"], window_df["finger_id"]))
        if not window_df.empty
        else set()
    )
    missing = sorted(events_pairs - window_pairs)
    return {
        "missing_pairs": missing,
        "missing_count": len(missing),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report-json", action="store_true")
    args = parser.parse_args()

    meta_path, meta = load_session_meta()
    features_path = Path(meta.get("features_path", "eeg_features.csv"))
    events_path = Path(meta.get("events_path", "events.csv"))
    raw_path = Path(meta.get("raw_path", "data/raw/UNKNOWN_raw.csv"))
    subject_id = meta.get("subject_id", "UNKNOWN")

    report = {
        "meta": meta,
        "paths": {
            "session_meta": abs_path(meta_path),
            "features_path": abs_path(features_path),
            "events_path": abs_path(events_path),
            "raw_path": abs_path(raw_path),
        },
        "candidates": [],
        "selected": None,
        "repair": {},
        "verification": {},
    }

    print_header("DIAGNOSTIC REPORT")
    print(f"session_meta.json       : {abs_path(meta_path)}")
    print(
        f"features_path           : {abs_path(features_path)} ({features_path.exists()}) {features_path.stat().st_size if features_path.exists() else 0} bytes"
    )
    print(
        f"events_path             : {abs_path(events_path)} ({events_path.exists()}) {events_path.stat().st_size if events_path.exists() else 0} bytes"
    )
    print(
        f"raw_path                : {abs_path(raw_path)} ({raw_path.exists()}) {raw_path.stat().st_size if raw_path.exists() else 0} bytes"
    )

    if not events_path.exists():
        print("❌ Missing events file")
        raise SystemExit(1)

    df_events = pd.read_csv(events_path)
    if df_events.empty:
        print("⚠️ events file is empty")
        ev_min = 0.0
        ev_max = 0.0
    else:
        ev_min = float(df_events["onset_s"].min())
        ev_max = float((df_events["onset_s"] + df_events["duration_s"]).max())
        print(f"events range            : onset {ev_min:.4f} s → end {ev_max:.4f} s")
    report["events_range"] = {"ev_min": ev_min, "ev_max": ev_max}
    report["events_table"] = action_finger_table(df_events)

    candidates = [features_path] + list_feature_candidates(subject_id)
    candidates = [p for p in dict.fromkeys(candidates) if p.exists()]
    if not candidates:
        print("❌ No feature candidates found")
        raise SystemExit(1)

    analyses = []
    for cand in candidates:
        try:
            info = analyze_features(cand, ev_min, ev_max)
            analyses.append(info)
        except Exception as exc:
            print(f"⚠️ Failed to analyze {cand}: {exc}")

    print_header("FEATURES CANDIDATES")
    for info in analyses:
        mismatch = "YES" if info["mismatch"] else "NO"
        print(f"candidate               : {abs_path(info['path'])}")
        print(
            f"  header cols           : {info['header_cols']} (mismatch sample: {mismatch})"
        )
        if info["notes"]:
            for note in info["notes"]:
                print(f"  note                  : {note}")
        lsl_min, lsl_max = info["lsl_range"]
        print(f"  lsl range             : {lsl_min:.6f} → {lsl_max:.6f}")
        if info["time_range"]:
            tmin, tmax = info["time_range"]
            print(f"  time_s range          : {tmin:.6f} → {tmax:.6f}")
            print(f"  time_s nonmono jumps  : {info['time_nonmono']}")
        print(f"  segments              : {len(info['segments'])}")
        if info["scored"]:
            best = info["scored"][0]
            print(
                f"  best segment          : {best['start']}:{best['end']} rows={best['rows']} "
                f"seg_end={best['seg_end']:.2f}s overlap={best['overlap']:.2f}s score={best['score']:.2f}"
            )
        else:
            print("  best segment          : <none>")
        report["candidates"].append(
            {
                "path": abs_path(info["path"]),
                "header_cols": info["header_cols"],
                "mismatch": bool(info["mismatch"]),
                "notes": info["notes"],
                "lsl_range": info["lsl_range"],
                "time_range": info["time_range"],
                "time_nonmono": info["time_nonmono"],
                "segments": len(info["segments"]),
                "best_segment": info["scored"][0] if info["scored"] else None,
            }
        )

    ranked = []
    for info in analyses:
        if info["scored"]:
            best = info["scored"][0]
            ranked.append((best["score"], best["seg_end"], best["rows"], info, best))
    if not ranked:
        print("❌ No valid segments with >=64 rows in any candidate")
        raise SystemExit(1)
    ranked.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
    _, _, _, best_info, best_seg = ranked[0]

    print_header("SELECTED SEGMENT")
    print(f"selected features        : {abs_path(best_info['path'])}")
    print(
        f"segment {best_seg['start']}:{best_seg['end']} rows={best_seg['rows']} "
        f"seg_end={best_seg['seg_end']:.2f}s overlap={best_seg['overlap']:.2f}s score={best_seg['score']:.2f}"
    )
    report["selected"] = {
        "path": abs_path(best_info["path"]),
        "segment": best_seg,
    }

    if args.dry_run and not args.apply:
        print("\nDry-run only. No files written.")
        print("Next commands:")
        print("  python scripts/repair_features_timebase.py --apply --report-json")
        print("  python 1b_extract_windows.py")
        if args.report_json:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = Path("scripts") / f"repair_report_{timestamp}.json"
            report_path.write_text(json.dumps(report, indent=2))
            print(f"✅ Report written: {abs_path(report_path)}")
        return

    df_feat = best_info["df"]
    header = best_info["header"]
    if header:
        df_feat, _ = normalize_columns(df_feat, header)

    repaired, offset_applied = repair_time_s(
        df_feat,
        "lsl_timestamp",
        "time_s",
        best_seg["start"],
        best_seg["end"],
        ev_min,
    )

    if len(repaired) < MIN_ROWS or best_seg["overlap"] < MIN_OVERLAP_SECONDS:
        print("❌ Safety guard tripped: insufficient rows or overlap")
        print(f"  repaired rows    : {len(repaired)}")
        print(f"  overlap seconds  : {best_seg['overlap']:.2f}")
        print("No files written. Resolve input mismatch and retry.")
        report["repair"]["guard_fail"] = True
        report["repair"]["rows"] = len(repaired)
        report["repair"]["overlap_seconds"] = best_seg["overlap"]
        if args.report_json:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report_path = Path("scripts") / f"repair_report_{timestamp}.json"
            report_path.write_text(json.dumps(report, indent=2))
            print(f"✅ Report written: {abs_path(report_path)}")
        raise SystemExit(1)

    experiment_hash = meta.get("experiment_hash", "UNKNOWN")
    out_path = (
        Path("data/processed")
        / f"{subject_id}_{experiment_hash}_eeg_features_repaired.csv"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repaired.to_csv(out_path, index=False)

    repaired_min = float(repaired["time_s"].min())
    repaired_max = float(repaired["time_s"].max())

    print_header("REPAIR OUTPUT")
    print(f"repaired features path   : {abs_path(out_path)}")
    print(f"repaired rows            : {len(repaired)}")
    print(f"repaired time_s range    : {repaired_min:.6f} → {repaired_max:.6f}")
    print(f"offset applied (ev_min)  : {offset_applied:.6f} s")

    if repaired_max < ev_min + 60.0:
        print("⚠️ time base sanity: repaired span is < 60s beyond ev_min")
    if repaired_max < ev_max - 10.0:
        print("⚠️ event tail may be lost (repaired_max < ev_max - 10s)")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = Path(f"session_meta.json.bak.{timestamp}")
    backup.write_text(meta_path.read_text())

    meta["features_path"] = str(out_path)
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"✅ session_meta.json updated (backup: {abs_path(backup)})")

    print_header("RUNNING 1b_extract_windows.py")
    res = run_extract_windows()
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr)
        raise SystemExit(res.returncode)

    stats = load_window_stats()
    if stats is None:
        print("❌ eeg_windows.csv not found after extraction")
        raise SystemExit(1)

    df_windows = pd.read_csv("eeg_windows.csv")
    non_rest_windows = df_windows[df_windows["action_id"] != 0]

    print_header("WINDOW LABEL STATS")
    print(f"rows: {stats['rows']}")
    print(f"action_id counts: {stats['action_counts']}")
    print(f"finger_id counts: {stats['finger_counts']}")
    print(f"non-REST rows: {stats['non_rest_rows']}")

    print_header("DISTRIBUTION COMPARISON")
    print("Events action_id:finger_id counts:")
    print(action_finger_table(df_events))
    print("Windows action_id:finger_id counts (non-REST only):")
    print(action_finger_table(non_rest_windows))

    coverage = coverage_report(df_events[df_events["action_id"] != 0], non_rest_windows)
    if coverage["missing_count"]:
        print(f"⚠️ Missing action/finger pairs in windows: {coverage['missing_pairs']}")
    else:
        print("✅ All action/finger pairs present in windows")

    report["repair"] = {
        "out_path": abs_path(out_path),
        "rows": len(repaired),
        "time_s_min": repaired_min,
        "time_s_max": repaired_max,
        "offset_applied": offset_applied,
        "overlap_seconds": best_seg["overlap"],
        "segment_end": best_seg["seg_end"],
    }
    report["verification"] = {
        "window_rows": stats["rows"],
        "window_non_rest": stats["non_rest_rows"],
        "action_counts": stats["action_counts"],
        "finger_counts": stats["finger_counts"],
        "events_table": action_finger_table(df_events),
        "windows_table": action_finger_table(non_rest_windows),
        "coverage": coverage,
    }

    if (df_events["action_id"] != 0).any() and stats["non_rest_rows"] == 0:
        print("❌ HARD ERROR: non-REST events exist, but all windows are REST")
        print("Likely causes:")
        print("- wrong segment chosen")
        print("- time base mismatch remains")
        print("- event types filtered as artifact")
        print("- MIN_OVERLAP gating too strict relative to event durations")
        report["status"] = "FAIL"
        if args.report_json:
            report_path = Path("scripts") / f"repair_report_{timestamp}.json"
            report_path.write_text(json.dumps(report, indent=2))
            print(f"✅ Report written: {abs_path(report_path)}")
        raise SystemExit(1)

    report["status"] = "PASS"

    if args.report_json:
        report_path = Path("scripts") / f"repair_report_{timestamp}.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"✅ Report written: {abs_path(report_path)}")

    print("\nPASS: Repair and verification complete.")
    print("Next commands:")
    print("  python 2_train_model.py")


if __name__ == "__main__":
    main()
