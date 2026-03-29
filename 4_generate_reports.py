"""
STEP 4 — Automatic Subject Report Generation
"""

import argparse
import json
from pathlib import Path

from utils.report_generator import (
    generate_subject_report,
    generate_run_report,
    generate_cross_subject_summary,
)
from utils.experiment_logger import LOG_DIR
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir

# Pipeline handoff: Step 4 summarizes Step 2/3 outputs from either a concrete run
# directory or an explicit experiment hash.

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Step 4: generate run-level or subject-level reports from Step 2 "
            "and Step 3 artifacts."
        )
    )
    selection_group = parser.add_argument_group("report selection")
    selection_group.add_argument(
        "--session-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Canonical session directory. Reports are written under processed/reports/<run_id>/.",
    )
    selection_group.add_argument(
        "--run-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Specific model run directory to summarize. Defaults to the latest run under the session.",
    )
    selection_group.add_argument(
        "--subject-id",
        type=str,
        default="",
        metavar="ID",
        help=(
            "Subject identifier used only for legacy --exp-hash report "
            "generation. Defaults to the subject recorded in the experiment log."
        ),
    )
    selection_group.add_argument(
        "--exp-hash",
        type=str,
        default=None,
        metavar="HASH",
        help="Legacy experiment hash override for report generation outside a session directory.",
    )
    args = parser.parse_args()

    if args.session_dir:
        session_dir_path = resolve_session_dir(str(args.session_dir))
        if not session_dir_path.exists():
            print("Session selection source: session_dir")
            print(f"Session dir not found: {session_dir_path}")
            raise SystemExit(2)
        if args.run_dir:
            print("⚠️ Explicit --run-dir provided with --session-dir; using explicit run dir.")
        run_dir_path = (
            Path(str(args.run_dir)).expanduser()
            if args.run_dir
            else resolve_latest_run_dir(session_dir_path)
        )
        if run_dir_path is None or not run_dir_path.exists():
            print("Session selection source: session_dir")
            print(
                "No model run directory found. Train a model first (Step 2), or pass --run-dir."
            )
            raise SystemExit(2)
        out_dir = SessionLayout(session_dir_path).reports_root / run_dir_path.name
        print("Session selection source: session_dir")
        print(f"Using run dir: {run_dir_path}")
        # Run report consumes artifacts that were produced under this exact run folder.
        report_path = generate_run_report(run_dir_path, out_dir=out_dir)
        print(f"Saving report to: {report_path}")
        print(f"✅ Run report generated: {report_path}")
        return

    if not args.run_dir and not args.exp_hash:
        print("Session selection source: legacy_explicit")
        print(
            "❌ Missing --session-dir. Provide --session-dir or explicit --run-dir/--exp-hash."
        )
        raise SystemExit(2)

    print("Session selection source: legacy_explicit")

    if args.run_dir:
        run_dir_path = Path(str(args.run_dir)).expanduser()
        if not run_dir_path.exists():
            print(f"Run dir not found: {run_dir_path}")
            raise SystemExit(2)
        out_dir = Path("reports") / "runs" / run_dir_path.name
        report_path = generate_run_report(run_dir_path, out_dir=out_dir)
        print(f"Saving report to: {report_path}")
        print(f"✅ Run report generated: {report_path}")
        return

    # Generate reports for explicit experiment hash
    exp_hash = args.exp_hash

    log_path = LOG_DIR / f"{exp_hash}.json"
    if not log_path.exists():
        print(f"Experiment log not found: {log_path}")
        raise SystemExit(2)
    logs = json.loads(log_path.read_text())
    subject_id = args.subject_id or logs.get("subject_id", "UNKNOWN")

    print(f"🧪 Generating report for subject {subject_id}")

    generate_subject_report(subject_id, exp_hash)
    generate_cross_subject_summary()

    print("✅ Subject & cross-subject reports generated")


if __name__ == "__main__":
    main()
