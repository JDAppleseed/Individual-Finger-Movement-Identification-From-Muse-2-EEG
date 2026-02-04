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
from utils.experiment_logger import get_latest_experiment_hash
from utils.experiment_logger import LOG_DIR
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Canonical session directory (writes reports under <session_dir>/processed/reports/<run_id>/).",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Model run directory (defaults to latest under <session_dir>/processed/models/).",
    )
    parser.add_argument(
        "--subject-id", type=str, default="8-M16", help="Subject ID to report"
    )
    parser.add_argument(
        "--exp-hash", type=str, default=None, help="Override experiment hash"
    )
    args = parser.parse_args()

    if args.session_dir:
        session_dir_path = resolve_session_dir(str(args.session_dir))
        if not session_dir_path.exists():
            raise SystemExit(f"Session dir not found: {session_dir_path}")
        run_dir_path = (
            Path(str(args.run_dir)).expanduser()
            if args.run_dir
            else resolve_latest_run_dir(session_dir_path)
        )
        if run_dir_path is None or not run_dir_path.exists():
            raise SystemExit(
                "No model run directory found. Train a model first (Step 2), or pass --run-dir."
            )
        out_dir = SessionLayout(session_dir_path).reports_root / run_dir_path.name
        report_path = generate_run_report(run_dir_path, out_dir=out_dir)
        print(f"✅ Run report generated: {report_path}")
        return

    # Generate reports for all subjects in latest experiment
    exp_hash = args.exp_hash or get_latest_experiment_hash()

    logs = json.loads((LOG_DIR / f"{exp_hash}.json").read_text())
    subject_id = args.subject_id or logs.get("subject_id", "UNKNOWN")

    print(f"🧪 Generating report for subject {subject_id}")

    generate_subject_report(subject_id, exp_hash)
    generate_cross_subject_summary()

    print("✅ Subject & cross-subject reports generated")


if __name__ == "__main__":
    main()
