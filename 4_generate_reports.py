"""
STEP 4 — Automatic Subject Report Generation
"""

import argparse
import json

from utils.report_generator import (
    generate_subject_report,
    generate_cross_subject_summary
)
from utils.experiment_logger import get_latest_experiment_hash
from utils.experiment_logger import LOG_DIR

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-id", type=str, default="1-M17", help="Subject ID to report")
    parser.add_argument("--exp-hash", type=str, default=None, help="Override experiment hash")
    args = parser.parse_args()

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
