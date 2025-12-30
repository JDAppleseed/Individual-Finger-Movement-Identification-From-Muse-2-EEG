"""
STEP 4 — Automatic Subject Report Generation
"""

from utils.report_generator import (
    generate_subject_report,
    generate_cross_subject_summary
)
from utils.experiment_logger import get_latest_experiment_hash
from utils.experiment_logger import LOG_DIR
import json

# Generate reports for all subjects in latest experiment

exp_hash = get_latest_experiment_hash()

logs = json.loads((LOG_DIR / f"{exp_hash}.json").read_text())
subject_id = logs["subject_id"]

print(f"🧪 Generating report for subject {subject_id}")

generate_subject_report(subject_id, exp_hash)
generate_cross_subject_summary()

print("✅ Subject & cross-subject reports generated")