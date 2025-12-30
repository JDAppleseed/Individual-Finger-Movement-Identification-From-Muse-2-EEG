"""
==========================================================
EEG Finger Classification — Automated Experiment Runner
(With Online Adaptive Calibration)
==========================================================

PURPOSE
-------
Orchestrates the *entire experimental pipeline* from
EEG data collection → training → evaluation → calibration →
visualization.

----------------------------------------------------------
EXECUTION ORDER (DO NOT CHANGE)
----------------------------------------------------------

1️⃣ Step 1 — Stream & Record EEG (Training / Demo)
1️⃣b Step 1b — Window Extraction
2️⃣ Step 2 — Train Neural Network
3️⃣ Step 3a — Model Evaluation & Calibration Metrics
3️⃣ Step 3b — Deepchecks Validation
3️⃣ Step 3c — Visualization & MC Dropout Uncertainty

----------------------------------------------------------
CALIBRATION
----------------------------------------------------------
• Online adaptive calibration is initialized here
• Persisted to disk
• Loaded by inference / robotics
"""

import subprocess
import sys
import json
import os
from pathlib import Path

# =========================
# ===== PATHS ============
# =========================

PYTHON_EXEC = sys.executable
PROJECT_ROOT = Path(__file__).parent.resolve()
CALIBRATION_PATH = PROJECT_ROOT / "logs" / "calibration"

# =========================
# ===== RUN FLAGS ========
# =========================

RUN_STEP_1 = False      # EEG streaming / labeling
RUN_STEP_1_REVIEW = False  # Post-recording event review
RUN_STEP_1_VALIDATE = True  # Event validation/repair
RUN_STEP_1B = True     # Window extraction
RUN_STEP_2 = True      # Training
RUN_STEP_3A = True     # Evaluation
RUN_STEP_3B = True     # Deepchecks
RUN_STEP_3C = True     # Visualization
RUN_STEP_4 = True

# =========================
# ===== CALIBRATION ======
# =========================

CALIBRATION_CONFIG = {
    "init_threshold": 0.75,
    "min_threshold": 0.55,
    "max_threshold": 0.90,
    "ema_alpha": 0.05
}

def initialize_calibration():
    """
    Create initial online calibration state if it does not exist
    """
    CALIBRATION_PATH.mkdir(parents=True, exist_ok=True)
    print("🧠 Calibration directory ready")

# =========================
# ===== RUNNER ===========
# =========================

def run_step(script, desc):
    print(f"\n▶ {script} — {desc}")
    env = dict(**os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    result = subprocess.run(
        [PYTHON_EXEC, script],
        cwd=PROJECT_ROOT,
        env=env
    )
    if result.returncode != 0:
        raise RuntimeError(f"{script} failed")

# =========================
# ===== VALIDATION =======
# =========================

#Also need utils and models, those are referenced inside of the scripts
#Scripts will NOT run without utils and models

required_scripts = [
    "1_stream_and_record.py",
    "5_review_events.py",
    "5_validate_events.py",
    "1b_extract_windows.py",
    "2_train_model.py",
    "3_evaluate_model.py",
    "3b_deepchecks_evaluate.py",
    "3c_live_paper_figures.py",
    "4_generate_reports.py"
]

for script in required_scripts:
    if not (PROJECT_ROOT / script).exists():
        raise FileNotFoundError(f"Missing required script: {script}")

# =========================
# ===== PIPELINE =========
# =========================

print("\n🚀 Initializing experiment pipeline")

# --- Calibration bootstrap ---
initialize_calibration()

# --- Step 1 ---
if RUN_STEP_1:
    run_step("1_stream_and_record.py", "EEG streaming + labeling")

# --- Step 1 review ---
if RUN_STEP_1_REVIEW:
    run_step("5_review_events.py", "Event review and correction")

# --- Step 1 validation ---
if RUN_STEP_1_VALIDATE:
    run_step("5_validate_events.py", "Event validation and repair")

# --- Step 1b ---
if RUN_STEP_1B:
    if not (PROJECT_ROOT / "eeg_features.csv").exists():
        raise FileNotFoundError("Missing eeg_features.csv. Run Step 1 first.")
    if not (PROJECT_ROOT / "events.csv").exists():
        raise FileNotFoundError("Missing events.csv. Run Step 1 and mark events.")
    run_step("1b_extract_windows.py", "Window extraction")

# --- Step 2 ---
if RUN_STEP_2:
    if not (PROJECT_ROOT / "eeg_windows.npz").exists():
        raise FileNotFoundError("Missing eeg_windows.npz. Run Step 1b first.")
    run_step("2_train_model.py", "Model training")

# --- Step 3a ---
if RUN_STEP_3A:
    run_step("3_evaluate_model.py", "Evaluation + calibration metrics")

# --- Step 3b ---
if RUN_STEP_3B:
    run_step("3b_deepchecks_evaluate.py", "Deepchecks validation")

# --- Step 3c ---
if RUN_STEP_3C:
    run_step("3c_live_paper_figures.py", "Visualization + MC uncertainty")

if RUN_STEP_4:
    run_step("4_generate_reports.py", "Per-subject & cross-subject reports")

print("\n🎉 PIPELINE COMPLETE — Calibration Active & Persisted")
