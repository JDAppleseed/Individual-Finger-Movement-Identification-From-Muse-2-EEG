"""
Per-Subject Calibration Tracking & Visualization
------------------------------------------------
• Stores confidence vs correctness per subject
• Computes per-subject ECE
• Generates calibration curves
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

CALIB_DIR = Path("logs/calibration")
CALIB_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# ===== STORAGE ===========
# =========================


def record_prediction(
    subject_id: str,
    experiment_hash: str,
    confidence: float,
    uncertainty: float,
    correct: bool,
    threshold: float,
):
    path = CALIB_DIR / f"{subject_id}_{experiment_hash}.json"

    entry = {
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "correct": bool(correct),
        "threshold": float(threshold),
    }

    if path.exists():
        data = json.loads(path.read_text())
        data.append(entry)
    else:
        data = [entry]

    path.write_text(json.dumps(data, indent=2))


# =========================
# ===== METRICS ===========
# =========================


def expected_calibration_error(conf, correct, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        idx = ((conf >= bins[i]) if i == 0 else (conf > bins[i])) & (conf <= bins[i + 1])
        if np.sum(idx) == 0:
            continue

        bin_acc = np.mean(correct[idx])
        bin_conf = np.mean(conf[idx])
        ece += abs(bin_acc - bin_conf) * (np.sum(idx) / len(conf))

    return ece


# =========================
# ===== PLOTTING =========
# =========================


def plot_subject_calibration(subject_id: str, experiment_hash: str):
    path = CALIB_DIR / f"{subject_id}_{experiment_hash}.json"
    if not path.exists():
        raise FileNotFoundError("No calibration data found")

    data = json.loads(path.read_text())

    conf = np.array([d["confidence"] for d in data])
    correct = np.array([d["correct"] for d in data])
    thresholds = np.array([d["threshold"] for d in data])

    ece = expected_calibration_error(conf, correct)

    # Reliability bins
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_acc = []

    for i in range(10):
        idx = (conf > bins[i]) & (conf <= bins[i + 1])
        if np.sum(idx) == 0:
            bin_acc.append(np.nan)
        else:
            bin_acc.append(np.mean(correct[idx]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # --- Reliability Diagram ---
    axs[0].plot([0, 1], [0, 1], "--", color="gray")
    axs[0].bar(bin_centers, bin_acc, width=0.08)
    axs[0].set_title(f"Reliability Diagram\nECE = {ece:.3f}")
    axs[0].set_xlabel("Confidence")
    axs[0].set_ylabel("Accuracy")

    # --- Confidence Histogram ---
    axs[1].hist(conf, bins=20)
    axs[1].set_title("Confidence Distribution")
    axs[1].set_xlabel("Confidence")

    # --- Threshold Drift ---
    axs[2].plot(thresholds)
    axs[2].set_title("Adaptive Threshold Drift")
    axs[2].set_xlabel("Prediction Index")
    axs[2].set_ylabel("Threshold")

    plt.tight_layout()
    plt.show()
