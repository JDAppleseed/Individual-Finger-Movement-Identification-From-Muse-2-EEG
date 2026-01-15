"""
Automatic Per-Subject Experiment Report Generator
------------------------------------------------
- Generates HTML reports
- Includes per-subject & cross-subject calibration summaries
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from utils.per_subject_calibration import expected_calibration_error
from utils.sequence_data import load_sequence_npz

REPORT_DIR = Path("reports/subjects")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_DIR = Path("logs/calibration")
EXP_LOG_DIR = Path("logs/experiments")


# =========================
# ===== UTILITIES =========
# =========================


def load_calibration_files():
    return list(CALIB_DIR.glob("*.json"))


def load_experiment_log(exp_hash):
    path = EXP_LOG_DIR / f"{exp_hash}.json"
    return json.loads(path.read_text())


def load_test_predictions(path="test_predictions.npz"):
    npz_path = Path(path)
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    return {
        "action_probs": data["action_probs"],
        "finger_probs": data["finger_probs"],
        "y_action": data["y_action"],
        "y_finger": data["y_finger"],
        "test_indices": data["test_indices"],
    }


def _safe_pct(value):
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value * 100:.2f}%"


# =========================
# ===== SUBJECT REPORT ====
# =========================


def generate_subject_report(subject_id, experiment_hash):
    calib_path = CALIB_DIR / f"{subject_id}_{experiment_hash}.json"
    has_calib = calib_path.exists()

    if has_calib:
        data = json.loads(calib_path.read_text())
        conf = np.array([d["confidence"] for d in data])
        unc = np.array([d["uncertainty"] for d in data])
        correct = np.array([d["correct"] for d in data])
        thresh = np.array([d["threshold"] for d in data])
        acc = float(correct.mean()) if correct.size else None
        ece = float(expected_calibration_error(conf, correct)) if conf.size else None
    else:
        conf = np.array([])
        unc = np.array([])
        correct = np.array([])
        thresh = np.array([])
        acc = None
        ece = None

    preds = load_test_predictions()

    subject_report_dir = REPORT_DIR / subject_id / experiment_hash
    subject_report_dir.mkdir(parents=True, exist_ok=True)

    action_acc = None
    finger_acc = None
    action_cm = None
    finger_cm = None

    if preds is not None:
        _, _, _, meta = load_sequence_npz("eeg_windows.npz")
        test_idx = preds["test_indices"]
        action_probs = preds["action_probs"]
        finger_probs = preds["finger_probs"]
        action_preds = np.argmax(action_probs, axis=1)
        finger_preds = np.argmax(finger_probs, axis=1)

        subj_ids = meta.get("subject_id")
        exp_hashes = meta.get("experiment_hash")

        subj_mask = np.ones_like(test_idx, dtype=bool)
        if subj_ids is not None:
            subj_mask = subj_ids[test_idx] == subject_id
        if exp_hashes is not None:
            subj_mask = subj_mask & (exp_hashes[test_idx] == experiment_hash)

        if subj_mask.any():
            y_action_subj = preds["y_action"][subj_mask]
            y_finger_subj = preds["y_finger"][subj_mask]
            action_preds_subj = action_preds[subj_mask]
            finger_preds_subj = finger_preds[subj_mask]

            action_acc = accuracy_score(y_action_subj, action_preds_subj)
            action_cm = confusion_matrix(y_action_subj, action_preds_subj)

            mask = y_action_subj != 0
            if mask.any():
                finger_acc = accuracy_score(
                    y_finger_subj[mask], finger_preds_subj[mask]
                )
                finger_cm = confusion_matrix(
                    y_finger_subj[mask], finger_preds_subj[mask]
                )

    # ===== Plots =====
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    if conf.size:
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_acc = []
        for i in range(10):
            idx = (conf > bins[i]) & (conf <= bins[i + 1])
            bin_acc.append(np.mean(correct[idx]) if idx.any() else np.nan)

        axs[0, 0].plot([0, 1], [0, 1], "--", color="gray")
        axs[0, 0].bar(bin_centers, bin_acc, width=0.08)
        axs[0, 0].set_title(
            f"Reliability Diagram (ECE={ece:.3f})"
            if ece is not None
            else "Reliability Diagram"
        )

        axs[0, 1].hist(conf, bins=20)
        axs[0, 1].set_title("Confidence Distribution")

        axs[1, 0].hist(unc, bins=20)
        axs[1, 0].set_title("Predictive Uncertainty")

        axs[1, 1].plot(thresh)
        axs[1, 1].set_title("Adaptive Threshold Drift")
    else:
        for ax in axs.flatten():
            ax.axis("off")
        axs[0, 0].set_title("No calibration data available")

    plt.tight_layout()

    fig_path = subject_report_dir / "calibration_figures.png"
    plt.savefig(fig_path)
    plt.close()

    # ===== Confusion matrices =====
    confusion_html = ""
    if action_cm is not None:
        cm_path = subject_report_dir / "action_confusion.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(action_cm, cmap="Blues")
        plt.title("Action Confusion")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        confusion_html += f'<img src="{cm_path.name}" width="400"/>'

    if finger_cm is not None:
        cm_path = subject_report_dir / "finger_confusion.png"
        plt.figure(figsize=(4, 4))
        plt.imshow(finger_cm, cmap="Greens")
        plt.title("Finger Confusion (non-REST)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()
        confusion_html += f'<img src="{cm_path.name}" width="400"/>'

    ece_str = f"{ece:.4f}" if ece is not None else "N/A"

    html = f"""
    <html>
    <head><title>BCI Subject Report - {subject_id}</title></head>
    <body>
    <h1>EEG BCI Subject Report</h1>

    <h2>Subject</h2>
    <p><b>ID:</b> {subject_id}</p>
    <p><b>Experiment Hash:</b> {experiment_hash}</p>
    <p><b>Generated:</b> {datetime.now(timezone.utc).isoformat()}</p>

    <h2>Performance</h2>
    <ul>
        <li>Calibration Accuracy: {_safe_pct(acc)}</li>
        <li>Expected Calibration Error (ECE): {ece_str}</li>
        <li>Action Accuracy (test set): {_safe_pct(action_acc)}</li>
        <li>Finger Accuracy (test set, non-REST): {_safe_pct(finger_acc)}</li>
    </ul>

    <h2>Calibration & Safety Metrics</h2>
    <img src="{fig_path.name}" width="800"/>
    {"<p><i>Calibration data unavailable for this subject/session.</i></p>" if not has_calib else ""}

    <h2>Confusion Matrices</h2>
    <p>Generated from the held-out test set (if available).</p>
    {confusion_html}

    <h2>Notes</h2>
    <p>
    This report summarizes subject-specific calibration behavior,
    uncertainty, and adaptive safety thresholds for EEG-based control.
    </p>

    </body>
    </html>
    """

    report_path = subject_report_dir / "report.html"
    report_path.write_text(html)

    return report_path


# =========================
# ===== CROSS-SUBJECT =====
# =========================


def generate_cross_subject_summary():
    files = load_calibration_files()

    if not files:
        print("⚠ No calibration files found")
        return None

    subject_ece = {}
    subject_acc = {}

    for f in files:
        data = json.loads(f.read_text())
        conf = np.array([d["confidence"] for d in data])
        correct = np.array([d["correct"] for d in data])

        subject = f.stem.split("_")[0]

        subject_ece[subject] = expected_calibration_error(conf, correct)
        subject_acc[subject] = correct.mean() if correct.size else np.nan

    subjects = list(subject_ece.keys())
    eces = [subject_ece[s] for s in subjects]
    accs = [subject_acc[s] for s in subjects]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(subjects, accs)
    ax1.set_ylabel("Accuracy")

    ax2 = ax1.twinx()
    ax2.plot(subjects, eces, color="red", marker="o")
    ax2.set_ylabel("ECE")

    ax1.set_title("Cross-Subject Calibration Comparison")

    plt.tight_layout()
    out = REPORT_DIR / "cross_subject_summary.png"
    plt.savefig(out)
    plt.close()

    print(f"📊 Cross-subject summary saved to {out}")
    return out


if __name__ == "__main__":
    files = load_calibration_files()
    if not files:
        print("⚠ No calibration files found")
    else:
        for f in files:
            subject_id, experiment_hash = f.stem.split("_", 1)
            print(f"Generating report for {subject_id} {experiment_hash}")
            generate_subject_report(subject_id, experiment_hash)
        generate_cross_subject_summary()
