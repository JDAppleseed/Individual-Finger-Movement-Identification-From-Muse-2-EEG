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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from utils.label_schema import (
    ACTION_REST,
    ACTION_NAMES,
    FINGER_NAMES,
    enforce_prediction_pairs,
)
from utils.per_subject_calibration import expected_calibration_error
from utils.sequence_data import load_sequence_npz

REPORT_DIR = Path("reports/subjects")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CALIB_DIR = Path("logs/calibration")
EXP_LOG_DIR = Path("logs/experiments")

ACTION_LABELS = sorted(ACTION_NAMES.keys())
ACTION_TICK_LABELS = [ACTION_NAMES[label] for label in ACTION_LABELS]
FINGER_LABELS = sorted(FINGER_NAMES.keys())
FINGER_TICK_LABELS = [FINGER_NAMES[label] for label in FINGER_LABELS]


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
        "test_indices": data["test_indices"]
        if "test_indices" in data
        else (
            data["test_indices_global"]
            if "test_indices_global" in data
            else (data["test_indices_local"] if "test_indices_local" in data else None)
        ),
    }


def _safe_pct(value):
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value * 100:.2f}%"


def _safe_float(value):
    if value is None or np.isnan(value):
        return "N/A"
    return f"{value:.3f}"


def _plot_confusion_matrix(cm, labels, tick_labels, *, title, cmap, out_path):
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if tick_labels:
        plt.xticks(
            range(len(labels)),
            tick_labels,
            rotation=45,
            ha="right",
            fontsize=8,
        )
        plt.yticks(range(len(labels)), tick_labels, fontsize=8)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


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
    joint_acc = None
    joint_acc_non_rest = None
    finger_acc = None
    action_cm = None
    finger_cm = None

    if preds is not None:
        _, _, _, meta = load_sequence_npz("eeg_windows.npz")
        test_idx = preds["test_indices"]
        action_probs = preds["action_probs"]
        finger_probs = preds["finger_probs"]
        action_preds = np.argmax(action_probs, axis=1)
        _, finger_preds = enforce_prediction_pairs(
            action_preds, np.argmax(finger_probs, axis=1)
        )

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
            action_cm = confusion_matrix(
                y_action_subj, action_preds_subj, labels=ACTION_LABELS
            )

            mask = y_action_subj != 0
            joint_correct = (
                (action_preds_subj == y_action_subj)
                & (finger_preds_subj == y_finger_subj)
            )
            joint_acc = float(np.mean(joint_correct)) if len(joint_correct) else None
            if mask.any():
                joint_acc_non_rest = float(np.mean(joint_correct[mask]))
            if mask.any():
                finger_acc = accuracy_score(
                    y_finger_subj[mask], finger_preds_subj[mask]
                )
                finger_cm = confusion_matrix(
                    y_finger_subj[mask],
                    finger_preds_subj[mask],
                    labels=FINGER_LABELS,
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
        _plot_confusion_matrix(
            action_cm,
            ACTION_LABELS,
            ACTION_TICK_LABELS,
            title="Action Confusion",
            cmap="Blues",
            out_path=cm_path,
        )
        confusion_html += f'<img src="{cm_path.name}" width="400"/>'

    if finger_cm is not None:
        cm_path = subject_report_dir / "finger_confusion.png"
        _plot_confusion_matrix(
            finger_cm,
            FINGER_LABELS,
            FINGER_TICK_LABELS,
            title="Finger Confusion (non-REST)",
            cmap="Greens",
            out_path=cm_path,
        )
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
        <li>Joint Accuracy (test set, overall/non-REST): {_safe_pct(joint_acc)} / {_safe_pct(joint_acc_non_rest)}</li>
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


def generate_run_report(run_dir: Path, *, out_dir: Path) -> Path:
    """
    Session/run-scoped report generator.

    This is the canonical pathing entrypoint used by Step 4 when a session_dir is provided:
      <session_dir>/processed/reports/<run_id>/report.html
    """
    run_dir = Path(run_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.json"
    train_config_path = run_dir / "train_config.json"
    preds_path = run_dir / "test_predictions.npz"
    eval_manifest_path = out_dir / "eval_manifest.json"

    metrics = {}
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
        except Exception:
            metrics = {}
    eval_manifest = {}
    if eval_manifest_path.exists():
        try:
            eval_manifest = json.loads(eval_manifest_path.read_text())
        except Exception:
            eval_manifest = {}

    preds = None
    if preds_path.exists():
        try:
            data = np.load(preds_path)
            action_probs = np.asarray(data["action_probs"])
            finger_probs = np.asarray(data["finger_probs"])
            y_action = np.asarray(data["y_action"]).astype(int)
            y_finger = np.asarray(data["y_finger"]).astype(int)
            preds = (action_probs, finger_probs, y_action, y_finger)
        except Exception:
            preds = None

    action_acc = None
    action_f1_macro = None
    action_f1_weighted = None
    raw_valid_pair_rate = None
    raw_invalid_pair_rate = None
    joint_acc = None
    joint_acc_non_rest = None
    finger_acc_non_rest = None
    finger_acc_overall = None
    finger_f1_non_rest_macro = None
    finger_f1_non_rest_weighted = None
    finger_f1_overall_macro = None
    finger_f1_overall_weighted = None
    rest_tpr = None
    rest_fpr = None
    rest_precision = None
    rest_f1 = None
    action_cm = None
    finger_cm = None
    if preds is not None:
        action_probs, finger_probs, y_action, y_finger = preds
        action_pred = np.argmax(action_probs, axis=1).astype(int)
        raw_finger_pred = np.argmax(finger_probs, axis=1).astype(int)
        raw_valid_pair_rate = (
            float(np.mean((action_pred != ACTION_REST) | (raw_finger_pred == 0)))
            if y_action.size
            else None
        )
        raw_invalid_pair_rate = (
            float(1.0 - raw_valid_pair_rate)
            if raw_valid_pair_rate is not None
            else None
        )
        _, finger_pred = enforce_prediction_pairs(
            action_pred, raw_finger_pred
        )
        if y_action.size:
            action_acc = float(accuracy_score(y_action, action_pred))
            action_f1_macro = float(
                f1_score(y_action, action_pred, average="macro", zero_division=0)
            )
            action_f1_weighted = float(
                f1_score(y_action, action_pred, average="weighted", zero_division=0)
            )
            action_cm = confusion_matrix(y_action, action_pred, labels=ACTION_LABELS)
            rest_mask = y_action == ACTION_REST
            if np.any(rest_mask):
                rest_tp = int(np.sum(rest_mask & (action_pred == ACTION_REST)))
                rest_fn = int(np.sum(rest_mask & (action_pred != ACTION_REST)))
                rest_fp = int(np.sum(~rest_mask & (action_pred == ACTION_REST)))
                rest_tn = int(np.sum(~rest_mask & (action_pred != ACTION_REST)))
                rest_tpr = float(rest_tp / (rest_tp + rest_fn)) if (rest_tp + rest_fn) else None
                rest_fpr = float(rest_fp / (rest_fp + rest_tn)) if (rest_fp + rest_tn) else None
                rest_precision = (
                    float(rest_tp / (rest_tp + rest_fp)) if (rest_tp + rest_fp) else None
                )
                if rest_precision is not None and rest_tpr is not None:
                    denom = rest_precision + rest_tpr
                    rest_f1 = float(2 * rest_precision * rest_tpr / denom) if denom else None
        mask = y_action != ACTION_REST
        if y_action.size:
            joint_correct = (action_pred == y_action) & (finger_pred == y_finger)
            joint_acc = float(np.mean(joint_correct))
            if np.any(mask):
                joint_acc_non_rest = float(np.mean(joint_correct[mask]))
        if np.any(mask):
            finger_acc_non_rest = float(accuracy_score(y_finger[mask], finger_pred[mask]))
            finger_f1_non_rest_macro = float(
                f1_score(y_finger[mask], finger_pred[mask], average="macro", zero_division=0)
            )
            finger_f1_non_rest_weighted = float(
                f1_score(
                    y_finger[mask], finger_pred[mask], average="weighted", zero_division=0
                )
            )
            finger_cm = confusion_matrix(
                y_finger[mask], finger_pred[mask], labels=FINGER_LABELS
            )
        if y_finger.size:
            finger_acc_overall = float(accuracy_score(y_finger, finger_pred))
            finger_f1_overall_macro = float(
                f1_score(y_finger, finger_pred, average="macro", zero_division=0)
            )
            finger_f1_overall_weighted = float(
                f1_score(y_finger, finger_pred, average="weighted", zero_division=0)
            )

    confusion_html = ""
    if action_cm is not None:
        cm_path = out_dir / "action_confusion.png"
        _plot_confusion_matrix(
            action_cm,
            ACTION_LABELS,
            ACTION_TICK_LABELS,
            title="Action Confusion",
            cmap="Blues",
            out_path=cm_path,
        )
        confusion_html += f'<img src="{cm_path.name}" width="400"/>'

    if finger_cm is not None:
        cm_path = out_dir / "finger_confusion.png"
        _plot_confusion_matrix(
            finger_cm,
            FINGER_LABELS,
            FINGER_TICK_LABELS,
            title="Finger Confusion (non-REST)",
            cmap="Greens",
            out_path=cm_path,
        )
        confusion_html += f'<img src="{cm_path.name}" width="400"/>'

    metrics_pre = json.dumps(metrics, indent=2) if metrics else "{}"
    manifest_metrics = eval_manifest.get("metrics", {}) if isinstance(eval_manifest, dict) else {}
    benchmarks = eval_manifest.get("benchmarks", {}) if isinstance(eval_manifest, dict) else {}
    primary_benchmark = benchmarks.get("primary_mixed_holdout") or {}
    aux_rest_benchmark = benchmarks.get("aux_rest_only") or {}
    core_replay_benchmark = benchmarks.get("core_full_session_replay") or {}
    repeated_summary = eval_manifest.get("repeated_split_summary") or {}
    candidate_event_flags = eval_manifest.get("candidate_event_flags") or []
    rest_event_breakdown = eval_manifest.get("rest_event_breakdown") or []

    if manifest_metrics:
        action_acc = manifest_metrics.get("action_acc", action_acc)
        action_f1_macro = manifest_metrics.get("action_f1_macro", action_f1_macro)
        action_f1_weighted = manifest_metrics.get("action_f1_weighted", action_f1_weighted)
        raw_valid_pair_rate = manifest_metrics.get("raw_valid_pair_rate", raw_valid_pair_rate)
        raw_invalid_pair_rate = manifest_metrics.get("raw_invalid_pair_rate", raw_invalid_pair_rate)
        joint_acc = manifest_metrics.get("joint_acc", joint_acc)
        joint_acc_non_rest = manifest_metrics.get("joint_acc_non_rest", joint_acc_non_rest)
        finger_acc_non_rest = manifest_metrics.get("finger_acc_non_rest", finger_acc_non_rest)
        finger_acc_overall = manifest_metrics.get("finger_acc_overall", finger_acc_overall)
        rest_tpr = manifest_metrics.get("rest_tpr", rest_tpr)
        rest_fpr = manifest_metrics.get("rest_fpr", rest_fpr)
        rest_precision = manifest_metrics.get("rest_precision", rest_precision)
        rest_f1 = manifest_metrics.get("rest_f1", rest_f1)

    primary_html = ""
    if primary_benchmark:
        primary_metrics = primary_benchmark.get("metrics", {})
        primary_html = f"""
        <h2>Primary Mixed Holdout</h2>
        <ul>
          <li>Test windows: {primary_benchmark.get("test_n", "N/A")}</li>
          <li>Action counts: {primary_benchmark.get("test_action_counts", "N/A")}</li>
          <li>Finger counts: {primary_benchmark.get("test_finger_counts", "N/A")}</li>
          <li>Action accuracy: {_safe_pct(primary_metrics.get("action_acc"))}</li>
          <li>Joint accuracy: {_safe_pct(primary_metrics.get("joint_acc"))}</li>
          <li>Finger accuracy (non-REST): {_safe_pct(primary_metrics.get("finger_acc_non_rest"))}</li>
          <li>REST TPR / Precision: {_safe_pct(primary_metrics.get("rest_tpr"))} / {_safe_pct(primary_metrics.get("rest_precision"))}</li>
        </ul>
        """

    aux_html = ""
    if aux_rest_benchmark:
        aux_html = f"""
        <h2>Auxiliary Quiet-REST Benchmark</h2>
        <ul>
          <li>Windows: {aux_rest_benchmark.get("n_windows", "N/A")}</li>
          <li>REST events: {aux_rest_benchmark.get("n_rest_events", "N/A")}</li>
          <li>Sessions: {", ".join(aux_rest_benchmark.get("sessions", [])) or "N/A"}</li>
          <li>Action accuracy: {_safe_pct(aux_rest_benchmark.get("action_acc"))}</li>
          <li>REST TPR / Precision / F1: {_safe_pct(aux_rest_benchmark.get("rest_tpr"))} / {_safe_pct(aux_rest_benchmark.get("rest_precision"))} / {_safe_float(aux_rest_benchmark.get("rest_f1"))}</li>
        </ul>
        """

    core_html = ""
    if core_replay_benchmark:
        core_metrics = core_replay_benchmark.get("metrics", {})
        core_html = f"""
        <h2>Core Full-Session Replay</h2>
        <ul>
          <li>Windows: {core_replay_benchmark.get("n_windows", "N/A")}</li>
          <li>Sessions: {", ".join(core_replay_benchmark.get("sessions", [])) or "N/A"}</li>
          <li>Action accuracy: {_safe_pct(core_metrics.get("action_acc"))}</li>
          <li>Joint accuracy: {_safe_pct(core_metrics.get("joint_acc"))}</li>
          <li>Finger accuracy (non-REST): {_safe_pct(core_metrics.get("finger_acc_non_rest"))}</li>
          <li>REST TPR / Precision: {_safe_pct(core_metrics.get("rest_tpr"))} / {_safe_pct(core_metrics.get("rest_precision"))}</li>
        </ul>
        """

    repeated_html = ""
    if repeated_summary:
        repeated_html = f"""
        <h2>Repeated Split Summary</h2>
        <ul>
          <li>Seeds: {", ".join(str(v) for v in repeated_summary.get("seeds", [])) or "N/A"}</li>
          <li>Action accuracy mean/std: {_safe_pct(repeated_summary.get("mean", {}).get("action_acc"))} / {_safe_pct(repeated_summary.get("std", {}).get("action_acc"))}</li>
          <li>Joint accuracy mean/std: {_safe_pct(repeated_summary.get("mean", {}).get("joint_acc"))} / {_safe_pct(repeated_summary.get("std", {}).get("joint_acc"))}</li>
          <li>Finger accuracy non-REST mean/std: {_safe_pct(repeated_summary.get("mean", {}).get("finger_acc_non_rest"))} / {_safe_pct(repeated_summary.get("std", {}).get("finger_acc_non_rest"))}</li>
          <li>REST TPR mean/std: {_safe_pct(repeated_summary.get("mean", {}).get("rest_tpr"))} / {_safe_pct(repeated_summary.get("std", {}).get("rest_tpr"))}</li>
          <li>REST precision mean/std: {_safe_pct(repeated_summary.get("mean", {}).get("rest_precision"))} / {_safe_pct(repeated_summary.get("std", {}).get("rest_precision"))}</li>
        </ul>
        """

    flags_html = ""
    if candidate_event_flags:
        rows = "".join(
            f"<tr><td>{flag.get('event_id')}</td><td>{flag.get('session_id')}</td><td>{_safe_pct(flag.get('rest_tpr'))}</td><td>{flag.get('dominant_non_rest_pair') or 'N/A'}</td><td>{_safe_pct(flag.get('dominant_non_rest_pair_share'))}</td><td>{flag.get('recommended_action')}</td><td>{flag.get('reason')}</td></tr>"
            for flag in candidate_event_flags
        )
        flags_html = f"""
        <h2>Flagged REST Events</h2>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr><th>Event</th><th>Session</th><th>REST TPR</th><th>Dominant non-REST pair</th><th>Dominant share</th><th>Recommendation</th><th>Reason</th></tr>
          {rows}
        </table>
        """

    rest_breakdown_html = ""
    if rest_event_breakdown:
        rows = "".join(
            f"<tr><td>{row.get('event_id')}</td><td>{row.get('session_id')}</td><td>{row.get('window_count')}</td><td>{_safe_pct(row.get('rest_tpr'))}</td><td>{row.get('pred_action_counts')}</td><td>{json.dumps(row.get('pred_pair_counts', {}))}</td><td>{_safe_float(row.get('median_p_rest'))}</td><td>{_safe_float(row.get('median_p_open'))}</td><td>{_safe_float(row.get('median_p_close'))}</td><td>{_safe_float(row.get('window_start'))}</td><td>{_safe_float(row.get('window_end'))}</td></tr>"
            for row in rest_event_breakdown
        )
        rest_breakdown_html = f"""
        <h2>REST Event Breakdown</h2>
        <table border="1" cellpadding="4" cellspacing="0">
          <tr><th>Event</th><th>Session</th><th>Windows</th><th>REST TPR</th><th>Pred action counts</th><th>Pred pair counts</th><th>Median p(REST)</th><th>Median p(OPEN)</th><th>Median p(CLOSE)</th><th>Start</th><th>End</th></tr>
          {rows}
        </table>
        """

    html = f"""
    <html>
    <head><title>BCI Run Report - {run_dir.name}</title></head>
    <body>
    <h1>EEG BCI Run Report</h1>

    <h2>Run</h2>
    <ul>
      <li><b>run_dir</b>: {run_dir}</li>
      <li><b>out_dir</b>: {out_dir}</li>
      <li><b>generated</b>: {datetime.now(timezone.utc).isoformat()}</li>
    </ul>

    <h2>Artifacts</h2>
    <ul>
      <li><b>metrics</b>: {metrics_path.name if metrics_path.exists() else "missing"}</li>
      <li><b>train_config</b>: {train_config_path.name if train_config_path.exists() else "missing"}</li>
      <li><b>test_predictions</b>: {preds_path.name if preds_path.exists() else "missing"}</li>
    </ul>

    <h2>Performance (from predictions)</h2>
    <ul>
      <li>Action accuracy: {_safe_pct(action_acc)}</li>
      <li>Action F1 (macro/weighted): {_safe_float(action_f1_macro)} / {_safe_float(action_f1_weighted)}</li>
      <li>Raw valid/invalid pair rate: {_safe_pct(raw_valid_pair_rate)} / {_safe_pct(raw_invalid_pair_rate)}</li>
      <li>Joint accuracy (overall/non-REST): {_safe_pct(joint_acc)} / {_safe_pct(joint_acc_non_rest)}</li>
      <li>Finger accuracy (non-REST): {_safe_pct(finger_acc_non_rest)}</li>
      <li>Finger F1 (non-REST macro/weighted): {_safe_float(finger_f1_non_rest_macro)} / {_safe_float(finger_f1_non_rest_weighted)}</li>
      <li>Finger accuracy (overall): {_safe_pct(finger_acc_overall)}</li>
      <li>Finger F1 (overall macro/weighted): {_safe_float(finger_f1_overall_macro)} / {_safe_float(finger_f1_overall_weighted)}</li>
      <li>REST TPR: {_safe_pct(rest_tpr)} | REST FPR: {_safe_pct(rest_fpr)} | REST Precision: {_safe_pct(rest_precision)} | REST F1: {_safe_float(rest_f1)}</li>
    </ul>

    {primary_html}
    {aux_html}
    {core_html}
    {repeated_html}
    {flags_html}
    {rest_breakdown_html}

    <h2>Confusion Matrices</h2>
    {confusion_html or "<p><i>Predictions unavailable; skipping matrices.</i></p>"}

    <h2>metrics.json</h2>
    <pre>{metrics_pre}</pre>

    </body>
    </html>
    """
    report_path = out_dir / "report.html"
    report_path.write_text(html)
    return report_path


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
