#!/usr/bin/env python3
"""
Build publication artifacts for paper/research_paper.tex from repo run bundles.

Strict rule: do not invent numbers. All quantitative claims come from:
  - Projects/**/processed/models/*/metrics.json
  - Projects/**/processed/models/*/test_predictions.npz
  - Projects/**/processed/models/*/train_config.json
  - Projects/**/processed/eeg_windows.npz
  - Projects/**/sessions/*/run_meta.json
  - Projects/**/sessions/*/events/events.jsonl
  - Projects/**/processed/reports/<run_id>/*.png

Outputs:
  - paper_artifacts/paper_stats.json (machine-readable)
  - paper_artifacts/paper_macros.tex (numbers as LaTeX macros)
  - paper_artifacts/tables.tex (LaTeX tables)
  - paper_artifacts/figures.tex (LaTeX figure includes)
  - paper_figures/* (copied/sanitized figure filenames)
"""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECTS_ROOT = REPO_ROOT / "Projects"
OUT_DIR = REPO_ROOT / "paper_artifacts"
FIG_DIR = REPO_ROOT / "paper_figures"

# Ensure repo root is importable even when running this script by path (sys.path[0] == scripts/).
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _safe_slug(s: str) -> str:
    s = str(s).strip()
    s = s.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _latex_escape(text: str) -> str:
    # Minimal escaping for tables/captions.
    rep = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(rep.get(ch, ch) for ch in str(text))


def expected_calibration_error(conf: np.ndarray, preds: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """
    Match 3_evaluate_model.py expected_calibration_error(): bins in (a,b].
    """
    conf = np.asarray(conf, dtype=float).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    labels = np.asarray(labels).reshape(-1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        idx = (conf > bins[i]) & (conf <= bins[i + 1])
        if int(np.sum(idx)) == 0:
            continue
        bin_acc = float(np.mean(preds[idx] == labels[idx]))
        bin_conf = float(np.mean(conf[idx]))
        ece += abs(bin_acc - bin_conf) * (float(np.sum(idx)) / float(len(conf)))
    return float(ece)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """
    Wilson score interval for binomial proportion.
    Returns (low, high) in [0,1]. If n==0 -> (nan,nan).
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    half = (z * math.sqrt((p * (1.0 - p) / n) + (z * z) / (4.0 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci_binary(x: np.ndarray, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """
    Percentile bootstrap CI for mean of binary array x in {0,1}.
    Returns (p2.5, p97.5) in [0,1]. If len(x)==0 -> (nan,nan).
    """
    x = np.asarray(x).astype(float).reshape(-1)
    if x.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return (float(lo), float(hi))


def mean_std(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr.mean()), float("nan"))
    return (float(arr.mean()), float(arr.std(ddof=1)))


def min_max(vals: List[float]) -> Tuple[float, float]:
    arr = np.asarray([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    return (float(arr.min()), float(arr.max()))


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_events_counts(events_path: Path) -> Dict[str, Any]:
    if not events_path.exists():
        return {"available": False}
    counts_by_type: Dict[str, int] = {}
    counts_by_action: Dict[str, int] = {}
    counts_by_finger: Dict[str, int] = {}
    counts_by_pair: Dict[str, int] = {}
    n_rows = 0
    suffix = events_path.suffix.lower()
    rows: List[Dict[str, Any]] = []
    if suffix == ".json":
        payload = _load_json(events_path)
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            payload = payload.get("events")
        if isinstance(payload, list):
            rows = [dict(ev) for ev in payload if isinstance(ev, dict)]
    elif suffix == ".jsonl":
        for line in events_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    else:
        with events_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    for row in rows:
        n_rows += 1
        t = (str(row.get("type") or "")).strip()
        counts_by_type[t] = counts_by_type.get(t, 0) + 1
        a = str(row.get("action_id") or "").strip()
        fi = str(row.get("finger_id") or "").strip()
        if a != "":
            counts_by_action[a] = counts_by_action.get(a, 0) + 1
        if fi != "":
            counts_by_finger[fi] = counts_by_finger.get(fi, 0) + 1
        if a != "" and fi != "":
            key = f"{a}:{fi}"
            counts_by_pair[key] = counts_by_pair.get(key, 0) + 1
    return {
        "available": True,
        "n_events": int(n_rows),
        "counts_by_type": counts_by_type,
        "counts_by_action_id": counts_by_action,
        "counts_by_finger_id": counts_by_finger,
        "counts_by_action_finger": counts_by_pair,
    }


def _npz_to_meta(npz: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the 'meta' dict shape used by utils.splitting.* (keys are 1D arrays).
    """
    meta: Dict[str, Any] = {}
    for key in [
        "trial_id",
        "block_id",
        "session_id",
        "window_start",
        "window_end",
        "subject_id",
        "experiment_hash",
    ]:
        if key in npz:
            meta[key] = np.asarray(npz[key]).reshape(-1)
    # include hop if present
    if "step_sec" in npz:
        try:
            meta["hop_seconds"] = float(np.asarray(npz["step_sec"]).item())
        except Exception:
            pass
    return meta


@dataclass(frozen=True)
class SubjectDemographics:
    subject_id: str
    age: Optional[float]
    sex: Optional[str]
    handedness: Optional[str]
    notes: Optional[str]


@dataclass(frozen=True)
class RunMetrics:
    subject_id: str
    session_id: str
    run_id: str
    created_utc: Optional[str]

    train_action_acc: Optional[float]
    train_finger_acc: Optional[float]
    train_avg_loss: Optional[float]
    train_epochs: Optional[int]
    train_batch_size: Optional[int]
    train_lr: Optional[float]
    train_seed: Optional[int]

    test_action_acc_metrics: Optional[float]
    test_finger_acc_non_rest_metrics: Optional[float]
    n_test_metrics: Optional[int]
    n_test_non_rest_metrics: Optional[int]

    test_action_acc_from_preds: Optional[float]
    test_finger_acc_non_rest_from_preds: Optional[float]
    test_action_ece: Optional[float]
    test_finger_ece_non_rest: Optional[float]

    action_ci95_wilson: Tuple[float, float]
    finger_non_rest_ci95_wilson: Tuple[float, float]
    action_ci95_boot: Tuple[float, float]
    finger_non_rest_ci95_boot: Tuple[float, float]

    # Dataset characteristics (from eeg_windows.npz)
    n_windows_total: Optional[int]
    window_sec: Optional[float]
    step_sec: Optional[float]
    target_fs: Optional[float]
    overlap_frac_mean: Optional[float]
    artifact_count: Optional[int]
    gap_count: Optional[int]

    # Label counts (test set) derived from predictions
    test_action_counts: Dict[str, int]
    test_finger_counts: Dict[str, int]

    # Per-class accuracies (test set) derived from predictions
    test_action_acc_by_class: Dict[str, float]
    test_finger_acc_by_class_non_rest: Dict[str, float]

    # Metadata richness indicators (from eeg_windows.npz, if present)
    n_unique_trial_id: Optional[int]
    n_unique_block_id: Optional[int]

    # Split metadata/config
    split_mode: Optional[str]
    test_size: Optional[float]
    purge_seconds: Optional[float]
    leakage_guard: Dict[str, Any]

    # Figure paths (copied/sanitized, relative to repo root)
    fig_action_confusion: Optional[str]
    fig_finger_confusion: Optional[str]
    fig_reliability: Optional[str]
    fig_scatter: Optional[str]


def _copy_figure(src: Path, dest_stem: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    dest = FIG_DIR / f"{dest_stem}{ext}"
    shutil.copy2(src, dest)
    return str(dest.relative_to(REPO_ROOT))


def _find_report_figs(report_dir: Path) -> Dict[str, Optional[Path]]:
    if not report_dir.exists():
        return {"action": None, "finger": None, "reliability": None, "scatter": None}
    action = report_dir / "action_confusion.png"
    finger = report_dir / "finger_confusion.png"
    mc_eval = next(iter(sorted(report_dir.glob("mc_eval_*.png"))), None)
    mc_scatter = next(iter(sorted(report_dir.glob("mc_scatter_*.png"))), None)
    return {
        "action": action if action.exists() else None,
        "finger": finger if finger.exists() else None,
        "reliability": mc_eval if mc_eval and mc_eval.exists() else None,
        "scatter": mc_scatter if mc_scatter and mc_scatter.exists() else None,
    }


def _count_labels(arr: np.ndarray) -> Dict[str, int]:
    u, c = np.unique(arr.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u.tolist(), c.tolist())}


def _compute_run_metrics(metrics_path: Path) -> RunMetrics:
    run_dir = metrics_path.parent
    run_id = run_dir.name
    session_dir = run_dir.parents[2]

    metrics = _load_json(metrics_path)
    train = metrics.get("train", {}) or {}
    test = metrics.get("test", {}) or {}

    train_cfg_path = run_dir / "train_config.json"
    train_cfg = _load_json(train_cfg_path) if train_cfg_path.exists() else {}

    preds_path = run_dir / (metrics.get("artifacts", {}) or {}).get("preds", "test_predictions.npz")
    if not preds_path.is_absolute():
        preds_path = run_dir / preds_path
    preds = np.load(preds_path, allow_pickle=True)

    action_probs = np.asarray(preds["action_probs"])
    finger_probs = np.asarray(preds["finger_probs"])
    y_action = np.asarray(preds["y_action"]).astype(int)
    y_finger = np.asarray(preds["y_finger"]).astype(int)

    action_conf = action_probs.max(axis=1)
    action_preds = action_probs.argmax(axis=1)
    finger_conf = finger_probs.max(axis=1)
    finger_preds = finger_probs.argmax(axis=1)

    non_rest_mask = y_action != 0  # ACTION_REST from utils/label_schema.py
    test_action_acc_from_preds = float(np.mean(action_preds == y_action)) if y_action.size else None
    test_finger_acc_non_rest_from_preds = (
        float(np.mean(finger_preds[non_rest_mask] == y_finger[non_rest_mask]))
        if int(np.sum(non_rest_mask)) > 0
        else None
    )

    action_ece = expected_calibration_error(action_conf, action_preds, y_action, n_bins=10)
    finger_ece = (
        expected_calibration_error(
            finger_conf[non_rest_mask],
            finger_preds[non_rest_mask],
            y_finger[non_rest_mask],
            n_bins=10,
        )
        if int(np.sum(non_rest_mask)) > 0
        else None
    )

    # CIs: Wilson on exact counts + bootstrap on correctness vectors
    k_action = int(np.sum(action_preds == y_action))
    n_action = int(y_action.size)
    action_ci_w = wilson_ci(k_action, n_action)
    action_ci_b = bootstrap_ci_binary((action_preds == y_action).astype(int), seed=0)

    k_f = int(np.sum(finger_preds[non_rest_mask] == y_finger[non_rest_mask]))
    n_f = int(np.sum(non_rest_mask))
    finger_ci_w = wilson_ci(k_f, n_f)
    finger_ci_b = bootstrap_ci_binary(
        (finger_preds[non_rest_mask] == y_finger[non_rest_mask]).astype(int), seed=0
    )

    # Dataset NPZ (for window parameters + flags)
    # Prefer metrics.npz_path; fall back to session processed/eeg_windows.npz
    dataset_npz_path = metrics.get("npz_path")
    eeg_npz = None
    if dataset_npz_path:
        p = Path(dataset_npz_path)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            eeg_npz = np.load(p, allow_pickle=True)
    if eeg_npz is None:
        fallback = session_dir / "processed" / "eeg_windows.npz"
        if fallback.exists():
            eeg_npz = np.load(fallback, allow_pickle=True)

    n_windows_total = int(len(eeg_npz["y_action"])) if eeg_npz is not None and "y_action" in eeg_npz else None
    window_sec = float(np.asarray(eeg_npz["window_sec"]).item()) if eeg_npz is not None and "window_sec" in eeg_npz else None
    step_sec = float(np.asarray(eeg_npz["step_sec"]).item()) if eeg_npz is not None and "step_sec" in eeg_npz else None
    target_fs = float(np.asarray(eeg_npz["target_fs"]).item()) if eeg_npz is not None and "target_fs" in eeg_npz else None
    overlap_frac_mean = (
        float(np.mean(np.asarray(eeg_npz["overlap_frac"]).astype(float)))
        if eeg_npz is not None and "overlap_frac" in eeg_npz
        else None
    )
    artifact_count = (
        int(np.sum(np.asarray(eeg_npz["artifact_flag"]).astype(int)))
        if eeg_npz is not None and "artifact_flag" in eeg_npz
        else None
    )
    gap_count = (
        int(np.sum(np.asarray(eeg_npz["gap_flag"]).astype(int)))
        if eeg_npz is not None and "gap_flag" in eeg_npz
        else None
    )

    n_unique_trial_id = None
    n_unique_block_id = None
    try:
        if eeg_npz is not None and "trial_id" in eeg_npz:
            n_unique_trial_id = int(np.unique(np.asarray(eeg_npz["trial_id"]).astype(int)).size)
        if eeg_npz is not None and "block_id" in eeg_npz:
            n_unique_block_id = int(np.unique(np.asarray(eeg_npz["block_id"]).astype(int)).size)
    except Exception:
        pass

    # Per-class accuracies on the test set
    test_action_acc_by_class: Dict[str, float] = {}
    for cls in sorted(np.unique(y_action).astype(int).tolist()):
        m = y_action == cls
        if int(np.sum(m)) == 0:
            continue
        test_action_acc_by_class[str(int(cls))] = float(np.mean(action_preds[m] == y_action[m]))

    test_finger_acc_by_class_non_rest: Dict[str, float] = {}
    y_f_nr = y_finger[non_rest_mask]
    f_pred_nr = finger_preds[non_rest_mask]
    for cls in sorted(np.unique(y_f_nr).astype(int).tolist()):
        m = y_f_nr == cls
        if int(np.sum(m)) == 0:
            continue
        test_finger_acc_by_class_non_rest[str(int(cls))] = float(
            np.mean(f_pred_nr[m] == y_f_nr[m])
        )

    # Split/leakage config (from train_config.json)
    leakage_guard = {
        "split_mode": train_cfg.get("split_mode"),
        "purge_seconds": train_cfg.get("purge_seconds"),
        "window_idx_leak_threshold": train_cfg.get("window_idx_leak_threshold"),
        "strict_leakage": train_cfg.get("strict_leakage"),
    }

    # Report figures
    report_dir = session_dir / "processed" / "reports" / run_id
    figs = _find_report_figs(report_dir)
    subj = str(train_cfg.get("subject_id_filter") or metrics.get("subject_id") or "UNKNOWN")
    # Use the canonical session directory name (avoid embedding absolute paths into filenames).
    sess = str(session_dir.name)
    stem_base = _safe_slug(f"{subj}__{sess}__{run_id}")
    fig_action = _copy_figure(figs["action"], f"{stem_base}__action_confusion") if figs["action"] else None
    fig_finger = _copy_figure(figs["finger"], f"{stem_base}__finger_confusion") if figs["finger"] else None
    fig_rel = _copy_figure(figs["reliability"], f"{stem_base}__mc_eval") if figs["reliability"] else None
    fig_scat = _copy_figure(figs["scatter"], f"{stem_base}__mc_scatter") if figs["scatter"] else None

    return RunMetrics(
        subject_id=subj,
        session_id=session_dir.name,
        run_id=run_id,
        created_utc=metrics.get("created_utc"),
        train_action_acc=train.get("action_acc"),
        train_finger_acc=train.get("finger_acc"),
        train_avg_loss=train.get("avg_loss"),
        train_epochs=train.get("epochs"),
        train_batch_size=train.get("batch_size"),
        train_lr=train.get("lr"),
        train_seed=train.get("seed"),
        test_action_acc_metrics=test.get("action_acc"),
        test_finger_acc_non_rest_metrics=test.get("finger_acc_non_rest"),
        n_test_metrics=test.get("n_test"),
        n_test_non_rest_metrics=test.get("n_test_non_rest"),
        test_action_acc_from_preds=test_action_acc_from_preds,
        test_finger_acc_non_rest_from_preds=test_finger_acc_non_rest_from_preds,
        test_action_ece=action_ece,
        test_finger_ece_non_rest=finger_ece,
        action_ci95_wilson=action_ci_w,
        finger_non_rest_ci95_wilson=finger_ci_w,
        action_ci95_boot=action_ci_b,
        finger_non_rest_ci95_boot=finger_ci_b,
        n_windows_total=n_windows_total,
        window_sec=window_sec,
        step_sec=step_sec,
        target_fs=target_fs,
        overlap_frac_mean=overlap_frac_mean,
        artifact_count=artifact_count,
        gap_count=gap_count,
        test_action_counts=_count_labels(y_action),
        test_finger_counts=_count_labels(y_finger),
        test_action_acc_by_class=test_action_acc_by_class,
        test_finger_acc_by_class_non_rest=test_finger_acc_by_class_non_rest,
        n_unique_trial_id=n_unique_trial_id,
        n_unique_block_id=n_unique_block_id,
        split_mode=train_cfg.get("split_mode"),
        test_size=train_cfg.get("test_size"),
        purge_seconds=train_cfg.get("purge_seconds"),
        leakage_guard=leakage_guard,
        fig_action_confusion=fig_action,
        fig_finger_confusion=fig_finger,
        fig_reliability=fig_rel,
        fig_scatter=fig_scat,
    )


def _scan_subject_demographics() -> List[SubjectDemographics]:
    def _infer_sex_from_subject_id(subject_id: Optional[str]) -> Optional[str]:
        if not subject_id:
            return None
        subj = str(subject_id).upper()
        has_m = "M" in subj
        has_f = "F" in subj
        if has_m and not has_f:
            return "Male"
        if has_f and not has_m:
            return "Female"
        return None

    demos: List[SubjectDemographics] = []
    for subj_json in sorted(PROJECTS_ROOT.rglob("subject.json")):
        data = _load_json(subj_json)
        subject_id = str(data.get("subject_id") or subj_json.parent.name)
        sex = data.get("sex") or data.get("gender") or _infer_sex_from_subject_id(subject_id)
        demos.append(
            SubjectDemographics(
                subject_id=subject_id,
                age=data.get("age"),
                sex=sex,
                handedness=data.get("handedness"),
                notes=data.get("notes"),
            )
        )
    return demos


def _scan_session_meta(subject_id: str) -> Dict[str, Any]:
    """
    Aggregate session durations and event counts for all sessions matching subject_id.
    """
    out: Dict[str, Any] = {"subject_id": subject_id, "sessions": []}
    for run_meta in sorted(PROJECTS_ROOT.rglob("run_meta.json")):
        meta = _load_json(run_meta)
        if str(meta.get("subject_id")) != str(subject_id):
            continue
        sess_dir = run_meta.parent
        stream = meta.get("stream") or {}
        srate = stream.get("nominal_srate")
        samples_received = meta.get("samples_received")
        duration_s = None
        if samples_received is not None and srate:
            try:
                duration_s = float(samples_received) / float(srate)
            except Exception:
                duration_s = None
        events_path = sess_dir / "events" / "events.jsonl"
        if not events_path.exists():
            events_path = sess_dir / "events" / "events.json"
        if not events_path.exists():
            events_path = sess_dir / "events" / "events.csv"
        events_counts = _read_events_counts(events_path)
        out["sessions"].append(
            {
                "session_id": meta.get("session_id") or sess_dir.name,
                "created_utc": meta.get("created_utc"),
                "samples_received": samples_received,
                "samples_written": meta.get("samples_written"),
                "events_written": meta.get("events_written"),
                "queue_max_depth": meta.get("queue_max_depth"),
                "nominal_srate_hz": srate,
                "duration_s_from_samples": duration_s,
                "events_counts": events_counts,
            }
        )
    return out


def _format_pct(x: float, digits: int = 2) -> str:
    if x is None or not np.isfinite(x):
        return r"\textit{n/a}"
    return f"{x*100.0:.{digits}f}"


def _format_num(x: float, digits: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return r"\textit{n/a}"
    return f"{x:.{digits}f}"


def _write_macros(runs: List[RunMetrics], demos: List[SubjectDemographics], repo_sha: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Per-run macros (index-based to keep names stable)
    lines: List[str] = []
    lines.append("% AUTO-GENERATED by scripts/build_paper_artifacts.py. DO NOT EDIT BY HAND.\n")
    lines.append(r"\newcommand{\RepoCommitSha}{" + _latex_escape(repo_sha) + r"}" + "\n")
    lines.append(r"\newcommand{\NumSubjects}{" + str(len({r.subject_id for r in runs})) + r"}" + "\n")

    # Aggregate performance (across runs)
    action_accs = [r.test_action_acc_metrics for r in runs]
    finger_accs = [r.test_finger_acc_non_rest_metrics for r in runs]
    action_mean, action_sd = mean_std([float(x) for x in action_accs if x is not None])
    finger_mean, finger_sd = mean_std([float(x) for x in finger_accs if x is not None])
    action_min, action_max = min_max([float(x) for x in action_accs if x is not None])
    finger_min, finger_max = min_max([float(x) for x in finger_accs if x is not None])

    lines.append(r"\newcommand{\ActionAccMean}{" + (_format_pct(action_mean, 2) if np.isfinite(action_mean) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\ActionAccSD}{" + (_format_pct(action_sd, 2) if np.isfinite(action_sd) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\ActionAccMin}{" + (_format_pct(action_min, 2) if np.isfinite(action_min) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\ActionAccMax}{" + (_format_pct(action_max, 2) if np.isfinite(action_max) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\FingerAccMean}{" + (_format_pct(finger_mean, 2) if np.isfinite(finger_mean) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\FingerAccSD}{" + (_format_pct(finger_sd, 2) if np.isfinite(finger_sd) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\FingerAccMin}{" + (_format_pct(finger_min, 2) if np.isfinite(finger_min) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\FingerAccMax}{" + (_format_pct(finger_max, 2) if np.isfinite(finger_max) else r"\textit{n/a}") + r"}" + "\n")

    # ECE summary
    action_eces = [r.test_action_ece for r in runs]
    finger_eces = [r.test_finger_ece_non_rest for r in runs if r.test_finger_ece_non_rest is not None]
    ece_a_mean, ece_a_sd = mean_std([float(x) for x in action_eces if x is not None])
    ece_f_mean, ece_f_sd = mean_std([float(x) for x in finger_eces if x is not None])
    lines.append(r"\newcommand{\ActionECEMean}{" + (_format_num(ece_a_mean, 4) if np.isfinite(ece_a_mean) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\ActionECESD}{" + (_format_num(ece_a_sd, 4) if np.isfinite(ece_a_sd) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\FingerECEMean}{" + (_format_num(ece_f_mean, 4) if np.isfinite(ece_f_mean) else r"\textit{n/a}") + r"}" + "\n")
    lines.append(r"\newcommand{\FingerECESD}{" + (_format_num(ece_f_sd, 4) if np.isfinite(ece_f_sd) else r"\textit{n/a}") + r"}" + "\n")

    # Demographics summary
    ages = [d.age for d in demos if d.age is not None]
    if ages:
        ages_arr = np.asarray(ages, dtype=float)
        lines.append(r"\newcommand{\AgeMean}{" + f"{float(ages_arr.mean()):.1f}" + r"}" + "\n")
        lines.append(
            r"\newcommand{\AgeSD}{"
            + (f"{float(ages_arr.std(ddof=1)):.1f}" if len(ages) > 1 else r"\textit{n/a}")
            + r"}"
            + "\n"
        )
        lines.append(r"\newcommand{\AgeMin}{" + f"{float(ages_arr.min()):.0f}" + r"}" + "\n")
        lines.append(r"\newcommand{\AgeMax}{" + f"{float(ages_arr.max()):.0f}" + r"}" + "\n")
    else:
        lines.append(r"\newcommand{\AgeMean}{\textit{n/a}}" + "\n")
        lines.append(r"\newcommand{\AgeSD}{\textit{n/a}}" + "\n")
        lines.append(r"\newcommand{\AgeMin}{\textit{n/a}}" + "\n")
        lines.append(r"\newcommand{\AgeMax}{\textit{n/a}}" + "\n")

    # Windowing constants (from eeg_windows.npz summaries in runs)
    def _all_close(vals: List[Optional[float]], tol: float = 1e-9) -> Optional[float]:
        arr = [float(v) for v in vals if v is not None and np.isfinite(v)]
        if not arr:
            return None
        first = arr[0]
        if all(abs(v - first) <= tol for v in arr[1:]):
            return first
        return None

    window_sec = _all_close([r.window_sec for r in runs])
    step_sec = _all_close([r.step_sec for r in runs])
    target_fs = _all_close([r.target_fs for r in runs])
    lines.append(
        r"\newcommand{\WindowSec}{"
        + (f"{window_sec:.3f}" if window_sec is not None else r"\textit{n/a}")
        + r"}"
        + "\n"
    )
    lines.append(
        r"\newcommand{\HopSec}{"
        + (f"{step_sec:.3f}" if step_sec is not None else r"\textit{n/a}")
        + r"}"
        + "\n"
    )
    lines.append(
        r"\newcommand{\TargetFsHz}{"
        + (f"{target_fs:.1f}" if target_fs is not None else r"\textit{n/a}")
        + r"}"
        + "\n"
    )

    # Model parameter count (from models/cnn_lstm_finger_action_net.py defaults)
    param_count = None
    try:
        import torch  # noqa: F401

        from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet

        m = CNNLSTMFingerActionNet(n_channels=4, n_fingers=6, n_actions=3)
        param_count = int(sum(p.numel() for p in m.parameters()))
    except Exception:
        param_count = None

    lines.append(
        r"\newcommand{\ModelParamCount}{"
        + (str(param_count) if param_count is not None else r"\textit{n/a}")
        + r"}"
        + "\n"
    )

    # Metadata richness indicators (trial_id / block_id uniqueness in eeg_windows.npz)
    trial_uniques = [r.n_unique_trial_id for r in runs if r.n_unique_trial_id is not None]
    block_uniques = [r.n_unique_block_id for r in runs if r.n_unique_block_id is not None]
    if trial_uniques:
        lines.append(r"\newcommand{\TrialIdUniqueMin}{" + str(int(min(trial_uniques))) + r"}" + "\n")
        lines.append(r"\newcommand{\TrialIdUniqueMax}{" + str(int(max(trial_uniques))) + r"}" + "\n")
    else:
        lines.append(r"\newcommand{\TrialIdUniqueMin}{\textit{n/a}}" + "\n")
        lines.append(r"\newcommand{\TrialIdUniqueMax}{\textit{n/a}}" + "\n")
    if block_uniques:
        lines.append(r"\newcommand{\BlockIdUniqueMin}{" + str(int(min(block_uniques))) + r"}" + "\n")
        lines.append(r"\newcommand{\BlockIdUniqueMax}{" + str(int(max(block_uniques))) + r"}" + "\n")
    else:
        lines.append(r"\newcommand{\BlockIdUniqueMin}{\textit{n/a}}" + "\n")
        lines.append(r"\newcommand{\BlockIdUniqueMax}{\textit{n/a}}" + "\n")

    (OUT_DIR / "paper_macros.tex").write_text("".join(lines), encoding="utf-8")


def _write_tables(runs: List[RunMetrics], demos: List[SubjectDemographics], session_meta: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Demographics table
    demo_lines: List[str] = []
    demo_lines.append("% AUTO-GENERATED by scripts/build_paper_artifacts.py. DO NOT EDIT BY HAND.\n")
    demo_lines.append("\\begin{table}[t]\n\\centering\n")
    demo_lines.append(
        "\\caption{Subject demographics (from \\texttt{subject.json}; sex inferred from subject ID when missing).}\n"
    )
    demo_lines.append("\\label{tab:demo}\n")
    demo_lines.append("\\begin{tabular}{llll}\n\\toprule\n")
    demo_lines.append("Subject & Age & Sex & Handedness \\\\\n\\midrule\n")
    for d in sorted(demos, key=lambda x: x.subject_id):
        age = str(d.age) if d.age is not None else r"\textit{n/a}"
        sex = _latex_escape(d.sex) if d.sex else r"\textit{n/a}"
        hand = _latex_escape(d.handedness) if d.handedness else r"\textit{n/a}"
        demo_lines.append(f"{_latex_escape(d.subject_id)} & {age} & {sex} & {hand} \\\\\n")
    demo_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    # Performance table per run
    perf_lines: List[str] = []
    perf_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")
    perf_lines.append("\\caption{Per-subject test performance and calibration (from \\texttt{metrics.json} and \\texttt{test\\_predictions.npz}).}\n")
    perf_lines.append("\\label{tab:perf}\n")
    perf_lines.append("\\resizebox{\\textwidth}{!}{%\n")
    perf_lines.append("\\begin{tabular}{lp{0.30\\textwidth}rrrrrrrr}\n\\toprule\n")
    perf_lines.append(
        "Subject & Session & $n_{test}$ & $n_{non\\text{-}REST}$ & Action Acc (\\%) & 95\\% CI & Finger Acc$_{non\\text{-}REST}$ (\\%) & 95\\% CI & Action ECE & Finger ECE$_{non\\text{-}REST}$ \\\\\n\\midrule\n"
    )
    for r in sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id)):
        action_acc = r.test_action_acc_metrics
        finger_acc = r.test_finger_acc_non_rest_metrics
        a_lo, a_hi = r.action_ci95_wilson
        f_lo, f_hi = r.finger_non_rest_ci95_wilson
        a_ci = (
            f"[{a_lo*100.0:.2f}, {a_hi*100.0:.2f}]"
            if np.isfinite(a_lo) and np.isfinite(a_hi)
            else "\\textit{n/a}"
        )
        f_ci = (
            f"[{f_lo*100.0:.2f}, {f_hi*100.0:.2f}]"
            if np.isfinite(f_lo) and np.isfinite(f_hi)
            else "\\textit{n/a}"
        )
        session_id = _latex_escape(r.session_id)
        session_id = session_id.replace(r"\_", r"\_\allowbreak")
        perf_lines.append(
            f"{_latex_escape(r.subject_id)} & {session_id} & {int(r.n_test_metrics or 0)} & {int(r.n_test_non_rest_metrics or 0)} & "
            f"{_format_pct(action_acc, 2)} & {a_ci} & {_format_pct(finger_acc, 2)} & {f_ci} & "
            f"{_format_num(r.test_action_ece, 4)} & {_format_num(r.test_finger_ece_non_rest, 4)} \\\\\n"
        )
    perf_lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n\n")

    # Dataset / windowing table per run
    data_lines: List[str] = []
    data_lines.append("\\begin{table*}[t]\n\\centering\n")
    data_lines.append("\\caption{Dataset and window extraction summary (from \\texttt{eeg\\_windows.npz} and \\texttt{run\\_meta.json}).}\n")
    data_lines.append("\\label{tab:dataset}\n")
    data_lines.append("\\begin{tabular}{llrrrrrr}\n\\toprule\n")
    data_lines.append(
        "Subject & Session & $N$ windows & Window (s) & Hop (s) & Mean overlap (\\%) & Artifacts (count) & Gaps (count) \\\\\n\\midrule\n"
    )
    for r in sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id)):
        ov_pct = (float(r.overlap_frac_mean) * 100.0) if r.overlap_frac_mean is not None else None
        win_str = f"{float(r.window_sec):.3f}" if r.window_sec is not None else "\\textit{n/a}"
        hop_str = f"{float(r.step_sec):.3f}" if r.step_sec is not None else "\\textit{n/a}"
        ov_str = f"{ov_pct:.1f}" if ov_pct is not None and np.isfinite(ov_pct) else "\\textit{n/a}"
        art_str = str(r.artifact_count) if r.artifact_count is not None else "\\textit{n/a}"
        gapc_str = str(r.gap_count) if r.gap_count is not None else "\\textit{n/a}"
        data_lines.append(
            f"{_latex_escape(r.subject_id)} & {_latex_escape(r.session_id)} & {int(r.n_windows_total or 0)} & "
            f"{win_str} & {hop_str} & {ov_str} & {art_str} & {gapc_str} \\\\\n"
        )
    data_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

    # Session durations / ingestion integrity table (per session, from run_meta.json)
    dur_lines: List[str] = []
    dur_lines.append("\\begin{table}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")
    dur_lines.append(
        "\\caption{Session durations and write integrity (computed from \\texttt{run\\_meta.json}).}\n"
    )
    dur_lines.append("\\label{tab:sessions}\n")
    dur_lines.append("\\resizebox{\\columnwidth}{!}{%\n")
    dur_lines.append("\\begin{tabular}{p{0.30\\linewidth}rrrr}\n\\toprule\n")
    dur_lines.append("Session & Samples recv. & Samples written & Drop (\\%) & Duration (min) \\\\\n\\midrule\n")
    for s in session_meta.get("sessions", []):
        dur_s = s.get("duration_s_from_samples")
        dur_min = (float(dur_s) / 60.0) if dur_s is not None else None
        dur_str = f"{dur_min:.1f}" if dur_min is not None and np.isfinite(dur_min) else "\\textit{n/a}"
        recv = s.get("samples_received")
        written = s.get("samples_written")
        drop_pct = None
        try:
            if recv is not None and written is not None and float(recv) > 0:
                drop_pct = (1.0 - (float(written) / float(recv))) * 100.0
        except Exception:
            drop_pct = None
        drop_str = f"{drop_pct:.2f}" if drop_pct is not None and np.isfinite(drop_pct) else "\\textit{n/a}"
        session_id = _latex_escape(str(s.get("session_id")))
        session_id = session_id.replace(r"\_", r"\_\allowbreak")
        dur_lines.append(
            f"{session_id} & {int(recv or 0)} & {int(written or 0)} & {drop_str} & {dur_str} \\\\\n"
        )
    dur_lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n\n")

    # Event / movement counts (exact, from events/events.jsonl)
    # Use union of event 'type' values across sessions and rotate headers to fit IEEE table* width.
    type_set = set()
    for s in session_meta.get("sessions", []):
        ec = (s.get("events_counts") or {}) if isinstance(s, dict) else {}
        if ec.get("available") and isinstance(ec.get("counts_by_type"), dict):
            type_set |= set(ec["counts_by_type"].keys())

    # Stabilize order: REST first, then NONE_OPEN, then per-finger open/close.
    preferred_order = [
        "rest",
        "none_open",
        "thumb_open",
        "thumb_close",
        "index_open",
        "index_close",
        "middle_open",
        "middle_close",
        "ring_open",
        "ring_close",
        "pinky_open",
        "pinky_close",
    ]
    types = [t for t in preferred_order if t in type_set] + sorted([t for t in type_set if t not in preferred_order])

    ev_lines: List[str] = []
    ev_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    ev_lines.append(
        "\\caption{Event label counts per session (from \\texttt{events/events.jsonl}).}\n"
    )
    ev_lines.append("\\label{tab:events}\n")
    ev_lines.append("\\begin{tabular}{l" + "r" * len(types) + "}\n\\toprule\n")
    header_cells = ["Session"] + [f"\\rotatebox{{90}}{{{_latex_escape(t)}}}" for t in types]
    ev_lines.append(" & ".join(header_cells) + " \\\\\n\\midrule\n")
    for s in session_meta.get("sessions", []):
        sess_id = _latex_escape(str(s.get("session_id")))
        ec = s.get("events_counts") or {}
        cbt = ec.get("counts_by_type") if ec.get("available") else {}
        row = [sess_id]
        for t in types:
            row.append(str(int((cbt or {}).get(t, 0))))
        ev_lines.append(" & ".join(row) + " \\\\\n")
    ev_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")

    (OUT_DIR / "tables_demographics.tex").write_text("".join(demo_lines), encoding="utf-8")
    (OUT_DIR / "tables_performance.tex").write_text("".join(perf_lines), encoding="utf-8")
    (OUT_DIR / "tables_dataset.tex").write_text("".join(data_lines), encoding="utf-8")
    (OUT_DIR / "tables_sessions.tex").write_text("".join(dur_lines), encoding="utf-8")
    (OUT_DIR / "tables_events.tex").write_text("".join(ev_lines), encoding="utf-8")

    # Per-class accuracy tables (derived from test_predictions.npz)
    try:
        from utils.label_schema import ACTION_NAMES, FINGER_NAMES  # type: ignore
    except Exception:  # pragma: no cover
        ACTION_NAMES = {0: "0", 1: "1", 2: "2"}
        FINGER_NAMES = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}

    pc_lines: List[str] = []
    pc_lines.append("\\begin{table*}[t]\n\\centering\n")
    pc_lines.append("\\caption{Per-class accuracies on the test split (computed from saved per-window predictions).}\n")
    pc_lines.append("\\label{tab:perclass}\n")
    pc_lines.append("\\begin{tabular}{lllrr}\n\\toprule\n")
    pc_lines.append("Task & Subject & Class & Count & Accuracy (\\%) \\\\\n\\midrule\n")
    for r in sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id)):
        # Action classes
        for cls_str, acc in sorted(r.test_action_acc_by_class.items(), key=lambda kv: int(kv[0])):
            cls = int(cls_str)
            name = ACTION_NAMES.get(cls, cls_str)
            count = int(r.test_action_counts.get(cls_str, 0))
            pc_lines.append(
                f"Action & {_latex_escape(r.subject_id)} & {_latex_escape(name)} & {count} & {acc*100.0:.2f} \\\\\n"
            )
        # Finger classes (non-REST only); count for NONE excludes action REST windows.
        rest_count = int(r.test_action_counts.get("0", 0))
        for cls_str, acc in sorted(r.test_finger_acc_by_class_non_rest.items(), key=lambda kv: int(kv[0])):
            cls = int(cls_str)
            name = FINGER_NAMES.get(cls, cls_str)
            if cls == 0:
                count = max(0, int(r.test_finger_counts.get("0", 0)) - rest_count)
            else:
                count = int(r.test_finger_counts.get(cls_str, 0))
            pc_lines.append(
                f"Finger (non-REST) & {_latex_escape(r.subject_id)} & {_latex_escape(name)} & {count} & {acc*100.0:.2f} \\\\\n"
            )
    pc_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
    (OUT_DIR / "tables_perclass.tex").write_text("".join(pc_lines), encoding="utf-8")

    # Bootstrap CI table (optional rigor supplement)
    boot_lines: List[str] = []
    boot_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    boot_lines.append("\\caption{Bootstrap 95\\% confidence intervals for accuracy (percentile bootstrap over test windows).}\n")
    boot_lines.append("\\label{tab:bootci}\n")
    boot_lines.append("\\begin{tabular}{lrrrr}\n\\toprule\n")
    boot_lines.append("Subject & Action Acc (\\%) & 95\\% CI & Finger Acc$_{non\\text{-}REST}$ (\\%) & 95\\% CI \\\\\n\\midrule\n")
    for r in sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id)):
        a_lo, a_hi = r.action_ci95_boot
        f_lo, f_hi = r.finger_non_rest_ci95_boot
        a_ci = (
            f"[{a_lo*100.0:.2f}, {a_hi*100.0:.2f}]"
            if np.isfinite(a_lo) and np.isfinite(a_hi)
            else "\\textit{n/a}"
        )
        f_ci = (
            f"[{f_lo*100.0:.2f}, {f_hi*100.0:.2f}]"
            if np.isfinite(f_lo) and np.isfinite(f_hi)
            else "\\textit{n/a}"
        )
        boot_lines.append(
            f"{_latex_escape(r.subject_id)} & {_format_pct(r.test_action_acc_metrics, 2)} & {a_ci} & {_format_pct(r.test_finger_acc_non_rest_metrics, 2)} & {f_ci} \\\\\n"
        )
    boot_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
    (OUT_DIR / "tables_bootstrap_ci.tex").write_text("".join(boot_lines), encoding="utf-8")

    # Train vs test generalization gaps (from metrics.json train/test blocks)
    gap_lines: List[str] = []
    gap_lines.append("\\begin{table*}[t]\n\\centering\n")
    gap_lines.append("\\caption{Train--test generalization gaps (percentage points) from saved \\texttt{metrics.json}.}\n")
    gap_lines.append("\\label{tab:gap}\n")
    gap_lines.append("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    gap_lines.append(
        "Subject & Train Action (\\%) & Test Action (\\%) & Gap (pp) & Train Finger (\\%) & Test Finger$_{non\\text{-}REST}$ (\\%) & Gap (pp) \\\\\n\\midrule\n"
    )
    for r in sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id)):
        def _gap_pp(train: Optional[float], test: Optional[float]) -> str:
            if train is None or test is None or (not np.isfinite(train)) or (not np.isfinite(test)):
                return "\\textit{n/a}"
            return f"{(float(train) - float(test)) * 100.0:.2f}"

        gap_lines.append(
            f"{_latex_escape(r.subject_id)} & "
            f"{_format_pct(r.train_action_acc, 2)} & {_format_pct(r.test_action_acc_metrics, 2)} & {_gap_pp(r.train_action_acc, r.test_action_acc_metrics)} & "
            f"{_format_pct(r.train_finger_acc, 2)} & {_format_pct(r.test_finger_acc_non_rest_metrics, 2)} & {_gap_pp(r.train_finger_acc, r.test_finger_acc_non_rest_metrics)} \\\\\n"
        )
    gap_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
    (OUT_DIR / "tables_generalization_gap.tex").write_text("".join(gap_lines), encoding="utf-8")


def _write_figures(runs: List[RunMetrics]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    lines: List[str] = []
    lines.append("% AUTO-GENERATED by scripts/build_paper_artifacts.py. DO NOT EDIT BY HAND.\n")

    for r in sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id)):
        lines.append("\\begin{figure*}[t]\n\\centering\n")
        # 2x2 grid using minipages (IEEE-friendly, no extra packages required)
        def inc(path: Optional[str], caption: str) -> str:
            if not path:
                return (
                    "\\begin{minipage}[b]{0.49\\linewidth}\\centering\n"
                    "\\fbox{\\parbox[c][0.28\\textheight][c]{0.95\\linewidth}{\\centering\\textit{Figure not available in current run artifacts.}}}\n"
                    + "\\\\"
                    + "\\footnotesize "
                    + caption
                    + "\n\\end{minipage}\n"
                )
            return (
                "\\begin{minipage}[b]{0.49\\linewidth}\\centering\n"
                + f"\\includegraphics[width=\\linewidth]{{{_latex_escape(path)}}}\n"
                + "\\\\"
                + "\\footnotesize "
                + caption
                + "\n\\end{minipage}\n"
            )

        subj = _latex_escape(r.subject_id)
        sess = _latex_escape(r.session_id)
        lines.append(inc(r.fig_action_confusion, f"Action confusion matrix ({subj})."))
        lines.append(inc(r.fig_finger_confusion, f"Finger confusion matrix ({subj})."))
        lines.append("\\\\[0.5em]\n")
        lines.append(inc(r.fig_reliability, f"Reliability / calibration summary ({subj})."))
        lines.append(inc(r.fig_scatter, f"MC dropout confidence scatter ({subj})."))
        run_id = _latex_escape(r.run_id)
        lines.append(
            f"\\caption{{Per-run evaluation figures for {subj} (session {sess}, run {run_id}).}}\n"
        )
        lines.append("\\end{figure*}\n\n")

    (OUT_DIR / "figures.tex").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    if not PROJECTS_ROOT.exists():
        raise SystemExit(f"Projects/ not found at {PROJECTS_ROOT}")

    # Deterministic rebuild: clear previous generated artifacts.
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    if FIG_DIR.exists():
        shutil.rmtree(FIG_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Discover all runs
    run_metrics: List[RunMetrics] = []
    for metrics_path in sorted(PROJECTS_ROOT.rglob("processed/models/*/metrics.json")):
        run_metrics.append(_compute_run_metrics(metrics_path))

    # Demographics (subject.json)
    demos = _scan_subject_demographics()

    # Session meta (run_meta + events) for each subject appearing in runs
    session_meta_by_subject: Dict[str, Any] = {}
    for subj in sorted({r.subject_id for r in run_metrics}):
        session_meta_by_subject[subj] = _scan_session_meta(subj)

    # Repo SHA for reproducibility section
    sha = "UNKNOWN"
    git_head = REPO_ROOT / ".git" / "HEAD"
    if git_head.exists():
        try:
            import subprocess

            sha = (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
                .decode("utf-8")
                .strip()
            )
        except Exception:
            pass

    # Write full JSON bundle
    stats = {
        # Avoid embedding absolute local paths in version-controlled artifacts.
        "repo_root": ".",
        "repo_commit_sha": sha,
        "n_runs": len(run_metrics),
        "runs": [asdict(r) for r in run_metrics],
        "subjects": [asdict(d) for d in demos],
        "session_meta_by_subject": session_meta_by_subject,
    }
    (OUT_DIR / "paper_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # Write LaTeX snippets
    _write_macros(run_metrics, demos, sha)
    # For tables: include session duration tables per subject by concatenating (small N here)
    # If multiple subjects exist, durations table will include all sessions across subjects.
    merged_session_meta = {"sessions": []}
    for subj in sorted(session_meta_by_subject.keys()):
        merged_session_meta["sessions"].extend(session_meta_by_subject[subj].get("sessions", []))
    _write_tables(run_metrics, demos, merged_session_meta)
    _write_figures(run_metrics)

    print(f"Wrote: {OUT_DIR / 'paper_stats.json'}")
    print(f"Wrote: {OUT_DIR / 'paper_macros.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_demographics.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_dataset.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_sessions.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_events.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_performance.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_perclass.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_bootstrap_ci.tex'}")
    print(f"Wrote: {OUT_DIR / 'tables_generalization_gap.tex'}")
    print(f"Wrote: {OUT_DIR / 'figures.tex'}")
    print(f"Copied figures into: {FIG_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
