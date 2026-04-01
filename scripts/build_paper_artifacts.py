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
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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


def _latex_breakable_id(text: str) -> str:
    escaped = _latex_escape(text)
    escaped = escaped.replace(r"\_", r"\_\allowbreak{}")
    escaped = escaped.replace("-", r"-\allowbreak{}")
    return escaped


def _display_subject_id(subject_id: str) -> str:
    subj = str(subject_id).strip()
    parts = subj.split()
    if len(parts) > 1 and all(part.isdigit() for part in parts[1:]):
        return parts[0]
    return subj


def _display_session_id(session_id: str) -> str:
    sess = str(session_id).strip()
    if not sess:
        return sess
    head, sep, tail = sess.partition("_")
    display_head = _display_subject_id(head)
    if not sep:
        return display_head
    return f"{display_head}{sep}{tail}"


def _paper_session_label(session_id: str) -> str:
    sess = _display_session_id(session_id)
    if not sess:
        return sess
    if sess.startswith("combined_"):
        return "Derived dataset (filtered)" if "pruned" in sess else "Derived dataset"
    return sess


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def _load_json_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _session_dir_from_rel(session_dir_rel: str) -> Path:
    return (REPO_ROOT / session_dir_rel).resolve()


def _session_provenance(session_dir: Path) -> Dict[str, Any]:
    manifest = _load_json_if_exists(session_dir / "manifest.json")
    filter_manifest = _load_json_if_exists(session_dir / "processed" / "filter_manifest.json")

    kind = "recorded"
    combined_from_sessions: List[str] = []
    core_sessions: List[str] = []
    aux_rest_sessions: List[str] = []
    filter_source_session = None
    filter_removed_n = None
    filter_source_n = None
    filter_kept_n = None
    filter_event_ids: List[int] = []
    filter_session_id = None
    filter_reason = None

    if isinstance(manifest.get("combined_from_sessions"), list):
        kind = "combined"
        combined_from_sessions = [Path(str(p)).name for p in manifest.get("combined_from_sessions", [])]
        deploy = manifest.get("deployment_training") or {}
        core_sessions = [str(s) for s in deploy.get("core_sessions", [])]
        aux_rest_sessions = [str(s) for s in deploy.get("aux_rest_sessions", [])]

    if filter_manifest:
        kind = "filtered"
        filter_source_session = Path(str(filter_manifest.get("source_session_dir") or "")).name or None
        counts = filter_manifest.get("counts") or {}
        filter_removed_n = counts.get("removed_n")
        filter_source_n = counts.get("source_n")
        filter_kept_n = counts.get("kept_n")
        filt = filter_manifest.get("filter") or {}
        filter_event_ids = [int(v) for v in filt.get("event_ids", []) if str(v).isdigit()]
        filter_session_id = filt.get("session_id")
        filter_reason = filt.get("reason")
        source_manifest = _load_json_if_exists(Path(str(filter_manifest.get("source_session_dir"))) / "manifest.json")
        if isinstance(source_manifest.get("combined_from_sessions"), list):
            combined_from_sessions = [
                Path(str(p)).name for p in source_manifest.get("combined_from_sessions", [])
            ]
            deploy = source_manifest.get("deployment_training") or {}
            core_sessions = [str(s) for s in deploy.get("core_sessions", [])]
            aux_rest_sessions = [str(s) for s in deploy.get("aux_rest_sessions", [])]

    return {
        "kind": kind,
        "session_id": session_dir.name,
        "session_dir_rel": _repo_rel(session_dir),
        "combined_from_sessions": combined_from_sessions,
        "core_sessions": core_sessions,
        "aux_rest_sessions": aux_rest_sessions,
        "filter_source_session": filter_source_session,
        "filter_removed_n": filter_removed_n,
        "filter_source_n": filter_source_n,
        "filter_kept_n": filter_kept_n,
        "filter_event_ids": filter_event_ids,
        "filter_session_id": filter_session_id,
        "filter_reason": filter_reason,
    }


def _raw_support_sessions_for_run(run: "RunMetrics") -> List[Dict[str, str]]:
    prov = _session_provenance(_session_dir_from_rel(run.session_dir_rel))
    if prov["kind"] == "recorded":
        return [{"session_id": run.session_id, "role": "recorded"}]

    rows: List[Dict[str, str]] = []
    combined_from = list(prov.get("combined_from_sessions") or [])
    core = set(prov.get("core_sessions") or [])
    aux = set(prov.get("aux_rest_sessions") or [])
    if not combined_from and prov.get("filter_source_session"):
        combined_from = [str(prov["filter_source_session"])]

    for session_id in combined_from:
        role = "supporting"
        if session_id in core:
            role = "core"
        elif session_id in aux:
            role = "aux_rest"
        rows.append({"session_id": session_id, "role": role})
    return rows


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
        idx = ((conf >= bins[i]) if i == 0 else (conf > bins[i])) & (conf <= bins[i + 1])
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


def _load_featured_bundle() -> Dict[str, Any]:
    manifest_path = next(PROJECTS_ROOT.glob("**/winning_model/winning_model_manifest.json"), None)
    if manifest_path is None or not manifest_path.exists():
        return {}

    winning_root = manifest_path.parent
    manifest = _load_json(manifest_path)

    model_metrics_path = winning_root / "model_run" / "metrics.json"
    eval_manifest_path = winning_root / "session_report" / "eval_manifest.json"

    source_session_dir = manifest.get("source_session_dir")
    replay_manifest_path = None
    if source_session_dir:
        source_session_name = Path(str(source_session_dir)).name
        candidate = winning_root / "pseudo_live" / source_session_name / "replay_manifest.json"
        if candidate.exists():
            replay_manifest_path = candidate
    if replay_manifest_path is None:
        replay_manifest_path = next(
            iter(sorted((winning_root / "pseudo_live").glob("*/replay_manifest.json"))),
            None,
        )

    return {
        "manifest": manifest,
        "model_metrics": _load_json(model_metrics_path) if model_metrics_path.exists() else {},
        "eval_manifest": _load_json(eval_manifest_path) if eval_manifest_path.exists() else {},
        "replay_manifest": _load_json(replay_manifest_path) if replay_manifest_path and replay_manifest_path.exists() else {},
        "replay_manifest_path": str(replay_manifest_path) if replay_manifest_path and replay_manifest_path.exists() else None,
    }


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
    session_dir_rel: str
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
    test_finger_acc_non_rest_raw_head: Optional[float]
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


def _metric_or_neg_inf(value: Optional[float]) -> float:
    if value is None or not np.isfinite(value):
        return float("-inf")
    return float(value)


def _run_rank_key(run: RunMetrics) -> Tuple[float, float, int, int, str, str]:
    """
    Deterministic ranking for manuscript run selection.
    Primary sort is held-out action accuracy because the paper's deployment path
    first depends on correctly separating REST vs action. Finger accuracy is the
    first tiebreak, followed by sample counts and stable lexical IDs.
    """
    return (
        _metric_or_neg_inf(run.test_action_acc_metrics),
        _metric_or_neg_inf(run.test_finger_acc_non_rest_metrics),
        int(run.n_test_non_rest_metrics or 0),
        int(run.n_test_metrics or 0),
        str(run.created_utc or ""),
        str(run.run_id),
    )


def _select_best_runs_per_subject(runs: List[RunMetrics]) -> List[RunMetrics]:
    best_by_subject: Dict[str, RunMetrics] = {}
    for run in runs:
        current = best_by_subject.get(run.subject_id)
        if current is None or _run_rank_key(run) > _run_rank_key(current):
            best_by_subject[run.subject_id] = run
    return sorted(best_by_subject.values(), key=lambda x: (x.subject_id, x.session_id, x.run_id))


def _copy_figure(src: Path, dest_stem: str) -> str:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    dest = FIG_DIR / f"{dest_stem}{ext}"
    shutil.copy2(src, dest)
    return str(dest.relative_to(REPO_ROOT))


def _find_report_figs(report_dir: Path, run_id: str) -> Dict[str, Optional[Path]]:
    repo_report_dir = REPO_ROOT / "reports" / "runs" / str(run_id)
    winning_report_dir = next(
        iter(sorted(PROJECTS_ROOT.glob(f"**/winning_model/repo_report"))),
        None,
    )

    def _first_existing(*paths: Optional[Path]) -> Optional[Path]:
        for path in paths:
            if path is not None and path.exists():
                return path
        return None

    def _first_glob(*patterns: Tuple[Optional[Path], str]) -> Optional[Path]:
        for directory, pattern in patterns:
            if directory is None or not directory.exists():
                continue
            match = next(iter(sorted(directory.glob(pattern))), None)
            if match is not None and match.exists():
                return match
        return None

    action = _first_existing(
        report_dir / "action_confusion.png" if report_dir.exists() else None,
        repo_report_dir / "action_confusion.png",
        winning_report_dir / "action_confusion.png" if winning_report_dir else None,
    )
    finger = _first_existing(
        report_dir / "finger_confusion.png" if report_dir.exists() else None,
        repo_report_dir / "finger_confusion.png",
        winning_report_dir / "finger_confusion.png" if winning_report_dir else None,
    )
    mc_eval = _first_glob(
        (report_dir if report_dir.exists() else None, "mc_eval_*.png"),
        (repo_report_dir, "mc_eval_*.png"),
        (winning_report_dir, "mc_eval_*.png"),
    )
    mc_scatter = _first_glob(
        (report_dir if report_dir.exists() else None, "mc_scatter_*.png"),
        (repo_report_dir, "mc_scatter_*.png"),
        (winning_report_dir, "mc_scatter_*.png"),
    )
    return {
        "action": action,
        "finger": finger,
        "reliability": mc_eval,
        "scatter": mc_scatter,
    }


def _resolve_test_indices(preds: Any) -> Optional[np.ndarray]:
    for key in ("test_indices_local", "test_indices", "test_indices_global"):
        if key in preds:
            return np.asarray(preds[key]).astype(int).reshape(-1)
    return None


def _write_standardized_mc_scatter(
    *,
    session_dir: Path,
    run_id: str,
    train_cfg: Dict[str, Any],
    dest_stem: str,
) -> Optional[str]:
    try:
        import torch

        from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
        from utils.model_outputs import infer_output_dims_from_state_dict
        from utils.runtime_utils import apply_channel_normalizer, load_normalizer
    except Exception:
        return None

    run_dir = session_dir / "processed" / "models" / run_id
    model_path = run_dir / "finger_action_model.pt"
    scaler_path = run_dir / "scaler.npz"
    preds_path = run_dir / "test_predictions.npz"
    windows_path = session_dir / "processed" / "eeg_windows.npz"
    if not all(path.exists() for path in (model_path, scaler_path, preds_path, windows_path)):
        return None

    try:
        state_dict = torch.load(model_path, map_location="cpu")
        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
            state_dict = state_dict["state_dict"]
        n_fingers, n_actions, has_applicability = infer_output_dims_from_state_dict(state_dict)
        n_channels = 4
        input_shape = train_cfg.get("input_shape")
        if isinstance(input_shape, list) and len(input_shape) >= 2:
            try:
                n_channels = int(input_shape[-1])
            except Exception:
                n_channels = 4
        model = CNNLSTMFingerActionNet(
            n_channels=n_channels,
            n_fingers=n_fingers,
            n_actions=n_actions,
            finger_applicability_head=has_applicability,
        )
        model.load_state_dict(state_dict)
        model.eval()

        preds = np.load(preds_path, allow_pickle=True)
        test_idx = _resolve_test_indices(preds)
        if test_idx is None or test_idx.size == 0:
            return None

        windows = np.load(windows_path, allow_pickle=True)
        X = np.asarray(windows["X"][test_idx], dtype=np.float32)
        normalizer = load_normalizer(scaler_path)
        X = apply_channel_normalizer(X, normalizer)

        confidences: List[np.ndarray] = []
        uncertainties: List[np.ndarray] = []
        torch.manual_seed(0)
        batch_size = 256
        mc_passes = 20
        with torch.no_grad():
            for start in range(0, int(X.shape[0]), batch_size):
                stop = min(int(X.shape[0]), start + batch_size)
                xb = torch.from_numpy(X[start:stop]).float()
                mc = model.mc_forward(xb, passes=mc_passes)
                action_mean = mc["action_mean"]
                action_std = mc["action_std"]
                confidences.append(action_mean.max(dim=-1).values.detach().cpu().numpy())
                uncertainties.append(action_std.mean(dim=-1).detach().cpu().numpy())

        conf = np.concatenate(confidences) if confidences else np.empty((0,), dtype=float)
        unc = np.concatenate(uncertainties) if uncertainties else np.empty((0,), dtype=float)
        if conf.size == 0 or unc.size == 0:
            return None

        fig, ax = plt.subplots(figsize=(6.0, 4.9), dpi=220)
        hb = ax.hexbin(
            conf,
            unc,
            gridsize=42,
            mincnt=1,
            bins="log",
            cmap="viridis",
            linewidths=0.0,
        )
        cbar = fig.colorbar(hb, ax=ax, pad=0.02)
        cbar.set_label("Counts", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.set_xlim(0.35, 1.0)
        ax.set_ylim(0.0, 0.32)
        ax.set_xlabel("MC-dropout action confidence")
        ax.set_ylabel("MC-dropout action uncertainty")
        ax.grid(linestyle=":", linewidth=0.5, alpha=0.45)
        ax.set_axisbelow(True)
        ax.text(
            0.98,
            0.98,
            f"n={conf.size}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.2,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#bbbbbb", linewidth=0.6, alpha=0.95),
        )
        anchor_x = float(np.quantile(conf, 0.82))
        anchor_y = float(np.quantile(unc, 0.18))
        ax.annotate(
            "Dense region:\nhigh confidence,\nlow uncertainty",
            xy=(anchor_x, anchor_y),
            xytext=(0.58, 0.83),
            textcoords="axes fraction",
            fontsize=7.2,
            ha="left",
            va="top",
            arrowprops=dict(arrowstyle="->", linewidth=0.8, color="#333333"),
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#bbbbbb", linewidth=0.6, alpha=0.95),
        )
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        out_path = FIG_DIR / f"{dest_stem}.png"
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return str(out_path.relative_to(REPO_ROOT))
    except Exception:
        return None


def _summarize_replay_predictions(replay_manifest: Dict[str, Any], replay_manifest_path: Optional[Path]) -> Dict[str, Optional[int]]:
    pred_path = None
    pred_entry = replay_manifest.get("predictions_jsonl")
    if pred_entry:
        pred_path = Path(str(pred_entry))
    if pred_path is None or not pred_path.exists():
        if replay_manifest_path is not None:
            candidate = replay_manifest_path.parent / "predictions.jsonl"
            if candidate.exists():
                pred_path = candidate
    if pred_path is None or not pred_path.exists():
        return {
            "rest_windows": None,
            "non_rest_windows": None,
            "positive_send_windows": None,
            "correct_send_windows": None,
            "false_actuation_rest_count": None,
        }

    rest_windows = 0
    non_rest_windows = 0
    positive_send_windows = 0
    correct_send_windows = 0
    false_actuation_rest_count = 0
    with pred_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            true_action = int(row.get("true_action_id", -1) or 0)
            true_finger = int(row.get("true_finger_id", 0) or 0)
            actuation_sent = bool(row.get("actuation_sent", False))
            act_action = int(row.get("actuation_target_action_id", 0) or 0)
            act_finger = int(row.get("actuation_target_finger_id", 0) or 0)
            if true_action == 0:
                rest_windows += 1
                false_actuation_rest_count += int(actuation_sent)
            else:
                non_rest_windows += 1
            positive_send_windows += int(actuation_sent and act_action != 0 and act_finger != 0)
            correct_send_windows += int(
                actuation_sent
                and true_action != 0
                and act_action == true_action
                and act_finger == true_finger
            )

    return {
        "rest_windows": int(rest_windows),
        "non_rest_windows": int(non_rest_windows),
        "positive_send_windows": int(positive_send_windows),
        "correct_send_windows": int(correct_send_windows),
        "false_actuation_rest_count": int(false_actuation_rest_count),
    }


def _count_labels(arr: np.ndarray) -> Dict[str, int]:
    u, c = np.unique(arr.astype(int), return_counts=True)
    return {str(int(k)): int(v) for k, v in zip(u.tolist(), c.tolist())}


def _compute_run_metrics(metrics_path: Path) -> RunMetrics:
    from utils.label_schema import decode_finger_predictions_for_actions, finger_confidences_for_ids

    run_dir = metrics_path.parent
    run_id = run_dir.name
    session_dir = run_dir.parents[2]
    session_dir_rel = _repo_rel(session_dir)

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

    non_rest_mask = y_action != 0  # ACTION_REST from utils/label_schema.py
    finger_preds = np.asarray(
        decode_finger_predictions_for_actions(action_preds, finger_probs), dtype=np.int64
    )
    finger_conf = np.asarray(finger_confidences_for_ids(finger_probs, finger_preds), dtype=float)
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
    figs = _find_report_figs(report_dir, run_id=run_id)
    subj = str(train_cfg.get("subject_id_filter") or metrics.get("subject_id") or "UNKNOWN")
    # Use the canonical session directory name (avoid embedding absolute paths into filenames).
    sess = str(session_dir.name)
    stem_base = _safe_slug(f"{subj}__{sess}__{run_id}")
    fig_action = _copy_figure(figs["action"], f"{stem_base}__action_confusion") if figs["action"] else None
    fig_finger = _copy_figure(figs["finger"], f"{stem_base}__finger_confusion") if figs["finger"] else None
    fig_rel = _copy_figure(figs["reliability"], f"{stem_base}__mc_eval") if figs["reliability"] else None
    fig_scat = _copy_figure(figs["scatter"], f"{stem_base}__mc_scatter") if figs["scatter"] else None
    if fig_scat is None:
        fig_scat = _write_standardized_mc_scatter(
            session_dir=session_dir,
            run_id=run_id,
            train_cfg=train_cfg,
            dest_stem=f"{stem_base}__mc_scatter",
        )

    return RunMetrics(
        subject_id=subj,
        session_id=session_dir.name,
        run_id=run_id,
        created_utc=metrics.get("created_utc"),
        session_dir_rel=session_dir_rel,
        train_action_acc=train.get("action_acc"),
        train_finger_acc=train.get("finger_acc"),
        train_avg_loss=train.get("avg_loss"),
        train_epochs=train.get("epochs"),
        train_batch_size=train.get("batch_size"),
        train_lr=train.get("lr"),
        train_seed=train.get("seed"),
        test_action_acc_metrics=test.get("action_acc"),
        # Use deployment-consistent finger decoding for manuscript metrics so
        # the point estimate, per-class table, and CI all refer to the same quantity.
        test_finger_acc_non_rest_metrics=test_finger_acc_non_rest_from_preds,
        test_finger_acc_non_rest_raw_head=test.get("finger_acc_non_rest"),
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


def _scan_session_meta(subject_id: str, session_ids: Optional[set[str]] = None) -> Dict[str, Any]:
    """
    Aggregate session durations and event counts for all sessions matching subject_id.
    """
    out: Dict[str, Any] = {"subject_id": subject_id, "sessions": []}
    for run_meta in sorted(PROJECTS_ROOT.rglob("run_meta.json")):
        meta = _load_json(run_meta)
        if str(meta.get("subject_id")) != str(subject_id):
            continue
        sess_dir = run_meta.parent
        session_id = str(meta.get("session_id") or sess_dir.name)
        if session_ids is not None and session_id not in session_ids:
            continue
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
                "subject_id": subject_id,
                "session_id": session_id,
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


def _write_finger_accuracy_bar_chart(runs: List[RunMetrics]) -> str:
    try:
        from utils.label_schema import FINGER_NAMES  # type: ignore
    except Exception:  # pragma: no cover
        FINGER_NAMES = {1: "THUMB", 2: "INDEX", 3: "MIDDLE", 4: "RING", 5: "PINKY"}

    finger_ids = [1, 2, 3, 4, 5]
    labels = [FINGER_NAMES.get(fid, str(fid)).title() for fid in finger_ids]
    x = np.arange(len(finger_ids))
    width = 0.36

    fig, ax = plt.subplots(figsize=(7.6, 4.3), dpi=220)
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#7f7f7f"]

    for idx, run in enumerate(runs):
        values = [100.0 * float(run.test_finger_acc_by_class_non_rest.get(str(fid), float("nan"))) for fid in finger_ids]
        offset = (idx - (len(runs) - 1) / 2.0) * width
        bars = ax.bar(
            x + offset,
            values,
            width=width,
            label=_display_subject_id(run.subject_id),
            color=colors[idx % len(colors)],
            edgecolor="black",
            linewidth=0.6,
        )
        for bar, fid in zip(bars, finger_ids):
            count = int(run.test_finger_counts.get(str(fid), 0))
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 1.6,
                    f"n={count}",
                    ha="center",
                    va="bottom",
                    fontsize=7.1,
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.92),
                )

    ax.axhline(20.0, color="#444444", linestyle="--", linewidth=1.0, label="5-way chance")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 110.0)
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="upper center")
    fig.tight_layout()

    out_path = FIG_DIR / "finger_accuracy_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def _load_event_rows(session_dir: Path) -> List[Dict[str, Any]]:
    events_path = session_dir / "events" / "events.jsonl"
    rows: List[Dict[str, Any]] = []
    if not events_path.exists():
        return rows
    for idx, line in enumerate(events_path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["_event_index"] = idx
            rows.append(payload)
    return rows


def _load_raw_segment_by_local_time(session_dir: Path, local_start: float, local_end: float) -> Tuple[np.ndarray, np.ndarray]:
    times: List[np.ndarray] = []
    samples: List[np.ndarray] = []
    raw_dir = session_dir / "raw"
    for shard in sorted(raw_dir.glob("eeg_raw_shard_*.npy")):
        arr = np.load(shard, allow_pickle=False)
        if "local_ts" not in arr.dtype.names or "sample" not in arr.dtype.names:
            continue
        mask = (arr["local_ts"] >= local_start) & (arr["local_ts"] <= local_end)
        if not np.any(mask):
            continue
        times.append(np.asarray(arr["local_ts"][mask], dtype=float))
        samples.append(np.asarray(arr["sample"][mask], dtype=float))
    if not times or not samples:
        return np.empty((0,), dtype=float), np.empty((0, 4), dtype=float)
    t = np.concatenate(times, axis=0)
    x = np.concatenate(samples, axis=0)
    order = np.argsort(t)
    return t[order], x[order]


def _write_featured_provenance_figure(runs: List[RunMetrics]) -> Optional[str]:
    if not runs:
        return None
    featured = max(runs, key=_run_rank_key)
    prov = _session_provenance(_session_dir_from_rel(featured.session_dir_rel))
    if prov.get("kind") != "filtered":
        return None

    def _short_session(session_id: str) -> str:
        sid = _display_session_id(session_id)
        parts = sid.split("_")
        if len(parts) >= 3:
            return f"{parts[0]}\n{parts[1]}_{parts[2]}"
        return sid.replace("_", "\n", 1)

    def _short_dataset(session_id: str) -> str:
        sid = _display_session_id(session_id)
        if sid.startswith("combined_"):
            parts = sid.split("_")
            if len(parts) >= 3:
                return f"{parts[1]}_{parts[2]}"
        return sid

    def _box(ax: plt.Axes, xy: Tuple[float, float], wh: Tuple[float, float], title: str, body: str, face: str) -> None:
        x, y = xy
        w, h = wh
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.1,
            edgecolor="#2f2f2f",
            facecolor=face,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2.0, y + h * 0.72, title, ha="center", va="center", fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2.0, y + h * 0.36, body, ha="center", va="center", fontsize=6.8)

    fig, ax = plt.subplots(figsize=(9.0, 3.2), dpi=200)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")

    left_boxes = [
        ("Movement Session 1", _short_session(str((prov.get("core_sessions") or ["unknown"])[0])), "#e8f1fb"),
        ("Movement Session 2", _short_session(str((prov.get("core_sessions") or ["unknown", "unknown"])[1])), "#e8f1fb"),
        ("Aux REST Session", _short_session(str((prov.get("aux_rest_sessions") or ["unknown"])[0])), "#eef6e8"),
    ]
    left_positions = [(0.03, 0.68), (0.03, 0.40), (0.03, 0.12)]
    for (title, body, face), pos in zip(left_boxes, left_positions):
        _box(ax, pos, (0.20, 0.18), title, body, face)

    combined_title = "Combined Dataset"
    combined_body = f"{_short_dataset(str(prov.get('filter_source_session') or 'combined'))}\nmerged movement + REST"
    filtered_body = (
        f"{_short_dataset(featured.session_id)} REST-pruned\n"
        f"remove REST {', '.join(str(v) for v in prov.get('filter_event_ids') or [])}\n"
        f"{int(prov.get('filter_kept_n') or 0):,} / {int(prov.get('filter_source_n') or 0):,} windows kept"
    )
    run_body = (
        f"{featured.run_id}\n"
        f"{100.0 * float(featured.test_action_acc_metrics or float('nan')):.2f}% action\n"
        f"{100.0 * float(featured.test_finger_acc_non_rest_metrics or float('nan')):.2f}% finger"
    )

    _box(ax, (0.34, 0.28), (0.18, 0.42), combined_title, combined_body, "#d9ecff")
    _box(ax, (0.58, 0.28), (0.18, 0.42), "Filtered Dataset", filtered_body, "#fff1cf")
    _box(ax, (0.82, 0.28), (0.15, 0.42), "Featured Run", run_body, "#ffe1de")

    arrow_kw = dict(arrowstyle="-|>", mutation_scale=12, linewidth=1.2, color="#444444")
    for y in [0.77, 0.49, 0.21]:
        ax.add_patch(FancyArrowPatch((0.23, y), (0.34, 0.49), connectionstyle="arc3,rad=0.0", **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.52, 0.49), (0.58, 0.49), connectionstyle="arc3,rad=0.0", **arrow_kw))
    ax.add_patch(FancyArrowPatch((0.76, 0.49), (0.82, 0.49), connectionstyle="arc3,rad=0.0", **arrow_kw))
    ax.text(
        0.55,
        0.64,
        "targeted REST pruning",
        fontsize=6.9,
        ha="center",
        va="center",
        color="#5a4b00",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#c7aa46", linewidth=0.7),
    )

    out_path = FIG_DIR / "featured_dataset_provenance.png"
    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.06)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def _write_raw_windowing_figure(runs: List[RunMetrics]) -> Optional[str]:
    source_run = next((r for r in sorted(runs, key=_run_rank_key, reverse=True) if _session_provenance(_session_dir_from_rel(r.session_dir_rel)).get("kind") == "recorded"), None)
    if source_run is None:
        return None

    session_dir = _session_dir_from_rel(source_run.session_dir_rel)
    events = _load_event_rows(session_dir)
    if not events:
        return None

    active_event = next((ev for ev in events if int(ev.get("action_id") or 0) != 0 and float(ev.get("duration_s") or 0.0) >= 0.6), None)
    if active_event is None:
        active_event = next((ev for ev in events if int(ev.get("action_id") or 0) != 0), None)
    if active_event is None:
        return None

    onset_s = float(active_event["onset_s"])
    end_s = float(active_event.get("end_s") or (onset_s + float(active_event.get("duration_s") or 0.0)))
    anchor_local = float(active_event["local_ts"]) - onset_s
    view_start = max(0.0, onset_s - 0.20)
    view_end = onset_s + 0.80
    local_start = anchor_local + view_start
    local_end = anchor_local + view_end
    raw_t_local, raw_samples = _load_raw_segment_by_local_time(session_dir, local_start, local_end)
    if raw_t_local.size == 0 or raw_samples.size == 0:
        return None
    raw_t = raw_t_local - anchor_local

    windows_npz_path = session_dir / "processed" / "eeg_windows.npz"
    if not windows_npz_path.exists():
        return None
    npz = np.load(windows_npz_path, allow_pickle=True)
    window_start = np.asarray(npz["window_start"], dtype=float)
    window_end = np.asarray(npz["window_end"], dtype=float)
    channel_names = [str(v) for v in np.asarray(npz["channel_names"]).tolist()] if "channel_names" in npz else ["TP9", "AF7", "AF8", "TP10"]
    event_key = "event_index" if "event_index" in npz else ("event_id" if "event_id" in npz else None)
    if event_key is not None:
        event_values = np.asarray(npz[event_key]).astype(int)
        mask = event_values == int(active_event["_event_index"])
    else:
        mask = (window_end >= view_start) & (window_start <= view_end)
    mask = mask & (window_end >= view_start) & (window_start <= view_end)
    idx = np.where(mask)[0]
    if idx.size == 0:
        idx = np.where((window_start >= onset_s - 0.15) & (window_start <= onset_s + 0.20))[0]
    if idx.size == 0:
        return None
    idx = idx[np.argsort(window_start[idx])]
    if idx.size > 8:
        center = int(np.argmin(np.abs(window_start[idx] - onset_s)))
        lo = max(0, center - 3)
        hi = min(idx.size, lo + 8)
        idx = idx[lo:hi]
    highlight_i = int(np.argmin(np.abs(window_start[idx] - onset_s)))
    h_start = float(window_start[idx[highlight_i]])
    h_end = float(window_end[idx[highlight_i]])
    next_start = float(window_start[idx[min(highlight_i + 1, idx.size - 1)]])

    fig = plt.figure(figsize=(7.2, 4.8), dpi=200)
    gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.6], hspace=0.18)
    ax = fig.add_subplot(gs[0])
    axw = fig.add_subplot(gs[1], sharex=ax)

    centered = raw_samples - np.median(raw_samples, axis=0, keepdims=True)
    amp = np.nanpercentile(np.abs(centered), 95, axis=0)
    amp[~np.isfinite(amp) | (amp < 1.0)] = 1.0
    spacing = float(np.nanmax(amp) * 2.8)
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#7f7f7f"]
    offsets = np.arange(raw_samples.shape[1])[::-1] * spacing

    for ch in range(raw_samples.shape[1]):
        ax.plot(raw_t, centered[:, ch] + offsets[ch], color=colors[ch % len(colors)], linewidth=1.0)
    ax.axvspan(onset_s, end_s, color="#f3d36b", alpha=0.28, lw=0.0)
    ax.axvline(onset_s, color="#8a5a00", linestyle="--", linewidth=1.0)
    ax.axvline(end_s, color="#8a5a00", linestyle="--", linewidth=1.0)
    ax.text(
        onset_s + 0.16,
        offsets[0] + spacing * 0.48,
        f"{str(active_event.get('type') or '').replace('_', ' ').title()} event",
        fontsize=7.6,
        ha="left",
        va="bottom",
        color="#5a4300",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="#fff6d7", edgecolor="none", alpha=0.95),
    )
    ax.set_xlim(view_start, view_end)
    ax.set_yticks(offsets)
    ax.set_yticklabels(channel_names[: raw_samples.shape[1]])
    ax.set_ylabel("Muse 2 channel")
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.text(0.0, 1.03, "(a) Raw recorded segment", transform=ax.transAxes, fontsize=9.5, fontweight="bold", ha="left", va="bottom")
    ax.tick_params(axis="x", labelbottom=False)

    axw.axvspan(onset_s, end_s, color="#f3d36b", alpha=0.28, lw=0.0)
    bar_h = 0.72
    y_positions = np.arange(idx.size)[::-1]
    for row, i in enumerate(idx):
        y = y_positions[row]
        color = "#4c78a8" if row != highlight_i else "#d95f02"
        axw.broken_barh([(float(window_start[i]), float(window_end[i] - window_start[i]))], (y - bar_h / 2.0, bar_h), facecolors=color, edgecolors="white", linewidth=0.6, alpha=0.9)
    bottom_i = int(idx[-1])
    bottom_start = float(window_start[bottom_i])
    bottom_end = float(window_end[bottom_i])
    axw.annotate("", xy=(bottom_start, -0.8), xytext=(bottom_end, -0.8), arrowprops=dict(arrowstyle="<->", linewidth=0.9, color="#333333"))
    axw.text(
        (bottom_start + bottom_end) / 2.0,
        -1.00,
        "0.25 s window",
        ha="center",
        va="top",
        fontsize=6.8,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.92),
    )
    if idx.size > 1:
        hop_start = float(window_start[idx[0]])
        hop_end = float(window_start[idx[1]])
        hop_y = y_positions[1] + 0.35 if idx.size > 1 else y_positions[0] + 0.35
        axw.annotate("", xy=(hop_start, hop_y), xytext=(hop_end, hop_y), arrowprops=dict(arrowstyle="<->", linewidth=0.9, color="#333333"))
        axw.text(
            (hop_start + hop_end) / 2.0 - 0.018,
            hop_y - 0.34,
            "0.05 s hop",
            ha="center",
            va="top",
            fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.92),
        )
    axw.set_yticks(y_positions)
    axw.set_yticklabels([f"W{n+1}" for n in range(idx.size)])
    axw.set_xlabel("Session time (s)")
    axw.set_ylabel("Windows")
    axw.set_ylim(-1.4, max(1.2, float(y_positions.max()) + 1.2))
    axw.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.6)
    axw.text(0.0, 1.03, "(b) Sliding windows", transform=axw.transAxes, fontsize=9.5, fontweight="bold", ha="left", va="bottom")

    for axis in [ax, axw]:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    out_path = FIG_DIR / "raw_eeg_windowing_example.png"
    fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.12, hspace=0.20)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return str(out_path.relative_to(REPO_ROOT))


def _write_macros(runs: List[RunMetrics], demos: List[SubjectDemographics], repo_sha: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    featured = _load_featured_bundle()

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
        from utils.label_schema import ACTIVE_FINGER_IDS

        m = CNNLSTMFingerActionNet(
            n_channels=4,
            n_fingers=len(ACTIVE_FINGER_IDS),
            n_actions=3,
            finger_applicability_head=True,
        )
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

    featured_metrics = ((featured.get("model_metrics") or {}).get("test") or {})
    featured_eval = ((featured.get("eval_manifest") or {}).get("metrics") or {})
    featured_replay = ((featured.get("replay_manifest") or {}).get("replay_metrics") or {})
    featured_replay_summary = ((featured.get("replay_manifest") or {}).get("summary") or {})
    featured_runtime = ((featured.get("replay_manifest") or {}).get("runtime_config") or {})
    replay_prediction_counts = _summarize_replay_predictions(
        featured.get("replay_manifest") or {},
        Path(str(featured.get("replay_manifest_path"))) if featured.get("replay_manifest_path") else None,
    )
    featured_source_dir = Path(str((featured.get("manifest") or {}).get("source_session_dir") or ""))
    featured_prov = _session_provenance(featured_source_dir) if featured_source_dir.exists() else {}

    def _macro_percent(name: str, value: Optional[float], digits: int = 2) -> None:
        lines.append(
            rf"\newcommand{{\{name}}}{{"
            + (_format_pct(value, digits) if value is not None and np.isfinite(value) else r"\textit{n/a}")
            + r"}"
            + "\n"
        )

    def _macro_number(name: str, value: Optional[float], digits: int = 4) -> None:
        lines.append(
            rf"\newcommand{{\{name}}}{{"
            + (_format_num(value, digits) if value is not None and np.isfinite(value) else r"\textit{n/a}")
            + r"}"
            + "\n"
        )

    def _macro_count(name: str, value: Optional[int]) -> None:
        lines.append(
            rf"\newcommand{{\{name}}}{{"
            + (str(int(value)) if value is not None else r"\textit{n/a}")
            + r"}"
            + "\n"
        )

    _macro_percent("FeaturedActionAcc", featured_eval.get("action_acc") or featured_metrics.get("action_acc"))
    _macro_percent("FeaturedFingerAcc", featured_eval.get("finger_acc_non_rest") or featured_metrics.get("finger_acc_non_rest"))
    _macro_count("FeaturedNTest", featured_metrics.get("n_test"))
    _macro_count("FeaturedNNonRest", featured_metrics.get("n_test_non_rest"))
    _macro_percent("FeaturedJointAcc", featured_eval.get("joint_acc"))
    _macro_percent("FeaturedReplayCommittedActionAcc", featured_replay.get("committed_action_acc"))
    _macro_percent("FeaturedReplayPrecision", featured_replay.get("would_send_window_precision_non_rest"))
    _macro_percent("FeaturedReplayRecall", featured_replay.get("would_send_window_recall_non_rest"))
    _macro_percent("FeaturedFalseActuationRateRest", featured_replay.get("false_actuation_rate_rest"))
    _macro_count("FeaturedReplayRestWindows", replay_prediction_counts.get("rest_windows"))
    _macro_count("FeaturedReplayNonRestWindows", replay_prediction_counts.get("non_rest_windows"))
    _macro_count("FeaturedReplayPositiveSendWindows", replay_prediction_counts.get("positive_send_windows"))
    _macro_count("FeaturedReplayCorrectSendWindows", replay_prediction_counts.get("correct_send_windows"))
    _macro_count("FeaturedReplayFalseActuationCount", replay_prediction_counts.get("false_actuation_rest_count"))
    _macro_count("FeaturedActuationStabilityWindows", featured_runtime.get("actuation_stability"))
    stability_ms = None
    if featured_runtime.get("actuation_stability") is not None and featured_runtime.get("hop_sec") is not None:
        stability_ms = 1000.0 * float(featured_runtime.get("actuation_stability")) * float(featured_runtime.get("hop_sec"))
    _macro_number("FeaturedActuationStabilityMs", stability_ms, digits=0)
    _macro_number("FeaturedActionECE", featured_eval.get("action_ece"))
    _macro_number("FeaturedFingerECE", featured_eval.get("finger_ece_non_rest"))
    _macro_number(
        "FeaturedReplayLatencyMeanMs",
        ((featured_replay_summary.get("latency_ms") or {}).get("mean")),
        digits=1,
    )
    _macro_count("FeaturedDerivedSourceN", featured_prov.get("filter_source_n"))
    _macro_count("FeaturedDerivedRemovedN", featured_prov.get("filter_removed_n"))
    _macro_count("FeaturedDerivedKeptN", featured_prov.get("filter_kept_n"))

    (OUT_DIR / "paper_macros.tex").write_text("".join(lines), encoding="utf-8")


def _write_tables(runs: List[RunMetrics], demos: List[SubjectDemographics], session_meta: Dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sorted_runs = sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id))
    raw_session_role: Dict[str, str] = {}
    role_rank = {"recorded": 0, "core": 1, "aux_rest": 2, "supporting": 3}
    for run in sorted_runs:
        for row in _raw_support_sessions_for_run(run):
            session_id = str(row["session_id"])
            role = str(row["role"])
            current = raw_session_role.get(session_id)
            if current is None or role_rank.get(role, 99) < role_rank.get(current, 99):
                raw_session_role[session_id] = role
    raw_sessions = sorted(
        session_meta.get("sessions", []),
        key=lambda s: (
            role_rank.get(raw_session_role.get(str(s.get("session_id")), "supporting"), 99),
            str(s.get("subject_id") or ""),
            str(s.get("session_id") or ""),
        ),
    )

    # Demographics table
    demo_lines: List[str] = []
    demo_lines.append("% AUTO-GENERATED by scripts/build_paper_artifacts.py. DO NOT EDIT BY HAND.\n")
    demo_lines.append("\\begin{table}[t]\n\\centering\n")
    demo_lines.append("\\caption{Subject demographics from the repository metadata.}\n")
    demo_lines.append("\\label{tab:demo}\n")
    demo_lines.append("\\begin{tabular}{llll}\n\\toprule\n")
    demo_lines.append("Subject & Age & Sex & Handedness \\\\\n\\midrule\n")
    for d in sorted(demos, key=lambda x: x.subject_id):
        age = str(d.age) if d.age is not None else r"\textit{n/a}"
        sex = _latex_escape(d.sex) if d.sex else r"\textit{n/a}"
        hand = _latex_escape(d.handedness) if d.handedness else r"\textit{n/a}"
        demo_lines.append(f"{_latex_escape(_display_subject_id(d.subject_id))} & {age} & {sex} & {hand} \\\\\n")
    demo_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

    # Performance table per run
    perf_lines: List[str] = []
    perf_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")
    perf_lines.append("\\caption{Representative subject-specific runs with held-out performance and calibration. Finger accuracy is computed with the deployment-consistent action-conditioned decode path on true non-REST windows.}\n")
    perf_lines.append("\\label{tab:perf}\n")
    perf_lines.append("\\resizebox{\\textwidth}{!}{%\n")
    perf_lines.append("\\begin{tabular}{llp{0.28\\textwidth}rrrrrrrr}\n\\toprule\n")
    perf_lines.append(
        "Subject & Run & Session & $n_{test}$ & $n_{non\\text{-}REST}$ & Action Acc (\\%) & 95\\% CI & Finger Acc$_{non\\text{-}REST}$ (\\%) & 95\\% CI & Action ECE & Finger ECE$_{non\\text{-}REST}$ \\\\\n\\midrule\n"
    )
    for r in sorted_runs:
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
        session_id = _latex_breakable_id(_paper_session_label(r.session_id))
        perf_lines.append(
            f"{_latex_escape(_display_subject_id(r.subject_id))} & {_latex_breakable_id(r.run_id)} & {session_id} & {int(r.n_test_metrics or 0)} & {int(r.n_test_non_rest_metrics or 0)} & "
            f"{_format_pct(action_acc, 2)} & {a_ci} & {_format_pct(finger_acc, 2)} & {f_ci} & "
            f"{_format_num(r.test_action_ece, 4)} & {_format_num(r.test_finger_ece_non_rest, 4)} \\\\\n"
        )
    perf_lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n\n")

    # Dataset / windowing table per session
    dataset_rows: Dict[Tuple[str, str], RunMetrics] = {}
    for r in sorted_runs:
        dataset_rows.setdefault((r.subject_id, r.session_id), r)
    data_lines: List[str] = []
    data_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")
    data_lines.append("\\caption{Dataset and window extraction summary for the representative datasets reported in this manuscript.}\n")
    data_lines.append("\\label{tab:dataset}\n")
    data_lines.append("\\resizebox{\\textwidth}{!}{%\n")
    data_lines.append("\\begin{tabular}{lp{0.34\\textwidth}rrrrrr}\n\\toprule\n")
    data_lines.append(
        "Subject & Session & $N$ windows & Window (s) & Hop (s) & Mean overlap (\\%) & Artifacts (count) & Gaps (count) \\\\\n\\midrule\n"
    )
    for (_, _), r in dataset_rows.items():
        ov_pct = (float(r.overlap_frac_mean) * 100.0) if r.overlap_frac_mean is not None else None
        win_str = f"{float(r.window_sec):.3f}" if r.window_sec is not None else "\\textit{n/a}"
        hop_str = f"{float(r.step_sec):.3f}" if r.step_sec is not None else "\\textit{n/a}"
        ov_str = f"{ov_pct:.1f}" if ov_pct is not None and np.isfinite(ov_pct) else "\\textit{n/a}"
        art_str = str(r.artifact_count) if r.artifact_count is not None else "\\textit{n/a}"
        gapc_str = str(r.gap_count) if r.gap_count is not None else "\\textit{n/a}"
        data_lines.append(
            f"{_latex_escape(_display_subject_id(r.subject_id))} & {_latex_breakable_id(_paper_session_label(r.session_id))} & {int(r.n_windows_total or 0)} & "
            f"{win_str} & {hop_str} & {ov_str} & {art_str} & {gapc_str} \\\\\n"
        )
    data_lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n\n")

    # Representative dataset provenance table
    dur_lines: List[str] = []
    dur_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n")
    dur_lines.append("\\caption{Representative dataset provenance. The featured 2-M16 run is trained on a filtered derived dataset rather than on a single recorded session.}\n")
    dur_lines.append("\\label{tab:sessions}\n")
    dur_lines.append("\\resizebox{\\textwidth}{!}{%\n")
    dur_lines.append("\\begin{tabular}{lllp{0.20\\textwidth}p{0.39\\textwidth}}\n\\toprule\n")
    dur_lines.append("Subject & Dataset & Kind & Source & Notes \\\\\n\\midrule\n")
    for r in sorted_runs:
        prov = _session_provenance(_session_dir_from_rel(r.session_dir_rel))
        subj = _latex_escape(_display_subject_id(r.subject_id))
        dataset_id = _latex_breakable_id(_paper_session_label(r.session_id))
        if prov["kind"] == "recorded":
            support = next((s for s in raw_sessions if str(s.get("session_id")) == r.session_id), None)
            note_parts: List[str] = []
            if support is not None:
                recv = support.get("samples_received")
                written = support.get("samples_written")
                dur_s = support.get("duration_s_from_samples")
                drop_pct = None
                try:
                    if recv is not None and written is not None and float(recv) > 0:
                        drop_pct = (1.0 - (float(written) / float(recv))) * 100.0
                except Exception:
                    drop_pct = None
                if dur_s is not None:
                    note_parts.append(f"{float(dur_s) / 60.0:.1f} min")
                if recv is not None:
                    note_parts.append(f"{int(recv)} samples")
                if drop_pct is not None and np.isfinite(drop_pct):
                    note_parts.append(f"{drop_pct:.2f}\\% write drop")
            notes = "; ".join(note_parts) if note_parts else "Direct recorded session."
            dur_lines.append(
                f"{subj} & {dataset_id} & Recorded session & {dataset_id} & {notes} \\\\\n"
            )
            continue

        source = _latex_breakable_id(_display_session_id(str(prov.get("filter_source_session") or r.session_id)))
        core_sessions = [str(s) for s in prov.get("core_sessions", [])]
        aux_sessions = [str(s) for s in prov.get("aux_rest_sessions", [])]
        notes: List[str] = []
        if core_sessions:
            notes.append(f"{len(core_sessions)} movement session" + ("" if len(core_sessions) == 1 else "s"))
        if aux_sessions:
            notes.append(f"{len(aux_sessions)} aux REST session" + ("" if len(aux_sessions) == 1 else "s"))
        if prov.get("filter_removed_n") is not None and prov.get("filter_source_n") is not None:
            notes.append(
                f"Pruned {int(prov['filter_removed_n'])}/{int(prov['filter_source_n'])} windows"
            )
        if prov.get("filter_event_ids"):
            evs = ",".join(str(v) for v in prov["filter_event_ids"])
            filt_session = str(prov.get("filter_session_id") or "")
            filt_parts = _display_session_id(filt_session).split("_")
            filt_short = "_".join(filt_parts[1:3]) if len(filt_parts) >= 3 else (_display_session_id(filt_session) or "source")
            notes.append(f"Removed REST {evs} from {_latex_escape(filt_short)}")
        dur_lines.append(
            f"{subj} & {dataset_id} & Filtered derived & {source} & {'; '.join(notes)} \\\\\n"
        )
    dur_lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n\n")

    # Event / movement counts (exact, from events/events.jsonl)
    # Use union of event 'type' values across sessions and rotate headers to fit IEEE table* width.
    type_set = set()
    for s in raw_sessions:
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
    event_header_labels = {
        "rest": r"\shortstack{REST}",
        "none_open": r"\shortstack{NONE\\OPEN}",
        "thumb_open": r"\shortstack{TH\\OPEN}",
        "thumb_close": r"\shortstack{TH\\CLOSE}",
        "index_open": r"\shortstack{IN\\OPEN}",
        "index_close": r"\shortstack{IN\\CLOSE}",
        "middle_open": r"\shortstack{MID\\OPEN}",
        "middle_close": r"\shortstack{MID\\CLOSE}",
        "ring_open": r"\shortstack{RING\\OPEN}",
        "ring_close": r"\shortstack{RING\\CLOSE}",
        "pinky_open": r"\shortstack{PINK\\OPEN}",
        "pinky_close": r"\shortstack{PINK\\CLOSE}",
    }

    ev_lines: List[str] = []
    ev_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    ev_lines.append(
        "\\caption{Raw-session event label counts for the recorded sessions underlying the representative datasets. Role indicates the representative recorded session (Recorded), a movement session merged into a derived dataset (Core), or an auxiliary REST-only session added for REST coverage (Aux REST). Column headers abbreviate thumb/index/middle/ring/pinky open and close events.}\n"
    )
    ev_lines.append("\\label{tab:events}\n")
    ev_lines.append("\\setlength{\\tabcolsep}{4pt}\n")
    ev_lines.append("\\begin{tabular}{lll" + "r" * len(types) + "}\n\\toprule\n")
    header_cells = ["Subject", "Session", "Role"] + [event_header_labels.get(t, _latex_escape(t)) for t in types]
    ev_lines.append(" & ".join(header_cells) + " \\\\\n\\midrule\n")
    role_labels = {
        "recorded": "Recorded",
        "core": "Core",
        "aux_rest": "Aux REST",
        "supporting": "Supporting",
    }
    for s in raw_sessions:
        subj = _latex_escape(_display_subject_id(str(s.get("subject_id") or "")))
        sess_id = _latex_breakable_id(_display_session_id(str(s.get("session_id"))))
        role = role_labels.get(raw_session_role.get(str(s.get("session_id")), "supporting"), "Supporting")
        ec = s.get("events_counts") or {}
        cbt = ec.get("counts_by_type") if ec.get("available") else {}
        row = [subj, sess_id, _latex_escape(role)]
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
    pc_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    pc_lines.append("\\caption{Action-class accuracies for the representative manuscript runs.}\n")
    pc_lines.append("\\label{tab:perclass-action}\n")
    pc_lines.append("\\setlength{\\tabcolsep}{3pt}\n")
    pc_lines.append("\\begin{tabular}{lp{0.30\\textwidth}lrr}\n\\toprule\n")
    pc_lines.append("Subject & Run & Class & Count & Accuracy (\\%) \\\\\n\\midrule\n")
    for r in sorted_runs:
        for cls_str, acc in sorted(r.test_action_acc_by_class.items(), key=lambda kv: int(kv[0])):
            cls = int(cls_str)
            name = ACTION_NAMES.get(cls, cls_str)
            count = int(r.test_action_counts.get(cls_str, 0))
            pc_lines.append(
                f"{_latex_escape(_display_subject_id(r.subject_id))} & {_latex_breakable_id(r.run_id)} & {_latex_escape(name)} & {count} & {acc*100.0:.2f} \\\\\n"
            )
    pc_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
    pc_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    pc_lines.append("\\caption{Finger-class accuracies on true non-REST test windows for the representative manuscript runs, using the deployment-consistent action-conditioned finger decode path.}\n")
    pc_lines.append("\\label{tab:perclass-finger}\n")
    pc_lines.append("\\setlength{\\tabcolsep}{3pt}\n")
    pc_lines.append("\\begin{tabular}{lp{0.30\\textwidth}lrr}\n\\toprule\n")
    pc_lines.append("Subject & Run & Class & Count & Accuracy (\\%) \\\\\n\\midrule\n")
    for r in sorted_runs:
        rest_count = int(r.test_action_counts.get("0", 0))
        for cls_str, acc in sorted(r.test_finger_acc_by_class_non_rest.items(), key=lambda kv: int(kv[0])):
            cls = int(cls_str)
            name = FINGER_NAMES.get(cls, cls_str)
            if cls == 0:
                count = max(0, int(r.test_finger_counts.get("0", 0)) - rest_count)
            else:
                count = int(r.test_finger_counts.get(cls_str, 0))
            pc_lines.append(
                f"{_latex_escape(_display_subject_id(r.subject_id))} & {_latex_breakable_id(r.run_id)} & {_latex_escape(name)} & {count} & {acc*100.0:.2f} \\\\\n"
            )
    pc_lines.append("\\bottomrule\n\\end{tabular}\n\\end{table*}\n\n")
    (OUT_DIR / "tables_perclass.tex").write_text("".join(pc_lines), encoding="utf-8")

    # Bootstrap CI table (optional rigor supplement)
    boot_lines: List[str] = []
    boot_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    boot_lines.append("\\caption{Bootstrap 95\\% confidence intervals for the representative manuscript runs.}\n")
    boot_lines.append("\\label{tab:bootci}\n")
    boot_lines.append("\\resizebox{\\textwidth}{!}{%\n")
    boot_lines.append("\\begin{tabular}{llrrrr}\n\\toprule\n")
    boot_lines.append("Subject & Run & Action Acc (\\%) & 95\\% CI & Finger Acc$_{non\\text{-}REST}$ (\\%) & 95\\% CI \\\\\n\\midrule\n")
    for r in sorted_runs:
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
            f"{_latex_escape(_display_subject_id(r.subject_id))} & {_latex_breakable_id(r.run_id)} & {_format_pct(r.test_action_acc_metrics, 2)} & {a_ci} & {_format_pct(r.test_finger_acc_non_rest_metrics, 2)} & {f_ci} \\\\\n"
        )
    boot_lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n\n")
    (OUT_DIR / "tables_bootstrap_ci.tex").write_text("".join(boot_lines), encoding="utf-8")

    # Train vs test generalization gaps (from metrics.json train/test blocks)
    gap_lines: List[str] = []
    gap_lines.append("\\begin{table*}[t]\n\\centering\n\\scriptsize\n")
    gap_lines.append("\\caption{Train--test generalization gaps (percentage points) for the representative manuscript runs.}\n")
    gap_lines.append("\\label{tab:gap}\n")
    gap_lines.append("\\resizebox{\\textwidth}{!}{%\n")
    gap_lines.append("\\begin{tabular}{llrrrrrr}\n\\toprule\n")
    gap_lines.append(
        "Subject & Run & Train Action (\\%) & Test Action (\\%) & Gap (pp) & Train Finger (\\%) & Test Finger$_{non\\text{-}REST}$ (\\%) & Gap (pp) \\\\\n\\midrule\n"
    )
    for r in sorted_runs:
        def _gap_pp(train: Optional[float], test: Optional[float]) -> str:
            if train is None or test is None or (not np.isfinite(train)) or (not np.isfinite(test)):
                return "\\textit{n/a}"
            return f"{(float(train) - float(test)) * 100.0:.2f}"

        gap_lines.append(
            f"{_latex_escape(_display_subject_id(r.subject_id))} & {_latex_breakable_id(r.run_id)} & "
            f"{_format_pct(r.train_action_acc, 2)} & {_format_pct(r.test_action_acc_metrics, 2)} & {_gap_pp(r.train_action_acc, r.test_action_acc_metrics)} & "
            f"{_format_pct(r.train_finger_acc, 2)} & {_format_pct(r.test_finger_acc_non_rest_raw_head, 2)} & {_gap_pp(r.train_finger_acc, r.test_finger_acc_non_rest_raw_head)} \\\\\n"
        )
    gap_lines.append("\\bottomrule\n\\end{tabular}%\n}\n\\end{table*}\n\n")
    (OUT_DIR / "tables_generalization_gap.tex").write_text("".join(gap_lines), encoding="utf-8")


def _write_figures(runs: List[RunMetrics]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_finger_accuracy_bar_chart(runs)
    _write_featured_provenance_figure(runs)
    _write_raw_windowing_figure(runs)

    lines: List[str] = []
    lines.append("% AUTO-GENERATED by scripts/build_paper_artifacts.py. DO NOT EDIT BY HAND.\n")

    for r in sorted(runs, key=lambda x: (x.subject_id, x.session_id, x.run_id)):
        if not any([r.fig_action_confusion, r.fig_finger_confusion, r.fig_reliability, r.fig_scatter]):
            continue
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
            safe_path = path.replace("\\", "/")
            if not safe_path.startswith("../"):
                safe_path = f"../{safe_path}"
            return (
                "\\begin{minipage}[b]{0.49\\linewidth}\\centering\n"
                + f"\\includegraphics[width=\\linewidth]{{{safe_path}}}\n"
                + "\\\\"
                + "\\footnotesize "
                + caption
                + "\n\\end{minipage}\n"
            )

        subj = _latex_escape(_display_subject_id(r.subject_id))
        lines.append(inc(r.fig_action_confusion, f"(a) Action confusion matrix ({subj})."))
        lines.append(inc(r.fig_finger_confusion, f"(b) Finger confusion matrix ({subj})."))
        lines.append("\\\\[0.5em]\n")
        lines.append(inc(r.fig_reliability, f"(c) Reliability / calibration summary ({subj})."))
        lines.append(inc(r.fig_scatter, f"(d) MC-dropout action confidence-uncertainty plot ({subj})."))
        if r == max(runs, key=_run_rank_key):
            fig_caption = f"Representative evaluation figures for the featured {subj} run."
        else:
            fig_caption = f"Representative evaluation figures for {subj}."
        lines.append(f"\\caption{{{fig_caption}}}\n")
        lines.append("\\end{figure*}\n\n")

    (OUT_DIR / "figures.tex").write_text("".join(lines), encoding="utf-8")


def _prune_unused_paper_figures() -> None:
    if not FIG_DIR.exists():
        return

    include_re = re.compile(r"includegraphics(?:\[[^\]]*\])?\{(?:\.\./)?paper_figures/([^}]+)\}")
    keep: set[str] = set()
    for tex_path in (REPO_ROOT / "paper" / "research_paper.tex", OUT_DIR / "figures.tex"):
        if not tex_path.exists():
            continue
        text = tex_path.read_text(encoding="utf-8")
        keep.update(m.group(1) for m in include_re.finditer(text))

    for path in FIG_DIR.iterdir():
        if not path.is_file():
            continue
        if path.name not in keep:
            path.unlink()


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
    all_run_metrics: List[RunMetrics] = []
    for metrics_path in sorted(PROJECTS_ROOT.rglob("processed/models/*/metrics.json")):
        all_run_metrics.append(_compute_run_metrics(metrics_path))

    run_metrics = _select_best_runs_per_subject(all_run_metrics)

    # Demographics (subject.json)
    demos = _scan_subject_demographics()
    selected_subjects = {r.subject_id for r in run_metrics}
    demos = [d for d in demos if d.subject_id in selected_subjects]

    # Session meta (run_meta + events) for each subject appearing in runs
    session_meta_by_subject: Dict[str, Any] = {}
    for subj in sorted(selected_subjects):
        subject_session_ids = set()
        for run in run_metrics:
            if run.subject_id != subj:
                continue
            for row in _raw_support_sessions_for_run(run):
                subject_session_ids.add(str(row["session_id"]))
        session_meta_by_subject[subj] = _scan_session_meta(subj, session_ids=subject_session_ids)

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
        "n_available_runs": len(all_run_metrics),
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
    _prune_unused_paper_figures()

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
