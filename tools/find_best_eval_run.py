#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


RE_ACTION_ACC = re.compile(r"Action Accuracy: ([0-9.]+)%")
RE_FINGER_ACC = re.compile(r"Finger Accuracy \(non-REST\): ([0-9.]+)%")
RE_ACTION_ECE = re.compile(r"Action ECE: ([0-9.]+)")
RE_FINGER_ECE = re.compile(r"Finger ECE \(non-REST\): ([0-9.]+)")


@dataclass
class EvalResult:
    npz: str
    model: str
    scaler: str
    subject_id: str
    command: List[str]
    action_acc: Optional[float]
    finger_acc: Optional[float]
    action_ece: Optional[float]
    finger_ece: Optional[float]
    score: Optional[float]
    returncode: int


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _unique_subject_id(npz_path: Path) -> Optional[str]:
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    if "subject_id" not in data:
        return None
    try:
        subjects = np.asarray(data["subject_id"]).astype(str)
    except Exception:
        return None
    if subjects.ndim == 0:
        return str(subjects)
    unique = np.unique(subjects)
    if len(unique) == 1:
        return str(unique[0])
    return None


def _find_windows(root: Path) -> List[Path]:
    windows = list(root.rglob("*windows*.npz"))
    default_npz = root / "eeg_windows.npz"
    if default_npz.exists():
        windows.append(default_npz)
    return sorted({p.resolve() for p in windows})


def _find_models(root: Path) -> List[Path]:
    return sorted({p.resolve() for p in root.rglob("*.pt")})


def _find_scalers(root: Path) -> List[Path]:
    candidates = []
    for pattern in ("scaler.save", "*normalizer*.save", "*scaler*.pkl", "*scaler*.joblib", "*normalizer*.pkl", "*normalizer*.joblib"):
        candidates.extend(root.rglob(pattern))
    return sorted({p.resolve() for p in candidates})


def _find_train_configs(root: Path) -> List[Path]:
    return sorted({p.resolve() for p in root.rglob("train_config.json")})


def _load_json(path: Path) -> Dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _candidate_from_config(cfg_path: Path, root: Path) -> Optional[Tuple[Path, Path, Path, Optional[str]]]:
    cfg = _load_json(cfg_path)
    if not cfg:
        return None
    npz_path = cfg.get("npz_path")
    model_path = cfg.get("save_model_path")
    scaler_path = cfg.get("save_scaler_path")
    subject_id = cfg.get("subject_id_filter")
    if npz_path:
        npz_path = (root / npz_path).resolve() if not os.path.isabs(npz_path) else Path(npz_path)
    if model_path:
        model_path = Path(model_path)
    if scaler_path:
        scaler_path = Path(scaler_path)
    if not (npz_path and model_path and scaler_path):
        return None
    if not (npz_path.exists() and model_path.exists() and scaler_path.exists()):
        return None
    return npz_path.resolve(), model_path.resolve(), scaler_path.resolve(), subject_id


def _pair_models_and_scalers(models: Iterable[Path], scalers: Iterable[Path]) -> List[Tuple[Path, Path]]:
    scalers_by_dir: Dict[Path, List[Path]] = {}
    for scaler in scalers:
        scalers_by_dir.setdefault(scaler.parent, []).append(scaler)
    pairs = []
    for model in models:
        candidates = scalers_by_dir.get(model.parent, [])
        for scaler in candidates:
            pairs.append((model, scaler))
    return pairs


def _build_candidates(root: Path) -> List[Tuple[Path, Path, Path, Optional[str]]]:
    windows = _find_windows(root)
    models = _find_models(root)
    scalers = _find_scalers(root)
    configs = _find_train_configs(root)

    candidates: List[Tuple[Path, Path, Path, Optional[str]]] = []
    for cfg_path in configs:
        candidate = _candidate_from_config(cfg_path, root)
        if candidate is not None:
            candidates.append(candidate)

    model_pairs = _pair_models_and_scalers(models, scalers)
    for model, scaler in model_pairs:
        for npz_path in windows:
            candidates.append((npz_path, model, scaler, None))

    deduped = []
    seen = set()
    for npz_path, model, scaler, subject_id in candidates:
        key = (str(npz_path), str(model), str(scaler), subject_id or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append((npz_path, model, scaler, subject_id))
    return deduped


def _parse_metrics(stdout: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    action_acc = None
    finger_acc = None
    action_ece = None
    finger_ece = None

    match = RE_ACTION_ACC.search(stdout)
    if match:
        action_acc = float(match.group(1))
    match = RE_FINGER_ACC.search(stdout)
    if match:
        finger_acc = float(match.group(1))
    match = RE_ACTION_ECE.search(stdout)
    if match:
        action_ece = float(match.group(1))
    match = RE_FINGER_ECE.search(stdout)
    if match:
        finger_ece = float(match.group(1))

    return action_acc, finger_acc, action_ece, finger_ece


def _run_eval(npz: Path, model: Path, scaler: Path, subject_id: Optional[str]) -> EvalResult:
    subject_id_arg = subject_id
    if subject_id_arg is None:
        subject_id_arg = ""
    cmd = [
        sys.executable,
        str(_repo_root() / "3_evaluate_model.py"),
        "--npz",
        str(npz),
        "--model",
        str(model),
        "--scaler",
        str(scaler),
        "--subject-id",
        subject_id_arg,
    ]
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "")
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    stdout = proc.stdout + (proc.stderr or "")
    action_acc, finger_acc, action_ece, finger_ece = _parse_metrics(stdout)
    score = None
    if action_acc is not None:
        score = action_acc + (finger_acc or 0.0)
    return EvalResult(
        npz=str(npz),
        model=str(model),
        scaler=str(scaler),
        subject_id=subject_id_arg,
        command=cmd,
        action_acc=action_acc,
        finger_acc=finger_acc,
        action_ece=action_ece,
        finger_ece=finger_ece,
        score=score,
        returncode=proc.returncode,
    )


def _rank_results(results: Sequence[EvalResult]) -> List[EvalResult]:
    return sorted(
        results,
        key=lambda r: (
            r.score is None,
            -(r.score or 0.0),
            -(r.action_acc or 0.0),
            -(r.finger_acc or 0.0),
        ),
    )


def _write_outputs(results: Sequence[EvalResult], out_json: Path, out_txt: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(r) for r in results]
    out_json.write_text(json.dumps(payload, indent=2))

    lines = []
    lines.append("rank\taction_acc\tfinger_acc\taction_ece\tfinger_ece\tscore\tnpz\tmodel\tscaler\tsubject_id")
    for idx, r in enumerate(results, start=1):
        lines.append(
            f"{idx}\t{r.action_acc}\t{r.finger_acc}\t{r.action_ece}\t{r.finger_ece}\t{r.score}\t{r.npz}\t{r.model}\t{r.scaler}\t{r.subject_id}"
        )
    out_txt.write_text("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of evaluations")
    args = parser.parse_args()

    root = _repo_root()
    candidates = _build_candidates(root)
    if args.max_runs:
        candidates = candidates[: args.max_runs]

    results: List[EvalResult] = []
    for npz, model, scaler, subject_id in candidates:
        result = _run_eval(npz, model, scaler, subject_id)
        results.append(result)

    ranked = _rank_results(results)
    out_json = root / "reports" / "best_eval_candidates.json"
    out_txt = root / "reports" / "best_eval_candidates.txt"
    _write_outputs(ranked, out_json, out_txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
