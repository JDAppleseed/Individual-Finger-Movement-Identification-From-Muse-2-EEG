#!/usr/bin/env python3
"""Evaluate every finite window of a subject corpus, including development data.

This diagnostic does not require a saved holdout and never creates a new split.
It does not replace held-out evaluation or reproduce gated command replay.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.sequence_data import load_sequence_npz
from utils.live_infer_common import load_model_artifacts


def evaluate(npz_path, run_dir, subject_id=None):
    npz_path, run_dir = Path(npz_path), Path(run_dir)
    X, a, f, meta = load_sequence_npz(npz_path)
    subjects = np.asarray(meta.get('subject_id', np.full(len(a), 'UNKNOWN'))).astype(str)
    if subject_id is None and len(np.unique(subjects)) != 1:
        raise ValueError('Multiple subjects: supply --subject-id for per-subject evaluation')
    mask = np.ones(len(a), dtype=bool) if subject_id is None else subjects == subject_id
    selected = np.flatnonzero(mask)
    if not len(selected):
        raise ValueError('No windows match the selected subject')
    channels = len(meta['channel_names']) if 'channel_names' in meta else 4
    if X.shape[-1] != channels:
        if X.shape[1] != channels:
            raise ValueError('Signal layout does not match channel count')
        X = X.transpose(0, 2, 1)
    finite = np.isfinite(X).all(axis=(1, 2))
    indices = selected[finite[selected]]
    if not len(indices):
        raise ValueError('No finite windows remain')
    spec = importlib.util.spec_from_file_location('full_dataset_evaluator', ROOT / '3_evaluate_model.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    device = torch.device('cpu')
    model, normalizer, temperature = load_model_artifacts(run_dir=run_dir, device=device, n_channels=channels)
    action, finger, applicability = module._run_deterministic_inference(
        X=X, indices=indices, model=model, normalizer=normalizer, temperature_state=temperature,
        n_actions=model.action_head.out_features, n_fingers=model.finger_head.out_features,
        batch_size=256, device=device, amp_dtype=None)
    metrics = module._compute_prediction_metrics(action_probs=action, finger_probs=finger,
        applicability_probs=applicability, y_action_true=a[indices], y_finger_true=f[indices])['metrics']
    sessions = np.asarray(meta.get('session_id', np.full(len(a), 'UNKNOWN'))).astype(str)
    hashes = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
              for name in ('finger_action_model.pt', 'scaler.npz', 'temperature_scaling.json')
              if (run_dir / name).exists()}
    return dict(scope='full_subject_corpus_including_training_and_calibration',
        independent_holdout=False, evaluation='unsmoothed_classifier; not command replay',
        subject_ids=np.unique(subjects[indices]).tolist(), dataset=str(npz_path), run=str(run_dir),
        dataset_sha256=hashlib.sha256(npz_path.read_bytes()).hexdigest(), artifact_sha256=hashes,
        selected_windows=len(selected), evaluated_windows=len(indices),
        excluded_nonfinite_windows=len(selected)-len(indices),
        sessions={sid: int(np.sum(sessions[indices] == sid)) for sid in np.unique(sessions[indices])},
        action_counts=np.bincount(a[indices], minlength=3).tolist(), metrics=metrics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preset', help='Name from configs/baselines/full_dataset_evaluations.json')
    parser.add_argument('--npz')
    parser.add_argument('--run-dir')
    parser.add_argument('--subject-id')
    parser.add_argument('--output', required=True, help='Separate full-corpus diagnostic JSON')
    args = parser.parse_args()
    if args.preset:
        if args.npz or args.run_dir:
            parser.error('--preset cannot be combined with --npz or --run-dir')
        presets = json.loads((ROOT / 'configs/baselines/full_dataset_evaluations.json').read_text())
        matches = [p for p in presets if p['name'] == args.preset]
        if not matches:
            parser.error('Unknown preset; choose: ' + ', '.join(p['name'] for p in presets))
        args.npz, args.run_dir = ROOT / matches[0]['dataset'], ROOT / matches[0]['run']
    elif not args.npz or not args.run_dir:
        parser.error('Supply --preset or both --npz and --run-dir')
    target = Path(args.output)
    if target.exists():
        raise FileExistsError(f'Refusing to overwrite existing report: {target}')
    torch.set_num_threads(1)
    result = evaluate(args.npz, args.run_dir, args.subject_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
