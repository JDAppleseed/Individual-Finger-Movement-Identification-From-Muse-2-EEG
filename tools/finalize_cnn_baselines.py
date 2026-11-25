#!/usr/bin/env python3
"""Audit, train, and summarize the frozen three-subject CNN-LSTM comparison.

Run each stage explicitly. Existing runs are never overwritten. No serial or
LSL access occurs here. Data stays local; aggregate reports can be published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.sequence_data import load_sequence_npz
from utils.splitting import compose_split_indices, infer_groups

SESSIONS = {
    '1-M16': 'Projects/1-M16/subjects/1-M16 500/sessions/combined_20260927_full',
    '2-M16': 'Projects/2-M16/subjects/2-M16/sessions/combined_20260428_with_fixed_session',
    '3-M16': 'Projects/3-M16/subjects/3-M16/sessions/combined_20260927_full',
}
RUN = '20261125_cnn_lstm_baseline_seed43'
REPORTS = ROOT / 'docs/research/finalization_20261125'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / name).write_text(json.dumps(data, indent=2) + '\n')


def audit():
    rows = []
    for subject, session in SESSIONS.items():
        path = ROOT / session / 'processed/eeg_windows.npz'
        X, a, f, meta = load_sequence_npz(path)
        n = len(a)
        if X.shape != (n, 64, 4) or not np.isfinite(X).all():
            raise ValueError(f'{subject}: invalid signal array')
        if not np.isin(a, [0, 1, 2]).all() or not np.isin(f, range(6)).all():
            raise ValueError(f'{subject}: invalid labels')
        if not np.array_equal(a == 0, f == 0):
            raise ValueError(f'{subject}: inconsistent REST/NONE labels')
        starts = np.asarray(meta['window_start'], float)
        ends = np.asarray(meta['window_end'], float)
        if not np.isfinite(starts).all() or not np.allclose(ends - starts, .25, atol=2e-5, rtol=0):
            raise ValueError(f'{subject}: invalid window timing')
        labels = np.asarray(meta['channel_names']).reshape(-1).tolist()
        if labels != ['TP9', 'AF7', 'AF8', 'TP10'] or float(meta['fs']) != 256:
            raise ValueError(f'{subject}: montage/sample rate mismatch: {labels}')
        sessions = np.asarray(meta['session_id']).astype(str)
        identities = list(zip(sessions, starts.tolist()))
        if len(set(identities)) != n:
            raise ValueError(f'{subject}: duplicate source windows')
        plan = compose_split_indices(a, f, meta, test_size=.2, random_state=43,
            split_mode='group_trial', purge_seconds=.25, hop_seconds=None,
            allow_fallback=False, aux_rest_session_policy='auto_train_only')
        groups = np.array([f"{sid}:{gid}" for sid, gid in zip(sessions, infer_groups(meta, n))])
        if np.intersect1d(groups[plan['train_idx']], groups[plan['test_idx']]).size:
            raise ValueError(f'{subject}: outer group overlap')
        source_rows = []
        for source in np.unique(sessions):
            mask = sessions == source
            source_path = path.parents[2] / source / 'processed/eeg_windows.npz'
            sx, sa, sf, sm = load_sequence_npz(source_path)
            lookup = {float(t): i for i, t in enumerate(sm['window_start'])}
            try:
                selected = np.array([lookup[float(t)] for t in starts[mask]])
            except KeyError as exc:
                raise ValueError(f'{subject}/{source}: source timestamp mismatch') from exc
            if not (np.array_equal(X[mask], sx[selected]) and
                    np.array_equal(a[mask], sa[selected]) and
                    np.array_equal(f[mask], sf[selected])):
                raise ValueError(f'{subject}/{source}: combined data differs from source')
            source_rows.append(dict(session=source, windows=int(mask.sum()),
                source_dataset_sha256=sha(source_path),
                excluded_source_windows=len(sa) - len(selected),
                source_content_match=True,
                action_counts=np.bincount(a[mask], minlength=3).tolist(),
                finger_counts=np.bincount(f[mask], minlength=6).tolist()))
        rows.append(dict(subject=subject, dataset=str(path.relative_to(ROOT)),
            sha256=sha(path), windows=n, channels=labels, fs=256,
            sources=source_rows, outer_train_n=len(plan['train_idx']),
            outer_test_n=len(plan['test_idx']), run=f'{session}/processed/models/{RUN}',
            scope='retrospective within-subject event holdout; not unseen-session validation'))
    save('datasets.json', rows)
    import platform
    import sklearn
    import torch
    save('protocol.json', dict(
        date='2026-11-25', architecture='CNNLSTMFingerActionNet', seed=43,
        epochs=60, batch_size=64, learning_rate=.001,
        test_size=.2, calibration_size=.1, outer_purge_seconds=.25,
        calibration_purge='maximum observed window duration',
        rest_weight=1., action_loss_weight=1.,
        data_scope='separate within-subject models; no pooled-subject training',
        selection='fixed recipe; final epoch; no test-driven model selection',
        limitations=['single seed', 'retrospective previously inspected data',
            'historically pruned 2-M16 REST events retained as exclusions',
            '3-M16 finger/session confounding', 'no new live validation'],
        environment=dict(python=platform.python_version(), numpy=np.__version__,
            sklearn=sklearn.__version__, torch=torch.__version__, device='cpu'),
        source_sha256={name: sha(ROOT / name) for name in [
            '2_train_model.py', 'models/cnn_lstm_finger_action_net.py',
            'utils/default_recipe.py', 'utils/runtime_utils.py', 'utils/splitting.py']}))
    return rows


def train(rows):
    protocol = json.loads((REPORTS / 'protocol.json').read_text())
    for name, expected in protocol['source_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f'Training source changed since audit: {name}')
    logs = ROOT / 'logs/finalization_20261125'
    logs.mkdir(parents=True, exist_ok=True)
    for row in rows:
        run = ROOT / row['run']
        if run.exists():
            raise FileExistsError(f'Refusing to overwrite {run}; use summarize for completed runs.')
        cmd = [sys.executable, str(ROOT / '2_train_model.py'), '--session-dir',
            str(ROOT / SESSIONS[row['subject']]), '--run-dir', str(run),
            '--epochs', '60', '--seed', '43', '--device', 'cpu',
            '--test-size', '.2', '--calibration-size', '.1', '--purge-seconds', '.25']
        print(f'Training {row["subject"]}: {run}', flush=True)
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                   VECLIB_MAXIMUM_THREADS='1', MPLBACKEND='Agg', PYTHONUNBUFFERED='1')
        with (logs / f'{row["subject"]}.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)


def summarize(rows):
    import torch
    from utils.live_infer_common import load_model_artifacts, predict_window
    torch.set_num_threads(1)
    results = []
    for row in rows:
        run = ROOT / row['run']
        X, a, f, meta = load_sequence_npz(ROOT / row['dataset'])
        with np.load(run / 'split_indices.npz') as z:
            folds = {k: z[k] for k in ['fit', 'calibration', 'test']}
        sessions = np.asarray(meta['session_id']).astype(str)
        groups = np.array([f'{sid}:{gid}' for sid, gid in zip(sessions, infer_groups(meta, len(a)))])
        start, end = np.asarray(meta['window_start']), np.asarray(meta['window_end'])
        for left, right in [('fit', 'calibration'), ('fit', 'test'), ('calibration', 'test')]:
            l, r = folds[left], folds[right]
            if np.intersect1d(groups[l], groups[r]).size:
                raise ValueError(f'{row["subject"]}: {left}/{right} group overlap')
            for i in r:
                same = l[sessions[l] == sessions[i]]
                if np.any((start[same] < end[i] - 1e-8) & (end[same] > start[i] + 1e-8)):
                    raise ValueError(f'{row["subject"]}: {left}/{right} signal overlap')
        with np.load(run / 'test_predictions.npz') as z:
            info = json.loads(str(z['dataset_info'].reshape(-1)[0]))
            if info['npz_sha256'] != row['sha256']:
                raise ValueError(f'{row["subject"]}: prediction cache belongs to a different dataset')
            indices = z['test_indices_global'] if 'test_indices_global' in z else z['test_indices']
            if not np.array_equal(indices, folds['test']):
                raise ValueError('Cached predictions do not match saved test indices')
            if not np.array_equal(z['y_action'], a[indices]) or not np.array_equal(z['y_finger'], f[indices]):
                raise ValueError('Cached labels do not match source dataset')
            pa = z['action_probs'].argmax(1)
            pf = z['finger_probs'].argmax(1) + 1
            pf[pa == 0] = 0
            model, scaler, temperature = load_model_artifacts(
                run_dir=run, device=torch.device('cpu'), n_channels=4)
            max_error = 0.
            for j in np.linspace(0, len(indices) - 1, 32, dtype=int):
                prediction = predict_window(X[indices[j]], scaler=scaler, model=model,
                    device=torch.device('cpu'), inference_engine=None, temperature_state=temperature)
                for head in ['action_probs', 'finger_probs']:
                    error = float(np.max(np.abs(prediction[head] - z[head][j])))
                    max_error = max(max_error, error)
                    if error > 1e-5:
                        raise ValueError(f'{row["subject"]}: saved-model/cache mismatch')
        ya, yf = a[indices], f[indices]
        active = ya != 0
        event_correct = []
        for group in np.unique(groups[indices]):
            mask = groups[indices] == group
            true = ya[mask] * 6 + yf[mask]
            pred = pa[mask] * 6 + pf[mask]
            event_correct.append(np.bincount(true, minlength=18).argmax() == np.bincount(pred, minlength=18).argmax())
        results.append(dict(subject=row['subject'], dataset_sha256=row['sha256'],
            run=row['run'], fold_counts={k: len(v) for k, v in folds.items()},
            cached_prediction_check=dict(windows=32, max_absolute_error=max_error),
            test_action_counts=np.bincount(ya, minlength=3).tolist(),
            test_event_count=len(event_correct),
            test_rest_event_count=len(np.unique(groups[indices][~active])),
            fold_audit='no shared event groups or overlapping signal intervals',
            action_accuracy=float(np.mean(pa == ya)),
            active_finger_accuracy=float(np.mean(pf[active] == yf[active])),
            joint_accuracy=float(np.mean((pa == ya) & (pf == yf))),
            rest_recall=float(np.mean(pa[~active] == 0)),
            event_majority_joint_accuracy=float(np.mean(event_correct)),
            per_finger_recall={str(k): float(np.mean(pf[yf == k] == k)) for k in range(1, 6)},
            hashes={name: sha(run / name) for name in ['finger_action_model.pt', 'scaler.npz',
                'temperature_scaling.json', 'train_config.json', 'split_indices.npz', 'test_predictions.npz']}))
    save('comparison.json', dict(protocol='CNN-LSTM; seed 43; 60 epochs; no smoothing; event holdout; 250 ms purge; independent calibration',
        deployment_status='offline research only; March 19 deployment reference unchanged', subjects=results))
    print(json.dumps(results, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['audit', 'train', 'summarize'])
    args = parser.parse_args()
    if args.stage == 'audit':
        print(json.dumps(audit(), indent=2))
        return
    rows = json.loads((REPORTS / 'datasets.json').read_text())
    for row in rows:
        if sha(ROOT / row['dataset']) != row['sha256']:
            raise ValueError('Dataset changed since audit; review and audit again.')
    (train if args.stage == 'train' else summarize)(rows)


if __name__ == '__main__':
    main()
