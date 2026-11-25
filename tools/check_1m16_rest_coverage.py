#!/usr/bin/env python3
"""Audit the 1-M16 REST regression and fixed-fold weight sensitivity.

This reads completed runs; it never selects a model using test scores.
"""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.sequence_data import load_sequence_npz
from utils.calibration_split import split_coverage

SESSION = ROOT / 'Projects/1-M16/subjects/1-M16 500/sessions/combined_20260927_full'
RUNS = {
    'initial_missing_core_rest': '20261125_cnn_lstm_baseline_seed43',
    'coverage_fixed_rest100': '20261125_cnn_lstm_baseline_seed43_coverage',
    'coverage_fixed_rest200': '20261125_cnn_lstm_coverage_rest200_seed43',
}


def main():
    _, actions, fingers, meta = load_sequence_npz(SESSION / 'processed/eeg_windows.npz')
    results = []
    folds = {}
    for name, run_id in RUNS.items():
        run = SESSION / 'processed/models' / run_id
        with np.load(run / 'split_indices.npz') as z:
            folds[name] = {k: z[k] for k in ['fit', 'calibration', 'test']}
        with np.load(run / 'test_predictions.npz') as z:
            indices = z['test_indices_global']
            assert np.array_equal(indices, folds[name]['test'])
            assert np.array_equal(z['y_action'], actions[indices])
            assert np.array_equal(z['y_finger'], fingers[indices])
            pa = z['action_probs'].argmax(1)
            pf = z['finger_probs'].argmax(1) + 1
            pf[pa == 0] = 0
        ya, yf = actions[indices], fingers[indices]
        rest = ya == 0
        events = np.asarray(meta['event_id'])[indices]
        correct_rest = int(np.sum((pa == 0) & rest))
        results.append(dict(variant=name, run=str(run.relative_to(ROOT)),
            rest_weight=json.loads((run / 'train_config.json').read_text())['rest_weight'],
            action_accuracy=float(np.mean(pa == ya)),
            active_finger_accuracy=float(np.mean(pf[~rest] == yf[~rest])),
            joint_accuracy=float(np.mean((pa == ya) & (pf == yf))),
            rest_recall=float(correct_rest / rest.sum()),
            rest_correct=correct_rest, rest_total=int(rest.sum()),
            movement_recall=float(np.mean(pa[~rest] != 0)),
            rest_events={str(e): dict(windows=int(np.sum(rest & (events == e))),
                correct=int(np.sum(rest & (events == e) & (pa == 0)))) for e in np.unique(events[rest])},
            folds=split_coverage(meta, actions, **folds[name]),
            temperatures=json.loads((run / 'temperature_scaling.json').read_text()),
            hashes={file: hashlib.sha256((run / file).read_bytes()).hexdigest()
                for file in ['finger_action_model.pt', 'scaler.npz', 'split_indices.npz', 'test_predictions.npz']}))
    for fold in ['fit', 'calibration', 'test']:
        assert np.array_equal(folds['coverage_fixed_rest100'][fold], folds['coverage_fixed_rest200'][fold])
    assert np.array_equal(folds['initial_missing_core_rest']['test'], folds['coverage_fixed_rest100']['test'])
    payload = dict(scope='retrospective diagnostic; weights are not selected on these test scores',
        dataset_windows=len(actions), dataset_rest_windows=int(np.sum(actions == 0)),
        identical_corrected_folds=True, unchanged_outer_test=True,
        published_reference=dict(run='20260305_052744', rest_correct=105, rest_total=114,
            rest_recall=105/114, rest_weight=1., seed=45,
            source='https://alphahand.org/reports/1-m16-500/report'), results=results)
    out = ROOT / 'docs/research/finalization_20261125/rest_coverage_diagnosis.json'
    out.write_text(json.dumps(payload, indent=2) + '\n')
    for row in results:
        print(row['variant'], {key: row[key] for key in ['rest_recall', 'movement_recall', 'joint_accuracy']})


if __name__ == '__main__':
    main()
