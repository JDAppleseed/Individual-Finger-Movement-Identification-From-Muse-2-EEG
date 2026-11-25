#!/usr/bin/env python3
"""Reproduce every November holdout prediction and its fitting-only scaler."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.finalize_cnn_baselines import sha
from utils.sequence_data import load_sequence_npz, fit_channel_normalizer
from utils.live_infer_common import load_model_artifacts, predict_window
from utils.saved_folds import load_saved_folds, assert_signal_disjoint


def main():
    torch.set_num_threads(1)
    rows = json.loads((ROOT / 'docs/research/finalization_20261125/datasets.json').read_text())
    results = []
    for row in rows:
        path, run = ROOT / row['dataset'], ROOT / row['run']
        X, a, f, meta = load_sequence_npz(path)
        assert sha(path) == row['sha256']
        folds = load_saved_folds(run, path, np.arange(len(a)), len(a))
        assert_signal_disjoint(meta, **{key: folds[key] for key in ('fit', 'calibration', 'test')})
        model, scaler, temperature = load_model_artifacts(run_dir=run, device=torch.device('cpu'), n_channels=4)
        recreated = fit_channel_normalizer(X[folds['fit']], preprocess=scaler.get('preprocess'))
        scaler_error = {key: float(np.max(np.abs(recreated[key] - scaler[key]))) for key in ('mean', 'std')}
        assert max(scaler_error.values()) == 0, scaler_error
        errors = {'action_probs': 0., 'finger_probs': 0.}
        applicability_error = 0.
        with np.load(run / 'test_predictions.npz') as z:
            indices = folds['test']
            assert np.array_equal(a[indices], z['y_action'])
            assert np.array_equal(f[indices], z['y_finger'])
            for j, index in enumerate(indices):
                pred = predict_window(X[index], scaler=scaler, model=model, device=torch.device('cpu'),
                                      inference_engine=None, temperature_state=temperature)
                for head in errors:
                    error = float(np.max(np.abs(pred[head] - z[head][j])))
                    assert np.isfinite(error) and error < 1e-5, (row['subject'], j, head, error)
                    errors[head] = max(errors[head], error)
                    assert np.argmax(pred[head]) == np.argmax(z[head][j]), (row['subject'], j, head)
                app_error = abs(pred['finger_applicable_prob'] - float(z['applicability_probs'][j]))
                assert np.isfinite(app_error) and app_error < 1e-5
                applicability_error = max(applicability_error, app_error)
            pa = z['action_probs'].argmax(1)
            pf = z['finger_probs'].argmax(1) + 1
            pf[pa == 0] = 0
            result = dict(subject=row['subject'], dataset_sha256=sha(path),
                          checked_holdout_windows=len(indices), scaler_max_absolute_error=scaler_error,
                          prediction_max_absolute_error=errors,
                          applicability_max_absolute_error=applicability_error,
                          all_argmax_predictions_match=True,
                          fit_calibration_test_signal_disjoint=True,
                          action_accuracy=float(np.mean(pa == a[indices])),
                          joint_accuracy=float(np.mean((pa == a[indices]) & (pf == f[indices]))),
                          rest_recall=float(np.mean(pa[a[indices] == 0] == 0)))
        results.append(result)
        print(json.dumps(result), flush=True)
    target = ROOT / 'docs/research/finalization_20261125/full_prediction_audit.json'
    target.write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
