"""Resolve the original training folds without re-splitting a fitted model."""
from pathlib import Path
import hashlib
import json
import numpy as np


def _indices(values, n):
    a = np.asarray(values)
    if a.ndim != 1 or a.dtype.kind not in 'iu':
        raise ValueError('Saved fold indices must be one-dimensional integers')
    if len(np.unique(a)) != len(a) or np.any(a < 0) or np.any(a >= n):
        raise ValueError('Saved fold indices are duplicated or outside the source dataset')
    return a.astype(np.int64)


def load_saved_folds(run_dir, dataset_path, source_indices, source_n):
    """Return folds in current-array coordinates, verified against source SHA256.

    Legacy local folds are accepted only with a matching prediction manifest and
    its explicit local-to-global test mapping. Unknown fitting membership is
    never inferred from the complement of the test fold.
    """
    run_dir = Path(run_dir)
    digest = hashlib.sha256(Path(dataset_path).read_bytes()).hexdigest()
    source_indices = _indices(source_indices, source_n)
    folds_path = run_dir / 'split_indices.npz'
    cache_path = run_dir / 'test_predictions.npz'
    folds = {}
    info = None
    cache_test = None
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if 'dataset_info' in z:
                info = json.loads(str(z['dataset_info'].reshape(-1)[0]))
            if info and info.get('npz_sha256') == digest:
                key = 'test_indices_global' if 'test_indices_global' in z else 'test_indices'
                if key in z:
                    if key != 'test_indices_global' and info.get('n_samples') != source_n:
                        raise ValueError('Legacy test coordinates cannot be established')
                    cache_test = _indices(z[key], source_n)
                    with np.load(dataset_path, allow_pickle=False) as source:
                        if (not np.array_equal(z['y_action'], source['y_action'][cache_test])
                                or not np.array_equal(z['y_finger'], source['y_finger'][cache_test])):
                            raise ValueError('Cached holdout labels disagree with the source dataset')
    if folds_path.exists():
        with np.load(folds_path, allow_pickle=False) as z:
            if 'index_space' in z and str(z['index_space'].item()) == 'source_npz':
                if str(z['dataset_sha256'].item()) != digest:
                    raise ValueError('Saved folds belong to a different dataset')
            else:
                # Old files saved local indices. Only the unfiltered identity
                # mapping is established by this legacy manifest.
                filters = (info or {}).get('filters', {})
                if (not info or info.get('npz_sha256') != digest
                        or info.get('n_samples') != source_n
                        or filters.get('max_samples') is not None):
                    raise ValueError('Legacy fold coordinates cannot be established')
            folds = {key: _indices(z[key], source_n)
                     for key in ('train', 'fit', 'calibration', 'test') if key in z}
    elif cache_test is not None:
        folds = {'test': cache_test}
    else:
        raise ValueError('Original holdout unavailable; refusing to re-split a trained model')
    if 'test' not in folds or not len(folds['test']):
        raise ValueError('Saved test fold is empty or missing')
    if cache_test is not None and not np.array_equal(folds['test'], cache_test):
        raise ValueError('Saved folds and prediction-cache holdout disagree')
    for left, right in [('fit', 'calibration'), ('fit', 'test'), ('calibration', 'test'), ('train', 'test')]:
        if left in folds and right in folds and np.intersect1d(folds[left], folds[right]).size:
            raise ValueError(f'Saved {left}/{right} folds overlap')
    if 'train' in folds:
        for key in ('fit', 'calibration'):
            if key in folds and not np.isin(folds[key], folds['train']).all():
                raise ValueError(f'Saved {key} is outside the outer training fold')
    lookup = np.full(source_n, -1, dtype=np.int64)
    lookup[source_indices] = np.arange(len(source_indices))
    mapped = {key: lookup[value] for key, value in folds.items()}
    # A subset is allowed, but retains only original fold members.
    mapped = {key: value[value >= 0] for key, value in mapped.items()}
    if not len(mapped['test']):
        raise ValueError('Current filters contain no original holdout rows')
    return mapped


def assert_signal_disjoint(meta, **folds):
    """Fail closed if completed fitting/calibration/test folds share raw signal."""
    if 'window_start' not in meta or 'window_end' not in meta:
        raise ValueError('Signal-overlap audit requires window_start and window_end')
    starts = np.asarray(meta['window_start'], dtype=float)
    ends = np.asarray(meta['window_end'], dtype=float)
    if (starts.ndim != 1 or ends.shape != starts.shape or not np.isfinite(starts).all()
            or not np.isfinite(ends).all() or np.any(ends <= starts)):
        raise ValueError('Invalid window intervals')
    sessions = np.asarray(meta.get('session_id', np.full(len(starts), 'GLOBAL'))).astype(str)
    names = list(folds)
    for j, left in enumerate(names):
        for right in names[j + 1:]:
            a, b = np.asarray(folds[left], dtype=int), np.asarray(folds[right], dtype=int)
            for sid in np.unique(sessions[b]):
                rows = b[sessions[b] == sid]
                order = np.argsort(starts[rows])
                boundaries = starts[rows][order]
                max_ends = np.maximum.accumulate(ends[rows][order])
                other = a[sessions[a] == sid]
                pos = np.searchsorted(boundaries, ends[other], side='left') - 1
                if np.any((pos >= 0) & (max_ends[np.maximum(pos, 0)] > starts[other])):
                    raise ValueError(f'{left}/{right} share raw signal; revise split/calibration before fitting')
