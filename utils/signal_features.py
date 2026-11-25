"""Deterministic signal-only descriptors for offline EEG classifier experiments."""
from __future__ import annotations
import numpy as np
from utils.runtime_utils import preprocess_eeg_windows


def extract_signal_features(X, *, fs=256., mode='center_detrend'):
    if mode not in {'center', 'center_detrend'}:
        raise ValueError('mode must be center or center_detrend')
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3 or X.shape[1] < 8 or not np.isfinite(X).all():
        raise ValueError('Expected finite [windows, time, channels] EEG')
    z = preprocess_eeg_windows(X, {'per_window_center': True, 'per_window_detrend': mode == 'center_detrend'})
    std = z.std(axis=1) + 1e-8
    d1 = np.diff(z, axis=1)
    d2 = np.diff(d1, axis=1)
    d1_std = d1.std(axis=1) + 1e-8
    norm = z / std[:, None, :]
    chunks = [np.log(std), np.log(np.mean(np.abs(z), axis=1)+1e-8),
              d1_std/std, d2.std(axis=1)/d1_std,
              np.mean(norm**3, axis=1), np.mean(norm**4, axis=1),
              np.mean(z[:, 1:, :]*z[:, :-1, :], axis=1)/(std**2),
              np.mean(np.diff(np.signbit(z), axis=1), axis=1)]
    quantiles = np.quantile(z, [.05, .25, .75, .95], axis=1)
    chunks.extend(q/std for q in quantiles)
    taper = np.hanning(z.shape[1]).astype(np.float32)
    power = np.abs(np.fft.rfft(z*taper[None, :, None], axis=1))**2
    freqs = np.fft.rfftfreq(z.shape[1], 1/float(fs))
    for lo, hi in [(1, 8), (8, 13), (13, 30), (30, 50), (50, 90), (90, 129)]:
        band = power[:, (freqs >= lo) & (freqs < hi), :].mean(axis=1)
        chunks.append(np.log(band+1e-8))
    # Cross-channel covariance and scale ratios; never use session/time/labels.
    for i in range(z.shape[2]):
        for j in range(i):
            chunks.append(np.mean(norm[:, :, i]*norm[:, :, j], axis=1)[:, None])
            chunks.append(np.log(std[:, i]/std[:, j])[:, None])
    # Coarse within-window morphology; retains temporal structure for the SVM.
    chunks.append(norm[:, np.linspace(0,z.shape[1]-1,16).astype(int), :].reshape(len(z), -1))
    result = np.concatenate(chunks, axis=1).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite signal features')
    return result


def joint_probabilities_to_heads(probabilities, classes):
    p = np.zeros((len(probabilities),11), dtype=np.float64)
    p[:,np.asarray(classes,dtype=int)] = probabilities
    action = np.column_stack([p[:,0],p[:,1:6].sum(axis=1),p[:,6:11].sum(axis=1)])
    finger = p[:,1:6]+p[:,6:11]
    mass = finger.sum(axis=1,keepdims=True)
    finger = np.divide(finger,mass,out=np.full_like(finger,.2),where=mass>0)
    return action.astype(np.float32),finger.astype(np.float32)
