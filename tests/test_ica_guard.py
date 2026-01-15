import numpy as np
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from utils.ica_guard import guard_ica_fit


def test_guard_ica_flatline_skips():
    X = np.zeros((200, 4))
    scaler = StandardScaler()
    ica = FastICA(n_components=4, random_state=42)
    result = guard_ica_fit(
        X,
        scaler=scaler,
        ica=ica,
        min_samples=50,
        min_var=1e-8,
    )
    assert not result.ok
    assert result.reason == "low_variance"


def test_guard_ica_nonfinite_skips():
    X = np.random.randn(200, 4)
    X[0, 0] = np.nan
    scaler = StandardScaler()
    ica = FastICA(n_components=4, random_state=42)
    result = guard_ica_fit(
        X,
        scaler=scaler,
        ica=ica,
        min_samples=50,
        min_var=1e-8,
    )
    assert not result.ok
    assert result.reason == "nonfinite_input"


def test_guard_ica_too_few_samples_skips():
    X = np.random.randn(10, 4)
    scaler = StandardScaler()
    ica = FastICA(n_components=4, random_state=42)
    result = guard_ica_fit(
        X,
        scaler=scaler,
        ica=ica,
        min_samples=50,
        min_var=1e-8,
    )
    assert not result.ok
    assert result.reason == "insufficient_samples"


def test_guard_ica_valid_runs():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((400, 4))
    scaler = StandardScaler()
    ica = FastICA(n_components=4, random_state=42)
    result = guard_ica_fit(
        X,
        scaler=scaler,
        ica=ica,
        min_samples=100,
        min_var=1e-8,
    )
    assert result.ok
    assert result.scaled is not None
    assert hasattr(ica, "components_")
