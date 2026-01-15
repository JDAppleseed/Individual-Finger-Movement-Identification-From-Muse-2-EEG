from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


@dataclass
class IcaGuardResult:
    ok: bool
    reason: Optional[str]
    diagnostics: Dict[str, Any]
    scaled: Optional[np.ndarray]


def _build_diagnostics(
    X: np.ndarray,
    variances: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    finite_mask = np.isfinite(X)
    nonfinite_counts = np.sum(~finite_mask, axis=0).tolist()
    total = X.size if X.size else 0
    nonfinite_fraction = (
        float(np.sum(~finite_mask)) / float(total) if total else 0.0
    )
    diag: Dict[str, Any] = {
        "shape": list(X.shape),
        "nonfinite_fraction": nonfinite_fraction,
        "nonfinite_counts": nonfinite_counts,
    }
    if variances is not None:
        diag["variances"] = variances.tolist()
    return diag


def validate_ica_input(
    X: np.ndarray,
    *,
    min_samples: Optional[int] = None,
    min_var: float = 1e-8,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if X.ndim != 2:
        return "invalid_shape", {"shape": list(X.shape)}
    if min_samples is not None and X.shape[0] < min_samples:
        return "insufficient_samples", {"shape": list(X.shape)}
    if not np.isfinite(X).all():
        return "nonfinite_input", _build_diagnostics(X)
    variances = np.var(X, axis=0)
    if np.any(variances < min_var):
        return "low_variance", _build_diagnostics(X, variances=variances)
    return None, _build_diagnostics(X, variances=variances)


def safe_standardize(
    X: np.ndarray, scaler: StandardScaler, *, min_var: float
) -> Tuple[np.ndarray, np.ndarray]:
    scaler.fit(X)
    scale = np.asarray(scaler.scale_, dtype=np.float64)
    safe_scale = np.where(scale < min_var, 1.0, scale)
    scaler.scale_ = safe_scale
    scaler.var_ = safe_scale**2
    scaled = scaler.transform(X)
    return scaled, safe_scale


def guard_ica_fit(
    X: np.ndarray,
    *,
    scaler: StandardScaler,
    ica: Any,
    min_samples: int,
    min_var: float,
) -> IcaGuardResult:
    reason, diagnostics = validate_ica_input(
        X, min_samples=min_samples, min_var=min_var
    )
    if reason is not None:
        return IcaGuardResult(False, reason, diagnostics, None)

    scaled, safe_scale = safe_standardize(X, scaler, min_var=min_var)
    if not np.isfinite(scaled).all():
        diagnostics["scale"] = safe_scale.tolist()
        diagnostics["post_scale_nonfinite"] = True
        return IcaGuardResult(False, "nonfinite_scaled", diagnostics, None)

    ica.fit(scaled)
    diagnostics["scale"] = safe_scale.tolist()
    return IcaGuardResult(True, None, diagnostics, scaled)
