from __future__ import annotations

from math import ceil
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.signal import welch


# Approximate 2D scalp coordinates for Muse 2 electrodes in a top-down view.
# These are sufficient for qualitative interpolation and intentionally keep the
# layout simple and dependency-free.
MUSE_2D_POSITIONS = {
    "AF7": np.array([-0.42, 0.78], dtype=np.float32),
    "AF8": np.array([0.42, 0.78], dtype=np.float32),
    "TP9": np.array([-0.86, -0.18], dtype=np.float32),
    "TP10": np.array([0.86, -0.18], dtype=np.float32),
}


def _channel_names_array(channel_names: Sequence[str] | np.ndarray) -> np.ndarray:
    names = np.asarray(channel_names).astype("U").reshape(-1)
    if names.size == 0:
        raise ValueError("channel_names is empty")
    return names


def muse_positions_for_channels(
    channel_names: Sequence[str] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    names = _channel_names_array(channel_names)
    positions = []
    kept_names = []
    for name in names:
        if name in MUSE_2D_POSITIONS:
            positions.append(MUSE_2D_POSITIONS[name])
            kept_names.append(name)
    if not positions:
        known = ", ".join(sorted(MUSE_2D_POSITIONS))
        raise ValueError(
            f"No Muse 2 electrode positions matched channel_names={names.tolist()}. "
            f"Known channels: {known}"
        )
    return np.vstack(positions).astype(np.float32), np.asarray(kept_names).astype("U")


def coerce_windows_ntc(X: np.ndarray, channel_count: int) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D windows array, got shape {arr.shape}")
    if arr.shape[-1] == channel_count:
        return arr
    if arr.shape[1] == channel_count:
        return np.transpose(arr, (0, 2, 1))
    raise ValueError(
        f"Could not infer channel axis for X with shape {arr.shape} and "
        f"channel_count={channel_count}"
    )


def compute_bandpower_windows(
    X: np.ndarray,
    fs: float,
    band: tuple[float, float] = (8.0, 12.0),
    *,
    channel_count: int,
) -> np.ndarray:
    if fs <= 0:
        raise ValueError(f"fs must be positive, got {fs}")
    low, high = band
    if not (0.0 <= low < high):
        raise ValueError(f"Invalid band {band}")

    X_ntc = coerce_windows_ntc(X, channel_count=channel_count)
    nperseg = min(int(round(fs)), X_ntc.shape[1])
    if nperseg < 8:
        raise ValueError(
            f"Window length too short for Welch bandpower estimate: {X_ntc.shape[1]} samples"
        )

    freqs, psd = welch(X_ntc, fs=fs, axis=1, nperseg=nperseg)
    band_mask = (freqs >= low) & (freqs <= high)
    if not np.any(band_mask):
        raise ValueError(
            f"Band {band} Hz not represented in Welch frequencies for fs={fs}"
        )
    return np.trapz(psd[:, band_mask, :], freqs[band_mask], axis=1).astype(np.float32)


def mean_bandpower_map(
    bandpower_windows: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    arr = np.asarray(bandpower_windows, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected bandpower_windows shape (N,C), got {arr.shape}")
    if mask is None:
        selected = arr
    else:
        mask_arr = np.asarray(mask)
        if mask_arr.dtype == bool:
            selected = arr[mask_arr]
        else:
            selected = arr[mask_arr.astype(np.int64)]
    if selected.size == 0:
        raise ValueError("Selection produced no windows for topomap")
    return selected.mean(axis=0).astype(np.float32)


def interpolate_muse_topomap(
    values: Sequence[float] | np.ndarray,
    channel_names: Sequence[str] | np.ndarray,
    *,
    grid_res: int = 220,
    smoothing_sigma: float = 1.2,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray, np.ndarray, np.ndarray]:
    values_arr = np.asarray(values, dtype=np.float32).reshape(-1)
    positions, kept_names = muse_positions_for_channels(channel_names)

    if values_arr.size != _channel_names_array(channel_names).size:
        raise ValueError(
            f"values length {values_arr.size} must match channel_names length "
            f"{_channel_names_array(channel_names).size}"
        )
    value_lookup = {
        name: float(values_arr[idx])
        for idx, name in enumerate(_channel_names_array(channel_names))
        if name in set(kept_names.tolist())
    }
    kept_values = np.asarray([value_lookup[name] for name in kept_names], dtype=np.float32)

    grid_x, grid_y = np.mgrid[-1.0:1.0:complex(grid_res), -1.0:1.0:complex(grid_res)]
    points = positions.astype(np.float32)
    zi_linear = griddata(points, kept_values, (grid_x, grid_y), method="linear")
    zi_nearest = griddata(points, kept_values, (grid_x, grid_y), method="nearest")
    zi = np.where(np.isnan(zi_linear), zi_nearest, zi_linear)
    if smoothing_sigma > 0.0:
        zi = gaussian_filter(zi.astype(np.float32), sigma=float(smoothing_sigma))

    scalp_mask = (grid_x ** 2 + grid_y ** 2) <= 1.0
    zi = np.ma.masked_where(~scalp_mask, zi)
    return grid_x, grid_y, zi, positions, kept_values


def plot_muse_topomap(
    values: Sequence[float] | np.ndarray,
    channel_names: Sequence[str] | np.ndarray,
    *,
    ax=None,
    title: str | None = None,
    cmap: str = "turbo",
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar: bool = False,
    blur_sigma: float = 1.2,
):
    grid_x, grid_y, zi, positions, kept_values = interpolate_muse_topomap(
        values,
        channel_names,
        smoothing_sigma=float(blur_sigma),
    )
    if ax is None:
        _, ax = plt.subplots(figsize=(4.2, 4.4))
    contour = ax.contourf(
        grid_x,
        grid_y,
        zi,
        levels=128,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.add_patch(Circle((0.0, 0.0), 1.0, fill=False, edgecolor="#263238", linewidth=1.3))
    ax.scatter(
        positions[:, 0],
        positions[:, 1],
        s=28,
        c="#ffd54f",
        edgecolors="#6d4c41",
        linewidths=0.9,
        zorder=5,
    )
    for idx, label in enumerate(muse_positions_for_channels(channel_names)[1]):
        ax.text(
            positions[idx, 0],
            positions[idx, 1] + 0.08,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f2937",
        )
    ax.set_xlim(-1.06, 1.06)
    ax.set_ylim(-1.08, 1.06)
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=11)
    cbar = None
    if colorbar:
        cbar = plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)
    return contour, cbar


def plot_muse_topomap_grid(
    maps: Sequence[tuple[str, np.ndarray]],
    channel_names: Sequence[str] | np.ndarray,
    *,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    cmap: str = "turbo",
    suptitle: str | None = None,
    blur_sigma: float = 1.2,
    vmin: float | None = None,
    vmax: float | None = None,
    colorbar_label: str = "Band Power",
):
    if not maps:
        raise ValueError("maps must contain at least one panel")

    n_panels = len(maps)
    ncols = max(1, min(int(ncols), n_panels))
    nrows = int(ceil(n_panels / ncols))
    if figsize is None:
        figsize = (4.2 * ncols, 4.2 * nrows)

    stack = np.stack([np.asarray(values, dtype=np.float32) for _, values in maps], axis=0)
    if vmin is None:
        vmin = float(np.nanmin(stack))
    if vmax is None:
        vmax = float(np.nanmax(stack))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.reshape(-1)
    last_contour = None
    for idx, (title, values) in enumerate(maps):
        last_contour, _ = plot_muse_topomap(
            values,
            channel_names,
            ax=flat_axes[idx],
            title=title,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            blur_sigma=blur_sigma,
        )
    for idx in range(n_panels, flat_axes.size):
        flat_axes[idx].axis("off")
    if last_contour is not None:
        colorbar = fig.colorbar(
            last_contour,
            ax=axes,
            fraction=0.025,
            pad=0.02,
            shrink=0.92,
        )
        if colorbar_label:
            colorbar.ax.set_ylabel(colorbar_label, rotation=90)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13)
    return fig


def compute_map_limits(
    maps: Sequence[tuple[str, np.ndarray]],
    *,
    robust_quantile: float = 0.02,
    center_zero: bool = False,
) -> tuple[float, float]:
    if not maps:
        raise ValueError("maps must contain at least one panel")
    stack = np.stack([np.asarray(values, dtype=np.float32) for _, values in maps], axis=0)
    flat = stack.reshape(-1)
    if robust_quantile > 0.0:
        low = float(np.nanquantile(flat, robust_quantile))
        high = float(np.nanquantile(flat, 1.0 - robust_quantile))
    else:
        low = float(np.nanmin(flat))
        high = float(np.nanmax(flat))
    if center_zero:
        limit = max(abs(low), abs(high))
        return -limit, limit
    return low, high


def split_indices_in_halves(indices: Iterable[int]) -> tuple[np.ndarray, np.ndarray]:
    idx = np.asarray(list(indices), dtype=np.int64)
    if idx.size == 0:
        return idx, idx
    split = max(1, idx.size // 2)
    first = idx[:split]
    second = idx[split:]
    if second.size == 0:
        second = first
    return first, second
