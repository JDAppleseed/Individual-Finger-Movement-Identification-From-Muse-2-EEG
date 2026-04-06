#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from muse_streaming.resample import resample_window, verify_alignment
from utils.live_infer_common import (
    load_model_artifacts_from_files,
    predict_window,
    sanitize_live_window,
)
from utils.runtime_utils import apply_channel_normalizer, load_normalizer

DEFAULT_CLIP_ABS_Z = 6.0
DEFAULT_BAD_CHANNEL_RMS_Z = 4.0
DEFAULT_BAD_CHANNEL_ABS_P95_Z = 6.0
DEFAULT_BAD_CHANNEL_CLIPPED_FRAC = 0.05
DEFAULT_BAD_WINDOW_CLIPPED_FRAC = 0.10
DEFAULT_BAD_WINDOW_MAX_MASKED_CHANNELS = 1
DEFAULT_SPECTRAL_BANDS = {
    "theta_4_8": (4.0, 8.0),
    "alpha_8_12": (8.0, 12.0),
    "beta_12_30": (12.0, 30.0),
    "gamma_30_45": (30.0, 45.0),
}


def _series(values: np.ndarray) -> dict[str, float | int | None]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "p50": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _load_raw_dir(raw_dir: Path) -> np.ndarray:
    shards = sorted(raw_dir.glob("eeg_raw_shard_*.npy"))
    if not shards:
        raise FileNotFoundError(f"No raw shards found under {raw_dir}")
    return np.concatenate([np.load(path, allow_pickle=False) for path in shards])


def _ensure_raw_array(raw_source: np.ndarray | Path) -> np.ndarray:
    if isinstance(raw_source, np.ndarray):
        return raw_source
    return _load_raw_dir(Path(raw_source).expanduser().resolve())


def _load_runtime_manifest(
    *,
    raw_dir: Path | None = None,
    runtime_manifest_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path | None]:
    if isinstance(runtime_manifest, dict) and runtime_manifest:
        return runtime_manifest, runtime_manifest_path
    if runtime_manifest_path is not None:
        path = Path(runtime_manifest_path).expanduser().resolve()
        if path.exists():
            return _load_json_object(path), path
    if raw_dir is not None:
        candidate = Path(raw_dir).expanduser().resolve().parent / "live_runtime_manifest.json"
        if candidate.exists():
            return _load_json_object(candidate), candidate
    return {}, None


def _validate_reorder(values: Any, n_channels: int) -> list[int] | None:
    if not isinstance(values, (list, tuple)):
        return None
    try:
        reorder = [int(item) for item in values]
    except Exception:
        return None
    if len(reorder) != n_channels:
        return None
    if sorted(reorder) != list(range(n_channels)):
        return None
    return reorder


def _default_channel_labels(prefix: str, n_channels: int) -> list[str]:
    return [f"{prefix}_{idx}" for idx in range(n_channels)]


def _resolve_reorder_metadata(
    *,
    runtime_manifest: dict[str, Any],
    n_channels: int,
) -> tuple[list[str], list[str], list[int] | None, bool, str | None, list[str]]:
    limitations: list[str] = []
    stream_resolution = (
        runtime_manifest.get("stream_resolution", {})
        if isinstance(runtime_manifest, dict)
        else {}
    )
    stream_contract = (
        runtime_manifest.get("stream_contract", {})
        if isinstance(runtime_manifest, dict)
        else {}
    )
    stream_selection = (
        runtime_manifest.get("stream_selection", {})
        if isinstance(runtime_manifest, dict)
        else {}
    )
    resolved_stream = (
        stream_contract.get("resolved", {})
        if isinstance(stream_contract.get("resolved"), dict)
        else {}
    )
    expected = (
        stream_contract.get("expected", {})
        if isinstance(stream_contract.get("expected"), dict)
        else {}
    )
    raw_labels = [
        str(label).strip()
        for label in (
            resolved_stream.get("channel_labels")
            or stream_resolution.get("channel_labels")
            or []
        )
        if str(label).strip()
    ]
    if len(raw_labels) != n_channels:
        raw_labels = _default_channel_labels("stream_ch", n_channels)
    model_labels = [
        str(label).strip()
        for label in (
            stream_selection.get("expected_channel_labels")
            or expected.get("required_labels")
            or []
        )
        if str(label).strip()
    ]
    if len(model_labels) != n_channels:
        model_labels = list(raw_labels)
    reorder = _validate_reorder(
        resolved_stream.get("channel_reorder_to_model_order"),
        n_channels,
    )
    reorder_source: str | None = None
    decisive = False
    if reorder is not None:
        reorder_source = "runtime_manifest.stream_contract.resolved.channel_reorder_to_model_order"
        decisive = True
    else:
        raw_norm = [label.lower() for label in raw_labels]
        model_norm = [label.lower() for label in model_labels]
        if raw_norm == model_norm and len(raw_norm) == n_channels:
            reorder = list(range(n_channels))
            reorder_source = "inferred_identity_from_matching_labels"
            limitations.append(
                "Runtime manifest did not record an explicit channel_reorder_to_model_order mapping; identity order was inferred from matching labels and is not treated as decisive proof."
            )
        else:
            limitations.append(
                "Runtime manifest is missing a valid channel_reorder_to_model_order mapping, so model-order distribution claims are not decisive."
            )
    return raw_labels, model_labels, reorder, decisive, reorder_source, limitations


def _compute_channel_stats(
    samples: np.ndarray,
    *,
    channel_labels: list[str],
) -> dict[str, Any]:
    if samples.ndim != 2:
        raise ValueError(f"Expected 2D samples, got shape {samples.shape}")
    channel_stats: list[dict[str, Any]] = []
    for channel_id in range(samples.shape[1]):
        values = np.asarray(samples[:, channel_id], dtype=float)
        finite = values[np.isfinite(values)]
        label = (
            str(channel_labels[channel_id])
            if channel_id < len(channel_labels)
            else f"ch_{channel_id}"
        )
        if finite.size == 0:
            channel_stats.append(
                {
                    "channel_id": int(channel_id),
                    "channel_label": label,
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "rms": None,
                    "abs_p95": None,
                }
            )
            continue
        channel_stats.append(
            {
                "channel_id": int(channel_id),
                "channel_label": label,
                "count": int(finite.size),
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "rms": float(np.sqrt(np.mean(finite**2))),
                "abs_p95": float(np.percentile(np.abs(finite), 95)),
            }
        )
    return {
        "channel_count": int(samples.shape[1]),
        "channel_labels": list(channel_labels),
        "channels": channel_stats,
    }


def _build_raw_channel_stats_payload(
    raw: np.ndarray,
    *,
    runtime_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime_manifest = runtime_manifest or {}
    samples = np.asarray(raw["sample"], dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"Expected raw sample payload with shape (N,C), got {samples.shape}")
    rows = int(samples.shape[0])
    finite_row_mask = np.all(np.isfinite(samples), axis=1)
    nonfinite_rows = int((~finite_row_mask).sum())
    nonfinite_values = int(np.size(samples) - np.isfinite(samples).sum())
    flagged_nonfinite_rows = 0
    if raw.dtype.names is not None and "flags" in raw.dtype.names:
        try:
            flags = np.asarray(raw["flags"], dtype=int)
            flagged_nonfinite_rows = int((flags & 1).astype(bool).sum())
        except Exception:
            flagged_nonfinite_rows = 0
    raw_labels, model_labels, reorder, decisive, reorder_source, limitations = (
        _resolve_reorder_metadata(
            runtime_manifest=runtime_manifest,
            n_channels=int(samples.shape[1]),
        )
    )
    raw_stream_stats = _compute_channel_stats(samples, channel_labels=raw_labels)
    if reorder is not None:
        model_samples = samples[:, np.asarray(reorder, dtype=np.int64)]
        model_order_stats = _compute_channel_stats(model_samples, channel_labels=model_labels)
    else:
        model_samples = samples
        model_order_stats = None
    return {
        "shard_count": 0,
        "rows": rows,
        "nonfinite_rows": int(nonfinite_rows),
        "nonfinite_values": int(nonfinite_values),
        "flagged_nonfinite_rows": int(flagged_nonfinite_rows),
        "channels": list(raw_stream_stats.get("channels", [])),
        "raw_stream_order": raw_stream_stats,
        "model_order": model_order_stats,
        "reorder": {
            "channel_reorder_to_model_order": list(reorder) if reorder is not None else None,
            "channel_reorder_applied": bool(
                reorder is not None and list(reorder) != list(range(len(reorder)))
            ),
            "proof_source": reorder_source,
        },
        "distribution_claim_decisive": bool(decisive),
        "limitations": limitations,
    }


def build_raw_channel_stats(
    *,
    raw_dir: Path,
    runtime_manifest_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_raw_dir = Path(raw_dir).expanduser().resolve()
    raw = _load_raw_dir(resolved_raw_dir)
    loaded_runtime_manifest, _ = _load_runtime_manifest(
        raw_dir=resolved_raw_dir,
        runtime_manifest_path=runtime_manifest_path,
        runtime_manifest=runtime_manifest,
    )
    payload = _build_raw_channel_stats_payload(
        raw,
        runtime_manifest=loaded_runtime_manifest,
    )
    payload["shard_count"] = int(len(sorted(resolved_raw_dir.glob("eeg_raw_shard_*.npy"))))
    return payload


def _spectral_proxies(
    prepared: np.ndarray,
    *,
    target_fs: float,
) -> dict[str, Any]:
    if prepared.size == 0:
        return {
            "relative_bandpower_mean": {
                band_name: [] for band_name in DEFAULT_SPECTRAL_BANDS
            }
        }
    centered = np.asarray(prepared, dtype=np.float64)
    centered = centered - np.mean(centered, axis=1, keepdims=True)
    taper = np.hanning(centered.shape[1]).reshape(1, centered.shape[1], 1)
    spectrum = np.abs(np.fft.rfft(centered * taper, axis=1)) ** 2
    freqs = np.fft.rfftfreq(centered.shape[1], d=(1.0 / float(target_fs)))
    total_mask = (freqs >= 4.0) & (freqs <= 45.0)
    if not np.any(total_mask):
        total_mask = np.ones_like(freqs, dtype=bool)
    total_power = np.sum(spectrum[:, total_mask, :], axis=1)
    total_power = np.maximum(total_power, 1e-12)
    proxies: dict[str, list[float]] = {}
    for band_name, (f_lo, f_hi) in DEFAULT_SPECTRAL_BANDS.items():
        band_mask = (freqs >= float(f_lo)) & (freqs < float(f_hi))
        if not np.any(band_mask):
            proxies[band_name] = [0.0 for _ in range(centered.shape[2])]
            continue
        band_power = np.sum(spectrum[:, band_mask, :], axis=1)
        rel_power = band_power / total_power
        proxies[band_name] = np.mean(rel_power, axis=0).astype(float).tolist()
    return {
        "relative_bandpower_mean": proxies,
    }


def _summarize_prepared_windows(
    prepared: np.ndarray,
    *,
    clip_abs_z: float,
    target_fs: float,
) -> dict[str, Any]:
    if prepared.size == 0:
        return {
            "window_count": 0,
            "per_channel_mean": [],
            "per_channel_std": [],
            "prepared_rms_mean": [],
            "prepared_abs_p95_mean": [],
            "prepared_clip_frac_mean": [],
            "prepared_total_clip_mean": None,
            "spectral_proxies": _spectral_proxies(
                np.empty((0, 1, 0), dtype=np.float32),
                target_fs=target_fs,
            ),
        }
    abs_prepared = np.abs(prepared)
    return {
        "window_count": int(prepared.shape[0]),
        "per_channel_mean": np.mean(prepared, axis=(0, 1)).astype(float).tolist(),
        "per_channel_std": np.std(prepared, axis=(0, 1)).astype(float).tolist(),
        "prepared_rms_mean": np.sqrt(np.mean(prepared**2, axis=1)).mean(axis=0).astype(float).tolist(),
        "prepared_abs_p95_mean": np.percentile(abs_prepared, 95, axis=1).mean(axis=0).astype(float).tolist(),
        "prepared_clip_frac_mean": np.mean(abs_prepared > float(clip_abs_z), axis=(0, 1)).astype(float).tolist(),
        "prepared_total_clip_mean": float(np.mean(abs_prepared > float(clip_abs_z))),
        "spectral_proxies": _spectral_proxies(prepared, target_fs=target_fs),
    }


def _offline_stats(
    npz_path: Path,
    *,
    scaler: Any,
    max_windows: int,
    clip_abs_z: float,
    target_fs: float,
) -> dict[str, Any]:
    npz = np.load(npz_path, allow_pickle=True)
    X = np.asarray(npz["X"], dtype=np.float32)
    y_action = np.asarray(npz["y_action"], dtype=np.int64)
    out: dict[str, Any] = {}
    masks = {
        "all": np.ones(len(X), dtype=bool),
        "rest": y_action == 0,
        "nonrest": y_action != 0,
    }
    for label, mask in masks.items():
        X_sel = X[mask]
        if len(X_sel) > max_windows:
            idx = np.linspace(0, len(X_sel) - 1, max_windows, dtype=int)
            X_sel = X_sel[idx]
        X_norm = apply_channel_normalizer(X_sel, scaler)
        summary = _summarize_prepared_windows(
            X_norm,
            clip_abs_z=clip_abs_z,
            target_fs=target_fs,
        )
        summary["window_count"] = int(len(X_sel))
        out[label] = summary
    return out


def _quality_thresholds_from_manifest(runtime_manifest: dict[str, Any]) -> dict[str, float | int]:
    runtime = runtime_manifest.get("runtime", {}) if isinstance(runtime_manifest, dict) else {}
    quality = runtime.get("quality_thresholds", {}) if isinstance(runtime.get("quality_thresholds"), dict) else {}
    return {
        "input_clip_abs_z": float(quality.get("input_clip_abs_z", DEFAULT_CLIP_ABS_Z)),
        "bad_channel_rms_z": float(quality.get("bad_channel_rms_z", DEFAULT_BAD_CHANNEL_RMS_Z)),
        "bad_channel_abs_p95_z": float(
            quality.get("bad_channel_abs_p95_z", DEFAULT_BAD_CHANNEL_ABS_P95_Z)
        ),
        "bad_channel_clipped_frac": float(
            quality.get("bad_channel_clipped_frac", DEFAULT_BAD_CHANNEL_CLIPPED_FRAC)
        ),
        "bad_window_clipped_frac": float(
            quality.get("bad_window_clipped_frac", DEFAULT_BAD_WINDOW_CLIPPED_FRAC)
        ),
        "bad_window_max_masked_channels": int(
            quality.get(
                "bad_window_max_masked_channels",
                DEFAULT_BAD_WINDOW_MAX_MASKED_CHANNELS,
            )
        ),
    }


def _window_stats(
    *,
    times: np.ndarray,
    values: np.ndarray,
    scaler: Any,
    window_sec: float,
    hop_sec: float,
    target_fs: float,
    strict_gap_s: float,
    max_gap_s: float,
    model_bundle: tuple[Any, Any, Any] | None,
    confidence_sample_windows: int,
    quality_thresholds: dict[str, float | int],
) -> dict[str, Any]:
    if times.size == 0 or values.size == 0:
        return {
            "candidate_count": 0,
            "accepted_count": 0,
            "accepted_rate": 0.0,
            "dropped_count": 0,
            "recovered_vs_strict_count": 0,
            "quality_bad_count": 0,
            "quality_bad_rate": None,
            "masked_window_count": 0,
            "masked_window_rate": None,
            "drop_reason_counts": {},
            "prepared_summary": _summarize_prepared_windows(
                np.empty((0, int(round(window_sec * target_fs)), values.shape[1]), dtype=np.float32),
                clip_abs_z=float(quality_thresholds["input_clip_abs_z"]),
                target_fs=target_fs,
            ),
            "sampled_joint_conf": _series(np.asarray([], dtype=float)),
        }
    latest = float(times[-1])
    next_start = 0.0
    candidate_count = 0
    accepted_count = 0
    recovered_vs_strict = 0
    quality_bad_count = 0
    masked_window_count = 0
    reasons: Counter[str] = Counter()
    prepared_windows: list[np.ndarray] = []
    sampled_joint_conf: list[float] = []

    while next_start + window_sec <= latest + 1e-12:
        candidate_count += 1
        start_s = next_start
        end_s = start_s + window_sec
        left = max(0, int(np.searchsorted(times, start_s, side="left")) - 1)
        right = min(
            int(times.size),
            int(np.searchsorted(times, end_s, side="right")) + 1,
        )
        if right - left < 2:
            reasons["insufficient_window_samples"] += 1
            next_start += hop_sec
            continue

        window_times = times[left:right]
        window_values = values[left:right]
        strict_alignment = verify_alignment(
            window_times,
            start_s=start_s,
            end_s=end_s,
            target_fs=target_fs,
            max_gap_s=strict_gap_s,
            max_edge_gap_s=strict_gap_s,
        )
        alignment = verify_alignment(
            window_times,
            start_s=start_s,
            end_s=end_s,
            target_fs=target_fs,
            max_gap_s=max_gap_s,
            max_edge_gap_s=strict_gap_s,
        )
        if not alignment.ok:
            reasons[str(alignment.reason or "alignment_fail")] += 1
            next_start += hop_sec
            continue

        accepted_count += 1
        if not strict_alignment.ok:
            recovered_vs_strict += 1

        _, window = resample_window(
            window_times,
            window_values,
            start_s=start_s,
            end_s=end_s,
            target_fs=target_fs,
        )
        quality = sanitize_live_window(
            window,
            scaler=scaler,
            enabled=True,
            input_clip_abs_z=float(quality_thresholds["input_clip_abs_z"]),
            bad_channel_rms_z=float(quality_thresholds["bad_channel_rms_z"]),
            bad_channel_abs_p95_z=float(quality_thresholds["bad_channel_abs_p95_z"]),
            bad_channel_clipped_frac=float(quality_thresholds["bad_channel_clipped_frac"]),
            bad_window_clipped_frac=float(quality_thresholds["bad_window_clipped_frac"]),
            bad_window_max_masked_channels=int(
                quality_thresholds["bad_window_max_masked_channels"]
            ),
        )
        prepared_windows.append(quality.prepared_window)
        quality_bad_count += int(bool(quality.window_quality_bad))
        masked_window_count += int(bool(quality.masked_channel_ids))

        if model_bundle is not None and len(sampled_joint_conf) < confidence_sample_windows:
            model, device, temperature_state = model_bundle
            inference = predict_window(
                window,
                scaler=scaler,
                model=model,
                device=device,
                inference_engine=None,
                direct_engine=None,
                temperature_state=temperature_state,
                prepared_window=quality.prepared_window,
            )
            action_probs = np.asarray(inference["action_probs"], dtype=float)
            finger_probs = np.asarray(inference["finger_probs"], dtype=float)
            sampled_joint_conf.append(float(min(action_probs.max(), finger_probs.max())))
        next_start += hop_sec

    prepared = (
        np.stack(prepared_windows).astype(np.float32)
        if prepared_windows
        else np.empty((0, int(round(window_sec * target_fs)), values.shape[1]), dtype=np.float32)
    )
    return {
        "candidate_count": int(candidate_count),
        "accepted_count": int(accepted_count),
        "accepted_rate": (
            float(accepted_count / candidate_count) if candidate_count > 0 else 0.0
        ),
        "dropped_count": int(candidate_count - accepted_count),
        "recovered_vs_strict_count": int(recovered_vs_strict),
        "quality_bad_count": int(quality_bad_count),
        "quality_bad_rate": (
            float(quality_bad_count / accepted_count) if accepted_count > 0 else None
        ),
        "masked_window_count": int(masked_window_count),
        "masked_window_rate": (
            float(masked_window_count / accepted_count) if accepted_count > 0 else None
        ),
        "drop_reason_counts": dict(reasons),
        "prepared_summary": _summarize_prepared_windows(
            prepared,
            clip_abs_z=float(quality_thresholds["input_clip_abs_z"]),
            target_fs=target_fs,
        ),
        "sampled_joint_conf": _series(np.asarray(sampled_joint_conf, dtype=float)),
    }


def _spectral_distance(
    live_proxies: dict[str, Any],
    offline_proxies: dict[str, Any],
) -> float | None:
    live_bands = (
        live_proxies.get("relative_bandpower_mean", {})
        if isinstance(live_proxies, dict)
        else {}
    )
    offline_bands = (
        offline_proxies.get("relative_bandpower_mean", {})
        if isinstance(offline_proxies, dict)
        else {}
    )
    diffs: list[float] = []
    for band_name, live_values in live_bands.items():
        off_values = offline_bands.get(band_name)
        if not isinstance(live_values, list) or not isinstance(off_values, list):
            continue
        if len(live_values) != len(off_values):
            continue
        diffs.extend(abs(float(a) - float(b)) for a, b in zip(live_values, off_values))
    if not diffs:
        return None
    return float(np.mean(np.asarray(diffs, dtype=float)))


def _rms_ratios(
    live_rms: list[float],
    offline_rms: list[float],
) -> list[float | None]:
    ratios: list[float | None] = []
    for live_value, offline_value in zip(live_rms, offline_rms):
        off = float(offline_value)
        if abs(off) <= 1e-12:
            ratios.append(None)
            continue
        ratios.append(float(live_value) / off)
    return ratios


def _classify_distribution_match(
    *,
    relaxed_stats: dict[str, Any],
    strict_stats: dict[str, Any],
    offline_reference: dict[str, Any],
    decisive: bool,
) -> dict[str, Any]:
    live_rms = list(relaxed_stats.get("prepared_summary", {}).get("prepared_rms_mean", []) or [])
    offline_rms = list(offline_reference.get("prepared_rms_mean", []) or [])
    rms_ratios = _rms_ratios(live_rms, offline_rms)
    finite_ratios = [float(value) for value in rms_ratios if value is not None and np.isfinite(value)]
    median_rms_ratio = float(np.median(finite_ratios)) if finite_ratios else None
    low_channels = int(sum(float(value) < 0.75 for value in finite_ratios))
    high_channels = int(sum(float(value) > 1.35 for value in finite_ratios))
    quality_bad_rate = relaxed_stats.get("quality_bad_rate")
    masked_window_rate = relaxed_stats.get("masked_window_rate")
    accepted_rate = float(relaxed_stats.get("accepted_rate", 0.0) or 0.0)
    recovered_vs_strict_count = int(relaxed_stats.get("recovered_vs_strict_count", 0) or 0)
    spectral_distance = _spectral_distance(
        relaxed_stats.get("prepared_summary", {}).get("spectral_proxies", {}),
        offline_reference.get("spectral_proxies", {}),
    )
    live_clip = relaxed_stats.get("prepared_summary", {}).get("prepared_total_clip_mean")
    offline_clip = offline_reference.get("prepared_total_clip_mean")
    verdict = "nominal"
    reason = "Accepted live windows remain within the expected model-input envelope."
    catastrophic = False
    if int(relaxed_stats.get("accepted_count", 0) or 0) <= 0:
        verdict = "catastrophic"
        reason = "No accepted live windows were available for model-order distribution analysis."
        catastrophic = True
    elif quality_bad_rate is not None and float(quality_bad_rate) >= 0.75:
        verdict = "catastrophic"
        reason = "Most accepted live windows still fail quality checks."
        catastrophic = True
    elif masked_window_rate is not None and float(masked_window_rate) >= 0.75:
        verdict = "catastrophic"
        reason = "Most accepted live windows require masking, indicating gross signal instability."
        catastrophic = True
    elif median_rms_ratio is not None and (
        float(median_rms_ratio) < 0.35 or float(median_rms_ratio) > 2.25
    ):
        verdict = "catastrophic"
        reason = "Prepared live-window amplitude is grossly outside the offline model-input range."
        catastrophic = True
    elif median_rms_ratio is not None and low_channels >= 2 and float(median_rms_ratio) < 0.78:
        verdict = "shifted_low_amplitude"
        reason = "Prepared live-window amplitude is materially quieter than the offline reference on multiple channels."
    elif (
        median_rms_ratio is not None
        and (
            (high_channels >= 2 and float(median_rms_ratio) > 1.25)
            or (
                quality_bad_rate is not None
                and float(quality_bad_rate) >= 0.20
                and float(median_rms_ratio) > 1.10
            )
        )
    ):
        verdict = "shifted_high_amplitude"
        reason = "Prepared live-window amplitude is materially stronger or noisier than the offline reference."
    return {
        "decisive": bool(decisive),
        "verdict": str(verdict),
        "reason": str(reason),
        "catastrophic": bool(catastrophic),
        "reference_split": "all",
        "accepted_rate_strict": float(strict_stats.get("accepted_rate", 0.0) or 0.0),
        "accepted_rate_relaxed": accepted_rate,
        "recovered_vs_strict_count": int(recovered_vs_strict_count),
        "quality_bad_rate_relaxed": (
            float(quality_bad_rate) if quality_bad_rate is not None else None
        ),
        "masked_window_rate_relaxed": (
            float(masked_window_rate) if masked_window_rate is not None else None
        ),
        "per_channel_rms_ratio": rms_ratios,
        "median_rms_ratio": median_rms_ratio,
        "spectral_distance_mean_abs": spectral_distance,
        "prepared_total_clip_mean_live": (
            float(live_clip) if live_clip is not None else None
        ),
        "prepared_total_clip_mean_offline": (
            float(offline_clip) if offline_clip is not None else None
        ),
    }


def _load_prediction_confidence(predictions_path: Path | None) -> dict[str, Any] | None:
    if predictions_path is None or not predictions_path.exists():
        return None
    joint_conf: list[float] = []
    action_conf: list[float] = []
    finger_conf: list[float] = []
    with predictions_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if not isinstance(row, dict):
                continue
            for key, bucket in (
                ("joint_conf", joint_conf),
                ("action_conf", action_conf),
                ("finger_conf", finger_conf),
            ):
                value = row.get(key)
                try:
                    if value is not None:
                        bucket.append(float(value))
                except Exception:
                    continue
    return {
        "joint_conf": _series(np.asarray(joint_conf, dtype=float)),
        "action_conf": _series(np.asarray(action_conf, dtype=float)),
        "finger_conf": _series(np.asarray(finger_conf, dtype=float)),
    }


def build_distribution_report(
    *,
    raw_source: np.ndarray | Path,
    run_dir: Path,
    offline_npz: Path,
    runtime_manifest_path: Path | None = None,
    runtime_manifest: dict[str, Any] | None = None,
    window_sec: float = 0.25,
    hop_sec: float = 0.05,
    target_fs: float = 256.0,
    relaxed_gap_s: float | None = None,
    max_offline_windows: int = 4000,
    confidence_sample_windows: int = 256,
    predictions_path: Path | None = None,
) -> dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    offline_npz = Path(offline_npz).expanduser().resolve()
    raw = _ensure_raw_array(raw_source)
    raw_dir = (
        Path(raw_source).expanduser().resolve()
        if isinstance(raw_source, Path)
        else None
    )
    loaded_runtime_manifest, loaded_runtime_manifest_path = _load_runtime_manifest(
        raw_dir=raw_dir,
        runtime_manifest_path=runtime_manifest_path,
        runtime_manifest=runtime_manifest,
    )
    quality_thresholds = _quality_thresholds_from_manifest(loaded_runtime_manifest)
    scaler = load_normalizer(run_dir / "scaler.npz")
    if scaler is None:
        raise RuntimeError(f"Failed to load scaler from {run_dir / 'scaler.npz'}")
    device = torch.device("cpu")
    model, _, temperature_state = load_model_artifacts_from_files(
        model_path=run_dir / "finger_action_model.pt",
        scaler_path=run_dir / "scaler.npz",
        device=device,
        n_channels=int(scaler.get("channels", 4)),
    )
    model_bundle = (model, device, temperature_state)

    raw_stats = _build_raw_channel_stats_payload(
        raw,
        runtime_manifest=loaded_runtime_manifest,
    )
    raw_stats["shard_count"] = (
        int(len(sorted(raw_dir.glob("eeg_raw_shard_*.npy"))))
        if raw_dir is not None and raw_dir.exists()
        else 1
    )

    samples = np.asarray(raw["sample"], dtype=float)
    finite_sample_mask = np.all(np.isfinite(samples), axis=1)
    if "lsl_ts_mono" not in (raw.dtype.names or ()):
        raise RuntimeError("Raw input payload is missing lsl_ts_mono")
    finite_times = np.asarray(raw["lsl_ts_mono"], dtype=float)[finite_sample_mask]
    finite_values = samples[finite_sample_mask]
    if finite_times.size > 0:
        finite_times = finite_times - float(finite_times[0])
    reorder = raw_stats.get("reorder", {}).get("channel_reorder_to_model_order")
    if isinstance(reorder, list):
        finite_model_values = finite_values[:, np.asarray(reorder, dtype=np.int64)]
    else:
        finite_model_values = finite_values
    dropped_nonfinite_samples = int(np.sum(~finite_sample_mask))
    filtered_diffs = np.diff(finite_times) if finite_times.size >= 2 else np.asarray([], dtype=float)
    strict_gap_s = 1.0 / float(target_fs) * 4.0
    runtime_relaxed_gap_s = (
        loaded_runtime_manifest.get("runtime", {}).get("alignment_internal_max_gap_s")
        if isinstance(loaded_runtime_manifest.get("runtime"), dict)
        else None
    )
    relaxed_gap_value = float(
        relaxed_gap_s
        if relaxed_gap_s is not None
        else (
            runtime_relaxed_gap_s
            if runtime_relaxed_gap_s is not None
            else max(strict_gap_s, 0.06)
        )
    )

    offline_stats = _offline_stats(
        offline_npz,
        scaler=scaler,
        max_windows=max(1, int(max_offline_windows)),
        clip_abs_z=float(quality_thresholds["input_clip_abs_z"]),
        target_fs=float(target_fs),
    )
    strict_stats = _window_stats(
        times=finite_times,
        values=finite_model_values,
        scaler=scaler,
        window_sec=float(window_sec),
        hop_sec=float(hop_sec),
        target_fs=float(target_fs),
        strict_gap_s=float(strict_gap_s),
        max_gap_s=float(strict_gap_s),
        model_bundle=model_bundle,
        confidence_sample_windows=max(0, int(confidence_sample_windows)),
        quality_thresholds=quality_thresholds,
    )
    relaxed_stats = _window_stats(
        times=finite_times,
        values=finite_model_values,
        scaler=scaler,
        window_sec=float(window_sec),
        hop_sec=float(hop_sec),
        target_fs=float(target_fs),
        strict_gap_s=float(strict_gap_s),
        max_gap_s=float(relaxed_gap_value),
        model_bundle=model_bundle,
        confidence_sample_windows=max(0, int(confidence_sample_windows)),
        quality_thresholds=quality_thresholds,
    )
    distribution_match = _classify_distribution_match(
        relaxed_stats=relaxed_stats,
        strict_stats=strict_stats,
        offline_reference=offline_stats.get("all", {}),
        decisive=bool(raw_stats.get("distribution_claim_decisive")),
    )

    report = {
        "status": "ok",
        "raw_dir": str(raw_dir) if raw_dir is not None else None,
        "run_dir": str(run_dir),
        "offline_npz": str(offline_npz),
        "runtime_manifest_path": (
            str(loaded_runtime_manifest_path)
            if loaded_runtime_manifest_path is not None
            else None
        ),
        "window_sec": float(window_sec),
        "hop_sec": float(hop_sec),
        "target_fs": float(target_fs),
        "strict_alignment_gap_s": float(strict_gap_s),
        "relaxed_alignment_gap_s": float(relaxed_gap_value),
        "quality_thresholds": quality_thresholds,
        "distribution_claim_decisive": bool(raw_stats.get("distribution_claim_decisive")),
        "limitations": list(raw_stats.get("limitations", [])),
        "reorder": dict(raw_stats.get("reorder", {})),
        "raw_channel_stats": raw_stats,
        "raw_stream": {
            "total_samples": int(len(raw)),
            "finite_samples": int(np.sum(finite_sample_mask)),
            "dropped_nonfinite_samples": int(dropped_nonfinite_samples),
            "filtered_gap_s": _series(filtered_diffs),
            "filtered_gap_counts": {
                str(threshold): int(np.sum(filtered_diffs > float(threshold)))
                for threshold in sorted(set([strict_gap_s, 0.05, 0.06, 0.1]))
            },
        },
        "offline_stats": offline_stats,
        "alignment": {
            "strict": strict_stats,
            "relaxed": relaxed_stats,
        },
        "distribution_match": distribution_match,
        "prediction_confidence": _load_prediction_confidence(predictions_path),
    }
    if not bool(report["distribution_claim_decisive"]):
        report["status"] = "partial"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare raw-captured live input windows against offline training-window statistics."
    )
    parser.add_argument("--raw-dir", required=True, help="Path to a Step 7 raw shard directory.")
    parser.add_argument("--run-dir", required=True, help="Model run directory with model/scaler artifacts.")
    parser.add_argument("--offline-npz", required=True, help="Offline eeg_windows.npz used for training/eval comparison.")
    parser.add_argument("--runtime-manifest", type=str, default=None, help="Optional live_runtime_manifest.json path.")
    parser.add_argument("--predictions-path", type=str, default=None, help="Optional predictions.jsonl path for confidence summaries.")
    parser.add_argument("--window-sec", type=float, default=0.25)
    parser.add_argument("--hop-sec", type=float, default=0.05)
    parser.add_argument("--target-fs", type=float, default=256.0)
    parser.add_argument("--relaxed-gap-s", type=float, default=None, help="Optional relaxed internal gap tolerance override.")
    parser.add_argument("--max-offline-windows", type=int, default=4000)
    parser.add_argument("--confidence-sample-windows", type=int, default=256)
    parser.add_argument("--report-out", type=str, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    report = build_distribution_report(
        raw_source=raw_dir,
        run_dir=Path(args.run_dir).expanduser().resolve(),
        offline_npz=Path(args.offline_npz).expanduser().resolve(),
        runtime_manifest_path=(
            Path(args.runtime_manifest).expanduser().resolve()
            if args.runtime_manifest
            else None
        ),
        window_sec=float(args.window_sec),
        hop_sec=float(args.hop_sec),
        target_fs=float(args.target_fs),
        relaxed_gap_s=(
            float(args.relaxed_gap_s) if args.relaxed_gap_s is not None else None
        ),
        max_offline_windows=max(1, int(args.max_offline_windows)),
        confidence_sample_windows=max(0, int(args.confidence_sample_windows)),
        predictions_path=(
            Path(args.predictions_path).expanduser().resolve()
            if args.predictions_path
            else None
        ),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.report_out:
        out_path = Path(args.report_out).expanduser().resolve()
        out_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report.get("distribution_match", {}).get("catastrophic") is not True else 1


if __name__ == "__main__":
    raise SystemExit(main())
