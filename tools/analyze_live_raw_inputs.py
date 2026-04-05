#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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


def _series(values: np.ndarray) -> dict[str, float | None]:
    if values.size == 0:
        return {"mean": None, "p50": None, "p95": None, "max": None}
    return {
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _load_raw_dir(raw_dir: Path) -> np.ndarray:
    shards = sorted(raw_dir.glob("eeg_raw_shard_*.npy"))
    if not shards:
        raise FileNotFoundError(f"No raw shards found under {raw_dir}")
    return np.concatenate([np.load(path) for path in shards])


def _offline_stats(npz_path: Path, scaler: Any, max_windows: int) -> dict[str, Any]:
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
        abs_norm = np.abs(X_norm)
        out[label] = {
            "window_count": int(len(X_sel)),
            "prepared_rms_mean": np.sqrt(np.mean(X_norm**2, axis=1)).mean(axis=0).tolist(),
            "prepared_abs_p95_mean": np.percentile(abs_norm, 95, axis=1).mean(axis=0).tolist(),
            "prepared_total_clip_mean": float(np.mean(abs_norm > 6.0)),
        }
    return out


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
) -> dict[str, Any]:
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
            input_clip_abs_z=6.0,
            bad_channel_rms_z=4.0,
            bad_channel_abs_p95_z=6.0,
            bad_channel_clipped_frac=0.05,
            bad_window_clipped_frac=0.10,
            bad_window_max_masked_channels=1,
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
    abs_prepared = np.abs(prepared)
    return {
        "candidate_count": int(candidate_count),
        "accepted_count": int(accepted_count),
        "dropped_count": int(candidate_count - accepted_count),
        "recovered_vs_strict_count": int(recovered_vs_strict),
        "quality_bad_count": int(quality_bad_count),
        "masked_window_count": int(masked_window_count),
        "drop_reason_counts": dict(reasons),
        "prepared_rms_mean": (
            np.sqrt(np.mean(prepared**2, axis=1)).mean(axis=0).tolist()
            if prepared.size
            else []
        ),
        "prepared_abs_p95_mean": (
            np.percentile(abs_prepared, 95, axis=1).mean(axis=0).tolist()
            if prepared.size
            else []
        ),
        "prepared_total_clip_mean": (
            float(np.mean(abs_prepared > 6.0)) if prepared.size else None
        ),
        "sampled_joint_conf": _series(np.asarray(sampled_joint_conf, dtype=float)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare raw-captured live input windows against offline training-window statistics."
    )
    parser.add_argument("--raw-dir", required=True, help="Path to a Step 7 raw shard directory.")
    parser.add_argument("--run-dir", required=True, help="Model run directory with model/scaler artifacts.")
    parser.add_argument("--offline-npz", required=True, help="Offline eeg_windows.npz used for training/eval comparison.")
    parser.add_argument("--window-sec", type=float, default=0.25)
    parser.add_argument("--hop-sec", type=float, default=0.05)
    parser.add_argument("--target-fs", type=float, default=256.0)
    parser.add_argument(
        "--alignment-gap-s",
        dest="alignment_gaps",
        action="append",
        type=float,
        default=None,
        help="Alignment internal-gap tolerance to evaluate. Repeatable. Defaults to strict and 0.06.",
    )
    parser.add_argument("--max-offline-windows", type=int, default=4000)
    parser.add_argument("--confidence-sample-windows", type=int, default=256)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir).expanduser().resolve()
    run_dir = Path(args.run_dir).expanduser().resolve()
    offline_npz = Path(args.offline_npz).expanduser().resolve()
    strict_gap_s = 1.0 / float(args.target_fs) * 4.0
    alignment_gaps = args.alignment_gaps or [strict_gap_s, 0.06]

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

    raw = _load_raw_dir(raw_dir)
    samples = np.stack(raw["sample"]).astype(float)
    finite_sample_mask = np.all(np.isfinite(samples), axis=1)
    finite_times = raw["lsl_ts_mono"].astype(float)[finite_sample_mask]
    finite_times -= float(finite_times[0])
    finite_values = samples[finite_sample_mask]
    dropped_nonfinite_samples = int(np.sum(~finite_sample_mask))
    filtered_diffs = np.diff(finite_times)

    report = {
        "raw_dir": str(raw_dir),
        "run_dir": str(run_dir),
        "offline_npz": str(offline_npz),
        "window_sec": float(args.window_sec),
        "hop_sec": float(args.hop_sec),
        "target_fs": float(args.target_fs),
        "strict_alignment_gap_s": float(strict_gap_s),
        "raw_stream": {
            "total_samples": int(len(raw)),
            "finite_samples": int(np.sum(finite_sample_mask)),
            "dropped_nonfinite_samples": dropped_nonfinite_samples,
            "filtered_gap_s": _series(filtered_diffs),
            "filtered_gap_counts": {
                str(threshold): int(np.sum(filtered_diffs > float(threshold)))
                for threshold in sorted(set([strict_gap_s, 0.05, 0.06, 0.1]))
            },
        },
        "offline_stats": _offline_stats(
            offline_npz,
            scaler=scaler,
            max_windows=max(1, int(args.max_offline_windows)),
        ),
        "alignment_evals": {},
    }

    for gap_s in alignment_gaps:
        report["alignment_evals"][str(float(gap_s))] = _window_stats(
            times=finite_times,
            values=finite_values,
            scaler=scaler,
            window_sec=float(args.window_sec),
            hop_sec=float(args.hop_sec),
            target_fs=float(args.target_fs),
            strict_gap_s=float(strict_gap_s),
            max_gap_s=float(gap_s),
            model_bundle=model_bundle,
            confidence_sample_windows=max(0, int(args.confidence_sample_windows)),
        )

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
