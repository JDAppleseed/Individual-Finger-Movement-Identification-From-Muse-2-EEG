from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from muse_streaming.resample import verify_alignment


def _load_live_module():
    module_path = Path(__file__).resolve().parents[1] / "7_live_infer_and_actuate.py"
    spec = importlib.util.spec_from_file_location("live_input_contract_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_verify_alignment_allows_bounded_internal_gap_with_strict_edges() -> None:
    times = np.arange(0.0, 0.25, 1.0 / 256.0, dtype=float)
    times = np.concatenate([times[:20], times[33:]])

    strict = verify_alignment(
        times,
        start_s=0.0,
        end_s=0.25,
        target_fs=256.0,
        max_gap_s=1.0 / 256.0 * 4.0,
        max_edge_gap_s=1.0 / 256.0 * 4.0,
    )
    relaxed = verify_alignment(
        times,
        start_s=0.0,
        end_s=0.25,
        target_fs=256.0,
        max_gap_s=0.06,
        max_edge_gap_s=1.0 / 256.0 * 4.0,
    )

    assert strict.ok is False
    assert strict.reason == "gap_exceeds_threshold"
    assert relaxed.ok is True
    assert relaxed.start_gap_s == 0.0
    assert relaxed.end_gap_s is not None and relaxed.end_gap_s <= (1.0 / 256.0 * 4.0)
    assert relaxed.max_gap_s is not None and relaxed.max_gap_s > (1.0 / 256.0 * 4.0)


def test_verify_alignment_keeps_window_edges_strict() -> None:
    times = np.arange(0.05, 0.25, 1.0 / 256.0, dtype=float)

    report = verify_alignment(
        times,
        start_s=0.0,
        end_s=0.25,
        target_fs=256.0,
        max_gap_s=0.06,
        max_edge_gap_s=1.0 / 256.0 * 4.0,
    )

    assert report.ok is False
    assert report.reason == "start_gap_exceeds_threshold"
    assert report.start_gap_s is not None and report.start_gap_s > 0.04


def test_channel_reorder_maps_stream_order_into_training_order() -> None:
    live_mod = _load_live_module()

    reorder = live_mod._build_channel_reorder(
        ["TP9", "AF7", "AF8", "TP10"],
        ["AF7", "TP9", "TP10", "AF8"],
    )

    assert reorder == (1, 0, 3, 2)


def test_resolve_effective_target_fs_canonicalizes_small_drift() -> None:
    live_mod = _load_live_module()

    effective, info = live_mod._resolve_effective_target_fs(
        train_config={"input_shape": [64, 4]},
        window_sec=0.25,
        requested_target_fs=256.12,
    )

    assert effective == 256.0
    assert info["adjusted"] is True
    assert info["model_input_time_samples"] == 64


def test_resolve_expected_channel_labels_falls_back_to_training_npz(tmp_path: Path) -> None:
    live_mod = _load_live_module()
    run_dir = tmp_path / "processed" / "models" / "run_001"
    run_dir.mkdir(parents=True)
    np.savez(
        tmp_path / "processed" / "eeg_windows.npz",
        channel_names=np.asarray(["TP9", "AF7", "AF8", "TP10"]),
    )

    labels, source = live_mod._resolve_expected_channel_labels({}, run_dir)

    assert labels == ["TP9", "AF7", "AF8", "TP10"]
    assert source == "training_npz.channel_names"
