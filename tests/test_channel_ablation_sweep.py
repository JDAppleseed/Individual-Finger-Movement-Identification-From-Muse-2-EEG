import csv

import numpy as np
import pytest

from tools.channel_ablation_sweep import (
    ChannelSubset,
    build_subset_plan,
    write_subset_npz,
    write_summary,
)


def test_build_subset_plan_defaults_cover_all_single_and_leave_one_out():
    names = ["TP9", "AF7", "AF8", "TP10"]

    subsets = build_subset_plan(
        names,
        modes={"all", "singles", "leave-one-out"},
    )

    ids = [subset.subset_id for subset in subsets]
    assert ids == [
        "all",
        "single_TP9",
        "single_AF7",
        "single_AF8",
        "single_TP10",
        "drop_TP9",
        "drop_AF7",
        "drop_AF8",
        "drop_TP10",
    ]
    assert subsets[-1].indices == (0, 1, 2)
    assert subsets[-1].omitted_channel == "TP10"


def test_write_subset_npz_slices_x_and_channel_names(tmp_path):
    source = tmp_path / "source.npz"
    out = tmp_path / "subset" / "eeg_windows.npz"
    x = np.arange(2 * 3 * 4, dtype=np.float32).reshape(2, 3, 4)
    np.savez(
        source,
        X=x,
        y_action=np.array([0, 1], dtype=np.int64),
        y_finger=np.array([0, 2], dtype=np.int64),
        channel_names=np.array(["TP9", "AF7", "AF8", "TP10"]),
        window_start=np.array([0.0, 0.05], dtype=np.float32),
    )
    subset = ChannelSubset(
        subset_id="pair_TP9_AF8",
        kind="pair",
        indices=(0, 2),
        names=("TP9", "AF8"),
    )

    write_subset_npz(source, out, subset)

    with np.load(out, allow_pickle=True) as data:
        np.testing.assert_array_equal(data["X"], x[:, :, [0, 2]])
        np.testing.assert_array_equal(data["channel_names"], np.array(["TP9", "AF8"]))
        np.testing.assert_array_equal(data["y_action"], np.array([0, 1]))
        np.testing.assert_array_equal(data["channel_ablation_indices"], np.array([0, 2]))


def test_write_summary_computes_drop_vs_all(tmp_path):
    rows = [
        {
            "subset_id": "all",
            "kind": "all",
            "channels": "TP9+AF7+AF8+TP10",
            "omitted_channel": "",
            "n_channels": 4,
            "seed": 43,
            "status": "ok",
            "action_acc": 0.90,
            "finger_acc_non_rest": 0.80,
            "n_test": 10,
            "n_test_non_rest": 8,
            "run_dir": "all",
            "npz_path": "all/eeg_windows.npz",
        },
        {
            "subset_id": "drop_TP9",
            "kind": "leave_one_out",
            "channels": "AF7+AF8+TP10",
            "omitted_channel": "TP9",
            "n_channels": 3,
            "seed": 43,
            "status": "ok",
            "action_acc": 0.75,
            "finger_acc_non_rest": 0.70,
            "n_test": 10,
            "n_test_non_rest": 8,
            "run_dir": "drop_TP9",
            "npz_path": "drop_TP9/eeg_windows.npz",
        },
    ]

    write_summary(tmp_path, rows)

    with (tmp_path / "summary.csv").open() as handle:
        parsed = list(csv.DictReader(handle))
    drop_row = next(row for row in parsed if row["subset_id"] == "drop_TP9")
    assert float(drop_row["action_drop_vs_all"]) == pytest.approx(0.15)
    assert float(drop_row["finger_drop_vs_all"]) == pytest.approx(0.10)
    assert "Leave-One-Out Importance" in (tmp_path / "summary.md").read_text()
