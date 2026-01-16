import csv
import json

import numpy as np

from diagnostics_cli import diagnostics


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_diagnostics_flags_prediction_order(tmp_path):
    features_path = tmp_path / "features.csv"
    events_path = tmp_path / "events.csv"
    predictions_path = tmp_path / "predictions.csv"

    _write_csv(
        features_path,
        [
            "lsl_timestamp",
            "lsl_timestamp_mono",
            "time_s",
            "ch1",
            "ch2",
            "ch3",
            "ch4",
        ],
        [
            [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.1, 1.1, 0.1, 0.0, 0.0, 0.0, 0.0],
            [1.2, 1.2, 0.2, 0.0, 0.0, 0.0, 0.0],
        ],
    )
    _write_csv(
        events_path,
        ["onset_s", "onset_lsl"],
        [
            [0.05, 1.05],
        ],
    )
    _write_csv(
        predictions_path,
        ["prediction_time_s", "prediction_lsl_ts"],
        [
            [0.15, 1.15],
            [0.1, 1.1],
        ],
    )

    session_meta = {
        "stream_start_lsl_ts": 1.0,
        "sampling_rate": 256.0,
    }

    report = diagnostics(features_path, events_path, predictions_path, session_meta)
    assert report["pred_monotonic"] is False
    assert report["verdict"] == "INVALID"
    assert report["event_out_of_range"] == 0
    assert report["pred_out_of_range"] == 0
    assert report["gap_count"] == 0
