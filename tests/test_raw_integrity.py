import os
import sys
import csv
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _load_stream_module():
    os.environ["STREAM_IMPORT_ONLY"] = "1"
    module_path = Path(__file__).resolve().parents[1] / "1_stream_and_record.py"
    spec = spec_from_file_location("stream_module_raw", module_path)
    module = module_from_spec(spec)
    sys.modules["stream_module_raw"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_analyze_lsl_timestamp_gaps_counts_missing():
    module = _load_stream_module()
    timestamps = [0.0, 0.1, 0.2, 0.5, 0.6]
    result = module.analyze_lsl_timestamp_gaps(timestamps, nominal_fs=10.0)
    assert result["gap_count"] == 1
    assert result["estimated_missing"] == 2
    assert result["expected_samples"] == 6


def test_raw_writer_writes_nonfinite(tmp_path):
    module = _load_stream_module()
    path = tmp_path / "raw.csv"
    file_obj, writer = module._open_raw_csv(path)
    sample = [1.0, float("nan"), 2.0, float("inf")]
    flags = module._raw_flags_for_sample(sample)
    assert flags & module.RAW_FLAG_NONFINITE
    row = module._build_raw_row(1.0, 1.0, 1.0, sample, flags)
    writer.writerow(row)
    file_obj.flush()
    file_obj.close()

    with open(path, "r", newline="") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2
    assert rows[0][-1] == "flags"
    assert rows[1][-1] == str(flags)
