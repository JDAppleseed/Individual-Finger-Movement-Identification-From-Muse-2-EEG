import importlib.util
from pathlib import Path


def _load_timebase_module():
    module_path = Path(__file__).resolve().parents[1] / "utils" / "session_timebase.py"
    spec = importlib.util.spec_from_file_location("session_timebase", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_session_continuous_time_resume():
    mod = _load_timebase_module()
    total_elapsed_s = 5.0
    run_start_lsl_ts = 100.0
    sample_lsl_ts = 102.0
    time_s = mod.compute_time_s(total_elapsed_s, run_start_lsl_ts, sample_lsl_ts)
    assert time_s == 7.0

    total_elapsed_s = time_s
    run_start_lsl_ts = 200.0
    sample_lsl_ts = 201.0
    resumed_time_s = mod.compute_time_s(total_elapsed_s, run_start_lsl_ts, sample_lsl_ts)
    assert resumed_time_s == 8.0


def test_event_time_from_local_clock():
    mod = _load_timebase_module()
    total_elapsed_s = 12.5
    run_start_lsl_ts = 400.0
    local_ts = 1000.0
    clock_offset = 5.0
    event_lsl_ts = mod.compute_event_lsl_ts(local_ts, clock_offset)
    event_time_s = mod.compute_event_time_s(total_elapsed_s, run_start_lsl_ts, event_lsl_ts)
    assert event_time_s == total_elapsed_s + (event_lsl_ts - run_start_lsl_ts)
