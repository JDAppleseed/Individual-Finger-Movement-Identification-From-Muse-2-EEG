import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from utils.stream_health import RollingHealthDecision

def _load_stream_module():
    os.environ["STREAM_IMPORT_ONLY"] = "1"
    module_path = Path(__file__).resolve().parents[1] / "1_stream_and_record.py"
    spec = spec_from_file_location("stream_module", module_path)
    module = module_from_spec(spec)
    sys.modules["stream_module"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stream_init_does_not_nameerror():
    module = _load_stream_module()
    payload = module._build_session_state_payload(module.state)
    assert "gap_count" in payload


def test_channel_reorder_applied_when_len_equals_channels():
    module = _load_stream_module()
    sample = [1.0, 2.0, 3.0, 4.0]
    indices = [1, 3, 0, 2]
    assert module._apply_channel_indices(sample, indices, 4) == [2.0, 4.0, 1.0, 3.0]
    assert module._apply_channel_indices(sample, [0, 1, 2, 3], 4) == sample


def test_segment_ica_future_discard():
    module = _load_stream_module()
    assert module._should_accept_ica_result(1, 0) is False
    assert module._should_accept_ica_result(2, 2) is True


def test_reset_segment_state_safe_before_buffers():
    module = _load_stream_module()
    module._reset_segment_state(module.state)


def test_reset_segment_state_resets_report_gates():
    module = _load_stream_module()
    module.last_report_time = 12.0
    module.last_report_samples_written = 99
    module.timebase_report_initialized = True
    module.last_live_viz_emit = 7.5
    module._reset_segment_state(module.state)
    assert module.last_report_time == 0.0
    assert module.last_report_samples_written == 0
    assert module.timebase_report_initialized is False
    assert module.last_live_viz_emit == 0.0


def test_label_check_uses_expected_channels():
    module = _load_stream_module()
    module.stream_requirements.expected_channels = 5
    module.stream_requirements.required_labels = ["TP9", "AF7", "AF8", "TP10", "X1"]
    module.stream_requirements.require_exact_channels = True
    status = module._evaluate_label_check(
        ["TP9", "AF7", "AF8", "TP10", "X1"], 5
    )
    assert status["ok"] is True
    status = module._evaluate_label_check(["TP9", "AF7", "AF8", "TP10", "X1"], 4)
    assert status["ok"] is False
    assert "channel_count_mismatch" in status["reason"]


def test_startup_defers_failed_writers_until_health_decision(tmp_path):
    module = _load_stream_module()
    module.failed_writers.close_failed_files()
    module.health_state.has_health_decision = False
    module.health_state.unhealthy_since_mono = None
    module.health_state.failed_write_until_mono = None
    module.health_state.label_check_status = None
    decision = RollingHealthDecision(
        healthy=False,
        reason="stall_no_samples",
        measured_fs=None,
        write_rate=0.0,
        event_allowed=False,
        queue_size=0,
        backwards_count=0,
        last_received_lsl_ts=None,
        last_written_lsl_ts=None,
    )
    module._route_writers_for_health(0.0, decision)
    assert module.failed_writers.is_open() is False

    module.hard_stop_policy.failed_dir = str(tmp_path)
    module.health_state.has_health_decision = True
    module._route_writers_for_health(1.0, decision)
    assert module.failed_writers.is_open() is True
    module.failed_writers.close_failed_files()
