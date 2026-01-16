import os
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


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
