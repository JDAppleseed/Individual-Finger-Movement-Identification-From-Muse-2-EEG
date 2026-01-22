import os
import sys
import queue
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


def _load_stream_module():
    os.environ["STREAM_IMPORT_ONLY"] = "1"
    module_path = Path(__file__).resolve().parents[1] / "1_stream_and_record.py"
    spec = spec_from_file_location("stream_module_backpressure", module_path)
    module = module_from_spec(spec)
    sys.modules["stream_module_backpressure"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_backpressure_triggers_shutdown():
    module = _load_stream_module()
    module.MODE = "train_record"
    module.ALLOW_DROP = False
    module.MAX_BACKPRESSURE_S = 0.0
    module.QUEUE_PUT_TIMEOUT_S = 0.01
    module.termination_reason = "normal"
    module.stop_event.clear()
    test_queue = queue.Queue(maxsize=1)
    test_queue.put_nowait(
        module.SamplePacket(
            seq=0,
            lsl_ts_raw=0.0,
            lsl_ts_mono=0.0,
            local_ts=0.0,
            sample=np.zeros(4, dtype=float),
            flags=0,
            segment_id=0,
            raw_path=Path("."),
            clamped=False,
            segment_break_reason=None,
        )
    )
    module._enqueue_with_overflow(
        test_queue,
        module.SamplePacket(
            seq=1,
            lsl_ts_raw=0.1,
            lsl_ts_mono=0.1,
            local_ts=0.1,
            sample=np.zeros(4, dtype=float),
            flags=0,
            segment_id=0,
            raw_path=Path("."),
            clamped=False,
            segment_break_reason=None,
        ),
        label="raw",
    )
    assert module.stop_event.is_set()
    assert module.termination_reason == "backpressure_abort"
