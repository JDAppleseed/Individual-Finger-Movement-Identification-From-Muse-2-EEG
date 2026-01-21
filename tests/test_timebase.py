from __future__ import annotations

import logging
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from console_utils import install_console_logging, get_console_tail
from timebase import TimebaseMapper, TimebaseSnapshot, map_lsl_to_mono_snapshot


def test_monotonic_mapping() -> None:
    mapper = TimebaseMapper()
    now = 100.0
    prev = None
    for lsl_raw in [10.0, 10.01, 10.02, 10.03]:
        now += 0.01
        continuous, mapped, discontinuity, clamped = mapper.map(lsl_raw, now)
        assert discontinuity is False
        assert clamped is False
        if prev is not None:
            assert continuous > prev
        prev = continuous
        assert continuous >= mapped


def test_backward_jump() -> None:
    mapper = TimebaseMapper()
    now = 200.0
    discontinuities = []
    continuous_values = []
    for lsl_raw in [10.0, 10.01, 9.50, 9.51]:
        now += 0.01
        continuous, _, discontinuity, _ = mapper.map(lsl_raw, now)
        discontinuities.append(discontinuity)
        continuous_values.append(continuous)
    assert discontinuities[2] is True
    assert all(b > a for a, b in zip(continuous_values, continuous_values[1:]))


def test_forward_jump() -> None:
    mapper = TimebaseMapper()
    now = 300.0
    discontinuities = []
    for lsl_raw in [10.0, 10.01, 15.0]:
        now += 0.01
        _, _, discontinuity, _ = mapper.map(lsl_raw, now)
        discontinuities.append(discontinuity)
    assert discontinuities[2] is True


def test_console_cap_warning() -> None:
    logger = install_console_logging(200, ring_chars=5000, level=logging.INFO)
    for _ in range(100):
        logger.info("x" * 10)
    tail = get_console_tail(5000)
    warning_line = "[console] output capped; further console output suppressed"
    assert tail.count(warning_line) == 1


def test_snapshot_mapping_requires_anchor() -> None:
    snapshot = TimebaseSnapshot()
    assert map_lsl_to_mono_snapshot(snapshot, 10.0) is None
    snapshot.has_anchor = True
    snapshot.offset = 5.0
    snapshot.prev_continuous_mono = 20.0
    snapshot.eps = 1e-6
    first = map_lsl_to_mono_snapshot(snapshot, 10.0)
    assert first is not None
    second = map_lsl_to_mono_snapshot(snapshot, 10.001)
    assert second is not None
    assert second >= first


if __name__ == "__main__":
    test_monotonic_mapping()
    test_backward_jump()
    test_forward_jump()
    test_console_cap_warning()
    test_snapshot_mapping_requires_anchor()
    print("OK")
