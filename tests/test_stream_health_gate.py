from utils.stream_health import RollingStreamHealthGate


def test_event_gating_on_stall():
    gate = RollingStreamHealthGate(
        expected_fs=1.0,
        health_window_s=1.0,
        stall_s=0.25,
        min_write_fraction=0.9,
        max_queue=4,
        recovery_s=0.0,
        backwards_threshold=3,
        backwards_window_s=1.0,
    )
    gate.record_received(0.0, 0.0)
    gate.record_written(0.0, 0.0)
    assert gate.evaluate(0.0).event_allowed is True
    assert gate.evaluate(0.3).event_allowed is False


def test_event_gating_recovers():
    gate = RollingStreamHealthGate(
        expected_fs=10.0,
        health_window_s=1.0,
        stall_s=2.0,
        min_write_fraction=0.9,
        max_queue=4,
        recovery_s=0.5,
        backwards_threshold=3,
        backwards_window_s=1.0,
    )
    for idx in range(10):
        t = idx * 0.1
        gate.record_received(float(idx), t)
        gate.record_written(float(idx), t)
    assert gate.evaluate(0.2).event_allowed is False
    assert gate.evaluate(0.8).event_allowed is True
