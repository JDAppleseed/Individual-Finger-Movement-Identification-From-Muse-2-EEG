from utils.stream_health import RollingStreamHealthGate


def test_event_gating_on_stall():
    gate = RollingStreamHealthGate(
        expected_fs=1.0,
        health_window_s=1.0,
        stall_s=0.25,
        max_queue=4,
        backlog_grace_s=0.0,
        recovery_s=0.0,
        backwards_threshold=3,
        backwards_window_s=1.0,
        gap_threshold_s=1.5,
        gap_count_threshold=3,
        gap_window_s=1.0,
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
        max_queue=4,
        backlog_grace_s=0.0,
        recovery_s=0.5,
        backwards_threshold=3,
        backwards_window_s=1.0,
        gap_threshold_s=0.2,
        gap_count_threshold=3,
        gap_window_s=1.0,
    )
    for idx in range(10):
        t = idx * 0.1
        lsl_ts = float(idx) * 0.1
        gate.record_received(lsl_ts, t)
        gate.record_written(lsl_ts, t)
    assert gate.evaluate(0.2).event_allowed is False
    assert gate.evaluate(0.8).event_allowed is True


def test_gate_does_not_flag_write_rate_low_without_writes():
    gate = RollingStreamHealthGate(
        expected_fs=10.0,
        health_window_s=1.0,
        stall_s=2.0,
        max_queue=100,
        backlog_grace_s=0.0,
        recovery_s=0.0,
        backwards_threshold=3,
        backwards_window_s=1.0,
        gap_threshold_s=0.2,
        gap_count_threshold=3,
        gap_window_s=1.0,
    )
    for idx in range(5):
        t = idx * 0.1
        lsl_ts = float(idx) * 0.1
        gate.record_received(lsl_ts, t)
    decision = gate.evaluate(0.5)
    assert decision.healthy is True
    assert decision.reason is None
