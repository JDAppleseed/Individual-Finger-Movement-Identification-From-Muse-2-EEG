from utils.stream_health import StreamHealthMonitor


def test_stream_health_stalls_when_no_writes():
    monitor = StreamHealthMonitor(timeout_s=1.0)
    status = monitor.check(0.0)
    assert not status.active
    assert status.stalled_reason == "no_csv_updates"


def test_stream_health_recovers_after_write():
    monitor = StreamHealthMonitor(timeout_s=1.0)
    monitor.mark_write(0.0, "2024-01-01T00:00:00Z")
    status = monitor.check(0.5)
    assert status.active
    assert status.last_write_utc == "2024-01-01T00:00:00Z"
    status = monitor.check(2.0)
    assert not status.active
    monitor.mark_write(2.5, "2024-01-01T00:00:03Z")
    status = monitor.check(2.6)
    assert status.active
    assert status.last_write_utc == "2024-01-01T00:00:03Z"
