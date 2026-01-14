import pytest

np = pytest.importorskip("numpy")


def test_timebase_selfcheck_warns_on_large_delta():
    from utils.timebase_selfcheck import evaluate_timebase_alignment

    recent_samples = [0.0, 0.1, 0.2, 0.3]
    event_times = [0.0, 0.35]
    result = evaluate_timebase_alignment(
        recent_samples, event_times, warn_threshold_s=0.02, error_threshold_s=0.1
    )
    assert result.max_abs_delta_s is not None
    assert result.warn is True
    assert result.error is True


def test_timebase_selfcheck_ok_when_aligned():
    from utils.timebase_selfcheck import evaluate_timebase_alignment

    recent_samples = [0.0, 0.1, 0.2, 0.3]
    event_times = [0.1, 0.2]
    result = evaluate_timebase_alignment(
        recent_samples, event_times, warn_threshold_s=0.05, error_threshold_s=0.2
    )
    assert result.warn is False
    assert result.error is False
