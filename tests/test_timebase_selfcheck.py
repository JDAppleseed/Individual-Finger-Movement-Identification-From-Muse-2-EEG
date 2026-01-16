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
    assert result.error is False


def test_timebase_selfcheck_ok_when_aligned():
    from utils.timebase_selfcheck import evaluate_timebase_alignment

    recent_samples = [0.0, 0.1, 0.2, 0.3]
    event_times = [0.1, 0.2]
    result = evaluate_timebase_alignment(
        recent_samples, event_times, warn_threshold_s=0.05, error_threshold_s=0.2
    )
    assert result.warn is False
    assert result.error is False


def test_timebase_selfcheck_errors_on_mean_or_max():
    from utils.timebase_selfcheck import evaluate_timebase_alignment

    recent_samples = [0.0, 0.1, 0.2, 0.3]
    event_times = [0.0, 0.9]
    result = evaluate_timebase_alignment(
        recent_samples, event_times, warn_threshold_s=0.05, error_threshold_s=0.2
    )
    assert result.warn is True
    assert result.error is True


def test_timebase_consistency_flags_sustained_warns():
    from utils.timebase_selfcheck import evaluate_timebase_consistency

    time_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    lsl_mono = np.array([1.0, 1.1, 1.2, 1.3, 1.4], dtype=float)
    stream_start = 1.0
    # Inject sustained drift beyond warn threshold
    time_s = time_s + np.array([0.0, 0.02, 0.03, 0.02, 0.03])

    result = evaluate_timebase_consistency(
        time_s, lsl_mono, stream_start, warn_threshold_s=0.01, error_threshold_s=0.05
    )
    assert result.warn is True
    assert result.error is True


def test_event_time_consistency_ok_when_aligned():
    from utils.timebase_selfcheck import evaluate_event_time_consistency

    onset_s = np.array([0.0, 0.5, 1.0], dtype=float)
    onset_lsl = np.array([10.0, 10.5, 11.0], dtype=float)
    result = evaluate_event_time_consistency(onset_s, onset_lsl, 10.0)
    assert result.warn is False
    assert result.error is False
