from utils.segmenting import SegmentBreaker


def test_segment_break_on_backwards_lsl():
    breaker = SegmentBreaker(gap_break_s=1.0)
    assert breaker.check(1.0).should_break is False
    result = breaker.check(0.9)
    assert result.should_break is True
    assert result.reason == "backwards"


def test_segment_break_on_large_gap():
    breaker = SegmentBreaker(gap_break_s=1.0)
    assert breaker.check(1.0).should_break is False
    result = breaker.check(2.5)
    assert result.should_break is True
    assert result.reason == "gap"
