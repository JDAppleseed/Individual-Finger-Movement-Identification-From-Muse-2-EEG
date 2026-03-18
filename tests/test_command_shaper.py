from utils.command_shaper import CommandShaper, CommandShaperConfig
from utils.palm_link import FLAG_HOLD, FLAG_WATCHDOG


def test_confidence_to_speed_mapping():
    shaper = CommandShaper(CommandShaperConfig(base_conf_thresh=0.75, speed_gamma=1.0))
    cmd = shaper.shape(
        action_id=1,
        finger_id=2,
        action_conf=0.6,
        timestamp_stream_ms=1000,
        timebase_ms=1000,
    )
    assert cmd.action_id == 0
    assert cmd.finger_id == 0
    assert cmd.speed_scalar == 0.0

    cmd = shaper.shape(
        action_id=1,
        finger_id=2,
        action_conf=1.0,
        timestamp_stream_ms=2000,
        timebase_ms=2000,
    )
    assert cmd.speed_scalar == 1.0


def test_speed_gamma():
    shaper = CommandShaper(CommandShaperConfig(base_conf_thresh=0.75, speed_gamma=2.0))
    cmd = shaper.shape(
        action_id=1,
        finger_id=1,
        action_conf=0.875,
        timestamp_stream_ms=1000,
        timebase_ms=1000,
    )
    assert abs(cmd.speed_scalar - 0.25) < 1e-6


def test_hold_on_change():
    shaper = CommandShaper(CommandShaperConfig(hold_ms=200))
    first = shaper.shape(
        action_id=1,
        finger_id=2,
        action_conf=0.9,
        timestamp_stream_ms=1000,
        timebase_ms=1000,
    )
    second = shaper.shape(
        action_id=2,
        finger_id=3,
        action_conf=0.9,
        timestamp_stream_ms=1100,
        timebase_ms=1100,
    )
    assert second.action_id == first.action_id
    assert second.finger_id == first.finger_id
    assert second.flags & FLAG_HOLD


def test_speed_override_is_used_and_preserved_across_hold():
    shaper = CommandShaper(CommandShaperConfig(hold_ms=200, base_conf_thresh=0.2))
    first = shaper.shape(
        action_id=1,
        finger_id=2,
        action_conf=0.9,
        speed_scalar_override=0.6,
        timestamp_stream_ms=1000,
        timebase_ms=1000,
    )
    second = shaper.shape(
        action_id=2,
        finger_id=3,
        action_conf=0.95,
        speed_scalar_override=0.2,
        timestamp_stream_ms=1100,
        timebase_ms=1100,
    )
    assert first.speed_scalar == 0.6
    assert second.flags & FLAG_HOLD
    assert second.action_id == first.action_id
    assert second.finger_id == first.finger_id
    assert second.speed_scalar == first.speed_scalar


def test_watchdog_trigger():
    shaper = CommandShaper(CommandShaperConfig(watchdog_ms=500))
    shaper.note_valid(timebase_ms=1000)
    assert shaper.watchdog_command(timebase_ms=1300) is None
    watchdog = shaper.watchdog_command(timebase_ms=1600)
    assert watchdog is not None
    assert watchdog.flags & FLAG_WATCHDOG
    assert watchdog.action_id == 0
