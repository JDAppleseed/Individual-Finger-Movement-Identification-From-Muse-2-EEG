from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from utils.palm_link import FLAG_HOLD, FLAG_THERMAL, FLAG_WATCHDOG


@dataclass
class ActuationCommand:
    action_id: int
    finger_id: int
    speed_scalar: float
    flags: int
    seq: int
    timestamp_stream_ms: int


@dataclass
class CommandShaperConfig:
    base_conf_thresh: float = 0.75
    speed_gamma: float = 1.0
    hold_ms: int = 150
    hold_conf_margin: float = 0.05
    watchdog_ms: int = 500
    thermal_limit_c: float = 42.0


class CommandShaper:
    def __init__(self, config: Optional[CommandShaperConfig] = None) -> None:
        self.config = config or CommandShaperConfig()
        self._seq = 0
        self._last_cmd: Optional[ActuationCommand] = None
        self._last_cmd_timebase_ms: Optional[int] = None
        self._last_valid_timebase_ms: Optional[int] = None
        self._hold_until_ms: int = 0

    def reset(self) -> None:
        self._seq = 0
        self._last_cmd = None
        self._last_cmd_timebase_ms = None
        self._last_valid_timebase_ms = None
        self._hold_until_ms = 0

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq

    def _confidence_to_speed(self, conf: float) -> float:
        base = float(self.config.base_conf_thresh)
        if conf < base:
            return 0.0
        speed = (conf - base) / max(1e-6, 1.0 - base)
        speed = max(0.0, min(1.0, speed))
        gamma = float(self.config.speed_gamma)
        if gamma != 1.0:
            speed = speed**gamma
        return speed

    def confidence_to_speed(self, conf: float) -> float:
        return self._confidence_to_speed(conf)

    def note_valid(self, timebase_ms: Optional[int] = None) -> None:
        if timebase_ms is None:
            return
        self._last_valid_timebase_ms = int(timebase_ms)

    def shape(
        self,
        action_id: int,
        finger_id: int,
        action_conf: float,
        timestamp_stream_ms: int,
        speed_scalar_override: Optional[float] = None,
        stability_ok: bool = True,
        timebase_ms: Optional[int] = None,
        thermal_c: Optional[float] = None,
    ) -> ActuationCommand:
        _ = stability_ok  # reserved for future safety hooks
        timebase_ms = (
            int(timestamp_stream_ms) if timebase_ms is None else int(timebase_ms)
        )
        self._last_valid_timebase_ms = timebase_ms

        target_action = int(action_id)
        target_finger = int(finger_id)
        conf = float(action_conf)

        # Confidence gating + speed map
        if conf < float(self.config.base_conf_thresh):
            target_action = 0
            target_finger = 0
            speed = 0.0
        else:
            if speed_scalar_override is None:
                speed = self._confidence_to_speed(conf)
            else:
                speed = max(0.0, min(1.0, float(speed_scalar_override)))
            if target_action == 0:
                target_finger = 0
                speed = 0.0

        previous_effective_cmd: Optional[ActuationCommand] = None
        if (
            self._last_cmd is not None
            and int(self._last_cmd.action_id) != 0
            and int(self._last_cmd.finger_id) != 0
        ):
            previous_effective_cmd = self._last_cmd

        flags = 0
        hold_requested = False
        if not bool(stability_ok):
            if previous_effective_cmd is not None:
                hold_requested = True
                self._hold_until_ms = max(
                    self._hold_until_ms, timebase_ms + int(self.config.hold_ms)
                )
            else:
                target_action = 0
                target_finger = 0
                speed = 0.0
        if previous_effective_cmd is not None:
            changed = (
                target_action != previous_effective_cmd.action_id
                or target_finger != previous_effective_cmd.finger_id
            )
            near_thresh = conf < float(self.config.base_conf_thresh) and conf >= (
                float(self.config.base_conf_thresh)
                - float(self.config.hold_conf_margin)
            )
            if changed:
                last_delta = None
                if self._last_cmd_timebase_ms is not None:
                    last_delta = timebase_ms - self._last_cmd_timebase_ms
                if last_delta is None or last_delta < int(self.config.hold_ms):
                    hold_requested = True
                    self._hold_until_ms = max(
                        self._hold_until_ms, timebase_ms + int(self.config.hold_ms)
                    )
            if near_thresh:
                hold_requested = True
                self._hold_until_ms = max(
                    self._hold_until_ms, timebase_ms + int(self.config.hold_ms)
                )

        if (
            hold_requested
            and previous_effective_cmd is not None
            and timebase_ms < self._hold_until_ms
        ):
            flags |= FLAG_HOLD
            target_action = previous_effective_cmd.action_id
            target_finger = previous_effective_cmd.finger_id
            speed = previous_effective_cmd.speed_scalar

        if thermal_c is not None and thermal_c >= float(self.config.thermal_limit_c):
            flags |= FLAG_THERMAL
            target_action = 0
            target_finger = 0
            speed = 0.0

        cmd = ActuationCommand(
            action_id=target_action,
            finger_id=target_finger,
            speed_scalar=float(speed),
            flags=flags,
            seq=self._next_seq(),
            timestamp_stream_ms=int(timestamp_stream_ms),
        )
        self._last_cmd = cmd
        self._last_cmd_timebase_ms = timebase_ms
        return cmd

    def watchdog_command(
        self, timebase_ms: Optional[int] = None
    ) -> Optional[ActuationCommand]:
        if timebase_ms is None:
            return None
        now_ms = int(timebase_ms)
        if self._last_valid_timebase_ms is None:
            return None
        if (now_ms - self._last_valid_timebase_ms) < int(self.config.watchdog_ms):
            return None
        cmd = ActuationCommand(
            action_id=0,
            finger_id=0,
            speed_scalar=0.0,
            flags=FLAG_WATCHDOG,
            seq=self._next_seq(),
            timestamp_stream_ms=now_ms,
        )
        self._last_cmd = cmd
        self._last_cmd_timebase_ms = now_ms
        return cmd
