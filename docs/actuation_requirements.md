# Actuation and Timestamp Requirements (Design Spec Rev A)

Source: internal Design Spec Rev A PDF (not committed in this repo)

## MUST requirements (backend actuation/timestamps)

- "Control: Python high-level intent; real-time palm controller generates PWM and enforces safety." (p.1)
- "Python/PyTorch inference ... generates finger ID + open/close command and a speed scalar based on confidence." (p.7)
- "Confidence gating: commands engage only when confidence >= 75%; speed increases with confidence." (p.7)
- "Comms: low-latency 2.4 GHz link transmits compact packets (finger, direction, speed, timestamp)." (p.7)
- "Interface: UART or SPI to palm controller and/or compute module, with timestamped commands." (p.5)
- "Safety: soft limits, command ramping, watchdog, and 'hold mode' to avoid constant hunting." (p.5)
- "Servo bus: regulated 8.0 V constant-performance rail." (p.1)
- "Thermal: ... skin-contact surfaces target <= 40-42 C." (p.1)
- "Thermal: ... fail-safe to reduce compute power or disable actuation on over-temp or pump failure." (p.8)

## Operational clarification

- Actuation cooldown is a per-finger actuator hold interval, not a global hand lockout.
- A finger should complete the last commanded open/close endpoint before any rest command or timeout can return it to rest.
- Return-to-rest is owned by the palm controller per finger; UI previews should not issue a full-hand rest before starting a different active finger.
- Commands to other fingers may be emitted during that interval, subject only to serial throughput and the target finger's own cooldown.
- The Arduino receiver must enforce the same per-finger hold behavior. Its servo updates should be non-blocking so one finger's movement does not prevent the controller from accepting another finger command.
- The current host default allows up to 20 serial commands/s, matching the 50 ms live inference hop. The model gate remains more conservative than that ceiling; the serial ceiling should not be interpreted as a target actuation rate.
- Window-level would-send recall is not an accuracy metric. It is affected by hop size, stability gates, duplicate suppression, cooldown, and whether repeated windows should generate repeated hardware commands. Report event-level hit rate and first-command latency alongside window-level send recall for usability claims.
