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
