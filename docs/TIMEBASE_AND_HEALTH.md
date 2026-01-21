# Timebase, Backlog, and Console Guardrails

This document explains how the streaming pipeline keeps time and health checks consistent while protecting real-time behavior.

## Canonical monotonic time domain

The pipeline now uses **one continuous monotonic timebase** derived from `time.monotonic()` for windowing, latency, and feature timestamps:

- **`lsl_ts_raw`**: Raw LSL timestamps from the inlet (unchanged).
- **`lsl_ts_mono`**: Raw LSL timestamps mapped into the monotonic domain using an offset (`lsl_ts_raw + offset`).
- **`time_s`**: Stream-relative seconds in the monotonic domain (`continuous_mono - stream_start_continuous_mono`).

The stream start anchor is captured from the first mapped monotonic sample, so `time_s` never jumps backward.

## Discontinuity detection and handling

LSL timestamps can jump due to reconnects, OS sleep, or stream restarts. The mapper detects discontinuities when:

- The LSL timestamp moves **backward by > 0.050s**, or
- The LSL timestamp jumps **forward by > 1.0s**.

On discontinuity:

- A new anchor offset is calculated.
- A new segment is started with reason `timebase_discontinuity`.
- The continuous monotonic time is clamped to remain strictly increasing.

This ensures that `time_s` is **non-decreasing** and latency calculations remain bounded.

## Backlog handling

When queues back up:

- **Oldest packets are dropped first** to preserve real-time behavior.
- The processing loop **throttles plotting** and **reduces feature-write frequency** while backlog is high.
- The raw backlog health policy uses warning and failure thresholds:
  - **Warn**: > 80% of max for the grace period (still healthy).
  - **Fail**: > 100% of max for the grace period (unhealthy).

This prevents backlog spirals from degrading latency or UI responsiveness.

## Console output guardrails

Console output is capped to prevent runaway logs from degrading UI performance:

- A **global character budget** caps emitted console output for the process lifetime.
- A **ring buffer** tracks the most recent console output for UI tail views.
- When the cap is hit, a **single warning line** is emitted:
  - `[console] output capped; further console output suppressed (cap=<CAP_CHARS>)`

Rate-limited logging helpers further reduce repeated log spam.

## Tuning

Key defaults (override in config or constants):

- `CONSOLE_CAP_CHARS`: total console character budget (default 200_000)
- `CONSOLE_RING_CHARS`: ring buffer size (default 50_000)
- `PROCESSING_BACKLOG_HIGH_WATERMARK_FRAC`: backlog threshold for throttling (default 0.8)
- `BACKLOG_FEATURE_WRITE_STRIDE`: write every Nth frame while backlogged (default 3)
- `FEATURE_WRITE_BATCH_MAX` / `FEATURE_WRITE_BATCH_INTERVAL_S`: feature write batching
