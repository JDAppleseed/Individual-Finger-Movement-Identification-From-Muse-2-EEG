# Pipeline Audit Change Log

## 2026-01-21

- muse_streaming/muse_lsl_streamer.py
  - Added simulator mode, buffer backpressure handling, and removed stray heartbeat code.
- muse_streaming/cli.py, cli.py, muse_streaming/recorder.py, muse_streaming/healthcheck.py
  - Introduced a unified CLI with stream/record/healthcheck/list commands, plus simulator support.
  - Added structured logging, timebase checks, and fixed-size resampling in the recorder.
- muse_streaming/config.py, muse_streaming/timebase.py, muse_streaming/io_paths.py, muse_streaming/resample.py
  - Centralized configuration, timebase invariants, resume-safe session paths, and resampling utilities.
- 1_stream_and_record.py
  - Hardened session ID uniqueness on new sessions and corrected inference latency timebase.
- README.md
  - Documented the CLI quickstart, timebase invariants, resume semantics, and artifact schemas.

## 2025-12-29

- utils/report_generator.py:1-292
  - Rebuilt the report generator into a single, consistent implementation.
  - Removed duplicated blocks and undefined variables, and added safe handling for missing calibration data.
  - Ensures per-subject reports and cross-subject summary generate without runtime errors.

## 2025-12-30

- 1_stream_and_record.py
  - Switched to session-scoped feature/event/raw files with persisted session_state for append/resume.
  - Unified time_s with event clock, added monotonic guard, and enforced 4-channel LSL invariant.
  - Added block/segment tracking and end-of-stream state persistence.
- scripts/preflight_check.py
  - Resolved paths via session_meta.json and added alignment/monotonicity checks.
- SCHEMAS.md
  - Updated Step 1 artifact paths to use session_id and documented session_state metadata.

## 2025-12-30 (Step 1 timing hardening)

- 1_stream_and_record.py
  - Enforced session-scoped outputs, session_state persistence with paths and UTC timestamps.
  - Unified time_s to event clock with monotonic clamp and 4-channel mapping.
  - Ensured q/ESC/end_stream share the same clean shutdown path and block_id increment.
- scripts/preflight_check.py
  - Validates session_meta paths, monotonic time_s, row length consistency, and event alignment.
- SCHEMAS.md
  - Documented session_state fields and time base alignment.
