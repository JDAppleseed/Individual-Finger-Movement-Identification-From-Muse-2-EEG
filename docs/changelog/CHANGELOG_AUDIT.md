# Pipeline Audit Change Log

## 2026-01-21

- muse_streaming/muse_lsl_streamer.py
  - Added simulator mode, buffer backpressure handling, and removed stray heartbeat code.
- muse_streaming/cli.py, cli.py, muse_streaming/recorder.py, muse_streaming/healthcheck.py
  - Introduced a unified CLI with stream/record/healthcheck/list commands, plus simulator support.
  - Added structured logging, timebase checks, and fixed-size resampling in the recorder.
  - Fixed stream listing output formatting and logger adapter handling.
- muse_streaming/config.py, muse_streaming/timebase.py, muse_streaming/io_paths.py, muse_streaming/resample.py
  - Centralized configuration, timebase invariants, resume-safe session paths, and resampling utilities.
- 1_stream_and_record.py
  - Hardened session ID uniqueness on new sessions and corrected inference latency timebase.
- README.md
  - Documented the CLI quickstart, timebase invariants, resume semantics, and artifact schemas.
  - Noted legacy output directory overrides via flags/environment.

## 2026-01-20

- muse_streaming/healthcheck.py
  - Added backward-compatible healthcheck parameters (`stream_name` + legacy `name`) with deprecation warnings.
  - Updated the standalone healthcheck CLI to expose `--stream-name` as the canonical flag.
- muse_streaming/cli.py
  - Added `--stream-name` alias support, auto-detect handling for unnamed healthchecks, and quote-stripping for list-streams output.
- eeglab_wrapper_ui.py
  - Switched the streamer launcher to `python muse_lsl_streamer.py` (canonical streamer path) and aligned defaults to shared config.
- scripts/smoke_live_pipeline.py
  - Updated healthcheck invocation to use `stream_name`.
- tests/test_muse_streaming_cli_healthcheck.py
  - Added CLI parsing, healthcheck compatibility, and list-streams formatting coverage.
- README.md
  - Documented canonical `list-streams` and `healthcheck --stream-name` usage.
- 1_stream_and_record.py
  - Added safe processed/raw directory resolution with resume-aware overrides and deterministic session seeds.
  - Wrapped the streaming loop with cleanup-on-error and streamer process teardown hooks.
- utils/output_paths.py, tests/test_output_paths.py
  - Centralized output directory resolution and added regression coverage for precedence rules.
- eeglab_wrapper_ui.py
  - Synced Step 1 output copy to use session_state paths when available.
- README.md
  - Documented processed/raw directory resolution rules and examples.

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
