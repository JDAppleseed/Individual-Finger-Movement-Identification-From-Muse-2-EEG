# Lossless Session Data Contract

This repository uses a **session directory** as the single source of truth for
lossless training capture. Raw EEG + events are the authoritative record; all
features/windows are derived offline.

## Session Layout

```
session_dir/
  meta.json
  manifest.json
  timebase_report.json
  raw/
    eeg_raw_shard_000.npy
    eeg_raw_shard_001.npy
  events/
    events.jsonl
```

## Raw Shards

Each shard is a NumPy structured array with:

* `seq`: global, monotonic, never resets
* `lsl_ts_raw`: original LSL timestamp
* `lsl_ts_mono`: monotonic LSL timestamp
* `local_ts`: local wall clock
* `flags`: bitfield (nonfinite samples, etc.)
* `segment_id`: segment counter
* `clamped`: whether the timestamp was clamped
* `sample`: EEG sample array (channels)

Shards are written via temp file + atomic rename to prevent partial writes.

## Manifest

`manifest.json` contains:

* `seq_min`, `seq_max`
* `expected_sample_count`, `actual_sample_count`
* `missing_seq_count`, `out_of_order_count`
* `termination_reason` (`normal`, `backpressure_abort`, `error`)
* `shard_list` with per-shard ranges
* `backpressure_max_duration_s`, `backpressure_total_duration_s`
* `max_queue_depth_observed`

## Timebase Report

`timebase_report.json` includes per-shard monotonic time ranges and is used
to audit timebase continuity.

## Events

`events/events.jsonl` is newline-delimited JSON. Each line is a single event.
Preferred fields (when available):

* `onset_s` (float) — session-relative onset (seconds)
* `duration_s` (float, optional) — duration in seconds (0.0 if unknown)
* `event_time_s` (float, legacy alias of `onset_s`)
* `type` / `label` (string) — event label (e.g., `thumb`, `index`, `rest`)
* `lsl_ts_mono` (float) — monotonic LSL timestamp
* `local_ts` (float) — local wall clock timestamp
* `metadata` (object, optional) — extra fields (finger/action IDs, notes, etc.)

If event marking is disabled or no labels are recorded, `events.jsonl` may be
empty; this is expected and should be reported in logs/diagnostics.

## Lossless Invariants

* **Training capture** (`train_record`) must never drop samples.
* Backpressure is enforced; if sustained, the run terminates with
  `termination_reason=backpressure_abort`.
* Offline window extraction must validate manifest continuity by default.

## Validation

Use:

```
python -m muse_streaming.validate_session --session <session_dir>
```

Add `--allow-partial` only when you explicitly accept incomplete sessions.
