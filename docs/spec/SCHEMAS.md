# EEG Pipeline Schemas

This document defines canonical artifacts, filenames, and schemas used across the pipeline.

## 1) Raw EEG Stream (Step 1)

Primary archive path (per session):
- `Projects/<project>/subjects/<subject>/sessions/<session_id>/raw/eeg_raw_shard_*.npy`

Record fields (numpy dtype):
- `seq` (int) — sample sequence
- `lsl_ts_raw` (float) — raw LSL timestamp
- `lsl_ts_mono` (float) — monotonic LSL timestamp
- `local_ts` (float) — local wall clock
- `flags` (int) — integrity flags
- `segment_id` (int) — segment index
- `clamped` (int) — monotonic clamp indicator
- `sample` (float[C]) — Muse 2 channel samples

Sampling:
- Nominal 256 Hz (see `session_meta.json` for actual sampling rate)

## 2) Cleaned Feature Stream (Step 1)

Legacy per-session file (appendable):
- `data/processed/<subject_id>_<session_id>_eeg_features.csv` (legacy)

Columns:
- `lsl_timestamp` (float)
- `time_s` (float) — seconds since stream start
- `ch1`, `ch2`, `ch3`, `ch4` (float) — cleaned EEG sample (ICA + artifact attenuation)
- `pred_action` (int)
- `pred_finger` (int)
- `action_confidence` (float)
- `action_uncertainty` (float)
- `finger_confidence` (float)
- `finger_uncertainty` (float)
- `velocity` (float)
- `latency_ms` (float)

Notes:
- Legacy artifact used by older review/diagnostic tools.

## 3) Event Marking (Step 1 UI)

Authoritative per-session file:
- `Projects/<project>/subjects/<subject>/sessions/<session_id>/events/events.jsonl`

Legacy per-session files:
- `data/processed/<subject_id>_<session_id>_events_autosave.csv` (legacy)
- `data/processed/<subject_id>_<session_id>_events.csv` (legacy)

Required fields (events.jsonl payloads):
- `onset_s` (float)
- `duration_s` (float)
- `type` (string) — `artifact`, `calibration`, `rest`, or derived from `action_id` + `finger_id`
- `channel` (string)
- `confidence` (float or empty)
- `notes` (string)
- `finger_id` (int)
- `action_id` (int)
- `trial_id` (int, optional) — defaults to 0 when unavailable
- `block_id` (int, optional) — defaults to 0 when unavailable
- `session_mode` (string, optional) — `physical` or `imagery` if provided
- `source` (string)

## 4) Window Extraction Outputs (Step 1b)

Primary training artifact:
- `eeg_windows.npz`

Contents:
- `X` (float32) — shape `[N, T, C]`, where `T = 64`, `C = 4`
- `y_action` (int64) — shape `[N]`
- `y_finger` (int64) — shape `[N]`
- `subject_id` (string) — shape `[N]`
- `experiment_hash` (string) — shape `[N]`
- `window_start` (float32)
- `window_end` (float32)
- `confidence_hint` (float32)
- `artifact_flag` (int64)
- `session_mode` (string, optional)
- `trial_id` (int64, optional)
- `block_id` (int64, optional)
- `fs` (int64)
- `window_sec` (float32)
- `step_sec` (float32)
- `channel_names` (string array)

Diagnostic summary:
- `eeg_windows.csv` — columns `ch1..ch4` (window means), `action_id`, `finger_id`, `subject_id`, `experiment_hash`, `window_start`, `window_end`, `confidence_hint`, `artifact_flag`, `session_mode`, `trial_id`, `block_id`

## 5) Normalization

- `scaler.npz` — per-channel normalizer
  - dict with `mean` and `std` arrays of shape `[C]`
  - fit on training windows only

## 6) Model Artifacts

- `finger_action_model.pt` — PyTorch state dict for `CNNLSTMFingerActionNet`
  - deployable runs require:
    - `finger_head.weight` shape `[5, ...]` (active fingers only)
    - `action_head.weight` shape `[3, ...]`
    - `finger_applicability_head.weight` / `.bias` present
- `test_predictions.npz` — held-out predictions for reproducibility
  - required arrays:
    - `action_probs`
    - `finger_probs`
    - `y_action`
    - `y_finger`
    - `test_indices_local`
    - `test_indices_global`
    - `action_temperature`
    - `finger_temperature`
    - `applicability_temperature`
  - optional / run-dependent arrays:
    - `applicability_probs` — required for deployable runs with the applicability head
    - `window_start`
    - `window_end`
    - `dataset_info` — JSON payload stored as a string array for cache validation / provenance

## 7) Calibration + Logs

- `temperature_scaling.json` — post-hoc calibration metadata
  - `action_temperature` (float)
  - `finger_temperature` (float)
  - `applicability_temperature` (float)
  - `has_applicability_temperature` (bool)
  - `fit_sample_count` (int)
  - `fit_non_rest_count` (int)
  - `source` (string)
  - `metrics` (object)
- `logs/experiments/<experiment_hash>.json` — step-wise experiment log
- `logs/calibration/<subject_id>_<experiment_hash>.json` — per-subject calibration stream
- `logs/calibration/calibration_state_<subject>_<experiment>.json` — online threshold state
- `logs/session_state_<subject_id>.json` — append/resume state (session_id, block_id, segment_id, total_elapsed_s, last_time_s, features_path, events_path, raw_path, created_utc, updated_utc)
- `processed/live_infer*/predictions.jsonl` — Step 7 per-window live predictions
  - core fields:
    - `window_start_s`, `window_end_s`, `ts_utc`
    - `committed_action_id`, `committed_finger_id`
    - `raw_top_action_id`, `raw_top_finger_id`
    - `action_conf`, `finger_conf`, `joint_conf`
    - `committed_pair_valid`
  - gate / uncertainty fields:
    - `finger_gate_ok`
    - `finger_applicable_prob`
    - `applicability_gate_ok`
    - `uncertainty_gate_ok`
    - `action_uncertainty`
    - `finger_uncertainty`
    - `applicability_uncertainty` (if MC inference produced it)
  - actuation fields:
    - `actuation_sent`
    - `actuation_target_action_id`
    - `actuation_target_finger_id`
    - `actuation_suppressed_reason`
    - `actuation_latency_ms`
    - `actuation_speed_scalar`

## 8) Reports

- `reports/subjects/<subject_id>/<experiment_hash>/report.html`
- `reports/subjects/<subject_id>/<experiment_hash>/*.png`
- `reports/subjects/cross_subject_summary.png`

## 9) Session Metadata

- `session_meta.json` contains:
  - `subject_id`, `session_id`, `segment_id`, `experiment_hash`
  - `sampling_rate`, `window_sec`, `channels`
  - `features_path`, `events_path`, `raw_path`
