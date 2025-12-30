# EEG Pipeline Schemas

This document defines canonical artifacts, filenames, and schemas used across the pipeline.

## 1) Raw EEG Stream (Step 1)

Primary archive path (per session):
- `data/raw/<subject_id>_<experiment_hash>_raw.csv`

Columns:
- `lsl_timestamp` (float) — raw LSL timestamp
- `ch1`, `ch2`, `ch3`, `ch4` (float) — Muse 2 channels

Sampling:
- Nominal 256 Hz (see `session_meta.json` for actual sampling rate)

## 2) Cleaned Feature Stream (Step 1)

Working file (latest session):
- `eeg_features.csv`

Archived file (per session):
- `data/processed/<subject_id>_<experiment_hash>_eeg_features.csv`

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
- The stream is saved one sample per iteration after the buffer is full.
- This file is used by the event review UI.

## 3) Event Marking (Step 1 UI)

Working files:
- `events.csv`
- `events_autosave.csv`

Archive file:
- `data/processed/<subject_id>_<experiment_hash>_events.csv`

Required columns:
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

- `scaler.save` — per-channel normalizer
  - dict with `mean` and `std` arrays of shape `[C]`
  - fit on training windows only

## 6) Model Artifacts

- `finger_action_model.pt` — PyTorch state dict for `CNNLSTMFingerActionNet`
- `test_predictions.npz` — held-out predictions for reproducibility
  - `action_probs`, `finger_probs`, `y_action`, `y_finger`, `test_indices`

## 7) Calibration + Logs

- `logs/experiments/<experiment_hash>.json` — step-wise experiment log
- `logs/calibration/<subject_id>_<experiment_hash>.json` — per-subject calibration stream
- `logs/calibration/calibration_state_<subject>_<experiment>.json` — online threshold state

## 8) Reports

- `reports/subjects/<subject_id>/<experiment_hash>/report.html`
- `reports/subjects/<subject_id>/<experiment_hash>/*.png`
- `reports/subjects/cross_subject_summary.png`

## 9) Session Metadata

- `session_meta.json` contains:
  - `subject_id`, `experiment_hash`
  - `sampling_rate`, `window_sec`, `channels`
  - `features_path`, `events_path`, `raw_path`
