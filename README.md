# EEG Finger Classification BCI (Muse 2)

Real-time EEG-based finger + action (rest/open/close) classification with uncertainty-aware gating, calibrated confidence, and experiment reporting. Pipeline is aligned to the SDS: streaming, event labeling, window extraction, training, evaluation, calibration, and reporting.

## Quick Start

1) Install deps:
```
python -m pip install -r requirements.txt
```

2) Stream + label:
```
python 1_stream_and_record.py
```

3) Review + validate events (optional but recommended):
```
python 5_review_events.py
python 5_validate_events.py --apply
```

4) Extract windows:
```
python 1b_extract_windows.py  # outputs eeg_windows.npz + eeg_windows.csv
```

5) Train + evaluate:
```
python 2_train_model.py
python 3_evaluate_model.py
python 3b_deepchecks_evaluate.py
python 3c_live_paper_figures.py
```

6) Reports:
```
python 4_generate_reports.py
```

You can also use `run_all.py` to orchestrate the full pipeline.

## Event Labeling (live)

Event marking happens inside `1_stream_and_record.py` via keyboard:
- Hold `Space` = start event, release `Space` = end event
- `o` = OPEN mode, `c` = CLOSE mode, `r` = REST mode
- `a` = artifact override, `k` = calibration override, `n` = clear override
- `0–5` = assign finger to most recent event (0 = NONE)

Events are saved to `events_autosave.csv` during capture and `events.csv` on exit.

## Label Schema

Action head (required):
- 0 = REST
- 1 = OPEN
- 2 = CLOSE

Finger head (conditional; only valid if action != REST):
- 0 = NONE
- 1 = THUMB
- 2 = INDEX
- 3 = MIDDLE
- 4 = RING
- 5 = PINKY

Validity rules:
- REST + NONE is valid
- REST + any finger is invalid
- OPEN/CLOSE + NONE is invalid
- OPEN/CLOSE + finger 1–5 is valid

During training, finger loss is masked when action == REST.

## Data Artifacts

- `data/raw/*.csv` raw EEG
- `eeg_features.csv` streamed feature frames
- `events.csv` event annotations
- `eeg_windows.npz` sequence window dataset (primary)
- `eeg_windows.csv` window summary (diagnostics)
- `scaler.save` (per-channel normalizer), `finger_action_model.pt`
- `logs/experiments/*.json` experiment logs
- `logs/calibration/*` calibration traces
- `reports/subjects/*` HTML + figures

## absolute_v1 timebase

All sessions use a single LSL-aligned timebase (`absolute_v1`).

Features CSV:
- `lsl_timestamp`: absolute LSL timestamp for each feature row (seconds, LSL domain)
- `time_s`: relative seconds since stream start
- `time_s = lsl_timestamp - stream_start_lsl_ts`

Events CSV:
- `onset_lsl`, `onset_s`, `duration_s`, `end_lsl`, `end_s`
- `onset_s = onset_lsl - stream_start_lsl_ts`
- `end_s = end_lsl - stream_start_lsl_ts`

A per-session metadata JSON is written to `data/processed/*_session_meta.json` with:
`timebase_version`, `stream_start_lsl_ts`, `local_clock_at_start`, `clock_offset`, and output paths.

Resume gating:
- Resume is allowed only if the subject matches and the existing features file has data rows.
- Use `--init-only` to preview the resume decision and resolved paths without writing files.
- Use `--force-new-session` to always start fresh.

Validate alignment after a run:
```
python tools/check_time_alignment.py --features data/processed/<subject>_<session>_eeg_features.csv --events data/processed/<subject>_<session>_events.csv
```

Legacy data may fail strict alignment checks. For extraction, you can override with:
```
python 1b_extract_windows.py --ignore-misalignment
```

Resampled extraction (fixed shape windows):
```
python 1b_extract_windows.py --target-fs 256
```

## Live Sliding-Window Inference (demo_backend)

The demo backend consumes LSL samples into fixed windows (default `window_sec=0.25`) and advances
by hop (`step_sec`, default `0.05`). Inference runs on every hop as soon as samples arrive.

Publishing and actuation are decoupled:
- `publish_hz` throttles websocket tick updates.
- `actuation_hz` throttles palm-controller packets.

Timebases:
- Stream timebase (`absolute_v1`) comes from LSL timestamps and is used for `window_start_s`, `window_end_s`, and actuation packet timestamps.
- Perf time (`time.perf_counter`) is used only for pacing/profiling and never for dataset time or actuation timing.

Backlog handling:
- `backlog_strategy="all"` processes every window in order.
- `backlog_strategy="latest"` drops intermediate windows to reduce latency.

Example control payload (partial):
```json
{
  "mode": "live",
  "publish_hz": 20,
  "actuation_hz": 20,
  "backlog_strategy": "latest",
  "fast_mode": false,
  "enable_actuation": true,
  "link_type": "udp",
  "udp_host": "127.0.0.1",
  "udp_port": 9010,
  "speed_gamma": 1.0,
  "hold_ms": 150,
  "watchdog_ms": 500
}
```

Actuation links:
- `link_type="null"` (default): no-op
- `link_type="udp"`: UDP sim link (2.4GHz radio stand-in)
- `link_type="serial"`: UART link (`serial_port`, `serial_baud`)

Find the best available evaluation run:
```
python tools/find_best_eval_run.py
```

## Notes

- Muse 2 sampling rate defaults to 256 Hz in code.
- `pynput` is required for live event marking.
- `5_validate_events.py` enforces validity and can auto-repair with `--apply`.

## EEGLAB-Style GUI Wrapper

Launch the operator-friendly GUI:
```
python eeglab_wrapper_ui.py
```

The GUI creates projects under `Projects/<ProjectName>/subjects/<subject_id>/` and writes per-step configs
to `config/` plus a session snapshot in `sessions/<session_id>/session_config.json`.

Packaging hint (PyInstaller):
```
pyinstaller --noconfirm --onefile --windowed eeglab_wrapper_ui.py
```

### Smoke Test Checklist
- create project + subject
- detect LSL / offline CSV
- run step1 dry run
- launch event marker
- confirm config JSON written
- confirm session folder created
