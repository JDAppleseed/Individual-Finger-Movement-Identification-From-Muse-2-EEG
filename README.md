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

## Muse Streaming CLI Quickstart

Use the new CLI to run the lightweight Muse → LSL → recorder pipeline (supports simulator mode):

1) Start the LSL streamer (BLE or simulator):
```
python -m cli start-streamer --sim
```

2) Run a healthcheck:
```
python -m cli healthcheck --sim --check-timebase
```

3) Record data:
```
python -m cli record --sim --output-dir data/muse_streaming --subject-id 8-M16
```

## Live Streaming (Desktop UI)

Launch the desktop UI and use the new live workflow (production path):

1) Open the desktop UI:
```
python eeglab_wrapper_ui.py
```
2) Click **Connect Muse 2** to start the internal BLE → LSL streamer and run a healthcheck.
3) If prompted about label/channel mismatches, review the warning and click **I understand** to proceed.
4) Click **Start Recording** to launch `1_stream_and_record.py` for live capture/inference.
5) If a hard stop triggers, a blocking modal will appear. Review the diagnostics and click **I understand**.
   The hard stop report is written under `logs/hard_stop_<subject>_<session>_<timestamp>.json`.

The UI gates Start Recording until the LSL stream is healthy (or operator-acknowledged).

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

## Timebase & Latency (absolute_v1)

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

### Timebase invariants

The pipeline enforces these invariants at runtime (warnings emitted on violation):

- **Single clock domain:** all comparisons and subtraction use LSL timestamps only.
- **absolute_v1:** `time_s := lsl_ts - stream_start_lsl_ts` for samples, features, and events.
- **Monotonic samples:** LSL timestamps are clamped on backward jumps.
- **Gap detection:** large gaps are flagged so windows/features can be skipped safely.
- **Latency definition:** `latency_ms := (lsl_now - lsl_ts) * 1000` (single LSL domain).

## Resume semantics (streaming CLI)

Resume is allowed **only** when required artifacts exist and are non-empty (features file at minimum).
If resume is requested but artifacts are missing/empty/corrupt:

- A **new session_id** and **new output paths** are created.
- Counters are reset.
- Existing event paths are **not** reused.
- If any output file already exists for a new session, a new unique session_id is generated.

## Muse streaming artifacts (CLI)

The CLI recorder writes under the output directory:

- `*_raw.csv`: raw samples
  - `time_s`, `lsl_ts`, `latency_ms`, `TP9`, `AF7`, `AF8`, `TP10`
- `*_features.csv`: fixed-size window features
  - `time_s` (window center), `lsl_ts`, `window_start_s`, `window_end_s`, `latency_ms`
  - `mean_*`, `std_*` per channel
- `*_events.csv`: session_start/session_end markers

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

## Demo backend (deprecated)

The `demo_backend/` FastAPI + Vite flow is deprecated and no longer in the run path.
It remains in the repo for reference only and will be removed once the desktop live flow
is fully validated. See `demo_backend/DEPRECATED.md`.

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
