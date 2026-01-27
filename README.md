# EEG Finger Classification BCI (Muse 2)

Real-time EEG-based finger + action (rest/open/close) classification with uncertainty-aware gating, calibrated confidence, and experiment reporting. Pipeline is aligned to the SDS: streaming, event labeling, window extraction, training, evaluation, calibration, and reporting.

## Python support (macOS)

- Supported versions: Python 3.11 or 3.12.
- Common failure mode: torch import can hang on Python 3.13. Use the setup flow below.

## Setup (macOS)

One-command setup (recommended):
```
./scripts/setup_venv.sh
```

If you use pyenv:
```
pyenv install 3.11.7
pyenv local 3.11.7
./scripts/setup_venv.sh
source .venv/bin/activate
```

Diagnostic command:
```
python scripts/diagnose_env.py
```

## How to run (UI)

```
source .venv/bin/activate
python eeglab_wrapper_ui.py
```

## What changed / How to run

- Connect Muse: `python eeglab_wrapper_ui.py` → click **Connect Muse 2** (Step 0), or run `python muse_lsl_streamer.py --name Muse2-EEG`.
- Record: `python 1_stream_and_record.py --enable-plot --plot-scale fixed --plot-fixed-ylim -200 200` (outputs raw.csv + events.csv).
- Extract/Train/Evaluate: `python 1b_extract_windows.py --session-dir <session_dir>` → `python 2_train_model.py` → `python 3_evaluate_model.py`.
- Live infer: `python 7_live_infer_and_actuate.py --model-path models/finger_action_model.pt --scaler-path scaler.save --stream-name Muse2-EEG` (safe mode). Add `--enable-actuation --i-understand-this-moves-the-hand` to actuate.

## Quick Start

1) Install deps:
```
./scripts/setup_venv.sh
source .venv/bin/activate
```

2) Record-only (raw + events):
```
python 1_stream_and_record.py --enable-plot --plot-scale fixed --plot-fixed-ylim -200 200
```

3) Review + validate events (optional but recommended):
```
python 5_review_events.py
python 5_validate_events.py --apply
```

4) Validate the session, then extract windows:
```
python -m muse_streaming.validate_session --session <session_dir>
python 1b_extract_windows.py --session-dir <session_dir>
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
python muse_lsl_streamer.py --sim
```

2) List available streams:
```
python -m cli list-streams
```

3) Run a healthcheck:
```
python -m cli healthcheck --stream-name Muse2-EEG --sim --check-timebase
```

4) Record data:
```
python -m cli record --sim --output-dir data/muse_streaming --subject-id 8-M16
```

## Live Streaming (Desktop UI)

Launch the desktop UI and use the new live workflow (production path):

1) Open the desktop UI:
```
python eeglab_wrapper_ui.py
```
2) Use **Step 0: Connect Muse / LSL Status** to connect and verify LSL health.
3) Create a Project and Subject to get the correct metadata for the steps to pull from.
4) Run **Step 1: Record** to capture raw EEG + events to the session directory.
5) Validate the session, then extract windows, train, evaluate, and run live inference.

The UI gates Start Recording until the LSL stream is healthy (or operator-acknowledged).

## Event Labeling (live)

Event marking happens inside `1_stream_and_record.py` via keyboard:
- `space` / `1–5` / `o` / `c` / `r` record events (configurable via `EVENT_KEYMAP`)

Events are saved to `events.csv` during capture.

## Live Inference (Dedicated Script)

Run live inference/actuation in a separate process:

```
python 7_live_infer_and_actuate.py \
  --model-path models/finger_action_model.pt \
  --scaler-path scaler.save \
  --stream-name Muse2-EEG \
  --stream-type EEG
```

Allow-drop is opt-in and logs dropped windows:

```
python 7_live_infer_and_actuate.py --allow-drop --latency-policy drop
```

See `DATA_CONTRACT.md` for session layout and validation requirements.

Legacy script output directories can be customized via:
- `--processed-dir` and `--raw-dir` flags, or
- `MUSE_PROCESSED_DIR` / `MUSE_RAW_DIR` environment variables.

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

### Processed/Raw Output Resolution (Step 1)

`1_stream_and_record.py` determines `processed_dir` and `raw_dir` in this order:
1) CLI flags `--processed-dir` / `--raw-dir`
2) Config JSON keys `processed_dir` / `processed_path` / `output_dir` (processed) and `raw_dir` (raw)
3) If the config includes `project_name`, `subject_id`, and `session_id`, defaults to:
   - `Projects/<project>/subjects/<subject>/sessions/<session>/processed`
   - `Projects/<project>/subjects/<subject>/sessions/<session>/raw`
4) Final fallback: `data/processed/<subject>/<session_id>` and `data/raw/<subject>/<session_id>`

Example:
```
python 1_stream_and_record.py --config Projects/Test1/subjects/Har/config/step1.json
python 1_stream_and_record.py --config Projects/Test1/subjects/Har/config/step1.json --processed-dir data/processed
```

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

### Notes:
- Everything here is a preliminary overview that will be updated to be more accurate when I get around to it
- This project is usually constantly changing
- For a simple pipeline run-through, use the ui (eeglab_wrapper_ui.py) and stick to the defaults
- raw.csv and events.csv are for human observation, we save metadata in a different stack and that is what gets fed to the rest of the pipeline
- Don't break confidentiality on any subjects, thats a big No-No
- Please use this ethically and whenever possible credit me for the usage
- Don't sell this code to other people, thats mean
- More information can be found in the eeg_hand_lessons.tex, compile that for a full 50ish pages of lessons on how everything works, exercisies, answer keys, and diagrams shouold be there when compiled into a pdf through LaTex
- Research Paper soon with more documentation and science heavy specifics
- Any questions can be answered by me (Jonathan Davanzo) just send me over an email or whatever you can find or prefer
- For anyone wanting to try this for themselves, good luck, recommended minimum hardware is a new-ish macbook air with minimum 16gb ram, its not gonna crash or mess anything up, one issue we ran into with those specs was write speed for when we had a more heavy stack of csvs being written at hundreds of thousands of lines with maybe 10-20 columns depending on what point we look back at
- Csv writing was optimized and is minimal now, look for and stay vigilant with the health checks and logs, stuff can go wrong pretty quick and usually if you pause it and the resume it fixes itself and the hardware/software catches up
- oh you also need a Muse 2 or equivalent EEG, but that's probably pretty obvious
