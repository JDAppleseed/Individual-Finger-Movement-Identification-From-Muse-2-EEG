# EEG Finger + Action Classification (Muse 2)

Real-time EEG-based finger + action (REST/OPEN/CLOSE) classification with uncertainty-aware gating, calibrated confidence, and reporting. The primary pipeline uses a **session directory** with lossless raw shards so all downstream steps are reproducible.

## Requirements

- Python 3.11 (enforced by `scripts/setup_venv.sh`).
- macOS or Linux (tested); Windows is unverified.
- Muse 2 or any LSL EEG stream with 4 channels for live recording.
- Optional: LaTeX for PDF reports (see `SYSTEM_DEPS.md`).
- Optional: `pyserial` for hardware actuation (`pip install pyserial`).

## Setup (macOS/Linux)

One-command setup:

```bash
./scripts/setup_venv.sh
```

If you use pyenv:

```bash
pyenv install 3.11.7
pyenv local 3.11.7
./scripts/setup_venv.sh
source .venv/bin/activate
```

Diagnostics:

```bash
python scripts/diagnose_env.py
```

## Quick Start (UI)

```bash
source .venv/bin/activate
python eeglab_wrapper_ui.py
```

Suggested UI flow:

1. Connect an LSL stream (Muse 2 BLE → LSL, or another LSL source).
2. Run **Step 1: Record** to create a session directory with raw shards + events.
3. Validate the session, extract windows, train, evaluate, and generate reports.

The UI writes per-step configs under `Projects/<ProjectName>/subjects/<subject_id>/config/`
and snapshots each step to `Projects/<ProjectName>/subjects/<subject_id>/sessions/<session_id>/session_config.json`.

## Session Directory (Core Concept)

Canonical session dirs live under:

```
Projects/<project>/subjects/<subject_id>/sessions/<subject_id>_<session_id>/
```

Notes:

- `session_id` is **auto-prefixed** with `subject_id` if it is not already.
- Step 1 creates `meta.json`, `manifest.json`, `timebase_report.json`, and `events/events.jsonl` **before** LSL resolution so partial runs still leave a durable trail.
- All downstream steps should be run with `--session-dir` for deterministic resolution.

## Pipeline Overview (Session-Dir Flow)

- Step 1: Stream & Record → creates a new session directory.
- Step 1b: Extract Windows → reads `<session_dir>/raw/` + `<session_dir>/events/`, writes `<session_dir>/processed/`.
- Step 2: Train Model → reads `<session_dir>/processed/eeg_windows.npz`, writes `<session_dir>/processed/models/<run_id>/`.
- Step 3+: Evaluate / Figures / Reports → read from the same session directory and the latest model run.
- Step 7: Live Infer + Actuate → uses the latest model/scaler unless explicitly overridden.

## CLI Run (Session-Dir Flow)

Record (Step 1):

```bash
python 1_stream_and_record.py --enable-plot --plot-scale fixed --plot-fixed-ylim -200 200
```

Validate + extract (Step 1b):

```bash
python -m muse_streaming.validate_session --session <session_dir>
python 1b_extract_windows.py --session-dir <session_dir>
```

Train + evaluate + reports:

```bash
python 2_train_model.py --session-dir <session_dir>
python 3_evaluate_model.py --session-dir <session_dir>
python 3b_deepchecks_evaluate.py --session-dir <session_dir>
python 3c_live_paper_figures.py --session-dir <session_dir>
python 4_generate_reports.py --session-dir <session_dir>
```

Optional flags:

- Use `--run-dir` with Steps 2–4 to target a specific model run.
- Use `--allow-partial` with validation/extraction only if you accept incomplete sessions.
- Use `--ignore-misalignment` in Step 1b to force extraction on legacy/misaligned data.

## Review + Validate Events

```bash
python 5_review_events.py --session-dir <session_dir>
python 5_validate_events.py --session-dir <session_dir> --apply
```

## Merging Two Sessions (Same Subject)

1) Extract windows for each session:

```bash
python 1b_extract_windows.py --session-dir <session_dir_1>
python 1b_extract_windows.py --session-dir <session_dir_2>
```

2) Merge the NPZs:

```bash
python tools/merge_windows_npz.py \
  --npz <session1>/processed/eeg_windows.npz \
  --npz <session2>/processed/eeg_windows.npz \
  --out <combined_session>/processed/eeg_windows.npz
```

3) Train/evaluate/report on the combined session:

```bash
python 2_train_model.py --session-dir <combined_session>
python 3_evaluate_model.py --session-dir <combined_session>
python 4_generate_reports.py --session-dir <combined_session>
```

The merge tool enforces matching window shapes and will error on incompatible metadata.

## Event Labeling (Live)

Default keymap (configurable via `--event-keymap` or config JSON):

```
space:mark,1:thumb,2:index,3:middle,4:ring,5:pinky,o:open,c:close,r:rest
```

Usage:

- Press `o`/`c`/`r` to select action (OPEN/CLOSE/REST).
- Press `1–5` to select finger.
- Hold `space` to mark an event (release to end the event).

Events are written to `events/events.jsonl` during capture (authoritative). `events.csv` is optional and for inspection only.

## Live Inference (Step 7)

Preferred (auto-resolve latest model/scaler from session dir):

```bash
python 7_live_infer_and_actuate.py \
  --config Projects/<project>/subjects/<subject>/config/infer.json \
  --session-dir <session_dir>
```

Explicit override (no session dir):

```bash
python 7_live_infer_and_actuate.py --config <infer_config.json>
```

When `--session-dir` is omitted, `model_path`, `scaler_path`, and `out_dir` must be present in the config JSON.

Actuation (optional):

```bash
python 7_live_infer_and_actuate.py \
  --config <infer_config.json> \
  --session-dir <session_dir> \
  --enable_actuation \
  --serial_port /dev/tty.usbmodemXXXX
```

Notes:

- `--serial_port` is required when `--enable_actuation` is set.
- Actuation is safety-gated; REST/NONE never actuate.
- See `docs/actuation_requirements.md` for design constraints.

## Muse Streaming CLI (Legacy CSV Pipeline)

This lightweight CLI records legacy CSVs under `data/` (separate from the session-dir pipeline).

Start a BLE → LSL streamer:

```bash
python -m cli start-streamer
```

List streams:

```bash
python -m cli list-streams
```

Healthcheck:

```bash
python -m cli healthcheck --stream-name Muse2-EEG --check-timebase
```

Record (legacy CSVs):

```bash
python -m cli record --output-dir data --subject-id <subject_id>
```

Legacy artifacts:

- `data/<subject>_<session>_raw.csv`
- `data/<subject>_<session>_features.csv` (only when features are written)
- `data/<subject>_<session>_events.csv`

Resume behavior:

- Resume is only allowed when a non-empty features file exists (legacy behavior).
- The default `record` mode writes raw + events only, so resume typically starts a new session.

## Label Schema

Action head:

- `0` = REST
- `1` = OPEN
- `2` = CLOSE

Finger head (only valid when action != REST):

- `0` = NONE
- `1` = THUMB
- `2` = INDEX
- `3` = MIDDLE
- `4` = RING
- `5` = PINKY

Validity rules:

- REST + NONE is valid
- REST + any finger is invalid
- OPEN/CLOSE + NONE is valid (whole-hand open/close)
- OPEN/CLOSE + finger 1–5 is valid

During training, finger loss is masked when action == REST.

## Data Artifacts (Session-Dir)

Canonical outputs:

- Raw EEG shards: `<session_dir>/raw/eeg_raw_shard_*.npy`
- Events: `<session_dir>/events/events.jsonl`
- Windowed data: `<session_dir>/processed/eeg_windows.(csv|npz)`
- Models: `<session_dir>/processed/models/<run_id>/`
- Reports: `<session_dir>/processed/reports/<run_id>/`

Supporting files:

- `manifest.json`, `meta.json`, `timebase_report.json`
- `raw/raw.csv` and `events/events.csv` (legacy inspection)
- `logs/step1.log`, `logs/resolved_settings.json`

See `DATA_CONTRACT.md` and `SCHEMAS.md` for detailed schemas.

## Timebase & Latency

All sessions use a single LSL-aligned timebase (`absolute_v1`).

Key invariants:

- Single clock domain (LSL timestamps only).
- `time_s := lsl_ts - stream_start_lsl_ts` for samples, features, and events.
- LSL timestamps are clamped on backward jumps.
- Latency is `latency_ms := (lsl_now - lsl_ts) * 1000`.

More detail: `docs/TIMEBASE_AND_HEALTH.md`.

## Smoke Test (No Muse Required)

```bash
python scripts/smoke_step1_record.py
python scripts/smoke_step1_record.py --full
```

The `--full` option runs Steps 2–4 after the mock recording (requires training dependencies).

## Troubleshooting

- If `pylsl` cannot find the LSL binary library on macOS, install it via Homebrew:
  - `brew install labstreaminglayer/tap/lsl`
  - `PYLSL_LIB=/opt/homebrew/Frameworks/lsl.framework/lsl`

## Papers & Figures

- `research_paper.pdf` / `research_paper.tex`
- `mc_dropout_uncertainty.pdf` / `mc_dropout_uncertainty.tex`

## License & Ethics

This repository is **proprietary**. See `LICENSE` for usage restrictions and authorization requirements.

Handle subject data confidentially and use the system responsibly.
