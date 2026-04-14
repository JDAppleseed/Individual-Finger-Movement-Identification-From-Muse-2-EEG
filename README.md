# EEG Finger + Action Classification (Muse 2)

This repository supports subject-level EEG recording, window extraction, model training, evaluation, reporting, and Step 7 live inference for finger + action classification (`REST`, `OPEN`, `CLOSE`).

The normal operator path is the UI in `eeglab_wrapper_ui.py`. The CLI remains available for advanced, manual, or automated runs.

## Who This Is For

Use this repo if you need to:

- record lossless EEG sessions from a Muse 2 or another 4-channel LSL EEG stream
- train and evaluate a subject/session/model/scaler deployment bundle
- run Step 7 live inference, with optional actuation
- review saved live predictions after a run

## Recommended Workflow

Use the UI unless you specifically need CLI or manual control.

Launch:

```bash
source .venv/bin/activate
python eeglab_wrapper_ui.py
```

The UI is the recommended path for:

- stream setup
- session recording
- extraction, training, evaluation, and reports
- Step 7 launch/preflight
- Step 7b live-review workflows

## Requirements

- Python 3.11
- macOS or Linux
- Muse 2 or another LSL EEG stream with 4 channels for live recording
- `requirements.txt` installed in a virtual environment
- optional: LaTeX for PDF reports; see `docs/ops/SYSTEM_DEPS.md`
- optional: Arduino/serial hardware for Step 7 actuation

## Quick Start

Set up the environment:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
python scripts/diagnose_env.py
```

Start the UI:

```bash
python eeglab_wrapper_ui.py
```

Important connection note:

- Step 1 and Step 7 read from an LSL EEG stream.
- They do not talk to the Muse over BLE directly.
- For Muse 2, create the LSL stream first, either with the UI's Muse Connector or another BLE -> LSL bridge.

## Core UI Workflow

1. Open the UI and choose the project and subject.
2. On `Stream Setup`, connect the Muse or select the existing LSL stream.
3. Run `Step 1: Record (Lossless)` to create a session directory with raw shards and events.
4. Use `Validate Session` to confirm the session looks healthy.
5. Run `Step 1b: Extract Windows`.
6. Run `Step 2: Train Model`.
7. Run `Step 3+` for evaluation, Deepchecks, figures, and reports.
8. Run `Step 7: Live Infer + Actuate` when you are ready for live inference.
9. Run `Step 7b: Review Live Predictions` to summarize saved predictions and export review CSVs.

Step 7 UI notes:

- The Step 7 `Session dir` field is the authoritative launch and preflight session.
- The main session selector only seeds that field when it is blank.
- In Step 7, `session_dir` resolves the deployment bundle and output location. It is not the live EEG source.
- For decisive runs, use a fresh `processed/live_infer_<run_tag>` output directory.

Labeling rules:

- `REST` must pair with finger `NONE`.
- `OPEN` and `CLOSE` must pair with an active finger.
- `Step 1b` fails fast on `OPEN/CLOSE + NONE`, so fix or prune those events before extraction.

## Optional CLI Flow

Use the CLI when you need manual control, scripting, or automation.

### Record And Prepare A Session

If you need a BLE -> LSL bridge outside the UI:

```bash
python -m cli start-streamer
```

Record a new lossless session:

```bash
python 1_stream_and_record.py --enable-plot --plot-scale fixed --plot-fixed-ylim -200 200
```

Validate and extract:

```bash
python -m muse_streaming.validate_session --session <session_dir>
python 1b_extract_windows.py --session-dir <session_dir>
```

Review or repair events if needed:

```bash
python 5_review_events.py --session-dir <session_dir>
python 5_validate_events.py --session-dir <session_dir> --apply
```

### Train, Evaluate, And Report

```bash
python 2_train_model.py --session-dir <session_dir>
python 3_evaluate_model.py --session-dir <session_dir>
python 3b_deepchecks_evaluate.py --session-dir <session_dir>
python 3c_live_paper_figures.py --session-dir <session_dir>
python 4_generate_reports.py --session-dir <session_dir>
```

### Step 7 Live Inference

For decisive live runs, follow `docs/ops/STEP7_LIVE_RUNBOOK.md`.

- Canonical Step 7 config: `Projects/<ProjectName>/subjects/<subject_id>/winning_model/configs/infer.json`
- Archived legacy Step 7 artifacts: `Projects/<ProjectName>/subjects/<subject_id>/archive/step7/`

Manual preflight:

```bash
python tools/live_preflight.py \
  --config <infer_config.json> \
  --session-dir <session_dir> \
  --out-dir <session_dir>/processed/live_infer_<run_tag> \
  --probe-stream \
  --probe-distribution
```

Manual launch:

```bash
python 7_live_infer_and_actuate.py \
  --config <infer_config.json> \
  --session-dir <session_dir> \
  --out-dir <session_dir>/processed/live_infer_<run_tag>
```

Optional actuation:

```bash
python 7_live_infer_and_actuate.py \
  --config <infer_config.json> \
  --session-dir <session_dir> \
  --enable_actuation
```

Step 7 notes:

- `session_dir` selects the deployment bundle and output location, not the live stream.
- When `--enable_actuation` is set and `--serial_port` is omitted, Step 7 auto-detects a likely Arduino serial port.
- If multiple serial devices are attached, pass `--serial_port` explicitly.
- Leave file output enabled for decisive runs. `no_file_io=true` suppresses required evidence.

### Step 7b Review

Summarize the latest saved live predictions under a session:

```bash
python tools/analyze_live_predictions.py --session-dir <session_dir>
```

Use an explicit log path if needed:

```bash
python tools/analyze_live_predictions.py \
  --pred-log <session_dir>/processed/live_infer*/predictions.jsonl \
  --out-json <summary.json> \
  --segments-csv <predicted_segments.csv> \
  --review-csv <predicted_segments_review.csv>
```

### Legacy Muse Streaming CLI

The legacy Muse streaming CLI writes CSV artifacts under `data/`. It is separate from the session-directory pipeline.

Common commands:

```bash
python -m cli list-streams
python -m cli healthcheck --stream-name Muse2-EEG --check-timebase
python -m cli record --output-dir data --subject-id <subject_id>
```

## Outputs And Artifacts

Canonical session directories live under:

```text
Projects/<project>/subjects/<subject_id>/sessions/<subject_id>_<session_id>/
```

Notes:

- If needed, `session_id` is auto-prefixed with `subject_id`.
- The UI writes per-subject configs under `Projects/<ProjectName>/subjects/<subject_id>/config/`.
- The UI snapshots each session launch to `sessions/<session_id>/session_config.json`.

Common outputs:

- raw EEG shards: `<session_dir>/raw/eeg_raw_shard_*.npy`
- event log: `<session_dir>/events/events.jsonl`
- extracted windows: `<session_dir>/processed/eeg_windows.npz`
- model runs: `<session_dir>/processed/models/<run_id>/`
- reports: `<session_dir>/processed/reports/<run_id>/`
- Step 7 live outputs: `<session_dir>/processed/live_infer_<run_tag>/`

Deployment bundle:

- `finger_action_model.pt`
- `scaler.npz`
- `temperature_scaling.json`

Step 7b prefers fresh `live_infer_<run_tag>` outputs over bare or legacy `live_infer` directories.

## Troubleshooting

- No live EEG stream found:
  Start or select an LSL stream first. Step 1 and Step 7 require LSL input.
- Step 1b rejects labels:
  Check for `OPEN/CLOSE + NONE` events and correct or remove them.
- Step 7 is pointing at the wrong session:
  Use the Step 7 `Session dir` field. That field controls Step 7 launch and preflight.
- Step 7 output directory already exists:
  Use a fresh `processed/live_infer_<run_tag>` directory for decisive runs.
- macOS `pylsl` library errors:
  Run `brew install labstreaminglayer/tap/lsl` and set `PYLSL_LIB=/opt/homebrew/Frameworks/lsl.framework/lsl`.

## Developer Notes

- `docs/ops/STEP7_LIVE_RUNBOOK.md` covers the decisive Step 7 manual run path.
- `docs/ops/SYSTEM_DEPS.md` lists optional system packages for PDF/report generation.
- For a decisive Step 7 run, use the active subject's `Projects/<ProjectName>/subjects/<subject_id>/winning_model/configs/infer.json` as the canonical CLI config. The matching `Projects/<ProjectName>/subjects/<subject_id>/config/infer.json` file is the UI working mirror.
- Checked-in configs under `Projects/<ProjectName>/subjects/<subject_id>/config/` are subject-specific reproducibility snapshots, not repo-wide defaults.
- `paper/`, `paper_artifacts/`, and `paper_figures/` are kept local and are ignored by Git.
- Detailed schemas live in `docs/spec/DATA_CONTRACT.md` and `docs/spec/SCHEMAS.md`.

## License And Ethics

This repository is proprietary. See `LICENSE` for usage restrictions and authorization requirements.

Handle subject data confidentially and use the system responsibly.
