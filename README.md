# AlphaHand EEG Finger Movement Identification

This repository contains the research pipeline behind AlphaHand: recording Muse 2 EEG, labeling finger/action trials, extracting model-ready windows, training and evaluating classifiers, generating reports, and running live inference with optional actuation.

The project is intended for developers and researchers who want to reproduce the current results, add new subjects and datasets, test better models, and improve the end-to-end system. Public project context, results, methods notes, and manuscript status live at `https://alphahand.org`.

The included `2-M16` bundle is a reference dataset and model snapshot. It is a starting point for validation and experimentation, not the whole scope of the repository.

## Start Here

Set up Python:

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
python3 scripts/diagnose_env.py
```

For lab/operator runs, launch the UI from the Python 3.11 environment used for Muse and PySide tooling:

```bash
conda activate muse311
python3 eeglab_wrapper_ui.py
```

The UI launches pipeline subprocesses with the same interpreter that launched the UI. If the UI starts inside `muse311`, Step 1 recording, Step 1b extraction, Step 2 training, Step 3 evaluation, and Step 7 live inference run inside `muse311` too.

Run the reference sanity checks:

```bash
python3 tools/build_2m16_reference_dataset.py --check-only

SESSION_DIR="Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2"
RUN_DIR="$SESSION_DIR/processed/models/20260319_075520"

python3 tools/smoke_inference.py \
  --npz "$SESSION_DIR/processed/eeg_windows.npz" \
  --model "$RUN_DIR/finger_action_model.pt" \
  --scaler "$RUN_DIR/scaler.npz"
```

Then choose a path:
- Use the published reference bundle: `docs/2M16_QUICKSTART.md`
- Run the full UI workflow: `conda activate muse311 && python3 eeglab_wrapper_ui.py`
- Record and train a new subject from CLI: follow the step map below
- Prepare live inference: `docs/ops/STEP7_LIVE_RUNBOOK.md`

## What This Repo Provides

- Lossless EEG capture from a Muse 2 or any compatible 4-channel LSL EEG stream.
- Session-level raw shards, event labels, manifests, and timebase diagnostics.
- Offline window extraction into `eeg_windows.npz`.
- PyTorch training for action classification (`REST`, `OPEN`, `CLOSE`) and active-finger classification.
- Evaluation, calibration, confusion figures, HTML reports, and cached test predictions.
- Pseudo-live replay and Step 7 live inference/actuation tooling.
- A curated `2-M16` reference bundle so new contributors can verify their setup without collecting data first.

## Repository Map

Important entrypoints:
- `eeglab_wrapper_ui.py`: main operator UI for setup, recording, extraction, training, evaluation, and live inference.
- `1_stream_and_record.py`: Step 1 lossless recording.
- `1b_extract_windows.py`: Step 1b raw/event to window extraction.
- `2_train_model.py`: Step 2 model training.
- `3_evaluate_model.py`: Step 3 evaluation and calibration.
- `3b_deepchecks_evaluate.py`: additional model checks.
- `3c_live_paper_figures.py`: figures used by reports and research writeups.
- `4_generate_reports.py`: report generation.
- `5_review_events.py`, `5_validate_events.py`: event review and repair tools.
- `7_live_infer_and_actuate.py`: Step 7 live inference and optional hardware actuation.

Important docs:
- `docs/2M16_QUICKSTART.md`: reproduce and experiment with the included reference bundle.
- `docs/spec/DATA_CONTRACT.md`: lossless session contract.
- `docs/spec/SCHEMAS.md`: artifact filenames and schemas.
- `docs/ops/RUNBOOK.md`: operational checks and deployment model gates.
- `docs/ops/STEP7_LIVE_RUNBOOK.md`: decisive live inference runbook.
- `docs/ops/SYSTEM_DEPS.md`: optional system dependencies.

## Pipeline Operating Manual

Use repo-relative paths in configs, commands, docs, and manifests. Public artifacts should not contain local absolute paths.

| Step | Goal | Main command or UI action | Output to inspect |
| --- | --- | --- | --- |
| 0 | Create/select project and subject | UI project selector | `Projects/<project>/subjects/<subject>/` |
| 1 | Record lossless EEG and events | `python3 1_stream_and_record.py` | `raw/eeg_raw_shard_*.npy`, `events/events.jsonl`, `manifest.json` |
| 1a | Validate capture health | `python3 -m muse_streaming.validate_session --session <session_dir>` | no missing sequence ranges, healthy timebase |
| 1b | Extract model windows | `python3 1b_extract_windows.py --session-dir <session_dir>` | `processed/eeg_windows.npz`, `extraction_report.json` |
| 2 | Train a model | `python3 2_train_model.py --session-dir <session_dir>` | `processed/models/<run_id>/` |
| 3 | Evaluate and calibrate | `python3 3_evaluate_model.py --session-dir <session_dir> --run-dir <run_dir>` | `eval_manifest.json`, figures, cached predictions |
| 3b/3c | Run deeper checks and figures | `python3 3b_deepchecks_evaluate.py`, `python3 3c_live_paper_figures.py` | diagnostics and paper/report figures |
| 4 | Publish a report | `python3 4_generate_reports.py --session-dir <session_dir>` | HTML report and report assets |
| 5 | Review labels | `python3 5_review_events.py`, `python3 5_validate_events.py` | fixed or documented event issues |
| 6 | Sweep ideas | `python3 6_sweep.py` | comparative runs and metrics |
| 7 | Live inference/actuation | `python3 7_live_infer_and_actuate.py --config <infer_json>` | `processed/live_infer_<run_id>/` evidence |

For live recording and Step 7, create or select an LSL EEG stream first. The pipeline reads from LSL; it does not talk directly to Muse over BLE unless you use a separate bridge such as the UI Muse connector or `python3 -m cli start-streamer`.

## Published Reference Bundle

The curated reference bundle lives at:

```text
Projects/2-M16/subjects/2-M16/
```

It includes:
- Three raw source sessions with events, manifests, metadata, timebase reports, and raw shards.
- Per-source extracted `processed/eeg_windows.npz` files.
- The final pruned combined dataset.
- Featured deployment run `20260319_075520`.
- `winning_model/` configs, model artifacts, figures, reports, and manifests.

The April 3 run `20260403_grouptrial_rest050` remains documented as an offline benchmark, but it is not the featured deployment model because the March 19 checkpoint wins the public safety metrics: `93.32%` would-send precision and `0.12%` false REST actuation on the cleaned pseudo-live corpus.

For `2-M16` live inference, the authoritative UI/Step 7 config is `Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json`. It pins the March 19 model and scaler, enables the tuned postprocess family, and should be run from the `muse311` UI environment.

Use it to verify your environment, compare model changes, and understand the expected artifact layout:

```bash
python3 tools/build_2m16_reference_dataset.py --check-only
```

Detailed inventory and hashes are in `Projects/2-M16/subjects/2-M16/PUBLISHED_ARTIFACTS.md`.

## Train A Baseline On The Reference Dataset

```bash
SESSION_DIR="Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2"
RUN_ID="$(date +%Y%m%d_%H%M%S)_local"

python3 2_train_model.py \
  --config Projects/2-M16/subjects/2-M16/config/train.json \
  --session-dir "$SESSION_DIR" \
  --run-dir "$SESSION_DIR/processed/models/$RUN_ID"

python3 3_evaluate_model.py \
  --config Projects/2-M16/subjects/2-M16/config/evaluate.json \
  --session-dir "$SESSION_DIR" \
  --run-dir "$SESSION_DIR/processed/models/$RUN_ID"
```

New local runs under `Projects/.../processed/models/` stay ignored unless deliberately added to a curated release.

## Add A New Subject Or Dataset

Use this process when expanding beyond `2-M16`:

1. Create a new subject under `Projects/<project>/subjects/<subject>/`.
2. Record multiple sessions with consistent channel order, clear trial prompts, and enough REST coverage.
3. Keep raw shards and `events/events.jsonl` as the source of truth. Derived windows and models should be reproducible from those inputs.
4. Validate every source session before extraction.
5. Fix invalid labels before training. Do not train on ambiguous labels.
6. Extract windows with documented settings and keep `extraction_report.json`.
7. Train a baseline with a run ID that records the recipe, split mode, seed, and dataset paths.
8. Evaluate on a held-out split that matches the research question. Prefer grouped splits when repeated windows from one trial could leak across train/test.
9. Compare against the reference bundle only as a sanity benchmark. New subjects may differ in signal quality, trial design, and class balance.
10. Publish only a curated artifact set: source sessions, final dataset, featured run, figures, reports, configs, and manifests.

When publishing a new bundle, update `.gitignore` with narrow exceptions for that subject instead of unignoring the whole `Projects/` tree.

## Labels And Data Contract

Canonical actions:
- `REST = 0`
- `OPEN = 1`
- `CLOSE = 2`

Canonical fingers:
- `NONE = 0`
- `THUMB = 1`
- `INDEX = 2`
- `MIDDLE = 3`
- `RING = 4`
- `PINKY = 5`

Rules:
- `REST` must pair with `NONE`.
- `OPEN` and `CLOSE` must pair with an active finger.
- `OPEN/CLOSE + NONE` is invalid for extraction, training, and deployment.
- `REST + active finger` is invalid as a committed/deployed label.

For artifact schemas, see `docs/spec/SCHEMAS.md`. For lossless raw-session invariants, see `docs/spec/DATA_CONTRACT.md`.

## Development And Improvement Areas

Good community research contributions include:
- Better data collection protocols for additional subjects and sessions.
- More robust preprocessing and artifact rejection.
- Architecture experiments that keep the public dataset contract stable.
- Cross-subject, leave-session-out, and grouped-trial evaluation improvements.
- Calibration, uncertainty, and actuation-gating improvements.
- Better reports, visualizations, and diagnostics that make failures easier to understand.
- Clear published bundles for new subjects with reproducible hashes and repo-relative configs.

Keep changes measurable. Every model or pipeline claim should point to a config, dataset, run directory, and evaluation manifest.

## Contributions And Contact

Use GitHub issues and pull requests for public, reproducible improvements to code, configs, docs, tests, reports, and model recipes.

For model changes:
- include the training/eval config, dataset or session path, run directory, and evaluation manifest
- compare against the current public `2-M16` deployment run when making deployment claims
- label results as research or offline improvements unless they pass the public replacement gate in `docs/2M16_MODEL_SELECTION_AUDIT.md`

Use the contact page at `https://alphahand.org/contact` instead of a public PR for:
- commercial or closed-source permission requests
- private collaboration proposals
- human-subject data release or consent questions
- safety-sensitive disclosures or anything else that should not start in a public repo thread

## Validation Before Publishing

Run the focused checks for this documentation/artifact bundle:

```bash
LEAK_PATTERN='/User''s/|Desktop/Individual''-Finger'
rg "$LEAK_PATTERN" README.md docs Projects/2-M16 \
  -g '!*.npy' -g '!*.pt' -g '!*.png' -g '!*.html'

python3 tools/build_2m16_reference_dataset.py --check-only

python3 tools/smoke_inference.py \
  --npz Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/eeg_windows.npz \
  --model Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260319_075520/finger_action_model.pt \
  --scaler Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260319_075520/scaler.npz

python3 -m pytest -q \
  tests/test_default_recipe.py \
  tests/test_label_schema.py \
  tests/test_extract_windows_logic.py \
  tests/test_cache_dataset_info_validation.py
```

For a full local development pass, also run:

```bash
python3 -m compileall .
python3 -m pytest -q
```

## Requirements

- Python 3.11 or 3.12 for the intended environment.
- macOS or Linux.
- `requirements.txt` installed in a virtual environment.
- Muse 2 or another 4-channel LSL EEG stream for new live recording.
- Optional: LaTeX for PDF reports; see `docs/ops/SYSTEM_DEPS.md`.
- Optional: Arduino/serial hardware for Step 7 actuation.

Some tests may run on other Python versions, but published workflows should state the version used.

## Publication, License, And Ethics

This is human-subject EEG research infrastructure. Treat raw data, subject metadata, and live actuation outputs with care.

Before distributing new data or model artifacts:
- Confirm the subject/data release is authorized.
- Remove local absolute paths from public files.
- Publish only a curated artifact set with hashes.
- State what was intentionally excluded.

License summary:
- Non-commercial research, education, nonprofit, charity, humanitarian, and public-interest use is allowed.
- Any use, modification, integration, deployment, or published work must remain public-source and must publish changes at no charge.
- For-profit commercial use requires prior written approval from Jonathan Davanzo.

See `LICENSE` for the full terms that govern use and redistribution.
