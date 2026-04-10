# EEG Finger + Action Classification (Muse 2)

Real-time EEG-based finger + action (REST/OPEN/CLOSE) classification with uncertainty-aware gating, calibrated confidence, and reporting. The primary pipeline uses a **session directory** with lossless raw shards so all downstream steps are reproducible.

The recommended interface for running the full pipeline (all steps) is `eeglab_wrapper_ui.py`.

## Requirements

- Python 3.11 (required by `eeglab_wrapper_ui.py`; 3.12 is not currently supported end-to-end).
- macOS or Linux (tested); Windows is unverified.
- Muse 2 or any LSL EEG stream with 4 channels for live recording.
- Python dependencies are listed in `requirements.txt`.
- Optional: LaTeX for PDF reports (see `docs/ops/SYSTEM_DEPS.md`).
- Optional: hardware actuation uses `pyserial` (already in `requirements.txt`).

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

Connection model:

- Step 1 and Step 7 both read live EEG from an LSL stream. They do not talk to the Muse over BLE by themselves.
- To use a Muse 2, start a separate BLE -> LSL streamer first, then point Step 1 or Step 7 at that LSL stream.
- In Step 7, `--session-dir` is for resolving the trained model/scaler and output paths; it is not the live EEG source, as is the case for other instances of `--session-dir`.

The UI writes per-step configs under `Projects/<ProjectName>/subjects/<subject_id>/config/`
and snapshots each step to `Projects/<ProjectName>/subjects/<subject_id>/sessions/<session_id>/session_config.json`.
Checked-in configs under `Projects/2-M16/.../config/` are subject-specific reproducibility snapshots, not the source of repo-wide generic defaults.
For the next decisive `2-M16` Step 7 live run, the canonical CLI config is
`Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json`; the checked-in
`Projects/2-M16/subjects/2-M16/config/infer.json` file is the UI working mirror of that same
deployment contract. Legacy Step 7 artifacts that should not drive the next live run are archived
under `Projects/2-M16/subjects/2-M16/archive/step7/`.

## Session Directory (Core Concept)

Canonical session directories live under:

```
Projects/<project>/subjects/<subject_id>/sessions/<subject_id>_<session_id>/
```

Notes:

- `session_id` is **auto-prefixed** with `subject_id` if it is not already.
- Step 1 creates `meta.json`, `manifest.json`, `timebase_report.json`, and `events/events.jsonl` **before** LSL resolution so partial runs still leave a durable trail.
- All downstream steps should be run with `--session-dir` for deterministic resolution.

## Pipeline Overview (Session-Directory Flow)

- Step 1: Stream & Record → creates a new session directory.
- Step 1b: Extract Windows → reads `<session_dir>/raw/` + `<session_dir>/events/`, writes `<session_dir>/processed/`.
- Step 2: Train Model → reads `<session_dir>/processed/eeg_windows.npz`, writes `<session_dir>/processed/models/<run_id>/` including `temperature_scaling.json`.
- Step 3+: Evaluate / Figures / Reports → read from the same session directory and the latest model run, and report action/finger/joint metrics, applicability diagnostics, and committed deployment-pair invariants.
- Step 7: Live Infer + Actuate → use `docs/ops/STEP7_LIVE_RUNBOOK.md` for the decisive `2-M16` live workflow, including the pinned config, deployment session, preflight, replay, and audit commands.
- Step 7b: Review Live Predictions → reads `processed/live_infer*/predictions.jsonl`, prefers fresh `live_infer_<run_tag>` outputs over bare or legacy names, summarizes runtime behavior, and exports segment CSVs for video-aligned validation.

## CLI Run (Session-Directory Flow)

Record (Step 1):

```bash
python 1_stream_and_record.py --enable-plot --plot-scale fixed --plot-fixed-ylim -200 200
```

Step 1 connection note:

- Step 1 records from an existing LSL EEG stream.
- If you are using a Muse 2, start the BLE -> LSL bridge first, for example:

```bash
python -m cli start-streamer
```

Validate + extract (Step 1b):

```bash
python -m muse_streaming.validate_session --session <session_dir>
python 1b_extract_windows.py --session-dir <session_dir>
```

Step 1b label constraint:

- Extraction now fails fast if any event encodes `OPEN` or `CLOSE` with `finger_id=NONE`.
- Fix or prune those events first; they are no longer silently accepted into the dataset.

Train + evaluate + reports:

```bash
python 2_train_model.py --session-dir <session_dir>
python 3_evaluate_model.py --session-dir <session_dir>
python 3b_deepchecks_evaluate.py --session-dir <session_dir>
python 3c_live_paper_figures.py --session-dir <session_dir>
python 4_generate_reports.py --session-dir <session_dir>
```

Training/evaluation notes:

- Step 2 now reserves a calibration holdout from the training split by default (`--calibration-size 0.1`) and fits post-hoc temperature scaling.
- The default Step 2 recipe is the current winning `2-M16` deployment recipe: `seed=43`, `split_mode=group_trial`, `aux_rest_session_policy=auto_train_only`, `rest_balance_mode=core_event_equalized`, and `window_preprocess=center_detrend`.
- Step 2 trains the default deployable architecture with a 5-way active-finger head plus a binary finger-applicability head, and saves `finger_action_model.pt`, `scaler.npz`, `test_predictions.npz`, and `temperature_scaling.json`.
- Step 2 still records `threshold_applicability=0.50` in train config/run metadata because that value exists in the winning train artifact. New Step 3 and Step 7 configs use the deployed postprocess default `threshold_applicability=0.40`.
- `temperature_scaling.json` now stores calibrated temperatures for action, finger, and applicability. `test_predictions.npz` stores `applicability_probs` for runs that include the applicability head.
- Step 3 defaults to deterministic evaluation with no smoothing or hysteresis and uses the winning deployment applicability threshold (`threshold_applicability=0.40`) when postprocess metrics are computed.
- Step 3 applies the saved temperature scaling when recomputing predictions and treats applicability FP/FN, action-applicability disagreement, and committed deployment-pair invariants as the primary deployment health signals. Raw `REST + active finger` remains available as a deprecated structural metric for active-finger heads.

Paper/manuscript note:

- `3c_live_paper_figures.py` and `scripts/build_paper_artifacts.py` generate local manuscript inputs and figures.
- The repository intentionally does not track `paper/`, `paper_artifacts/`, or `paper_figures/`; keep those outputs local unless you have a specific reason to publish them elsewhere.

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

Labeling constraint:

- `rest` must pair with finger `none`.
- Finger-specific movement events should pair `open`/`close` with one of `thumb/index/middle/ring/pinky`.
- `open`/`close` with finger `none` is now treated as an invalid dataset label during Step 1b extraction.

## Live Inference (Step 7)

For the next decisive `2-M16` live run, use `docs/ops/STEP7_LIVE_RUNBOOK.md`.
The canonical config there is
`Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json`, and decisive runs should write
to a fresh `processed/live_infer_<run_tag>` directory instead of reusing the runtime default
`processed/live_infer`.
In the UI, the Step 7 `Session dir` field is the authoritative launch/preflight session; do not
assume the main session selector overrides a pinned Step 7 deployment session.

Preferred (auto-resolve latest model/scaler from the session directory):

```bash
python tools/live_preflight.py \
  --config Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json \
  --session-dir Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2 \
  --out-dir Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/live_infer_<run_tag> \
  --probe-stream \
  --probe-distribution

python 7_live_infer_and_actuate.py \
  --config Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json \
  --session-dir Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2 \
  --out-dir Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/live_infer_<run_tag>
```

Explicit override (no session directory):

```bash
python 7_live_infer_and_actuate.py --config <infer_config.json>
```

When `--session-dir` is omitted, `model_path`, `scaler_path`, and `out_dir` must be present in the config JSON.
Legacy compare/tuning sessions and superseded live outputs for `2-M16` are archived under
`Projects/2-M16/subjects/2-M16/archive/step7/` and should not be used for the decisive run.

Step 7 connection note:

- Step 7 also reads from a live LSL EEG stream, typically the same `Muse2-EEG` stream produced by the BLE -> LSL streamer.
- `--session-dir` only selects the trained artifacts and output location for inference; it does not replace the live stream, same as other instances of `--session-dir`.

Actuation (optional):

```bash
python 7_live_infer_and_actuate.py \
  --config <infer_config.json> \
  --session-dir <session_dir> \
  --enable_actuation
```

Notes:

- When `--enable_actuation` is set and `--serial_port` is omitted, Step 7 auto-detects a likely USB serial Arduino port (for example `/dev/cu.usbmodem*` on macOS).
- Pass `--serial_port` explicitly if multiple candidate serial devices are attached or auto-detection is ambiguous.
- Actuation is safety-gated; REST/NONE never actuate.
- Current live defaults use raw committed predictions with `postprocess=false`, `smoothing_enabled=false`, `finger_mode=raw`, `threshold_action=0.0`, `threshold_finger=0.0`, and `threshold_applicability=0.0`.
- Current actuation defaults use `actuation_min_prob=0.20`, `actuation_stability=2`, `actuation_repeat_ms=100`, and `actuation_cooldown_ms=0`.
- Current live-quality defaults are enabled: `input_clip_abs_z=6.0`, `bad_channel_rms_z=4.0`, `bad_channel_abs_p95_z=6.0`, `bad_channel_clipped_frac=0.05`, `bad_window_clipped_frac=0.10`, and `bad_window_max_masked_channels=1`.
- Step 7 logs prediction latency by default in `predictions.jsonl`; actuation events now also record `actuation_sent`, `actuation_latency_ms`, `actuation_speed_scalar`, `finger_applicable_prob`, and `applicability_gate_ok`.
- Raw shards and prediction logs are preserved by default. They are only disabled when `no_file_io` is set to `true`.
- Optional MC-dropout backend: set `use_inference_engine: true` in the infer config to route Step 7 through `utils/inference.py`. Relevant keys are `mc_passes`, `uncertainty_base_threshold`, and `uncertainty_weight`.
- When enabled, live inference uses mean probabilities across MC passes, logs `action_uncertainty` / `finger_uncertainty`, and adds an adaptive uncertainty gate before hardware actuation.
- Step 7 automatically loads `temperature_scaling.json` from the selected run when present and applies it before softmax in both direct and MC-dropout inference modes.
- The deployed decode path now enforces `REST => finger NONE` and `OPEN/CLOSE => active finger`, so committed/exported/live commands are always valid action-finger pairs.
- Low finger confidence and low applicability are actuation gates only. They do not rewrite a committed non-REST label back to `NONE`.
- Step 7 actuation only starts for deployable runs: `active_finger_head=true`, `finger_applicability_head=true`, a 5-output finger head, and `temperature_scaling.json` with `applicability_temperature`.
- Confidence-modulated actuation speed is enabled by default. Use `modulate_actuation_speed` and `actuation_speed_gamma` to adjust it; the Arduino receiver in `hardware/arduino/blue_hand_receive_upload/blue_hand_receive_upload.ino` now accepts an optional third `speed_u8` field.
- See `docs/actuation_requirements.md` for design constraints.

## Live Review (Step 7b)

After a Step 7 session, summarize the saved prediction log and export segment CSVs for manual review:

```bash
python tools/analyze_live_predictions.py \
  --session-dir <session_dir>
```

Explicit log path:

```bash
python tools/analyze_live_predictions.py \
  --pred-log <session_dir>/processed/live_infer*/predictions.jsonl \
  --out-json <summary.json> \
  --segments-csv <predicted_segments.csv> \
  --review-csv <predicted_segments_review.csv>
```

Notes:

- If `--pred-log` is omitted, the tool auto-resolves the latest `processed/live_infer*/predictions.jsonl` under `--session-dir`, preferring fresh `live_infer_<run_tag>` directories over bare or legacy names.
- Fresh decisive runs are expected to use `processed/live_infer_<run_tag>`; archived legacy outputs are not part of this auto-resolve path.
- The summary JSON reports latency, transition rate, actuation counts, uncertainty-gate failures, applicability-gate failures, and committed/sent deployment-pair invariants.
- `predicted_segments.csv` converts per-window predictions into contiguous predicted command intervals.
- `predicted_segments_review.csv` adds blank reviewer columns so you can compare predicted intervals against recorded video or robot-hand motion.
- The UI exposes this as `Step 7b: Review Live Predictions` and auto-fills the latest Step 7 log/output paths for the selected session when available.

## Muse Streaming CLI (Legacy CSV Pipeline)

This lightweight CLI records legacy CSVs under `data/` (separate from the session-directory pipeline).

Start a BLE → LSL streamer:

```bash
python -m cli start-streamer
```

List streams:

```bash
python -m cli list-streams
```

Health check:

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

Committed finger labels:

- `0` = NONE
- `1` = THUMB
- `2` = INDEX
- `3` = MIDDLE
- `4` = RING
- `5` = PINKY

Default deployable model heads:

- Finger head: active fingers only (`THUMB..PINKY`, 5 outputs)
- Action head: `REST/OPEN/CLOSE` (3 outputs)
- Applicability head: binary `finger meaningful on this window?`

Validity rules:

- REST + NONE is valid
- REST + any finger is invalid
- OPEN/CLOSE + NONE is invalid for extracted/trainable datasets and Step 1b now fails fast on it
- OPEN/CLOSE + finger 1–5 is valid

During training, active-finger loss is masked when action == REST, and the applicability head is trained on all windows with target `1[action != REST]`.

## Data Artifacts (Session Directory)

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

See `docs/spec/DATA_CONTRACT.md` and `docs/spec/SCHEMAS.md` for detailed schemas.

## Timebase & Latency

All sessions use a single LSL-aligned timebase (`absolute_v1`).

Key invariants:

- Single clock domain (LSL timestamps only).
- `time_s := lsl_ts_mono - stream_start_lsl_ts_mono` for samples, features, and events.
- LSL timestamps are clamped on backward jumps to produce `lsl_ts_mono`.
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

Paper sources and generated figure assets are kept local and are ignored by Git:

- `paper/`
- `paper_artifacts/`
- `paper_figures/`

## License & Ethics

This repository is **proprietary**. See `LICENSE` for usage restrictions and authorization requirements.

Handle subject data confidentially and use the system responsibly.
