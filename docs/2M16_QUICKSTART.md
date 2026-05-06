# 2-M16 Quickstart

This guide is the clone-first path for the published `2-M16` bundle. It lets a new user validate the dataset, rebuild the reference windows, train a new model, and compare against the featured deployment run without collecting new EEG.

## Published Bundle

Root:

```text
Projects/2-M16/subjects/2-M16/
```

Source sessions:

| Session | Role | Raw shards | Published windows |
| --- | --- | ---: | ---: |
| `2-M16_20260216_150056_01` | Core movement session | 1,387 | 10,266 |
| `2-M16_20260317_190134` | Core mixed REST/movement session | 268 | 1,644 |
| `2-M16_20260315_145838_01` | Auxiliary quiet REST session | 135 | 1,059 |

Final dataset:

```text
Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/eeg_windows.npz
```

The final dataset is rebuilt from the three source-session NPZ files, then removes REST event IDs `0`, `1`, and `2` from `2-M16_20260216_150056_01`. Those three REST events contributed `522` windows that were repeatedly identified as problematic REST examples. The final dataset has `12,447` windows.

Current published size is about `225 MB` on this checkout, and users should budget about `244 MB` before Git compression depending on filesystem block accounting. Most of that is raw EEG shards so users can practice extraction.

## Setup

```bash
./scripts/setup_venv.sh
source .venv/bin/activate
python3 scripts/diagnose_env.py
```

Use commands from the repository root.

For live Muse collection, live inference, or the full operator workflow, use one Python 3.11/3.12 environment for Muse, LSL, PySide, and the pipeline:

```bash
source .venv/bin/activate
python3 eeglab_wrapper_ui.py
```

If you prefer conda, activate your own Python 3.11/3.12 environment before launching the UI. The UI starts child pipeline steps with its own interpreter, so recording, extraction, training, evaluation, and Step 7 live inference stay in the same environment.

## Validate Source Sessions

```bash
for session in \
  2-M16_20260216_150056_01 \
  2-M16_20260317_190134 \
  2-M16_20260315_145838_01
do
  python3 -m muse_streaming.validate_session \
    --session "Projects/2-M16/subjects/2-M16/sessions/$session"
done
```

## Rebuild Or Check The Reference Dataset

Check that the published final NPZ matches the source-session windows and prune rule:

```bash
python3 tools/build_2m16_reference_dataset.py --check-only
```

Build a scratch copy:

```bash
python3 tools/build_2m16_reference_dataset.py \
  --out /tmp/2m16_reference_eeg_windows.npz
```

Expected final dataset:

- `X`: `(12447, 64, 4)`
- Action counts: `REST=2404`, `OPEN=4814`, `CLOSE=5229`
- Finger counts: `NONE=2404`, `THUMB=2252`, `INDEX=1742`, `MIDDLE=2051`, `RING=1922`, `PINKY=2076`
- Channels: `TP9`, `AF7`, `AF8`, `TP10`

## Practice Extraction From Raw Sessions

Each source session includes raw shards, event labels, session metadata, and the currently published extracted windows.

```bash
python3 1b_extract_windows.py \
  --session-dir Projects/2-M16/subjects/2-M16/sessions/2-M16_20260317_190134
```

That command writes `processed/eeg_windows.npz` in the selected session. If you are comparing against the published artifact, run `git diff --stat` afterward or restore the file before committing.

## Train A New Model

```bash
SESSION_DIR="Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2"
RUN_ID="$(date +%Y%m%d_%H%M%S)_local"

python3 2_train_model.py \
  --config Projects/2-M16/subjects/2-M16/config/train.json \
  --session-dir "$SESSION_DIR" \
  --run-dir "$SESSION_DIR/processed/models/$RUN_ID"
```

The published training recipe uses:

- Seed `43`
- `60` epochs
- Batch size `64`
- Learning rate `0.001`
- `group_trial` split mode
- `auto_train_only` auxiliary REST policy
- Active-finger head plus finger-applicability head
- `center_detrend` window preprocessing

## Evaluate A Run

Evaluate your new run:

```bash
python3 3_evaluate_model.py \
  --config Projects/2-M16/subjects/2-M16/config/evaluate.json \
  --session-dir "$SESSION_DIR" \
  --run-dir "$SESSION_DIR/processed/models/$RUN_ID"
```

Evaluate the featured deployment run:

```bash
REF_RUN="$SESSION_DIR/processed/models/20260319_075520"

python3 3_evaluate_model.py \
  --config Projects/2-M16/subjects/2-M16/config/evaluate.json \
  --session-dir "$SESSION_DIR" \
  --run-dir "$REF_RUN"
```

The reference evaluation writes reports under:

```text
Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/reports/20260319_075520/
```

## Reference Results

Featured deployment run:

```text
20260319_075520
```

Metrics from `winning_model/session_report/eval_manifest.json`:

| Metric | Value |
| --- | ---: |
| Action accuracy | `89.79%` |
| Joint action+finger accuracy | `84.66%` |
| Non-REST finger accuracy | `85.96%` |
| REST true positive rate | `98.37%` |
| REST precision | `80.11%` |
| Action ECE | `0.0232` |
| Non-REST finger ECE | `0.0273` |
| Event-level action accuracy | `92.56%` |
| Event-level joint accuracy | `87.60%` |
| Event-level non-REST finger accuracy | `90.68%` |

Selection note:
- `20260403_grouptrial_rest050` wins the offline holdout and event-level leaderboard.
- `20260319_075520` is the public/deployment model because it wins the deployment-safety selection audit. With the current per-finger actuation path, report `95.37%` would-send precision, `0.25%` false REST actuation, and `91.11%` event hit rate; treat window-level send recall as throughput, not classifier accuracy.
- Website-facing metrics are staged in `docs/public_metrics_2m16_current.json`.
- For live `2-M16` inference, use `Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json`; it pins the March 19 model/scaler and tuned Step 7 postprocess settings.
- See `docs/2M16_MODEL_SELECTION_AUDIT.md` for the concrete ranking table and diagnosis.

Smoke-test the deployable artifacts:

```bash
python3 tools/smoke_inference.py \
  --npz "$SESSION_DIR/processed/eeg_windows.npz" \
  --model "$REF_RUN/finger_action_model.pt" \
  --scaler "$REF_RUN/scaler.npz"
```

## Artifact Boundary

Published:

- Raw shards, events, manifests, run metadata, and timebase reports for the three source sessions
- Per-source extracted `processed/eeg_windows.npz`
- Final pruned combined `processed/eeg_windows.npz`
- Featured deployment model run, scaler, cached predictions, temperature scaling, report figures, and manifests
- Stable `winning_model/` snapshot

Not published:

- Historical archives under `archive/`
- Full pseudo-live replay logs, CSVs, and JSONL files
- Live inference run directories
- Exploratory topomap/event-space outputs
- Local paper build products

See `Projects/2-M16/subjects/2-M16/PUBLISHED_ARTIFACTS.md` for hashes and exact file groups.
