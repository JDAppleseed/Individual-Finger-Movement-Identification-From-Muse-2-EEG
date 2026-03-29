# RUNBOOK

## Timebase report

Each Step 1 session writes a timebase report to:

- `data/processed/<subject>_<session>_timebase_report.json`
- `reports/last_timebase_report.json` (latest pointer)

Key values to eyeball:

- `time_s_clamped_count` should be `0` (non-zero means LSL time moved backward and was clamped).
- `event_clamped_count` should be `0` (non-zero means event timestamps fell behind stream samples).
- `nearest_sample_delta_s_abs_max` ideally stays below ~0.05–0.10s for tight alignment.
- `dt_median_ms`/`dt_p95_ms` should roughly match the sample period (for 256 Hz, ~3.9 ms).

## AFTER CHANGES — RUN FULL CHECKS

```
python -m compileall .
pytest -q
python 3_evaluate_model.py --run-dir <run_dir> --deterministic --threshold-applicability 0.4 --no-manifest
python 1b_extract_windows.py --help
python -c "import json; from pathlib import Path; print('ok')"
python -c "from utils.timebase import clamp_monotonic_time; print('ok')"
```

## Step 1 Recording Smoke Test (No Muse Required)

This uses a mock LSL outlet and runs `1_stream_and_record.py` for a short capture.

```bash
python scripts/smoke_step1_record.py
```

Expected:
- non-empty `raw/eeg_raw_shard_*.npy` in the temp output dir
- `events/events.jsonl` may be empty if no labels were sent (this is OK)
- log lines with `[alive] recv=... wrote=...`

## Deployment Model Gate

Deployment candidates must satisfy all of the following:

- `train_config.json` has `active_finger_head: true`
- `train_config.json` has `finger_applicability_head: true`
- the saved model finger head has exactly `5` outputs
- the saved model includes `finger_applicability_head.weight` / `.bias`
- `temperature_scaling.json` includes `applicability_temperature`
- Step 7 actuation is only started with that model class
- pseudo-live replay reports:
  - `committed_non_rest_none_count == 0`
  - `committed_rest_non_none_count == 0`
  - `sent_non_rest_none_count == 0`
  - `sent_rest_non_none_count == 0`
  - `deployment_pair_invariant_ok == true`

Required deployment validation flow:

```bash
python tools/smoke_inference.py --npz eeg_windows.npz --model <run_dir>/finger_action_model.pt --scaler <run_dir>/scaler.npz
python tools/pseudo_live_replay.py --run-dir <run_dir> --session-dir <session_dir> --target-session-dir <target_session_dir> --infer-config <infer_json>
python 7_live_infer_and_actuate.py --config <infer_json> --enable-actuation
```

Notes:

- `OPEN/CLOSE + NONE` is a deployment failure, even if actuation would have been blocked later.
- `REST + active finger` is also a deployment failure for committed or sent outputs, even though raw active-finger logits remain structural on REST windows.
- Low finger confidence is allowed only as an actuation gate (`finger_gate_ok=false`), not as a committed label rewrite.
- Low applicability is allowed only as an actuation gate (`applicability_gate_ok=false`), not as a committed label rewrite.
- For active-finger deployment runs, use applicability FP/FN and action-applicability disagreement as the primary REST-side health metrics. Raw `REST + active finger` is retained only as a deprecated diagnostic.
- Legacy 6-class finger-head runs remain readable for historical analysis, but they are not deployable.

## Environment setup (clean clone)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt
```

System-level dependencies (LaTeX/Node) are listed in `docs/ops/SYSTEM_DEPS.md`.
