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
python 3_evaluate_model.py --npz eeg_windows.npz --model finger_action_model.pt --scaler scaler.save --subject-id "" --deterministic --split-seed 42 --no-manifest
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
- non-empty `*_raw.csv` in the temp output dir
- `events/events.jsonl` may be empty if no labels were sent (this is OK)
- log lines with `[alive] recv=... wrote=...`

## Environment setup (clean clone)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt
```

System-level dependencies (LaTeX/Node) are listed in `SYSTEM_DEPS.md`.
