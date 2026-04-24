# Winning Model Snapshot

This folder is the stable copy of the current best `2-M16` deployment candidate.

Source:
- Session: `combined_20260319_081200_pruned_rest_events_0_1_2`
- Run: `20260403_grouptrial_rest050`

Published contents:
- `configs/`: deployment-facing configs for inference and pseudo-live replay setup.
- `model_run/`: model, scaler, temperature scaling, cached test predictions, metrics, and train config.
- `session_report/`: report and figures for the canonical session evaluation.
- `repo_report/`: compact repo-level report figures and HTML.

Not published here:
- Full pseudo-live replay logs, CSVs, JSONL files, and historical live run directories.
- Legacy compare/tuning artifacts under `../archive/`.

Use `configs/infer.json` for the next decisive Step 7 live run. See `winning_model_manifest.json` for exact source paths and copied artifact locations.
