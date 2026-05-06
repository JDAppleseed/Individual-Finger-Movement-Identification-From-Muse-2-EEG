# Winning Model Snapshot

This folder is the stable copy of the current best `2-M16` deployment candidate.

Source:
- Session: `combined_20260319_081200_pruned_rest_events_0_1_2`
- Run: `20260319_075520`

Selection:
- `20260319_075520` is the featured public/deployment checkpoint.
- `20260403_grouptrial_rest050` remains the better offline holdout/event-level benchmark, but it is not the deployment snapshot because its cleaned-corpus would-send precision is lower and its false REST actuation is higher.
- Current public reporting should use the per-finger command path: `95.37%` would-send precision, `0.25%` false REST actuation, and `91.11%` event hit rate. See `../../../../../docs/public_metrics_2m16_current.json` from the repository root.
- See `../../../../../docs/2M16_MODEL_SELECTION_AUDIT.md` from the repository root for the historical comparison table.

Published contents:
- `configs/`: deployment-facing configs for inference and pseudo-live replay setup.
- `model_run/`: model, scaler, temperature scaling, cached test predictions, metrics, and train config.
- `session_report/`: report and figures for the canonical session evaluation.
- `repo_report/`: compact repo-level report figures and HTML.

Not published here:
- Full pseudo-live replay logs, CSVs, JSONL files, and historical live run directories.
- Legacy compare/tuning artifacts under `../archive/`.

Use `configs/infer.json` for the next decisive Step 7 live run. See `winning_model_manifest.json` for exact source paths and copied artifact locations.
