# Subject Config Guide

This directory stores the public `2-M16` configs used by the quickstart, UI, and reproducibility workflows.

Published configs:
- `train.json`: reference training recipe for the combined pruned dataset.
- `evaluate.json`, `evaluate_deepchecks.json`, `evaluate_figures.json`: evaluation and figure entrypoints.
- `infer.json`: subject-local live inference config aligned with the winning model.
- `live_review.json`: review config for live inference outputs.
- `pseudo_live.json`: offline replay config skeleton without the full historical replay logs.
- `step1b.json`: extraction settings for rebuilding source-session windows.

For a decisive live run, launch `eeglab_wrapper_ui.py` from `muse311`, use `../winning_model/configs/infer.json`, and follow `../../../../../docs/ops/STEP7_LIVE_RUNBOOK.md`. The pinned live model is `processed/models/20260319_075520/finger_action_model.pt`; `20260403_grouptrial_rest050` is documented only as an offline benchmark.

Configs not listed above are local or exploratory and are intentionally not part of the published bundle.
