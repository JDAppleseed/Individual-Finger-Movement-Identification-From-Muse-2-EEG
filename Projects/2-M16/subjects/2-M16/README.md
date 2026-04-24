# 2-M16 Published Bundle

This subject bundle is the curated public entrypoint for `2-M16`. It includes enough data and outputs to validate the source sessions, rebuild the published training dataset, train a new model, and compare against the featured deployment run without collecting new EEG first.

Start here:
- Repo quickstart: `../../../../README.md`
- Full 2-M16 workflow: `../../../../docs/2M16_QUICKSTART.md`
- Published file and hash inventory: `PUBLISHED_ARTIFACTS.md`

Main paths:
- `config/`: public train, evaluate, extraction, inference, and review configs.
- `sessions/2-M16_20260216_150056_01`: core movement source session.
- `sessions/2-M16_20260315_145838_01`: auxiliary quiet REST source session.
- `sessions/2-M16_20260317_190134`: mixed REST/movement source session.
- `sessions/combined_20260319_081200_pruned_rest_events_0_1_2`: final pruned combined dataset and featured deployment run.
- `winning_model/`: stable copy of the current best deployable model, configs, reports, and manifests.

Featured deployment run:
- Run ID: `20260319_075520`
- Action accuracy: `89.79%`
- Joint action+finger accuracy: `84.66%`
- Non-REST finger accuracy: `85.96%` from the evaluation manifest, with `87.01%` in the saved model-card test metric.
- Cleaned pseudo-live would-send precision: `93.32%`
- Cleaned pseudo-live false REST actuation: `0.12%`

Model-selection note:
- `20260403_grouptrial_rest050` wins the offline holdout and event-level scores.
- `20260319_075520` is the public/deployment model because it wins would-send precision, false REST actuation, holdout REST TPR, and action calibration.
- See `../../../../docs/2M16_MODEL_SELECTION_AUDIT.md` for the ranking table.

Label rule:
- `REST` pairs only with `NONE`.
- `OPEN` and `CLOSE` pair only with an active finger.
- `OPEN/CLOSE + NONE` is invalid for extraction, training, and deployment.

The bundle intentionally excludes archives, full pseudo-live logs, live inference run directories, exploratory event-space/topomap outputs, `.DS_Store`, and unrelated project trees.
