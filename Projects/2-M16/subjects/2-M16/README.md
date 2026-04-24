# 2-M16 Published Bundle

This subject bundle is the curated public entrypoint for `2-M16`. It includes enough data and outputs to validate the source sessions, rebuild the published training dataset, train a new model, and compare against the reference run without collecting new EEG first.

Start here:
- Repo quickstart: `../../../../README.md`
- Full 2-M16 workflow: `../../../../docs/2M16_QUICKSTART.md`
- Published file and hash inventory: `PUBLISHED_ARTIFACTS.md`

Main paths:
- `config/`: public train, evaluate, extraction, inference, and review configs.
- `sessions/2-M16_20260216_150056_01`: core movement source session.
- `sessions/2-M16_20260315_145838_01`: auxiliary quiet REST source session.
- `sessions/2-M16_20260317_190134`: mixed REST/movement source session.
- `sessions/combined_20260319_081200_pruned_rest_events_0_1_2`: final pruned combined dataset and reference run.
- `winning_model/`: stable copy of the current best deployable model, configs, reports, and manifests.

Reference run:
- Run ID: `20260403_grouptrial_rest050`
- Action accuracy: `91.83%`
- Joint action+finger accuracy: `86.66%`
- Non-REST finger accuracy: `88.11%`

Label rule:
- `REST` pairs only with `NONE`.
- `OPEN` and `CLOSE` pair only with an active finger.
- `OPEN/CLOSE + NONE` is invalid for extraction, training, and deployment.

The bundle intentionally excludes archives, full pseudo-live logs, live inference run directories, exploratory event-space/topomap outputs, `.DS_Store`, and unrelated project trees.
