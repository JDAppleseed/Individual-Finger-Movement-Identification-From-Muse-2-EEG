# Tools

The numbered root scripts remain the primary pipeline. Tools are explicit,
operator-invoked diagnostics and experiments; they do not run automatically.

- **Current comparison:** `finalize_cnn_baselines.py` audits the newest combined
  datasets, trains the fixed CNN-LSTM recipe, and checks saved folds/results.
  `check_1m16_rest_coverage.py` reconstructs the coverage regression and fixed-fold
  REST-weight sensitivity diagnosis.
- **Reference validation:** `build_2m16_reference_dataset.py`, `smoke_inference.py`.
- **Streaming and live diagnostics:** `live_preflight.py`, `lsl_sanity_probe.py`,
  `audit_live_parity.py`, `analyze_live_raw_inputs.py`, `analyze_live_predictions.py`,
  `check_time_alignment.py`, `profile_live_loop.py`.
- **Replay:** `pseudo_live_replay.py`, `replay_live_capture.py`.
- **Research:** channel ablation, event-space plotting, and experimental topomaps.
- **Archived September experiments:** `compare_subject_models.py`,
  `predict_subject_ensemble.py`, `assess_subject_live_readiness.py`,
  `select_balanced_subject_gates.py`, `check_initial_subject_cnns.py`, and
  `check_2m16_session_transfer.py`. These rely on local September artifacts and
  may create/update dated experiment outputs. They are not the current baseline
  workflow. Ensemble presets are rejected by Step 7.

See [workflow/layout](../docs/repository_layout.md) for artifact ownership and
[trimming review](../docs/maintenance/REMOVAL_REVIEW.md) before removing anything.

`audit_saved_baselines.py` independently reproduces every saved November holdout
prediction and its fitting-only scaler, checks final signal separation, and writes
`docs/research/finalization_20261125/full_prediction_audit.json`. It does not retrain
or replace baseline artifacts.
