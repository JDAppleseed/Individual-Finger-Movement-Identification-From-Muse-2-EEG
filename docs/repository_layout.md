# Workflow and repository layout

This final cleanup preserves the numbered entrypoints and existing artifact paths.
Moving those files now would break saved configs, UI launches, and old research
provenance. New documentation, baseline presets, and aggregate evidence have
dedicated directories; larger code reorganization belongs in the separate rewrite.

```text
1_stream_and_record.py       Capture authoritative raw signal and events
1b_extract_windows.py        Validate timing and extract labeled windows
2_train_model.py             Train subject-specific CNN-LSTM and calibrate
3_evaluate_model.py          Evaluate a saved model and split
4_generate_reports.py       Render evaluation evidence
5_*                         Review and validate events
6_sweep.py                  Explicit training sweeps
7_live_infer_and_actuate.py  CNN-LSTM live inference and optional actuation
app/                        UI launch, config, and process contracts
muse_streaming/             Recording, clocks, transport, recovery
models/                     Neural architecture definitions
utils/                      Shared preprocessing, splits, decoding, command gates
tools/                      Operator diagnostics and explicit research commands
scripts/                    Build, setup, repair, and smoke-test automation
tests/                      Contract and regression coverage
configs/baselines/          Pinned reference observation presets and hashes
Projects/<project>/subjects/<subject>/
  sessions/<session>/       Raw/events plus derived processed artifacts
  winning_model/            Published historical deployment snapshot
docs/spec/                  Data and schema contracts
docs/ops/                   Operating procedures
docs/research/              Dated results and provenance
docs/maintenance/           Trimming review and rewrite handoff
logs/                       Local run logs and historical experiments
paper*/                     Local manuscript, generated tables, and figures
output/, tmp/               Local deliverables, backups, and scratch work
```

Use one canonical dataset per subject for a declared experiment. A combined
dataset and its constituent recordings are different representations of the same
observations: never concatenate both. Keep source session/event identities through
windowing, split by events, and audit temporal overlap. Cross-subject comparison
means comparing separate subject-specific models, not silently pooling people.

Run `./run_py tools/finalize_cnn_baselines.py audit`, then `train`, then `summarize`
for the frozen November comparison. Training refuses to overwrite existing run
directories. Aggregate provenance lives under `docs/research/finalization_20261125/`.

Historical ensemble code is retained for reproducibility, but is outside the live
CNN-LSTM workflow. No training or research utility should promote a live model.
Generated outputs are ignored by Git; ignoring a file does not remove it locally.
