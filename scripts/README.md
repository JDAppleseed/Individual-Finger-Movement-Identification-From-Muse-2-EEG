# Scripts

Setup/environment checks, smoke tests, repairs, and report builders live here.
Use `./run_py` from the repository root to select the project interpreter.

Repair and salvage scripts change derived/session artifacts: inspect their CLI
and preserve original data before use. Paper builders read local manuscript and
experiment assets; those assets are deliberately excluded from public source.
`build_subject_comparison_paper_artifacts.py` reproduces the archived September
ensemble tables, not the November CNN-LSTM comparison.

The current reproducible comparison entrypoint is
`../tools/finalize_cnn_baselines.py`; its aggregate evidence belongs under
`docs/research/finalization_20261125/`. Never promote generated runs automatically.
