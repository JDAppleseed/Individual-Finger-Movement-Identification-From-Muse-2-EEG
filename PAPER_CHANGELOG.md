# Paper Build Changelog (Auto-Expanded IEEE Manuscript)

This changelog summarizes the modifications made to expand `research_paper.tex` into a publication-grade IEEE-style paper **without inventing numbers**.

## What was added/changed

- Added an artifact-driven paper build script: `scripts/build_paper_artifacts.py`
  - Scans `Projects/**/processed/models/*/metrics.json` and `test_predictions.npz`
  - Computes test accuracies, ECE (10-bin), Wilson 95% CIs, and bootstrap 95% CIs directly from saved predictions
  - Extracts demographics from `Projects/**/subject.json`
  - Extracts session durations and movement/event counts from `Projects/**/run_meta.json` and `events/events.csv`
  - Copies all required figures from `processed/reports/<run_id>/` into `paper_figures/` with LaTeX-safe filenames
  - Writes LaTeX snippets + macros into `paper_artifacts/`

- Expanded manuscript: `research_paper.tex`
  - Added major scientific sections: Related Work, Uncertainty & Calibration, Real-Time Inference, Statistical Rigor, Threats to Validity, Reproducibility, and dataset/schema notes
  - Replaced hand-typed performance numbers with `paper_artifacts/paper_macros.tex` macros
  - Automatically inserts all per-run figures via `\input{paper_artifacts/figures.tex}`
  - Adds tables via `\input{paper_artifacts/*.tex}` generated from repository artifacts
  - Adds a TikZ system diagram in the System Overview section
  - Explicitly flags missing data as “not available in current run artifacts” (e.g., Step 7 latency logs, sex field)

- Added bibliography: `references.bib`
  - Includes MC Dropout (Gal & Ghahramani, 2016) and calibration (Guo et al., 2017)
  - Includes standard EEG deep learning citations (EEGNet; Schirrmeister et al.)
  - Adds placeholder BibTeX entries for consumer-EEG and finger-level prior work where the repo did not provide citations

## Generated outputs (in the working tree)

- `paper_artifacts/paper_stats.json`: machine-readable extracted metrics and metadata
- `paper_artifacts/paper_macros.tex`: LaTeX macros for all reported quantitative values
- `paper_artifacts/tables_*.tex`: LaTeX tables (demographics, dataset/windowing, sessions, events, performance, per-class accuracy, bootstrap CIs)
- `paper_artifacts/tables_generalization_gap.tex`: train–test gap table from `metrics.json`
- `paper_artifacts/figures.tex`: LaTeX figure includes for each run (action confusion, finger confusion, reliability/calibration, scatter)
- `paper_figures/*`: copied evaluation figures with sanitized filenames
- `research_paper.pdf`: compiled PDF (8 pages)

## Missing-data flags (explicitly stated in LaTeX)

- Step 7 real-time latency logs (mean/p50/p95/max latency and drop rate): not present in current run artifacts
- Sex field in `subject.json`: not present in current run artifacts
- Early stopping / gradient clipping: not logged in `train_config.json`

## Rebuild commands

- Regenerate paper artifacts: `python3 scripts/build_paper_artifacts.py`
- Compile PDF: `latexmk -pdf -interaction=nonstopmode research_paper.tex`
