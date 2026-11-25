# Removal review — approval pending

No file or folder listed here is authorized for deletion. The finalization keeps
all recordings, models, experimental results, and prior manuscript versions.
Review `removal_candidates.json` for measured sizes and paths. Sizes overlap when
a candidate folder contains another candidate; do not sum them blindly.

| Candidate | Proposed eventual action | Condition before removal |
| --- | --- | --- |
| `__pycache__/`, `.pytest_cache/`, `.DS_Store` | Remove generated caches | User approves; reproducible from code |
| `tmp/pdfs/` | Archive then trim scratch builds | Compare with final manuscript and audit reports first |
| `alphahand-overleaf.zip` | Archive as a dated manuscript snapshot | Confirm source and bibliography are preserved |
| `output/` historical deliverables | Keep final deliverables; review duplicates | Preserve September audit and November backup |
| `logs/` historical sweeps | Compress/archive, not blanket deletion | Retain configs, splits, hashes, predictions, and selected results |
| `Projects/**/archive/` | Review duplicated deployment snapshots | Hash-check against canonical weights/scalers/configs |
| `ablation_runs/` | Preserve research evidence | Do not trim until repeated-seed work and paper references are checked |
| Root screenshot | Move to a named historical asset folder or remove | Confirm no external/document reference relies on it |
| Old ensemble tooling | Optional archive after rewrite | Preserve reproducible September experiment and tests together |

Code-only follow-up candidates: the now-unused `_rebalance_calibration_split`
and `_calibration_stratify_labels` helpers in Step 2. They remain for review, but
the training path no longer calls them. Do not reuse the individual-window
rebalancer in the rewrite.

Protected: raw recordings, authoritative event labels, repaired-label provenance,
published March model and normalizer, saved split indices, private manuscript,
historical negative results, and any artifact needed to substantiate a public claim.
No `git clean`, bulk deletion, or model pruning is part of this cleanup.
