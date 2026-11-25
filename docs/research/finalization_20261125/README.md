# CNN-LSTM baseline finalization — November 25, 2026

The live pipeline is restored to CNN-LSTM inference. The original March 19
2-M16 checkpoint, normalizer, temperatures, and deployment settings are preserved.
The newer feature-classifier ensembles remain reproducible offline experiments;
they are not a live backend. This is a conservative cleanup before a separate
rewrite, not a new live-performance certification.

## What changed since the public release

The website's latest listed release is the
[May 17 electrode-ablation study](https://alphahand.org/updates/2-m16-electrode-ablation-2026-05-17).
It still features the March 19 CNN-LSTM checkpoint. Later local work combined
1-M16/3-M16 recordings, tuned neural recipes, explored feature ensembles, revised
the private manuscript, and added opt-in ensemble shadow inference. Those studies
did not replace the March weights. The September studies also exposed calibration
independence and replay-provenance gaps.

This cleanup removes the live ensemble adapter, rejects old ensemble presets,
retains session/gap-aware evaluation and consistent Deepchecks splitting, and fixes
future calibration splits. Calibration now preserves groups, purges overlapping
signal, fails rather than falling back to random windows, and saves exact folds.
Historical calibration claims are not retroactively repaired.

## Current datasets and protocol

| Subject | Canonical dataset | Windows | Source recordings |
| --- | --- | ---: | ---: |
| 1-M16 | September 27 full combination | 14,672 | 3 |
| 2-M16 | April 28 combination with corrected session | 14,673 | 4 |
| 3-M16 | September 27 full combination | 7,998 | 2 |

`datasets.json` pins every selected dataset and source hash. Vetting checks finite
64-by-4 signal windows, channel order, 256 Hz sampling, consistent REST/NONE labels,
250 ms timing within stored timestamp precision, duplicate source timestamps,
source signal/label identity, and outer group separation. The final comparison
also audits every pair of fitting/calibration/test folds for shared events and
overlapping source-signal intervals. This is a technical data audit; it does not
establish physiological signal specificity or independently verify the performed
task.

1-M16/3-M16 include every window from their available source datasets. The 2-M16
combination preserves 522 previously excluded February REST windows as exclusions;
these are documented historical outcome-informed pruning, not newly discarded
data. It adds all 2,226 corrected April windows to the March reference. The April
session includes 291 active windows and 1,935 background REST-labeled windows;
background REST is not equivalent to intentionally cued REST. Do not concatenate
combined datasets with their source copies and count observations twice.

Each subject is trained separately using the original CNN-LSTM family, 60 epochs,
seed 43, batch size 64, learning rate 0.001, REST/action weights 1.0, 20% nominal
event-group test holdout, 10% calibration allocation from the training pool, and
a 250 ms outer purge. Auxiliary REST-only sessions remain fitting-only. The
calibration purge uses the maximum observed window duration. After diagnosing the initial 1-M16 coverage failure, the fitting-coverage guard
was added and only 1-M16 was retrained. The 2-M16/3-M16 folds are unchanged, and
their original run provenance is retained under `initial_attempt/protocol.json`.
For 1-M16, all 114 remaining movement-session REST windows stay in fitting; action
and applicability temperature fitting is skipped because calibration has no REST.
The 60-epoch recipe remains fixed; no epoch or threshold is selected from test performance.

`protocol.json` records environment and source hashes. `comparison.json` records
fold counts, metrics, and artifact hashes. Metrics use unsmoothed argmax predictions with available calibration
(identity temperatures where calibration lacks a class); REST maps to finger NONE. Finger accuracy is measured on true
movement windows, with predicted REST counting as an error. Joint accuracy requires
both labels to match. These are not command-gated or hardware-actuation metrics.

## Reproduce

From the repository root, with the local source datasets available:

```sh
./run_py tools/finalize_cnn_baselines.py audit
./run_py tools/finalize_cnn_baselines.py train
./run_py tools/finalize_cnn_baselines.py summarize
```

Training refuses to overwrite existing runs. For completed runs, use `summarize`.
Individual EEG recordings and new model runs remain local; public aggregate
reports identify their hashes and paths. The published 2-M16 reference remains
available through the existing quickstart and smoke test.

For a later, attended observation with an available LSL stream:

```sh
./run_py 7_live_infer_and_actuate.py --config configs/baselines/2-M16_march19_shadow.json
```

This preset pins the historical winning snapshot, enables full audit/parity
logging, and disables serial. Supply a fresh `--out-dir` within the session for
repeat observations. No hardware observation was performed during finalization.

## Interpretation and release boundaries

See [the generated comparison](COMPARISON.md) for current aggregate and per-finger
results. Joint accuracy is 69.59%, 76.07%, and 55.12% for 1-M16, 2-M16, and 3-M16.
The corrected 1-M16 retrain has 68.20% held-out REST recall and remains unpromoted.
The initial 0% result came from a calibration coverage error, now preserved in
`initial_attempt/`. See [the REST diagnosis](REST_DIAGNOSIS.md) for the correction
and controlled weight-2 comparison. The other research runs also remain unpromoted. Identical CNN-LSTM architecture does not
guarantee identical performance after changing the fitting/calibration allocation.

The comparison is retrospective, single-seed, and within-subject. It is not a
cross-subject transfer experiment or independent replication. Different label
distributions and session coverage limit comparisons between subjects. In 3-M16,
thumb/index/middle occur in one recording and ring/pinky in another. In 1-M16,
movement-session REST evidence is sparse. The corrected April sitting exposed the
March model's all-REST failure; including April in training cannot establish
unseen-session generalization.

The historical 95.37% command precision, 91.11% event-hit rate, and 0.25% false
REST-actuation summary refers to the full 12,447-window March development corpus,
not exclusively the 2,301-window holdout. Its exact May replay outputs were not
recovered, and another replay manifest points to a different checkpoint. Preserve
these as historical summaries, not newly reproduced held-out claims.

See [workflow/layout](../../repository_layout.md),
[removal review](../../maintenance/REMOVAL_REVIEW.md), and
[rewrite requirements](../../maintenance/REWRITE_REQUIREMENTS.md). Proposed
trimming requires the owner's approval. The manuscript and website release copy
are prepared for review; publication is separate from pushing the repository work.

## Training, evaluation, and data-handling re-audit — November 25

The follow-up audit reproduced **all 7,321 held-out predictions**, including the
applicability probabilities, from the three saved November models. Action and
finger argmax predictions match exactly; maximum probability differences are
below 1.5e-6. Each scaler matches statistics recomputed exclusively from its
fitting fold exactly. Fitting, calibration, and test folds share no raw signal.
See [full verification evidence](full_prediction_audit.json) and run
`./run_py tools/audit_saved_baselines.py` to reproduce it. The November comparison
numbers are unchanged, including corrected 1-M16 REST recall of **68.20%**.

Confirmed problems corrected in the surrounding pipeline:

- Step 3 generated additional random test splits for an already fitted model.
  Those could include training rows and cannot estimate independent multi-seed
  performance. This report is disabled; independent seeds require retraining.
- Cache rejection or absence triggered a new split. Evaluation now resolves the
  saved training holdout, verifies dataset SHA256, validates every cached label,
  and performs fresh inference. It stops if original membership is unknown.
  Deepchecks also requires saved fitting membership instead of using the test
  complement (which includes calibration and purged rows).
- Filtered training saved local fold positions as though they were source-file
  positions. New fold artifacts store original NPZ indices, an explicit coordinate
  marker, and the dataset hash. A one-epoch synthetic integration run verified
  subject filtering plus removal of a non-finite row, followed by successful
  evaluation without a prediction cache. These synthetic metrics are not research
  results. Legacy local folds require evidence of an unchanged identity mapping.
- Appending locked REST rows after splitting could reintroduce overlapping signal.
  The completed outer split now purges actual interval overlaps, and training
  checks the completed fit/calibration/test folds before fitting. An incompatible
  locked-REST calibration split fails rather than silently leaking signal.
- Temperature fitting placed the optimized parameter on the requested accelerator
  while fitting logits on CPU. Both now use CPU; current CPU runs are unaffected.
- Label loading flattened matrices and truncated fractional labels. It now rejects
  malformed labels. New extraction and prediction timestamps retain float64
  precision. Existing datasets are preserved. Evaluation also matches training's
  channel-layout conversion and tracks original indices through filtering.
- Smoothed evaluation did not forward applicability probabilities or report the
  resulting gates. Gate diagnostics are now recorded separately from classifier
  labels, matching live command gating semantics. The redundant action-only
  override that bypassed postprocessing decisions is removed.
- REST diagnostics merged identical event numbers across sessions and recommended
  pruning/relabeling based on prediction errors. Events are now session-scoped,
  and flags request independent signal/label review. Model errors alone must not
  justify exclusion or relabeling. Full-corpus replay and auxiliary REST summaries
  explicitly identify their inclusion of training data.

The fixed 60-epoch recipe saves the final epoch; it does not automatically select
an epoch using test accuracy. Test metrics are nevertheless exposed during
training, and the existing holdouts have been inspected repeatedly. They are
research/development evidence, not untouched confirmation sets. The weight-2
1-M16 experiment remains diagnostic, not a promoted model. Prospective session
validation and acquisition/artifact controls remain necessary for generalization
claims. Historical outcome-informed pruning is documented rather than undone or
reclassified as independent validation.

No raw recordings, baseline model artifacts, or historical reports were deleted
or rewritten by this audit. The private manuscript and website draft retain the
same November metrics. This is a targeted logic audit, not a claim that every
archived script or live hardware behavior has been certified.

## Full-subject corpus evaluation

Full-corpus evaluation is available again as a separate diagnostic. It evaluates
all finite windows for one subject, including fitting and calibration data, using
the saved model, preprocessing, and temperatures. It performs no retraining,
random re-splitting, smoothing, or command gating.

| Model | Windows | Action | Finger on true movement | Joint | REST recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1-M16 march5 | 13,188 | 86.93% | 88.21% | 79.65% | 97.45% |
| 2-M16 march19 | 12,447 | 89.23% | 85.71% | 85.40% | 96.55% |
| 1-M16 november | 14,672 | 85.35% | 86.06% | 77.19% | 95.86% |
| 2-M16 november | 14,673 | 88.24% | 85.61% | 85.50% | 91.98% |
| 3-M16 november | 7,998 | 77.81% | 66.24% | 64.80% | 89.86% |

These full-corpus results measure fit across the available recording corpus. They
are separate from held-out evaluation and gated command replay. Historical models
use their original corpora; November models use the expanded vetted corpora.
Their denominators and preprocessing differ, so these are not controlled model
comparisons. Missing original replay artifacts mean the May command-policy
headline is preserved as historical evidence, not claimed as reproduced by this
new classifier-only evaluation.

## Preserved published headlines

The published March 2-M16 saved-test figures remain **89.79% action**, **85.96%
finger on non-rest windows**, and **84.66% joint**. The March 1-M16 report remains
**83.94% action**, **80.61% finger**, and **92.11% REST recall** on its original
saved test split. The May full-corpus command summary remains **95.37% would-send
precision**, **91.11% event hit rate**, **86.42% committed joint accuracy**, and
**0.25% false REST actuation**. These historical values and their metric meanings
are preserved; the new evaluation does not overwrite them.

Run a full-corpus diagnostic without changing any saved holdout:

```bash
./run_py tools/evaluate_full_dataset.py --preset 1-M16_november --output output/full-corpus-1-M16.json
```

Other presets: `2-M16_november`, `3-M16_november`, `1-M16_march5`, and
`2-M16_march19`. For another dataset/model, supply `--npz PATH --run-dir PATH`
and optionally `--subject-id ID` instead of `--preset`. Existing report files
are never overwritten. Aggregate results are in [full_corpus/](full_corpus/);
original public headlines are pinned in [historical_headlines.json](historical_headlines.json).
The existing `tools/pseudo_live_replay.py` remains the separate command-replay path.
