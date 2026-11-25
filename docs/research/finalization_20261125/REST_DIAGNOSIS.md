# 1-M16 REST regression: cause, correction, and weighting

The initial November 0% REST recall was a regression in the newly introduced
calibration policy. The combined dataset was correct and included all available
REST recordings. The policy preserved group separation but failed to preserve
fitting coverage within the movement recording. That was an implementation error,
not evidence that CNN-LSTM could no longer classify REST.

## All recorded REST was included

| Source | Total windows | REST windows |
| --- | ---: | ---: |
| Original movement recording | 13,188 | 353 |
| First additional REST-only recording | 537 | 537 |
| Second additional REST-only recording | 947 | 947 |
| Combined dataset | 14,672 | 1,837 |

Source arrays, labels, and hashes match the combination. No additional REST
recording was omitted. The outer test contains 239 movement-session REST windows
from events 0 and 2, leaving event 1's 114 windows in the training pool.

## What went wrong

The initial group-aware calibration split moved all 114 remaining movement-session
REST windows into calibration. Fitting therefore saw 1,484 REST windows, but all
came from separate REST-only recordings. It saw **zero** REST windows from the
movement session represented in the test. Aggregate class counts hid this failure.

The correction prioritizes fitting coverage per source/action pair, moves only
whole calibration groups, and rechecks temporal separation. It preserves all 114
movement-session REST windows in fitting, along with both auxiliary recordings.
The corrected fitting/calibration/test sizes are 10,803 / 1,049 / 2,815. The outer
test indices are unchanged. All pairwise group and signal-overlap audits pass.

Because only one movement-session REST event remains outside the test, it cannot
be independently used for both fitting and calibration. Action and applicability
temperature fitting are therefore explicitly skipped (identity temperature 1.0);
finger temperature scaling still uses the movement-only calibration fold. More
independent REST events are needed for representative three-way separation.

## Controlled diagnostic results

Both corrected runs use identical fitting/calibration/test indices, seed 43,
60 epochs, and the same preprocessing. Only the REST loss weight differs.

| Run | REST weight | Correct REST | REST recall | Action accuracy | Joint accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Initial coverage error | 1.0 | 0/239 | 0.00% | 69.41% | 58.72% |
| Corrected coverage | 1.0 | 163/239 | 68.20% | 79.36% | 69.59% |
| Corrected coverage, doubled weight | 2.0 | 170/239 | 71.13% | 80.85% | 70.66% |

Increasing the weight helps modestly in this diagnostic: seven more REST windows,
or 2.93 percentage points of recall. It does not replace the missing training
examples or restore the website's number. The major recovery comes from fixing
coverage, not increasing the weight. Existing sampling already emphasizes core
REST events; the loss weight is an additional adjustment.

The remaining error is concentrated in one event:

| Test REST event | Weight 1.0 correct | Weight 2.0 correct |
| --- | ---: | ---: |
| 0 | 118/119 | 118/119 |
| 2 | 45/120 | 52/120 |

Movement-versus-REST recall on true movement is 99.96% and 100.00%, respectively;
that does not imply correct OPEN/CLOSE or finger decoding. With only two test
REST events and one seed, this is exploratory sensitivity evidence. Weight 2.0
is not selected as a deployment default using these test results.

## Why the website is higher

The [published March 5 report](https://alphahand.org/reports/1-m16-500/report)
reports 92.11% REST recall, 83.94% action accuracy, and 80.61% non-REST finger
accuracy for run `20260305_052744`. The local saved predictions reproduce 105/114
correct REST windows. Its REST test is event 1; the current test uses events 0/2.

That older run used seed 45, the original movement dataset alone, REST weight 1.0,
a six-class finger head, per-channel normalization without recorded per-window
centering/detrending, and no temporal purge. The new run uses the March 2-M16
reference recipe: seed 43, five active-finger classes plus applicability,
centering/detrending, additional REST data, and separated/purged folds. Both are
CNN-LSTM models, but they are not identical experiments. The published score must
remain attached to its own checkpoint and test split.

## Reproduction and preserved evidence

- `rest_coverage_diagnosis.json`: exact counts, per-event results, fold composition,
  temperatures, and artifact hashes for all three diagnostic runs.
- `initial_attempt/`: the superseded initial comparison, retained without erasure.
- `tools/check_1m16_rest_coverage.py`: regenerates the diagnostic from saved runs.
- `utils/calibration_split.py`: fitting-coverage protection shared by training.
- Regression tests cover lost REST support, coverage after temporal purging, and
  identity temperatures when calibration lacks a class.

The corrected weight-1 run remains the common-recipe research baseline. The
historical deployment reference is unchanged. No new hardware or unseen-session
validation was performed, and no files were trimmed.
