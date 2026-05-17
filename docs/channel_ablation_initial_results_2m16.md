# Initial 2-M16 Channel Ablation Results

Branch: `ablation/electrode-channel-importance`  
Tooling commit: `b4ef753`  
Locked manuscript base: `c8a5f52`  
Command:

```bash
python3 tools/channel_ablation_sweep.py --epochs 60 --force --no-skip-existing --keep-going
```

Additional temporal-pair run:

```bash
python3 tools/channel_ablation_sweep.py --epochs 60 --subsets 'TP9+TP10' --keep-going
```

This first pass uses one seed (`43`) and the featured 2-M16 derived dataset. It
should be treated as an exploratory subject-specific result until repeated
across seeds and, for any candidate command-path claim, replayed through the
deployment gates.

All class-level analyses below were computed from each run's saved
`test_predictions.npz` file. Action accuracy uses the action head directly.
Finger accuracy decodes the active-finger head on true non-REST windows and is
not yet deployment-gated. Deltas are percentage-point changes relative to the
fresh full-montage retrain in this ablation branch.

## Full-Montage Baseline

| Channels | Action Acc (%) | Finger Acc Non-REST (%) |
| --- | ---: | ---: |
| TP9+AF7+AF8+TP10 | 90.96 | 86.26 |

The full-montage retrain is close to, but not identical to, the locked paper
checkpoint because this is a fresh training run.

## Leave-One-Out Importance

Leave-one-out drops estimate whether an electrode is necessary for this model
and dataset. Larger positive drops indicate stronger dependence.

| Omitted Electrode | Kept Channels | Action Drop vs All (pp) | Finger Drop vs All (pp) |
| --- | --- | ---: | ---: |
| TP10 | TP9+AF7+AF8 | 12.08 | 9.13 |
| TP9 | AF7+AF8+TP10 | 11.73 | 13.29 |
| AF7 | TP9+AF8+TP10 | 1.13 | -0.40 |
| AF8 | TP9+AF7+TP10 | 0.56 | -1.30 |

Initial interpretation: the temporal electrodes, TP9 and TP10, appear to carry
most of the usable signal in this single-seed retraining ablation. Dropping AF7
or AF8 caused little or no degradation, and finger accuracy slightly improved in
those two runs, which may reflect optimizer variance, redundancy, or noisy
frontal-channel contribution.

## Single-Channel Sufficiency

Single-channel accuracy estimates whether an electrode can carry useful
information alone.

| Electrode | Action Acc (%) | Finger Acc Non-REST (%) |
| --- | ---: | ---: |
| TP9 | 73.01 | 72.22 |
| TP10 | 72.66 | 67.80 |
| AF7 | 48.89 | 35.46 |
| AF8 | 48.33 | 36.16 |

Initial interpretation: TP9 and TP10 each preserve substantial subject-specific
classification signal on their own, while AF7 and AF8 perform much closer to a
weak model. This supports a working hypothesis that the featured 2-M16 model is
driven primarily by temporal-channel information rather than frontal-channel
information.

## Temporal-Pair Sufficiency

This run removes both frontal electrodes, AF7 and AF8, and trains/tests only on
the temporal electrodes TP9 and TP10.

| Channels | Action Acc (%) | Finger Acc Non-REST (%) | Action Drop vs All (pp) | Finger Drop vs All (pp) |
| --- | ---: | ---: | ---: | ---: |
| TP9+TP10 | 88.31 | 83.90 | 2.65 | 2.36 |

Initial interpretation: the TP9+TP10 pair preserves most of the full-montage
performance while using only half the channels. This strengthens the working
hypothesis that the useful signal in the featured 2-M16 model is concentrated in
the temporal electrodes. It does not prove AF7/AF8 are useless in general; it
only shows that this subject-specific retraining ablation did not need them to
retain most held-out accuracy.

## Class-Level Test Set Counts

The same held-out test split is used across the channel subsets.

| Label Group | Class | Test Windows |
| --- | --- | ---: |
| Action | REST | 307 |
| Action | OPEN | 921 |
| Action | CLOSE | 1073 |
| Finger | THUMB | 466 |
| Finger | INDEX | 327 |
| Finger | MIDDLE | 443 |
| Finger | RING | 365 |
| Finger | PINKY | 393 |

## Per-Action Accuracy

| Run | Overall Action (%) | REST (%) | OPEN (%) | CLOSE (%) | REST Delta (pp) | OPEN Delta (pp) | CLOSE Delta (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| All (TP9+AF7+AF8+TP10) | 90.96 | 97.72 | 94.68 | 85.83 | +0.00 | +0.00 | +0.00 |
| TP9 only | 73.01 | 70.68 | 82.19 | 65.80 | -27.04 | -12.49 | -20.04 |
| AF7 only | 48.89 | 40.39 | 47.56 | 52.47 | -57.33 | -47.12 | -33.36 |
| AF8 only | 48.33 | 29.97 | 41.15 | 59.74 | -67.75 | -53.53 | -26.10 |
| TP10 only | 72.66 | 100.00 | 83.28 | 55.73 | +2.28 | -11.40 | -30.10 |
| Drop TP9 | 79.23 | 100.00 | 67.75 | 83.13 | +2.28 | -26.93 | -2.70 |
| Drop AF7 | 89.83 | 98.37 | 92.29 | 85.27 | +0.65 | -2.39 | -0.56 |
| Drop AF8 | 90.40 | 100.00 | 91.31 | 86.86 | +2.28 | -3.37 | +1.03 |
| Drop TP10 | 78.88 | 97.72 | 75.57 | 76.33 | +0.00 | -19.11 | -9.51 |
| TP9+TP10 only | 88.31 | 100.00 | 90.34 | 83.22 | +2.28 | -4.34 | -2.61 |

Action interpretation: dropping a frontal electrode did not reduce REST
accuracy. With AF7 removed, REST accuracy increased by 0.65 pp, OPEN decreased
by 2.39 pp, and CLOSE decreased by 0.56 pp. With AF8 removed, REST increased by
2.28 pp, OPEN decreased by 3.37 pp, and CLOSE increased by 1.03 pp. Therefore,
the small overall action decrease from frontal-channel removal is not a REST
failure and is not a uniform action decline; it is driven primarily by reduced
OPEN accuracy.

The TP9+TP10-only temporal pair also keeps REST perfect in this split and loses
4.34 pp on OPEN and 2.61 pp on CLOSE. This is a modest degradation compared with
dropping either temporal electrode individually, especially the 26.93 pp OPEN
loss when TP9 is omitted and the 19.11 pp OPEN loss when TP10 is omitted.

## Per-Finger Accuracy

| Run | Overall Non-REST Finger (%) | THUMB (%) | INDEX (%) | MIDDLE (%) | RING (%) | PINKY (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All (TP9+AF7+AF8+TP10) | 86.26 | 100.00 | 96.33 | 85.10 | 58.08 | 89.06 |
| TP9 only | 72.22 | 100.00 | 80.73 | 57.79 | 47.67 | 71.25 |
| AF7 only | 35.46 | 61.59 | 11.01 | 21.44 | 7.12 | 66.92 |
| AF8 only | 36.16 | 47.85 | 0.92 | 39.73 | 8.49 | 73.28 |
| TP10 only | 67.80 | 86.48 | 67.58 | 56.88 | 54.52 | 70.48 |
| Drop TP9 | 72.97 | 75.32 | 76.45 | 83.75 | 39.73 | 86.01 |
| Drop AF7 | 86.66 | 100.00 | 88.38 | 81.94 | 70.68 | 89.57 |
| Drop AF8 | 87.56 | 100.00 | 93.88 | 83.30 | 76.99 | 82.19 |
| Drop TP10 | 77.13 | 98.71 | 73.09 | 78.56 | 39.18 | 88.55 |
| TP9+TP10 only | 83.90 | 99.57 | 81.04 | 74.72 | 70.41 | 90.59 |

## Per-Finger Deltas vs Full Montage

| Run | Overall Delta (pp) | THUMB Delta (pp) | INDEX Delta (pp) | MIDDLE Delta (pp) | RING Delta (pp) | PINKY Delta (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All (TP9+AF7+AF8+TP10) | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 | +0.00 |
| TP9 only | -14.04 | +0.00 | -15.60 | -27.31 | -10.41 | -17.81 |
| AF7 only | -50.80 | -38.41 | -85.32 | -63.66 | -50.96 | -22.14 |
| AF8 only | -50.10 | -52.15 | -95.41 | -45.37 | -49.59 | -15.78 |
| TP10 only | -18.46 | -13.52 | -28.75 | -28.22 | -3.56 | -18.58 |
| Drop TP9 | -13.29 | -24.68 | -19.88 | -1.35 | -18.36 | -3.05 |
| Drop AF7 | +0.40 | +0.00 | -7.95 | -3.16 | +12.60 | +0.51 |
| Drop AF8 | +1.30 | +0.00 | -2.45 | -1.81 | +18.90 | -6.87 |
| Drop TP10 | -9.13 | -1.29 | -23.24 | -6.55 | -18.90 | -0.51 |
| TP9+TP10 only | -2.36 | -0.43 | -15.29 | -10.38 | +12.33 | +1.53 |

Finger interpretation: the temporal electrodes remain the strongest individual
channels. TP9 alone and TP10 alone retain far more non-REST finger accuracy than
AF7 alone or AF8 alone. Removing TP9 causes a larger overall finger loss
(-13.29 pp) than removing TP10 (-9.13 pp), suggesting TP9 is the stronger
temporal contributor for this split, although TP10 alone remains highly
informative.

Frontal-channel removal does not produce a broad finger collapse. Dropping AF7
slightly improves overall non-REST finger accuracy (+0.40 pp), mainly because
RING improves by 12.60 pp while INDEX and MIDDLE decline. Dropping AF8 improves
overall non-REST finger accuracy by 1.30 pp, with RING improving by 18.90 pp and
PINKY decreasing by 6.87 pp. The TP9+TP10-only run follows the same pattern:
overall finger accuracy remains close to full montage (-2.36 pp), RING improves
by 12.33 pp, and the main finger losses are INDEX (-15.29 pp) and MIDDLE
(-10.38 pp).

These per-finger results make the channel effect more specific than the overall
accuracy table. AF7/AF8 do not appear necessary for the aggregate finger result
in this seed, but they may help some digit boundaries, especially INDEX and
MIDDLE. They may also add noise or conflicting information for RING in this
subject-specific split. This should be treated as a retraining-ablation finding,
not anatomical localization.

## Next Validation Steps

1. Repeat the same subset plan across at least three seeds, for example:

   ```bash
   python3 tools/channel_ablation_sweep.py --seeds 43,44,45 --force --no-skip-existing --keep-going
   ```

2. Add pairwise subsets with:

   ```bash
   python3 tools/channel_ablation_sweep.py --subset-mode all,singles,leave-one-out,pairs --seeds 43,44,45
   ```

3. Run deployment-consistent evaluation and pseudo-live replay for the strongest
   reduced montages before making any command-path claim.

4. Report the result as subject-specific electrode importance, not as anatomical
   localization.
