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
