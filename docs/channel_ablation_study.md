# AlphaHand Channel Ablation Study

## Goal

Estimate which Muse 2 electrodes contribute the most useful information to the
subject-specific AlphaHand model by retraining the same CNN+LSTM recipe on
controlled channel subsets.

This is a model-input ablation study, not a physiological localization claim.
The current montage is TP9, AF7, AF8, and TP10, and the featured result remains
subject-specific to 2-M16.

## Primary Comparisons

1. Full montage baseline: train and test with all four channels.
2. Leave-one-out subsets: train and test after removing one electrode.
3. Single-channel subsets: train and test with one electrode at a time.
4. Optional pair subsets: train and test all two-electrode combinations.

Leave-one-out drops estimate whether an electrode is necessary for performance.
Single-channel accuracy estimates whether an electrode is sufficient on its own.
Both can be affected by optimizer variance, so repeated seeds are preferred
before ranking electrodes.

## Runner

Use:

```bash
python3 tools/channel_ablation_sweep.py
```

Default inputs:

- Source dataset: `Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/eeg_windows.npz`
- Base recipe: `Projects/2-M16/subjects/2-M16/winning_model/model_run/train_config.json`
- Output root: `ablation_runs/channel_importance_2m16`
- Subsets: full montage, single channels, and leave-one-out subsets.

Useful variants:

```bash
python3 tools/channel_ablation_sweep.py --dry-run
python3 tools/channel_ablation_sweep.py --epochs 5 --max-subsets 2
python3 tools/channel_ablation_sweep.py --subset-mode all,singles,leave-one-out,pairs
python3 tools/channel_ablation_sweep.py --seeds 43,44,45
```

The runner writes:

- `manifest.json`: channel plan, source dataset, train recipe, seeds.
- `datasets/<subset>/eeg_windows.npz`: generated channel-subset datasets.
- `runs/<subset>/seed_<seed>/`: Step 2 training outputs.
- `summary.csv`: machine-readable metrics and full-baseline drops.
- `summary.md`: ranked summary tables for review.

## Interpretation Standard

Report at least:

- Full-montage action accuracy and non-REST finger accuracy.
- Leave-one-out action and finger drops in percentage points.
- Single-channel action and finger accuracies.
- Number of seeds and whether rankings are stable across seeds.

For paper claims, avoid saying an electrode "contains" finger intent by itself.
Prefer phrasing such as:

> Removing TP9 produced the largest held-out accuracy drop in this
> subject-specific retraining ablation, suggesting that TP9 carried the most
> usable information for this model and dataset.

## Caveats

- The held-out windows are overlapping, so raw window counts overstate
  independent evidence.
- Retraining ablations answer a different question than post-hoc channel
  occlusion of the already trained model.
- Accuracy changes can reflect optimization variance, signal quality,
  subject-specific geometry, and class imbalance.
- Strong offline subsets should still be tested through deployment-consistent
  decoding and pseudo-live replay before making command-path claims.
