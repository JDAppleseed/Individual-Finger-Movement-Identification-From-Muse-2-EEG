# 2-M16 Model Selection Audit

Date: 2026-04-24

Decision: use `20260319_075520` as the public `2-M16` deployment model and keep `20260403_grouptrial_rest050` as an offline benchmark.

The April 3 model is stronger on the standard holdout and event-level checks. It should not be the main displayed deployment model because its cleaned pseudo-live would-send precision and REST safety are worse.

## Ranking Policy

For public deployment claims:
- Would-send precision ranks first.
- False REST actuation ranks second.
- Holdout REST TPR and calibration rank next.
- Offline holdout/event-level accuracy is still important, but it does not override a deployment safety regression.

For research experiments:
- Offline holdout and event-level scores are valid targets.
- A future model should keep the April 3 offline gains while beating March 19 on pseudo-live precision and false REST actuation.

## Head-To-Head

| Metric | `20260319_075520` | `20260403_grouptrial_rest050` | Winner |
| --- | ---: | ---: | --- |
| Holdout action accuracy | `89.79%` | `91.83%` | April 3 |
| Holdout joint accuracy | `84.66%` | `86.66%` | April 3 |
| Holdout non-REST finger accuracy | `85.96%` eval / `87.01%` model card | `88.11%` | April 3 |
| Holdout REST TPR | `98.37%` | `94.79%` | March 19 |
| Holdout REST precision | `80.11%` | `84.59%` | April 3 |
| Action ECE, lower is better | `2.32%` | `3.98%` | March 19 |
| Non-REST finger ECE, lower is better | `2.73%` | `1.60%` | April 3 |
| Event-level action accuracy | `92.56%` | `95.87%` | April 3 |
| Event-level joint accuracy | `87.60%` | `93.39%` | April 3 |
| Event-level non-REST finger accuracy | `90.68%` | `94.92%` | April 3 |
| Cleaned pseudo-live committed joint | `86.64%` | `86.04%` | March 19 |
| Cleaned pseudo-live would-send precision | `93.32%` | `80.06%` | March 19 |
| Cleaned pseudo-live would-send recall | `10.57%` | `36.49%` | April 3 |
| Cleaned pseudo-live false REST actuation | `0.12%` | `6.74%` | March 19 |

## Diagnosis

The April 3 checkpoint is more aggressive. It improves offline action, joint, and event-level scores and raises would-send recall from `10.57%` to `36.49%`. That is useful for research, but it also lowers cleaned-corpus would-send precision by `13.26` percentage points and raises false REST actuation by `6.61` percentage points.

The regression has two causes:
- The April 3 model is less conservative around REST: holdout REST TPR drops from `98.37%` to `94.79%`.
- The April 3 public replay used the raw-gated Step 7 path with postprocessing disabled, while the March 19 displayed metric set uses the tuned deployment family with EMA smoothing, low action/finger thresholds, applicability gating, stability, and cooldown.

## Public Bundle Role

The current curated public bundle publishes:
- Final dataset: `Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/eeg_windows.npz`
- Featured deployment run: `Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260319_075520`
- Stable snapshot: `Projects/2-M16/subjects/2-M16/winning_model/`

The April 3 checkpoint remains useful as a historical offline comparison point, but it is not part of the current minimal public model set.

## Replacement Gate

A future model should not replace `20260319_075520` as the displayed/public deployment model unless it improves or matches:
- Cleaned pseudo-live would-send precision: at least `93.32%`
- Cleaned pseudo-live false REST actuation: at most `0.12%`
- Holdout REST TPR: at least `98.37%`, unless a lower TPR is justified by a stronger pseudo-live safety result
- Invalid committed and sent pair rates: `0.00%`

Offline improvements should still be reported, but they should be labeled as offline until they pass this deployment gate.
