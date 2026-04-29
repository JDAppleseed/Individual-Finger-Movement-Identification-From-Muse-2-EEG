# 2-M16 Live Inference Failure Investigation

Date prepared: 2026-04-29  
Incident session: `2-M16_20260428_161659_01`  
Latest live inference examined: `live_infer_20260428_161232`  
Runtime environment: `muse311`

## Executive Summary

The April 28, 2026 live-control failure was not explained by the event-label repair alone. After correcting the newly recorded session to the intended low-count sequence, pseudo-live replay on the same sitting still failed to commit to true non-REST movement.

The strongest finding is that the deployed March 19 model classified all 291 movement-only replay windows as raw REST before postprocessing. The failure therefore happens in the model action head, not only in actuation gating, cooldown, pair stability, or UI actuation state.

The live input distribution report now flags the latest live run as `shifted_low_amplitude`. Prepared live-window RMS was materially lower than the offline reference on two channels, especially TP9 and TP10:

| Channel | RMS Ratio vs Offline Reference |
| --- | ---: |
| TP9 | 0.460 |
| AF7 | 0.894 |
| AF8 | 1.016 |
| TP10 | 0.718 |

This points to a session/input-distribution shift, likely from electrode contact, placement, or signal amplitude differences in the April 28 sitting. It is not a simple hidden channel-order issue: testing all 24 channel permutations did not recover meaningful movement recall. It is also not fixed by a single quick retrain: models adapted on this one short session either stayed too conservative or became unsafe with high false REST actuation.

No new model should be promoted from this investigation. The correct next step is to collect a larger same-sitting calibration set with stronger signal quality and enough repetitions per finger/action, then retrain and accept only a model that passes pseudo-live movement recall and REST-safety gates.

## Original Symptoms

- Step 1 recorded an event sequence inconsistent with the intended protocol.
- The user reported no intentional REST event.
- The first event should have been `thumb_close`.
- The last event should have been `pinky_open`.
- Live inference around `20260428_161232` showed too little actuation and did not commit to non-REST often enough.
- The session was run through `eeglab_wrapper_ui.py`, so both the step implementation and the UI launch harness were considered possible failure points.

## Corrected Event Sequence

Corrected event files:

- `Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01/events/events.csv`
- `Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01/events/events.jsonl`

Final corrected sequence:

```text
thumb_close, thumb_open,
index_close, index_open,
middle_close, middle_open,
ring_close, ring_open,
pinky_close, pinky_open
```

First event:

```text
onset_s=20.008214994333684
duration_s=1.106765791773796
type=thumb_close
finger_id=1
action_id=2
source=keyboard
```

Last event:

```text
onset_s=105.048867078498
duration_s=1.3204107079654932
type=pinky_open
finger_id=5
action_id=1
source=keyboard
```

Validation after repair:

```text
./run_py 5_validate_events.py --session-dir Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01 --strict
./run_py muse_streaming/validate_session.py --session Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01
```

Both validators passed after the event repair.

## Step 1 Event-Marking Root Cause

The original event issue was caused by an invalid keyboard event pair being allowed to continue through the Step 1 writer path.

Observed failure mode:

- Step 1 logged the first mark as `none_open`.
- That produced `action_id=1` and `finger_id=0`.
- The downstream event writer treated the invalid open/no-finger pair as REST.
- This silently converted a user-intended movement event into a REST-like event.

Why this can happen in the UI workflow:

- Step 1 relies on keyboard capture and current finger/action state.
- Running through `eeglab_wrapper_ui.py` adds another focus layer.
- If the keyboard event arrives while no valid finger is selected, Step 1 must reject the mark instead of coercing it.

Implemented Step 1 protection:

- Invalid open/close marks with no selected finger are ignored.
- `_emit_event` drops invalid non-REST pairs instead of converting them to REST.
- The UI launch path now surfaces keyboard-capture context so this failure is easier to identify.

## April 28 Window Extraction

The corrected session was processed with `1b_extract_windows.py` under `muse311`.

Movement-only, label-gated extraction:

```text
conda run -n muse311 python 1b_extract_windows.py \
  --session-dir Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01 \
  --target-fs 256 \
  --rest-policy label_gated
```

Result:

| Class | Windows |
| --- | ---: |
| OPEN | 145 |
| CLOSE | 146 |
| REST | 0 |
| Total | 291 |

Rest-by-exclusion extraction:

```text
conda run -n muse311 python 1b_extract_windows.py \
  --session-dir Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01 \
  --target-fs 256 \
  --rest-policy rest_by_exclusion
```

Result:

| Class | Windows |
| --- | ---: |
| REST | 1,935 |
| OPEN | 145 |
| CLOSE | 146 |
| Total | 2,226 |

Finger counts in the rest-by-exclusion dataset:

| Finger | Windows |
| --- | ---: |
| NONE | 1,935 |
| THUMB | 56 |
| INDEX | 54 |
| MIDDLE | 55 |
| RING | 63 |
| PINKY | 63 |

## Pseudo-Live Replay Evidence

Pseudo-live replay was run against the corrected April 28 session using the deployment-style Step 7 path.

Deployed March 19 model on movement-only windows:

```text
target_session_dir=Projects/2-M16/subjects/2-M16/sessions/2-M16_20260428_161659_01
run_dir=Projects/2-M16/subjects/2-M16/winning_model/model_run
latency_mode=ignore
```

Raw action probabilities over the 291 true movement windows:

| Action Head Output | Mean Probability | Median | P95 |
| --- | ---: | ---: | ---: |
| REST | 0.841 | 0.859 | 0.948 |
| OPEN | 0.106 | 0.092 | 0.227 |
| CLOSE | 0.054 | 0.048 | 0.101 |
| Applicability | 0.172 | 0.154 | 0.333 |

Raw top action counts:

| Raw Top Action | Windows |
| --- | ---: |
| REST | 291 |
| OPEN | 0 |
| CLOSE | 0 |

This is the key result: the model was not producing movement logits that downstream thresholds could recover. The action head itself considered every corrected movement window REST.

## Rest-By-Exclusion Replay Comparison

The rest-by-exclusion replay includes the full sitting timeline, including implicit REST between events. The apparent action/joint accuracy for conservative models is high only because REST dominates the dataset. The deployment-relevant metric is whether movement windows would send non-REST actuation while REST remains safe.

| Model | Committed Action Acc | Committed Joint Acc | Would-Send Non-REST Recall | Would-Send Non-REST Precision | False REST Actuation | Sent Count | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `model_run` March 19 deployment | 86.25% | 86.25% | 0.00% | 0.00% | 0.155% | 3 | Too conservative, no movement recall |
| `20260404_grouptrial_rest050_t10_c05` archived | 86.43% | 86.43% | 0.00% | 0.00% | 0.052% | 1 | Too conservative, no movement recall |
| `20260428_fixed_session_adapt_rest050` merged adaptation | 86.39% | 86.21% | 0.00% | n/a | 0.000% | 0 | Too conservative, no actuation |
| `20260428_session_only_rest025` session-only aggressive | 20.80% | 17.52% | 17.18% | 15.72% | 12.92% | 318 | Unsafe, high false actuation |
| `20260428_session_only_rest050` session-only conservative | 54.49% | 51.71% | 11.34% | 22.45% | 5.37% | 147 | Unsafe, still low recall |

Interpretation:

- The deployed and archived models are safe against REST but miss all true movement actuation.
- The session-only models can be made to move, but only by producing too much false actuation.
- The merged adaptation model remained conservative and did not solve same-sitting recall.
- The April 28 sitting is too small to tune a reliable deployment model.

## Offline Training Checks

Three candidate training experiments were run to test whether same-sitting data could recover movement decoding.

| Candidate | Training Data | Rest Weight | Train Action Acc | Test Action Acc | Test Non-REST Finger Acc | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `20260428_fixed_session_adapt_rest050` | March corpus + April 28 | 0.50 | 87.16% | 82.86% | 80.09% | Offline split looked acceptable, pseudo-live recall stayed 0% |
| `20260428_session_only_rest025` | April 28 only | 0.25 | 87.32% | 21.48% | 1.64% | Overfit/unstable, unsafe in replay |
| `20260428_session_only_rest050` | April 28 only | 0.50 | 86.63% | 33.19% | 1.64% | Overfit/unstable, unsafe in replay |

The merged model's standard test metrics looked much better than its pseudo-live performance. That gap reinforces that normal offline split metrics are not enough for live-control readiness. Replay metrics must stay in the acceptance loop.

## Live Distribution Diagnosis

Latest live report:

```text
Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/live_infer_20260428_161232/live_input_distribution_report.json
```

Distribution match result:

```json
{
  "verdict": "shifted_low_amplitude",
  "reason": "Prepared live-window amplitude is materially quieter than the offline reference on multiple channels.",
  "median_rms_ratio": 0.8059538264445342,
  "min_rms_ratio": 0.45956374652767196,
  "max_rms_ratio": 1.016196880627181,
  "per_channel_rms_ratio": [
    0.45956374652767196,
    0.894000688979339,
    1.016196880627181,
    0.7179069639097293
  ],
  "low_rms_channel_count": 2
}
```

Channel mapping:

| Index | Channel | Ratio |
| ---: | --- | ---: |
| 0 | TP9 | 0.460 |
| 1 | AF7 | 0.894 |
| 2 | AF8 | 1.016 |
| 3 | TP10 | 0.718 |

This is consistent with the pseudo-live movement failure. The temporal channels, especially TP9, were substantially quieter than the offline reference distribution used to train/calibrate the deployed model.

## Ruled-Out Explanations

| Hypothesis | Result | Interpretation |
| --- | --- | --- |
| Event labels alone caused live failure | Ruled out | Corrected same-sitting replay still predicted REST on all movement-only windows. |
| UI actuation state alone caused failure | Ruled out as primary cause | Replay used the model path and still failed before actuation. UI warnings were added, but model logits are the core issue. |
| Channel order mismatch | Unlikely | Testing all 24 channel permutations did not recover meaningful recall. Best permutation was still approximately 0.34% non-REST recall/action accuracy. |
| Postprocess thresholds were too conservative | Not primary | Raw top action was REST for all movement-only windows. Thresholds cannot recover absent movement logits. |
| Need only lower REST threshold | Unsafe | Session-only low-rest-weight tuning increased actuation but also produced high false REST actuation. |
| Model artifact path issue | Partly fixed, not root cause | Temperature-path fallback was repaired for archived models; archived replay still had 0% movement recall. |

## Implemented Tooling Fixes

Tracked code changes from this investigation:

| File | Change |
| --- | --- |
| `tools/pseudo_live_replay.py` | Added `_coerce_bool` so pseudo-live runtime config can parse boolean-like settings reliably. |
| `utils/live_infer_common.py` | `resolve_temperature_path` now falls back to the model-local `temperature_scaling.json` when `train_config.json` points to a stale absolute path. |
| `3_evaluate_model.py` | Matched the temperature-path fallback used by live inference helpers. |
| `tools/merge_windows_npz.py` | Allows scalar `config` metadata mismatch when merging NPZ window datasets. |
| `tools/analyze_live_raw_inputs.py` | Distribution classifier now detects local low-amplitude channels, not only median-wide amplitude collapse. |
| `tools/live_preflight.py` | Compact preflight output now includes min/max/per-channel RMS ratios for faster diagnosis. |
| `tests/test_deployment_tooling.py` | Added coverage for the local low-amplitude live-distribution verdict. |

Step 1/UI fixes from the event-label investigation:

| File | Change |
| --- | --- |
| `1_stream_and_record.py` | Rejects invalid open/close marks with no selected finger instead of writing REST-like records. |
| `eeglab_wrapper_ui.py` | Adds launch/runtime warnings around keyboard capture and actuation overrides. |
| `7_live_infer_and_actuate.py` | Live summaries include full runtime actuation state. |
| `tests/test_stream_fix_pack.py` | Covers the invalid event-mark handling path. |
| `tests/test_ui_step7_launch_contract.py` | Covers the UI Step 7 launch contract and warnings. |

## Validation Commands

Focused tooling validation:

```text
conda run -n muse311 python -m pytest \
  tests/test_deployment_tooling.py::test_live_prediction_summary_includes_full_runtime_metrics \
  tests/test_deployment_tooling.py::test_live_raw_distribution_flags_channel_local_low_amplitude \
  tests/test_deployment_tooling.py::test_live_preflight_distribution_probe_assessment_shifted_low_amplitude \
  -q
```

Result:

```text
3 passed
```

Broader regression subset:

```text
conda run -n muse311 python -m pytest \
  tests/test_stream_fix_pack.py \
  tests/test_deployment_tooling.py::test_live_prediction_summary_includes_full_runtime_metrics \
  tests/test_deployment_tooling.py::test_live_raw_distribution_flags_channel_local_low_amplitude \
  tests/test_ui_step7_launch_contract.py \
  -q
```

Result:

```text
47 passed
```

Syntax check:

```text
conda run -n muse311 python -m py_compile \
  1_stream_and_record.py \
  3_evaluate_model.py \
  7_live_infer_and_actuate.py \
  eeglab_wrapper_ui.py \
  tools/analyze_live_raw_inputs.py \
  tools/merge_windows_npz.py \
  tools/pseudo_live_replay.py \
  tools/live_preflight.py \
  utils/live_infer_common.py
```

Result: passed.

## Why This Happened

There were two separate failure classes:

1. Event capture/serialization allowed an invalid mark to become a REST-like event.
2. Live inference saw an April 28 signal distribution that was quieter than the model's offline reference on key channels, so the action head preferred REST even during intended movement.

The first issue explains the incorrect event file. The second issue explains why fixing the event file did not make the model actuate correctly.

The most likely live-inference cause is a combination of:

- TP9/TP10 contact or placement differences in the April 28 sitting.
- The March 19 model being calibrated to stronger or differently distributed temporal-channel signal.
- A short April 28 correction dataset with too few repetitions to support robust same-session adaptation.
- Offline split metrics overestimating deployability when pseudo-live recall and REST safety are not both enforced.

## Possible Fixes and Research Directions

Recommended near-term fixes:

1. Add a live preflight rule that strongly warns, or blocks decisive live control, when `shifted_low_amplitude` is detected on two or more channels.
2. Require a short calibration replay before each live-control run: at least several open/close repetitions per finger plus intentional REST intervals.
3. Use pseudo-live replay as the deployment gate, not only offline holdout metrics.
4. Expand the session-adaptation dataset before retraining; the April 28 10-event sitting is too small.
5. Add visible UI feedback for Step 1 keyboard capture and selected finger/action state.
6. Preserve the corrected event sequence in validation reports so label drift is obvious before training.

Modeling ideas to test:

- Per-session amplitude normalization or adaptive gain correction, validated against REST false-actuation risk.
- Training-time amplitude augmentation, especially channel-local attenuation on TP9/TP10.
- A separate REST-vs-movement detector trained to be robust to low-amplitude movement.
- Model acceptance thresholds optimized jointly for movement recall and false REST actuation.
- A calibration head or lightweight fine-tuning path that uses several repetitions per class from the same sitting.
- Segment-level loss or event-level training objectives, since the deployment objective is sustained actuation, not isolated window accuracy.

Data-collection recommendations:

- Collect at least 5 to 10 repetitions for each finger/action pair in the same sitting.
- Include explicit REST baselines and natural gaps for rest-by-exclusion replay.
- Record headset fit/contact observations, especially TP9 and TP10.
- Do a quick preflight amplitude comparison before starting robot actuation.
- Do not promote any model that has 0% would-send movement recall or elevated false REST actuation on same-sitting pseudo-live replay.

## Current Decision

Do not deploy any model trained during this investigation.

Keep the March 19 deployment model as the current conservative baseline, but treat the April 28 finding as an input-quality and robustness blocker. The model can be safe and still fail to actuate. The next success criterion is a model/run combination that shows both:

- Meaningful same-sitting non-REST would-send recall.
- Low false REST actuation on rest-by-exclusion replay.

