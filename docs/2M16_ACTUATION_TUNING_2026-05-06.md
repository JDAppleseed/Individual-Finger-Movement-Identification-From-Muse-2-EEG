# 2-M16 Actuation Tuning Notes - 2026-05-06

## Why window recall read too low

The old pseudo-live "would-send recall" was a window-level metric:

```text
correct emitted non-REST windows / all true non-REST windows
```

That denominator includes every overlapping 50 ms hop window, even though the hand should not necessarily receive a repeated command for every overlapping window. A servo position command persists until a new command changes that finger. A low window-level send recall can therefore reflect conservative command emission, cooldown, stability, duplicate suppression, or command persistence. It is not the same quantity as held-out action or finger accuracy.

For usability, report at least three quantities together:

- Window-level would-send precision and false REST actuation for safety.
- Event-level hit rate: whether each true movement event received at least one correct command.
- First-command latency for responsive control.

## Behavior change

Cooldown semantics are now per finger. A command to thumb no longer blocks index, middle, ring, or pinky commands during the thumb cooldown interval. The cooldown still blocks rapid reversals or repeats for the same finger.

The command shaper now applies the short hold behavior to same-finger changes only. Different fingers can actuate while the previous finger remains in its commanded position.

The host serial ceiling is now 20 commands/s, matching the 50 ms live hop. This is a transport ceiling, not a target command rate. The default model gate still uses 3-window stability and a 250 ms per-finger hold because that setting had the best safety-preserving replay profile.

The Arduino receiver now mirrors the host semantics: each finger has its own 250 ms hold timer, and servo movement updates are non-blocking. This prevents a slow or recently commanded finger from delaying command intake for another finger.

## Featured 2-M16 replay sweep

Command path:

```text
python3 tools/pseudo_live_replay.py \
  --config <temp pseudo config> \
  --infer-config <temp infer config> \
  --target-session-dir Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2 \
  --device cpu
```

The sweep used the March 19 deployment checkpoint `20260319_075520` and the featured filtered 2-M16 dataset. Outputs were written to `/tmp/alphahand_tuning_per_finger`.

| Variant | Precision | Window recall | False REST actuation | Event hit rate | Misses / events | Median first-hit latency | P95 first-hit latency | Sent windows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline_per_finger | 95.37% | 31.56% | 0.25% | 91.11% | 53 / 596 | 0.082 s | 0.377 s | 3324 |
| stability2_cd250 | 92.02% | 33.06% | 0.37% | 92.45% | 45 / 596 | 0.082 s | 0.323 s | 3608 |
| stability2_cd100 | 92.21% | 33.13% | 0.37% | 93.12% | 41 / 596 | 0.082 s | 0.343 s | 3608 |
| responsive_mid | 92.17% | 33.15% | 0.37% | 93.12% | 41 / 596 | 0.082 s | 0.343 s | 3612 |
| aggressive_probe | 83.44% | 35.06% | 1.46% | 93.96% | 36 / 596 | 0.081 s | 0.206 s | 4220 |

## Current recommendation

Use `baseline_per_finger` as the next safety-preserving behavior because it improves responsiveness substantially while preserving high precision. Treat `stability2_cd100` as the next shadow-mode candidate if more responsiveness is needed; it increases event hit rate but raises false REST actuation from 0.25% to 0.37% in this replay.

Do not use `aggressive_probe` for actuation without further safety work. It increases event hit rate slightly, but precision drops to 83.44% and false REST actuation rises to 1.46%.

## Throughput levers to test next

- Keep model thresholds fixed and test hardware transport first: serial queue depth, serial ceiling, Arduino non-blocking motion, and per-finger hold behavior should improve perceived responsiveness without changing classifier risk.
- Tune stability before thresholds. Moving from 3 to 2 stable windows increases event hit rate, but it currently drops precision below the public deployment target. Treat this as a shadow-mode candidate, not the new actuation default.
- Tune cooldown only per finger. Shorter same-finger cooldowns may improve repeated-command responsiveness, but they should be evaluated against false reversals and duplicate commands rather than raw window recall alone.
- Report website-facing metrics as a bundle: held-out action/finger accuracy, would-send precision, false REST actuation, event hit rate, and first-command latency. Do not publish window-level send recall by itself.
