# Explicit CNN-LSTM presets

All presets here are observation-only: serial is disabled, and full audit/parity
logging is enabled. They never promote models automatically.

| Preset | Purpose |
| --- | --- |
| `2-M16_march19_shadow.json` | Preserved public deployment reference, pinned to `winning_model/model_run` |
| `1-M16_november_research_shadow.json` | New fixed-recipe research run; coverage-corrected fitting; sparse REST evidence; no deployment promotion |
| `2-M16_november_research_shadow.json` | New expanded-corpus research run; no new-session validation |
| `3-M16_november_research_shadow.json` | New research run; weak finger separation and session confounding |

Use from the repository root with `./run_py 7_live_infer_and_actuate.py --config
<preset>`. New research artifacts remain local; a public checkout alone does not
contain those weights. Output paths are relative to this config folder and resolve
inside the selected session. Use a fresh `--out-dir` for repeated observations.

`march19_manifest.json` freezes the unchanged reference artifacts. New model,
normalizer, temperature, and split hashes are in
`docs/research/finalization_20261125/comparison.json`. The original winning config
is preserved independently; nothing here changes its historical command policy.

`1-M16_march5_reference_manifest.json` also pins the published 1-M16 offline
reference with 105/114 correct REST test windows. Its dataset, split, preprocessing,
and six-class finger head differ from the November common-recipe experiment.
