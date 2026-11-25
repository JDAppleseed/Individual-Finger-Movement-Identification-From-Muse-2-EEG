# AlphaHand documentation

Start with the [repository workflow and layout](repository_layout.md). The
[November finalization record](research/finalization_20261125/README.md) identifies
the current datasets, comparison protocol, restored baseline, and limitations.

| Task | Source of truth |
| --- | --- |
| Install and run | [Root README](../README.md) |
| Reproduce the public reference | [2-M16 quickstart](2M16_QUICKSTART.md) |
| Understand files and timestamps | [Data contract](spec/DATA_CONTRACT.md), [schemas](spec/SCHEMAS.md) |
| Operate capture and inference | [Runbook](ops/RUNBOOK.md), [Step 7](ops/STEP7_LIVE_RUNBOOK.md) |
| Explain historical model selection | [Selection audit](2M16_MODEL_SELECTION_AUDIT.md) |
| Understand the April transfer failure | [Incident report](research/2M16_live_inference_failure_2026-04-28.md) |
| Review proposed trimming | [Removal review](maintenance/REMOVAL_REVIEW.md) |
| Plan the separate rewrite | [Rewrite requirements](maintenance/REWRITE_REQUIREMENTS.md) |

Dated research reports describe their own experiments. They do not implicitly
replace a model, dataset, gate policy, or published metric in another report.
