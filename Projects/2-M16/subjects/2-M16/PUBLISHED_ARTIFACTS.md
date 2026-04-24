# Published Artifacts

This file defines the curated `2-M16` publication set. Paths are relative to `Projects/2-M16/subjects/2-M16/`.

Expected size:
- About `225 MB` on this checkout.
- Budget about `244 MB` before Git compression depending on filesystem block accounting.
- Logical file size for the curated files is about `105 MB`.

## Source Sessions

| Session | Raw shards | Raw bytes | Raw tree SHA256 | Windows | Source role |
| --- | ---: | ---: | --- | ---: | --- |
| `2-M16_20260216_150056_01` | 1,387 | 65,134,012 | `7f4d72a431b6effcc7221e52fcc029c0ff39836abf3a6f8f18c3d5d7ca4da30c` | 10,266 | Core movement session |
| `2-M16_20260315_145838_01` | 135 | 6,459,480 | `d8b6bb0fc64063315ddad989393a61e10657e4e4869be2fce09bb00c77b8a30f` | 1,059 | Auxiliary quiet REST session |
| `2-M16_20260317_190134` | 268 | 12,854,296 | `784e4b96a7cd02f62f04053a1f3d5d5ecc7f7ad33cf91816b29c0cd168a1e119` | 1,644 | Mixed REST/movement session |

The raw tree hash is computed by hashing each `raw/eeg_raw_shard_*.npy`, then hashing `relative_path + NUL + file_sha256 + newline` for all shards in sorted order. The session `manifest.json` files contain the full shard path and sample-range lists.

Published for each source session:
- `manifest.json`
- `meta.json`
- `run_meta.json`
- `timebase_report.json`
- `events/events.jsonl`
- `events/events.csv` when present
- `logs/resolved_settings.json`
- `logs/session_state.json`
- `raw/eeg_raw_shard_*.npy`
- `processed/eeg_windows.npz`
- `processed/extraction_report.json`

## Source Session Hashes

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `sessions/2-M16_20260216_150056_01/manifest.json` | 417,111 | `7394d0344fff6ebca63eaa34a8528140157c16266c41a8470c62c2a8faf53a18` |
| `sessions/2-M16_20260216_150056_01/meta.json` | 1,599 | `308af86b5a2df7da3886c9432c47f4ea0ed154a3e960455479cd080794020107` |
| `sessions/2-M16_20260216_150056_01/run_meta.json` | 617 | `a301be6c7cbf2c2e6feb1c9270f8f77488ffe6a150484ecf8997ae8aba42d8bb` |
| `sessions/2-M16_20260216_150056_01/timebase_report.json` | 416,689 | `c09409d0fad3fe98786d72ca8a63a6916133c96961833e3418ea12450ef5f368` |
| `sessions/2-M16_20260216_150056_01/events/events.jsonl` | 335,602 | `2b741cdbc75bede1a6c6c7fe27b183bacb71b00e5281a3b8ed9241925abe29ee` |
| `sessions/2-M16_20260216_150056_01/events/events.csv` | 95,534 | `b9ce0d1eae58742b29b8bcd8fc736e9d7d32e9410fd4e78a9c59a2e911afe1dd` |
| `sessions/2-M16_20260216_150056_01/logs/resolved_settings.json` | 977 | `7528bc4f5b186bec3647773dfd8e663e887d5aa9e9b5df26ab5bd5e888250e0d` |
| `sessions/2-M16_20260216_150056_01/logs/session_state.json` | 431 | `655104848e8650ff994b9317c4d722fbd97ff8db1c3cdcfd1985eeb9628ef370` |
| `sessions/2-M16_20260216_150056_01/processed/eeg_windows.npz` | 9,472,665 | `f53c1f59078f8e55fafbfff56d9f23d4909cfaeb668a5f03361caf59be996c61` |
| `sessions/2-M16_20260216_150056_01/processed/extraction_report.json` | 1,230 | `6bbd96ed5bb582ff8184ff2dd93693bad86175547178bff0016e3e2d40da65ef` |
| `sessions/2-M16_20260315_145838_01/manifest.json` | 40,575 | `6f41817b4cdd777a4350e36d3e990961ed51385cd2b1bc3163aed7a9a1ad0deb` |
| `sessions/2-M16_20260315_145838_01/meta.json` | 1,644 | `0d19ab90edb899d99b7891f1afeaf322fcf7cfd35911a397651d84e457f0c7fc` |
| `sessions/2-M16_20260315_145838_01/run_meta.json` | 718 | `33d87f3e4ddcf42a650c412bdace3c00501732bd5e7604f4ed7cfc33eb980c4b` |
| `sessions/2-M16_20260315_145838_01/timebase_report.json` | 40,156 | `eb83d68bbebe1ef28a019a004f69e904ef0fd792e16ea5d77733a1564b25e3fb` |
| `sessions/2-M16_20260315_145838_01/events/events.jsonl` | 2,102 | `0b83d5eecb1de17fc91f540e5f6318e45161bef4b7b8ca7f5d569e66800e5d6a` |
| `sessions/2-M16_20260315_145838_01/events/events.csv` | 605 | `1efb4033b323c496cd04b7437112122218a15cf58536b6eeb1798b83217163de` |
| `sessions/2-M16_20260315_145838_01/logs/resolved_settings.json` | 977 | `ff75701365ffc4785458b2527f9ad3168a4152a16dd3a8813d9c438c26c03f17` |
| `sessions/2-M16_20260315_145838_01/logs/session_state.json` | 427 | `e9625c7fe25bc1b6725ea3eb6cc019f72297b69ef1f890719b710b5823859074` |
| `sessions/2-M16_20260315_145838_01/processed/eeg_windows.npz` | 963,524 | `2259862b5ce3d126bb2bdaea23a01bbbc4031a07e6d13664e5696d5a8b92c425` |
| `sessions/2-M16_20260315_145838_01/processed/extraction_report.json` | 1,225 | `a6ba66396932ecff05c2c3d8027fe71059e2955023f341a4cac53c9e0b20f11e` |
| `sessions/2-M16_20260317_190134/manifest.json` | 80,260 | `984f02e70313b7c52c38e90b08603b2cbacb832e0bf5e3b4e962cc7a173629f9` |
| `sessions/2-M16_20260317_190134/meta.json` | 1,627 | `5c11898a720f314a53102a0243647968146b1b5411885ee142462ad7a8c2983c` |
| `sessions/2-M16_20260317_190134/run_meta.json` | 703 | `285a2b940f791b3444b2a1ea7d57aacc6009507a40fcc4e94e0c7c428099ec9c` |
| `sessions/2-M16_20260317_190134/timebase_report.json` | 79,838 | `db8edd711c46217a27231175c27f371429191fa17c1ec1681589dd25e3a0e0b8` |
| `sessions/2-M16_20260317_190134/events/events.jsonl` | 8,558 | `29bd25b045a9a64598f8677413ed75f7cabf1981bbebedb366fdb356e3ffac8c` |
| `sessions/2-M16_20260317_190134/logs/resolved_settings.json` | 965 | `bd2cc518dcdff902f6a05b82e772e335d1b8924d40315c91f25ddee952307127` |
| `sessions/2-M16_20260317_190134/logs/session_state.json` | 421 | `dc796e519fb51b774dde3cd399a60366b11f88586a8c4bd6fe373c24813a097a` |
| `sessions/2-M16_20260317_190134/processed/eeg_windows.npz` | 1,508,424 | `284e0f22cfdb2d6ed78f6fb00e787ca46b1d48235801e85a0f14bbaf51b9d5fc` |
| `sessions/2-M16_20260317_190134/processed/extraction_report.json` | 1,222 | `3a79492bea4b44c4a07aea429bc7aa0f0872472ef80d94783ca6c34c2f65bbe7` |

## Final Dataset And Pruning Rule

Final dataset path:

```text
sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/eeg_windows.npz
```

Build rule:
- Source order: `2-M16_20260216_150056_01`, `2-M16_20260317_190134`, `2-M16_20260315_145838_01`.
- Remove REST event IDs `0`, `1`, and `2` from `2-M16_20260216_150056_01`.
- Keep all windows from the other two source sessions.

Expected final dataset:
- `X=(12447,64,4)`
- Actions: `REST=2404`, `OPEN=4814`, `CLOSE=5229`
- Fingers: `NONE=2404`, `THUMB=2252`, `INDEX=1742`, `MIDDLE=2051`, `RING=1922`, `PINKY=2076`
- Sessions: `2-M16_20260216_150056_01=9744`, `2-M16_20260315_145838_01=1059`, `2-M16_20260317_190134=1644`
- Channels: `TP9`, `AF7`, `AF8`, `TP10`

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/session_config.json` | 4,670 | `4db62ee7f6033521ae8c80e3af65f59296c8d5595c6e0f538f64e5944d1d7721` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/eeg_windows.npz` | 11,444,609 | `8d823df8926d113ecff45412a7d170f2b7feffa1a7aff1cf410429bdc34c7914` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/filter_manifest.json` | 645 | `6b7ef550c2fb2c9e2ccaf276d4c281f5b92e023717e79147d4cd55319130718b` |

## Reference Run

Reference run ID: `20260403_grouptrial_rest050`

| Path | Bytes | SHA256 |
| --- | ---: | --- |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260403_grouptrial_rest050/finger_action_model.pt` | 117,453 | `8bf5b652a215cde72097cdb21d1ca499e9c18d78e566c7e2c424b6fe804a6f57` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260403_grouptrial_rest050/scaler.npz` | 1,854 | `189edceb4c7a1b9eb55ad70b4992483140b9e5e48a8288026ec99e7811d3246d` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260403_grouptrial_rest050/temperature_scaling.json` | 554 | `4f5baa75caec22d0ec165e255758b66ebb623cee03f27f80eaab1094cd032f77` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260403_grouptrial_rest050/test_predictions.npz` | 96,248 | `83a498ece6a55111600cc6c984f99a555b1a35b2271e651d50c91e530c34e5cf` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260403_grouptrial_rest050/metrics.json` | 1,744 | `2f075bd732f309756743d893098688983b63f7d04246a6a932c04197e58fc129` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/models/20260403_grouptrial_rest050/train_config.json` | 5,557 | `efd319d85546b98ef0f724dbfedaac003817bc9302ac93a40aa0b833bfd7ab2e` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/reports/20260403_grouptrial_rest050/eval_manifest.json` | 24,246 | `42027752c5fb2eb7039eaee2740d18e4487bd4d516e70c4d63b6e0a3fa4c5756` |
| `sessions/combined_20260319_081200_pruned_rest_events_0_1_2/processed/reports/20260403_grouptrial_rest050/report.html` | 10,149 | `b0439d4ffb709796544497abf2955f79ae7b8461826e0cf9e4e5c3a06058cd2f` |
| `winning_model/winning_model_manifest.json` | 834 | `b2e116a6a1fb2034aa0543f48f1cfaee80e1860e8e903be173892cfe9d908eae` |

The report images `action_confusion.png`, `finger_confusion.png`, and `eval_UNKNOWN.png` are also published in both the canonical report directory and the `winning_model/session_report/` snapshot.

Reference metrics from `winning_model/session_report/eval_manifest.json`:
- Action accuracy: `91.83%`
- Joint action+finger accuracy: `86.66%`
- Non-REST finger accuracy: `88.11%`
- REST true positive rate: `94.79%`

## Excluded

Intentionally excluded from publication:
- `archive/`
- Full `processed/pseudo_live/` and `winning_model/pseudo_live/` replay logs, CSVs, and JSONL files
- Live inference run directories such as `processed/live_infer_*`
- Exploratory `outputs/` event-space HTML files
- Exploratory `topomaps/` outputs
- `.DS_Store`
- Unrelated projects and local-only subject folders

These exclusions keep the clone entrypoint focused on validating the published data, training a comparable model, and testing against the current reference artifacts.
