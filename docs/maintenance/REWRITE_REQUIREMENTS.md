# Requirements for the later lightweight rewrite

The rewrite must start in a separate folder after this repository's finalization
is pushed. This cleanup does not start that rewrite. Use the finalization commit
as a frozen reference and migrate only behaviors backed by contracts and fixtures.

## Essential processes to preserve

1. **Capture and timebase:** lossless raw samples; authoritative JSONL events;
   explicit clock domains; monotonic timestamps; packet loss, reconnect, restart,
   and backpressure accounting. Preserve channel labels and units.
2. **Extraction:** 256 Hz, four ordered channels, 250 ms windows, 50 ms hop;
   explicit gap/interpolation, overlap, REST, and artifact policies. Source
   session/event identity must survive every transformation.
3. **Learning:** the CNN-LSTM with action, active-finger, and applicability heads;
   saved centering/detrending and channel normalizer; fitting-only preprocessing;
   group-separated fitting/calibration/test folds, temporal purging, and saved
   indices. Audit per-session/per-class coverage as well as leakage: a grouped
   calibration split must not consume the only fitting event for a class. Skip
   unsupported temperature fitting explicitly instead of claiming calibrated
   confidence for missing classes. Never fall back silently to randomly split overlapping windows.
4. **Evaluation:** raw predictions, optional explicitly named postprocessing,
   session/gap state resets, per-finger results, REST errors, event summaries, and
   denominator-specific command metrics. Distinguish full-corpus replay, event
   holdout, unseen-session evaluation, and actual live hardware observations.
5. **Live parity:** use the same model artifact loader, channel ordering,
   normalization, calibration, and decoding offline and online. Replay should be
   a first-class source feeding the same decision pipeline as LSL.
6. **Command control:** REST/NONE semantics, applicability, signal-quality and
   latency checks, stability, cooldown, repeats, per-finger shaping, bounded serial
   writes, and immediate shutdown. Shadow mode must never open serial hardware.
7. **Provenance:** immutable data/model/config/split hashes, code/environment
   identity, explicit experiment manifests, and reproducible aggregate reports.
   No timestamped folder should become a default merely because it is newest.
8. **Operator experience:** one validated config schema, clear session/subject/run
   selection, actionable errors, subprocess lifecycle management, and a compact
   UI over the same command layer. Avoid duplicating logic in UI callbacks.

## Suggested small architecture

Use one package with `capture`, `dataset`, `model`, `evaluation`, and `runtime`
modules; a single CLI; and a thin optional UI. Put experiment presets and report
templates outside runtime code. Use adapters for LSL, replay, and serial so tests
need no hardware. Preserve old artifact loading through a small compatibility
adapter rather than importing the old application wholesale.

## Acceptance gates

- Load the frozen March snapshot and reproduce deterministic reference logits,
  decoded labels, and replay decisions within documented numerical tolerances.
- Reproduce the November dataset counts and split/interval audits.
- Pass fixtures for gaps, reordered channels, partial packets, rest-only input,
  reconnects, stale windows, and clean serial-free shutdown.
- Reproduce aggregate subject results from cached predictions and exact indices.
- Demonstrate a separately recorded session before claiming improved live utility.

## Outstanding scientific limitations to carry forward

1-M16 has sparse movement-session REST evidence. 3-M16 finger coverage is
confounded with recording session. 2-M16 has outcome-informed historical REST
pruning and a documented cross-sitting failure. Historical calibration folds are
not retroactively certified by fixing the code. The May command summary lacks its
complete original replay evidence chain and includes development-corpus data.
Do not erase these limitations during migration or relabel old tests as untouched.

Measure end-to-end physical latency separately from model computation and
successful-event replay detection delay. Treat future physiological artifact
controls and acquisition-protocol documentation as research requirements.
