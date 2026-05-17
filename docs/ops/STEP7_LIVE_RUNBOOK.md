# Step 7 Live Runbook

Use the UI for normal Step 7 work. Use this runbook when you need the manual CLI path for the decisive `2-M16` live run.

## Scope

This is the fail-closed, auditable manual flow for a real Step 7 live inference run.

Legacy compare/tuning sessions and superseded Step 7 outputs for `2-M16` are archived under `Projects/2-M16/subjects/2-M16/archive/step7/`. Keep them for reference only; they are not part of the active live workflow.

Normal live work should be launched through `eeglab_wrapper_ui.py` from the same Python 3.11/3.12 environment used for Muse, LSL, PySide, and the pipeline:

```bash
source .venv/bin/activate
python3 eeglab_wrapper_ui.py
```

If you prefer conda, activate your own Python 3.11/3.12 environment before launching the UI. The UI runs Step 7 with the same interpreter that launched the UI. This avoids mixing the Muse/LSL/PySide runtime with another Python environment.

## Pinned Inputs

- Config: `Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json`
- Session dir: `Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2`
- Deployment bundle: March 19 run `20260319_075520`; model, scaler, and temperature scaling resolve from the pinned config and runtime manifest
- Live stream identity: set `LSL_SOURCE_ID` to the exact headset you want
- Output dir: use a fresh `processed/live_infer_<run_tag>` directory every run

## UI Notes

- On the Step 7 page, the Step 7 `Session dir` field is the authoritative launch/preflight session.
- For `2-M16`, the UI should resolve `finger_action_model.pt` and `scaler.npz` under `processed/models/20260319_075520/`.
- Do not assume the main session selector overrides a pinned Step 7 session.
- Step 7b resolves the latest live output under that Step 7 session and prefers fresh `live_infer_<run_tag>` directories over bare or legacy names.
- `Show Step 1-style live EEG plot` is enabled by default and uses the Step 1 defaults: display FS `64.0`, redraw FPS `20.0`, 5 s window, fixed `[-200, 200] uV`, and `120 uV` channel spacing.
- `live_logging_mode=lean_decisive` is the default. It keeps preflight evidence, raw shards, `predictions.jsonl`, the runtime manifest, segment breaks, and the prediction summary. `full_audit` adds high-volume `runtime_events.jsonl`, `window_audit.jsonl`, parity capture/replay, and distribution report artifacts.
- Live inference must use `mc_passes=1`. MC dropout experiments belong outside Step 7 live deployment configs.

## Optional CLI Flow

### 1. Preflight

```bash
export CONFIG="Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json"
export SESSION_DIR="Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2"
export RUN_TAG="$(date +%Y%m%d_%H%M%S)"
export OUT_DIR="$SESSION_DIR/processed/live_infer_$RUN_TAG"
export LSL_SOURCE_ID="<live-headset-source-id>"

python3 tools/live_preflight.py \
  --config "$CONFIG" \
  --session-dir "$SESSION_DIR" \
  --out-dir "$OUT_DIR" \
  --lsl-source-id "$LSL_SOURCE_ID" \
  --probe-stream \
  --probe-distribution \
  --distribution-probe-seconds 15
```

Before launch, confirm all of the following:

- `requested_source_id` is the headset you expect
- `selected_source_id` matches `requested_source_id`
- `stream_contract_ok` is `True`
- `Distribution probe verdict` is not `catastrophic`
- `out_dir` does not already contain files
- there are no errors about missing model, scaler, temperature scaling, or deployable-run invariants

### 2. Launch

```bash
python3 7_live_infer_and_actuate.py \
  --config "$CONFIG" \
  --session-dir "$SESSION_DIR" \
  --out-dir "$OUT_DIR" \
  --lsl-source-id "$LSL_SOURCE_ID" \
  --live-logging-mode lean_decisive \
  --live-eeg-plot
```

### 3. Expected Outputs

When `no_file_io=false`, these files must exist after the run:

- `$OUT_DIR/live_infer.log`
- `$OUT_DIR/live_runtime_manifest.json`
- `$OUT_DIR/predictions.jsonl`
- `$OUT_DIR/raw/eeg_raw_shard_*.npy`
- `$OUT_DIR/segment_breaks.jsonl`
- `$OUT_DIR/live_prediction_summary.json`

In `full_audit` mode, these additional files must also exist:

- `$OUT_DIR/runtime_events.jsonl`
- `$OUT_DIR/window_audit.jsonl`
- `$OUT_DIR/live_input_distribution_report.json`
- `$OUT_DIR/parity_capture/capture_manifest.json`
- `$OUT_DIR/parity_capture/captured_windows.json`
- `$OUT_DIR/parity_report.json`

### 4. Immediate Post-Run Check

```bash
python3 - <<'PY' "$OUT_DIR"
import json, sys
from pathlib import Path
out_dir = Path(sys.argv[1])
manifest = json.loads((out_dir / "live_runtime_manifest.json").read_text())
payload = {
    "termination_reason": manifest.get("finalization", {}).get("termination_reason"),
    "required_outputs_ok": manifest.get("finalization", {}).get("required_outputs_ok"),
    "required_output_errors": manifest.get("finalization", {}).get("required_output_errors"),
    "stream_contract_ok": manifest.get("stream_contract", {}).get("contract_ok"),
    "selected_source_id": manifest.get("stream_resolution", {}).get("selected_source_id"),
    "replay_cmd": manifest.get("finalization", {}).get("post_run_commands", {}).get("replay"),
    "audit_cmd": manifest.get("finalization", {}).get("post_run_commands", {}).get("audit"),
    "live_logging_mode": manifest.get("runtime", {}).get("live_logging_mode"),
    "distribution_report_path": manifest.get("finalization", {}).get("distribution_report_path"),
    "distribution_report_write_error": manifest.get("finalization", {}).get("distribution_report_write_error"),
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
```

Healthy output should show:

- `stream_contract_ok=true`
- `required_outputs_ok=true`
- `required_output_errors=null`
- `selected_source_id` matching the pinned headset id
- `termination_reason=ok` or `termination_reason=interrupted`
- In lean mode, `distribution_report_path=null` and `distribution_report_write_error=null` are expected.

### 5. Replay (`full_audit` only)

```bash
python3 tools/replay_live_capture.py \
  --capture-dir "$OUT_DIR/parity_capture"
```

This must write `$OUT_DIR/parity_report.json`. The command returns nonzero if the capture is malformed or replay parity fails.

### 6. Audit

```bash
python3 tools/audit_live_parity.py \
  --live-dir "$OUT_DIR" \
  --write-json \
  --write-md
```

Healthy lean audit output should show:

- `evidence.completeness=complete`
- `evidence.accepted_window_parity_evidence=not_required_lean`
- `evidence.distribution_evidence=not_required_lean`
- `blocking_errors` is empty

For `full_audit`, pass `--parity-report "$OUT_DIR/parity_report.json"` and `--distribution-report "$OUT_DIR/live_input_distribution_report.json"` and require confirmed parity/distribution evidence.

## Immediate Failure Signatures

- Preflight error: `No explicit live LSL source_id is pinned`
- Preflight or startup error: non-empty `out_dir`
- Startup error: `stream_contract_mismatch`
- Startup error: `artifact_load_error`, `temperature_artifact_missing`, or `temperature_artifact_load_error`
- Final manifest shows `required_outputs_ok=false`
- `mc_passes` is not `1` in the frozen config or runtime manifest
- `full_audit` replay returns nonzero with `status=error` or `status=parity_failure`
- Audit returns nonzero with `blocking_errors`
- `full_audit` audit shows `accepted_window_parity_evidence=none` or `partial` after replay
