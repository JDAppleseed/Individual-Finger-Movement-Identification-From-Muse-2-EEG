# Step 7 Live Runbook

## Scope

Use this runbook for the next real Step 7 live inference run. The goal is an explicit, fail-closed, auditable run that can settle accepted-window inference parity on captured live data.

## Pin These Inputs

- Config: `Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json`
- Session dir: `Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2`
- Model: pinned in the config and runtime manifest
- Scaler: pinned in the config and runtime manifest
- Temperature scaling: resolved beside the pinned model and required at startup
- Live stream identity: set `LSL_SOURCE_ID` for the actual headset before launch
- Output dir: use a fresh per-run directory every time

## Preflight

```bash
export CONFIG="/Users/jonathandavanzo/Desktop/Individual-Finger-Movement-Identification-From-Muse-2-EEG/Projects/2-M16/subjects/2-M16/winning_model/configs/infer.json"
export SESSION_DIR="/Users/jonathandavanzo/Desktop/Individual-Finger-Movement-Identification-From-Muse-2-EEG/Projects/2-M16/subjects/2-M16/sessions/combined_20260319_081200_pruned_rest_events_0_1_2"
export RUN_TAG="$(date +%Y%m%d_%H%M%S)"
export OUT_DIR="$SESSION_DIR/processed/live_infer_$RUN_TAG"
export LSL_SOURCE_ID="<live-headset-source-id>"

python3 tools/live_preflight.py \
  --config "$CONFIG" \
  --session-dir "$SESSION_DIR" \
  --out-dir "$OUT_DIR" \
  --lsl-source-id "$LSL_SOURCE_ID" \
  --probe-stream
```

Preflight must show all of the following before launch:

- `requested_source_id` is the value you expect for this headset
- `selected_source_id` matches `requested_source_id`
- `stream_contract_ok` is `True`
- `out_dir` does not already exist with files in it
- no errors about missing model, scaler, temperature scaling, or deployable run invariants

## Live Run

```bash
python3 7_live_infer_and_actuate.py \
  --config "$CONFIG" \
  --session-dir "$SESSION_DIR" \
  --out-dir "$OUT_DIR" \
  --lsl-source-id "$LSL_SOURCE_ID" \
  --parity-capture-enabled \
  --parity-capture-max-windows 128 \
  --parity-capture-flush-every 1
```

## Expected Outputs

These files must exist after the run when `no_file_io=false`:

- `$OUT_DIR/live_infer.log`
- `$OUT_DIR/live_runtime_manifest.json`
- `$OUT_DIR/predictions.jsonl`
- `$OUT_DIR/window_audit.jsonl`
- `$OUT_DIR/segment_breaks.jsonl`
- `$OUT_DIR/live_prediction_summary.json`
- `$OUT_DIR/parity_capture/capture_manifest.json`
- `$OUT_DIR/parity_capture/captured_windows.json`

## Immediate Post-Run Checks

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
}
print(json.dumps(payload, indent=2, sort_keys=True))
PY
```

Healthy output looks like:

- `stream_contract_ok` is `true`
- `required_outputs_ok` is `true`
- `required_output_errors` is `null`
- `selected_source_id` matches the pinned headset id
- `termination_reason` is `ok` or `interrupted`

## Replay

```bash
python3 tools/replay_live_capture.py \
  --capture-dir "$OUT_DIR/parity_capture"
```

This must write `$OUT_DIR/parity_report.json`. The command returns nonzero if the capture is malformed or if replay parity fails.

## Audit

```bash
python3 tools/audit_live_parity.py \
  --live-dir "$OUT_DIR" \
  --parity-report "$OUT_DIR/parity_report.json" \
  --write-json \
  --write-md
```

Healthy audit output after replay looks like:

- `evidence.completeness` is `complete`
- `evidence.accepted_window_parity_evidence` is `confirmed`
- `blocking_errors` is empty

The audit may still confirm pre-inference window/alignment loss. That does not invalidate parity capture if replay evidence is complete.

## Immediate Failure Signatures

- Preflight error: `No explicit live LSL source_id is pinned`
- Preflight or startup error: non-empty `out_dir`
- Startup error: `stream_contract_mismatch`
- Startup error: `artifact_load_error`, `temperature_artifact_missing`, or `temperature_artifact_load_error`
- Final manifest shows `required_outputs_ok=false`
- Replay returns nonzero with `status=error` or `status=parity_failure`
- Audit returns nonzero with `blocking_errors`
- Audit shows `accepted_window_parity_evidence=none` or `partial` after replay
