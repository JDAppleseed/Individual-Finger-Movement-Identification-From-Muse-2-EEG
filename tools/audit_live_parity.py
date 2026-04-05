#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_live_predictions import summarize_records
from utils.live_parity import summarize_counter_rows, write_json
from utils.runtime_utils import now_utc_iso


def _load_jsonl(path: Path, *, label: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, errors
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception as exc:
                errors.append(
                    {
                        "path": str(path),
                        "label": str(label),
                        "line_no": int(line_no),
                        "error": str(exc),
                    }
                )
                continue
            if not isinstance(payload, dict):
                errors.append(
                    {
                        "path": str(path),
                        "label": str(label),
                        "line_no": int(line_no),
                        "error": "JSONL row must decode to an object",
                    }
                )
                continue
            payload["_line_no"] = int(line_no)
            rows.append(payload)
    return rows, errors


def _load_json_object(path: Path, *, label: str) -> tuple[dict[str, Any], list[str]]:
    if not path.exists():
        return {}, []
    try:
        payload = json.loads(path.read_text())
    except Exception as exc:
        return {}, [f"{label} parse error at {path}: {exc}"]
    if not isinstance(payload, dict):
        return {}, [f"{label} must be a JSON object: {path}"]
    return payload, []


def _load_prediction_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if not path.exists():
        return rows, errors
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            text = raw_line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception as exc:
                errors.append(
                    {
                        "line_no": int(line_no),
                        "error": str(exc),
                    }
                )
                continue
            if isinstance(payload, dict):
                payload["_line_no"] = int(line_no)
                rows.append(payload)
    return rows, errors


def _parity_evidence_status(
    parity: dict[str, Any],
) -> tuple[str, str]:
    if not isinstance(parity, dict) or not parity:
        return "none", "unknown"
    required_sections = [
        parity.get("preprocessed_tensor_values"),
        parity.get("logits"),
        parity.get("probabilities"),
        parity.get("decoded_outputs"),
    ]
    if all(isinstance(section, dict) and "ok" in section for section in required_sections):
        passed = all(section.get("ok") is True for section in required_sections)
        return "confirmed", ("pass" if passed else "fail")
    return "partial", "unknown"


def _safe_counter(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = int(raw)
        except Exception:
            continue
    return out


def _merge_counter(primary: dict[str, int], fallback: dict[str, int]) -> dict[str, int]:
    return dict(primary) if primary else dict(fallback)


def _sum_counter_keys(counter: dict[str, int], keys: Iterable[str]) -> int:
    return sum(int(counter.get(str(key), 0)) for key in keys)


def _status(
    *,
    confirmed: bool,
    ruled_out: bool,
) -> str:
    if confirmed:
        return "confirmed"
    if ruled_out:
        return "ruled_out"
    return "still_plausible"


def _top_reason(counter: dict[str, int]) -> Optional[tuple[str, int]]:
    if not counter:
        return None
    return max(counter.items(), key=lambda item: int(item[1]))


def _parse_partial_drop_counts(paths: list[Path]) -> dict[str, Any]:
    total = 0
    hits: list[dict[str, Any]] = []
    pattern = re.compile(r"partial_dropped(?:=|:)\s*(\d+)")
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line_no, line in enumerate(handle, start=1):
                match = pattern.search(line)
                if match is None:
                    continue
                count = int(match.group(1))
                total += count
                hits.append(
                    {
                        "path": str(path),
                        "line_no": int(line_no),
                        "partial_dropped": int(count),
                    }
                )
    return {
        "total_partial_dropped": int(total),
        "hits": hits[:50],
    }


def audit_live_dir(
    *,
    live_dir: Path,
    connector_logs: list[Path],
    parity_report_path: Optional[Path],
) -> dict[str, Any]:
    predictions_path = live_dir / "predictions.jsonl"
    summary_path = live_dir / "live_prediction_summary.json"
    runtime_manifest_path = live_dir / "live_runtime_manifest.json"
    window_audit_path = live_dir / "window_audit.jsonl"
    segment_break_path = live_dir / "segment_breaks.jsonl"

    predictions, prediction_parse_errors = _load_prediction_rows(predictions_path)
    blocking_errors: list[str] = []
    predictions_summary = (
        summarize_records(predictions).get("summary", {}) if predictions else {}
    )
    saved_summary, summary_errors = _load_json_object(
        summary_path, label="live_prediction_summary"
    )
    runtime_manifest, runtime_manifest_errors = _load_json_object(
        runtime_manifest_path, label="live_runtime_manifest"
    )
    parity_report = (
        _load_json_object(parity_report_path, label="parity_report")[0]
        if parity_report_path is not None and parity_report_path.exists()
        else _load_json_object(
            live_dir / "parity_report.json",
            label="parity_report",
        )[0]
    )
    parity_report_errors = (
        _load_json_object(parity_report_path, label="parity_report")[1]
        if parity_report_path is not None and parity_report_path.exists()
        else _load_json_object(
            live_dir / "parity_report.json",
            label="parity_report",
        )[1]
    )
    window_rows, window_parse_errors = _load_jsonl(
        window_audit_path,
        label="window_audit",
    )
    segment_break_rows, segment_break_parse_errors = _load_jsonl(
        segment_break_path,
        label="segment_breaks",
    )
    connector_drop_info = _parse_partial_drop_counts(connector_logs)
    if prediction_parse_errors:
        blocking_errors.append(
            f"predictions.jsonl contains {len(prediction_parse_errors)} malformed line(s)"
        )
    if window_parse_errors:
        blocking_errors.append(
            f"window_audit.jsonl contains {len(window_parse_errors)} malformed line(s)"
        )
    if segment_break_parse_errors:
        blocking_errors.append(
            f"segment_breaks.jsonl contains {len(segment_break_parse_errors)} malformed line(s)"
        )
    blocking_errors.extend(summary_errors)
    blocking_errors.extend(runtime_manifest_errors)
    blocking_errors.extend(parity_report_errors)

    artifact_presence = {
        "predictions_jsonl": bool(predictions_path.exists()),
        "live_prediction_summary_json": bool(summary_path.exists()),
        "live_runtime_manifest_json": bool(runtime_manifest_path.exists()),
        "window_audit_jsonl": bool(window_audit_path.exists()),
        "segment_breaks_jsonl": bool(segment_break_path.exists()),
        "parity_report_json": bool(parity_report),
    }
    any_evidence = any(artifact_presence.values())
    modern_core_present = all(
        artifact_presence[key]
        for key in (
            "predictions_jsonl",
            "live_prediction_summary_json",
            "live_runtime_manifest_json",
            "window_audit_jsonl",
            "segment_breaks_jsonl",
            "parity_report_json",
        )
    )
    if not any_evidence:
        evidence_completeness = "none"
    elif modern_core_present:
        evidence_completeness = "complete"
    else:
        evidence_completeness = "partial"
    evidence_limitations: list[str] = []
    if not artifact_presence["live_runtime_manifest_json"]:
        evidence_limitations.append(
            "Missing live_runtime_manifest.json, so stream identity and artifact provenance cannot be conclusively audited."
        )
    if not artifact_presence["window_audit_jsonl"]:
        evidence_limitations.append(
            "Missing window_audit.jsonl, so accepted-vs-dropped candidate-window evidence is incomplete."
        )
    if not artifact_presence["segment_breaks_jsonl"]:
        evidence_limitations.append(
            "Missing segment_breaks.jsonl, so segment reset evidence is incomplete."
        )
    if not artifact_presence["parity_report_json"]:
        evidence_limitations.append(
            "Missing parity_report.json, so accepted-window inference parity remains unproven."
        )
    parity = parity_report.get("parity", {}) if isinstance(parity_report, dict) else {}
    parity_evidence_status, parity_evidence_result = _parity_evidence_status(parity)

    candidate_window_count = int(
        saved_summary.get(
            "candidate_window_count",
            len(window_rows),
        )
    )
    accepted_window_count = int(
        saved_summary.get(
            "accepted_window_count",
            predictions_summary.get(
                "valid_window_count",
                sum(str(row.get("status") or "") == "accepted" for row in window_rows),
            ),
        )
    )
    alignment_fail_count = int(
        predictions_summary.get(
            "alignment_fail_count",
            saved_summary.get("alignment_fail_count", 0),
        )
    )
    if candidate_window_count <= 0 and accepted_window_count > 0:
        candidate_window_count = int(accepted_window_count + alignment_fail_count)
    dropped_window_reason_counts = _merge_counter(
        _safe_counter(saved_summary.get("dropped_window_reason_counts")),
        summarize_counter_rows(
            [row for row in window_rows if str(row.get("status") or "") == "dropped"],
            "drop_reason",
        ),
    )
    segment_break_reason_counts = _merge_counter(
        _safe_counter(saved_summary.get("segment_break_reason_counts")),
        summarize_counter_rows(segment_break_rows, "reason"),
    )
    actuation_suppressed_counts = _merge_counter(
        _safe_counter(saved_summary.get("actuation_suppressed_counts")),
        _safe_counter(predictions_summary.get("actuation_suppressed_reason_counts")),
    )
    quality_bad_reason_counts = _merge_counter(
        _safe_counter(saved_summary.get("quality_bad_reason_counts")),
        _safe_counter(predictions_summary.get("quality_bad_reason_counts")),
    )
    masked_channel_counts = _merge_counter(
        _safe_counter(saved_summary.get("masked_channel_counts")),
        _safe_counter(predictions_summary.get("masked_channel_counts")),
    )

    stream_resolution = runtime_manifest.get("stream_resolution", {})
    stream_contract = runtime_manifest.get("stream_contract", {})
    runtime = runtime_manifest.get("runtime", {})
    artifacts = runtime_manifest.get("artifacts", {})
    pair_stability_count = int(actuation_suppressed_counts.get("pair_stability", 0))
    uncertainty_gate_count = int(
        actuation_suppressed_counts.get("uncertainty_gate", 0)
    )
    alignment_drop_count = _sum_counter_keys(
        dropped_window_reason_counts,
        [
            "gap_exceeds_threshold",
            "end_gap_exceeds_threshold",
            "start_gap_exceeds_threshold",
            "non_monotonic",
        ],
    )
    live_quality_mutation_count = int(
        predictions_summary.get("masked_window_count", 0)
        or saved_summary.get("masked_window_count", 0)
    )
    quality_bad_window_count = int(
        predictions_summary.get("window_quality_bad_count", 0)
        or saved_summary.get("window_quality_bad_count", 0)
    )

    parity = parity_report.get("parity", {}) if isinstance(parity_report, dict) else {}
    prepared_parity_ok = bool(
        isinstance(parity.get("preprocessed_tensor_values"), dict)
        and parity["preprocessed_tensor_values"].get("ok") is True
    )
    logits_parity_ok = bool(
        isinstance(parity.get("logits"), dict) and parity["logits"].get("ok") is True
    )
    probs_parity_ok = bool(
        isinstance(parity.get("probabilities"), dict)
        and parity["probabilities"].get("ok") is True
    )
    decoded_parity_ok = bool(
        isinstance(parity.get("decoded_outputs"), dict)
        and parity["decoded_outputs"].get("ok") is True
    )

    suspected_issues = [
        {
            "issue": "stale_lsl_source_id_wrong_stream",
            "status": _status(
                confirmed=bool(
                    stream_resolution
                    and stream_resolution.get("requested_source_id")
                    and stream_resolution.get("source_id_source") == "config"
                    and (
                        bool(stream_resolution.get("recovery_used"))
                        or not bool(stream_resolution.get("selection_matched_by_source_id"))
                    )
                ),
                ruled_out=bool(
                    stream_resolution
                    and (
                        stream_resolution.get("selection_matched_by_source_id") is True
                        or not stream_resolution.get("requested_source_id")
                    )
                ),
            ),
            "evidence": {
                "requested_source_id": stream_resolution.get("requested_source_id"),
                "selected_source_id": stream_resolution.get("selected_source_id"),
                "source_id_source": stream_resolution.get("source_id_source"),
                "source_id_match_mode": stream_resolution.get("source_id_match_mode"),
                "recovery_used": stream_resolution.get("recovery_used"),
            },
        },
        {
            "issue": "strict_live_window_alignment_drops_windows",
            "status": _status(
                confirmed=bool(alignment_drop_count > 0 or alignment_fail_count > 0),
                ruled_out=bool(
                    candidate_window_count > 0
                    and alignment_drop_count == 0
                    and alignment_fail_count == 0
                ),
            ),
            "evidence": {
                "candidate_window_count": int(candidate_window_count),
                "accepted_window_count": int(accepted_window_count),
                "alignment_fail_count": int(alignment_fail_count),
                "dropped_window_reason_counts": dropped_window_reason_counts,
            },
        },
        {
            "issue": "streamer_partial_packet_drops_drive_gap_rejection",
            "status": _status(
                confirmed=bool(
                    connector_drop_info["total_partial_dropped"] > 0 and alignment_drop_count > 0
                ),
                ruled_out=bool(
                    connector_logs and connector_drop_info["total_partial_dropped"] == 0
                ),
            ),
            "evidence": connector_drop_info,
        },
        {
            "issue": "segment_break_logic_clears_state_frequently",
            "status": _status(
                confirmed=int(saved_summary.get("segment_break_count", len(segment_break_rows))) > 0,
                ruled_out=bool(
                    saved_summary.get("segment_break_count", len(segment_break_rows)) == 0
                ),
            ),
            "evidence": {
                "segment_break_count": int(
                    saved_summary.get("segment_break_count", len(segment_break_rows))
                ),
                "segment_break_reason_counts": segment_break_reason_counts,
            },
        },
        {
            "issue": "live_quality_enabled_changes_tensors_relative_to_raw_live_windows",
            "status": _status(
                confirmed=bool(
                    runtime.get("live_quality_enabled") is True
                    and (live_quality_mutation_count > 0 or quality_bad_window_count > 0)
                ),
                ruled_out=bool(
                    runtime.get("live_quality_enabled") is False
                    or (
                        runtime.get("live_quality_enabled") is True
                        and live_quality_mutation_count == 0
                        and quality_bad_window_count == 0
                    )
                ),
            ),
            "evidence": {
                "live_quality_enabled": runtime.get("live_quality_enabled"),
                "masked_window_count": int(live_quality_mutation_count),
                "window_quality_bad_count": int(quality_bad_window_count),
                "quality_bad_reason_counts": quality_bad_reason_counts,
                "masked_channel_counts": masked_channel_counts,
            },
        },
        {
            "issue": "model_scaler_calibration_artifacts_may_not_match_intended_run",
            "status": _status(
                confirmed=False,
                ruled_out=bool(
                    artifacts
                    and artifacts.get("model_sha256")
                    and artifacts.get("scaler_sha256")
                    and artifacts.get("temperature_sha256")
                    and runtime_manifest.get("deployment")
                )
                and bool(
                    str(artifacts.get("model_path", "")).startswith(
                        str(artifacts.get("run_dir", ""))
                    )
                    and str(artifacts.get("scaler_path", "")).startswith(
                        str(artifacts.get("run_dir", ""))
                    )
                ),
            ),
            "evidence": {
                "run_dir": artifacts.get("run_dir"),
                "model_path": artifacts.get("model_path"),
                "scaler_path": artifacts.get("scaler_path"),
                "temperature_path": artifacts.get("temperature_path"),
                "model_sha256": artifacts.get("model_sha256"),
                "scaler_sha256": artifacts.get("scaler_sha256"),
                "temperature_sha256": artifacts.get("temperature_sha256"),
            },
        },
        {
            "issue": "true_model_inference_parity_failure",
            "status": _status(
                confirmed=bool(parity) and not (
                    prepared_parity_ok and logits_parity_ok and probs_parity_ok and decoded_parity_ok
                ),
                ruled_out=bool(
                    parity
                    and prepared_parity_ok
                    and logits_parity_ok
                    and probs_parity_ok
                    and decoded_parity_ok
                ),
            ),
            "evidence": {
                "parity_report_present": bool(parity),
                "preprocessed_tensor_values": parity.get("preprocessed_tensor_values"),
                "logits": parity.get("logits"),
                "probabilities": parity.get("probabilities"),
                "decoded_outputs": parity.get("decoded_outputs"),
            },
        },
        {
            "issue": "commit_actuation_layer_suppresses_otherwise_valid_predictions",
            "status": _status(
                confirmed=bool(pair_stability_count > 0),
                ruled_out=bool(pair_stability_count == 0 and actuation_suppressed_counts),
            ),
            "evidence": {
                "actuation_suppressed_counts": actuation_suppressed_counts,
                "postprocess_enabled": runtime.get("postprocess_enabled"),
                "actuation_settings": runtime.get("actuation"),
            },
        },
        {
            "issue": "exact_pair_stability_required_for_actuation",
            "status": _status(
                confirmed=bool(
                    pair_stability_count > 0
                    and int(runtime.get("actuation", {}).get("actuation_stability", 2) or 2) >= 2
                ),
                ruled_out=bool(pair_stability_count == 0),
            ),
            "evidence": {
                "actuation_stability": runtime.get("actuation", {}).get(
                    "actuation_stability"
                ),
                "pair_stability_suppression_count": int(pair_stability_count),
                "top_suppression_reason": _top_reason(actuation_suppressed_counts),
            },
        },
        {
            "issue": "uncertainty_gating_is_the_primary_suppression_cause",
            "status": _status(
                confirmed=bool(
                    uncertainty_gate_count > 0
                    and uncertainty_gate_count >= pair_stability_count
                ),
                ruled_out=bool(uncertainty_gate_count == 0),
            ),
            "evidence": {
                "uncertainty_gate_suppression_count": int(uncertainty_gate_count),
                "actuation_suppressed_counts": actuation_suppressed_counts,
                "inference_backend": runtime.get("inference_backend"),
            },
        },
    ]

    layer_rows = [
        {
            "layer": "stream problems",
            "status": _status(
                confirmed=any(
                    row["issue"] in {
                        "stale_lsl_source_id_wrong_stream",
                        "streamer_partial_packet_drops_drive_gap_rejection",
                    }
                    and row["status"] == "confirmed"
                    for row in suspected_issues
                ),
                ruled_out=bool(
                    stream_resolution
                    and suspected_issues[0]["status"] == "ruled_out"
                    and suspected_issues[2]["status"] == "ruled_out"
                ),
            ),
            "evidence": {
                "stream_resolution": stream_resolution,
                "stream_contract": stream_contract,
                "connector_partial_dropped_total": connector_drop_info[
                    "total_partial_dropped"
                ],
            },
        },
        {
            "layer": "windowing/resampling problems",
            "status": _status(
                confirmed=bool(
                    suspected_issues[1]["status"] == "confirmed"
                    or suspected_issues[3]["status"] == "confirmed"
                ),
                ruled_out=bool(
                    candidate_window_count > 0
                    and suspected_issues[1]["status"] == "ruled_out"
                    and suspected_issues[3]["status"] == "ruled_out"
                ),
            ),
            "evidence": {
                "candidate_window_count": int(candidate_window_count),
                "accepted_window_count": int(accepted_window_count),
                "alignment_fail_count": int(alignment_fail_count),
                "dropped_window_reason_counts": dropped_window_reason_counts,
                "segment_break_reason_counts": segment_break_reason_counts,
            },
        },
        {
            "layer": "model/scaler/calibration inference problems",
            "status": _status(
                confirmed=bool(
                    suspected_issues[4]["status"] == "confirmed"
                    or suspected_issues[6]["status"] == "confirmed"
                ),
                ruled_out=bool(
                    suspected_issues[4]["status"] == "ruled_out"
                    and suspected_issues[5]["status"] == "ruled_out"
                    and suspected_issues[6]["status"] == "ruled_out"
                ),
            ),
            "evidence": {
                "artifacts": artifacts,
                "inference_backend": runtime.get("inference_backend"),
                "parity": parity,
            },
        },
        {
            "layer": "commit/postprocess/actuation problems",
            "status": _status(
                confirmed=bool(
                    suspected_issues[7]["status"] == "confirmed"
                    or suspected_issues[8]["status"] == "confirmed"
                ),
                ruled_out=bool(
                    actuation_suppressed_counts == {}
                    and predictions_summary.get("actuation_sent_count", 0) == 0
                ),
            ),
            "evidence": {
                "actuation_suppressed_counts": actuation_suppressed_counts,
                "pair_stability_suppression_count": int(pair_stability_count),
                "actuation_sent_count": int(
                    predictions_summary.get(
                        "actuation_sent_count",
                        saved_summary.get("actuation_sent_count", 0),
                    )
                ),
                "postprocess_enabled": runtime.get("postprocess_enabled"),
            },
        },
    ]

    executive_summary: list[str] = []
    if evidence_completeness == "none":
        executive_summary.append(
            "No Step 7 live evidence files were found; this directory cannot support a parity audit."
        )
    elif evidence_completeness == "partial":
        executive_summary.append(
            "Audit evidence is partial; missing modern artifacts limit what can be concluded."
        )
    if parity_evidence_status == "none":
        executive_summary.append(
            "No parity report is present yet, so accepted-window inference parity remains unproven."
        )
    elif parity_evidence_status == "partial":
        executive_summary.append(
            "Parity evidence is partial or malformed; accepted-window inference parity is not yet settled."
        )
    elif parity_evidence_result == "pass":
        executive_summary.append(
            "Accepted-window replay evidence is complete and supports inference parity for the captured windows."
        )
    elif parity_evidence_result == "fail":
        executive_summary.append(
            "Accepted-window replay evidence is complete and shows an inference parity failure."
        )
    if layer_rows[0]["status"] == "confirmed":
        executive_summary.append(
            "Stream layer shows unresolved risk or confirmed mismatch evidence."
        )
    if layer_rows[1]["status"] == "confirmed":
        alignment_summary = (
            dropped_window_reason_counts
            if dropped_window_reason_counts
            else {"alignment_fail_count": int(alignment_fail_count)}
        )
        executive_summary.append(
            f"Windowing drops are present before inference: {alignment_summary}."
        )
    if layer_rows[2]["status"] == "confirmed":
        executive_summary.append(
            "Inference parity is broken or live-only tensor mutation is active."
        )
    if layer_rows[3]["status"] == "confirmed":
        executive_summary.append(
            f"Commit/actuation suppression is dominated by pair stability: {pair_stability_count} windows."
        )
    if not executive_summary:
        executive_summary.append("No confirmed live parity break was detected from the available artifacts.")

    return {
        "generated_at": now_utc_iso(),
        "live_dir": str(live_dir),
        "blocking_errors": blocking_errors[:25],
        "paths": {
            "predictions": str(predictions_path),
            "summary": str(summary_path),
            "runtime_manifest": str(runtime_manifest_path),
            "window_audit": str(window_audit_path),
            "segment_breaks": str(segment_break_path),
            "parity_report": str(parity_report_path) if parity_report_path else None,
            "connector_logs": [str(path) for path in connector_logs],
        },
        "evidence": {
            "completeness": evidence_completeness,
            "accepted_window_parity_evidence": parity_evidence_status,
            "accepted_window_parity_result": parity_evidence_result,
            "available_artifacts": artifact_presence,
            "limitations": evidence_limitations,
        },
        "executive_summary": executive_summary,
        "evidence_table": layer_rows,
        "suspected_issues": suspected_issues,
        "metrics": {
            "prediction_summary": predictions_summary,
            "saved_summary": saved_summary,
            "summary_reconciliation": (
                saved_summary.get("reconciliation", {})
                if isinstance(saved_summary, dict)
                else {}
            ),
            "runtime_manifest_finalization": (
                runtime_manifest.get("finalization", {})
                if isinstance(runtime_manifest, dict)
                else {}
            ),
            "prediction_parse_errors": prediction_parse_errors[:25],
            "window_audit_parse_errors": window_parse_errors[:25],
            "segment_break_parse_errors": segment_break_parse_errors[:25],
            "candidate_window_count": int(candidate_window_count),
            "accepted_window_count": int(accepted_window_count),
            "dropped_window_reason_counts": dropped_window_reason_counts,
            "segment_break_reason_counts": segment_break_reason_counts,
            "actuation_suppressed_counts": actuation_suppressed_counts,
            "quality_bad_reason_counts": quality_bad_reason_counts,
            "masked_channel_counts": masked_channel_counts,
            "connector_partial_drop_info": connector_drop_info,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Live Parity Audit")
    lines.append("")
    lines.append("## Evidence Status")
    evidence = report.get("evidence", {})
    lines.append(
        f"- completeness: `{evidence.get('completeness', 'unknown')}`"
    )
    lines.append(
        f"- accepted_window_parity_evidence: `{evidence.get('accepted_window_parity_evidence', 'unknown')}`"
    )
    lines.append(
        f"- accepted_window_parity_result: `{evidence.get('accepted_window_parity_result', 'unknown')}`"
    )
    for limitation in evidence.get("limitations", []):
        lines.append(f"- limitation: {limitation}")
    lines.append("")
    lines.append("## Executive Summary")
    for line in report.get("executive_summary", []):
        lines.append(f"- {line}")
    lines.append("")
    if report.get("blocking_errors"):
        lines.append("## Blocking Errors")
        for error in report.get("blocking_errors", []):
            lines.append(f"- {error}")
        lines.append("")
    lines.append("## Evidence Table")
    lines.append("| Layer | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    for row in report.get("evidence_table", []):
        evidence_text = json.dumps(row.get("evidence", {}), sort_keys=True)
        lines.append(
            f"| {row.get('layer')} | {row.get('status')} | `{evidence_text}` |"
        )
    lines.append("")
    lines.append("## Ranked Root Causes")
    for idx, issue in enumerate(report.get("suspected_issues", []), start=1):
        evidence_text = json.dumps(issue.get("evidence", {}), sort_keys=True)
        lines.append(
            f"{idx}. `{issue.get('issue')}`: `{issue.get('status')}`. Evidence: `{evidence_text}`"
        )
    lines.append("")
    lines.append("## Reproduction Files")
    for key, value in report.get("paths", {}).items():
        lines.append(f"- `{key}`: `{value}`")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit Step 7 live parity artifacts by layer and suspected root cause."
    )
    parser.add_argument(
        "--live-dir",
        required=True,
        type=str,
        help="Path to processed/live_infer.",
    )
    parser.add_argument(
        "--connector-log",
        action="append",
        default=[],
        help="Optional streamer/connector log file to parse for partial_dropped counts.",
    )
    parser.add_argument(
        "--parity-report",
        type=str,
        default=None,
        help="Optional explicit parity_report.json path.",
    )
    parser.add_argument(
        "--write-json",
        action="store_true",
        help="Write live_parity_audit.json under the live dir.",
    )
    parser.add_argument(
        "--write-md",
        action="store_true",
        help="Write live_parity_audit.md under the live dir.",
    )
    return parser


def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    live_dir = Path(args.live_dir).expanduser().resolve()
    report = audit_live_dir(
        live_dir=live_dir,
        connector_logs=[Path(path).expanduser().resolve() for path in args.connector_log],
        parity_report_path=(
            Path(args.parity_report).expanduser().resolve()
            if args.parity_report
            else None
        ),
    )
    if args.write_json:
        write_json(live_dir / "live_parity_audit.json", report)
    if args.write_md:
        (live_dir / "live_parity_audit.md").write_text(
            render_markdown(report), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "evidence": report.get("evidence", {}),
                "blocking_errors": report.get("blocking_errors", []),
                "executive_summary": report.get("executive_summary", []),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 1 if report.get("blocking_errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
