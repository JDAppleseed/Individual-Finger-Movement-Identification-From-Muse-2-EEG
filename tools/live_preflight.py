#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui_config_validation import validate_live_infer
from tools.analyze_live_raw_inputs import build_distribution_report
from utils.live_infer_common import require_deployable_run
from utils.lsl_stream_select import resolve_source_id_preference
from utils.session_layout import SessionLayout


def _load_live_module():
    module_path = REPO_ROOT / "7_live_infer_and_actuate.py"
    spec = importlib.util.spec_from_file_location("live_infer_preflight", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_settings(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if isinstance(settings, dict):
        return settings
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"Unsupported config format: {path}")


def _resolve_repo_path(raw: Any) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(str(raw)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (REPO_ROOT / path).resolve()


def _print_kv(label: str, value: Any) -> None:
    print(f"{label:24}: {value}")


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts if str(part))


def _parse_required_labels(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        return [part.strip() for part in cleaned.split(",") if part.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _resolve_effective_expected_channel_labels(
    *,
    live_mod,
    settings: dict[str, Any],
    deployment_run_dir: Path | None,
) -> tuple[list[str], str | None]:
    if deployment_run_dir is None:
        return [], None
    expected_labels, expected_labels_source = live_mod._resolve_expected_channel_labels(
        settings,
        deployment_run_dir,
    )
    require_fn = getattr(live_mod, "_require_expected_channel_labels", None)
    if callable(require_fn):
        expected_labels = require_fn(expected_labels, expected_labels_source)
    else:
        expected_labels = [str(label).strip() for label in expected_labels if str(label).strip()]
        if not expected_labels:
            raise RuntimeError(
                "No expected live channel labels could be derived. Step 7 cannot prove "
                "model-order channel mapping without REQUIRED_LSL_LABELS or "
                "training_npz.channel_names."
            )
    return list(expected_labels), expected_labels_source


def _probe_stream(
    *,
    live_mod,
    settings: dict[str, Any],
    deployment_run_dir: Path | None,
    cli_source_id: str | None,
    env_source_id: str | None,
    config_source_id: str | None,
) -> tuple[Any, dict[str, Any] | None, dict[str, Any] | None]:
    lsl_name = (
        settings.get("stream_name")
        or settings.get("lsl_name")
        or settings.get("LSL_STREAM_NAME")
        or "Muse2-EEG"
    )
    lsl_type = (
        settings.get("stream_type")
        or settings.get("lsl_type")
        or settings.get("LSL_STREAM_TYPE")
        or "EEG"
    )
    try:
        timeout_s = float(settings.get("LSL_RESOLVE_TIMEOUT", 25.0))
    except Exception:
        timeout_s = 25.0
    resolved = live_mod._resolve_lsl_inlet(
        str(lsl_name),
        str(lsl_type),
        timeout_s=float(timeout_s),
        cli_source_id=cli_source_id,
        env_source_id=env_source_id,
        config_source_id=config_source_id,
    )
    source_pref = resolve_source_id_preference(
        cli_source_id=cli_source_id,
        env_source_id=env_source_id,
        config_source_id=config_source_id,
    )
    source_pref_payload = {
        "cli_source_id": source_pref.cli_source_id,
        "env_source_id": source_pref.env_source_id,
        "config_source_id": source_pref.config_source_id,
        "requested_source_id": source_pref.requested_source_id,
        "source": source_pref.source,
    }
    expected_labels: list[str] = []
    expected_labels_source: str | None = None
    expected_rate: float | None = None
    if deployment_run_dir is not None:
        expected_labels, expected_labels_source = _resolve_effective_expected_channel_labels(
            live_mod=live_mod,
            settings=settings,
            deployment_run_dir=deployment_run_dir,
        )
        train_config = live_mod._load_train_config(deployment_run_dir)
        expected_rate, _ = live_mod._resolve_effective_target_fs(
            train_config=train_config,
            window_sec=float(settings.get("window_sec", 0.25)),
            requested_target_fs=float(settings.get("target_fs", 256.0)),
        )
    stream_contract = live_mod._stream_contract_summary(
        config_settings=settings,
        expected_name=str(lsl_name),
        expected_type=str(lsl_type),
        source_id_preference=source_pref_payload,
        resolved_stream=resolved.resolution,
        expected_labels=expected_labels,
        expected_rate=expected_rate,
        expected_labels_source=expected_labels_source,
    )
    channel_reorder = live_mod._build_channel_reorder(
        expected_labels,
        resolved.resolution.get("channel_labels", []) or [],
    )
    channel_reorder_applied = bool(
        channel_reorder is not None
        and list(channel_reorder) != list(range(len(channel_reorder)))
    )
    stream_contract.setdefault("resolved", {})
    stream_contract["resolved"]["channel_reorder_to_model_order"] = (
        list(channel_reorder) if channel_reorder is not None else None
    )
    stream_contract["resolved"]["channel_reorder_applied"] = bool(
        channel_reorder_applied
    )
    live_mod._require_stream_contract_ok(stream_contract)
    if (
        expected_labels
        and (resolved.resolution.get("channel_labels") or [])
        and channel_reorder is None
    ):
        raise RuntimeError(
            "Resolved stream labels passed the set check but could not be mapped into a deterministic model channel order."
        )
    return resolved, resolved.resolution, stream_contract


def _collect_distribution_probe_samples(
    *,
    live_mod,
    inlet: Any,
    duration_s: float,
) -> np.ndarray:
    records: list[tuple[int, float, float, float, int, int, int, np.ndarray]] = []
    seq = 0
    stream_origin_mono = None
    stream_origin_lsl = None
    prev_lsl_mono = None
    deadline = time.monotonic() + max(1.0, float(duration_s))
    while time.monotonic() < deadline:
        chunk, timestamps = inlet.pull_chunk(timeout=0.1, max_samples=64)
        if not timestamps:
            continue
        for sample, lsl_ts in zip(chunk, timestamps):
            sample_mono = time.monotonic()
            (
                _time_s,
                lsl_ts_mono,
                clamped,
                stream_origin_mono,
                stream_origin_lsl,
                prev_lsl_mono,
            ) = live_mod._resolve_live_sample_time(
                lsl_ts=float(lsl_ts),
                sample_mono=float(sample_mono),
                stream_origin_mono=stream_origin_mono,
                stream_origin_lsl=stream_origin_lsl,
                prev_lsl_mono=prev_lsl_mono,
            )
            values = np.asarray(sample, dtype=np.float64)
            flags = 0
            if not np.all(np.isfinite(values)):
                flags |= int(live_mod.RAW_FLAG_NONFINITE)
            records.append(
                (
                    int(seq),
                    float(lsl_ts),
                    float(lsl_ts_mono),
                    float(sample_mono),
                    int(flags),
                    0,
                    int(bool(clamped)),
                    values,
                )
            )
            seq += 1
            if time.monotonic() >= deadline:
                break
    if not records:
        raise RuntimeError("distribution probe captured no stream samples")
    channel_count = int(np.asarray(records[0][7], dtype=np.float64).size)
    raw_dtype = np.dtype(
        [
            ("seq", "<i8"),
            ("lsl_ts_raw", "<f8"),
            ("lsl_ts_mono", "<f8"),
            ("local_ts", "<f8"),
            ("flags", "<i8"),
            ("segment_id", "<i8"),
            ("clamped", "i1"),
            ("sample", "<f8", (channel_count,)),
        ]
    )
    raw = np.zeros(len(records), dtype=raw_dtype)
    for idx, record in enumerate(records):
        raw["seq"][idx] = int(record[0])
        raw["lsl_ts_raw"][idx] = float(record[1])
        raw["lsl_ts_mono"][idx] = float(record[2])
        raw["local_ts"][idx] = float(record[3])
        raw["flags"][idx] = int(record[4])
        raw["segment_id"][idx] = int(record[5])
        raw["clamped"][idx] = int(record[6])
        raw["sample"][idx] = np.asarray(record[7], dtype=np.float64)
    return raw


def _build_distribution_probe_runtime_manifest(
    *,
    settings: dict[str, Any],
    stream_resolution: dict[str, Any],
    stream_contract: dict[str, Any],
    launch_plan: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    expected_sampling_rate = (
        (stream_contract or {}).get("expected", {}).get("sampling_rate")
        if isinstance((stream_contract or {}).get("expected"), dict)
        else None
    )
    target_fs = float(expected_sampling_rate or settings.get("target_fs", 256.0))
    return {
        "stream_resolution": dict(stream_resolution or {}),
        "stream_contract": dict(stream_contract or {}),
        "stream_selection": {
            "expected_channel_labels": list(
                (stream_contract or {}).get("expected", {}).get("required_labels") or []
            ),
        },
        "runtime": {
            "window_sec": float(settings.get("window_sec", 0.25)),
            "hop_sec": float(settings.get("hop_sec", 0.05)),
            "target_fs": float(target_fs),
            "alignment_internal_max_gap_s": float(
                settings.get("alignment_internal_max_gap_s", 0.06)
            ),
            "alignment_edge_max_gap_s": float(1.0 / float(target_fs) * 4.0),
            "live_quality_enabled": bool(settings.get("live_quality_enabled", True)),
            "quality_thresholds": {
                "input_clip_abs_z": float(settings.get("input_clip_abs_z", 6.0)),
                "bad_channel_rms_z": float(settings.get("bad_channel_rms_z", 4.0)),
                "bad_channel_abs_p95_z": float(settings.get("bad_channel_abs_p95_z", 6.0)),
                "bad_channel_clipped_frac": float(
                    settings.get("bad_channel_clipped_frac", 0.05)
                ),
                "bad_window_clipped_frac": float(
                    settings.get("bad_window_clipped_frac", 0.10)
                ),
                "bad_window_max_masked_channels": int(
                    settings.get("bad_window_max_masked_channels", 1)
                ),
            },
        },
        "artifacts": {
            "run_dir": (
                str(launch_plan.model_path.parent)
                if launch_plan is not None and getattr(launch_plan, "model_path", None) is not None
                else None
            ),
        },
        "probe": {
            "lsl_source_id": args.lsl_source_id,
            "distribution_probe_seconds": float(args.distribution_probe_seconds),
        },
    }


def _assess_distribution_probe(report: dict[str, Any]) -> tuple[list[str], list[str], dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    distribution_match = (
        report.get("distribution_match", {})
        if isinstance(report.get("distribution_match"), dict)
        else {}
    )
    relaxed = (
        report.get("alignment", {}).get("relaxed", {})
        if isinstance(report.get("alignment"), dict)
        else {}
    )
    accepted_count = int(relaxed.get("accepted_count", 0) or 0)
    quality_bad_rate = relaxed.get("quality_bad_rate")
    if accepted_count <= 0:
        errors.append("distribution probe found no valid accepted windows")
    if quality_bad_rate is not None and float(quality_bad_rate) >= 0.75:
        errors.append(
            f"distribution probe quality rejection is overwhelming (quality_bad_rate={float(quality_bad_rate):.3f})"
        )
    if bool(distribution_match.get("catastrophic")):
        errors.append(
            f"distribution probe detected catastrophic live-vs-offline mismatch ({distribution_match.get('reason')})"
        )
    decisive = bool(
        report.get("distribution_claim_decisive") is True
        and distribution_match.get("decisive") is True
    )
    if not decisive:
        errors.append(
            "distribution probe reorder proof is non-decisive; decisive Step 7 launch is blocked until model-order live input proof is available"
        )
    verdict = str(distribution_match.get("verdict") or "unknown")
    if verdict not in {"nominal", "unknown"} and not bool(distribution_match.get("catastrophic")):
        warnings.append(
            f"distribution probe verdict={verdict}: {distribution_match.get('reason')}"
        )
    compact = {
        "verdict": verdict,
        "decisive": decisive,
        "accepted_count": accepted_count,
        "candidate_count": int(relaxed.get("candidate_count", 0) or 0),
        "quality_bad_rate": (
            float(quality_bad_rate) if quality_bad_rate is not None else None
        ),
        "median_rms_ratio": distribution_match.get("median_rms_ratio"),
        "recovered_vs_strict_count": distribution_match.get("recovered_vs_strict_count"),
    }
    return errors, warnings, compact


def _build_step7_command(
    *,
    config_path: Path,
    launch_plan,
    source_id: str | None,
) -> str:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "7_live_infer_and_actuate.py"),
        "--config",
        str(config_path),
    ]
    if launch_plan.selected_session_dir is not None:
        cmd += ["--session-dir", str(launch_plan.selected_session_dir)]
    cmd += ["--out-dir", str(launch_plan.out_dir)]
    if source_id:
        cmd += ["--lsl-source-id", str(source_id)]
    cmd += [
        "--parity-capture-enabled",
        "--parity-capture-max-windows",
        "128",
        "--parity-capture-flush-every",
        "1",
    ]
    return _shell_join(cmd)


def _serialize_launch_plan(launch_plan: Any) -> dict[str, Any] | None:
    if launch_plan is None:
        return None
    return {
        "project_name": getattr(launch_plan, "project_name", None),
        "subject_id": getattr(launch_plan, "subject_id", None),
        "selection_source": getattr(launch_plan, "selection_source", None),
        "session_dir_inferred": bool(
            getattr(launch_plan, "session_dir_inferred", False)
        ),
        "selected_session_dir": (
            str(getattr(launch_plan, "selected_session_dir", ""))
            if getattr(launch_plan, "selected_session_dir", None) is not None
            else None
        ),
        "explicit_overrides": list(
            getattr(launch_plan, "explicit_overrides", ()) or ()
        ),
        "chosen_run_dir": (
            str(getattr(launch_plan, "chosen_run_dir", ""))
            if getattr(launch_plan, "chosen_run_dir", None) is not None
            else None
        ),
        "model_path": (
            str(getattr(launch_plan, "model_path", ""))
            if getattr(launch_plan, "model_path", None) is not None
            else None
        ),
        "scaler_path": (
            str(getattr(launch_plan, "scaler_path", ""))
            if getattr(launch_plan, "scaler_path", None) is not None
            else None
        ),
        "temperature_path": (
            str(getattr(launch_plan, "temperature_path", ""))
            if getattr(launch_plan, "temperature_path", None) is not None
            else None
        ),
        "out_dir": (
            str(getattr(launch_plan, "out_dir", ""))
            if getattr(launch_plan, "out_dir", None) is not None
            else None
        ),
        "no_file_io": bool(getattr(launch_plan, "no_file_io", False)),
        "record_raw": bool(getattr(launch_plan, "record_raw", False)),
    }


def run_live_preflight(
    *,
    config_path: Path,
    session_dir: str | None = None,
    model_path: str | None = None,
    scaler_path: str | None = None,
    out_dir: str | None = None,
    project_name: str | None = None,
    subject_id: str | None = None,
    lsl_source_id: str | None = None,
    probe_stream: bool = False,
    probe_distribution: bool = False,
    distribution_probe_seconds: float = 15.0,
    allow_no_source_id: bool = False,
    allow_no_parity_capture: bool = False,
    smoke_device: str = "cpu",
    smoke_index: int = 0,
    skip_smoke: bool = False,
) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    settings = _load_settings(config_path)
    validation = validate_live_infer(deepcopy(settings))
    live_mod = _load_live_module()
    _, live_defaults = live_mod._build_arg_parser()

    errors = list(validation.errors)
    warnings = list(validation.warnings)
    parse_labels = getattr(live_mod, "parse_required_labels", _parse_required_labels)
    configured_required_labels = parse_labels(settings.get("REQUIRED_LSL_LABELS"))

    launch_plan = None
    try:
        launch_plan = live_mod.resolve_live_launch_plan(
            config_path=config_path,
            config_payload=json.loads(config_path.read_text()),
            config_settings=settings,
            session_dir_override=session_dir,
            project_name_override=project_name,
            subject_id_override=subject_id,
            model_path_override=model_path,
            scaler_path_override=scaler_path,
            out_dir_override=out_dir,
            allow_outside_base=False,
            no_file_io_override=None,
        )
    except Exception as exc:
        errors.append(str(exc))

    source_pref = resolve_source_id_preference(
        cli_source_id=lsl_source_id,
        env_source_id=os.environ.get("LSL_SOURCE_ID"),
        config_source_id=settings.get("lsl_source_id") or settings.get("LSL_SOURCE_ID"),
    )
    source_pref_payload = {
        "cli_source_id": source_pref.cli_source_id,
        "env_source_id": source_pref.env_source_id,
        "config_source_id": source_pref.config_source_id,
        "requested_source_id": source_pref.requested_source_id,
        "source": source_pref.source,
    }
    if source_pref.requested_source_id is not None:
        warnings = [
            warning
            for warning in warnings
            if "lsl_source_id is blank in config" not in warning
        ]
    if source_pref.requested_source_id is None and not allow_no_source_id:
        errors.append(
            "No explicit live LSL source_id is pinned. Export LSL_SOURCE_ID or pass --lsl-source-id before the run."
        )
    parity_capture_enabled = bool(
        settings.get(
            "parity_capture_enabled",
            live_defaults.get("parity_capture_enabled", False),
        )
    )
    if not parity_capture_enabled and not allow_no_parity_capture:
        errors.append(
            "parity_capture_enabled is not enabled in config. Enable it for the next decisive live run."
        )

    stream_resolution = None
    stream_contract = None
    resolved_stream = None
    expected_channel_labels = list(configured_required_labels)
    expected_channel_labels_source = (
        "config.REQUIRED_LSL_LABELS" if configured_required_labels else None
    )
    if probe_stream or probe_distribution:
        try:
            resolved_stream, stream_resolution, stream_contract = _probe_stream(
                live_mod=live_mod,
                settings=settings,
                deployment_run_dir=(
                    launch_plan.model_path.parent if launch_plan is not None else None
                ),
                cli_source_id=lsl_source_id,
                env_source_id=os.environ.get("LSL_SOURCE_ID"),
                config_source_id=settings.get("lsl_source_id")
                or settings.get("LSL_SOURCE_ID"),
            )
        except Exception as exc:
            errors.append(f"stream probe failed: {exc}")
        else:
            expected_section = (
                stream_contract.get("expected", {})
                if isinstance(stream_contract, dict)
                else {}
            )
            expected_channel_labels = list(
                expected_section.get("required_labels") or []
            )
            expected_channel_labels_source = expected_section.get(
                "required_labels_source"
            )

    deployable_info: dict[str, Any] | None = None
    smoke_session_dir = None
    windows_npz = None
    if launch_plan is not None:
        if launch_plan.no_file_io:
            errors.append("no_file_io is enabled. Disable it for the next live run.")
        if not launch_plan.model_path.exists():
            errors.append(f"model_path not found: {launch_plan.model_path}")
        if not launch_plan.scaler_path.exists():
            errors.append(f"scaler_path not found: {launch_plan.scaler_path}")
        if not launch_plan.temperature_path.exists():
            errors.append(
                f"temperature_scaling.json not found: {launch_plan.temperature_path}"
            )
        if settings.get("enable_actuation"):
            try:
                deployable_info = require_deployable_run(launch_plan.model_path.parent)
            except RuntimeError as exc:
                errors.append(str(exc))
        deployment_session_dir = _resolve_repo_path(settings.get("deployment_session_dir"))
        smoke_session_dir = launch_plan.selected_session_dir
        if smoke_session_dir is not None and smoke_session_dir.exists():
            candidate = SessionLayout(smoke_session_dir).windows_npz
            if candidate.exists():
                windows_npz = candidate
        if (
            windows_npz is None
            and deployment_session_dir is not None
            and deployment_session_dir.exists()
        ):
            candidate = SessionLayout(deployment_session_dir).windows_npz
            if candidate.exists():
                smoke_session_dir = deployment_session_dir
                windows_npz = candidate
        if windows_npz is None:
            warnings.append("No windows NPZ was found for smoke inference.")
        if not (probe_stream or probe_distribution):
            try:
                (
                    expected_channel_labels,
                    expected_channel_labels_source,
                ) = _resolve_effective_expected_channel_labels(
                    live_mod=live_mod,
                    settings=settings,
                    deployment_run_dir=launch_plan.model_path.parent,
                )
            except Exception as exc:
                errors.append(f"expected channel labels unavailable: {exc}")

    distribution_probe_compact = None
    if probe_distribution:
        if resolved_stream is None:
            errors.append("distribution probe requires a resolved live stream")
        elif launch_plan is None or not launch_plan.model_path.exists():
            errors.append("distribution probe requires a valid deployment model path")
        elif windows_npz is None or not windows_npz.exists():
            errors.append(
                "distribution probe requires an offline windows NPZ for comparison"
            )
        else:
            try:
                raw_probe = _collect_distribution_probe_samples(
                    live_mod=live_mod,
                    inlet=resolved_stream.inlet,
                    duration_s=float(distribution_probe_seconds),
                )
                probe_runtime_manifest = _build_distribution_probe_runtime_manifest(
                    settings=settings,
                    stream_resolution=stream_resolution or {},
                    stream_contract=stream_contract or {},
                    launch_plan=launch_plan,
                    args=argparse.Namespace(
                        lsl_source_id=lsl_source_id,
                        distribution_probe_seconds=distribution_probe_seconds,
                    ),
                )
                distribution_probe_report = build_distribution_report(
                    raw_source=raw_probe,
                    run_dir=launch_plan.model_path.parent,
                    offline_npz=windows_npz,
                    runtime_manifest=probe_runtime_manifest,
                    window_sec=float(settings.get("window_sec", 0.25)),
                    hop_sec=float(settings.get("hop_sec", 0.05)),
                    target_fs=float(
                        probe_runtime_manifest.get("runtime", {}).get(
                            "target_fs", 256.0
                        )
                    ),
                    relaxed_gap_s=float(
                        settings.get("alignment_internal_max_gap_s", 0.06)
                    ),
                    confidence_sample_windows=64,
                )
                probe_errors, probe_warnings, distribution_probe_compact = (
                    _assess_distribution_probe(distribution_probe_report)
                )
                errors.extend(probe_errors)
                warnings.extend(probe_warnings)
            except Exception as exc:
                errors.append(f"distribution probe failed: {exc}")

    smoke_ok = True
    smoke_cmd = None
    smoke_stdout = ""
    smoke_stderr = ""
    smoke_returncode = None
    if (
        not skip_smoke
        and launch_plan is not None
        and windows_npz is not None
        and windows_npz.exists()
        and launch_plan.model_path.exists()
        and launch_plan.scaler_path.exists()
    ):
        smoke_cmd_parts = [
            sys.executable,
            str(REPO_ROOT / "tools" / "smoke_inference.py"),
            "--npz",
            str(windows_npz),
            "--model",
            str(launch_plan.model_path),
            "--scaler",
            str(launch_plan.scaler_path),
            "--index",
            str(int(smoke_index)),
            "--device",
            str(smoke_device),
        ]
        smoke_cmd = _shell_join(smoke_cmd_parts)
        smoke = subprocess.run(smoke_cmd_parts, capture_output=True, text=True)
        smoke_returncode = int(smoke.returncode)
        smoke_stdout = smoke.stdout.strip()
        smoke_stderr = smoke.stderr.strip()
        if smoke.returncode != 0:
            smoke_ok = False
            errors.append(
                f"smoke inference failed with exit code {smoke.returncode}"
            )

    recommended_commands = None
    if launch_plan is not None:
        recommended_commands = {
            "live": _build_step7_command(
                config_path=config_path,
                launch_plan=launch_plan,
                source_id=source_pref.requested_source_id,
            ),
            "replay": _shell_join(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "replay_live_capture.py"),
                    "--capture-dir",
                    str(launch_plan.out_dir / "parity_capture"),
                ]
            ),
            "audit": _shell_join(
                [
                    sys.executable,
                    str(REPO_ROOT / "tools" / "audit_live_parity.py"),
                    "--live-dir",
                    str(launch_plan.out_dir),
                    "--parity-report",
                    str(launch_plan.out_dir / "parity_report.json"),
                    "--distribution-report",
                    str(launch_plan.out_dir / "live_input_distribution_report.json"),
                    "--write-json",
                    "--write-md",
                ]
            ),
        }

    effective_contract = {
        "stream_name": settings.get("stream_name") or settings.get("LSL_STREAM_NAME"),
        "stream_type": settings.get("stream_type") or settings.get("LSL_STREAM_TYPE"),
        "requested_source_id": source_pref.requested_source_id,
        "source_id_source": source_pref.source,
        "parity_capture_enabled": parity_capture_enabled,
        "no_file_io": (
            getattr(launch_plan, "no_file_io", settings.get("no_file_io"))
            if launch_plan is not None
            else bool(settings.get("no_file_io"))
        ),
        "config_required_lsl_labels": list(configured_required_labels),
        "expected_channel_labels": list(expected_channel_labels),
        "expected_channel_labels_source": expected_channel_labels_source,
        "require_exactly_4_channels": bool(
            settings.get("REQUIRE_EXACTLY_4_CHANNELS", True)
        ),
        "alignment_internal_max_gap_s": float(
            settings.get("alignment_internal_max_gap_s", 0.06)
        ),
        "latency_policy": settings.get("latency_policy"),
        "latency_threshold_ms": settings.get("latency_threshold_ms"),
        "live_quality_enabled": bool(settings.get("live_quality_enabled", True)),
        "enable_actuation": bool(settings.get("enable_actuation", False)),
        "smoke_session_dir": str(smoke_session_dir) if smoke_session_dir else None,
    }

    return {
        "ready": bool(not errors and smoke_ok),
        "config_path": str(config_path),
        "errors": errors,
        "warnings": warnings,
        "launch_plan": _serialize_launch_plan(launch_plan),
        "source_preference": source_pref_payload,
        "stream_probe": {
            "resolution": stream_resolution,
            "contract": stream_contract,
        }
        if stream_resolution is not None
        else None,
        "distribution_probe": distribution_probe_compact,
        "deployment": deployable_info,
        "smoke": {
            "requested": bool(not skip_smoke),
            "ok": bool(smoke_ok),
            "cmd": smoke_cmd,
            "returncode": smoke_returncode,
            "stdout": smoke_stdout,
            "stderr": smoke_stderr,
        },
        "recommended_commands": recommended_commands,
        "effective_contract": effective_contract,
    }


def _print_live_preflight_report(report: dict[str, Any]) -> None:
    launch_plan = report.get("launch_plan") or {}
    effective_contract = report.get("effective_contract") or {}
    stream_probe = report.get("stream_probe") or {}
    stream_resolution = stream_probe.get("resolution") or {}
    stream_contract = stream_probe.get("contract") or {}
    distribution_probe = report.get("distribution_probe") or {}
    smoke = report.get("smoke") or {}
    recommended_commands = report.get("recommended_commands") or {}

    print("Live preflight")
    print("-" * 40)
    _print_kv("config", report.get("config_path"))
    _print_kv("session_dir", launch_plan.get("selected_session_dir"))
    _print_kv("out_dir", launch_plan.get("out_dir"))
    _print_kv("model_path", launch_plan.get("model_path"))
    _print_kv("scaler_path", launch_plan.get("scaler_path"))
    _print_kv("temperature_path", launch_plan.get("temperature_path"))
    _print_kv("selection_source", launch_plan.get("selection_source"))
    _print_kv("session_dir_inferred", launch_plan.get("session_dir_inferred"))
    _print_kv("explicit_overrides", launch_plan.get("explicit_overrides"))
    _print_kv("stream_name", effective_contract.get("stream_name"))
    _print_kv("stream_type", effective_contract.get("stream_type"))
    _print_kv("requested_source_id", effective_contract.get("requested_source_id"))
    _print_kv("source_id_source", effective_contract.get("source_id_source"))
    _print_kv(
        "parity_capture_enabled", effective_contract.get("parity_capture_enabled")
    )
    _print_kv("no_file_io", effective_contract.get("no_file_io"))
    _print_kv(
        "config_required_labels",
        effective_contract.get("config_required_lsl_labels"),
    )
    _print_kv(
        "expected_channel_labels",
        effective_contract.get("expected_channel_labels"),
    )
    _print_kv(
        "expected_labels_source",
        effective_contract.get("expected_channel_labels_source"),
    )
    _print_kv("smoke_session_dir", effective_contract.get("smoke_session_dir"))

    if stream_resolution:
        print("\nStream probe")
        print("-" * 40)
        _print_kv("selected_source_id", stream_resolution.get("selected_source_id"))
        _print_kv("stream_uid", stream_resolution.get("uid"))
        _print_kv("channel_labels", stream_resolution.get("channel_labels"))
        _print_kv("stream_contract_ok", stream_contract.get("contract_ok"))
        _print_kv("stream_contract_mismatches", stream_contract.get("mismatches"))

    if distribution_probe:
        print("\nDistribution probe")
        print("-" * 40)
        _print_kv("probe_seconds", distribution_probe.get("probe_seconds"))
        _print_kv("verdict", distribution_probe.get("verdict"))
        _print_kv("decisive", distribution_probe.get("decisive"))
        _print_kv("accepted_count", distribution_probe.get("accepted_count"))
        _print_kv("candidate_count", distribution_probe.get("candidate_count"))
        _print_kv("quality_bad_rate", distribution_probe.get("quality_bad_rate"))
        _print_kv("median_rms_ratio", distribution_probe.get("median_rms_ratio"))
        _print_kv(
            "recovered_vs_strict_count",
            distribution_probe.get("recovered_vs_strict_count"),
        )

    if report.get("deployment") is not None:
        deployment = report["deployment"]
        print("\nDeployment")
        print("-" * 40)
        _print_kv("deployable_run", deployment.get("deployable"))
        _print_kv("active_finger_head", deployment.get("active_finger_head"))
        _print_kv(
            "finger_applicability_head",
            deployment.get("finger_applicability_head"),
        )

    if smoke.get("cmd"):
        print("\nSmoke inference")
        print("-" * 40)
        print(smoke["cmd"])
        if smoke.get("stdout"):
            print(smoke["stdout"])
        if smoke.get("stderr"):
            print(smoke["stderr"])

    if report.get("warnings"):
        print("\nWarnings")
        print("-" * 40)
        for warning in report["warnings"]:
            print(f"- {warning}")

    if recommended_commands:
        print("\nRecommended commands")
        print("-" * 40)
        print(recommended_commands.get("live"))
        print(recommended_commands.get("replay"))
        print(recommended_commands.get("audit"))

    if report.get("errors"):
        print("\nErrors")
        print("-" * 40)
        for error in report["errors"]:
            print(f"- {error}")
        return

    print("\nReady for live run.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the current Step 7 deployment config before a live run."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to infer.json")
    parser.add_argument("--session-dir", type=str, default=None, help="Optional session directory override")
    parser.add_argument("--model-path", type=str, default=None, help="Optional model path override")
    parser.add_argument("--scaler-path", type=str, default=None, help="Optional scaler path override")
    parser.add_argument("--out-dir", type=str, default=None, help="Explicit live output directory to use")
    parser.add_argument("--project-name", type=str, default=None, help="Optional project name override")
    parser.add_argument("--subject-id", type=str, default=None, help="Optional subject ID override")
    parser.add_argument("--lsl-source-id", type=str, default=None, help="Explicit live LSL source_id to require")
    parser.add_argument("--probe-stream", action="store_true", help="Resolve the live LSL stream using the same Step 7 rules")
    parser.add_argument("--probe-distribution", action="store_true", help="Capture a short live sample and compare model-order live inputs against the offline deployment NPZ.")
    parser.add_argument("--distribution-probe-seconds", type=float, default=15.0, help="Live sampling duration for --probe-distribution.")
    parser.add_argument("--allow-no-source-id", action="store_true", help="Do not fail when no explicit LSL source_id is pinned")
    parser.add_argument("--allow-no-parity-capture", action="store_true", help="Do not fail when parity capture is disabled in config")
    parser.add_argument("--smoke-device", type=str, default="cpu", help="Device for the optional smoke inference check")
    parser.add_argument("--smoke-index", type=int, default=0, help="Window index for smoke inference")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the smoke inference command")
    parser.add_argument("--json", action="store_true", help="Emit a structured JSON report instead of human-readable text.")
    args = parser.parse_args()
    report = run_live_preflight(
        config_path=Path(args.config),
        session_dir=args.session_dir,
        model_path=args.model_path,
        scaler_path=args.scaler_path,
        out_dir=args.out_dir,
        project_name=args.project_name,
        subject_id=args.subject_id,
        lsl_source_id=args.lsl_source_id,
        probe_stream=bool(args.probe_stream),
        probe_distribution=bool(args.probe_distribution),
        distribution_probe_seconds=float(args.distribution_probe_seconds),
        allow_no_source_id=bool(args.allow_no_source_id),
        allow_no_parity_capture=bool(args.allow_no_parity_capture),
        smoke_device=str(args.smoke_device),
        smoke_index=int(args.smoke_index),
        skip_smoke=bool(args.skip_smoke),
    )
    distribution_probe = report.get("distribution_probe")
    if isinstance(distribution_probe, dict):
        distribution_probe.setdefault(
            "probe_seconds", float(args.distribution_probe_seconds)
        )
    if args.json:
        print(json.dumps(report, sort_keys=True))
    else:
        _print_live_preflight_report(report)
    return 0 if bool(report.get("ready")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
