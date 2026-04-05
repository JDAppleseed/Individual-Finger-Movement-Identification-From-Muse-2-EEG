#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui_config_validation import validate_live_infer
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


def _probe_stream(
    *,
    live_mod,
    settings: dict[str, Any],
    deployment_run_dir: Path | None,
    cli_source_id: str | None,
    env_source_id: str | None,
    config_source_id: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
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
        expected_labels, expected_labels_source = live_mod._resolve_expected_channel_labels(
            settings,
            deployment_run_dir,
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
    live_mod._require_stream_contract_ok(stream_contract)
    return resolved.resolution, stream_contract


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
    parser.add_argument("--allow-no-source-id", action="store_true", help="Do not fail when no explicit LSL source_id is pinned")
    parser.add_argument("--allow-no-parity-capture", action="store_true", help="Do not fail when parity capture is disabled in config")
    parser.add_argument("--smoke-device", type=str, default="cpu", help="Device for the optional smoke inference check")
    parser.add_argument("--smoke-index", type=int, default=0, help="Window index for smoke inference")
    parser.add_argument("--skip-smoke", action="store_true", help="Skip the smoke inference command")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    settings = _load_settings(config_path)
    validation = validate_live_infer(deepcopy(settings))
    live_mod = _load_live_module()
    _, live_defaults = live_mod._build_arg_parser()

    errors = list(validation.errors)
    warnings = list(validation.warnings)

    try:
        launch_plan = live_mod.resolve_live_launch_plan(
            config_path=config_path,
            config_payload=json.loads(config_path.read_text()),
            config_settings=settings,
            session_dir_override=args.session_dir,
            project_name_override=args.project_name,
            subject_id_override=args.subject_id,
            model_path_override=args.model_path,
            scaler_path_override=args.scaler_path,
            out_dir_override=args.out_dir,
            allow_outside_base=False,
            no_file_io_override=None,
        )
    except Exception as exc:
        launch_plan = None
        errors.append(str(exc))

    source_pref = resolve_source_id_preference(
        cli_source_id=args.lsl_source_id,
        env_source_id=os.environ.get("LSL_SOURCE_ID"),
        config_source_id=settings.get("lsl_source_id") or settings.get("LSL_SOURCE_ID"),
    )
    if source_pref.requested_source_id is not None:
        warnings = [
            warning
            for warning in warnings
            if "lsl_source_id is blank in config" not in warning
        ]
    if source_pref.requested_source_id is None and not args.allow_no_source_id:
        errors.append(
            "No explicit live LSL source_id is pinned. Export LSL_SOURCE_ID or pass --lsl-source-id before the run."
        )
    parity_capture_enabled = bool(
        settings.get(
            "parity_capture_enabled",
            live_defaults.get("parity_capture_enabled", False),
        )
    )
    if not parity_capture_enabled and not args.allow_no_parity_capture:
        errors.append(
            "parity_capture_enabled is not enabled in config. Enable it for the next decisive live run."
        )

    stream_resolution = None
    stream_contract = None
    if args.probe_stream:
        try:
            stream_resolution, stream_contract = _probe_stream(
                live_mod=live_mod,
                settings=settings,
                deployment_run_dir=(
                    launch_plan.model_path.parent if launch_plan is not None else None
                ),
                cli_source_id=args.lsl_source_id,
                env_source_id=os.environ.get("LSL_SOURCE_ID"),
                config_source_id=settings.get("lsl_source_id") or settings.get("LSL_SOURCE_ID"),
            )
        except Exception as exc:
            errors.append(f"stream probe failed: {exc}")

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
            errors.append(f"temperature_scaling.json not found: {launch_plan.temperature_path}")
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
        if windows_npz is None and deployment_session_dir is not None and deployment_session_dir.exists():
            candidate = SessionLayout(deployment_session_dir).windows_npz
            if candidate.exists():
                smoke_session_dir = deployment_session_dir
                windows_npz = candidate
        if windows_npz is None:
            warnings.append("No windows NPZ was found for smoke inference.")

    print("Live preflight")
    print("-" * 40)
    _print_kv("config", config_path)
    _print_kv("session_dir", getattr(launch_plan, "selected_session_dir", None))
    _print_kv("out_dir", getattr(launch_plan, "out_dir", None))
    _print_kv("model_path", getattr(launch_plan, "model_path", None))
    _print_kv("scaler_path", getattr(launch_plan, "scaler_path", None))
    _print_kv("temperature_path", getattr(launch_plan, "temperature_path", None))
    _print_kv("selection_source", getattr(launch_plan, "selection_source", None))
    _print_kv("session_dir_inferred", getattr(launch_plan, "session_dir_inferred", None))
    _print_kv("explicit_overrides", getattr(launch_plan, "explicit_overrides", None))
    _print_kv("stream_name", settings.get("stream_name") or settings.get("LSL_STREAM_NAME"))
    _print_kv("stream_type", settings.get("stream_type") or settings.get("LSL_STREAM_TYPE"))
    _print_kv("requested_source_id", source_pref.requested_source_id)
    _print_kv("source_id_source", source_pref.source)
    _print_kv("parity_capture_enabled", parity_capture_enabled)
    _print_kv("no_file_io", getattr(launch_plan, "no_file_io", settings.get("no_file_io")))
    _print_kv("smoke_session_dir", smoke_session_dir)

    if stream_resolution is not None:
        print("\nStream probe")
        print("-" * 40)
        _print_kv("selected_source_id", stream_resolution.get("selected_source_id"))
        _print_kv("stream_uid", stream_resolution.get("uid"))
        _print_kv("channel_labels", stream_resolution.get("channel_labels"))
        _print_kv("stream_contract_ok", stream_contract.get("contract_ok") if stream_contract else None)
        _print_kv("stream_contract_mismatches", stream_contract.get("mismatches") if stream_contract else None)

    if deployable_info is not None:
        print("\nDeployment")
        print("-" * 40)
        _print_kv("deployable_run", deployable_info.get("deployable"))
        _print_kv("active_finger_head", deployable_info.get("active_finger_head"))
        _print_kv(
            "finger_applicability_head",
            deployable_info.get("finger_applicability_head"),
        )

    smoke_ok = True
    if (
        not args.skip_smoke
        and launch_plan is not None
        and windows_npz is not None
        and windows_npz.exists()
        and launch_plan.model_path.exists()
        and launch_plan.scaler_path.exists()
    ):
        smoke_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "smoke_inference.py"),
            "--npz",
            str(windows_npz),
            "--model",
            str(launch_plan.model_path),
            "--scaler",
            str(launch_plan.scaler_path),
            "--index",
            str(int(args.smoke_index)),
            "--device",
            str(args.smoke_device),
        ]
        print("\nSmoke inference")
        print("-" * 40)
        print(_shell_join(smoke_cmd))
        smoke = subprocess.run(smoke_cmd, capture_output=True, text=True)
        if smoke.stdout.strip():
            print(smoke.stdout.strip())
        if smoke.returncode != 0:
            smoke_ok = False
            errors.append(f"smoke inference failed with exit code {smoke.returncode}")
            if smoke.stderr.strip():
                print(smoke.stderr.strip())

    if warnings:
        print("\nWarnings")
        print("-" * 40)
        for warning in warnings:
            print(f"- {warning}")

    if launch_plan is not None:
        live_cmd = _build_step7_command(
            config_path=config_path,
            launch_plan=launch_plan,
            source_id=source_pref.requested_source_id,
        )
        replay_cmd = _shell_join(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "replay_live_capture.py"),
                "--capture-dir",
                str(launch_plan.out_dir / "parity_capture"),
            ]
        )
        audit_cmd = _shell_join(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "audit_live_parity.py"),
                "--live-dir",
                str(launch_plan.out_dir),
                "--parity-report",
                str(launch_plan.out_dir / "parity_report.json"),
                "--write-json",
                "--write-md",
            ]
        )
        print("\nRecommended commands")
        print("-" * 40)
        print(live_cmd)
        print(replay_cmd)
        print(audit_cmd)

    if errors:
        print("\nErrors")
        print("-" * 40)
        for error in errors:
            print(f"- {error}")
        return 1
    if not smoke_ok:
        return 1

    print("\nReady for live run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
