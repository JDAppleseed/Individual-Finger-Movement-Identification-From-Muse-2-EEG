#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ui_config_validation import validate_live_infer
from utils.live_infer_common import require_deployable_run, resolve_temperature_path
from utils.session_layout import SessionLayout


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the current Step 7 deployment config before a live run."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to infer.json",
    )
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Optional session directory override for smoke inference",
    )
    parser.add_argument(
        "--smoke-device",
        type=str,
        default="cpu",
        help="Device for the optional smoke inference check",
    )
    parser.add_argument(
        "--smoke-index",
        type=int,
        default=0,
        help="Window index for smoke inference",
    )
    parser.add_argument(
        "--skip-smoke",
        action="store_true",
        help="Skip the smoke inference command",
    )
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    settings = _load_settings(config_path)
    validation = validate_live_infer(settings)

    session_dir = (
        Path(args.session_dir).expanduser().resolve()
        if args.session_dir
        else _resolve_repo_path(settings.get("session_dir"))
    )
    model_path = _resolve_repo_path(settings.get("model_path"))
    scaler_path = _resolve_repo_path(settings.get("scaler_path"))
    run_dir = model_path.parent.resolve() if model_path is not None else None
    temperature_path = resolve_temperature_path(run_dir) if run_dir is not None else None
    windows_npz = SessionLayout(session_dir).windows_npz if session_dir is not None else None

    errors = list(validation.errors)
    warnings = list(validation.warnings)

    if model_path is None or not model_path.exists():
        errors.append(f"model_path not found: {model_path}")
    if scaler_path is None or not scaler_path.exists():
        errors.append(f"scaler_path not found: {scaler_path}")
    if temperature_path is None or not temperature_path.exists():
        errors.append(f"temperature_scaling.json not found: {temperature_path}")
    if session_dir is None or not session_dir.exists():
        warnings.append(f"session_dir missing or unresolved: {session_dir}")
    elif not windows_npz.exists():
        warnings.append(f"session windows NPZ missing for smoke inference: {windows_npz}")

    deployable_info: dict[str, Any] | None = None
    if run_dir is not None and run_dir.exists():
        try:
            deployable_info = require_deployable_run(run_dir)
        except RuntimeError as exc:
            errors.append(str(exc))

    print("Live preflight")
    print("-" * 40)
    _print_kv("config", config_path)
    _print_kv("session_dir", session_dir)
    _print_kv("run_dir", run_dir)
    _print_kv("model_path", model_path)
    _print_kv("scaler_path", scaler_path)
    _print_kv("temperature_path", temperature_path)
    _print_kv("stream_name", settings.get("stream_name") or settings.get("LSL_STREAM_NAME"))
    _print_kv("stream_type", settings.get("stream_type") or settings.get("LSL_STREAM_TYPE"))
    _print_kv("lsl_source_id", settings.get("lsl_source_id"))
    _print_kv("postprocess", settings.get("postprocess"))
    _print_kv("actuation_min_prob", settings.get("actuation_min_prob"))
    _print_kv("actuation_stability", settings.get("actuation_stability"))
    _print_kv("actuation_repeat_ms", settings.get("actuation_repeat_ms"))
    _print_kv("live_quality_enabled", settings.get("live_quality_enabled"))

    if deployable_info is not None:
        _print_kv("deployable_run", deployable_info.get("deployable"))
        _print_kv("active_finger_head", deployable_info.get("active_finger_head"))
        _print_kv(
            "finger_applicability_head",
            deployable_info.get("finger_applicability_head"),
        )

    if warnings:
        print("\nWarnings")
        print("-" * 40)
        for warning in warnings:
            print(f"- {warning}")

    smoke_ok = True
    if not args.skip_smoke and windows_npz is not None and windows_npz.exists():
        smoke_cmd = [
            sys.executable,
            str(REPO_ROOT / "tools" / "smoke_inference.py"),
            "--npz",
            str(windows_npz),
            "--model",
            str(model_path),
            "--scaler",
            str(scaler_path),
            "--index",
            str(int(args.smoke_index)),
            "--device",
            str(args.smoke_device),
        ]
        print("\nSmoke inference")
        print("-" * 40)
        print(" ".join(smoke_cmd))
        smoke = subprocess.run(smoke_cmd, capture_output=True, text=True)
        if smoke.stdout.strip():
            print(smoke.stdout.strip())
        if smoke.returncode != 0:
            smoke_ok = False
            errors.append(f"smoke inference failed with exit code {smoke.returncode}")
            if smoke.stderr.strip():
                print(smoke.stderr.strip())
    elif not args.skip_smoke:
        warnings.append("Smoke inference skipped because session windows were unavailable.")

    probe_cmd = (
        f"{sys.executable} tools/lsl_sanity_probe.py "
        f"--name-contains \"{settings.get('stream_name') or settings.get('LSL_STREAM_NAME') or 'Muse2-EEG'}\" "
        "--min-channels 4 --seconds 10"
    )
    live_cmd = (
        f"{sys.executable} 7_live_infer_and_actuate.py "
        f"--config {config_path} "
        f"--session-dir {session_dir}"
    )

    print("\nSuggested commands")
    print("-" * 40)
    print(probe_cmd)
    print(live_cmd)

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
