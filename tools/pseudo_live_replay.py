#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analyze_live_predictions import (
    load_prediction_log,
    summarize_records,
    write_segments_csv,
)
from utils.live_infer_common import (
    ReplayRuntimeConfig,
    compute_replay_metrics,
    load_model_artifacts,
    replay_ordered_windows,
    require_deployable_run,
    write_predictions_jsonl,
)
from utils.postprocess import PostprocessSettings
from utils.runtime_utils import now_utc_iso
from utils.sequence_data import load_sequence_npz
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir


def _load_config(path: str | None) -> tuple[dict[str, Any], Path | None]:
    if not path:
        return {}, None
    config_path = Path(path).expanduser()
    if config_path.exists():
        try:
            config_path = config_path.resolve()
        except Exception:
            pass
    payload = json.loads(config_path.read_text())
    settings = payload.get("settings", payload)
    return dict(settings or {}), config_path.parent


def _resolve_path(value: Any, *, base_dir: Path | None = None) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)
        candidates.append(REPO_ROOT / path)
        if base_dir is not None:
            candidates.append(base_dir / path)
    for candidate in candidates:
        if candidate.exists():
            try:
                return candidate.resolve()
            except Exception:
                return candidate
    if base_dir is not None and not path.is_absolute():
        return base_dir / path
    if not path.is_absolute():
        return REPO_ROOT / path
    return path


def _coerce_bool(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _resolve_device(requested: str) -> torch.device:
    text = str(requested or "auto").strip().lower()
    if text == "auto":
        if sys.platform == "darwin" and getattr(torch.backends, "mps", None) is not None:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(text)


def _resolve_run_and_session(
    *,
    run_dir_arg: str | None,
    session_dir_arg: str | None,
    settings: dict[str, Any],
    base_dir: Path | None,
) -> tuple[Path, Path]:
    run_dir = _resolve_path(
        run_dir_arg or settings.get("run_dir") or settings.get("deployment_run_dir"),
        base_dir=base_dir,
    )
    session_dir = _resolve_path(
        session_dir_arg
        or settings.get("session_dir")
        or settings.get("deployment_session_dir"),
        base_dir=base_dir,
    )
    if run_dir is None:
        if session_dir is None:
            raise SystemExit("Provide --run-dir or --session-dir (or configure one in --config).")
        latest = resolve_latest_run_dir(resolve_session_dir(session_dir))
        if latest is None:
            raise SystemExit(f"No model run found under session: {session_dir}")
        run_dir = latest
    run_dir = Path(run_dir).expanduser().resolve()
    if session_dir is None:
        session_dir = run_dir.parents[2]
    session_dir = resolve_session_dir(session_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    if not session_dir.exists():
        raise SystemExit(f"Session directory not found: {session_dir}")
    return run_dir, session_dir


def _resolve_target_session_dirs(
    *,
    cli_values: Optional[Iterable[str]],
    settings: dict[str, Any],
    base_dir: Path | None,
) -> list[Path]:
    raw_values: list[Any] = []
    if cli_values:
        raw_values.extend(list(cli_values))
    else:
        for key in ("target_session_dirs", "current_replay_targets"):
            value = settings.get(key)
            if isinstance(value, list):
                raw_values.extend(value)
        for key in ("target_session_dir", "realism_target_session_dir"):
            value = settings.get(key)
            if value:
                raw_values.append(value)
    targets: list[Path] = []
    seen: set[str] = set()
    for value in raw_values:
        path = _resolve_path(value, base_dir=base_dir)
        if path is None:
            continue
        resolved = resolve_session_dir(path)
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        targets.append(resolved)
    if not targets:
        raise SystemExit(
            "Provide at least one --target-session-dir or configure target_session_dirs/current_replay_targets."
        )
    return targets


def _load_json_file(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    settings = payload.get("settings")
    return dict(settings) if isinstance(settings, dict) else dict(payload)


def _build_postprocess_settings(infer_settings: dict[str, Any]) -> PostprocessSettings:
    defaults = PostprocessSettings()
    kwargs = {
        field.name: infer_settings.get(field.name, getattr(defaults, field.name))
        for field in fields(PostprocessSettings)
    }
    return PostprocessSettings(**kwargs)


def _build_runtime_config(
    *,
    infer_settings: dict[str, Any],
    settings: dict[str, Any],
    latency_mode: str | None,
    fixed_latency_ms: float | None,
    reset_on_trial_change: Optional[bool],
    deterministic: Optional[bool],
) -> ReplayRuntimeConfig:
    defaults = ReplayRuntimeConfig()
    kwargs = {field.name: getattr(defaults, field.name) for field in fields(ReplayRuntimeConfig)}
    for key in kwargs:
        if key in infer_settings:
            kwargs[key] = infer_settings[key]
        if key in settings:
            kwargs[key] = settings[key]
    kwargs["latency_mode"] = (
        str(latency_mode)
        if latency_mode is not None
        else str(settings.get("latency_mode", kwargs["latency_mode"]))
    )
    kwargs["fixed_latency_ms"] = (
        float(fixed_latency_ms)
        if fixed_latency_ms is not None
        else (
            float(settings["fixed_latency_ms"])
            if settings.get("fixed_latency_ms") is not None
            else kwargs["fixed_latency_ms"]
        )
    )
    kwargs["reset_on_trial_change"] = (
        bool(reset_on_trial_change)
        if reset_on_trial_change is not None
        else _coerce_bool(settings.get("reset_on_trial_change"), kwargs["reset_on_trial_change"])
    )
    kwargs["deterministic"] = (
        bool(deterministic)
        if deterministic is not None
        else _coerce_bool(settings.get("deterministic"), kwargs["deterministic"])
    )
    if str(kwargs["latency_mode"]).strip().lower() == "fixed" and kwargs["fixed_latency_ms"] is None:
        raise SystemExit("--fixed-latency-ms is required when --latency-mode=fixed")
    return ReplayRuntimeConfig(
        window_sec=float(kwargs["window_sec"]),
        hop_sec=float(kwargs["hop_sec"]),
        latency_threshold_ms=float(kwargs["latency_threshold_ms"]),
        actuation_min_prob=float(kwargs["actuation_min_prob"]),
        actuation_stability=int(kwargs["actuation_stability"]),
        actuation_cooldown_ms=int(kwargs["actuation_cooldown_ms"]),
        actuation_repeat_ms=int(kwargs["actuation_repeat_ms"]),
        actuation_min_speed=float(kwargs["actuation_min_speed"]),
        modulate_actuation_speed=_coerce_bool(kwargs["modulate_actuation_speed"], True),
        actuation_speed_gamma=float(kwargs["actuation_speed_gamma"]),
        use_inference_engine=_coerce_bool(kwargs["use_inference_engine"], False),
        mc_passes=int(kwargs["mc_passes"]),
        uncertainty_base_threshold=float(kwargs["uncertainty_base_threshold"]),
        uncertainty_weight=float(kwargs["uncertainty_weight"]),
        latency_mode=str(kwargs["latency_mode"]),
        fixed_latency_ms=(
            float(kwargs["fixed_latency_ms"])
            if kwargs["fixed_latency_ms"] is not None
            else None
        ),
        reset_on_trial_change=bool(kwargs["reset_on_trial_change"]),
        deterministic=bool(kwargs["deterministic"]),
    )


def _ensure_windows_ntc(X: np.ndarray, meta: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Expected X to be 3D, got shape {arr.shape}")
    channel_names = meta.get("channel_names")
    if channel_names is not None:
        try:
            channel_count = int(len(np.asarray(channel_names)))
        except Exception:
            channel_count = None
        else:
            if arr.shape[2] == channel_count:
                return arr
            if arr.shape[1] == channel_count and arr.shape[2] != channel_count:
                return np.transpose(arr, (0, 2, 1))
    if arr.shape[1] <= 16 and arr.shape[2] > 16:
        return np.transpose(arr, (0, 2, 1))
    return arr


def _meta_array(
    meta: dict[str, Any],
    key: str,
    n: int,
    *,
    dtype: Any | None = None,
) -> np.ndarray | None:
    if key not in meta:
        return None
    arr = np.asarray(meta[key])
    if arr.ndim == 0:
        arr = np.full(n, arr.item(), dtype=arr.dtype if dtype is None else dtype)
    elif len(arr) != n:
        return None
    if dtype is None:
        return arr
    return np.asarray(arr, dtype=dtype)


def _resolve_output_dir(
    *,
    explicit_output_dir: str | None,
    base_dir: Path | None,
    target_session_dir: Path,
    run_dir: Path,
    multi_target: bool,
) -> Path:
    if explicit_output_dir:
        root = _resolve_path(explicit_output_dir, base_dir=base_dir)
        if root is None:
            raise SystemExit(f"Unable to resolve output directory: {explicit_output_dir}")
        return root / target_session_dir.name if multi_target else root
    return SessionLayout(target_session_dir).pseudo_live_root / run_dir.name


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _assert_deployment_replay_ok(
    *,
    target_session_dir: Path,
    summary: dict[str, Any],
    replay_metrics: dict[str, Any],
) -> None:
    summary_committed = int(summary.get("committed_non_rest_none_count", 0) or 0)
    summary_rest = int(summary.get("committed_rest_non_none_count", 0) or 0)
    summary_sent = int(summary.get("sent_non_rest_none_count", 0) or 0)
    summary_sent_rest = int(summary.get("sent_rest_non_none_count", 0) or 0)
    replay_committed = int(replay_metrics.get("committed_non_rest_none_count", 0) or 0)
    replay_rest = int(replay_metrics.get("committed_rest_non_none_count", 0) or 0)
    replay_sent = int(replay_metrics.get("sent_non_rest_none_count", 0) or 0)
    replay_sent_rest = int(replay_metrics.get("sent_rest_non_none_count", 0) or 0)
    summary_ok = bool(summary.get("deployment_pair_invariant_ok", False))
    replay_ok = bool(replay_metrics.get("deployment_pair_invariant_ok", False))
    if (
        summary_committed != 0
        or summary_rest != 0
        or summary_sent != 0
        or summary_sent_rest != 0
        or replay_committed != 0
        or replay_rest != 0
        or replay_sent != 0
        or replay_sent_rest != 0
        or not summary_ok
        or not replay_ok
    ):
        raise SystemExit(
            "Deployment replay invariant failed for "
            f"{target_session_dir}: "
            f"summary committed_non_rest_none={summary_committed}, "
            f"summary committed_rest_non_none={summary_rest}, "
            f"summary sent_non_rest_none={summary_sent}, "
            f"summary sent_rest_non_none={summary_sent_rest}, "
            f"replay committed_non_rest_none={replay_committed}, "
            f"replay committed_rest_non_none={replay_rest}, "
            f"replay sent_non_rest_none={replay_sent}, "
            f"replay sent_rest_non_none={replay_sent_rest}, "
            f"summary_ok={summary_ok}, replay_ok={replay_ok}"
        )


def _run_single_replay(
    *,
    run_dir: Path,
    source_session_dir: Path,
    target_session_dir: Path,
    infer_config_path: Path,
    output_dir: Path,
    device: torch.device,
    benchmark_label: Optional[str],
    runtime_config: ReplayRuntimeConfig,
    postprocess_enabled: bool,
    postprocess_settings: PostprocessSettings,
) -> dict[str, Any]:
    layout = SessionLayout(target_session_dir)
    if not layout.windows_npz.exists():
        raise SystemExit(f"Target dataset not found: {layout.windows_npz}")

    X, y_action, y_finger, meta = load_sequence_npz(layout.windows_npz)
    X = _ensure_windows_ntc(X, meta)
    n = int(len(y_action))
    if n == 0:
        raise SystemExit(f"Target dataset has no windows: {layout.windows_npz}")

    window_start = _meta_array(meta, "window_start", n, dtype=np.float32)
    window_end = _meta_array(meta, "window_end", n, dtype=np.float32)
    if window_start is None or window_end is None:
        raise SystemExit(f"Target dataset is missing window_start/window_end: {layout.windows_npz}")

    order = np.argsort(window_start.astype(np.float64), kind="stable")
    X = X[order]
    y_action = np.asarray(y_action, dtype=np.int64)[order]
    y_finger = np.asarray(y_finger, dtype=np.int64)[order]
    window_start = np.asarray(window_start, dtype=np.float32)[order]
    window_end = np.asarray(window_end, dtype=np.float32)[order]
    trial_ids = _meta_array(meta, "trial_id", n, dtype=np.int64)
    if trial_ids is not None:
        trial_ids = trial_ids[order]
    session_ids = _meta_array(meta, "session_id", n)
    if session_ids is not None:
        session_ids = session_ids.astype("U")[order]
    event_ids = _meta_array(meta, "event_id", n, dtype=np.int64)
    if event_ids is not None:
        event_ids = event_ids[order]
    event_onset_s = _meta_array(meta, "event_onset_s", n, dtype=np.float32)
    if event_onset_s is not None:
        event_onset_s = event_onset_s[order]

    model, scaler, temperature_state = load_model_artifacts(
        run_dir=run_dir,
        device=device,
        n_channels=int(X.shape[-1]),
    )
    replay = replay_ordered_windows(
        X=X,
        window_start_s=window_start,
        window_end_s=window_end,
        y_action_true=y_action,
        y_finger_true=y_finger,
        trial_ids=trial_ids,
        session_ids=session_ids,
        event_ids=event_ids,
        event_onset_s=event_onset_s,
        scaler=scaler,
        model=model,
        device=device,
        postprocess_enabled=postprocess_enabled,
        postprocess_settings=postprocess_settings,
        runtime_config=runtime_config,
        temperature_state=temperature_state,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_log_path = output_dir / "predictions.jsonl"
    summary_json_path = output_dir / "live_prediction_summary.json"
    segments_csv_path = output_dir / "predicted_segments.csv"
    review_csv_path = output_dir / "predicted_segments_review.csv"
    manifest_path = output_dir / "replay_manifest.json"

    write_predictions_jsonl(replay["records"], pred_log_path)
    records = load_prediction_log(pred_log_path)
    summary_bundle = summarize_records(records)
    summary = dict(summary_bundle["summary"])
    summary["pred_log_path"] = str(pred_log_path)
    summary["target_session_dir"] = str(target_session_dir)
    summary["run_dir"] = str(run_dir)
    _write_json(summary_json_path, summary)
    write_segments_csv(
        segments_csv_path,
        summary_bundle["segments"],
        include_review_columns=False,
    )
    write_segments_csv(
        review_csv_path,
        summary_bundle["segments"],
        include_review_columns=True,
    )

    replay_metrics = compute_replay_metrics(
        records=records,
        y_action_true=y_action,
        y_finger_true=y_finger,
        window_start_s=window_start,
        window_end_s=window_end,
        trial_ids=trial_ids,
        session_ids=session_ids,
        event_ids=event_ids,
        event_onset_s=event_onset_s,
        applicability_threshold=float(postprocess_settings.threshold_applicability),
    )

    manifest = {
        "created_at": now_utc_iso(),
        "benchmark_label": benchmark_label,
        "source_session_dir": str(source_session_dir),
        "target_session_dir": str(target_session_dir),
        "run_dir": str(run_dir),
        "infer_config": str(infer_config_path),
        "predictions_jsonl": str(pred_log_path),
        "live_prediction_summary_json": str(summary_json_path),
        "predicted_segments_csv": str(segments_csv_path),
        "predicted_segments_review_csv": str(review_csv_path),
        "record_count": int(len(records)),
        "device": str(device),
        "postprocess": {
            "enabled": bool(postprocess_enabled),
            "settings": asdict(postprocess_settings),
        },
        "runtime_config": asdict(runtime_config),
        "summary": summary,
        "replay_metrics": replay_metrics,
    }
    _write_json(manifest_path, manifest)
    _assert_deployment_replay_ok(
        target_session_dir=target_session_dir,
        summary=summary,
        replay_metrics=replay_metrics,
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay Step 7-style inference and actuation logic on recorded windows "
            "without contacting hardware."
        )
    )
    parser.add_argument("--config", type=str, default=None, help="Optional pseudo-live config JSON.")
    parser.add_argument("--run-dir", type=str, default=None, help="Specific model run directory.")
    parser.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Session directory used to resolve the latest model run when --run-dir is omitted.",
    )
    parser.add_argument(
        "--target-session-dir",
        action="append",
        default=None,
        help="Target session directory to replay (repeatable).",
    )
    parser.add_argument(
        "--infer-config",
        type=str,
        default=None,
        help="Step 7 infer.json used to source postprocess and actuation settings.",
    )
    parser.add_argument(
        "--latency-mode",
        type=str,
        default=None,
        choices=["ignore", "compute", "fixed"],
        help="Latency policy used during offline replay.",
    )
    parser.add_argument(
        "--fixed-latency-ms",
        type=float,
        default=None,
        help="Fixed latency override when --latency-mode=fixed.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional output directory override.")
    parser.add_argument(
        "--reset-on-trial-change",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reset postprocess and actuation state when trial_id changes.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable deterministic replay settings.",
    )
    parser.add_argument("--device", type=str, default="auto", help="Torch device override.")
    parser.add_argument(
        "--benchmark-label",
        type=str,
        default=None,
        help="Optional label recorded in the replay manifest.",
    )
    args = parser.parse_args()

    settings, config_dir = _load_config(args.config)
    run_dir, session_dir = _resolve_run_and_session(
        run_dir_arg=args.run_dir,
        session_dir_arg=args.session_dir,
        settings=settings,
        base_dir=config_dir,
    )
    try:
        require_deployable_run(run_dir)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    infer_config_path = _resolve_path(
        args.infer_config or settings.get("infer_config"),
        base_dir=config_dir,
    )
    if infer_config_path is None or not infer_config_path.exists():
        raise SystemExit("Provide --infer-config or configure infer_config in --config.")
    infer_config_path = infer_config_path.resolve()
    infer_settings = _load_json_file(infer_config_path)
    postprocess_enabled = _coerce_bool(infer_settings.get("postprocess"), True)
    postprocess_settings = _build_postprocess_settings(infer_settings)
    runtime_config = _build_runtime_config(
        infer_settings=infer_settings,
        settings=settings,
        latency_mode=args.latency_mode,
        fixed_latency_ms=args.fixed_latency_ms,
        reset_on_trial_change=args.reset_on_trial_change,
        deterministic=args.deterministic,
    )
    target_session_dirs = _resolve_target_session_dirs(
        cli_values=args.target_session_dir,
        settings=settings,
        base_dir=config_dir,
    )
    device = _resolve_device(args.device)

    manifests: List[dict[str, Any]] = []
    multi_target = len(target_session_dirs) > 1
    for target_session_dir in target_session_dirs:
        output_dir = _resolve_output_dir(
            explicit_output_dir=args.output_dir,
            base_dir=config_dir,
            target_session_dir=target_session_dir,
            run_dir=run_dir,
            multi_target=multi_target,
        )
        manifest = _run_single_replay(
            run_dir=run_dir,
            source_session_dir=session_dir,
            target_session_dir=target_session_dir,
            infer_config_path=infer_config_path,
            output_dir=output_dir,
            device=device,
            benchmark_label=args.benchmark_label or settings.get("benchmark_label"),
            runtime_config=runtime_config,
            postprocess_enabled=postprocess_enabled,
            postprocess_settings=postprocess_settings,
        )
        manifests.append(manifest)

    if len(manifests) == 1:
        print(json.dumps(manifests[0], indent=2, sort_keys=True))
    else:
        print(json.dumps({"replays": manifests}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
