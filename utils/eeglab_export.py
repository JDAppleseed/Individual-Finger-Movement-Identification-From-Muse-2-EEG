from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.io import savemat

from utils.session_event_io import (
    load_event_payloads,
    normalize_event_payload,
    resolve_raw_shard_paths,
)


@dataclass(frozen=True)
class EeglabExportSummary:
    session_dir: Path
    out_path: Path
    sample_count: int
    channel_count: int
    event_count: int
    skipped_event_count: int
    sampling_rate: float


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _session_meta(session_dir: Path) -> dict[str, Any]:
    for name in ("meta.json", "session_meta.json", "manifest.json"):
        payload = _read_json(session_dir / name)
        if payload:
            return payload
    return {}


def _channel_labels(meta: dict[str, Any], channel_count: int) -> list[str]:
    labels = meta.get("channel_labels")
    if isinstance(labels, list) and len(labels) == channel_count:
        return [str(label) for label in labels]
    return [f"ch{i + 1}" for i in range(channel_count)]


def _sampling_rate(meta: dict[str, Any], lsl_ts_mono: np.ndarray) -> float:
    for key in ("sampling_rate", "sampling_rate_hz", "expected_sampling_rate"):
        try:
            value = float(meta.get(key))
        except Exception:
            value = 0.0
        if np.isfinite(value) and value > 0:
            return value
    if lsl_ts_mono.size >= 2:
        diffs = np.diff(lsl_ts_mono.astype(float))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(1.0 / np.median(diffs))
    return 256.0


def _existing_file(candidate: Path) -> Optional[Path]:
    try:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    except Exception:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _event_path_from_value(session_dir: Path, value: Any) -> Optional[Path]:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = session_dir / candidate
    return _existing_file(candidate)


def resolve_eeglab_export_events_path(session_dir: Path | str) -> Optional[Path]:
    session_dir = Path(session_dir).expanduser()
    if session_dir.exists():
        session_dir = session_dir.resolve()

    seen: set[str] = set()

    def _remember(candidate: Optional[Path]) -> Optional[Path]:
        if candidate is None:
            return None
        key = str(candidate)
        if key in seen:
            return None
        seen.add(key)
        return candidate

    for rel in ("events/events.jsonl", "events/events.json", "events/events.csv"):
        candidate = _remember(_existing_file(session_dir / rel))
        if candidate is not None:
            return candidate

    metadata_keys = (
        "events_jsonl_path",
        "events_json_path",
        "events_csv_path",
        "events_jsonl",
        "events_json",
        "events_csv",
        "events_path",
    )
    for name in ("meta.json", "session_meta.json", "manifest.json", "run_meta.json"):
        payload = _read_json(session_dir / name)
        if not isinstance(payload, dict):
            continue
        for key in metadata_keys:
            candidate = _remember(_event_path_from_value(session_dir, payload.get(key)))
            if candidate is not None:
                return candidate
        files = payload.get("files")
        if isinstance(files, dict):
            for key in metadata_keys:
                candidate = _remember(_event_path_from_value(session_dir, files.get(key)))
                if candidate is not None:
                    return candidate

    events_dir = session_dir / "events"
    if events_dir.exists():
        for pattern in ("*.jsonl", "*.json", "*.csv"):
            for path in sorted(events_dir.glob(pattern)):
                candidate = _remember(_existing_file(path))
                if candidate is not None:
                    return candidate
    return None


def _mat_struct_array(rows: list[dict[str, Any]], fields: list[str]) -> np.ndarray:
    dtype = [(field, "O") for field in fields]
    arr = np.empty((1, len(rows)), dtype=dtype)
    for idx, row in enumerate(rows):
        for field in fields:
            arr[field][0, idx] = row.get(field)
    return arr


def default_eeglab_export_path(session_dir: Path) -> Path:
    session_dir = Path(session_dir).expanduser()
    return session_dir / "exports" / f"{session_dir.name}_eeglab.set"


def export_session_to_eeglab(
    session_dir: Path | str,
    out_path: Path | str | None = None,
) -> EeglabExportSummary:
    session_dir = Path(session_dir).expanduser()
    if session_dir.exists():
        session_dir = session_dir.resolve()
    if not session_dir.exists() or not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    shard_paths = resolve_raw_shard_paths(session_dir)
    if not shard_paths:
        raise FileNotFoundError(f"No raw shards found under {session_dir / 'raw'}")

    raw_chunks = [np.load(path) for path in shard_paths]
    raw = np.concatenate(raw_chunks) if len(raw_chunks) > 1 else raw_chunks[0]
    if raw.size == 0:
        raise RuntimeError(f"Raw shards are empty for {session_dir}")
    if "sample" not in raw.dtype.names:
        raise RuntimeError("Raw shard format missing required 'sample' field.")
    if "lsl_ts_mono" not in raw.dtype.names:
        raise RuntimeError("Raw shard format missing required 'lsl_ts_mono' field.")

    signal = np.asarray(raw["sample"], dtype=np.float64)
    if signal.ndim != 2:
        raise RuntimeError("Expected raw['sample'] to have shape (samples, channels).")
    sample_count, channel_count = signal.shape

    meta = _session_meta(session_dir)
    srate = _sampling_rate(meta, np.asarray(raw["lsl_ts_mono"], dtype=np.float64))
    labels = _channel_labels(meta, channel_count)

    event_rows: list[dict[str, Any]] = []
    skipped_event_count = 0
    events_path = resolve_eeglab_export_events_path(session_dir)
    if events_path is not None:
        for idx, payload in enumerate(load_event_payloads(events_path)):
            event = normalize_event_payload(payload, fallback_index=idx)
            if event is None:
                continue
            onset_s = float(event.get("onset_s", 0.0))
            latency = 1.0 + onset_s * srate
            if latency < 1.0 or latency > float(sample_count):
                skipped_event_count += 1
                continue
            duration_s = max(0.0, float(event.get("duration_s", 0.0)))
            event_rows.append(
                {
                    "type": str(event.get("type") or event.get("label") or f"event_{idx + 1}"),
                    "latency": float(latency),
                    "duration": float(duration_s * srate),
                    "urevent": float(len(event_rows) + 1),
                }
            )

    chanlocs = _mat_struct_array([{"labels": label} for label in labels], ["labels"])
    events = _mat_struct_array(event_rows, ["type", "latency", "duration", "urevent"])

    out_path = Path(out_path) if out_path is not None else default_eeglab_export_path(session_dir)
    out_path = out_path.expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    subject_id = str(meta.get("subject_id") or "")
    session_id = str(meta.get("session_id") or session_dir.name)
    set_name = f"{subject_id}_{session_id}".strip("_") or session_dir.name
    eeg = {
        "setname": set_name,
        "filename": out_path.name,
        "filepath": str(out_path.parent),
        "subject": subject_id,
        "session": session_id,
        "comments": f"Exported from {session_dir}",
        "nbchan": float(channel_count),
        "trials": float(1),
        "pnts": float(sample_count),
        "srate": float(srate),
        "xmin": float(0.0),
        "xmax": float((sample_count - 1) / srate) if sample_count else float(0.0),
        "times": (np.arange(sample_count, dtype=np.float64) / srate * 1000.0).reshape(1, -1),
        "data": np.ascontiguousarray(signal.T),
        "chanlocs": chanlocs,
        "event": events,
        "urevent": events.copy(),
        "icaact": np.empty((0, 0), dtype=np.float64),
        "icawinv": np.empty((0, 0), dtype=np.float64),
        "icasphere": np.empty((0, 0), dtype=np.float64),
        "icaweights": np.empty((0, 0), dtype=np.float64),
        "saved": "yes",
        "etc": {
            "source_session_dir": str(session_dir),
            "source_raw_shards": [str(path) for path in shard_paths],
            "source_events_path": str(events_path) if events_path else "",
            "skipped_event_count": float(skipped_event_count),
        },
    }
    savemat(str(out_path), {"EEG": eeg}, do_compression=True, long_field_names=True)

    return EeglabExportSummary(
        session_dir=session_dir,
        out_path=out_path,
        sample_count=sample_count,
        channel_count=channel_count,
        event_count=len(event_rows),
        skipped_event_count=skipped_event_count,
        sampling_rate=srate,
    )
