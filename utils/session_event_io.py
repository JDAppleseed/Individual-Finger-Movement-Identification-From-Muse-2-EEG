from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

PRIVATE_PAYLOAD_KEY = "_source_payload"
DEFAULT_EVENT_COLUMNS = {
    "duration_s": 0.0,
    "type": "",
    "channel": "n/a",
    "confidence": "",
    "notes": "",
    "finger_id": 0,
    "action_id": 0,
    "trial_id": 0,
    "block_id": 0,
    "session_mode": "",
    "source": "unknown",
}


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _pick_first(*values: Any, default: Any = None) -> Any:
    for value in values:
        if _is_missing(value):
            continue
        return value
    return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def json_safe(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def json_safe_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {str(key): json_safe(value) for key, value in data.items()}


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _sorted_raw_shards(raw_dir: Path) -> list[Path]:
    shard_paths = list(raw_dir.glob("eeg_raw_shard_*.npy"))
    if not shard_paths:
        return []

    def _key(path: Path) -> tuple[int, str]:
        match = re.search(r"eeg_raw_shard_(\d+)\.npy$", path.name)
        if match:
            return int(match.group(1)), path.name
        return (10**12, path.name)

    return sorted(shard_paths, key=_key)


def resolve_raw_shard_paths(session_dir: Path) -> list[Path]:
    session_dir = Path(session_dir).expanduser().resolve()
    manifest = _read_json(session_dir / "manifest.json")
    shard_paths: list[Path] = []
    if isinstance(manifest, dict):
        shard_list = manifest.get("shard_list") or []
        for item in shard_list:
            if not isinstance(item, dict):
                continue
            raw_path = item.get("path")
            if _is_missing(raw_path):
                continue
            shard_path = Path(str(raw_path))
            if not shard_path.is_absolute():
                shard_path = (session_dir / shard_path).resolve()
            if shard_path.exists():
                shard_paths.append(shard_path)
    if shard_paths:
        return shard_paths
    return _sorted_raw_shards(session_dir / "raw")


def load_event_payloads(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload, dict) and isinstance(payload.get("events"), list):
            payload = payload.get("events")
        if not isinstance(payload, list):
            return []
        return [dict(ev) for ev in payload if isinstance(ev, dict)]
    if suffix == ".jsonl":
        payloads: list[dict[str, Any]] = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                payloads.append(dict(payload))
        return payloads
    if suffix == ".csv":
        df = pd.read_csv(path)
        return [dict(row) for row in df.to_dict(orient="records")]
    return []


def normalize_event_payload(
    payload: dict[str, Any], fallback_index: int = 0
) -> Optional[dict[str, Any]]:
    if not isinstance(payload, dict):
        return None
    source_payload = dict(payload)
    metadata_raw = source_payload.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    onset_s = safe_float(
        _pick_first(
            source_payload.get("onset_s"),
            source_payload.get("event_time_s"),
            metadata.get("onset_s"),
            metadata.get("event_time_s"),
            default=np.nan,
        )
    )
    if not np.isfinite(onset_s):
        return None

    duration_s = safe_float(
        _pick_first(
            source_payload.get("duration_s"),
            metadata.get("duration_s"),
            metadata.get("duration"),
            default=0.0,
        ),
        0.0,
    )
    duration_s = max(0.0, duration_s)

    end_s = safe_float(
        _pick_first(source_payload.get("end_s"), metadata.get("end_s"), default=np.nan)
    )
    if not np.isfinite(end_s):
        end_s = float(onset_s + duration_s)

    event = {
        "onset_s": float(onset_s),
        "event_time_s": float(onset_s),
        "duration_s": float(duration_s),
        "end_s": float(end_s),
        "type": str(
            _pick_first(
                source_payload.get("type"),
                source_payload.get("label"),
                metadata.get("type"),
                metadata.get("label"),
                default="",
            )
            or ""
        ).strip(),
        "channel": str(
            _pick_first(
                source_payload.get("channel"), metadata.get("channel"), default="n/a"
            )
            or "n/a"
        ).strip()
        or "n/a",
        "confidence": _pick_first(
            source_payload.get("confidence"), metadata.get("confidence"), default=""
        ),
        "notes": str(
            _pick_first(source_payload.get("notes"), metadata.get("notes"), default="")
            or ""
        ).strip(),
        "finger_id": safe_int(
            _pick_first(
                source_payload.get("finger_id"),
                metadata.get("finger_id"),
                metadata.get("finger"),
                default=0,
            ),
            0,
        ),
        "action_id": safe_int(
            _pick_first(
                source_payload.get("action_id"),
                metadata.get("action_id"),
                metadata.get("action"),
                default=0,
            ),
            0,
        ),
        "trial_id": safe_int(
            _pick_first(
                source_payload.get("trial_id"),
                metadata.get("trial_id"),
                metadata.get("trial"),
                default=0,
            ),
            0,
        ),
        "block_id": safe_int(
            _pick_first(
                source_payload.get("block_id"),
                metadata.get("block_id"),
                metadata.get("block"),
                default=0,
            ),
            0,
        ),
        "session_mode": str(
            _pick_first(
                source_payload.get("session_mode"),
                metadata.get("session_mode"),
                metadata.get("mode"),
                default="",
            )
            or ""
        ).strip(),
        "source": str(
            _pick_first(
                source_payload.get("source"), metadata.get("source"), default="unknown"
            )
            or "unknown"
        ).strip()
        or "unknown",
        PRIVATE_PAYLOAD_KEY: source_payload,
    }

    for passthrough in ("lsl_ts_mono", "local_ts", "event_id", "event_index", "label"):
        value = source_payload.get(passthrough)
        if not _is_missing(value):
            event[passthrough] = value

    if "event_id" not in event:
        alt_id = _pick_first(source_payload.get("id"), metadata.get("event_id"))
        event["event_id"] = safe_int(alt_id, fallback_index)
    if "event_index" not in event:
        event["event_index"] = safe_int(metadata.get("event_index"), fallback_index)
    else:
        event["event_index"] = safe_int(event["event_index"], fallback_index)

    return event


def normalize_events_df(events_df: pd.DataFrame) -> pd.DataFrame:
    if events_df.empty:
        return events_df
    if "onset_s" not in events_df.columns and "event_time_s" in events_df.columns:
        events_df["onset_s"] = events_df["event_time_s"]
    if "type" not in events_df.columns and "label" in events_df.columns:
        events_df["type"] = events_df["label"]
    for col, value in DEFAULT_EVENT_COLUMNS.items():
        if col not in events_df.columns:
            events_df[col] = value
    for col in ("onset_s", "event_time_s", "duration_s", "end_s", "lsl_ts_mono", "local_ts"):
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce")
    for col in ("finger_id", "action_id", "trial_id", "block_id", "event_id", "event_index"):
        if col in events_df.columns:
            events_df[col] = pd.to_numeric(events_df[col], errors="coerce").fillna(0).astype(int)
    events_df["onset_s"] = events_df["onset_s"].fillna(0.0).astype(float)
    events_df["event_time_s"] = events_df.get("event_time_s", events_df["onset_s"])
    events_df["event_time_s"] = pd.to_numeric(
        events_df["event_time_s"], errors="coerce"
    ).fillna(events_df["onset_s"]).astype(float)
    events_df["duration_s"] = events_df["duration_s"].fillna(0.0).astype(float)
    if "end_s" in events_df.columns:
        events_df["end_s"] = events_df["end_s"].fillna(
            events_df["onset_s"] + events_df["duration_s"]
        ).astype(float)
    return events_df


def load_events_dataframe(path: Path) -> pd.DataFrame:
    payloads = load_event_payloads(path)
    events = []
    for idx, payload in enumerate(payloads):
        event = normalize_event_payload(payload, fallback_index=idx)
        if event is not None:
            events.append(event)
    return normalize_events_df(pd.DataFrame(events))


def _clean_canonical_record(row: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in row.items():
        if str(key).startswith("_"):
            continue
        cleaned[str(key)] = json_safe(value)
    onset = safe_float(cleaned.get("onset_s", cleaned.get("event_time_s", 0.0)), 0.0)
    duration = max(0.0, safe_float(cleaned.get("duration_s", 0.0), 0.0))
    cleaned["onset_s"] = float(onset)
    cleaned["event_time_s"] = float(onset)
    cleaned["duration_s"] = float(duration)
    cleaned["end_s"] = float(onset + duration)
    if "type" not in cleaned and "label" in cleaned:
        cleaned["type"] = str(cleaned.get("label", ""))
    return cleaned


def event_row_to_payload(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = _clean_canonical_record(row)
    source_payload = row.get(PRIVATE_PAYLOAD_KEY)
    payload = dict(source_payload) if isinstance(source_payload, dict) else {}

    metadata_raw = payload.get("metadata")
    metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}

    payload["onset_s"] = float(cleaned["onset_s"])
    payload["event_time_s"] = float(cleaned["event_time_s"])
    payload["duration_s"] = float(cleaned["duration_s"])
    payload["end_s"] = float(cleaned["end_s"])
    payload["type"] = str(cleaned.get("type", "") or "")
    payload["label"] = str(cleaned.get("type", "") or "")

    for field in (
        "channel",
        "confidence",
        "notes",
        "finger_id",
        "action_id",
        "trial_id",
        "block_id",
        "session_mode",
        "source",
        "lsl_ts_mono",
        "local_ts",
        "event_id",
        "event_index",
    ):
        if field in cleaned:
            payload[field] = json_safe(cleaned[field])

    metadata["duration_s"] = float(cleaned["duration_s"])
    metadata["action_id"] = safe_int(cleaned.get("action_id"), 0)
    metadata["finger_id"] = safe_int(cleaned.get("finger_id"), 0)
    metadata["source"] = str(cleaned.get("source", "unknown") or "unknown")

    for field in ("channel", "confidence", "notes", "trial_id", "block_id", "session_mode"):
        if field in cleaned and not _is_missing(cleaned[field]):
            metadata[field] = json_safe(cleaned[field])

    payload["metadata"] = metadata
    return payload


def event_row_to_flat_record(row: dict[str, Any]) -> dict[str, Any]:
    return _clean_canonical_record(row)


def event_rows_to_payloads(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event_row_to_payload(row) for row in rows]


def event_rows_to_flat_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event_row_to_flat_record(row) for row in rows]
