from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence


class LSLStreamSelectError(RuntimeError):
    pass


class NoStreamFoundError(LSLStreamSelectError):
    pass


class NoStreamMatchedError(LSLStreamSelectError):
    pass


class MultipleStreamsMatchedError(LSLStreamSelectError):
    pass

try:
    from pylsl import StreamInfo, resolve_streams

    LSL_AVAILABLE = True
except Exception:
    StreamInfo = Any  # type: ignore[assignment,misc]
    resolve_streams = None
    LSL_AVAILABLE = False


@dataclass(frozen=True)
class StreamSelector:
    name_contains: Optional[str]
    type_equals: Optional[str]
    min_channels: int
    require_unique: bool = True


def _safe_attr(callable_obj) -> Optional[str]:
    try:
        value = callable_obj()
    except Exception:
        return None
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def stream_signature(info: StreamInfo) -> Dict[str, Any]:
    source_id = None
    uid = None
    if hasattr(info, "source_id"):
        source_id = _safe_attr(info.source_id)
    if hasattr(info, "uid"):
        uid = _safe_attr(info.uid)
    labels = extract_channel_labels(info)
    return {
        "name": str(info.name()),
        "type": str(info.type()),
        "channel_count": int(info.channel_count()),
        "nominal_srate": float(info.nominal_srate()),
        "source_id": source_id,
        "uid": uid,
        "channel_labels": labels,
        "labels": labels,
    }


def extract_channel_labels(info: StreamInfo) -> List[str]:
    labels: List[str] = []
    try:
        ch = info.desc().child("channels").child("channel")
    except Exception:
        ch = None
    if ch is None:
        return labels
    for _ in range(info.channel_count()):
        try:
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
        except Exception:
            break
    return [label for label in labels if label]


def list_streams() -> List[Dict[str, Any]]:
    if not LSL_AVAILABLE or resolve_streams is None:
        raise RuntimeError("pylsl is not available")
    streams = resolve_streams()
    return [stream_signature(info) for info in streams]


def _format_candidates(candidates: Iterable[Dict[str, Any]]) -> str:
    lines = []
    for candidate in candidates:
        name = candidate.get("name")
        stream_type = candidate.get("type")
        channels = candidate.get("channel_count")
        source_id = candidate.get("source_id")
        uid = candidate.get("uid")
        parts = [
            f"name={name}",
            f"type={stream_type}",
            f"channels={channels}",
        ]
        if source_id:
            parts.append(f"source_id={source_id}")
        if uid:
            parts.append(f"uid={uid}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _normalize_selector(selector: StreamSelector) -> StreamSelector:
    return StreamSelector(
        name_contains=selector.name_contains.lower()
        if selector.name_contains
        else None,
        type_equals=selector.type_equals.lower() if selector.type_equals else None,
        min_channels=int(selector.min_channels),
        require_unique=bool(selector.require_unique),
    )


def select_stream_candidate(
    candidates: Sequence[Dict[str, Any]], selector: StreamSelector
) -> Dict[str, Any]:
    normalized = _normalize_selector(selector)
    filtered: List[Dict[str, Any]] = []
    for candidate in candidates:
        name = str(candidate.get("name") or "").lower()
        stream_type = str(candidate.get("type") or "").lower()
        channel_count = int(candidate.get("channel_count") or 0)
        if channel_count < normalized.min_channels:
            continue
        if normalized.name_contains and normalized.name_contains not in name:
            continue
        if normalized.type_equals and normalized.type_equals != stream_type:
            continue
        filtered.append(candidate)

    if not filtered:
        if not candidates:
            raise NoStreamFoundError("No LSL streams found.")
        candidate_list = _format_candidates(candidates)
        raise NoStreamMatchedError(
            "No LSL streams matched the selector. "
            "Set LSL_STREAM_NAME or LSL_STREAM_TYPE to disambiguate. "
            f"Selector={normalized}.\nStreams found:\n{candidate_list}"
        )

    if normalized.require_unique and len(filtered) > 1:
        candidate_list = _format_candidates(filtered)
        raise MultipleStreamsMatchedError(
            "Multiple LSL streams matched the selector. "
            "Set LSL_STREAM_NAME or LSL_STREAM_TYPE to disambiguate. "
            f"Selector={normalized}.\nMatches:\n{candidate_list}"
        )

    return filtered[0]


def pick_stream(selector: StreamSelector) -> StreamInfo:
    if not LSL_AVAILABLE or resolve_streams is None:
        raise RuntimeError("pylsl is not available")
    streams = resolve_streams()
    candidates = [
        {
            **stream_signature(info),
            "info": info,
        }
        for info in streams
    ]
    chosen = select_stream_candidate(candidates, selector)
    return chosen["info"]


def log_stream_signature(signature: Dict[str, Any]) -> None:
    parts = [
        f"name={signature.get('name')}",
        f"type={signature.get('type')}",
        f"channels={signature.get('channel_count')}",
        f"nominal_srate={signature.get('nominal_srate')}",
    ]
    labels = signature.get("labels")
    if labels:
        parts.append(f"labels={labels}")
    source_id = signature.get("source_id")
    uid = signature.get("uid")
    if source_id:
        parts.append(f"source_id={source_id}")
    if uid:
        parts.append(f"uid={uid}")
    print("🔗 LSL stream selected: " + ", ".join(parts))
