from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from utils.channel_labels import extract_lsl_channel_labels, normalize_channel_labels

AUTO_SOURCE_ID_TOKENS = frozenset({"auto", "muse2_internal", "internal"})


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


@dataclass(frozen=True)
class SourceIdPreference:
    cli_source_id: Optional[str]
    env_source_id: Optional[str]
    config_source_id: Optional[str]
    requested_source_id: Optional[str]
    source: str


@dataclass(frozen=True)
class SourceIdSelection:
    selected: Dict[str, Any]
    requested_source_id: Optional[str]
    selected_source_id: Optional[str]
    recovery_used: bool


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
    return normalize_channel_labels(extract_lsl_channel_labels(info), dedupe=False)


def normalize_source_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in AUTO_SOURCE_ID_TOKENS:
        return None
    return text


def resolve_source_id_preference(
    cli_source_id: Any,
    env_source_id: Any,
    config_source_id: Any,
) -> SourceIdPreference:
    cli = normalize_source_id(cli_source_id)
    env = normalize_source_id(env_source_id)
    config = normalize_source_id(config_source_id)
    if cli:
        requested = cli
        source = "cli"
    elif env:
        requested = env
        source = "env"
    elif config:
        requested = config
        source = "config"
    else:
        requested = None
        source = "none"
    return SourceIdPreference(
        cli_source_id=cli,
        env_source_id=env,
        config_source_id=config,
        requested_source_id=requested,
        source=source,
    )


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
        rate = candidate.get("nominal_srate")
        if rate is not None:
            parts.append(f"rate={rate}")
        if source_id:
            parts.append(f"source_id={source_id}")
        if uid:
            parts.append(f"uid={uid}")
        labels = candidate.get("labels") or candidate.get("channel_labels")
        if labels:
            parts.append(f"labels={labels}")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def _candidate_source_id(candidate: Dict[str, Any]) -> Optional[str]:
    return normalize_source_id(candidate.get("source_id"))


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


def select_stream_by_source_id(
    candidates: Sequence[Dict[str, Any]],
    *,
    requested_source_id: Any,
    require_unique_when_unspecified: bool = True,
) -> SourceIdSelection:
    candidate_list = list(candidates)
    if not candidate_list:
        raise NoStreamFoundError("No LSL stream candidates found.")

    requested = normalize_source_id(requested_source_id)
    if requested:
        exact = [
            candidate
            for candidate in candidate_list
            if _candidate_source_id(candidate) == requested
        ]
        if len(exact) == 1:
            selected = exact[0]
            return SourceIdSelection(
                selected=selected,
                requested_source_id=requested,
                selected_source_id=_candidate_source_id(selected),
                recovery_used=False,
            )
        if len(exact) > 1:
            raise MultipleStreamsMatchedError(
                f"Multiple LSL streams matched requested source_id={requested}.\n"
                f"Matches:\n{_format_candidates(exact)}"
            )
        if len(candidate_list) == 1:
            selected = candidate_list[0]
            return SourceIdSelection(
                selected=selected,
                requested_source_id=requested,
                selected_source_id=_candidate_source_id(selected),
                recovery_used=True,
            )
        raise NoStreamMatchedError(
            f"Requested LSL source_id={requested} was not found among "
            f"{len(candidate_list)} candidate stream(s). "
            "Multiple candidates present; refusing ambiguous recovery.\n"
            f"Candidates:\n{_format_candidates(candidate_list)}"
        )

    if require_unique_when_unspecified and len(candidate_list) > 1:
        raise MultipleStreamsMatchedError(
            "Multiple LSL streams matched the selector. "
            "Set LSL_STREAM_NAME, LSL_STREAM_TYPE, or --lsl-source-id to disambiguate.\n"
            f"Matches:\n{_format_candidates(candidate_list)}"
        )

    selected = candidate_list[0]
    return SourceIdSelection(
        selected=selected,
        requested_source_id=None,
        selected_source_id=_candidate_source_id(selected),
        recovery_used=False,
    )


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
