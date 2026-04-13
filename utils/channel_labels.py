from __future__ import annotations

from typing import Any, Iterable, Sequence
from xml.etree import ElementTree

_QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "`": "`",
    "“": "”",
    "‘": "’",
    "«": "»",
}


def strip_outer_label_quotes(value: Any) -> str:
    text = str(value or "").strip()
    while len(text) >= 2:
        closing = _QUOTE_PAIRS.get(text[0])
        if closing is None or text[-1] != closing:
            break
        text = text[1:-1].strip()
    return text


def normalize_channel_label(value: Any) -> str:
    text = strip_outer_label_quotes(value)
    return text.upper() if text else ""


def normalize_channel_labels(
    values: Iterable[Any],
    *,
    dedupe: bool = False,
) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = normalize_channel_label(value)
        if not label:
            continue
        if dedupe:
            if label in seen:
                continue
            seen.add(label)
        labels.append(label)
    return labels


def parse_channel_label_list(
    value: Any,
    *,
    dedupe: bool = False,
) -> list[str]:
    if value is None:
        return []
    parts: Sequence[Any]
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            cleaned = cleaned[1:-1]
        parts = [part.strip() for part in cleaned.split(",")]
    elif isinstance(value, (list, tuple, set)):
        parts = list(value)
    else:
        parts = [value]
    return normalize_channel_labels(parts, dedupe=dedupe)


def append_lsl_channel_metadata(
    desc: Any,
    labels: Sequence[Any],
    *,
    unit: str = "microvolts",
    channel_type: str = "EEG",
) -> list[str]:
    canonical_labels = normalize_channel_labels(labels, dedupe=False)
    channels = desc.append_child("channels")
    for label in canonical_labels:
        channel = channels.append_child("channel")
        channel.append_child_value("label", label)
        channel.append_child_value("unit", unit)
        channel.append_child_value("type", channel_type)
    return canonical_labels


def _safe_channel_count(info: Any) -> int:
    try:
        return max(0, int(info.channel_count()))
    except Exception:
        return 0


def _extract_labels_from_xml(info: Any) -> list[str]:
    xml_getter = getattr(info, "as_xml", None)
    if not callable(xml_getter):
        return []
    try:
        xml_text = str(xml_getter() or "").strip()
    except Exception:
        return []
    if not xml_text:
        return []
    try:
        root = ElementTree.fromstring(xml_text)
    except Exception:
        return []
    labels: list[str] = []
    for node in root.findall(".//channels/channel/label"):
        text = str(node.text or "").strip()
        if text:
            labels.append(text)
    return labels


def _extract_labels_from_desc(info: Any) -> list[str]:
    try:
        desc = info.desc()
        channels = desc.child("channels")
        channel = channels.child("channel")
    except Exception:
        return []
    labels: list[str] = []
    max_labels = max(_safe_channel_count(info), 1)
    for _ in range(max_labels):
        if channel is None:
            break
        try:
            text = str(channel.child_value("label") or "").strip()
        except Exception:
            text = ""
        if text:
            labels.append(text)
        next_channel = None
        next_sibling = getattr(channel, "next_sibling", None)
        if callable(next_sibling):
            try:
                next_channel = next_sibling("channel")
            except TypeError:
                try:
                    next_channel = next_sibling()
                except Exception:
                    next_channel = None
            except Exception:
                next_channel = None
        if next_channel is None or next_channel is channel:
            break
        channel = next_channel
    return labels


def extract_lsl_channel_labels(info: Any) -> list[str]:
    labels = _extract_labels_from_xml(info)
    if labels:
        return labels
    return _extract_labels_from_desc(info)


def describe_lsl_channel_labels(info: Any) -> dict[str, Any]:
    raw_labels = extract_lsl_channel_labels(info)
    normalized_labels = normalize_channel_labels(raw_labels, dedupe=False)
    return {
        "raw_labels": list(raw_labels),
        "normalized_labels": list(normalized_labels),
        "metadata_present": bool(raw_labels),
    }
