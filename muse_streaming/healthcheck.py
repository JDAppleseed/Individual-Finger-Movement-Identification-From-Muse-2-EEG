from __future__ import annotations

import json
import statistics
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from muse_streaming.config import (
    DEFAULT_LABELS,
    DEFAULT_NOMINAL_SRATE,
    DEFAULT_STREAM_NAME,
    DEFAULT_STREAM_TYPE,
    StreamSettings,
)
from muse_streaming.timebase import check_timebase_invariants
from utils.channel_labels import (
    describe_lsl_channel_labels,
    parse_channel_label_list,
)
from utils.lsl_stream_select import (
    MultipleStreamsMatchedError,
    NoStreamFoundError,
    NoStreamMatchedError,
    normalize_source_id,
    select_stream_by_source_id,
    stream_signature,
)

try:
    from pylsl import StreamInfo, StreamInlet, local_clock, resolve_streams

    LSL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StreamInfo = None
    StreamInlet = None
    local_clock = None
    resolve_streams = None
    LSL_AVAILABLE = False


@dataclass
class HealthcheckResult:
    ok: bool
    reason: str
    summary: str
    name: str
    stype: str
    channel_count: int
    labels: List[str]
    samples_received: int
    measured_sps: Optional[float]
    expected_sps: float
    jitter_s: Optional[float]
    latency_s: Optional[float]
    nominal_srate: float
    timebase_ok: bool
    timebase_warnings: List[str]
    source_id: Optional[str] = None
    requested_source_id: Optional[str] = None
    raw_metadata_labels: List[str] = field(default_factory=list)
    normalized_metadata_labels: List[str] = field(default_factory=list)
    label_metadata_present: bool = False
    stream_uid: Optional[str] = None
    matching_candidate_count: int = 0
    matching_candidates: List[Dict[str, object]] = field(default_factory=list)
    inlet_created: bool = False
    inlet_opened: bool = False
    pull_attempts: int = 0
    first_sample_timestamp: Optional[float] = None
    first_sample_length: Optional[int] = None
    sample_validation_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "summary": self.summary,
            "name": self.name,
            "type": self.stype,
            "channel_count": self.channel_count,
            "labels": self.labels,
            "samples_received": self.samples_received,
            "measured_sps": self.measured_sps,
            "expected_sps": self.expected_sps,
            "jitter_s": self.jitter_s,
            "latency_s": self.latency_s,
            "nominal_srate": self.nominal_srate,
            "timebase_ok": self.timebase_ok,
            "timebase_warnings": self.timebase_warnings,
            "source_id": self.source_id,
            "requested_source_id": self.requested_source_id,
            "raw_metadata_labels": list(self.raw_metadata_labels),
            "normalized_metadata_labels": list(self.normalized_metadata_labels),
            "label_metadata_present": bool(self.label_metadata_present),
            "stream_uid": self.stream_uid,
            "matching_candidate_count": int(self.matching_candidate_count),
            "matching_candidates": list(self.matching_candidates),
            "inlet_created": bool(self.inlet_created),
            "inlet_opened": bool(self.inlet_opened),
            "pull_attempts": int(self.pull_attempts),
            "first_sample_timestamp": self.first_sample_timestamp,
            "first_sample_length": self.first_sample_length,
            "sample_validation_error": self.sample_validation_error,
        }


def _match_labels(found: Iterable[str], required: Iterable[str]) -> bool:
    found_norm = {label for label in parse_channel_label_list(list(found), dedupe=False)}
    required_norm = {label for label in parse_channel_label_list(list(required), dedupe=False)}
    return required_norm.issubset(found_norm)


def _resolve_streams_with_timeout(timeout_s: float) -> List[StreamInfo]:
    if resolve_streams is None:
        return []
    try:
        return list(resolve_streams(timeout=timeout_s))
    except TypeError:
        return list(resolve_streams())


def _source_id_of(info: StreamInfo) -> Optional[str]:
    getter = getattr(info, "source_id", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except Exception:
        return None
    return normalize_source_id(value)


def _uid_of(info: StreamInfo) -> Optional[str]:
    getter = getattr(info, "uid", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except Exception:
        return None
    text = str(value).strip() if value is not None else ""
    return text or None


def _candidate_signatures(candidates: Iterable[StreamInfo]) -> List[Dict[str, object]]:
    return [stream_signature(candidate) for candidate in candidates]


def _hydrate_stream_info(inlet: StreamInlet, fallback_info: StreamInfo) -> StreamInfo:
    info_getter = getattr(inlet, "info", None)
    if not callable(info_getter):
        return fallback_info
    try:
        return info_getter(timeout=0.5)
    except TypeError:
        try:
            return info_getter()
        except Exception:
            return fallback_info
    except Exception:
        return fallback_info


def _open_inlet_stream(inlet: StreamInlet, timeout_s: float) -> None:
    opener = getattr(inlet, "open_stream", None)
    if not callable(opener):
        return
    try:
        opener(timeout=timeout_s)
        return
    except TypeError:
        pass
    try:
        opener(timeout_s)
        return
    except TypeError:
        pass
    opener()


def run_healthcheck(
    *,
    stream_name: Optional[str] = None,
    name: Optional[str] = None,
    stype: Optional[str] = None,
    required_labels: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[str]] = None,
    nominal_srate: Optional[float] = None,
    require_exact_channels: bool = True,
    min_sample_window_s: float = 1.0,
    timeout_s: float = 3.0,
    check_timebase: bool = True,
    nominal_srate_tolerance: float = 1.0,
    source_id: Optional[str] = None,
    stream: Optional[StreamSettings] = None,
    **kwargs,
) -> HealthcheckResult:
    if name is not None:
        if stream_name is not None and name != stream_name:
            raise TypeError(
                "run_healthcheck() got both stream_name and legacy name with different values."
            )
        warnings.warn(
            "`name` is deprecated; use stream_name or stream=StreamSettings(...)",
            DeprecationWarning,
            stacklevel=2,
        )
        if stream_name is None:
            stream_name = name

    if "stream_type" in kwargs:
        if stype is None:
            stype = kwargs.pop("stream_type")
        else:
            kwargs.pop("stream_type")
    if "type" in kwargs:
        if stype is None:
            stype = kwargs.pop("type")
        else:
            kwargs.pop("type")
    if "lsl_source_id" in kwargs:
        if source_id is None:
            source_id = kwargs.pop("lsl_source_id")
        else:
            kwargs.pop("lsl_source_id")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"run_healthcheck() got unexpected keyword argument(s): {unexpected}")

    if stream is not None:
        legacy_args = (stream_name, name, stype, required_labels, labels, nominal_srate)
        if any(value is not None for value in legacy_args):
            raise TypeError(
                "run_healthcheck() got both stream and legacy stream parameters."
            )
    else:
        if (
            stream_name is None
            and name is None
            and stype is None
            and required_labels is None
            and labels is None
            and nominal_srate is None
        ):
            stream = StreamSettings()
        else:
            label_source = required_labels if required_labels is not None else labels
            stream = StreamSettings(
                name=stream_name,
                stype=stype or DEFAULT_STREAM_TYPE,
                nominal_srate=(
                    float(nominal_srate)
                    if nominal_srate is not None
                    else DEFAULT_NOMINAL_SRATE
                ),
                labels=list(label_source) if label_source is not None else list(DEFAULT_LABELS),
            )

    if not LSL_AVAILABLE or resolve_streams is None:
        raise RuntimeError("pylsl is required for health checks.")

    required_labels = parse_channel_label_list(stream.labels, dedupe=False)
    requested_source_id = normalize_source_id(source_id)
    resolve_timeout_s = max(0.1, float(timeout_s))
    streams = _resolve_streams_with_timeout(resolve_timeout_s)
    match: Optional[StreamInfo] = None
    matching_candidates = []
    for candidate in streams:
        if stream.name and candidate.name() != stream.name:
            continue
        if stream.stype and candidate.type() != stream.stype:
            continue
        matching_candidates.append(candidate)

    expected_sps = float(
        stream.nominal_srate
        if stream.nominal_srate
        else DEFAULT_NOMINAL_SRATE
    )
    expected_name = stream.name or "auto"
    expected_type = stream.stype or DEFAULT_STREAM_TYPE
    candidate_signatures = _candidate_signatures(matching_candidates)

    if not matching_candidates:
        summary = (
            f"Stream '{expected_name}' (type {expected_type}) not found within "
            f"{resolve_timeout_s:.1f}s. Start the streamer and confirm the LSL name/type."
        )
        return HealthcheckResult(
            ok=False,
            reason="stream_not_found",
            summary=summary,
            name=stream.name or "",
            stype=expected_type,
            channel_count=0,
            labels=[],
            samples_received=0,
            measured_sps=None,
            expected_sps=expected_sps,
            jitter_s=None,
            latency_s=None,
            nominal_srate=0.0,
            timebase_ok=False,
            timebase_warnings=[],
            source_id=None,
            requested_source_id=requested_source_id,
            matching_candidate_count=0,
        )

    if requested_source_id:
        try:
            selection = select_stream_by_source_id(
                [dict(stream_signature(candidate), _stream=candidate) for candidate in matching_candidates],
                requested_source_id=requested_source_id,
                require_unique_when_unspecified=True,
            )
            if bool(selection.recovery_used):
                raise NoStreamMatchedError(
                    f"Requested source_id={requested_source_id} was not found exactly; "
                    f"single-candidate recovery selected source_id={selection.selected_source_id}."
                )
            match = selection.selected.get("_stream")
        except (NoStreamFoundError, NoStreamMatchedError, MultipleStreamsMatchedError) as exc:
            match = None
            discovered_source_ids = sorted(
                {
                    source
                    for source in (_source_id_of(candidate) for candidate in matching_candidates)
                    if source
                }
            )
            summary = (
                f"Stream '{expected_name}' ({expected_type}) was found, but the requested "
                f"source_id={requested_source_id} did not match the live candidate set. "
                f"Discovered source_ids={discovered_source_ids or []}. {exc}"
            )
            return HealthcheckResult(
                ok=False,
                reason="stream_found_source_id_mismatch",
                summary=summary,
                name=expected_name if expected_name != "auto" else "",
                stype=expected_type,
                channel_count=0,
                labels=[],
                samples_received=0,
                measured_sps=None,
                expected_sps=expected_sps,
                jitter_s=None,
                latency_s=None,
                nominal_srate=0.0,
                timebase_ok=False,
                timebase_warnings=[],
                source_id=None,
                requested_source_id=requested_source_id,
                matching_candidate_count=len(matching_candidates),
                matching_candidates=candidate_signatures,
            )
    else:
        if len(matching_candidates) > 1:
            summary = (
                f"Multiple live streams matched name='{expected_name}' type='{expected_type}'. "
                "Set lsl_source_id to bind Step 7 to a single stream. "
                f"Matches={candidate_signatures}."
            )
            return HealthcheckResult(
                ok=False,
                reason="stream_selection_ambiguous",
                summary=summary,
                name=expected_name if expected_name != "auto" else "",
                stype=expected_type,
                channel_count=0,
                labels=[],
                samples_received=0,
                measured_sps=None,
                expected_sps=expected_sps,
                jitter_s=None,
                latency_s=None,
                nominal_srate=0.0,
                timebase_ok=False,
                timebase_warnings=[],
                source_id=None,
                requested_source_id=None,
                matching_candidate_count=len(matching_candidates),
                matching_candidates=candidate_signatures,
            )
        match = matching_candidates[0]

    inlet = None
    try:
        inlet = StreamInlet(match)
    except Exception as exc:
        summary = (
            f"Stream '{match.name()}' ({match.type()}) resolved, but StreamInlet creation failed: "
            f"{exc!r}. Selected stream={stream_signature(match)}."
        )
        return HealthcheckResult(
            ok=False,
            reason="stream_resolved_but_inlet_open_failed",
            summary=summary,
            name=match.name(),
            stype=match.type(),
            channel_count=int(match.channel_count() or 0),
            labels=[],
            samples_received=0,
            measured_sps=None,
            expected_sps=expected_sps,
            jitter_s=None,
            latency_s=None,
            nominal_srate=float(match.nominal_srate() or 0.0),
            timebase_ok=False,
            timebase_warnings=[],
            source_id=_source_id_of(match),
            requested_source_id=requested_source_id,
            stream_uid=_uid_of(match),
            matching_candidate_count=len(matching_candidates),
            matching_candidates=candidate_signatures,
            inlet_created=False,
            inlet_opened=False,
            sample_validation_error=repr(exc),
        )

    inlet_opened = False
    try:
        _open_inlet_stream(inlet, timeout_s=max(0.5, min(float(timeout_s), 2.0)))
        inlet_opened = True
    except Exception as exc:
        summary = (
            f"Stream '{match.name()}' ({match.type()}) resolved, but inlet open failed: "
            f"{exc!r}. Selected stream={stream_signature(match)}."
        )
        return HealthcheckResult(
            ok=False,
            reason="stream_resolved_but_inlet_open_failed",
            summary=summary,
            name=match.name(),
            stype=match.type(),
            channel_count=int(match.channel_count() or 0),
            labels=[],
            samples_received=0,
            measured_sps=None,
            expected_sps=expected_sps,
            jitter_s=None,
            latency_s=None,
            nominal_srate=float(match.nominal_srate() or 0.0),
            timebase_ok=False,
            timebase_warnings=[],
            source_id=_source_id_of(match),
            requested_source_id=requested_source_id,
            stream_uid=_uid_of(match),
            matching_candidate_count=len(matching_candidates),
            matching_candidates=candidate_signatures,
            inlet_created=True,
            inlet_opened=False,
            sample_validation_error=repr(exc),
        )

    resolved_info = _hydrate_stream_info(inlet, match)
    channel_count = int(resolved_info.channel_count())
    label_report = describe_lsl_channel_labels(resolved_info)
    labels = list(label_report["normalized_labels"])
    raw_metadata_labels = list(label_report["raw_labels"])
    label_metadata_present = bool(label_report["metadata_present"])
    nominal_srate = float(resolved_info.nominal_srate() or 0.0)
    expected_sps = float(stream.nominal_srate or nominal_srate or DEFAULT_NOMINAL_SRATE)
    resolved_source_id = _source_id_of(resolved_info)
    selected_uid = _uid_of(resolved_info) or _uid_of(match)

    if requested_source_id and resolved_source_id and resolved_source_id != requested_source_id:
        summary = (
            f"Stream '{match.name()}' ({match.type()}) metadata resolved to source_id="
            f"{resolved_source_id}, but requested source_id={requested_source_id}. "
            f"Selected stream={stream_signature(match)}."
        )
        return HealthcheckResult(
            ok=False,
            reason="source_id_resolved_mismatch",
            summary=summary,
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
            measured_sps=None,
            expected_sps=expected_sps,
            jitter_s=None,
            latency_s=None,
            nominal_srate=nominal_srate,
            timebase_ok=False,
            timebase_warnings=[],
            source_id=resolved_source_id,
            requested_source_id=requested_source_id,
            raw_metadata_labels=raw_metadata_labels,
            normalized_metadata_labels=labels,
            label_metadata_present=label_metadata_present,
            stream_uid=selected_uid,
            matching_candidate_count=len(matching_candidates),
            matching_candidates=candidate_signatures,
            inlet_created=True,
            inlet_opened=inlet_opened,
        )

    if require_exact_channels and channel_count != len(required_labels):
        summary = (
            f"Stream '{match.name()}' ({match.type()}) channel count "
            f"{channel_count} != expected {len(required_labels)}. "
            f"Expected labels: {required_labels}."
        )
        return HealthcheckResult(
            ok=False,
            reason="stream_found_channel_count_mismatch",
            summary=summary,
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
            measured_sps=None,
            expected_sps=expected_sps,
            jitter_s=None,
            latency_s=None,
            nominal_srate=float(match.nominal_srate() or 0.0),
            timebase_ok=False,
            timebase_warnings=[],
            source_id=resolved_source_id,
            requested_source_id=requested_source_id,
            raw_metadata_labels=raw_metadata_labels,
            normalized_metadata_labels=labels,
            label_metadata_present=label_metadata_present,
            stream_uid=selected_uid,
            matching_candidate_count=len(matching_candidates),
            matching_candidates=candidate_signatures,
            inlet_created=True,
            inlet_opened=inlet_opened,
        )

    if not label_metadata_present:
        summary = (
            f"Stream '{match.name()}' ({match.type()}) is present, but channel labels are "
            "absent from the published LSL metadata. "
            f"Expected labels: {required_labels}."
        )
        return HealthcheckResult(
            ok=False,
            reason="stream_found_labels_missing",
            summary=summary,
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
            measured_sps=None,
            expected_sps=expected_sps,
            jitter_s=None,
            latency_s=None,
            nominal_srate=float(match.nominal_srate() or 0.0),
            timebase_ok=False,
            timebase_warnings=[],
            source_id=resolved_source_id,
            requested_source_id=requested_source_id,
            raw_metadata_labels=raw_metadata_labels,
            normalized_metadata_labels=labels,
            label_metadata_present=label_metadata_present,
            stream_uid=selected_uid,
            matching_candidate_count=len(matching_candidates),
            matching_candidates=candidate_signatures,
            inlet_created=True,
            inlet_opened=inlet_opened,
        )

    if not _match_labels(labels, required_labels):
        summary = (
            f"Stream '{match.name()}' ({match.type()}) label mismatch. "
            f"Expected subset: {required_labels}; "
            f"found normalized labels: {labels}; raw metadata labels: {raw_metadata_labels}."
        )
        return HealthcheckResult(
            ok=False,
            reason="stream_found_label_mismatch",
            summary=summary,
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
            measured_sps=None,
            expected_sps=expected_sps,
            jitter_s=None,
            latency_s=None,
            nominal_srate=float(match.nominal_srate() or 0.0),
            timebase_ok=False,
            timebase_warnings=[],
            source_id=resolved_source_id,
            requested_source_id=requested_source_id,
            raw_metadata_labels=raw_metadata_labels,
            normalized_metadata_labels=labels,
            label_metadata_present=label_metadata_present,
            stream_uid=selected_uid,
            matching_candidate_count=len(matching_candidates),
            matching_candidates=candidate_signatures,
            inlet_created=True,
            inlet_opened=inlet_opened,
        )

    if expected_sps and nominal_srate:
        if abs(nominal_srate - float(expected_sps)) > nominal_srate_tolerance:
            summary = (
                f"Stream '{match.name()}' ({match.type()}) nominal_srate={nominal_srate:.2f} "
                f"differs from expected {expected_sps:.2f} (tol ±{nominal_srate_tolerance:.2f}). "
                "Verify the streamer rate."
            )
            return HealthcheckResult(
                ok=False,
                reason="nominal_srate_mismatch",
                summary=summary,
                name=match.name(),
                stype=match.type(),
                channel_count=channel_count,
                labels=labels,
                samples_received=0,
                measured_sps=None,
                expected_sps=expected_sps,
                jitter_s=None,
                latency_s=None,
                nominal_srate=nominal_srate,
                timebase_ok=False,
                timebase_warnings=[],
                source_id=resolved_source_id,
                requested_source_id=requested_source_id,
                raw_metadata_labels=raw_metadata_labels,
                normalized_metadata_labels=labels,
                label_metadata_present=label_metadata_present,
                stream_uid=selected_uid,
                matching_candidate_count=len(matching_candidates),
                matching_candidates=candidate_signatures,
                inlet_created=True,
                inlet_opened=inlet_opened,
            )

    start = time.monotonic()
    samples = 0
    target_samples = max(1, int(min_sample_window_s * expected_sps))
    timestamps: List[float] = []
    arrival_times: List[float] = []
    latency_samples: List[float] = []
    pull_attempts = 0
    first_sample_timestamp: Optional[float] = None
    first_sample_length: Optional[int] = None
    sample_validation_error: Optional[str] = None
    pull_contract_error = False
    while time.monotonic() - start < timeout_s:
        pull_attempts += 1
        try:
            sample, lsl_ts = inlet.pull_sample(timeout=0.2)
        except Exception as exc:
            sample_validation_error = repr(exc)
            pull_contract_error = True
            break
        if sample is None:
            continue
        if first_sample_timestamp is None and lsl_ts is not None:
            first_sample_timestamp = float(lsl_ts)
        if first_sample_length is None:
            try:
                first_sample_length = len(sample)
            except Exception:
                first_sample_length = None
        try:
            sample_length = len(sample)
        except Exception:
            sample_length = None
        if sample_length != channel_count:
            sample_validation_error = (
                f"expected {channel_count} channels but received {sample_length}"
            )
            break
        samples += 1
        arrival_times.append(time.monotonic())
        if lsl_ts is not None:
            ts = float(lsl_ts)
            timestamps.append(ts)
            if local_clock is not None:
                latency_samples.append(float(local_clock() - ts))
        if samples >= target_samples:
            break

    measured_sps: Optional[float] = None
    window_s: Optional[float] = None
    if len(timestamps) >= 2:
        window_s = float(timestamps[-1] - timestamps[0])
        if window_s > 0:
            measured_sps = float((len(timestamps) - 1) / window_s)
    elif len(arrival_times) >= 2:
        window_s = float(arrival_times[-1] - arrival_times[0])
        if window_s > 0:
            measured_sps = float((len(arrival_times) - 1) / window_s)

    jitter_s: Optional[float] = None
    if len(timestamps) >= 3:
        intervals = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
        if len(intervals) >= 2:
            jitter_s = float(statistics.pstdev(intervals))
    elif len(arrival_times) >= 3:
        intervals = [
            arrival_times[i] - arrival_times[i - 1] for i in range(1, len(arrival_times))
        ]
        if len(intervals) >= 2:
            jitter_s = float(statistics.pstdev(intervals))

    latency_s: Optional[float] = None
    if latency_samples:
        latency_s = float(sum(latency_samples) / len(latency_samples))

    ok = samples > 0 and sample_validation_error is None
    reason = "ok" if ok else "stream_resolved_but_no_samples_pulled"
    if sample_validation_error is not None:
        if pull_contract_error:
            reason = "healthcheck_stream_pull_contract_violation"
        else:
            reason = "healthcheck_sample_shape_invalid"
        ok = False
    elif ok and measured_sps is not None:
        if abs(measured_sps - expected_sps) > nominal_srate_tolerance:
            ok = False
            reason = "measured_sps_mismatch"

    timebase_ok = True
    timebase_warnings: List[str] = []
    if check_timebase and len(timestamps) >= 2:
        check = check_timebase_invariants(timestamps, max_gap_s=1.0)
        timebase_ok = check.ok
        timebase_warnings = check.warnings

    if reason == "stream_resolved_but_no_samples_pulled":
        summary = (
            f"Stream '{match.name()}' ({match.type()}) resolved and inlet opened, "
            f"but no samples were pulled within {timeout_s:.1f}s after {pull_attempts} "
            f"attempts. Selected stream={stream_signature(match)}."
        )
    elif reason == "healthcheck_sample_shape_invalid":
        summary = (
            f"Stream '{match.name()}' ({match.type()}) produced a malformed sample: "
            f"{sample_validation_error}. Expected channel_count={channel_count}, "
            f"first_sample_length={first_sample_length!r}, first_sample_timestamp={first_sample_timestamp!r}."
        )
    elif reason == "healthcheck_stream_pull_contract_violation":
        summary = (
            f"Stream '{match.name()}' ({match.type()}) resolved, but sample pull failed: "
            f"{sample_validation_error}. Selected stream={stream_signature(match)}."
        )
    elif reason == "measured_sps_mismatch" and measured_sps is not None:
        summary = (
            f"Stream '{match.name()}' ({match.type()}) rate mismatch: measured "
            f"{measured_sps:.2f} Hz vs expected {expected_sps:.2f} "
            f"(tol ±{nominal_srate_tolerance:.2f})."
        )
    else:
        window_text = f" over {window_s:.2f}s" if window_s is not None else ""
        measured_text = f"{measured_sps:.2f}" if measured_sps is not None else "sample-flow"
        summary = (
            f"Stream '{match.name()}' ({match.type()}) healthy: measured "
            f"{measured_text} Hz (expected {expected_sps:.2f} Hz){window_text}, "
            f"samples={samples}"
        )
        if jitter_s is not None:
            summary += f", jitter {jitter_s * 1000.0:.2f} ms"
        if latency_s is not None:
            summary += f", latency {latency_s * 1000.0:.1f} ms"
        if check_timebase and not timebase_ok and timebase_warnings:
            summary += "; timebase warnings: " + ", ".join(timebase_warnings)

    return HealthcheckResult(
        ok=ok,
        reason=reason,
        summary=summary,
        name=match.name(),
        stype=match.type(),
        channel_count=channel_count,
        labels=labels,
        samples_received=samples,
        measured_sps=measured_sps,
        expected_sps=expected_sps,
        jitter_s=jitter_s,
        latency_s=latency_s,
        nominal_srate=nominal_srate,
        timebase_ok=timebase_ok,
        timebase_warnings=timebase_warnings,
        source_id=resolved_source_id,
        requested_source_id=requested_source_id,
        raw_metadata_labels=raw_metadata_labels,
        normalized_metadata_labels=labels,
        label_metadata_present=label_metadata_present,
        stream_uid=selected_uid,
        matching_candidate_count=len(matching_candidates),
        matching_candidates=candidate_signatures,
        inlet_created=True,
        inlet_opened=inlet_opened,
        pull_attempts=pull_attempts,
        first_sample_timestamp=first_sample_timestamp,
        first_sample_length=first_sample_length,
        sample_validation_error=sample_validation_error,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Muse 2 LSL healthcheck")
    parser.add_argument(
        "--stream-name",
        "--name",
        dest="stream_name",
        type=str,
        default=DEFAULT_STREAM_NAME,
        help="LSL stream name (deprecated alias: --name)",
    )
    parser.add_argument("--type", type=str, default=DEFAULT_STREAM_TYPE)
    parser.add_argument("--labels", type=str, default=",".join(DEFAULT_LABELS))
    parser.add_argument("--exact", action="store_true", help="Require exact channel count")
    parser.add_argument("--check-timebase", action="store_true", help="Validate timestamps")
    parser.add_argument("--srate-tol", type=float, default=1.0, help="Nominal srate tolerance")
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    stream = StreamSettings(name=args.stream_name, stype=args.type, labels=labels)
    result = run_healthcheck(
        stream=stream,
        require_exact_channels=args.exact,
        check_timebase=args.check_timebase,
        nominal_srate_tolerance=args.srate_tol,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
