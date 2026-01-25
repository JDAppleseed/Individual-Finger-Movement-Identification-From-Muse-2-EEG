from __future__ import annotations

import json
import statistics
import time
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional

from muse_streaming.config import (
    DEFAULT_LABELS,
    DEFAULT_NOMINAL_SRATE,
    DEFAULT_STREAM_NAME,
    DEFAULT_STREAM_TYPE,
    StreamSettings,
)
from muse_streaming.timebase import check_timebase_invariants

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
        }


def _extract_channel_labels(info: StreamInfo) -> List[str]:
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


def _match_labels(found: Iterable[str], required: Iterable[str]) -> bool:
    found_norm = {label.strip().lower() for label in found if label}
    required_norm = {label.strip().lower() for label in required if label}
    return required_norm.issubset(found_norm)


def _resolve_streams_with_timeout(timeout_s: float) -> List[StreamInfo]:
    if resolve_streams is None:
        return []
    try:
        return list(resolve_streams(timeout=timeout_s))
    except TypeError:
        return list(resolve_streams())


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

    resolve_timeout_s = max(0.1, float(timeout_s))
    streams = _resolve_streams_with_timeout(resolve_timeout_s)
    match: Optional[StreamInfo] = None
    for candidate in streams:
        if stream.name and candidate.name() != stream.name:
            continue
        if stream.stype and candidate.type() != stream.stype:
            continue
        match = candidate
        break

    expected_sps = float(
        stream.nominal_srate
        if stream.nominal_srate
        else DEFAULT_NOMINAL_SRATE
    )
    expected_name = stream.name or "auto"
    expected_type = stream.stype or DEFAULT_STREAM_TYPE

    if match is None:
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
        )

    channel_count = int(match.channel_count())
    labels = _extract_channel_labels(match)
    nominal_srate = float(match.nominal_srate() or 0.0)
    expected_sps = float(stream.nominal_srate or nominal_srate or DEFAULT_NOMINAL_SRATE)

    if require_exact_channels and channel_count != len(list(stream.labels)):
        summary = (
            f"Stream '{match.name()}' ({match.type()}) channel count "
            f"{channel_count} != expected {len(list(stream.labels))}. "
            f"Expected labels: {list(stream.labels)}."
        )
        return HealthcheckResult(
            ok=False,
            reason="channel_count_mismatch",
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
        )

    if not _match_labels(labels, stream.labels):
        summary = (
            f"Stream '{match.name()}' ({match.type()}) label mismatch. "
            f"Expected subset: {list(stream.labels)}; found: {labels}."
        )
        return HealthcheckResult(
            ok=False,
            reason="label_mismatch",
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
            )

    inlet = StreamInlet(match)
    start = time.monotonic()
    samples = 0
    target_samples = max(1, int(min_sample_window_s * expected_sps))
    timestamps: List[float] = []
    arrival_times: List[float] = []
    latency_samples: List[float] = []
    while time.monotonic() - start < timeout_s:
        sample, lsl_ts = inlet.pull_sample(timeout=0.2)
        if sample is None:
            continue
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

    ok = samples >= max(1, target_samples)
    reason = "ok" if ok else "no_samples"
    if ok and measured_sps is not None:
        if abs(measured_sps - expected_sps) > nominal_srate_tolerance:
            ok = False
            reason = "measured_sps_mismatch"

    timebase_ok = True
    timebase_warnings: List[str] = []
    if check_timebase and len(timestamps) >= 2:
        check = check_timebase_invariants(timestamps, max_gap_s=1.0)
        timebase_ok = check.ok
        timebase_warnings = check.warnings

    if reason == "no_samples":
        summary = (
            f"Stream '{match.name()}' ({match.type()}) resolved, but no samples "
            f"arrived within {timeout_s:.1f}s. Ensure the Muse is streaming."
        )
    elif reason == "measured_sps_mismatch" and measured_sps is not None:
        summary = (
            f"Stream '{match.name()}' ({match.type()}) rate mismatch: measured "
            f"{measured_sps:.2f} Hz vs expected {expected_sps:.2f} "
            f"(tol ±{nominal_srate_tolerance:.2f})."
        )
    else:
        window_text = f" over {window_s:.2f}s" if window_s is not None else ""
        measured_text = f"{measured_sps:.2f}" if measured_sps is not None else "n/a"
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
