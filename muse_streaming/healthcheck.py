from __future__ import annotations

import json
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
    from pylsl import StreamInfo, StreamInlet, resolve_streams

    LSL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StreamInfo = None
    StreamInlet = None
    resolve_streams = None
    LSL_AVAILABLE = False


@dataclass
class HealthcheckResult:
    ok: bool
    reason: str
    name: str
    stype: str
    channel_count: int
    labels: List[str]
    samples_received: int
    nominal_srate: float
    timebase_ok: bool
    timebase_warnings: List[str]

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "name": self.name,
            "type": self.stype,
            "channel_count": self.channel_count,
            "labels": self.labels,
            "samples_received": self.samples_received,
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


def run_healthcheck(
    *,
    stream: Optional[StreamSettings] = None,
    stream_name: Optional[str] = None,
    name: Optional[str] = None,
    stype: Optional[str] = None,
    required_labels: Optional[Iterable[str]] = None,
    labels: Optional[Iterable[str]] = None,
    nominal_srate: Optional[float] = None,
    require_exact_channels: bool = True,
    min_sample_window_s: float = 0.5,
    timeout_s: float = 3.0,
    check_timebase: bool = True,
    nominal_srate_tolerance: float = 1.0,
    **kwargs,
) -> HealthcheckResult:
    if name is not None:
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

    streams = resolve_streams()
    match: Optional[StreamInfo] = None
    for candidate in streams:
        if stream.name and candidate.name() != stream.name:
            continue
        if stream.stype and candidate.type() != stream.stype:
            continue
        match = candidate
        break

    if match is None:
        return HealthcheckResult(
            ok=False,
            reason="stream_not_found",
            name=stream.name,
            stype=stream.stype,
            channel_count=0,
            labels=[],
            samples_received=0,
            nominal_srate=0.0,
            timebase_ok=False,
            timebase_warnings=[],
        )

    channel_count = int(match.channel_count())
    labels = _extract_channel_labels(match)
    nominal_srate = float(match.nominal_srate() or 0.0)
    if require_exact_channels and channel_count != len(list(stream.labels)):
        return HealthcheckResult(
            ok=False,
            reason="channel_count_mismatch",
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
            nominal_srate=float(match.nominal_srate() or 0.0),
            timebase_ok=False,
            timebase_warnings=[],
        )

    if not _match_labels(labels, stream.labels):
        return HealthcheckResult(
            ok=False,
            reason="label_mismatch",
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
            nominal_srate=float(match.nominal_srate() or 0.0),
            timebase_ok=False,
            timebase_warnings=[],
        )

    if stream.nominal_srate and nominal_srate:
        if abs(nominal_srate - float(stream.nominal_srate)) > nominal_srate_tolerance:
            return HealthcheckResult(
                ok=False,
                reason="nominal_srate_mismatch",
                name=match.name(),
                stype=match.type(),
                channel_count=channel_count,
                labels=labels,
                samples_received=0,
                nominal_srate=nominal_srate,
                timebase_ok=False,
                timebase_warnings=[],
            )

    inlet = StreamInlet(match)
    start = time.monotonic()
    samples = 0
    nominal_srate = float(match.nominal_srate() or 1.0)
    target_samples = int(min_sample_window_s * nominal_srate)
    timestamps: List[float] = []
    while time.monotonic() - start < timeout_s:
        sample, lsl_ts = inlet.pull_sample(timeout=0.2)
        if sample is None:
            continue
        samples += 1
        if lsl_ts is not None:
            timestamps.append(float(lsl_ts))
        if samples >= max(1, target_samples):
            break

    ok = samples >= max(1, target_samples)
    timebase_ok = True
    timebase_warnings: List[str] = []
    if check_timebase and len(timestamps) >= 2:
        check = check_timebase_invariants(timestamps, max_gap_s=1.0)
        timebase_ok = check.ok
        timebase_warnings = check.warnings
    return HealthcheckResult(
        ok=ok,
        reason="ok" if ok else "no_samples",
        name=match.name(),
        stype=match.type(),
        channel_count=channel_count,
        labels=labels,
        samples_received=samples,
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
