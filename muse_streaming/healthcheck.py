from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

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

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "reason": self.reason,
            "name": self.name,
            "type": self.stype,
            "channel_count": self.channel_count,
            "labels": self.labels,
            "samples_received": self.samples_received,
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
    name: str,
    stype: str,
    required_labels: Iterable[str],
    require_exact_channels: bool = True,
    min_sample_window_s: float = 0.5,
    timeout_s: float = 3.0,
) -> HealthcheckResult:
    if not LSL_AVAILABLE or resolve_streams is None:
        raise RuntimeError("pylsl is required for health checks.")

    streams = resolve_streams()
    match: Optional[StreamInfo] = None
    for stream in streams:
        if name and stream.name() != name:
            continue
        if stype and stream.type() != stype:
            continue
        match = stream
        break

    if match is None:
        return HealthcheckResult(
            ok=False,
            reason="stream_not_found",
            name=name,
            stype=stype,
            channel_count=0,
            labels=[],
            samples_received=0,
        )

    channel_count = int(match.channel_count())
    labels = _extract_channel_labels(match)
    if require_exact_channels and channel_count != len(list(required_labels)):
        return HealthcheckResult(
            ok=False,
            reason="channel_count_mismatch",
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
        )

    if not _match_labels(labels, required_labels):
        return HealthcheckResult(
            ok=False,
            reason="label_mismatch",
            name=match.name(),
            stype=match.type(),
            channel_count=channel_count,
            labels=labels,
            samples_received=0,
        )

    inlet = StreamInlet(match)
    start = time.monotonic()
    samples = 0
    target_samples = int(min_sample_window_s * float(match.nominal_srate() or 1))
    while time.monotonic() - start < timeout_s:
        sample, _ = inlet.pull_sample(timeout=0.2)
        if sample is None:
            continue
        samples += 1
        if samples >= max(1, target_samples):
            break

    ok = samples >= max(1, target_samples)
    return HealthcheckResult(
        ok=ok,
        reason="ok" if ok else "no_samples",
        name=match.name(),
        stype=match.type(),
        channel_count=channel_count,
        labels=labels,
        samples_received=samples,
    )


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Muse 2 LSL healthcheck")
    parser.add_argument("--name", type=str, default="Muse2-EEG")
    parser.add_argument("--type", type=str, default="EEG")
    parser.add_argument("--labels", type=str, default="TP9,AF7,AF8,TP10")
    parser.add_argument("--exact", action="store_true", help="Require exact channel count")
    args = parser.parse_args()

    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    result = run_healthcheck(
        name=args.name,
        stype=args.type,
        required_labels=labels,
        require_exact_channels=args.exact,
    )
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
