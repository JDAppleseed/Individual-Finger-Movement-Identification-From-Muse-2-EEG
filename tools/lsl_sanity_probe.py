from __future__ import annotations

import argparse
import statistics
import time
from typing import Optional

from pylsl import StreamInlet, StreamInfo

from utils.lsl_stream_select import (
    StreamSelector,
    log_stream_signature,
    pick_stream,
    stream_signature,
)
from utils.stream_timebase import clamp_lsl_timestamp, gap_threshold_s, is_gap


def _drain_inlet(inlet: StreamInlet, drain_s: float = 0.75) -> int:
    drained = 0
    start = time.monotonic()
    while time.monotonic() - start < drain_s:
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            time.sleep(0.005)
            continue
        drained += 1
    return drained


def _default_stream_selector(
    name_contains: Optional[str], type_equals: Optional[str], min_channels: int
) -> StreamInfo:
    if name_contains:
        return pick_stream(
            StreamSelector(
                name_contains=name_contains,
                type_equals=None,
                min_channels=min_channels,
            )
        )
    if type_equals:
        return pick_stream(
            StreamSelector(
                name_contains=None,
                type_equals=type_equals,
                min_channels=min_channels,
            )
        )
    try:
        return pick_stream(
            StreamSelector(
                name_contains=None, type_equals="EEG", min_channels=min_channels
            )
        )
    except Exception as exc:
        message = str(exc)
        if "No LSL streams matched" not in message:
            raise
    return pick_stream(
        StreamSelector(name_contains="eeg", type_equals=None, min_channels=min_channels)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="LSL sanity probe")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--name-contains", type=str, default=None)
    parser.add_argument("--type-equals", type=str, default=None)
    parser.add_argument("--min-channels", type=int, default=4)
    parser.add_argument("--epsilon-s", type=float, default=0.010)
    parser.add_argument("--hard-backwards-s", type=float, default=0.200)
    args = parser.parse_args()

    stream = _default_stream_selector(
        args.name_contains, args.type_equals, args.min_channels
    )
    signature = stream_signature(stream)
    log_stream_signature(signature)

    inlet = StreamInlet(stream, max_buflen=5)
    drained = _drain_inlet(inlet, drain_s=0.75)
    if drained:
        print(f"🧹 Drained {drained} stale LSL samples before probe.")

    nominal_srate = signature.get("nominal_srate") or 0.0
    if nominal_srate and nominal_srate > 0:
        nominal_dt = 1.0 / float(nominal_srate)
    else:
        nominal_dt = 1.0 / 256.0
    gap_threshold = gap_threshold_s(nominal_dt)

    samples = 0
    soft_backwards = 0
    hard_backwards = 0
    max_backwards_delta = 0.0
    gap_count = 0
    max_gap = 0.0
    dt_samples = []
    prev_mono = None

    start = time.monotonic()
    while time.monotonic() - start < args.seconds:
        sample, ts = inlet.pull_sample(timeout=0.1)
        if sample is None:
            continue
        samples += 1
        result = clamp_lsl_timestamp(
            prev_mono,
            float(ts),
            epsilon_s=args.epsilon_s,
            hard_backwards_s=args.hard_backwards_s,
        )
        if result.clamped:
            if result.is_hard_backwards:
                hard_backwards += 1
            else:
                soft_backwards += 1
            max_backwards_delta = max(max_backwards_delta, result.backwards_delta_s)
        elif prev_mono is not None:
            dt_s = float(result.mono_ts - prev_mono)
            if dt_s > 0:
                dt_samples.append(dt_s)
            if is_gap(dt_s, nominal_dt):
                gap_count += 1
                max_gap = max(max_gap, dt_s)
        prev_mono = result.mono_ts

    measured_fs = None
    if dt_samples:
        median_dt = statistics.median(dt_samples)
        if median_dt > 0:
            measured_fs = 1.0 / median_dt

    print("\nLSL sanity probe summary")
    print("-" * 30)
    print(f"Samples             : {samples}")
    if measured_fs is not None:
        print(f"Measured Fs         : {measured_fs:.2f} Hz (median dt)")
    else:
        print("Measured Fs         : n/a")
    print(f"Nominal gap threshold: {gap_threshold:.4f} s")
    print(f"Backwards soft count: {soft_backwards}")
    print(f"Backwards hard count: {hard_backwards}")
    print(f"Max backwards delta : {max_backwards_delta:.6f} s")
    print(f"Gap count           : {gap_count}")
    print(f"Max gap             : {max_gap:.6f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
