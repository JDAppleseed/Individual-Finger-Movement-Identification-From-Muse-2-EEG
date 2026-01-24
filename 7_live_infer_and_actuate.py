#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from itertools import count
from pathlib import Path
from typing import Deque, Optional, Tuple

import joblib
import numpy as np
import torch
from pylsl import StreamInlet, resolve_streams

try:
    from pylsl import resolve_byprop as _resolve_byprop
except Exception:
    _resolve_byprop = None

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from muse_streaming.packets import SamplePacket
from muse_streaming.session_writer import SessionWriter


logger = logging.getLogger("live_infer")


def standardize_window_TxC(window_TxC: np.ndarray, scaler_obj) -> np.ndarray:
    if scaler_obj is None:
        return window_TxC
    if isinstance(scaler_obj, dict):
        mean = np.asarray(scaler_obj.get("mean"), dtype=np.float32)
        std = np.asarray(scaler_obj.get("std"), dtype=np.float32)
        if mean.ndim == 0 or std.ndim == 0:
            return window_TxC
        std = np.where(std == 0, 1.0, std)
        return (window_TxC - mean) / std
    if hasattr(scaler_obj, "mean_") and hasattr(scaler_obj, "scale_"):
        mean = np.asarray(scaler_obj.mean_, dtype=np.float32)
        scale = np.asarray(scaler_obj.scale_, dtype=np.float32)
        scale = np.where(scale == 0, 1.0, scale)
        return (window_TxC - mean) / scale
    return window_TxC


def _resample_window(
    window_times: np.ndarray,
    window_values: np.ndarray,
    start_s: float,
    end_s: float,
    target_fs: float,
) -> Optional[np.ndarray]:
    if window_times.size < 2:
        return None
    window_samples = int(round((end_s - start_s) * target_fs))
    if window_samples <= 1:
        return None
    grid = np.linspace(start_s, end_s, window_samples, endpoint=False)
    window = np.zeros((window_samples, window_values.shape[1]), dtype=np.float32)
    for ch_idx in range(window_values.shape[1]):
        window[:, ch_idx] = np.interp(grid, window_times, window_values[:, ch_idx])
    return window


def _resolve_stream(
    name: Optional[str], stype: Optional[str], source_id: Optional[str]
):
    def _format_stream(candidate) -> str:
        parts = [
            f"name={candidate.name()}",
            f"type={candidate.type()}",
            f"ch={candidate.channel_count()}",
        ]
        if hasattr(candidate, "source_id"):
            try:
                value = candidate.source_id()
                if value:
                    parts.append(f"source_id={value}")
            except Exception:
                pass
        if hasattr(candidate, "uid"):
            try:
                value = candidate.uid()
                if value:
                    parts.append(f"uid={value}")
            except Exception:
                pass
        return ", ".join(parts)

    if source_id:
        if _resolve_byprop is not None:
            candidates = _resolve_byprop("source_id", source_id, timeout=2.0)
        else:
            candidates = []
            for candidate in resolve_streams():
                if not hasattr(candidate, "source_id"):
                    continue
                try:
                    value = candidate.source_id()
                except Exception:
                    continue
                if str(value).strip() == source_id:
                    candidates.append(candidate)
        if name or stype:
            candidates = [
                candidate
                for candidate in candidates
                if (not name or candidate.name() == name)
                and (not stype or candidate.type() == stype)
            ]
        if not candidates:
            raise RuntimeError(f"No matching LSL stream found for source_id={source_id}.")
        if len(candidates) > 1:
            details = "\n".join(f"- {_format_stream(c)}" for c in candidates)
            raise RuntimeError(
                "Multiple LSL streams matched source_id. "
                f"Refine with --stream-name/--stream-type.\n{details}"
            )
        return candidates[0]

    matches = [
        info
        for info in resolve_streams()
        if (not name or info.name() == name) and (not stype or info.type() == stype)
    ]
    if not matches:
        raise RuntimeError("No matching LSL stream found.")
    if len(matches) > 1:
        details = "\n".join(f"- {_format_stream(c)}" for c in matches)
        raise RuntimeError(
            "Multiple LSL streams matched. Use --lsl-source-id to disambiguate.\n"
            + details
        )
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="models/finger_action_model.pt")
    parser.add_argument("--scaler-path", type=str, default="scaler.save")
    parser.add_argument("--stream-name", type=str, default=None)
    parser.add_argument("--stream-type", type=str, default="EEG")
    parser.add_argument("--lsl-source-id", type=str, default=None)
    parser.add_argument("--window-sec", type=float, default=0.25)
    parser.add_argument("--hop-sec", type=float, default=0.05)
    parser.add_argument("--target-fs", type=float, default=256.0)
    parser.add_argument("--log-every", type=float, default=1.0)
    parser.add_argument("--allow-drop", action="store_true", help="Allow dropping windows")
    parser.add_argument(
        "--latency-threshold-ms",
        type=float,
        default=250.0,
        help="Warn/drop if rolling p95 latency exceeds this threshold",
    )
    parser.add_argument(
        "--latency-policy",
        type=str,
        choices=["warn", "drop", "degrade"],
        default="warn",
        help="Behavior when latency threshold is exceeded",
    )
    parser.add_argument(
        "--enable-actuation",
        action="store_true",
        help="Enable actuation (placeholder hook)",
    )
    parser.add_argument(
        "--i-understand-this-moves-the-hand",
        action="store_true",
        help="Required safety acknowledgement for actuation.",
    )
    parser.add_argument(
        "--bluetooth-target",
        type=str,
        default="",
        help="Bluetooth target name/address for actuation (if enabled)",
    )
    parser.add_argument(
        "--record-raw",
        action="store_true",
        help="Record raw EEG during live inference",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw",
        help="Raw session root directory for live recording",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default="LIVE",
        help="Subject ID for optional raw recording",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Session ID for optional raw recording",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.enable_actuation and not args.i_understand_this_moves_the_hand:
        logger.warning(
            "Actuation requested without --i-understand-this-moves-the-hand; running in safe mode."
        )
        args.enable_actuation = False

    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    model_path = Path(args.model_path)
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNLSTMFingerActionNet(n_channels=4, n_fingers=6, n_actions=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    scaler = None
    scaler_path = Path(args.scaler_path)
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
    if args.enable_actuation:
        logger.info(
            "Actuation enabled (target=%s). Placeholder hook; implement device control here.",
            args.bluetooth_target or "unspecified",
        )
    else:
        logger.info("Actuation disabled (safe mode). Predictions only.")

    info = _resolve_stream(args.stream_name, args.stream_type, args.lsl_source_id)
    inlet = StreamInlet(info, max_buflen=2, max_chunklen=32)
    logger.info("Connected to stream %s (%s)", info.name(), info.type())
    session_writer = None
    raw_buffer: list[SamplePacket] = []
    seq_counter = count(0)
    raw_flush_size = 256
    if args.record_raw:
        session_id = args.session_id or time.strftime("%Y%m%d_%H%M%S")
        channel_count = int(info.channel_count())
        channel_labels = [f"ch{i+1}" for i in range(channel_count)]
        session_writer = SessionWriter(
            output_root=Path(args.raw_dir),
            subject_id=args.subject_id,
            session_id=session_id,
            channel_labels=channel_labels,
            sampling_rate=float(info.nominal_srate() or args.target_fs),
            timebase_version="absolute_v1",
            shard_size_samples=2048,
            resume=False,
            mode="live_infer",
        )
        logger.info("Recording raw to %s", session_writer.paths.session_dir)

    buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=4096)
    stream_start: Optional[float] = None
    next_window_start_s = 0.0
    dropped_windows = 0
    last_log = time.monotonic()
    latency_window: Deque[float] = deque(maxlen=200)
    last_latency_warn = 0.0
    cooldown_s = 0.5

    termination_reason = "normal"
    try:
        while True:
            sample, lsl_ts = inlet.pull_sample(timeout=0.1)
            if sample is None:
                continue
            lsl_ts = float(lsl_ts)
            if stream_start is None:
                stream_start = lsl_ts
            time_s = lsl_ts - stream_start
            buffer.append((time_s, np.asarray(sample, dtype=float)))
            if session_writer is not None:
                raw_buffer.append(
                    SamplePacket(
                        seq=next(seq_counter),
                        lsl_ts_raw=lsl_ts,
                        lsl_ts_mono=lsl_ts,
                        local_ts=time.time(),
                        sample=np.asarray(sample, dtype=float),
                        flags=0,
                        segment_id=0,
                        clamped=False,
                        raw_path=None,
                        segment_break_reason=None,
                    )
                )
                if len(raw_buffer) >= raw_flush_size:
                    session_writer.append_packets(raw_buffer)
                    raw_buffer = []

            while (next_window_start_s + args.window_sec) <= time_s:
                window_start = next_window_start_s
                window_end = window_start + args.window_sec
                times = np.array([t for t, _ in buffer], dtype=float)
                values = np.array([v for _, v in buffer], dtype=float)
                mask = (times >= window_start) & (times < window_end)
                if not np.any(mask):
                    dropped_windows += 1
                    next_window_start_s += args.hop_sec
                    continue
                window_times = times[mask]
                window_values = values[mask]
                window = _resample_window(
                    window_times,
                    window_values,
                    start_s=window_start,
                    end_s=window_end,
                    target_fs=args.target_fs,
                )
                if window is None:
                    dropped_windows += 1
                    next_window_start_s += args.hop_sec
                    continue

                window_input = standardize_window_TxC(window.astype(np.float32), scaler)
                x = torch.tensor(window_input, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.inference_mode():
                    finger_logits, action_logits = model(x)
                    action_probs = torch.softmax(action_logits, dim=1).squeeze(0)
                    finger_probs = torch.softmax(finger_logits, dim=1).squeeze(0)
                pred_action = int(torch.argmax(action_probs).item())
                pred_finger = int(torch.argmax(finger_probs).item())

                now = time.monotonic()
                window_center_lsl = stream_start + window_start + args.window_sec / 2.0
                latency_ms = (now - window_center_lsl) * 1000.0
                latency_window.append(latency_ms)
                if latency_window:
                    p95_latency = float(np.percentile(latency_window, 95))
                else:
                    p95_latency = latency_ms
                if p95_latency > args.latency_threshold_ms:
                    should_warn = now - last_latency_warn >= cooldown_s
                    if should_warn:
                        logger.warning(
                            "p95 latency %.1fms exceeds threshold %.1fms (policy=%s)",
                            p95_latency,
                            args.latency_threshold_ms,
                            args.latency_policy,
                        )
                        last_latency_warn = now
                    if args.allow_drop and args.latency_policy in {"drop", "degrade"}:
                        backlog = int(
                            max(
                                0.0,
                                ((time_s - args.window_sec) - next_window_start_s)
                                / args.hop_sec,
                            )
                        )
                        if backlog > 0:
                            dropped_windows += backlog
                            next_window_start_s += backlog * args.hop_sec
                            logger.warning(
                                "Dropping %s windows to recover latency (p95=%.1fms).",
                                backlog,
                                p95_latency,
                            )
                logger.info(
                    "pred_action=%s pred_finger=%s latency_ms=%.1f dropped_windows=%s",
                    pred_action,
                    pred_finger,
                    latency_ms,
                    dropped_windows,
                )
                if args.enable_actuation:
                    logger.debug(
                        "Actuation hook placeholder (action=%s finger=%s).",
                        pred_action,
                        pred_finger,
                    )
                next_window_start_s += args.hop_sec

            now = time.monotonic()
            if now - last_log >= args.log_every:
                logger.info("buffer=%s dropped_windows=%s", len(buffer), dropped_windows)
                last_log = now
    except KeyboardInterrupt:
        logger.info("Stopping live inference.")
    except Exception as exc:
        termination_reason = "error"
        logger.error("Live inference error: %s", exc)
    finally:
        if session_writer is not None:
            if raw_buffer:
                session_writer.append_packets(raw_buffer)
            session_writer.finalize(termination_reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
