"""
7_live_infer_and_actuate.py (updated)

Adds real actuation support for an Arduino-controlled robotic hand via Serial (USB serial or Bluetooth SPP serial port).

Protocol sent to Arduino (newline-terminated ASCII):
  "{finger_id},{action_id}\n"
Where:
  finger_id: 0=none, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
  action_id: 0=rest, 1=open, 2=close

This matches the project conventions used in event logs (rest down-weighting, etc.).

Invariant:
  finger_id=0 is NONE and is always a no-op; never actuate hardware.

Manual test (serial):
  - Send "0,1\n" -> should do nothing (no-op).
  - Send "1,1\n" -> should open thumb.
  - Send "1,2\n" -> should close thumb.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np
import joblib

# Torch is required for inference
import torch

# LSL is required for live stream
from pylsl import StreamInlet, resolve_byprop  # type: ignore

# Project-local imports (keep as-is; this file is intended to be a drop-in replacement)
# NOTE: If these imports differ in your repo, keep the same ones you already had.
# They exist in the user's original file; we preserve names/structure.
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from muse_streaming.resample import resample_window
from utils.runtime_utils import apply_channel_normalizer
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir

# Pipeline handoff: Step 7 runs online inference with the trained Step 2 model
# and optional hardware actuation, while writing live-session artifacts.

logger = logging.getLogger("live_infer")


# -------------------- Serial Actuation --------------------

@dataclass
class ActuationDecision:
    finger_id: int
    action_id: int
    prob: float


class SerialHandActuator:
    """
    Best-effort serial actuator.
    - Uses pyserial if installed
    - Sends ASCII protocol: "{finger},{action}\\n"
    """
    def __init__(self, port: str, baud: int = 9600, write_timeout: float = 0.2):
        try:
            import serial  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pyserial is required for --enable_actuation with --serial_port. "
                "Install with: pip install pyserial"
            ) from exc

        self._serial_mod = serial
        self.port = port
        self.baud = baud
        self.write_timeout = write_timeout
        self.ser = None

    def open(self) -> None:
        self.ser = self._serial_mod.Serial(
            self.port,
            self.baud,
            timeout=0,          # non-blocking reads (we don't read)
            write_timeout=self.write_timeout,
        )
        # Give Arduino time to reset after opening USB serial
        time.sleep(1.2)

    def close(self) -> None:
        try:
            if self.ser is not None:
                self.ser.close()
        finally:
            self.ser = None

    def send(self, finger_id: int, action_id: int) -> None:
        if self.ser is None:
            return
        line = f"{finger_id},{action_id}\n".encode("ascii", errors="ignore")
        self.ser.write(line)
        # don't force flush; OS buffers are fine for this use-case


# -------------------- Helpers --------------------

def ensure_dir(path: str) -> None:
    Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text())


def setup_logger(log_path: str, level: int = logging.INFO) -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
    )
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    if log_path:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def standardize_window_TxC(window_TxC: np.ndarray, scaler: object) -> np.ndarray:
    return apply_channel_normalizer(window_TxC, scaler)


def _resample_window(
    times: np.ndarray,
    values: np.ndarray,
    *,
    start_s: float,
    end_s: float,
    target_fs: float,
) -> Optional[np.ndarray]:
    try:
        _, window = resample_window(
            times,
            values,
            start_s=start_s,
            end_s=end_s,
            target_fs=target_fs,
        )
        return window
    except Exception:
        return None


@dataclass(frozen=True)
class Packet:
    seq: int
    lsl_ts_raw: float
    lsl_ts_mono: float
    local_ts: float
    sample: np.ndarray
    flags: int
    segment_id: int
    clamped: bool
    raw_path: Optional[Path] = None
    segment_break_reason: Optional[str] = None


class SessionWriter:
    def __init__(self, out_dir: str) -> None:
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._raw_path = self.out_dir / "raw.csv"
        self._raw_handle = self._raw_path.open("a", newline="")
        import csv

        self._raw_writer = csv.writer(self._raw_handle)
        self._header_written = self._raw_path.stat().st_size > 0

    def append_packets(self, packets: list[Packet]) -> None:
        if not packets:
            return
        if not self._header_written:
            sample_len = int(np.asarray(packets[0].sample).reshape(-1).shape[0])
            header = [
                "seq",
                "lsl_ts_raw",
                "lsl_ts_mono",
                "local_ts",
                "flags",
                "segment_id",
                "clamped",
            ] + [f"ch{i + 1}" for i in range(sample_len)]
            self._raw_writer.writerow(header)
            self._header_written = True
        for packet in packets:
            sample = np.asarray(packet.sample, dtype=float).reshape(-1)
            row = [
                int(packet.seq),
                float(packet.lsl_ts_raw),
                float(packet.lsl_ts_mono),
                float(packet.local_ts),
                int(packet.flags),
                int(packet.segment_id),
                int(bool(packet.clamped)),
            ]
            row.extend(sample.tolist())
            self._raw_writer.writerow(row)
        self._raw_handle.flush()

    def close(self) -> None:
        try:
            self._raw_handle.flush()
            self._raw_handle.close()
        except Exception:
            pass


def load_model_and_scaler(
    model_path: str, scaler_path: str, *, device: torch.device
) -> tuple[torch.nn.Module, object]:
    model_path_p = Path(model_path).expanduser().resolve()
    if not model_path_p.exists():
        raise FileNotFoundError(f"Model not found: {model_path_p}")
    state = torch.load(model_path_p, map_location=device)
    try:
        in_ch = int(state["conv.0.weight"].shape[1])
    except Exception:
        in_ch = 4
    n_fingers = int(state["finger_head.weight"].shape[0])
    n_actions = int(state["action_head.weight"].shape[0])
    model = CNNLSTMFingerActionNet(
        n_channels=in_ch, n_fingers=n_fingers, n_actions=n_actions
    )
    model.load_state_dict(state)
    model.to(device)

    scaler = None
    scaler_path_p = Path(scaler_path).expanduser().resolve()
    if scaler_path_p.exists():
        try:
            scaler = joblib.load(scaler_path_p)
        except Exception:
            scaler = None
    return model, scaler

def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _is_noop_decision(finger_id: int, action_id: int) -> bool:
    """
    Returns True if the decision represents a guaranteed no-op.
    Semantics:
      finger_id == 0 -> NONE
      action_id == 0 -> REST
    """
    return int(finger_id) == 0 or int(action_id) == 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live inference + optional robotic hand actuation")

    # Existing args (preserved from original file)
    p.add_argument("--config", required=True, type=str, help="Path to step7 config JSON")
    p.add_argument("--device", default=None, type=str, help="torch device override (e.g., cpu, mps, cuda)")
    p.add_argument(
        "--session-dir",
        type=str,
        default=None,
        help="Session directory (derives model/scaler + output dir defaults).",
    )

    p.add_argument("--window_sec", type=float, default=0.25)
    p.add_argument("--hop_sec", type=float, default=0.05)
    p.add_argument("--target_fs", type=float, default=256.0)

    p.add_argument("--latency_threshold_ms", type=float, default=750.0)
    p.add_argument("--latency_policy", type=str, default="warn", choices=["warn", "drop", "degrade"])
    p.add_argument("--allow_drop", action="store_true")
    p.add_argument("--log_every", type=float, default=5.0)

    # New: actuation knobs
    p.add_argument("--enable_actuation", action="store_true", help="Enable sending commands to Arduino hand")
    p.add_argument("--serial_port", type=str, default=None, help="Serial port (e.g. /dev/tty.usbmodem*, /dev/tty.*)")
    p.add_argument("--serial_baud", type=int, default=9600, help="Baud rate (must match Arduino sketch)")
    p.add_argument("--actuation_min_prob", type=float, default=0.65, help="Min joint confidence to actuate")
    p.add_argument("--actuation_stability", type=int, default=2, help="Require same decision N windows in a row")
    p.add_argument("--actuation_cooldown_ms", type=int, default=150, help="Min time between sends")

    return p


def _apply_config_to_args(
    args_obj: argparse.Namespace, settings: dict, defaults: dict
) -> None:
    if not isinstance(settings, dict):
        return
    for key, default in defaults.items():
        if key in settings and getattr(args_obj, key) == default:
            setattr(args_obj, key, settings[key])


def _select_device(device_override: Optional[str]) -> torch.device:
    if device_override:
        return torch.device(device_override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _resolve_lsl_inlet(name: str, type_: str, timeout_s: float = 5.0) -> StreamInlet:
    streams = resolve_byprop("name", name, timeout=timeout_s)
    streams = [s for s in streams if (s.type() == type_)]
    if not streams:
        raise RuntimeError(f"No LSL streams found for name={name} type={type_}.")
    # pick first match
    return StreamInlet(streams[0], max_chunklen=64)


def _choose_actuation(
    finger_probs: torch.Tensor,
    action_probs: torch.Tensor,
) -> ActuationDecision:
    pred_finger = int(torch.argmax(finger_probs).item())
    pred_action = int(torch.argmax(action_probs).item())
    # Joint confidence heuristic: min of the two max probs
    conf = float(min(float(finger_probs[pred_finger].item()), float(action_probs[pred_action].item())))
    return ActuationDecision(finger_id=pred_finger, action_id=pred_action, prob=conf)


def _debounced_should_send(
    decision: ActuationDecision,
    last_sent: Optional[Tuple[int, int]],
    stable_count: int,
    required_stability: int,
    last_send_ts: float,
    cooldown_ms: int,
) -> bool:
    if decision.prob <= 0.0:
        return False
    # Invariant: finger_id=0 is NONE and must never actuate hardware.
    if int(decision.finger_id) == 0:
        return False
    # Invariant: action_id=0 (REST) must never actuate hardware.
    if int(decision.action_id) == 0:
        return False
    if stable_count < required_stability:
        return False
    if last_sent is not None and (decision.finger_id, decision.action_id) == last_sent:
        return False
    if (time.monotonic() - last_send_ts) * 1000.0 < float(cooldown_ms):
        return False
    return True


# -------------------- Main --------------------

def main() -> int:
    parser = _build_arg_parser()
    args = parser.parse_args()
    defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
    cfg = load_json(args.config)
    _apply_config_to_args(args, cfg, defaults)

    # Required config keys (as in original file)
    lsl_name = cfg.get("lsl_name", cfg.get("stream_name", "Muse2-EEG"))
    lsl_type = cfg.get("lsl_type", cfg.get("stream_type", "EEG"))
    model_path = cfg.get("model_path")
    scaler_path = cfg.get("scaler_path")
    out_dir = cfg.get("out_dir")
    session_dir_value = args.session_dir or cfg.get("session_dir")

    def _resolve_path(path_str: str, base_dir: Optional[Path]) -> str:
        candidate = Path(path_str).expanduser()
        if not candidate.is_absolute():
            base = base_dir if base_dir is not None else Path.cwd()
            candidate = (base / candidate).resolve()
        return str(candidate)

    selection_source = "legacy_explicit"
    # Prefer session-dir resolution so model/scaler/output paths come from a
    # single run context, matching the training/evaluation layout.
    if session_dir_value:
        session_dir_path = resolve_session_dir(str(session_dir_value))
        if not session_dir_path.exists():
            print("Session selection source: session_dir")
            print(f"Session dir not found: {session_dir_path}")
            return 2
        base_dir = session_dir_path
        explicit_overrides = []
        if model_path:
            explicit_overrides.append("model_path")
        if scaler_path:
            explicit_overrides.append("scaler_path")
        if out_dir:
            explicit_overrides.append("out_dir")
        if explicit_overrides:
            print(
                f"⚠️ Explicit paths provided with --session-dir; using overrides: {explicit_overrides}"
            )
            selection_source = "legacy_explicit"
        else:
            selection_source = "session_dir"

        run_dir = resolve_latest_run_dir(session_dir_path)
        if run_dir is None or not run_dir.exists():
            print("Session selection source: session_dir")
            print(
                "No model run directory found. Train a model first (Step 2), or pass explicit model_path/scaler_path."
            )
            return 2
        if not model_path:
            model_path = str(run_dir / "finger_action_model.pt")
        else:
            model_path = _resolve_path(str(model_path), base_dir)
        if not scaler_path:
            scaler_path = str(run_dir / "scaler.save")
        else:
            scaler_path = _resolve_path(str(scaler_path), base_dir)
        if not out_dir:
            out_dir = str(SessionLayout(session_dir_path).processed_dir / "live_infer")
        else:
            out_dir = _resolve_path(str(out_dir), base_dir)
    else:
        config_dir = Path(args.config).expanduser().resolve().parent
        if not model_path or not scaler_path or not out_dir:
            print("Session selection source: legacy_explicit")
            print(
                "Missing --session-dir. Config must include model_path, scaler_path, and out_dir."
            )
            return 2
        model_path = _resolve_path(str(model_path), config_dir)
        scaler_path = _resolve_path(str(scaler_path), config_dir)
        out_dir = _resolve_path(str(out_dir), config_dir)

    print(f"Session selection source: {selection_source}")
    print(f"Using model file: {model_path}")
    print(f"Using scaler file: {scaler_path}")
    print(f"Saving outputs to: {out_dir}")

    ensure_dir(out_dir)

    setup_logger(
        log_path=str(Path(out_dir) / "live_infer.log"),
        level=logging.INFO,
    )

    device = _select_device(args.device)
    logger.info("Using device=%s", device)

    model, scaler = load_model_and_scaler(model_path, scaler_path, device=device)
    model.eval()

    inlet = _resolve_lsl_inlet(lsl_name, lsl_type, timeout_s=8.0)
    info = inlet.info()
    sfreq = float(info.nominal_srate())
    ch = int(info.channel_count())
    logger.info("Connected LSL stream name=%s type=%s sfreq=%s ch=%s", lsl_name, lsl_type, sfreq, ch)

    # Session writer (preserved)
    session_writer = SessionWriter(out_dir=str(out_dir))
    raw_flush_size = int(cfg.get("raw_flush_size", 256))
    raw_buffer = []

    # Serial actuator
    actuator: Optional[SerialHandActuator] = None
    if args.enable_actuation:
        if not args.serial_port:
            raise RuntimeError("--enable_actuation requires --serial_port (e.g. /dev/tty.usbmodem*, /dev/tty.*)")
        actuator = SerialHandActuator(args.serial_port, baud=args.serial_baud)
        actuator.open()
        logger.info("Actuation enabled via serial port %s @ %s baud", args.serial_port, args.serial_baud)

    # Live buffers
    from collections import deque
    buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=int(max(5, args.window_sec * args.target_fs * 4)))
    latency_window: Deque[float] = deque(maxlen=200)

    stream_start = time.monotonic()  # approximate, for latency calculation
    dropped_windows = 0
    last_log = time.monotonic()

    next_window_start_s = 0.0
    start_ts = time.monotonic()

    # Debounce state
    last_decision: Optional[Tuple[int, int]] = None
    stable_count = 0
    last_sent: Optional[Tuple[int, int]] = None
    last_send_ts = 0.0
    sample_seq = 0

    termination_reason = "ok"
    try:
        while True:
            # Pull a chunk from LSL
            chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=64)
            if timestamps:
                for sample, lsl_ts in zip(chunk, timestamps):
                    # time since start (monotonic-based)
                    time_s = time.monotonic() - start_ts
                    vec = np.asarray(sample, dtype=np.float32)
                    buffer.append((time_s, vec))

                    # Persist raw packets
                    raw_buffer.append(
                        Packet(
                            seq=sample_seq,
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
                    sample_seq += 1
                    if len(raw_buffer) >= raw_flush_size:
                        session_writer.append_packets(raw_buffer)
                        raw_buffer = []

            # Infer over available windows
            time_s = time.monotonic() - start_ts
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

                decision = _choose_actuation(finger_probs, action_probs)

                # Latency tracking
                now = time.monotonic()
                window_center_lsl = stream_start + window_start + args.window_sec / 2.0
                latency_ms = (now - window_center_lsl) * 1000.0
                latency_window.append(latency_ms)

                p95_latency = float(np.percentile(latency_window, 95)) if latency_window else float(latency_ms)

                if _is_noop_decision(decision.finger_id, decision.action_id):
                    logger.info(
                        "PREDICT NO-OP finger=%s action=%s joint_prob=%.3f latency_ms=%.1f dropped_windows=%s",
                        decision.finger_id,
                        decision.action_id,
                        decision.prob,
                        latency_ms,
                        dropped_windows,
                    )
                else:
                    logger.info(
                        "PREDICT ACTUATABLE finger=%s action=%s joint_prob=%.3f latency_ms=%.1f dropped_windows=%s",
                        decision.finger_id,
                        decision.action_id,
                        decision.prob,
                        latency_ms,
                        dropped_windows,
                    )

                # Stability / debounce
                key = (decision.finger_id, decision.action_id)
                if last_decision == key:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_decision = key

                # Decide to actuate
                if args.enable_actuation and actuator is not None:
                    if decision.prob >= float(args.actuation_min_prob):
                        # Hard safety gate: NEVER actuate on NONE/REST.
                        if int(decision.finger_id) == 0 or int(decision.action_id) == 0:
                            logger.info(
                                "NO-OP decision suppressed (finger=%s action=%s)",
                                decision.finger_id,
                                decision.action_id,
                            )
                        elif _debounced_should_send(
                            decision=decision,
                            last_sent=last_sent,
                            stable_count=stable_count,
                            required_stability=int(args.actuation_stability),
                            last_send_ts=last_send_ts,
                            cooldown_ms=int(args.actuation_cooldown_ms),
                        ):
                            # Send command
                            actuator.send(decision.finger_id, decision.action_id)
                            last_sent = key
                            last_send_ts = time.monotonic()
                            logger.info("ACTUATE sent finger=%s action=%s prob=%.3f", decision.finger_id, decision.action_id, decision.prob)
                    else:
                        logger.debug("Actuation suppressed by min_prob (%.3f < %.3f)", decision.prob, float(args.actuation_min_prob))

                next_window_start_s += args.hop_sec

            # periodic status log
            now = time.monotonic()
            if now - last_log >= args.log_every:
                logger.info("buffer=%s dropped_windows=%s", len(buffer), dropped_windows)
                last_log = now

    except KeyboardInterrupt:
        logger.info("Stopping live inference.")
    except Exception as exc:
        termination_reason = "error"
        logger.error("Live inference error: %s", exc)
        raise
    finally:
        try:
            if session_writer is not None:
                if raw_buffer:
                    session_writer.append_packets(raw_buffer)
                session_writer.close()
        finally:
            if actuator is not None:
                actuator.close()
            logger.info("Shutdown complete (reason=%s).", termination_reason)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
