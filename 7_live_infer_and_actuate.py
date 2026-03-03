"""
7_live_infer_and_actuate.py (updated)

Adds real actuation support for an Arduino-controlled robotic hand via Serial (USB serial or Bluetooth SPP serial port).

Protocol sent to Arduino (newline-terminated ASCII):
  "{finger_id},{action_id}\n"
Where:
  finger_id: 0=none, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
  action_id: 0=rest (midpoint), 1=open, 2=close

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

# Torch is required for inference
import torch

# LSL is required for live stream
try:
    from pylsl import StreamInlet, resolve_byprop  # type: ignore
    LSL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StreamInlet = None
    resolve_byprop = None
    LSL_AVAILABLE = False

# Project-local imports (keep as-is; this file is intended to be a drop-in replacement)
# NOTE: If these imports differ in your repo, keep the same ones you already had.
# They exist in the user's original file; we preserve names/structure.
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from muse_streaming.resample import resample_window, verify_alignment
from utils.runtime_utils import apply_channel_normalizer, load_normalizer
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir
from utils.postprocess import PostprocessSettings, PostprocessState, postprocess_predictions

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


def _warmup_actuation(actuator: SerialHandActuator, *, pause_s: float = 0.8, inter_cmd_s: float = 0.03) -> None:
    """
    Visual sanity check: open all fingers, close all, then return to rest (midpoint).
    This is intentionally a best-effort sequence to confirm connectivity.
    """
    for action_id, label in [(1, "open"), (2, "close"), (0, "rest")]:
        for finger_id in range(1, 6):
            actuator.send(finger_id, action_id)
            time.sleep(inter_cmd_s)
        logger.info("Warmup: %s sent for all fingers; waiting %.2fs", label, pause_s)
        time.sleep(pause_s)


# -------------------- Helpers --------------------

def ensure_dir(path: str) -> None:
    Path(path).expanduser().resolve().mkdir(parents=True, exist_ok=True)


def load_json(path: str) -> dict:
    return json.loads(Path(path).expanduser().resolve().read_text())


def _load_config_file(path: Path) -> tuple[dict, dict]:
    payload = load_json(str(path))
    settings = payload.get("settings")
    if isinstance(settings, dict):
        return payload, settings
    return payload, payload


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
    except Exception as exc:
        logger.warning(
            "Resampling failed for window [%.3f, %.3f]: %s", start_s, end_s, exc
        )
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
    def __init__(
        self,
        out_dir: str,
        *,
        channel_count: int,
        shard_size_samples: int = 2048,
    ) -> None:
        from muse_streaming.session_writer import RawShardWriter

        self.out_dir = Path(out_dir).expanduser().resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.out_dir / "raw"
        self._raw_writer = RawShardWriter(
            raw_dir=self.raw_dir,
            channel_count=channel_count,
            shard_size_samples=shard_size_samples,
        )

    def append_packets(self, packets: list[Packet]) -> None:
        if not packets:
            return
        record_arr = self._raw_writer.empty_record_array(len(packets))
        for idx, packet in enumerate(packets):
            record_arr["seq"][idx] = int(packet.seq)
            record_arr["lsl_ts_raw"][idx] = float(packet.lsl_ts_raw)
            record_arr["lsl_ts_mono"][idx] = float(packet.lsl_ts_mono)
            record_arr["local_ts"][idx] = float(packet.local_ts)
            record_arr["flags"][idx] = int(packet.flags)
            record_arr["segment_id"][idx] = int(packet.segment_id)
            record_arr["clamped"][idx] = int(bool(packet.clamped))
            record_arr["sample"][idx] = np.asarray(packet.sample, dtype=float).reshape(-1)
        self._raw_writer.append(record_arr)

    def close(self) -> None:
        try:
            self._raw_writer.flush()
        except Exception:
            pass


def load_model_and_scaler(
    model_path: str, scaler_path: str, *, device: torch.device
) -> tuple[torch.nn.Module, object]:
    model_path_p = Path(model_path).expanduser().resolve()
    if not model_path_p.exists():
        raise FileNotFoundError(f"Model not found: {model_path_p}")
    if not model_path_p.suffix.lower() in {".pt", ".pth"}:
        raise ValueError(f"Unexpected model file extension: {model_path_p.suffix}")
    state = torch.load(model_path_p, map_location=device, weights_only=True)
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
        if scaler_path_p.suffix.lower() != ".npz":
            logger.warning("Unexpected scaler extension: %s", scaler_path_p.suffix)
        scaler = load_normalizer(scaler_path_p)
        if scaler is None:
            logger.warning("Failed to load scaler from %s", scaler_path_p)
    return model, scaler


def _safe_float(x: float) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _latest_dir_by_mtime(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def _resolve_repo_root(config_path: Path) -> Path:
    parts = list(config_path.resolve().parts)
    for idx, part in enumerate(parts):
        if part == "Projects":
            if idx == 0:
                return Path("/")
            return Path(*parts[:idx])
    return config_path.parent


def _derive_project_subject(
    config_payload: dict,
    config_path: Path,
    project_override: Optional[str],
    subject_override: Optional[str],
    config_settings: Optional[dict] = None,
) -> tuple[Optional[str], Optional[str]]:
    settings = config_settings or {}
    project_name = (
        project_override
        or config_payload.get("project_name")
        or config_payload.get("project")
        or settings.get("project_name")
        or settings.get("project")
    )
    subject_id = (
        subject_override
        or config_payload.get("subject_id")
        or settings.get("subject_id")
    )
    if project_name and subject_id:
        return str(project_name), str(subject_id)
    parts = config_path.resolve().parts
    for idx in range(len(parts) - 3):
        if parts[idx] == "Projects" and parts[idx + 2] == "subjects":
            return parts[idx + 1], parts[idx + 3]
    return None, None


def _ensure_unique_output_dir(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.name
    parent = path.parent
    for i in range(2, 1000):
        candidate = parent / f"{stem}_v{i}"
        if not candidate.exists():
            return candidate
    return path


def _is_noop_decision(finger_id: int, action_id: int) -> bool:
    """
    Returns True if the decision represents a guaranteed no-op.
    Semantics:
      finger_id == 0 -> NONE
      action_id == 0 -> REST (suppressed for safety during inference)
    """
    return int(finger_id) == 0 or int(action_id) == 0


def _build_arg_parser() -> tuple[argparse.ArgumentParser, dict]:
    pp_defaults = PostprocessSettings()
    defaults = {
        "device": None,
        "session_dir": None,
        "LIVE_VIZ_ENABLED": False,
        "LIVE_VIZ_FPS": 2.0,
        "window_sec": 0.25,
        "hop_sec": 0.05,
        "target_fs": 256.0,
        "latency_threshold_ms": 750.0,
        "latency_policy": "warn",
        "allow_drop": False,
        "log_every": 5.0,
        "enable_actuation": False,
        "serial_port": None,
        "serial_baud": 9600,
        "actuation_min_prob": 0.65,
        "actuation_stability": 2,
        "actuation_cooldown_ms": 150,
        "allow_outside_base": False,
        "no_file_io": False,
        "subject_id": None,
        "project_name": None,
        "postprocess": True,
        "smoothing_enabled": bool(pp_defaults.smoothing_enabled),
        "smoothing_method": str(pp_defaults.smoothing_method),
        "smoothing_window": int(pp_defaults.smoothing_window),
        "hysteresis_enabled": bool(pp_defaults.hysteresis_enabled),
        "hysteresis_frames": int(pp_defaults.hysteresis_frames),
        "threshold_action": float(pp_defaults.threshold_action),
        "threshold_finger": float(pp_defaults.threshold_finger),
        "adjacency_enabled": bool(pp_defaults.adjacency_enabled),
        "hysteresis_margin": float(pp_defaults.hysteresis_margin),
        "finger_delta": float(pp_defaults.finger_delta),
        "finger_mode": str(pp_defaults.finger_mode),
        "pred_log": None,
    }
    p = argparse.ArgumentParser(description="Live inference + optional robotic hand actuation")
    p.set_defaults(**defaults)

    # Existing args (preserved from original file)
    p.add_argument("--config", required=True, type=str, help="Path to step7 config JSON")
    p.add_argument("--model-path", type=str, help="Override model file path.")
    p.add_argument("--scaler-path", type=str, help="Override scaler file path.")
    p.add_argument("--out-dir", type=str, help="Override output directory.")
    p.add_argument("--device", type=str, help="torch device override (e.g., cpu, mps, cuda)")
    p.add_argument(
        "--session-dir",
        type=str,
        help="Session directory (derives model/scaler + output dir defaults).",
    )
    p.add_argument("--subject-id", type=str, help="Subject ID (auto-resolve latest session).")
    p.add_argument("--project-name", type=str, help="Project name (auto-resolve latest session).")

    p.add_argument("--window_sec", type=float, help="Window length (s).")
    p.add_argument("--hop_sec", type=float, help="Window hop (s).")
    p.add_argument("--target_fs", type=float, help="Target FS for resampling.")

    p.add_argument("--latency_threshold_ms", type=float, help="Latency p95 threshold (ms).")
    p.add_argument(
        "--latency_policy",
        type=str,
        choices=["warn", "drop", "degrade"],
        help="Latency policy (warn/drop/degrade).",
    )
    p.add_argument("--allow_drop", action="store_true")
    p.add_argument("--log_every", type=float, help="Log interval (s).")
    p.add_argument(
        "--live_viz",
        dest="LIVE_VIZ_ENABLED",
        action="store_true",
        help="Emit live visualization updates (UI model view).",
    )
    p.add_argument(
        "--live_viz_fps",
        dest="LIVE_VIZ_FPS",
        type=float,
        help="Live visualization update rate (Hz).",
    )

    # Postprocess knobs
    post_group = p.add_mutually_exclusive_group()
    post_group.add_argument(
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="Enable postprocess smoothing/thresholding.",
    )
    post_group.add_argument(
        "--no-postprocess",
        dest="postprocess",
        action="store_false",
        help="Disable postprocess and use raw argmax predictions.",
    )
    smooth_group = p.add_mutually_exclusive_group()
    smooth_group.add_argument(
        "--smoothing-enabled",
        dest="smoothing_enabled",
        action="store_true",
        help="Enable postprocess smoothing.",
    )
    smooth_group.add_argument(
        "--no-smoothing",
        dest="smoothing_enabled",
        action="store_false",
        help="Disable postprocess smoothing.",
    )
    hyst_group = p.add_mutually_exclusive_group()
    hyst_group.add_argument(
        "--hysteresis-enabled",
        dest="hysteresis_enabled",
        action="store_true",
        help="Enable postprocess hysteresis.",
    )
    hyst_group.add_argument(
        "--no-hysteresis",
        dest="hysteresis_enabled",
        action="store_false",
        help="Disable postprocess hysteresis.",
    )
    adj_group = p.add_mutually_exclusive_group()
    adj_group.add_argument(
        "--adjacency-enabled",
        dest="adjacency_enabled",
        action="store_true",
        help="Enable adjacency correction for fingers.",
    )
    adj_group.add_argument(
        "--no-adjacency",
        dest="adjacency_enabled",
        action="store_false",
        help="Disable adjacency correction for fingers.",
    )
    p.add_argument(
        "--smoothing-method",
        type=str,
        choices=["vote", "ema"],
        help="Postprocess smoothing method.",
    )
    p.add_argument(
        "--smoothing-window",
        type=int,
        help="Postprocess smoothing window size.",
    )
    p.add_argument(
        "--hysteresis-frames",
        type=int,
        help="Postprocess hysteresis frames.",
    )
    p.add_argument(
        "--threshold-action",
        type=float,
        help="Postprocess action confidence threshold.",
    )
    p.add_argument(
        "--threshold-finger",
        type=float,
        help="Postprocess finger confidence threshold.",
    )
    p.add_argument(
        "--hysteresis-margin",
        type=float,
        help="Postprocess hysteresis margin.",
    )
    p.add_argument(
        "--finger-delta",
        type=float,
        help="Postprocess finger delta threshold.",
    )
    p.add_argument(
        "--finger-mode",
        type=str,
        choices=["raw", "smooth"],
        help="Finger smoothing mode (raw/smooth).",
    )

    # New: actuation knobs
    p.add_argument("--enable_actuation", action="store_true", help="Enable sending commands to Arduino hand")
    p.add_argument("--serial_port", type=str, help="Serial port (e.g. /dev/tty.usbmodem*, /dev/tty.*)")
    p.add_argument("--serial_baud", type=int, help="Baud rate (must match Arduino sketch)")
    p.add_argument("--actuation_min_prob", type=float, help="Min joint confidence to actuate")
    p.add_argument("--actuation_stability", type=int, help="Require same decision N windows in a row")
    p.add_argument("--actuation_cooldown_ms", type=int, help="Min time between sends")
    p.add_argument("--pred-log", type=str, help="Optional prediction log JSONL path override.")
    p.add_argument(
        "--allow_outside_base",
        action="store_true",
        help="Allow output dir outside session/config base directory.",
    )
    p.add_argument(
        "--no_file_io",
        action="store_true",
        help="Disable all file outputs (raw shards + log file) for max performance.",
    )

    return p, defaults


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
    if not LSL_AVAILABLE or resolve_byprop is None or StreamInlet is None:
        raise RuntimeError("pylsl is required for live inference.")
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


def _postprocess_decision(
    action_probs: np.ndarray,
    finger_probs: np.ndarray,
    *,
    enabled: bool,
    settings: PostprocessSettings,
    state: PostprocessState,
) -> dict:
    if not enabled:
        raw_action = int(np.argmax(action_probs)) if action_probs.size else 0
        raw_finger = int(np.argmax(finger_probs)) if finger_probs.size else 0
        action_conf = float(np.max(action_probs)) if action_probs.size else 0.0
        finger_conf = float(np.max(finger_probs)) if finger_probs.size else 0.0
        return {
            "committed_action_id": raw_action,
            "committed_finger_id": raw_finger,
            "raw_top_action_id": raw_action,
            "raw_top_finger_id": raw_finger,
            "action_conf": action_conf,
            "finger_conf": finger_conf,
            "smoothed_action_id": raw_action,
            "smoothed_finger_id": raw_finger,
            "decision_reason": "raw_argmax",
            "frames_in_state": 1,
        }
    return postprocess_predictions(action_probs, finger_probs, settings, state)


def _compute_hidden_mag(model: CNNLSTMFingerActionNet, x: torch.Tensor) -> Optional[float]:
    """
    Returns the last-step LSTM hidden magnitude for a window, or None on failure.
    x: [B, T, C]
    """
    try:
        x = x.permute(0, 2, 1)
        x = model.conv(x)
        x = x.permute(0, 2, 1)
        out, _ = model.lstm(x)
        hidden_mag = torch.linalg.norm(out, dim=2).squeeze(0)
        if hidden_mag.numel() == 0:
            return None
        value = float(hidden_mag[-1].item())
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


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
    parser, defaults = _build_arg_parser()
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config_payload, config_settings = _load_config_file(config_path)
    _apply_config_to_args(args, config_settings, defaults)

    # Required config keys (as in original file)
    lsl_name = config_settings.get("lsl_name", config_settings.get("stream_name", "Muse2-EEG"))
    lsl_type = config_settings.get("lsl_type", config_settings.get("stream_type", "EEG"))
    session_dir_value = args.session_dir or config_settings.get("session_dir")
    project_name, subject_id = _derive_project_subject(
        config_payload, config_path, args.project_name, args.subject_id, config_settings
    )
    session_dir_inferred = False
    if not session_dir_value and project_name and subject_id:
        repo_root = _resolve_repo_root(config_path)
        sessions_root = (
            repo_root
            / "Projects"
            / project_name
            / "subjects"
            / subject_id
            / "sessions"
        )
        latest_session = _latest_dir_by_mtime(sessions_root)
        if latest_session is not None:
            session_dir_value = str(latest_session)
            session_dir_inferred = True
    model_path = args.model_path or (None if session_dir_value else config_settings.get("model_path"))
    scaler_path = args.scaler_path or (None if session_dir_value else config_settings.get("scaler_path"))
    out_dir = args.out_dir or (None if session_dir_value else config_settings.get("out_dir"))

    def _resolve_path(path_str: str, base_dir: Optional[Path]) -> str:
        candidate = Path(path_str).expanduser()
        if not candidate.is_absolute():
            base = base_dir if base_dir is not None else Path.cwd()
            candidate = (base / candidate).resolve()
        return str(candidate)

    def _is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    selection_source = "legacy_explicit"
    base_dir: Optional[Path] = None
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
        if args.model_path:
            explicit_overrides.append("model_path")
        if args.scaler_path:
            explicit_overrides.append("scaler_path")
        if args.out_dir:
            explicit_overrides.append("out_dir")
        if explicit_overrides:
            print(
                f"⚠️ Explicit paths provided with --session-dir; using overrides: {explicit_overrides}"
            )
            selection_source = "legacy_explicit"
        else:
            selection_source = "subject_latest" if session_dir_inferred else "session_dir"

        run_dir = resolve_latest_run_dir(session_dir_path)
        if run_dir is None or not run_dir.exists():
            print("Session selection source: session_dir")
            print(
                "No model run directory found. Train a model first (Step 2), or pass explicit model_path/scaler_path."
            )
            return 2
        base_dir = session_dir_path
        if not model_path:
            model_path = str(run_dir / "finger_action_model.pt")
        else:
            model_path = _resolve_path(str(model_path), base_dir)
        if not scaler_path:
            scaler_path = str(run_dir / "scaler.npz")
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
        base_dir = config_dir
        model_path = _resolve_path(str(model_path), config_dir)
        scaler_path = _resolve_path(str(scaler_path), config_dir)
        out_dir = _resolve_path(str(out_dir), config_dir)

    base_dir = base_dir if base_dir is not None else Path.cwd()
    out_dir_path = Path(out_dir).expanduser().resolve()
    if not args.allow_outside_base:
        if not _is_relative_to(out_dir_path, base_dir):
            raise ValueError(
                f"out_dir must be within {base_dir} (got {out_dir_path}). "
                "Pass --allow_outside_base to override."
            )
    out_dir = str(out_dir_path)

    config_no_file_io = config_settings.get("no_file_io")
    if config_no_file_io is None and "record_raw" in config_settings:
        config_no_file_io = not bool(config_settings.get("record_raw"))
    no_file_io = bool(args.no_file_io or config_no_file_io)
    record_raw = not no_file_io
    if record_raw:
        unique_dir = _ensure_unique_output_dir(out_dir_path)
        if unique_dir != out_dir_path:
            out_dir_path = unique_dir
            out_dir = str(out_dir_path)
            print(f"Output dir exists; using: {out_dir}")

    print(f"Session selection source: {selection_source}")
    print(f"Using model file: {model_path}")
    print(f"Using scaler file: {scaler_path}")
    if no_file_io:
        print("File outputs disabled: raw shards + log file")
    else:
        print(f"Saving outputs to: {out_dir}")

    if not no_file_io:
        ensure_dir(out_dir)

    setup_logger(
        log_path="" if no_file_io else str(Path(out_dir) / "live_infer.log"),
        level=logging.INFO,
    )

    postprocess_enabled = bool(args.postprocess)
    post_settings = PostprocessSettings(
        smoothing_enabled=bool(args.smoothing_enabled),
        smoothing_method=str(args.smoothing_method),
        smoothing_window=int(args.smoothing_window),
        hysteresis_enabled=bool(args.hysteresis_enabled),
        hysteresis_frames=int(args.hysteresis_frames),
        threshold_action=float(args.threshold_action),
        threshold_finger=float(args.threshold_finger),
        adjacency_enabled=bool(args.adjacency_enabled),
        hysteresis_margin=float(args.hysteresis_margin),
        finger_delta=float(args.finger_delta),
        finger_mode=str(args.finger_mode),
    )
    post_state = PostprocessState()

    pred_log = None
    pred_log_path = None
    pred_log_flush_every = 50
    pred_log_count = 0
    if not no_file_io:
        pred_log_path = args.pred_log or str(Path(out_dir) / "predictions.jsonl")
        try:
            pred_log = Path(pred_log_path).open("a")
            logger.info("Prediction log: %s", pred_log_path)
        except Exception as exc:
            logger.warning("Failed to open prediction log %s: %s", pred_log_path, exc)

    device = _select_device(args.device)
    logger.info("Using device=%s", device)

    model, scaler = load_model_and_scaler(model_path, scaler_path, device=device)
    model.eval()

    live_viz_enabled = bool(getattr(args, "LIVE_VIZ_ENABLED", False))
    live_viz_fps = float(getattr(args, "LIVE_VIZ_FPS", 0.0) or 0.0)
    if live_viz_fps <= 0.0:
        live_viz_enabled = False
    live_viz_interval = (1.0 / live_viz_fps) if live_viz_enabled else 0.0
    last_live_viz_emit = 0.0

    inlet = _resolve_lsl_inlet(lsl_name, lsl_type, timeout_s=8.0)
    info = inlet.info()
    sfreq = float(info.nominal_srate())
    ch = int(info.channel_count())
    logger.info("Connected LSL stream name=%s type=%s sfreq=%s ch=%s", lsl_name, lsl_type, sfreq, ch)

    # Session writer (raw shards, optional)
    session_writer = None
    raw_buffer: list[Packet] = []
    raw_flush_size = int(config_settings.get("raw_flush_size", 256))
    if record_raw:
        raw_shard_samples = int(config_settings.get("raw_shard_samples", 2048))
        session_writer = SessionWriter(
            out_dir=str(out_dir),
            channel_count=ch,
            shard_size_samples=raw_shard_samples,
        )
    else:
        logger.info("Raw recording disabled (no_file_io).")

    # Serial actuator
    actuator: Optional[SerialHandActuator] = None
    if args.enable_actuation:
        if not args.serial_port:
            raise RuntimeError("--enable_actuation requires --serial_port (e.g. /dev/tty.usbmodem*, /dev/tty.*)")
        actuator = SerialHandActuator(args.serial_port, baud=args.serial_baud)
        actuator.open()
        logger.info("Actuation enabled via serial port %s @ %s baud", args.serial_port, args.serial_baud)
        _warmup_actuation(actuator)

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

                    # Persist raw packets (optional)
                    if record_raw and session_writer is not None:
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

                alignment = verify_alignment(
                    window_times,
                    start_s=window_start,
                    end_s=window_end,
                    target_fs=args.target_fs,
                    max_gap_s=1.0 / float(args.target_fs) * 4.0,
                )
                if not alignment.ok:
                    dropped_windows += 1
                    if pred_log is not None:
                        payload = {
                            "ts_utc": time.time(),
                            "window_start_s": float(window_start),
                            "window_end_s": float(window_end),
                            "latency_ms": None,
                            "alignment_ok": False,
                            "alignment_reason": alignment.reason,
                            "decision_reason": "alignment_fail",
                            "committed_action_id": 0,
                            "committed_finger_id": 0,
                        }
                        pred_log.write(json.dumps(payload) + "\n")
                        pred_log_count += 1
                        if pred_log_count % pred_log_flush_every == 0:
                            pred_log.flush()
                    next_window_start_s += args.hop_sec
                    continue

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

                emit_viz = False
                viz_ts = None
                now_mono = time.monotonic()
                if live_viz_enabled and (now_mono - last_live_viz_emit) >= live_viz_interval:
                    emit_viz = True
                    viz_ts = float(window_end)

                with torch.inference_mode():
                    finger_logits, action_logits = model(x)
                    action_probs_t = torch.softmax(action_logits, dim=1).squeeze(0)
                    finger_probs_t = torch.softmax(finger_logits, dim=1).squeeze(0)
                    hidden_mag = _compute_hidden_mag(model, x) if emit_viz else None

                action_probs = action_probs_t.detach().cpu().numpy()
                finger_probs = finger_probs_t.detach().cpu().numpy()

                decision_info = _postprocess_decision(
                    action_probs,
                    finger_probs,
                    enabled=postprocess_enabled,
                    settings=post_settings,
                    state=post_state,
                )
                decision = ActuationDecision(
                    finger_id=int(decision_info["committed_finger_id"]),
                    action_id=int(decision_info["committed_action_id"]),
                    prob=float(min(decision_info["action_conf"], decision_info["finger_conf"])),
                )

                # Latency tracking
                now = time.monotonic()
                window_center_lsl = stream_start + window_start + args.window_sec / 2.0
                latency_ms = (now - window_center_lsl) * 1000.0
                latency_window.append(latency_ms)

                p95_latency = float(np.percentile(latency_window, 95)) if latency_window else float(latency_ms)

                if _is_noop_decision(decision.finger_id, decision.action_id):
                    logger.info(
                        "PREDICT NO-OP finger=%s action=%s joint_prob=%.3f raw_finger=%s raw_action=%s reason=%s latency_ms=%.1f dropped_windows=%s",
                        decision.finger_id,
                        decision.action_id,
                        decision.prob,
                        decision_info.get("raw_top_finger_id"),
                        decision_info.get("raw_top_action_id"),
                        decision_info.get("decision_reason"),
                        latency_ms,
                        dropped_windows,
                    )
                else:
                    logger.info(
                        "PREDICT ACTUATABLE finger=%s action=%s joint_prob=%.3f raw_finger=%s raw_action=%s reason=%s latency_ms=%.1f dropped_windows=%s",
                        decision.finger_id,
                        decision.action_id,
                        decision.prob,
                        decision_info.get("raw_top_finger_id"),
                        decision_info.get("raw_top_action_id"),
                        decision_info.get("decision_reason"),
                        latency_ms,
                        dropped_windows,
                    )

                if emit_viz and hidden_mag is not None and viz_ts is not None:
                    last_live_viz_emit = now_mono
                    print(f"VIZ t={viz_ts:.3f} hidden_mag={hidden_mag:.6f}", flush=True)

                if pred_log is not None:
                    payload = {
                        "ts_utc": time.time(),
                        "window_start_s": float(window_start),
                        "window_end_s": float(window_end),
                        "latency_ms": float(latency_ms),
                        "alignment_ok": True,
                        "action_probs": action_probs.tolist(),
                        "finger_probs": finger_probs.tolist(),
                        "raw_top_action_id": int(decision_info.get("raw_top_action_id", 0)),
                        "raw_top_finger_id": int(decision_info.get("raw_top_finger_id", 0)),
                        "smoothed_action_id": int(decision_info.get("smoothed_action_id", 0)),
                        "smoothed_finger_id": int(decision_info.get("smoothed_finger_id", 0)),
                        "committed_action_id": int(decision_info.get("committed_action_id", 0)),
                        "committed_finger_id": int(decision_info.get("committed_finger_id", 0)),
                        "action_conf": float(decision_info.get("action_conf", 0.0)),
                        "finger_conf": float(decision_info.get("finger_conf", 0.0)),
                        "joint_conf": float(decision.prob),
                        "decision_reason": str(decision_info.get("decision_reason", "")),
                        "postprocess_enabled": bool(postprocess_enabled),
                        "dropped_windows": int(dropped_windows),
                    }
                    pred_log.write(json.dumps(payload) + "\n")
                    pred_log_count += 1
                    if pred_log_count % pred_log_flush_every == 0:
                        pred_log.flush()

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
            if record_raw and session_writer is not None:
                if raw_buffer:
                    session_writer.append_packets(raw_buffer)
                session_writer.close()
        finally:
            if pred_log is not None:
                try:
                    pred_log.flush()
                    pred_log.close()
                except Exception:
                    pass
            if actuator is not None:
                actuator.close()
            logger.info("Shutdown complete (reason=%s).", termination_reason)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
