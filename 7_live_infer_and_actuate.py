"""
7_live_infer_and_actuate.py (updated)

Adds real actuation support for an Arduino-controlled robotic hand via Serial (USB serial or Bluetooth SPP serial port).

Protocol sent to Arduino (newline-terminated ASCII):
  "{finger_id},{action_id},{speed_u8}\n"
Where:
  finger_id: 0=none, 1=thumb, 2=index, 3=middle, 4=ring, 5=pinky
  action_id: 0=rest (midpoint), 1=open, 2=close
  speed_u8: 0-255 scalar derived from prediction confidence

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
import collections
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Optional, Tuple

import numpy as np

# Torch is required for inference
import torch

# LSL is required for live stream
try:
    from pylsl import StreamInlet, resolve_byprop  # type: ignore
    try:
        from pylsl import resolve_streams  # type: ignore
    except Exception:  # pragma: no cover - older pylsl builds
        resolve_streams = None
    LSL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StreamInlet = None
    resolve_byprop = None
    resolve_streams = None
    LSL_AVAILABLE = False

# Project-local imports (keep as-is; this file is intended to be a drop-in replacement)
# NOTE: If these imports differ in your repo, keep the same ones you already had.
# They exist in the user's original file; we preserve names/structure.
from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet
from muse_streaming.resample import resample_window, verify_alignment
from utils.command_shaper import CommandShaper, CommandShaperConfig
from utils.default_recipe import LIVE_INFER_RECIPE_DEFAULTS
from utils.inference import InferenceConfig, InferenceEngine
from utils.label_schema import (
    decode_finger_prediction,
    decode_prediction_pair,
    finger_confidence_for_id,
    is_valid_action_finger,
)
from utils.live_infer_common import (
    ReplayRuntimeConfig,
    applicability_gate_passed as _shared_applicability_gate_passed,
    build_actuation_command_shaper as _shared_build_actuation_command_shaper,
    build_actuation_speed_mapper as _shared_build_actuation_speed_mapper,
    compute_actuation_speed_scalar as _shared_compute_actuation_speed_scalar,
    debounced_should_send as _shared_debounced_should_send,
    finger_gate_passed as _shared_finger_gate_passed,
    is_noop_decision as _shared_is_noop_decision,
    latency_gate_passed as _shared_latency_gate_passed,
    require_deployable_run as _shared_require_deployable_run,
    resolve_actuation_candidate as _shared_resolve_actuation_candidate,
    resolve_temperature_path as _shared_resolve_temperature_path,
    uncertainty_gate_passed as _shared_uncertainty_gate_passed,
)
from utils.model_outputs import infer_output_dims_from_state_dict, unpack_model_outputs
from utils.postprocess import PostprocessSettings, PostprocessState, postprocess_predictions
from utils.runtime_utils import (
    TemperatureScalingState,
    apply_channel_normalizer,
    apply_temperature_to_logits,
    load_normalizer,
    load_temperature_scaling,
)
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir
from utils.stream_timebase import clamp_lsl_timestamp

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
    - Sends ASCII protocol: "{finger},{action},{speed_u8}\\n"
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

    def send(
        self, finger_id: int, action_id: int, speed_scalar: Optional[float] = None
    ) -> None:
        if self.ser is None:
            return
        if speed_scalar is None:
            line = f"{finger_id},{action_id}\n".encode("ascii", errors="ignore")
        else:
            speed_u8 = int(
                max(0, min(255, round(float(speed_scalar) * 255.0)))
            )
            line = f"{finger_id},{action_id},{speed_u8}\n".encode(
                "ascii", errors="ignore"
            )
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


def _load_train_config(run_dir: Path) -> dict:
    path = Path(run_dir).expanduser().resolve() / "train_config.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _resolve_temperature_path(run_dir: Path) -> Path:
    return _shared_resolve_temperature_path(run_dir)


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
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            logger.warning(
                "Resampling skipped for window [%.3f, %.3f]: non-finite input samples",
                start_s,
                end_s,
            )
            return None
        _, window = resample_window(
            times,
            values,
            start_s=start_s,
            end_s=end_s,
            target_fs=target_fs,
        )
        if not np.all(np.isfinite(window)):
            logger.warning(
                "Resampling skipped for window [%.3f, %.3f]: non-finite values after interpolation",
                start_s,
                end_s,
            )
            return None
        return window
    except Exception as exc:
        logger.warning(
            "Resampling failed for window [%.3f, %.3f]: %s", start_s, end_s, exc
        )
        return None


def _resolve_live_sample_time(
    *,
    lsl_ts: float,
    sample_mono: float,
    stream_origin_mono: Optional[float],
    stream_origin_lsl: Optional[float],
    prev_lsl_mono: Optional[float],
) -> Tuple[float, float, bool, Optional[float], Optional[float], Optional[float]]:
    lsl_ts_mono = float(lsl_ts)
    if np.isfinite(lsl_ts_mono):
        clamp_result = clamp_lsl_timestamp(prev_lsl_mono, lsl_ts_mono)
        lsl_ts_mono = float(clamp_result.mono_ts)
        prev_lsl_mono = lsl_ts_mono
        if stream_origin_lsl is None:
            stream_origin_lsl = lsl_ts_mono
            stream_origin_mono = float(sample_mono)
        time_s = lsl_ts_mono - float(stream_origin_lsl)
        return (
            float(time_s),
            lsl_ts_mono,
            bool(clamp_result.clamped),
            stream_origin_mono,
            stream_origin_lsl,
            prev_lsl_mono,
        )

    if stream_origin_mono is None:
        stream_origin_mono = float(sample_mono)
    time_s = float(sample_mono) - float(stream_origin_mono)
    return (
        float(time_s),
        lsl_ts_mono,
        False,
        stream_origin_mono,
        stream_origin_lsl,
        prev_lsl_mono,
    )


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


RAW_FLAG_NONFINITE = 1


@dataclass
class RestFingerBiasCorrection:
    enabled: bool = True
    min_rest_windows: int = 10
    strength: float = 1.5
    ratio_clip_min: float = 0.25
    ratio_clip_max: float = 4.0
    rest_sum: Optional[np.ndarray] = None
    rest_count: int = 0

    def _active_slice(self, probs: np.ndarray) -> slice:
        # Active-finger heads use length 5; legacy heads may include NONE at index 0.
        return slice(1, None) if int(np.asarray(probs).size) == 6 else slice(None)

    @property
    def ready(self) -> bool:
        return bool(self.enabled) and int(self.rest_count) >= max(1, int(self.min_rest_windows))

    def prior(self) -> Optional[np.ndarray]:
        if self.rest_sum is None or int(self.rest_count) <= 0:
            return None
        prior = np.asarray(self.rest_sum, dtype=float) / float(self.rest_count)
        total = float(np.sum(prior))
        if prior.size == 0 or not np.isfinite(total) or total <= 0.0:
            return None
        return prior / total

    def update(self, action_probs: np.ndarray, finger_probs: np.ndarray) -> bool:
        if not bool(self.enabled):
            return False
        action_probs = np.asarray(action_probs, dtype=float).reshape(-1)
        finger_probs = np.asarray(finger_probs, dtype=float).reshape(-1)
        if (
            action_probs.size == 0
            or finger_probs.size == 0
            or not np.all(np.isfinite(action_probs))
            or not np.all(np.isfinite(finger_probs))
            or int(np.argmax(action_probs)) != 0
        ):
            return False
        was_ready = self.ready
        if self.rest_sum is None or self.rest_sum.shape != finger_probs.shape:
            self.rest_sum = np.zeros_like(finger_probs, dtype=float)
            self.rest_count = 0
        self.rest_sum += finger_probs
        self.rest_count += 1
        return (not was_ready) and self.ready

    def apply(self, finger_probs: np.ndarray) -> np.ndarray:
        finger_probs = np.asarray(finger_probs, dtype=float).reshape(-1)
        if finger_probs.size == 0 or not np.all(np.isfinite(finger_probs)):
            return finger_probs
        prior = self.prior()
        if prior is None or not self.ready or prior.shape != finger_probs.shape:
            return finger_probs
        active_slice = self._active_slice(prior)
        prior_active = np.asarray(prior[active_slice], dtype=float)
        if prior_active.size == 0:
            return finger_probs
        uniform = 1.0 / float(prior_active.size)
        ratio = prior_active / float(uniform)
        ratio = np.clip(
            ratio,
            float(self.ratio_clip_min),
            float(self.ratio_clip_max),
        )
        correction = np.power(ratio, float(max(0.0, self.strength)))
        adjusted = finger_probs.copy()
        adjusted[active_slice] = adjusted[active_slice] / correction
        total = float(np.sum(adjusted))
        if not np.all(np.isfinite(adjusted)) or total <= 0.0:
            return finger_probs
        return adjusted / total


@dataclass(frozen=True)
class LiveWindowQuality:
    prepared_window: np.ndarray
    channel_rms_z: np.ndarray
    channel_abs_p95_z: np.ndarray
    channel_clipped_frac: np.ndarray
    masked_channel_ids: tuple[int, ...]
    bad_channel_ids: tuple[int, ...]
    total_clipped_frac: float
    window_quality_bad: bool
    quality_bad_reason: Optional[str] = None


def _sanitize_live_window(
    window_TxC: np.ndarray,
    *,
    scaler: object,
    enabled: bool,
    input_clip_abs_z: float,
    bad_channel_rms_z: float,
    bad_channel_abs_p95_z: float,
    bad_channel_clipped_frac: float,
    bad_window_clipped_frac: float,
    bad_window_max_masked_channels: int,
) -> LiveWindowQuality:
    normalized = np.asarray(standardize_window_TxC(window_TxC, scaler), dtype=np.float32)
    if normalized.ndim != 2:
        raise ValueError(f"Expected 2D window, got shape {normalized.shape}")

    clip_abs_z = float(max(1e-6, input_clip_abs_z))
    if normalized.size == 0:
        empty = np.zeros(normalized.shape[1] if normalized.ndim == 2 else 0, dtype=float)
        return LiveWindowQuality(
            prepared_window=normalized,
            channel_rms_z=empty,
            channel_abs_p95_z=empty,
            channel_clipped_frac=empty,
            masked_channel_ids=tuple(),
            bad_channel_ids=tuple(),
            total_clipped_frac=0.0,
            window_quality_bad=False,
            quality_bad_reason=None,
        )

    abs_normalized = np.abs(normalized)
    channel_rms_z = np.sqrt(np.mean(normalized**2, axis=0)).astype(float)
    channel_abs_p95_z = np.percentile(abs_normalized, 95, axis=0).astype(float)
    clipped_mask = abs_normalized > clip_abs_z
    channel_clipped_frac = np.mean(clipped_mask, axis=0).astype(float)
    total_clipped_frac = float(np.mean(clipped_mask))

    bad_mask = (
        (channel_rms_z > float(bad_channel_rms_z))
        | (channel_abs_p95_z > float(bad_channel_abs_p95_z))
        | (channel_clipped_frac > float(bad_channel_clipped_frac))
    )
    bad_channel_ids = tuple(int(idx) for idx in np.flatnonzero(bad_mask))
    masked_channel_ids: tuple[int, ...] = tuple()
    window_quality_bad = False
    quality_bad_reason: Optional[str] = None

    prepared_window = np.clip(normalized, -clip_abs_z, clip_abs_z).astype(
        np.float32, copy=True
    )
    if enabled and len(bad_channel_ids) == 1:
        bad_idx = int(bad_channel_ids[0])
        prepared_window[:, bad_idx] = 0.0
        masked_channel_ids = (bad_idx,)

    if enabled:
        if len(bad_channel_ids) > int(max(0, bad_window_max_masked_channels)):
            window_quality_bad = True
            quality_bad_reason = "too_many_bad_channels"
        elif total_clipped_frac > float(bad_window_clipped_frac):
            window_quality_bad = True
            quality_bad_reason = "total_clipped_frac"

    return LiveWindowQuality(
        prepared_window=prepared_window if enabled else normalized.astype(np.float32, copy=True),
        channel_rms_z=channel_rms_z,
        channel_abs_p95_z=channel_abs_p95_z,
        channel_clipped_frac=channel_clipped_frac,
        masked_channel_ids=masked_channel_ids,
        bad_channel_ids=bad_channel_ids,
        total_clipped_frac=total_clipped_frac,
        window_quality_bad=bool(window_quality_bad),
        quality_bad_reason=quality_bad_reason,
    )


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
    n_fingers, n_actions, has_applicability_head = infer_output_dims_from_state_dict(
        state
    )
    model = CNNLSTMFingerActionNet(
        n_channels=in_ch,
        n_fingers=n_fingers,
        n_actions=n_actions,
        finger_applicability_head=bool(has_applicability_head),
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
    return _shared_is_noop_decision(finger_id, action_id)


def _build_arg_parser() -> tuple[argparse.ArgumentParser, dict]:
    pp_defaults = PostprocessSettings()
    infer_defaults = InferenceConfig()
    runtime_defaults = ReplayRuntimeConfig()
    defaults = {
        "device": None,
        "session_dir": None,
        "stream_name": None,
        "stream_type": None,
        "bluetooth_target": None,
        "LIVE_VIZ_ENABLED": bool(LIVE_INFER_RECIPE_DEFAULTS["LIVE_VIZ_ENABLED"]),
        "LIVE_VIZ_FPS": float(LIVE_INFER_RECIPE_DEFAULTS["LIVE_VIZ_FPS"]),
        "window_sec": float(runtime_defaults.window_sec),
        "hop_sec": float(runtime_defaults.hop_sec),
        "target_fs": float(LIVE_INFER_RECIPE_DEFAULTS["target_fs"]),
        "latency_threshold_ms": float(runtime_defaults.latency_threshold_ms),
        "latency_policy": str(LIVE_INFER_RECIPE_DEFAULTS["latency_policy"]),
        "allow_drop": bool(LIVE_INFER_RECIPE_DEFAULTS["allow_drop"]),
        "log_every": float(LIVE_INFER_RECIPE_DEFAULTS["log_every"]),
        "enable_actuation": bool(LIVE_INFER_RECIPE_DEFAULTS["enable_actuation"]),
        "serial_port": None,
        "serial_baud": int(LIVE_INFER_RECIPE_DEFAULTS["serial_baud"]),
        "actuation_min_prob": float(runtime_defaults.actuation_min_prob),
        "actuation_stability": int(runtime_defaults.actuation_stability),
        "actuation_cooldown_ms": int(runtime_defaults.actuation_cooldown_ms),
        "actuation_repeat_ms": int(runtime_defaults.actuation_repeat_ms),
        "actuation_min_speed": float(runtime_defaults.actuation_min_speed),
        "modulate_actuation_speed": bool(runtime_defaults.modulate_actuation_speed),
        "actuation_speed_gamma": float(runtime_defaults.actuation_speed_gamma),
        "allow_outside_base": False,
        "no_file_io": bool(LIVE_INFER_RECIPE_DEFAULTS["no_file_io"]),
        "subject_id": None,
        "project_name": None,
        "postprocess": bool(LIVE_INFER_RECIPE_DEFAULTS["postprocess"]),
        "smoothing_enabled": bool(pp_defaults.smoothing_enabled),
        "smoothing_method": str(pp_defaults.smoothing_method),
        "smoothing_window": int(pp_defaults.smoothing_window),
        "hysteresis_enabled": bool(pp_defaults.hysteresis_enabled),
        "hysteresis_frames": int(pp_defaults.hysteresis_frames),
        "threshold_action": float(pp_defaults.threshold_action),
        "threshold_finger": float(pp_defaults.threshold_finger),
        "threshold_applicability": float(pp_defaults.threshold_applicability),
        "adjacency_enabled": bool(pp_defaults.adjacency_enabled),
        "hysteresis_margin": float(pp_defaults.hysteresis_margin),
        "finger_delta": float(pp_defaults.finger_delta),
        "finger_mode": str(pp_defaults.finger_mode),
        "rest_bias_correction_enabled": bool(
            LIVE_INFER_RECIPE_DEFAULTS["rest_bias_correction_enabled"]
        ),
        "rest_bias_strength": float(LIVE_INFER_RECIPE_DEFAULTS["rest_bias_strength"]),
        "rest_bias_min_windows": int(
            LIVE_INFER_RECIPE_DEFAULTS["rest_bias_min_windows"]
        ),
        "live_quality_enabled": bool(LIVE_INFER_RECIPE_DEFAULTS["live_quality_enabled"]),
        "input_clip_abs_z": float(LIVE_INFER_RECIPE_DEFAULTS["input_clip_abs_z"]),
        "bad_channel_rms_z": float(LIVE_INFER_RECIPE_DEFAULTS["bad_channel_rms_z"]),
        "bad_channel_abs_p95_z": float(
            LIVE_INFER_RECIPE_DEFAULTS["bad_channel_abs_p95_z"]
        ),
        "bad_channel_clipped_frac": float(
            LIVE_INFER_RECIPE_DEFAULTS["bad_channel_clipped_frac"]
        ),
        "bad_window_clipped_frac": float(
            LIVE_INFER_RECIPE_DEFAULTS["bad_window_clipped_frac"]
        ),
        "bad_window_max_masked_channels": int(
            LIVE_INFER_RECIPE_DEFAULTS["bad_window_max_masked_channels"]
        ),
        "use_inference_engine": bool(runtime_defaults.use_inference_engine),
        "mc_passes": int(infer_defaults.mc_passes),
        "uncertainty_base_threshold": float(infer_defaults.base_threshold),
        "uncertainty_weight": float(infer_defaults.uncertainty_weight),
        "pred_log": None,
    }
    p = argparse.ArgumentParser(
        description=(
            "Step 7: run live EEG inference from an LSL stream and optionally "
            "send commands to the robotic hand."
        )
    )
    p.set_defaults(**defaults)

    selection_group = p.add_argument_group("session and model")
    selection_group.add_argument(
        "--config",
        required=True,
        type=str,
        metavar="PATH",
        help="Path to the Step 7 JSON config file.",
    )
    selection_group.add_argument(
        "--model-path",
        type=str,
        metavar="PATH",
        help="Override the model weights path.",
    )
    selection_group.add_argument(
        "--scaler-path",
        type=str,
        metavar="PATH",
        help="Override the channel normalizer path.",
    )
    selection_group.add_argument(
        "--out-dir",
        type=str,
        metavar="PATH",
        help="Override the output directory used for live-session artifacts.",
    )
    selection_group.add_argument(
        "--device",
        type=str,
        metavar="NAME",
        help="Torch device override (for example: cpu, mps, cuda).",
    )
    selection_group.add_argument(
        "--session-dir",
        type=str,
        metavar="PATH",
        help="Session directory used to derive default model, scaler, and output paths.",
    )
    selection_group.add_argument(
        "--subject-id",
        type=str,
        metavar="ID",
        help="Subject ID used to auto-resolve the latest session when --session-dir is omitted.",
    )
    selection_group.add_argument(
        "--project-name",
        type=str,
        metavar="NAME",
        help="Project name used with --subject-id to auto-resolve the latest session.",
    )

    stream_group = p.add_argument_group("stream and timing")
    stream_group.add_argument(
        "--stream-name",
        dest="stream_name",
        type=str,
        metavar="NAME",
        help="Override the LSL stream name used for live inference.",
    )
    stream_group.add_argument(
        "--stream-type",
        dest="stream_type",
        type=str,
        metavar="TYPE",
        help="Override the LSL stream type used for live inference.",
    )
    stream_group.add_argument(
        "--window-sec",
        "--window_sec",
        dest="window_sec",
        type=float,
        metavar="SECONDS",
        help="Window length, in seconds, for each inference step.",
    )
    stream_group.add_argument(
        "--hop-sec",
        "--hop_sec",
        dest="hop_sec",
        type=float,
        metavar="SECONDS",
        help="Hop size, in seconds, between successive inference windows.",
    )
    stream_group.add_argument(
        "--target-fs",
        "--target_fs",
        dest="target_fs",
        type=float,
        metavar="HZ",
        help="Target sampling rate, in Hz, for resampling incoming windows.",
    )

    stream_group.add_argument(
        "--latency-threshold-ms",
        "--latency_threshold_ms",
        dest="latency_threshold_ms",
        type=float,
        metavar="MS",
        help="Warn/drop/degrade threshold for p95 latency, in milliseconds.",
    )
    stream_group.add_argument(
        "--latency-policy",
        "--latency_policy",
        dest="latency_policy",
        type=str,
        choices=["warn", "drop", "degrade"],
        help="What to do when latency exceeds the threshold: warn, drop, or degrade.",
    )
    stream_group.add_argument(
        "--allow-drop",
        "--allow_drop",
        dest="allow_drop",
        action="store_true",
        help="Allow dropping work instead of blocking when the live loop falls behind.",
    )
    stream_group.add_argument(
        "--log-every",
        "--log_every",
        dest="log_every",
        type=float,
        metavar="SECONDS",
        help="Emit progress logs at this interval, in seconds.",
    )
    stream_group.add_argument(
        "--live-viz",
        "--live_viz",
        dest="LIVE_VIZ_ENABLED",
        action="store_true",
        help="Emit live visualization updates for the UI model view.",
    )
    stream_group.add_argument(
        "--live-viz-fps",
        "--live_viz_fps",
        dest="LIVE_VIZ_FPS",
        type=float,
        metavar="HZ",
        help="Live visualization update rate, in Hz.",
    )

    # Postprocess knobs
    postprocess_group = p.add_argument_group("postprocessing")
    post_group = postprocess_group.add_mutually_exclusive_group()
    post_group.add_argument(
        "--postprocess",
        dest="postprocess",
        action="store_true",
        help="Enable postprocessing before predictions are emitted or actuated.",
    )
    post_group.add_argument(
        "--no-postprocess",
        dest="postprocess",
        action="store_false",
        help="Disable postprocessing and use raw argmax predictions.",
    )
    smooth_group = postprocess_group.add_mutually_exclusive_group()
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
        help="Disable the smoothing stage inside postprocessing.",
    )
    hyst_group = postprocess_group.add_mutually_exclusive_group()
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
        help="Disable the hysteresis stage inside postprocessing.",
    )
    adj_group = postprocess_group.add_mutually_exclusive_group()
    adj_group.add_argument(
        "--adjacency-enabled",
        dest="adjacency_enabled",
        action="store_true",
        help="Enable adjacency correction for finger predictions.",
    )
    adj_group.add_argument(
        "--no-adjacency",
        dest="adjacency_enabled",
        action="store_false",
        help="Disable adjacency correction for fingers.",
    )
    postprocess_group.add_argument(
        "--smoothing-method",
        type=str,
        choices=["vote", "ema"],
        help="Postprocess smoothing method.",
    )
    postprocess_group.add_argument(
        "--smoothing-window",
        type=int,
        help="Window size used by the smoothing stage.",
    )
    postprocess_group.add_argument(
        "--hysteresis-frames",
        type=int,
        help="Number of consecutive frames required by hysteresis.",
    )
    postprocess_group.add_argument(
        "--threshold-action",
        type=float,
        help="Minimum action confidence required after postprocessing.",
    )
    postprocess_group.add_argument(
        "--threshold-finger",
        type=float,
        help="Minimum finger confidence required after postprocessing.",
    )
    postprocess_group.add_argument(
        "--threshold-applicability",
        type=float,
        help="Minimum applicability probability required before non-REST actuation.",
    )
    postprocess_group.add_argument(
        "--hysteresis-margin",
        type=float,
        help="Postprocess hysteresis margin.",
    )
    postprocess_group.add_argument(
        "--finger-delta",
        type=float,
        help="Minimum finger-score gap used by postprocessing.",
    )
    postprocess_group.add_argument(
        "--finger-mode",
        type=str,
        choices=["raw", "smooth"],
        help="Which finger signal to use after postprocessing: raw or smoothed.",
    )
    rest_bias_group = postprocess_group.add_mutually_exclusive_group()
    rest_bias_group.add_argument(
        "--rest-bias-correction-enabled",
        "--rest_bias_correction_enabled",
        dest="rest_bias_correction_enabled",
        action="store_true",
        help="Debias finger probabilities online using a live rest-window prior.",
    )
    rest_bias_group.add_argument(
        "--no-rest-bias-correction",
        "--no_rest_bias_correction",
        dest="rest_bias_correction_enabled",
        action="store_false",
        help="Disable the live rest-window finger debiasing stage.",
    )
    postprocess_group.add_argument(
        "--rest-bias-strength",
        "--rest_bias_strength",
        dest="rest_bias_strength",
        type=float,
        help="Strength of the online rest-window finger debiasing correction.",
    )
    postprocess_group.add_argument(
        "--rest-bias-min-windows",
        "--rest_bias_min_windows",
        dest="rest_bias_min_windows",
        type=int,
        help="Number of rest windows required before finger debiasing activates.",
    )
    quality_group = p.add_argument_group("live signal quality")
    quality_toggle = quality_group.add_mutually_exclusive_group()
    quality_toggle.add_argument(
        "--live-quality-enabled",
        "--live_quality_enabled",
        dest="live_quality_enabled",
        action="store_true",
        help="Enable live-only clipping, channel masking, and quality gating.",
    )
    quality_toggle.add_argument(
        "--no-live-quality",
        "--no_live_quality",
        dest="live_quality_enabled",
        action="store_false",
        help="Disable the live-only signal quality sanitizer.",
    )
    quality_group.add_argument(
        "--input-clip-abs-z",
        "--input_clip_abs_z",
        dest="input_clip_abs_z",
        type=float,
        help="Clip normalized live inputs to +/- this absolute z-score.",
    )
    quality_group.add_argument(
        "--bad-channel-rms-z",
        "--bad_channel_rms_z",
        dest="bad_channel_rms_z",
        type=float,
        help="Mark a channel bad when its normalized RMS exceeds this threshold.",
    )
    quality_group.add_argument(
        "--bad-channel-abs-p95-z",
        "--bad_channel_abs_p95_z",
        dest="bad_channel_abs_p95_z",
        type=float,
        help="Mark a channel bad when its normalized abs p95 exceeds this threshold.",
    )
    quality_group.add_argument(
        "--bad-channel-clipped-frac",
        "--bad_channel_clipped_frac",
        dest="bad_channel_clipped_frac",
        type=float,
        help="Mark a channel bad when its clipped fraction exceeds this threshold.",
    )
    quality_group.add_argument(
        "--bad-window-clipped-frac",
        "--bad_window_clipped_frac",
        dest="bad_window_clipped_frac",
        type=float,
        help="Skip actuation when total clipped fraction exceeds this threshold.",
    )
    quality_group.add_argument(
        "--bad-window-max-masked-channels",
        "--bad_window_max_masked_channels",
        dest="bad_window_max_masked_channels",
        type=int,
        help="Maximum bad-channel count that can be masked instead of quality-gating the window.",
    )
    postprocess_group.add_argument(
        "--use-inference-engine",
        dest="use_inference_engine",
        action="store_true",
        help="Use utils.inference.InferenceEngine for MC-dropout mean probabilities and uncertainty.",
    )
    postprocess_group.add_argument(
        "--mc-passes",
        dest="mc_passes",
        type=int,
        help="Monte Carlo dropout passes when --use-inference-engine is enabled.",
    )
    postprocess_group.add_argument(
        "--uncertainty-base-threshold",
        dest="uncertainty_base_threshold",
        type=float,
        help="Base action threshold used for adaptive uncertainty gating.",
    )
    postprocess_group.add_argument(
        "--uncertainty-weight",
        dest="uncertainty_weight",
        type=float,
        help="Weight applied to action uncertainty for adaptive actuation gating.",
    )

    # New: actuation knobs
    actuation_group = p.add_argument_group("actuation")
    actuation_group.add_argument(
        "--enable-actuation",
        "--enable_actuation",
        dest="enable_actuation",
        action="store_true",
        help="Enable sending commands to the Arduino-controlled hand.",
    )
    actuation_group.add_argument(
        "--serial-port",
        "--serial_port",
        dest="serial_port",
        type=str,
        metavar="PORT",
        help="Serial port to use. Auto-detected when omitted and actuation is enabled.",
    )
    actuation_group.add_argument(
        "--serial-baud",
        "--serial_baud",
        dest="serial_baud",
        type=int,
        help="Serial baud rate. Must match the Arduino sketch.",
    )
    actuation_group.add_argument(
        "--actuation-min-prob",
        "--actuation_min_prob",
        dest="actuation_min_prob",
        type=float,
        help="Minimum joint confidence required before a command is sent.",
    )
    actuation_group.add_argument(
        "--actuation-stability",
        "--actuation_stability",
        dest="actuation_stability",
        type=int,
        help="Require the same decision for N consecutive windows before actuating.",
    )
    actuation_group.add_argument(
        "--actuation-cooldown-ms",
        "--actuation_cooldown_ms",
        dest="actuation_cooldown_ms",
        type=int,
        metavar="MS",
        help="Minimum time, in milliseconds, between actuation commands.",
    )
    actuation_group.add_argument(
        "--actuation-repeat-ms",
        "--actuation_repeat_ms",
        dest="actuation_repeat_ms",
        type=int,
        metavar="MS",
        help="Milliseconds after which the same stable command may be resent.",
    )
    actuation_group.add_argument(
        "--actuation-min-speed",
        "--actuation_min_speed",
        dest="actuation_min_speed",
        type=float,
        help="Minimum non-zero speed scalar to use for any actuated command.",
    )
    actuation_group.add_argument(
        "--modulate-actuation-speed",
        dest="modulate_actuation_speed",
        action="store_true",
        help="Scale actuation speed from prediction confidence.",
    )
    actuation_group.add_argument(
        "--actuation-speed-gamma",
        dest="actuation_speed_gamma",
        type=float,
        help="Gamma curve applied to confidence-based actuation speed.",
    )
    actuation_group.add_argument(
        "--bluetooth-target",
        dest="bluetooth_target",
        type=str,
        metavar="NAME",
        help="Compatibility option for the UI connector. Ignored by the inference script itself.",
    )
    output_group = p.add_argument_group("outputs")
    output_group.add_argument(
        "--pred-log",
        type=str,
        metavar="PATH",
        help="Optional JSONL path override for per-window prediction logs.",
    )
    output_group.add_argument(
        "--allow-outside-base",
        "--allow_outside_base",
        action="store_true",
        help="Allow the output directory to live outside the session/config base directory.",
    )
    output_group.add_argument(
        "--no-file-io",
        "--no_file_io",
        dest="no_file_io",
        action="store_true",
        help="Disable all file outputs, including raw shards and the live log file.",
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
        text = str(device_override).strip().lower()
        if text and text != "auto":
            return torch.device(text)
    # Step 7 runs single-window inference. On Apple silicon that latency-sensitive
    # path is materially faster on CPU than MPS for this model, so keep "auto"
    # CPU-first here and reserve MPS for explicit opt-in benchmarking.
    if sys.platform == "darwin" and getattr(torch.backends, "mps", None) is not None:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _safe_lsl_attr(callable_obj) -> str:
    try:
        value = callable_obj()
    except Exception:
        return ""
    if value is None:
        return ""
    return str(value).strip()


def _is_default_infer_artifact_path(path_value: Optional[str], filename: str) -> bool:
    if not path_value:
        return True
    normalized = str(path_value).strip().replace("\\", "/")
    return normalized in {
        filename,
        f"models/{filename}",
        f"./{filename}",
        f"./models/{filename}",
    }


def _resolve_latest_run_dir_across_subject_sessions(
    repo_root: Path,
    project_name: Optional[str],
    subject_id: Optional[str],
    *,
    exclude_session_dir: Optional[Path] = None,
) -> Optional[tuple[Path, Path]]:
    if not project_name or not subject_id:
        return None
    sessions_root = (
        repo_root / "Projects" / str(project_name) / "subjects" / str(subject_id) / "sessions"
    )
    if not sessions_root.exists():
        return None
    best_pair: Optional[tuple[Path, Path]] = None
    best_mtime = float("-inf")
    for session_dir in sessions_root.iterdir():
        if not session_dir.is_dir():
            continue
        try:
            if exclude_session_dir is not None and session_dir.resolve() == exclude_session_dir.resolve():
                continue
        except Exception:
            pass
        run_dir = resolve_latest_run_dir(session_dir)
        if run_dir is None or not run_dir.exists():
            continue
        try:
            score = run_dir.stat().st_mtime
        except Exception:
            score = float("-inf")
        if score > best_mtime:
            best_mtime = score
            best_pair = (session_dir, run_dir)
    return best_pair


def _stream_source_id(info: Any) -> str:
    getter = getattr(info, "source_id", None)
    if getter is None:
        return ""
    return _safe_lsl_attr(getter)


def _format_lsl_stream(info: Any) -> str:
    parts = [
        f"name={_safe_lsl_attr(getattr(info, 'name', lambda: ''))}",
        f"type={_safe_lsl_attr(getattr(info, 'type', lambda: ''))}",
    ]
    try:
        parts.append(f"ch={int(info.channel_count())}")
    except Exception:
        pass
    try:
        parts.append(f"rate={float(info.nominal_srate())}")
    except Exception:
        pass
    source_id = _stream_source_id(info)
    uid = _safe_lsl_attr(getattr(info, "uid", lambda: ""))
    if source_id:
        parts.append(f"source_id={source_id}")
    if uid:
        parts.append(f"uid={uid}")
    return ", ".join(parts)


def _serial_port_score(port: Any) -> int:
    device = str(getattr(port, "device", "") or "")
    text = " ".join(
        str(getattr(port, attr, "") or "")
        for attr in ("name", "description", "manufacturer", "product", "interface")
    ).lower()
    device_l = device.lower()
    score = 0
    if "bluetooth" in text or "bluetooth" in device_l:
        score -= 200
    if "debug-console" in device_l or "incoming-port" in device_l:
        score -= 250
    if "arduino" in text:
        score += 200
    if "usbmodem" in device_l:
        score += 140
    if "usbserial" in device_l:
        score += 120
    if "wch" in text or "ch340" in text:
        score += 100
    if "cp210" in text or "silicon labs" in text:
        score += 100
    if "ftdi" in text:
        score += 100
    if "usb serial" in text:
        score += 80
    if device_l.startswith("/dev/cu."):
        score += 10
    vid = getattr(port, "vid", None)
    pid = getattr(port, "pid", None)
    if vid is not None and pid is not None:
        score += 10
    return score


def _choose_auto_serial_port(ports: list[Any]) -> Optional[str]:
    if not ports:
        return None
    scored = []
    for port in ports:
        device = str(getattr(port, "device", "") or "").strip()
        if not device:
            continue
        scored.append((_serial_port_score(port), device))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    if len(scored) == 1:
        return scored[0][1] if scored[0][0] > -100 else None
    if scored[0][0] <= 0:
        return None
    if scored[0][0] == scored[1][0]:
        return None
    return scored[0][1]


def _autodetect_serial_port() -> str:
    try:
        from serial.tools import list_ports  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyserial is required for auto-detecting the Arduino serial port. "
            "Install with: pip install pyserial"
        ) from exc

    ports = list(list_ports.comports())
    chosen = _choose_auto_serial_port(ports)
    if chosen:
        return chosen
    available = ", ".join(str(getattr(port, "device", "") or "?") for port in ports) or "(none)"
    raise RuntimeError(
        "Unable to auto-detect Arduino serial port. "
        f"Available ports: {available}. Pass --serial_port explicitly if needed."
    )


def _resolve_lsl_inlet(
    name: str,
    type_: str,
    timeout_s: float = 5.0,
    source_id: Optional[str] = None,
) -> StreamInlet:
    if not LSL_AVAILABLE or StreamInlet is None or (
        resolve_streams is None and resolve_byprop is None
    ):
        raise RuntimeError("pylsl is required for live inference.")
    timeout_s = max(0.1, float(timeout_s))
    desired_source_id = str(source_id or os.environ.get("LSL_SOURCE_ID") or "").strip()
    logger.info(
        "Resolving LSL stream name=%s type=%s source_id=%s timeout=%.1fs",
        name,
        type_,
        desired_source_id or "-",
        timeout_s,
    )
    deadline = time.monotonic() + timeout_s
    last_seen: list[Any] = []
    while True:
        remaining = max(0.0, deadline - time.monotonic())
        query_wait = min(0.5, remaining)
        all_streams: list[Any] = []
        if resolve_streams is not None:
            try:
                all_streams = list(resolve_streams(wait_time=query_wait))
            except TypeError:
                try:
                    all_streams = list(resolve_streams(timeout=query_wait))
                except TypeError:
                    all_streams = list(resolve_streams())
        elif resolve_byprop is not None:
            all_streams = list(resolve_byprop("name", name, timeout=query_wait))
        last_seen = all_streams

        candidates: list[Any] = []
        for stream in all_streams:
            try:
                if name and str(stream.name()) != str(name):
                    continue
                if type_ and str(stream.type()) != str(type_):
                    continue
            except Exception:
                continue
            candidates.append(stream)

        if desired_source_id:
            exact_source = [s for s in candidates if _stream_source_id(s) == desired_source_id]
            if exact_source:
                candidates = exact_source
            elif not candidates:
                fallback = []
                for stream in all_streams:
                    try:
                        if type_ and str(stream.type()) != str(type_):
                            continue
                    except Exception:
                        continue
                    if _stream_source_id(stream) == desired_source_id:
                        fallback.append(stream)
                if fallback:
                    candidates = fallback

        if candidates:
            candidates = sorted(
                candidates,
                key=lambda stream: (
                    1 if desired_source_id and _stream_source_id(stream) == desired_source_id else 0,
                    1 if name and _safe_lsl_attr(stream.name) == str(name) else 0,
                    1 if type_ and _safe_lsl_attr(stream.type) == str(type_) else 0,
                    float(getattr(stream, "nominal_srate", lambda: 0.0)() or 0.0),
                    float(getattr(stream, "channel_count", lambda: 0)() or 0),
                ),
                reverse=True,
            )
            chosen = candidates[0]
            inlet = StreamInlet(chosen, max_chunklen=64)
            try:
                sample, ts = inlet.pull_sample(timeout=min(0.25, max(0.05, remaining or 0.25)))
            except Exception:
                sample, ts = None, None
            if sample is not None and ts is not None:
                logger.info("Resolved LSL stream: %s", _format_lsl_stream(chosen))
                return inlet
            logger.info(
                "LSL stream resolved but not yet producing samples; retrying: %s",
                _format_lsl_stream(chosen),
            )

        if remaining <= 0.0:
            break
        time.sleep(min(0.25, remaining))

    suffix = ""
    if last_seen:
        rendered = "; ".join(_format_lsl_stream(stream) for stream in last_seen[:8])
        suffix = f" Available streams: {rendered}"
    raise RuntimeError(
        f"No LSL streams found for name={name} type={type_} "
        f"source_id={desired_source_id or '-'} within {timeout_s:.1f}s.{suffix}"
    )


def _choose_actuation(
    finger_probs: torch.Tensor,
    action_probs: torch.Tensor,
) -> ActuationDecision:
    action_probs_np = action_probs.detach().cpu().numpy()
    finger_probs_np = finger_probs.detach().cpu().numpy()
    pred_action, pred_finger = decode_prediction_pair(action_probs_np, finger_probs_np)
    # Joint confidence heuristic: min of the two max probs
    conf = float(
        min(
            float(finger_confidence_for_id(finger_probs_np, pred_finger)),
            float(action_probs_np[pred_action]),
        )
    )
    return ActuationDecision(finger_id=pred_finger, action_id=pred_action, prob=conf)


def _postprocess_decision(
    action_probs: np.ndarray,
    finger_probs: np.ndarray,
    *,
    enabled: bool,
    settings: PostprocessSettings,
    state: PostprocessState,
    finger_applicable_prob: Optional[float] = None,
) -> dict:
    if not enabled:
        raw_action = int(np.argmax(action_probs)) if action_probs.size else 0
        raw_finger = decode_finger_prediction(finger_probs)
        committed_action, committed_finger = decode_prediction_pair(
            action_probs, finger_probs
        )
        action_conf = float(np.max(action_probs)) if action_probs.size else 0.0
        finger_conf = (
            finger_confidence_for_id(finger_probs, committed_finger)
            if finger_probs.size
            else 0.0
        )
        finger_gate_ok = bool(
            committed_action == 0 or finger_conf >= float(settings.threshold_finger)
        )
        applicability_gate_ok = bool(
            committed_action == 0
            or finger_applicable_prob is None
            or float(finger_applicable_prob) >= float(settings.threshold_applicability)
        )
        return {
            "committed_action_id": committed_action,
            "committed_finger_id": committed_finger,
            "raw_top_action_id": raw_action,
            "raw_top_finger_id": raw_finger,
            "action_conf": action_conf,
            "finger_conf": finger_conf,
            "finger_gate_ok": finger_gate_ok,
            "finger_applicable_prob": (
                float(finger_applicable_prob)
                if finger_applicable_prob is not None
                else None
            ),
            "applicability_gate_ok": applicability_gate_ok,
            "committed_pair_valid": bool(
                is_valid_action_finger(committed_action, committed_finger)
            ),
            "smoothed_action_id": committed_action,
            "smoothed_finger_id": committed_finger,
            "decision_reason": "raw_argmax_gated",
            "frames_in_state": 1,
        }
    return postprocess_predictions(
        action_probs,
        finger_probs,
        settings,
        state,
        finger_applicable_prob=finger_applicable_prob,
    )


def _build_inference_engine(
    model: torch.nn.Module,
    scaler: object,
    device: torch.device,
    args: argparse.Namespace,
    temperature_state: Optional[TemperatureScalingState],
) -> Optional[InferenceEngine]:
    if not bool(getattr(args, "use_inference_engine", False)):
        return None
    config = InferenceConfig(
        base_threshold=float(args.uncertainty_base_threshold),
        uncertainty_weight=float(args.uncertainty_weight),
        stability_frames=max(1, int(args.actuation_stability)),
        mc_passes=max(1, int(args.mc_passes)),
    )
    return InferenceEngine(
        model=model,
        normalizer=scaler,
        device=device,
        action_names={},
        finger_names={},
        config=config,
        temperature_state=temperature_state,
    )


def _build_direct_inference_engine(
    model: torch.nn.Module,
    scaler: object,
    device: torch.device,
    temperature_state: Optional[TemperatureScalingState],
) -> Optional[InferenceEngine]:
    if not hasattr(model, "to"):
        return None
    return InferenceEngine(
        model=model,
        normalizer=scaler,
        device=device,
        action_names={},
        finger_names={},
        config=InferenceConfig(mc_passes=1),
        temperature_state=temperature_state,
    )


def _predict_window(
    window: np.ndarray,
    *,
    scaler: object,
    model: torch.nn.Module,
    device: torch.device,
    inference_engine: Optional[InferenceEngine],
    direct_engine: Optional[InferenceEngine] = None,
    temperature_state: Optional[TemperatureScalingState] = None,
    emit_viz: bool,
    prepared_window: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    window_f32 = np.asarray(window, dtype=np.float32)
    model_window_f32 = (
        np.asarray(prepared_window, dtype=np.float32)
        if prepared_window is not None
        else window_f32
    )
    model_window_is_normalized = prepared_window is not None
    hidden_mag: Optional[float] = None
    live_viz_payload: Optional[dict[str, Any]] = None

    if inference_engine is None:
        if direct_engine is not None:
            _, x = direct_engine.prepare_input(
                model_window_f32,
                normalized=model_window_is_normalized,
            )
            finger_probs_t, action_probs_t, applicability_prob_t = (
                direct_engine.forward_probabilities(x)
            )
            action_probs_t = action_probs_t.squeeze(0)
            finger_probs_t = finger_probs_t.squeeze(0)
            applicability_prob_t = (
                applicability_prob_t.squeeze(0)
                if applicability_prob_t is not None
                else None
            )
            if emit_viz:
                live_viz_payload, hidden_mag = _build_live_viz_payload(model, x)
        else:
            window_input = (
                model_window_f32
                if model_window_is_normalized
                else standardize_window_TxC(window_f32, scaler)
            )
            x = torch.from_numpy(window_input).unsqueeze(0).to(device)
            with torch.inference_mode():
                finger_logits, action_logits, applicability_logits = unpack_model_outputs(
                    model(x)
                )
                finger_logits = apply_temperature_to_logits(
                    finger_logits,
                    temperature_state.finger_temperature if temperature_state is not None else 1.0,
                )
                action_logits = apply_temperature_to_logits(
                    action_logits,
                    temperature_state.action_temperature if temperature_state is not None else 1.0,
                )
                if applicability_logits is not None:
                    applicability_logits = apply_temperature_to_logits(
                        applicability_logits,
                        temperature_state.applicability_temperature
                        if temperature_state is not None
                        else 1.0,
                    )
                action_probs_t = torch.softmax(action_logits, dim=1).squeeze(0)
                finger_probs_t = torch.softmax(finger_logits, dim=1).squeeze(0)
                applicability_prob_t = (
                    torch.sigmoid(applicability_logits).squeeze(0)
                    if applicability_logits is not None
                    else None
                )
                if emit_viz:
                    live_viz_payload, hidden_mag = _build_live_viz_payload(model, x)
        return {
            "backend": "direct",
            "action_probs": action_probs_t.detach().cpu().numpy(),
            "finger_probs": finger_probs_t.detach().cpu().numpy(),
            "finger_applicable_prob": (
                float(applicability_prob_t.detach().cpu().item())
                if applicability_prob_t is not None
                else None
            ),
            "action_uncertainty": 0.0,
            "finger_uncertainty": 0.0,
            "applicability_uncertainty": None,
            "adaptive_threshold": None,
            "health_score": None,
            "hidden_mag": hidden_mag,
            "live_viz_payload": live_viz_payload,
        }

    (
        action_probs,
        finger_probs,
        action_uncertainty,
        finger_uncertainty,
        diagnostics,
    ) = inference_engine.predict_proba(
        model_window_f32,
        normalized=model_window_is_normalized,
    )
    if action_probs is None or finger_probs is None:
        raise RuntimeError("InferenceEngine returned empty probabilities for a loaded model.")

    if emit_viz:
        _, x = inference_engine.prepare_input(
            model_window_f32,
            normalized=model_window_is_normalized,
        )
        live_viz_payload, hidden_mag = _build_live_viz_payload(model, x)

    adaptive_threshold = min(
        0.99,
        max(
            float(inference_engine.config.base_threshold),
            float(inference_engine.config.base_threshold)
            + float(inference_engine.config.uncertainty_weight)
            * float(action_uncertainty),
        ),
    )
    return {
        "backend": "inference_engine",
        "action_probs": action_probs,
        "finger_probs": finger_probs,
        "finger_applicable_prob": diagnostics.get("finger_applicable_prob"),
        "action_uncertainty": float(action_uncertainty),
        "finger_uncertainty": float(finger_uncertainty),
        "applicability_uncertainty": diagnostics.get("applicability_uncertainty"),
        "adaptive_threshold": float(adaptive_threshold),
        "health_score": diagnostics.get("health_score") if isinstance(diagnostics, dict) else None,
        "hidden_mag": hidden_mag,
        "live_viz_payload": live_viz_payload,
    }


def _compute_saliency(model: CNNLSTMFingerActionNet, x: torch.Tensor) -> Optional[np.ndarray]:
    try:
        x_grad = x.detach().clone().requires_grad_(True)
        _, action_logits, _ = unpack_model_outputs(model(x_grad))
        target_idx = int(torch.argmax(action_logits, dim=1).item())
        loss = action_logits[0, target_idx]
        model.zero_grad(set_to_none=True)
        loss.backward()
        grad = x_grad.grad
        if grad is None:
            return None
        return np.abs(grad.detach().cpu().numpy()[0])
    except Exception:
        return None


def _compute_live_viz_arrays(
    model: CNNLSTMFingerActionNet, x: torch.Tensor
) -> tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[float],
]:
    try:
        with torch.inference_mode():
            z = x.permute(0, 2, 1)
            feature_map_t = model.conv(z)
            lstm_in = feature_map_t.permute(0, 2, 1)
            out, _ = model.lstm(lstm_in)
            hidden_t = torch.linalg.norm(out, dim=2).squeeze(0)
            head_out = model.head_dropout(out)
            finger_logits = model.finger_head(head_out)
            action_logits = model.action_head(head_out)
            feature_map = feature_map_t.squeeze(0).detach().cpu().numpy()
            hidden_timeline = hidden_t.detach().cpu().numpy()
            finger_probs = (
                torch.softmax(finger_logits, dim=2).squeeze(0).detach().cpu().numpy()
            )
            action_probs = (
                torch.softmax(action_logits, dim=2).squeeze(0).detach().cpu().numpy()
            )
        hidden_mag = None
        if hidden_timeline.size:
            value = float(hidden_timeline[-1])
            if np.isfinite(value):
                hidden_mag = value
        return feature_map, hidden_timeline, finger_probs, action_probs, hidden_mag
    except Exception:
        return None, None, None, None, None


def _build_live_viz_payload(
    model: CNNLSTMFingerActionNet,
    x: torch.Tensor,
) -> tuple[Optional[dict[str, Any]], Optional[float]]:
    (
        feature_map,
        hidden_timeline,
        finger_probs,
        action_probs,
        hidden_mag,
    ) = _compute_live_viz_arrays(model, x)
    saliency = _compute_saliency(model, x)
    if (
        feature_map is None
        and hidden_timeline is None
        and finger_probs is None
        and action_probs is None
        and saliency is None
        and hidden_mag is None
    ):
        return None, hidden_mag
    return (
        {
            "hidden_mag": float(hidden_mag) if hidden_mag is not None else None,
            "feature_map": feature_map.tolist() if feature_map is not None else None,
            "hidden_timeline": (
                hidden_timeline.tolist() if hidden_timeline is not None else None
            ),
            "finger_probs": finger_probs.tolist() if finger_probs is not None else None,
            "action_probs": action_probs.tolist() if action_probs is not None else None,
            "saliency": saliency.tolist() if saliency is not None else None,
        },
        hidden_mag,
    )


def _debounced_should_send(
    decision: ActuationDecision,
    last_sent: Optional[Tuple[int, int]],
    stable_count: int,
    required_stability: int,
    last_send_ts: float,
    cooldown_ms: int,
    repeat_same_ms: int = 0,
) -> bool:
    last_send_time_ms = (
        None if float(last_send_ts) <= 0.0 else float(last_send_ts) * 1000.0
    )
    current_time_ms = float(time.monotonic()) * 1000.0
    return _shared_debounced_should_send(
        decision,
        last_sent=last_sent,
        stable_count=stable_count,
        required_stability=required_stability,
        last_send_time_ms=last_send_time_ms,
        current_time_ms=current_time_ms,
        cooldown_ms=cooldown_ms,
        repeat_same_ms=repeat_same_ms,
    )


def _uncertainty_gate_passed(
    decision_info: dict[str, Any],
    inference_result: dict[str, Any],
) -> bool:
    return _shared_uncertainty_gate_passed(decision_info, inference_result)


def _finger_gate_passed(decision_info: dict[str, Any]) -> bool:
    return _shared_finger_gate_passed(decision_info)


def _build_actuation_speed_mapper(args: argparse.Namespace) -> Optional[CommandShaper]:
    return _shared_build_actuation_speed_mapper(
        modulate_actuation_speed=bool(getattr(args, "modulate_actuation_speed", True)),
        actuation_speed_gamma=float(args.actuation_speed_gamma),
    )


def _compute_actuation_speed_scalar(
    decision_prob: float,
    action_uncertainty: float,
    speed_mapper: Optional[CommandShaper],
    min_speed: float = 0.0,
) -> float:
    return _shared_compute_actuation_speed_scalar(
        decision_prob,
        action_uncertainty,
        speed_mapper,
        min_speed=min_speed,
    )


def _build_actuation_command_shaper(args: argparse.Namespace) -> CommandShaper:
    return _shared_build_actuation_command_shaper(
        actuation_min_prob=float(args.actuation_min_prob),
        actuation_speed_gamma=float(args.actuation_speed_gamma),
        hop_sec=float(args.hop_sec),
        actuation_stability=int(args.actuation_stability),
        actuation_cooldown_ms=int(args.actuation_cooldown_ms),
    )


def _estimate_window_center_mono(
    *,
    latest_sample_mono: Optional[float],
    latest_stream_time_s: float,
    window_center_stream_s: float,
    fallback_mono: Optional[float] = None,
) -> float:
    if latest_sample_mono is None:
        if fallback_mono is not None:
            return float(fallback_mono)
        return time.monotonic()
    stream_delta_s = float(latest_stream_time_s) - float(window_center_stream_s)
    if not np.isfinite(stream_delta_s):
        return float(latest_sample_mono)
    if stream_delta_s < 0.0:
        stream_delta_s = 0.0
    return float(latest_sample_mono) - stream_delta_s


def _latency_gate_passed(latency_ms: float, threshold_ms: float) -> bool:
    return _shared_latency_gate_passed(latency_ms, threshold_ms)


def _require_deployable_run(run_dir: Path) -> dict[str, Any]:
    return _shared_require_deployable_run(run_dir)


def _resolve_actuation_candidate(
    history: Deque[ActuationDecision],
    *,
    required_finger_stability: int,
) -> dict[str, Any]:
    return _shared_resolve_actuation_candidate(
        history,
        required_finger_stability=required_finger_stability,
    )


def _resolve_live_actuation_vote(
    history: Deque[ActuationDecision],
    decision: ActuationDecision,
    *,
    required_pair_stability: int,
    ignore_window: bool,
    ignore_reason: str = "quality_gate",
) -> dict[str, Any]:
    if ignore_window:
        return {
            "decision": ActuationDecision(finger_id=0, action_id=0, prob=0.0),
            "reason": str(ignore_reason),
            "finger_votes": {},
            "action_votes": {},
            "pair_votes": {},
            "resolved_finger_id": 0,
            "history_appended": False,
        }
    history.append(decision)
    out = _resolve_actuation_candidate(
        history,
        required_finger_stability=required_pair_stability,
    )
    out["history_appended"] = True
    return out


def _stringify_counter(counter: collections.Counter[Any]) -> dict[str, int]:
    return {str(key): int(value) for key, value in counter.items()}


def _top_counter_snapshot(
    counter: collections.Counter[Any], *, top_k: int = 2
) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in counter.most_common(max(0, int(top_k)))
    }


def _pair_key(finger_id: Any, action_id: Any) -> str:
    return f"{int(finger_id)}:{int(action_id)}"


def _load_prediction_records(pred_log_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not pred_log_path.exists():
        return records
    with pred_log_path.open("r") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                payload = json.loads(text)
            except Exception:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return records


def _compute_raw_channel_stats(raw_dir: Optional[Path]) -> Optional[dict[str, Any]]:
    if raw_dir is None or not raw_dir.exists():
        return None
    per_channel_values: list[list[np.ndarray]] = []
    rows = 0
    nonfinite_rows = 0
    nonfinite_values = 0
    flagged_nonfinite_rows = 0
    shard_count = 0
    for path in sorted(raw_dir.glob("*.npy")):
        shard = np.load(path, allow_pickle=False)
        samples = np.asarray(shard["sample"], dtype=float)
        if samples.ndim != 2:
            continue
        if not per_channel_values:
            per_channel_values = [[] for _ in range(samples.shape[1])]
        shard_count += 1
        rows += int(samples.shape[0])
        finite_rows = np.all(np.isfinite(samples), axis=1)
        nonfinite_rows += int((~finite_rows).sum())
        nonfinite_values += int(np.size(samples) - np.isfinite(samples).sum())
        if shard.dtype.names is not None and "flags" in shard.dtype.names:
            flagged_nonfinite_rows += int(((np.asarray(shard["flags"], dtype=int) & RAW_FLAG_NONFINITE) != 0).sum())
        for ch_idx in range(samples.shape[1]):
            values = samples[:, ch_idx]
            finite = values[np.isfinite(values)]
            if finite.size:
                per_channel_values[ch_idx].append(finite.astype(float, copy=False))

    channel_stats: list[dict[str, Any]] = []
    for channel_id, chunks in enumerate(per_channel_values):
        if not chunks:
            channel_stats.append(
                {
                    "channel_id": int(channel_id),
                    "count": 0,
                    "mean": None,
                    "std": None,
                    "abs_p95": None,
                }
            )
            continue
        values = np.concatenate(chunks)
        channel_stats.append(
            {
                "channel_id": int(channel_id),
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "abs_p95": float(np.percentile(np.abs(values), 95)),
            }
        )

    return {
        "shard_count": int(shard_count),
        "rows": int(rows),
        "nonfinite_rows": int(nonfinite_rows),
        "nonfinite_values": int(nonfinite_values),
        "flagged_nonfinite_rows": int(flagged_nonfinite_rows),
        "channels": channel_stats,
    }


def _build_live_prediction_summary(
    *,
    pred_log_path: Path,
    summary_path: Path,
    raw_dir: Optional[Path],
    dropped_windows: int,
    dropped_nonfinite_samples: int,
    dropped_nonfinite_windows: int,
) -> None:
    records = _load_prediction_records(pred_log_path)
    if not records:
        summary_path.write_text(
            json.dumps(
                {
                    "record_count": 0,
                    "dropped_windows": int(dropped_windows),
                    "dropped_nonfinite_samples": int(dropped_nonfinite_samples),
                    "dropped_nonfinite_windows": int(dropped_nonfinite_windows),
                    "raw_channel_stats": _compute_raw_channel_stats(raw_dir),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return

    committed_pairs = [
        (
            int(row.get("committed_finger_id", 0) or 0),
            int(row.get("committed_action_id", 0) or 0),
        )
        for row in records
    ]
    sent_pairs = [
        (
            int(row.get("actuation_target_finger_id", 0) or 0),
            int(row.get("actuation_target_action_id", 0) or 0),
        )
        for row in records
        if bool(row.get("actuation_sent"))
    ]
    committed_transitions = sum(
        1 for prev, cur in zip(committed_pairs, committed_pairs[1:]) if prev != cur
    )
    sent_transitions = sum(
        1 for prev, cur in zip(sent_pairs, sent_pairs[1:]) if prev != cur
    )

    segments: list[dict[str, Any]] = []
    seg_start = 0
    prev_pair = committed_pairs[0]
    for idx, pair in enumerate(committed_pairs[1:], start=1):
        if pair != prev_pair:
            if prev_pair != (0, 0):
                segments.append(
                    {
                        "finger_id": int(prev_pair[0]),
                        "action_id": int(prev_pair[1]),
                        "frames": int(idx - seg_start),
                        "start_s": float(records[seg_start].get("window_start_s", 0.0)),
                        "end_s": float(records[idx - 1].get("window_end_s", 0.0)),
                    }
                )
            seg_start = idx
            prev_pair = pair
    if prev_pair != (0, 0):
        segments.append(
            {
                "finger_id": int(prev_pair[0]),
                "action_id": int(prev_pair[1]),
                "frames": int(len(records) - seg_start),
                "start_s": float(records[seg_start].get("window_start_s", 0.0)),
                "end_s": float(records[-1].get("window_end_s", 0.0)),
            }
        )
    segments.sort(key=lambda item: int(item["frames"]), reverse=True)

    masked_channel_counts: collections.Counter[int] = collections.Counter()
    for row in records:
        for channel_id in row.get("masked_channel_ids", []) or []:
            masked_channel_counts[int(channel_id)] += 1

    summary = {
        "record_count": int(len(records)),
        "raw_action_counts": _stringify_counter(
            collections.Counter(row.get("raw_top_action_id") for row in records)
        ),
        "raw_finger_counts": _stringify_counter(
            collections.Counter(row.get("raw_top_finger_id") for row in records)
        ),
        "committed_action_counts": _stringify_counter(
            collections.Counter(int(row.get("committed_action_id", 0) or 0) for row in records)
        ),
        "committed_finger_counts": _stringify_counter(
            collections.Counter(int(row.get("committed_finger_id", 0) or 0) for row in records)
        ),
        "actuation_sent_pair_counts": _stringify_counter(
            collections.Counter(_pair_key(fid, aid) for fid, aid in sent_pairs)
        ),
        "actuation_suppressed_counts": _stringify_counter(
            collections.Counter(
                str(row.get("actuation_suppressed_reason") or "none")
                for row in records
            )
        ),
        "actuation_vote_reason_counts": _stringify_counter(
            collections.Counter(str(row.get("actuation_vote_reason") or "none") for row in records)
        ),
        "window_quality_bad_count": int(
            sum(bool(row.get("window_quality_bad")) for row in records)
        ),
        "quality_bad_reason_counts": _stringify_counter(
            collections.Counter(str(row.get("quality_bad_reason") or "none") for row in records)
        ),
        "masked_window_count": int(
            sum(bool(row.get("masked_channel_ids")) for row in records)
        ),
        "masked_channel_counts": _stringify_counter(masked_channel_counts),
        "pair_transition_rate": float(
            committed_transitions / max(1, len(committed_pairs) - 1)
        ),
        "actuation_sent_pair_transition_rate": float(
            sent_transitions / max(1, len(sent_pairs) - 1)
        ),
        "longest_committed_non_rest_segments": segments[:10],
        "dropped_windows": int(dropped_windows),
        "dropped_nonfinite_samples": int(dropped_nonfinite_samples),
        "dropped_nonfinite_windows": int(dropped_nonfinite_windows),
        "raw_channel_stats": _compute_raw_channel_stats(raw_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))


# -------------------- Main --------------------

def main() -> int:
    parser, defaults = _build_arg_parser()
    args = parser.parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config_payload, config_settings = _load_config_file(config_path)
    _apply_config_to_args(args, config_settings, defaults)

    # Required config keys (as in original file)
    lsl_name = (
        args.stream_name
        or config_settings.get("lsl_name")
        or config_settings.get("stream_name")
        or "Muse2-EEG"
    )
    lsl_type = (
        args.stream_type
        or config_settings.get("lsl_type")
        or config_settings.get("stream_type")
        or "EEG"
    )
    lsl_source_id = (
        config_settings.get("lsl_source_id")
        or config_settings.get("LSL_SOURCE_ID")
        or os.environ.get("LSL_SOURCE_ID")
    )
    try:
        lsl_resolve_timeout_s = float(config_settings.get("LSL_RESOLVE_TIMEOUT", 25.0))
    except Exception:
        lsl_resolve_timeout_s = 25.0
    session_dir_value = args.session_dir or config_settings.get("session_dir")
    repo_root = _resolve_repo_root(config_path)
    project_name, subject_id = _derive_project_subject(
        config_payload, config_path, args.project_name, args.subject_id, config_settings
    )
    session_dir_inferred = False
    if not session_dir_value and project_name and subject_id:
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
    model_path = args.model_path or config_settings.get("model_path")
    scaler_path = args.scaler_path or config_settings.get("scaler_path")
    out_dir = args.out_dir or config_settings.get("out_dir")

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
        resolved_model_override = None
        resolved_scaler_override = None
        explicit_overrides = []
        if model_path:
            resolved_candidate = _resolve_path(str(model_path), config_path.parent)
            if Path(resolved_candidate).exists():
                resolved_model_override = resolved_candidate
                explicit_overrides.append("model_path")
            elif args.model_path:
                print("Session selection source: session_dir")
                print(f"Model path not found: {resolved_candidate}")
                return 2
            elif not _is_default_infer_artifact_path(str(model_path), "finger_action_model.pt"):
                print(
                    f"⚠️ Config model_path not found; falling back to latest session model: {resolved_candidate}"
                )
        if scaler_path:
            resolved_candidate = _resolve_path(str(scaler_path), config_path.parent)
            if Path(resolved_candidate).exists():
                resolved_scaler_override = resolved_candidate
                explicit_overrides.append("scaler_path")
            elif args.scaler_path:
                print("Session selection source: session_dir")
                print(f"Scaler path not found: {resolved_candidate}")
                return 2
            elif not _is_default_infer_artifact_path(str(scaler_path), "scaler.npz"):
                print(
                    f"⚠️ Config scaler_path not found; falling back to latest session scaler: {resolved_candidate}"
                )
        if out_dir:
            explicit_overrides.append("out_dir")
        if explicit_overrides:
            print(
                f"⚠️ Explicit paths provided with --session-dir; using overrides: {explicit_overrides}"
            )
            selection_source = "legacy_explicit"
        else:
            selection_source = "subject_latest" if session_dir_inferred else "session_dir"

        run_dir = None
        if resolved_model_override is None or resolved_scaler_override is None:
            run_dir = resolve_latest_run_dir(session_dir_path)
            if (run_dir is None or not run_dir.exists()) and (project_name and subject_id):
                fallback_pair = _resolve_latest_run_dir_across_subject_sessions(
                    _resolve_repo_root(config_path),
                    project_name,
                    subject_id,
                    exclude_session_dir=session_dir_path,
                )
                if fallback_pair is not None:
                    fallback_session_dir, fallback_run_dir = fallback_pair
                    print(
                        "⚠️ Selected session has no model run; "
                        f"using latest trained session for artifacts: {fallback_session_dir}"
                    )
                    run_dir = fallback_run_dir
        if (resolved_model_override is None or resolved_scaler_override is None) and (
            run_dir is None or not run_dir.exists()
        ):
            print("Session selection source: session_dir")
            print(
                "No model run directory found. Train a model first (Step 2), or pass explicit model_path/scaler_path."
            )
            return 2
        base_dir = session_dir_path
        if resolved_model_override is not None:
            model_path = resolved_model_override
        else:
            assert run_dir is not None
            model_path = str(run_dir / "finger_action_model.pt")
        if resolved_scaler_override is not None:
            scaler_path = resolved_scaler_override
        else:
            assert run_dir is not None
            scaler_path = str(run_dir / "scaler.npz")
        if not out_dir:
            out_dir = str(SessionLayout(session_dir_path).processed_dir / "live_infer")
        else:
            out_dir = _resolve_path(str(out_dir), config_path.parent)
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
        threshold_applicability=float(args.threshold_applicability),
        adjacency_enabled=bool(args.adjacency_enabled),
        hysteresis_margin=float(args.hysteresis_margin),
        finger_delta=float(args.finger_delta),
        finger_mode=str(args.finger_mode),
    )
    post_state = PostprocessState()
    rest_bias = RestFingerBiasCorrection(
        enabled=bool(args.rest_bias_correction_enabled),
        min_rest_windows=max(1, int(args.rest_bias_min_windows)),
        strength=float(args.rest_bias_strength),
    )
    logger.info(
        "Rest-bias correction enabled=%s strength=%.3f min_rest_windows=%s",
        bool(rest_bias.enabled),
        float(rest_bias.strength),
        int(rest_bias.min_rest_windows),
    )
    logger.info(
        "Live quality enabled=%s clip_abs_z=%.2f bad_channel_rms_z=%.2f bad_channel_abs_p95_z=%.2f bad_channel_clipped_frac=%.3f bad_window_clipped_frac=%.3f bad_window_max_masked_channels=%s",
        bool(args.live_quality_enabled),
        float(args.input_clip_abs_z),
        float(args.bad_channel_rms_z),
        float(args.bad_channel_abs_p95_z),
        float(args.bad_channel_clipped_frac),
        float(args.bad_window_clipped_frac),
        int(args.bad_window_max_masked_channels),
    )

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
    temperature_state = load_temperature_scaling(
        _resolve_temperature_path(Path(model_path).resolve().parent)
    )
    if temperature_state is not None:
        logger.info(
            "Temperature scaling loaded: action=%.4f finger=%.4f applicability=%.4f source=%s",
            float(temperature_state.action_temperature),
            float(temperature_state.finger_temperature),
            float(temperature_state.applicability_temperature),
            str(temperature_state.source),
        )
    else:
        logger.info("Temperature scaling: not found; using identity.")
    inference_engine = _build_inference_engine(
        model, scaler, device, args, temperature_state
    )
    direct_inference_engine = (
        None
        if inference_engine is not None
        else _build_direct_inference_engine(model, scaler, device, temperature_state)
    )
    actuation_speed_mapper = _build_actuation_speed_mapper(args)
    if inference_engine is not None:
        logger.info(
            "Inference backend=inference_engine mc_passes=%s uncertainty_base_threshold=%.3f uncertainty_weight=%.3f",
            args.mc_passes,
            float(args.uncertainty_base_threshold),
            float(args.uncertainty_weight),
        )
    else:
        logger.info("Inference backend=direct")
    logger.info(
        "Actuation speed modulation=%s gamma=%.3f",
        bool(args.modulate_actuation_speed),
        float(args.actuation_speed_gamma),
    )
    deploy_info = None
    if args.enable_actuation:
        deployment_run_dir = Path(model_path).expanduser().resolve().parent
        deploy_info = _require_deployable_run(deployment_run_dir)
        logger.info(
            "Deployment model validated run_dir=%s active_finger_head=%s finger_applicability_head=%s n_fingers=%s n_actions=%s",
            deployment_run_dir,
            deploy_info.get("active_finger_head"),
            deploy_info.get("finger_applicability_head"),
            deploy_info.get("n_fingers"),
            deploy_info.get("n_actions"),
        )

    live_viz_enabled = bool(getattr(args, "LIVE_VIZ_ENABLED", False))
    live_viz_fps = float(getattr(args, "LIVE_VIZ_FPS", 0.0) or 0.0)
    if live_viz_fps <= 0.0:
        live_viz_enabled = False
    live_viz_interval = (1.0 / live_viz_fps) if live_viz_enabled else 0.0
    last_live_viz_emit = 0.0

    inlet = _resolve_lsl_inlet(
        lsl_name,
        lsl_type,
        timeout_s=lsl_resolve_timeout_s,
        source_id=str(lsl_source_id) if lsl_source_id else None,
    )
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
        serial_port = args.serial_port or config_settings.get("serial_port")
        if not serial_port:
            serial_port = _autodetect_serial_port()
            logger.info("Actuation serial port auto-detected: %s", serial_port)
        actuator = SerialHandActuator(str(serial_port), baud=args.serial_baud)
        actuator.open()
        logger.info("Actuation enabled via serial port %s @ %s baud", serial_port, args.serial_baud)
        _warmup_actuation(actuator)

    # Live buffers
    from collections import deque
    buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=int(max(5, args.window_sec * args.target_fs * 4)))
    latency_window: Deque[float] = deque(maxlen=200)

    stream_origin_mono: Optional[float] = None
    stream_origin_lsl: Optional[float] = None
    prev_lsl_mono: Optional[float] = None
    latest_sample_mono: Optional[float] = None
    latest_stream_time_s = 0.0
    dropped_windows = 0
    dropped_nonfinite_samples = 0
    dropped_nonfinite_windows = 0
    quality_bad_windows = 0
    quality_masked_windows = 0
    masked_channel_counts: collections.Counter[int] = collections.Counter()
    last_masked_channel_warning: Optional[Tuple[int, int]] = None
    last_log = time.monotonic()

    next_window_start_s = 0.0

    # Debounce state
    last_sent: Optional[Tuple[int, int]] = None
    last_send_ts = 0.0
    sample_seq = 0
    actuation_history: Deque[ActuationDecision] = deque(
        maxlen=max(1, int(args.actuation_stability))
    )
    actuation_command_shaper = _build_actuation_command_shaper(args)

    termination_reason = "ok"
    try:
        while True:
            # Pull a chunk from LSL
            chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=64)
            if timestamps:
                for sample, lsl_ts in zip(chunk, timestamps):
                    sample_mono = time.monotonic()
                    latest_sample_mono = float(sample_mono)
                    (
                        time_s,
                        lsl_ts_mono,
                        clamped,
                        stream_origin_mono,
                        stream_origin_lsl,
                        prev_lsl_mono,
                    ) = _resolve_live_sample_time(
                        lsl_ts=float(lsl_ts),
                        sample_mono=float(sample_mono),
                        stream_origin_mono=stream_origin_mono,
                        stream_origin_lsl=stream_origin_lsl,
                        prev_lsl_mono=prev_lsl_mono,
                    )
                    latest_stream_time_s = max(float(latest_stream_time_s), float(time_s))
                    vec = np.asarray(sample, dtype=np.float32)
                    sample_flags = 0
                    if not np.all(np.isfinite(vec)):
                        sample_flags |= RAW_FLAG_NONFINITE
                        dropped_nonfinite_samples += 1

                    # Persist raw packets (optional)
                    if record_raw and session_writer is not None:
                        raw_buffer.append(
                            Packet(
                                seq=sample_seq,
                                lsl_ts_raw=lsl_ts,
                                lsl_ts_mono=lsl_ts_mono,
                                local_ts=time.time(),
                                sample=np.asarray(sample, dtype=float),
                                flags=sample_flags,
                                segment_id=0,
                                clamped=clamped,
                                raw_path=None,
                                segment_break_reason=None,
                            )
                        )
                        sample_seq += 1
                        if len(raw_buffer) >= raw_flush_size:
                            session_writer.append_packets(raw_buffer)
                            raw_buffer = []

                    if sample_flags & RAW_FLAG_NONFINITE:
                        continue

                    buffer.append((time_s, vec))

            # Infer over available windows
            time_s = float(latest_stream_time_s)
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
                if not np.all(np.isfinite(window_values)):
                    dropped_windows += 1
                    dropped_nonfinite_windows += 1
                    next_window_start_s += args.hop_sec
                    continue

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
                            "finger_gate_ok": True,
                            "committed_pair_valid": True,
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

                emit_viz = False
                viz_ts = None
                now_mono = time.monotonic()
                if live_viz_enabled and (now_mono - last_live_viz_emit) >= live_viz_interval:
                    emit_viz = True
                    viz_ts = float(window_end)

                quality = _sanitize_live_window(
                    window,
                    scaler=scaler,
                    enabled=bool(args.live_quality_enabled),
                    input_clip_abs_z=float(args.input_clip_abs_z),
                    bad_channel_rms_z=float(args.bad_channel_rms_z),
                    bad_channel_abs_p95_z=float(args.bad_channel_abs_p95_z),
                    bad_channel_clipped_frac=float(args.bad_channel_clipped_frac),
                    bad_window_clipped_frac=float(args.bad_window_clipped_frac),
                    bad_window_max_masked_channels=int(
                        args.bad_window_max_masked_channels
                    ),
                )
                if quality.window_quality_bad:
                    quality_bad_windows += 1
                if quality.masked_channel_ids:
                    quality_masked_windows += 1
                    for channel_id in quality.masked_channel_ids:
                        masked_channel_counts[int(channel_id)] += 1

                inference_result = _predict_window(
                    window,
                    scaler=scaler,
                    model=model,
                    device=device,
                    inference_engine=inference_engine,
                    direct_engine=direct_inference_engine,
                    temperature_state=temperature_state,
                    emit_viz=emit_viz,
                    prepared_window=quality.prepared_window,
                )
                action_probs = np.asarray(inference_result["action_probs"], dtype=float)
                model_raw_finger_probs = np.asarray(
                    inference_result["finger_probs"], dtype=float
                )
                finger_applicable_prob = inference_result.get("finger_applicable_prob")
                hidden_mag = inference_result.get("hidden_mag")
                action_uncertainty = float(
                    inference_result.get("action_uncertainty", 0.0) or 0.0
                )
                finger_uncertainty = float(
                    inference_result.get("finger_uncertainty", 0.0) or 0.0
                )
                applicability_uncertainty = inference_result.get(
                    "applicability_uncertainty"
                )
                model_raw_top_finger_id = decode_finger_prediction(model_raw_finger_probs)
                rest_bias_became_ready = rest_bias.update(
                    action_probs, model_raw_finger_probs
                )
                if rest_bias_became_ready:
                    prior = rest_bias.prior()
                    logger.info(
                        "Rest-bias correction armed rest_windows=%s prior=%s strength=%.3f",
                        int(rest_bias.rest_count),
                        (
                            np.round(np.asarray(prior, dtype=float), 4).tolist()
                            if prior is not None
                            else None
                        ),
                        float(rest_bias.strength),
                    )
                finger_probs = np.asarray(
                    rest_bias.apply(model_raw_finger_probs), dtype=float
                )
                rest_bias_applied = bool(
                    rest_bias.ready
                    and not np.allclose(
                        finger_probs,
                        model_raw_finger_probs,
                        rtol=1e-6,
                        atol=1e-8,
                        equal_nan=True,
                    )
                )

                decision_info = _postprocess_decision(
                    action_probs,
                    finger_probs,
                    enabled=postprocess_enabled,
                    settings=post_settings,
                    state=post_state,
                    finger_applicable_prob=(
                        float(finger_applicable_prob)
                        if finger_applicable_prob is not None
                        else None
                    ),
                )
                decision = ActuationDecision(
                    finger_id=int(decision_info["committed_finger_id"]),
                    action_id=int(decision_info["committed_action_id"]),
                    prob=float(min(decision_info["action_conf"], decision_info["finger_conf"])),
                )
                finger_gate_ok = _finger_gate_passed(decision_info)
                applicability_gate_ok = _shared_applicability_gate_passed(
                    decision_info
                )
                uncertainty_gate_ok = _uncertainty_gate_passed(
                    decision_info=decision_info,
                    inference_result=inference_result,
                )
                actuation_speed_scalar = _compute_actuation_speed_scalar(
                    decision.prob,
                    action_uncertainty,
                    actuation_speed_mapper,
                    min_speed=float(args.actuation_min_speed),
                )

                # Latency tracking
                now = time.monotonic()
                window_center_stream_s = window_start + args.window_sec / 2.0
                window_center_mono = _estimate_window_center_mono(
                    latest_sample_mono=latest_sample_mono,
                    latest_stream_time_s=float(latest_stream_time_s),
                    window_center_stream_s=float(window_center_stream_s),
                    fallback_mono=stream_origin_mono,
                )
                latency_ms = (now - window_center_mono) * 1000.0
                latency_window.append(latency_ms)

                p95_latency = float(np.percentile(latency_window, 95)) if latency_window else float(latency_ms)

                if _is_noop_decision(decision.finger_id, decision.action_id):
                    logger.info(
                        "PREDICT NO-OP finger=%s action=%s joint_prob=%.3f model_raw_finger=%s post_bias_finger=%s raw_action=%s reason=%s quality_bad=%s masked=%s latency_ms=%.1f dropped_windows=%s",
                        decision.finger_id,
                        decision.action_id,
                        decision.prob,
                        model_raw_top_finger_id,
                        decision_info.get("raw_top_finger_id"),
                        decision_info.get("raw_top_action_id"),
                        decision_info.get("decision_reason"),
                        bool(quality.window_quality_bad),
                        list(quality.masked_channel_ids),
                        latency_ms,
                        dropped_windows,
                    )
                else:
                    logger.info(
                        "PREDICT ACTUATABLE finger=%s action=%s joint_prob=%.3f model_raw_finger=%s post_bias_finger=%s raw_action=%s reason=%s quality_bad=%s masked=%s latency_ms=%.1f dropped_windows=%s",
                        decision.finger_id,
                        decision.action_id,
                        decision.prob,
                        model_raw_top_finger_id,
                        decision_info.get("raw_top_finger_id"),
                        decision_info.get("raw_top_action_id"),
                        decision_info.get("decision_reason"),
                        bool(quality.window_quality_bad),
                        list(quality.masked_channel_ids),
                        latency_ms,
                        dropped_windows,
                    )

                if emit_viz and viz_ts is not None:
                    live_viz_payload = inference_result.get("live_viz_payload")
                    if isinstance(live_viz_payload, dict):
                        last_live_viz_emit = now_mono
                        payload = dict(live_viz_payload)
                        payload["t"] = float(viz_ts)
                        print(
                            "VIZJSON " + json.dumps(payload, separators=(",", ":")),
                            flush=True,
                        )
                    elif hidden_mag is not None:
                        last_live_viz_emit = now_mono
                        print(f"VIZ t={viz_ts:.3f} hidden_mag={hidden_mag:.6f}", flush=True)

                # Decide to actuate
                actuation_sent = False
                actuation_latency_ms = None
                actuation_decision_delay_ms = None
                actuation_vote = _resolve_live_actuation_vote(
                    actuation_history,
                    decision,
                    required_pair_stability=int(args.actuation_stability),
                    ignore_window=bool(quality.window_quality_bad),
                    ignore_reason="quality_gate",
                )
                voted_decision = actuation_vote["decision"]
                actuation_target_finger_id = int(voted_decision.finger_id)
                actuation_target_action_id = int(voted_decision.action_id)
                actuation_suppressed_reason = None
                latency_policy = str(getattr(args, "latency_policy", "warn")).strip().lower()
                actuation_latency_gate_ok = (
                    True
                    if latency_policy == "warn"
                    else _latency_gate_passed(latency_ms, float(args.latency_threshold_ms))
                )
                if args.enable_actuation and actuator is not None:
                    if quality.window_quality_bad:
                        actuation_suppressed_reason = "quality_gate"
                        logger.info(
                            "Actuation suppressed by quality gate reason=%s bad_channels=%s masked_channels=%s total_clipped_frac=%.3f",
                            quality.quality_bad_reason or "quality_gate",
                            list(quality.bad_channel_ids),
                            list(quality.masked_channel_ids),
                            float(quality.total_clipped_frac),
                        )
                    elif not actuation_latency_gate_ok:
                        actuation_suppressed_reason = "latency_gate"
                        logger.info(
                            "Actuation suppressed by latency gate latency_ms=%.1f threshold_ms=%.1f",
                            latency_ms,
                            float(args.latency_threshold_ms),
                        )
                    elif not finger_gate_ok:
                        actuation_suppressed_reason = "finger_gate"
                        logger.info(
                            "Actuation suppressed by finger gate finger=%s finger_conf=%.3f threshold=%.3f",
                            decision.finger_id,
                            float(decision_info.get("finger_conf", 0.0)),
                            float(args.threshold_finger),
                        )
                    elif not applicability_gate_ok:
                        actuation_suppressed_reason = "applicability_gate"
                        logger.info(
                            "Actuation suppressed by applicability gate action=%s finger=%s applicability_prob=%.3f threshold=%.3f",
                            decision.action_id,
                            decision.finger_id,
                            float(decision_info.get("finger_applicable_prob", 0.0) or 0.0),
                            float(args.threshold_applicability),
                        )
                    elif _is_noop_decision(
                        voted_decision.finger_id, voted_decision.action_id
                    ):
                        actuation_suppressed_reason = str(
                            actuation_vote.get("reason", "noop")
                        )
                        logger.info(
                            "NO-OP decision suppressed (finger=%s action=%s)",
                            voted_decision.finger_id,
                            voted_decision.action_id,
                        )
                    elif not uncertainty_gate_ok:
                        actuation_suppressed_reason = "uncertainty_gate"
                        logger.info(
                            "Actuation suppressed by uncertainty gate action_conf=%.3f adaptive_threshold=%.3f action_unc=%.4f",
                            float(decision_info.get("action_conf", 0.0)),
                            float(inference_result.get("adaptive_threshold", 0.0)),
                            action_uncertainty,
                        )
                    else:
                        shaped_command = actuation_command_shaper.shape(
                            action_id=int(voted_decision.action_id),
                            finger_id=int(voted_decision.finger_id),
                            action_conf=float(voted_decision.prob),
                            speed_scalar_override=float(actuation_speed_scalar),
                            timestamp_stream_ms=int(round(window_center_stream_s * 1000.0)),
                            stability_ok=True,
                            timebase_ms=int(round(window_center_stream_s * 1000.0)),
                        )
                        actuation_target_finger_id = int(shaped_command.finger_id)
                        actuation_target_action_id = int(shaped_command.action_id)
                        actuation_speed_scalar = float(shaped_command.speed_scalar)
                        actuation_decision = ActuationDecision(
                            finger_id=actuation_target_finger_id,
                            action_id=actuation_target_action_id,
                            prob=float(voted_decision.prob),
                        )
                        actuation_key = (
                            int(actuation_decision.finger_id),
                            int(actuation_decision.action_id),
                        )
                        if _is_noop_decision(
                            actuation_decision.finger_id, actuation_decision.action_id
                        ):
                            actuation_suppressed_reason = "min_prob"
                            logger.debug(
                                "Actuation suppressed by min_prob (%.3f < %.3f)",
                                voted_decision.prob,
                                float(args.actuation_min_prob),
                            )
                        elif _debounced_should_send(
                            decision=actuation_decision,
                            last_sent=last_sent,
                            stable_count=1,
                            required_stability=1,
                            last_send_ts=last_send_ts,
                            cooldown_ms=int(args.actuation_cooldown_ms),
                            repeat_same_ms=int(args.actuation_repeat_ms),
                        ):
                            send_start = time.monotonic()
                            actuator.send(
                                actuation_decision.finger_id,
                                actuation_decision.action_id,
                                speed_scalar=actuation_speed_scalar,
                            )
                            send_end = time.monotonic()
                            last_sent = actuation_key
                            last_send_ts = send_end
                            actuation_sent = True
                            actuation_latency_ms = (send_end - window_center_mono) * 1000.0
                            actuation_decision_delay_ms = (send_start - now) * 1000.0
                            logger.info(
                                "ACTUATE sent finger=%s action=%s prob=%.3f speed=%.3f prediction_latency_ms=%.1f actuation_latency_ms=%.1f decision_to_send_ms=%.1f",
                                actuation_decision.finger_id,
                                actuation_decision.action_id,
                                voted_decision.prob,
                                actuation_speed_scalar,
                                latency_ms,
                                actuation_latency_ms,
                                actuation_decision_delay_ms,
                            )
                        else:
                            actuation_suppressed_reason = "cooldown_or_duplicate"

                if pred_log is not None:
                    payload = {
                        "ts_utc": time.time(),
                        "window_start_s": float(window_start),
                        "window_end_s": float(window_end),
                        "latency_ms": float(latency_ms),
                        "prediction_latency_ms": float(latency_ms),
                        "alignment_ok": True,
                        "action_probs": action_probs.tolist(),
                        "model_raw_finger_probs": model_raw_finger_probs.tolist(),
                        "finger_probs": finger_probs.tolist(),
                        "raw_top_action_id": int(decision_info.get("raw_top_action_id", 0)),
                        "raw_top_finger_id": int(decision_info.get("raw_top_finger_id", 0)),
                        "model_raw_top_finger_id": int(model_raw_top_finger_id),
                        "smoothed_action_id": int(decision_info.get("smoothed_action_id", 0)),
                        "smoothed_finger_id": int(decision_info.get("smoothed_finger_id", 0)),
                        "committed_action_id": int(decision_info.get("committed_action_id", 0)),
                        "committed_finger_id": int(decision_info.get("committed_finger_id", 0)),
                        "action_conf": float(decision_info.get("action_conf", 0.0)),
                        "finger_conf": float(decision_info.get("finger_conf", 0.0)),
                        "finger_gate_ok": bool(decision_info.get("finger_gate_ok", True)),
                        "finger_applicable_prob": decision_info.get(
                            "finger_applicable_prob"
                        ),
                        "applicability_gate_ok": bool(
                            decision_info.get("applicability_gate_ok", True)
                        ),
                        "committed_pair_valid": bool(
                            decision_info.get("committed_pair_valid", True)
                        ),
                        "joint_conf": float(decision.prob),
                        "action_uncertainty": action_uncertainty,
                        "finger_uncertainty": finger_uncertainty,
                        "applicability_uncertainty": applicability_uncertainty,
                        "adaptive_threshold": inference_result.get("adaptive_threshold"),
                        "uncertainty_gate_ok": bool(uncertainty_gate_ok),
                        "health_score": inference_result.get("health_score"),
                        "window_quality_bad": bool(quality.window_quality_bad),
                        "quality_bad_reason": quality.quality_bad_reason,
                        "masked_channel_ids": list(quality.masked_channel_ids),
                        "quality_bad_channel_ids": list(quality.bad_channel_ids),
                        "channel_rms_z": quality.channel_rms_z.tolist(),
                        "channel_abs_p95_z": quality.channel_abs_p95_z.tolist(),
                        "channel_clipped_frac": quality.channel_clipped_frac.tolist(),
                        "total_clipped_frac": float(quality.total_clipped_frac),
                        "inference_backend": str(inference_result.get("backend", "direct")),
                        "decision_reason": str(decision_info.get("decision_reason", "")),
                        "postprocess_enabled": bool(postprocess_enabled),
                        "rest_bias_correction_enabled": bool(rest_bias.enabled),
                        "rest_bias_correction_ready": bool(rest_bias.ready),
                        "rest_bias_correction_applied": bool(rest_bias_applied),
                        "rest_bias_rest_window_count": int(rest_bias.rest_count),
                        "rest_bias_strength": float(rest_bias.strength),
                        "dropped_windows": int(dropped_windows),
                        "actuation_speed_scalar": float(actuation_speed_scalar),
                        "actuation_target_finger_id": int(actuation_target_finger_id),
                        "actuation_target_action_id": int(actuation_target_action_id),
                        "actuation_vote_reason": str(
                            actuation_vote.get("reason", "")
                        ),
                        "actuation_vote_finger_counts": actuation_vote.get(
                            "finger_votes", {}
                        ),
                        "actuation_vote_action_counts": actuation_vote.get(
                            "action_votes", {}
                        ),
                        "actuation_vote_pair_counts": actuation_vote.get(
                            "pair_votes", {}
                        ),
                        "actuation_latency_gate_ok": bool(actuation_latency_gate_ok),
                        "actuation_suppressed_reason": actuation_suppressed_reason,
                        "actuation_sent": bool(actuation_sent),
                        "actuation_latency_ms": (
                            float(actuation_latency_ms)
                            if actuation_latency_ms is not None
                            else None
                        ),
                        "actuation_decision_delay_ms": (
                            float(actuation_decision_delay_ms)
                            if actuation_decision_delay_ms is not None
                            else None
                        ),
                    }
                    pred_log.write(json.dumps(payload) + "\n")
                    pred_log_count += 1
                    if pred_log_count % pred_log_flush_every == 0:
                        pred_log.flush()

                next_window_start_s += args.hop_sec

            # periodic status log
            now = time.monotonic()
            if now - last_log >= args.log_every:
                masked_snapshot = _top_counter_snapshot(masked_channel_counts, top_k=2)
                logger.info(
                    "buffer=%s dropped_windows=%s dropped_nonfinite_samples=%s dropped_nonfinite_windows=%s quality_bad_windows=%s quality_masked_windows=%s masked_channels=%s rest_bias_ready=%s rest_bias_windows=%s",
                    len(buffer),
                    dropped_windows,
                    dropped_nonfinite_samples,
                    dropped_nonfinite_windows,
                    quality_bad_windows,
                    quality_masked_windows,
                    masked_snapshot or None,
                    bool(rest_bias.ready),
                    int(rest_bias.rest_count),
                )
                if masked_channel_counts:
                    top_channel, top_count = masked_channel_counts.most_common(1)[0]
                    should_warn = top_count >= 20
                    if should_warn and last_masked_channel_warning is not None:
                        last_channel, last_count = last_masked_channel_warning
                        should_warn = bool(
                            int(top_channel) != int(last_channel)
                            or int(top_count) >= int(last_count) + 20
                        )
                    if should_warn:
                        logger.warning(
                            "Live quality warning: channel_id=%s has been masked in %s windows. Check headset contact, hair obstruction, and motion on that sensor.",
                            int(top_channel),
                            int(top_count),
                        )
                        last_masked_channel_warning = (int(top_channel), int(top_count))
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
            if (
                not no_file_io
                and pred_log_path is not None
                and Path(pred_log_path).exists()
            ):
                summary_path = Path(out_dir) / "live_prediction_summary.json"
                try:
                    _build_live_prediction_summary(
                        pred_log_path=Path(pred_log_path),
                        summary_path=summary_path,
                        raw_dir=(Path(out_dir) / "raw") if record_raw else None,
                        dropped_windows=dropped_windows,
                        dropped_nonfinite_samples=dropped_nonfinite_samples,
                        dropped_nonfinite_windows=dropped_nonfinite_windows,
                    )
                    logger.info("Prediction summary written: %s", summary_path)
                except Exception as exc:
                    logger.warning(
                        "Failed to write prediction summary %s: %s",
                        summary_path,
                        exc,
                    )
            if actuator is not None:
                actuator.close()
            logger.info(
                "Shutdown complete (reason=%s, dropped_nonfinite_samples=%s, dropped_nonfinite_windows=%s, quality_bad_windows=%s, quality_masked_windows=%s, masked_channels=%s, rest_bias_ready=%s, rest_bias_windows=%s).",
                termination_reason,
                dropped_nonfinite_samples,
                dropped_nonfinite_windows,
                quality_bad_windows,
                quality_masked_windows,
                _top_counter_snapshot(masked_channel_counts, top_k=4) or None,
                bool(rest_bias.ready),
                int(rest_bias.rest_count),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
