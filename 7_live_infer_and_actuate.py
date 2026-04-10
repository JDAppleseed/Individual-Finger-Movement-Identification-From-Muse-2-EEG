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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Deque, Optional, Sequence, Tuple

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
    LiveWindowQuality,
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
    sanitize_live_window as _shared_sanitize_live_window,
    uncertainty_gate_passed as _shared_uncertainty_gate_passed,
)
from utils.live_parity import (
    LiveParityCapture,
    ParityCaptureSettings,
    parse_required_labels,
    sha256_file,
    write_json,
    write_jsonl_row,
)
from utils.lsl_stream_select import (
    MultipleStreamsMatchedError,
    NoStreamFoundError,
    NoStreamMatchedError,
    resolve_source_id_preference,
    select_stream_by_source_id,
    stream_signature,
)
from utils.model_outputs import infer_output_dims_from_state_dict, unpack_model_outputs
from utils.postprocess import PostprocessSettings, PostprocessState, postprocess_predictions
from utils.runtime_utils import (
    TemperatureScalingState,
    apply_channel_normalizer,
    apply_temperature_to_logits,
    load_normalizer,
    load_temperature_scaling,
    now_utc_iso,
)
from utils.session_layout import SessionLayout, resolve_latest_run_dir, resolve_session_dir
from utils.stream_timebase import (
    clamp_lsl_timestamp,
    is_gap,
    should_segment_break_backwards,
)

# Pipeline handoff: Step 7 runs online inference with the trained Step 2 model
# and optional hardware actuation, while writing live-session artifacts.

logger = logging.getLogger("live_infer")


# -------------------- Serial Actuation --------------------

@dataclass
class ActuationDecision:
    finger_id: int
    action_id: int
    prob: float


@dataclass(frozen=True)
class LSLResolutionResult:
    inlet: Any
    resolution: dict[str, Any]


@dataclass(frozen=True)
class LiveLaunchPlan:
    project_name: Optional[str]
    subject_id: Optional[str]
    selection_source: str
    session_dir_inferred: bool
    selected_session_dir: Optional[Path]
    explicit_overrides: tuple[str, ...]
    chosen_run_dir: Optional[Path]
    model_path: Path
    scaler_path: Path
    temperature_path: Path
    out_dir: Path
    no_file_io: bool
    record_raw: bool


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


def _safe_send_actuation(
    actuator: Optional[SerialHandActuator],
    *,
    finger_id: int,
    action_id: int,
    speed_scalar: Optional[float] = None,
) -> bool:
    if actuator is None:
        return False
    try:
        actuator.send(
            finger_id=int(finger_id),
            action_id=int(action_id),
            speed_scalar=speed_scalar,
        )
        return True
    except Exception as exc:
        logger.warning(
            "Actuation send failed finger=%s action=%s speed=%s error=%s",
            int(finger_id),
            int(action_id),
            None if speed_scalar is None else float(speed_scalar),
            exc,
        )
        try:
            actuator.close()
        except Exception:
            pass
        return False


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


def _load_channel_labels_from_npz(npz_path: Path) -> list[str]:
    path = Path(npz_path).expanduser().resolve()
    if not path.exists():
        return []
    try:
        with np.load(path, allow_pickle=False) as npz:
            if "channel_names" not in npz:
                return []
            raw = np.asarray(npz["channel_names"]).reshape(-1)
            return [str(item).strip() for item in raw if str(item).strip()]
    except Exception:
        return []


def _resolve_expected_channel_labels(
    config_settings: dict[str, Any],
    deployment_run_dir: Path,
) -> tuple[list[str], Optional[str]]:
    config_labels = parse_required_labels(config_settings.get("REQUIRED_LSL_LABELS"))
    if config_labels:
        return config_labels, "config.REQUIRED_LSL_LABELS"

    ack_labels = parse_required_labels(config_settings.get("LABEL_CHECK_EXPECTED_LABELS"))
    if ack_labels:
        return ack_labels, "config.LABEL_CHECK_EXPECTED_LABELS"

    npz_labels = _load_channel_labels_from_npz(
        Path(deployment_run_dir).expanduser().resolve().parent.parent / "eeg_windows.npz"
    )
    if npz_labels:
        return npz_labels, "training_npz.channel_names"
    return [], None


def _require_expected_channel_labels(
    expected_labels: Sequence[str],
    expected_labels_source: Optional[str],
) -> list[str]:
    labels = [
        str(label).strip()
        for label in expected_labels
        if str(label).strip()
    ]
    if labels:
        return labels
    source_hint = (
        str(expected_labels_source)
        if expected_labels_source
        else "config.REQUIRED_LSL_LABELS or training_npz.channel_names"
    )
    raise RuntimeError(
        "No expected live channel labels could be derived from "
        f"{source_hint}. Step 7 cannot prove model-order channel mapping without "
        "REQUIRED_LSL_LABELS or training_npz.channel_names."
    )


def _build_channel_reorder(
    expected_labels: Sequence[str],
    resolved_labels: Sequence[str],
) -> Optional[tuple[int, ...]]:
    expected = [str(label).strip().lower() for label in expected_labels if str(label).strip()]
    found = [str(label).strip().lower() for label in resolved_labels if str(label).strip()]
    if not expected or not found:
        return None
    index_by_label: dict[str, int] = {}
    for idx, label in enumerate(found):
        if label in index_by_label:
            return None
        index_by_label[label] = int(idx)
    try:
        return tuple(int(index_by_label[label]) for label in expected)
    except KeyError:
        return None


def _resolve_effective_target_fs(
    *,
    train_config: dict[str, Any],
    window_sec: float,
    requested_target_fs: float,
) -> tuple[float, dict[str, Any]]:
    window_sec_f = float(window_sec)
    requested_f = float(requested_target_fs)
    info: dict[str, Any] = {
        "requested_target_fs": requested_f,
        "effective_target_fs": requested_f,
        "canonical_target_fs": None,
        "model_input_time_samples": None,
        "adjusted": False,
        "reason": None,
    }
    input_shape = train_config.get("input_shape")
    if not isinstance(input_shape, (list, tuple)) or not input_shape:
        return requested_f, info
    try:
        expected_samples = int(input_shape[0])
    except Exception:
        return requested_f, info
    if expected_samples < 1 or window_sec_f <= 0.0:
        return requested_f, info

    canonical_target_fs = float(expected_samples) / window_sec_f
    allowed_delta_hz = max(0.25, 0.005 * canonical_target_fs)
    effective_target_fs = requested_f
    info["canonical_target_fs"] = canonical_target_fs
    info["model_input_time_samples"] = expected_samples

    if abs(requested_f - canonical_target_fs) <= allowed_delta_hz:
        effective_target_fs = canonical_target_fs
        if abs(requested_f - canonical_target_fs) > 1e-9:
            info["adjusted"] = True
            info["reason"] = "canonicalized_from_model_time_axis"
    elif int(round(window_sec_f * requested_f)) != expected_samples:
        raise RuntimeError(
            "Configured window_sec/target_fs does not match the trained model input "
            f"time axis. window_sec={window_sec_f:.6f} target_fs={requested_f:.6f} "
            f"produces {int(round(window_sec_f * requested_f))} samples, but the model "
            f"expects {expected_samples}. Use target_fs={canonical_target_fs:.6f} or "
            "a compatible window_sec."
        )

    info["effective_target_fs"] = float(effective_target_fs)
    return float(effective_target_fs), info


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
    return _shared_sanitize_live_window(
        window_TxC,
        scaler=scaler,
        enabled=enabled,
        input_clip_abs_z=input_clip_abs_z,
        bad_channel_rms_z=bad_channel_rms_z,
        bad_channel_abs_p95_z=bad_channel_abs_p95_z,
        bad_channel_clipped_frac=bad_channel_clipped_frac,
        bad_window_clipped_frac=bad_window_clipped_frac,
        bad_window_max_masked_channels=bad_window_max_masked_channels,
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

    scaler_path_p = Path(scaler_path).expanduser().resolve()
    if not scaler_path_p.exists():
        raise FileNotFoundError(f"Scaler not found: {scaler_path_p}")
    if scaler_path_p.suffix.lower() != ".npz":
        raise ValueError(f"Unexpected scaler file extension: {scaler_path_p.suffix}")
    scaler = load_normalizer(scaler_path_p)
    if scaler is None:
        raise RuntimeError(f"Failed to load scaler from {scaler_path_p}")
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
        "lsl_source_id": None,
        "bluetooth_target": None,
        "LIVE_VIZ_ENABLED": bool(LIVE_INFER_RECIPE_DEFAULTS["LIVE_VIZ_ENABLED"]),
        "LIVE_VIZ_FPS": float(LIVE_INFER_RECIPE_DEFAULTS["LIVE_VIZ_FPS"]),
        "window_sec": float(runtime_defaults.window_sec),
        "hop_sec": float(runtime_defaults.hop_sec),
        "target_fs": float(LIVE_INFER_RECIPE_DEFAULTS["target_fs"]),
        "alignment_internal_max_gap_s": float(
            LIVE_INFER_RECIPE_DEFAULTS["alignment_internal_max_gap_s"]
        ),
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
        "parity_capture_enabled": bool(
            LIVE_INFER_RECIPE_DEFAULTS["parity_capture_enabled"]
        ),
        "parity_capture_max_windows": 64,
        "parity_capture_flush_every": 8,
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
        "--lsl-source-id",
        "--lsl_source_id",
        dest="lsl_source_id",
        type=str,
        metavar="SOURCE_ID",
        help="Explicit LSL source_id override. Precedence is CLI, then env, then config.",
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
        "--alignment-internal-max-gap-s",
        "--alignment_internal_max_gap_s",
        dest="alignment_internal_max_gap_s",
        type=float,
        metavar="SECONDS",
        help=(
            "Maximum internal sample gap tolerated inside a live window before "
            "alignment drops it. Window-edge coverage remains strict."
        ),
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
    audit_group = p.add_argument_group("audit and parity")
    parity_toggle = audit_group.add_mutually_exclusive_group()
    parity_toggle.add_argument(
        "--parity-capture-enabled",
        "--parity_capture_enabled",
        dest="parity_capture_enabled",
        action="store_true",
        help="Persist a bounded rolling sample of accepted live windows for replay parity checks.",
    )
    parity_toggle.add_argument(
        "--no-parity-capture",
        "--no_parity_capture",
        dest="parity_capture_enabled",
        action="store_false",
        help="Disable accepted-window parity capture.",
    )
    audit_group.add_argument(
        "--parity-capture-max-windows",
        "--parity_capture_max_windows",
        dest="parity_capture_max_windows",
        type=int,
        help="Maximum number of accepted windows retained in the rolling parity capture buffer.",
    )
    audit_group.add_argument(
        "--parity-capture-flush-every",
        "--parity_capture_flush_every",
        dest="parity_capture_flush_every",
        type=int,
        help="Flush parity capture files after this many accepted windows.",
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


def _dir_has_entries(path: Path) -> bool:
    try:
        return any(path.iterdir())
    except FileNotFoundError:
        return False


def _collect_required_output_status(
    *,
    no_file_io: bool,
    out_dir: Path,
    pred_log_path: Optional[Path],
    window_audit_path: Optional[Path],
    segment_break_path: Optional[Path],
    summary_path: Optional[Path],
    distribution_report_path: Optional[Path],
    parity_capture: Optional[LiveParityCapture],
    parity_capture_required: bool,
    cleanup_errors: Optional[list[str]] = None,
    summary_write_error: Optional[str] = None,
    distribution_report_write_error: Optional[str] = None,
) -> tuple[dict[str, Optional[str]], list[str]]:
    output_hashes = {
        "live_log_sha256": None if no_file_io else sha256_file(Path(out_dir) / "live_infer.log"),
        "prediction_log_sha256": sha256_file(pred_log_path),
        "window_audit_sha256": sha256_file(window_audit_path),
        "segment_break_sha256": sha256_file(segment_break_path),
        "summary_sha256": sha256_file(summary_path),
        "distribution_report_sha256": sha256_file(distribution_report_path),
        "parity_capture_manifest_sha256": (
            sha256_file(parity_capture.manifest_path) if parity_capture is not None else None
        ),
        "parity_capture_records_sha256": (
            sha256_file(parity_capture.records_path) if parity_capture is not None else None
        ),
    }
    errors = list(cleanup_errors or [])
    if summary_write_error:
        errors.append(f"summary_write_error: {summary_write_error}")
    if distribution_report_write_error:
        errors.append(f"distribution_report_write_error: {distribution_report_write_error}")
    if no_file_io:
        return output_hashes, errors

    required_outputs = [
        ("live_log", Path(out_dir) / "live_infer.log", output_hashes["live_log_sha256"]),
        ("prediction_log", pred_log_path, output_hashes["prediction_log_sha256"]),
        ("window_audit", window_audit_path, output_hashes["window_audit_sha256"]),
        ("segment_break", segment_break_path, output_hashes["segment_break_sha256"]),
        ("summary", summary_path, output_hashes["summary_sha256"]),
        (
            "distribution_report",
            distribution_report_path,
            output_hashes["distribution_report_sha256"],
        ),
    ]
    for label, path, sha_value in required_outputs:
        if path is None:
            errors.append(f"{label}_path_missing")
            continue
        if sha_value is None:
            errors.append(f"{label}_missing_or_unreadable: {path}")

    if parity_capture_required:
        if parity_capture is None:
            errors.append("parity_capture_missing")
        else:
            if output_hashes["parity_capture_manifest_sha256"] is None:
                errors.append(
                    f"parity_capture_manifest_missing_or_unreadable: {parity_capture.manifest_path}"
                )
            if output_hashes["parity_capture_records_sha256"] is None:
                errors.append(
                    f"parity_capture_records_missing_or_unreadable: {parity_capture.records_path}"
                )

    return output_hashes, errors


def resolve_live_launch_plan(
    *,
    config_path: Path,
    config_payload: dict[str, Any],
    config_settings: dict[str, Any],
    session_dir_override: Optional[str],
    project_name_override: Optional[str],
    subject_id_override: Optional[str],
    model_path_override: Optional[str],
    scaler_path_override: Optional[str],
    out_dir_override: Optional[str],
    allow_outside_base: bool,
    no_file_io_override: Optional[bool] = None,
) -> LiveLaunchPlan:
    repo_root = _resolve_repo_root(config_path)
    project_name, subject_id = _derive_project_subject(
        config_payload,
        config_path,
        project_name_override,
        subject_id_override,
        config_settings,
    )
    session_dir_value = session_dir_override or config_settings.get("session_dir")
    session_dir_inferred = False
    if not session_dir_value and project_name and subject_id:
        sessions_root = (
            repo_root
            / "Projects"
            / str(project_name)
            / "subjects"
            / str(subject_id)
            / "sessions"
        )
        latest_session = _latest_dir_by_mtime(sessions_root)
        if latest_session is not None:
            session_dir_value = str(latest_session)
            session_dir_inferred = True

    config_dir = config_path.parent

    def _resolve_path(
        path_str: str,
        base_dir: Optional[Path],
        *,
        prefer_config_dir: bool,
    ) -> Path:
        candidate = Path(path_str).expanduser()
        if candidate.is_absolute():
            return candidate.resolve()

        search_roots: list[Path] = []
        base_dir_resolved = base_dir.expanduser().resolve() if base_dir is not None else None
        cwd_resolved = Path.cwd().resolve()
        repo_root_resolved = repo_root.expanduser().resolve()

        ordered_roots = (
            [base_dir_resolved, cwd_resolved, repo_root_resolved]
            if prefer_config_dir
            else [cwd_resolved, repo_root_resolved, base_dir_resolved]
        )
        for root in ordered_roots:
            if root is None or root in search_roots:
                continue
            search_roots.append(root)

        for root in search_roots:
            resolved = (root / candidate).resolve()
            if resolved.exists():
                return resolved
        return (search_roots[0] / candidate).resolve()

    def _is_relative_to(path: Path, base: Path) -> bool:
        try:
            path.relative_to(base)
            return True
        except ValueError:
            return False

    raw_model_path = model_path_override or config_settings.get("model_path")
    raw_scaler_path = scaler_path_override or config_settings.get("scaler_path")
    raw_out_dir = out_dir_override or config_settings.get("out_dir")

    explicit_overrides: list[str] = []
    chosen_run_dir: Optional[Path] = None
    selected_session_dir: Optional[Path] = None
    selection_source = "legacy_explicit"

    if session_dir_value:
        session_dir_path = resolve_session_dir(str(session_dir_value))
        if not session_dir_path.exists():
            raise RuntimeError(f"Session dir not found: {session_dir_path}")
        selected_session_dir = session_dir_path
        base_dir = session_dir_path

        model_explicit = bool(model_path_override) or not _is_default_infer_artifact_path(
            raw_model_path, "finger_action_model.pt"
        )
        scaler_explicit = bool(
            scaler_path_override
        ) or not _is_default_infer_artifact_path(raw_scaler_path, "scaler.npz")

        if model_explicit:
            explicit_overrides.append("model_path")
        if scaler_explicit:
            explicit_overrides.append("scaler_path")
        if raw_out_dir:
            explicit_overrides.append("out_dir")

        run_dir = None
        if not model_explicit or not scaler_explicit:
            run_dir = resolve_latest_run_dir(session_dir_path)
            if run_dir is None or not run_dir.exists():
                fallback_pair = _resolve_latest_run_dir_across_subject_sessions(
                    repo_root,
                    project_name,
                    subject_id,
                    exclude_session_dir=session_dir_path,
                )
                if fallback_pair is None:
                    raise RuntimeError(
                        "Selected session has no model run directory. "
                        "Pin model_path/scaler_path explicitly or choose a session with "
                        "a processed/models run."
                    )
                _, run_dir = fallback_pair
                selection_source = "session_dir_subject_latest_run"
            chosen_run_dir = Path(run_dir).resolve()

        if model_explicit:
            assert raw_model_path
            model_path = _resolve_path(
                str(raw_model_path),
                config_dir,
                prefer_config_dir=not bool(model_path_override),
            )
        else:
            assert chosen_run_dir is not None
            model_path = chosen_run_dir / "finger_action_model.pt"

        if scaler_explicit:
            assert raw_scaler_path
            scaler_path = _resolve_path(
                str(raw_scaler_path),
                config_dir,
                prefer_config_dir=not bool(scaler_path_override),
            )
        else:
            assert chosen_run_dir is not None
            scaler_path = chosen_run_dir / "scaler.npz"

        if raw_out_dir:
            out_dir = _resolve_path(
                str(raw_out_dir),
                config_dir,
                prefer_config_dir=not bool(out_dir_override),
            )
        else:
            out_dir = SessionLayout(session_dir_path).processed_dir / "live_infer"

        if explicit_overrides:
            selection_source = "legacy_explicit"
        elif selection_source != "session_dir_subject_latest_run":
            selection_source = "subject_latest" if session_dir_inferred else "session_dir"
    else:
        if not raw_model_path or not raw_scaler_path or not raw_out_dir:
            raise RuntimeError(
                "Missing session_dir. Pin model_path, scaler_path, and out_dir explicitly."
            )
        base_dir = config_dir
        model_path = _resolve_path(
            str(raw_model_path),
            config_dir,
            prefer_config_dir=not bool(model_path_override),
        )
        scaler_path = _resolve_path(
            str(raw_scaler_path),
            config_dir,
            prefer_config_dir=not bool(scaler_path_override),
        )
        out_dir = _resolve_path(
            str(raw_out_dir),
            config_dir,
            prefer_config_dir=not bool(out_dir_override),
        )

    out_dir = out_dir.expanduser().resolve()
    base_dir = base_dir.expanduser().resolve()
    if not allow_outside_base and not _is_relative_to(out_dir, base_dir):
        raise ValueError(
            f"out_dir must be within {base_dir} (got {out_dir}). "
            "Pass --allow_outside_base to override."
        )

    config_no_file_io = config_settings.get("no_file_io")
    if config_no_file_io is None and "record_raw" in config_settings:
        config_no_file_io = not bool(config_settings.get("record_raw"))
    no_file_io = (
        bool(no_file_io_override)
        if no_file_io_override is not None
        else bool(config_no_file_io)
    )
    record_raw = not no_file_io
    if record_raw and out_dir.exists() and _dir_has_entries(out_dir):
        raise RuntimeError(
            f"Output dir already exists and is not empty: {out_dir}. "
            "Choose a fresh --out-dir for an unambiguous live run."
        )

    return LiveLaunchPlan(
        project_name=str(project_name) if project_name is not None else None,
        subject_id=str(subject_id) if subject_id is not None else None,
        selection_source=str(selection_source),
        session_dir_inferred=bool(session_dir_inferred),
        selected_session_dir=selected_session_dir,
        explicit_overrides=tuple(explicit_overrides),
        chosen_run_dir=chosen_run_dir,
        model_path=model_path.expanduser().resolve(),
        scaler_path=scaler_path.expanduser().resolve(),
        temperature_path=_resolve_temperature_path(model_path.parent).expanduser().resolve(),
        out_dir=out_dir,
        no_file_io=bool(no_file_io),
        record_raw=bool(record_raw),
    )


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


def _stream_labels(info: Any) -> list[str]:
    try:
        signature = stream_signature(info)
    except Exception:
        return []
    labels = signature.get("labels") or signature.get("channel_labels") or []
    return [str(label).strip() for label in labels if str(label).strip()]


def _stream_contract_summary(
    *,
    config_settings: dict[str, Any],
    expected_name: str,
    expected_type: str,
    source_id_preference: dict[str, Any],
    resolved_stream: dict[str, Any],
    expected_labels: Optional[Sequence[str]] = None,
    expected_rate: Optional[float] = None,
    expected_labels_source: Optional[str] = None,
) -> dict[str, Any]:
    expected_labels_list = (
        [str(label).strip() for label in expected_labels if str(label).strip()]
        if expected_labels is not None
        else parse_required_labels(config_settings.get("REQUIRED_LSL_LABELS"))
    )
    expected_rate_value = (
        float(expected_rate) if expected_rate is not None else config_settings.get("SAMPLING_RATE")
    )
    require_exactly_4 = bool(config_settings.get("REQUIRE_EXACTLY_4_CHANNELS", True))
    resolved_labels = [
        str(label).strip()
        for label in resolved_stream.get("channel_labels", []) or []
        if str(label).strip()
    ]
    mismatches: list[str] = []
    if expected_name and str(resolved_stream.get("name") or "") != str(expected_name):
        mismatches.append("stream_name")
    if expected_type and str(resolved_stream.get("type") or "") != str(expected_type):
        mismatches.append("stream_type")
    if expected_labels_list:
        found_norm = {label.lower() for label in resolved_labels}
        required_norm = {label.lower() for label in expected_labels_list}
        if not required_norm.issubset(found_norm):
            mismatches.append("labels")
    if require_exactly_4 and int(resolved_stream.get("channel_count") or 0) != 4:
        mismatches.append("channel_count")
    if expected_rate_value is not None:
        try:
            expected_rate_f = float(expected_rate_value)
            resolved_rate_f = float(resolved_stream.get("nominal_srate") or 0.0)
            if abs(resolved_rate_f - expected_rate_f) > 1.0:
                mismatches.append("sampling_rate")
        except Exception:
            pass
    return {
        "expected": {
            "stream_name": str(expected_name),
            "stream_type": str(expected_type),
            "required_labels": expected_labels_list,
            "required_labels_source": expected_labels_source,
            "sampling_rate": (
                float(expected_rate_value) if expected_rate_value is not None else None
            ),
            "require_exactly_4_channels": bool(require_exactly_4),
            "source_id_preference": source_id_preference,
        },
        "resolved": resolved_stream,
        "mismatches": mismatches,
        "contract_ok": bool(not mismatches),
    }


def _require_stream_contract_ok(stream_contract: dict[str, Any]) -> None:
    mismatches = list(stream_contract.get("mismatches") or [])
    if not mismatches:
        return
    resolved = dict(stream_contract.get("resolved") or {})
    expected = dict(stream_contract.get("expected") or {})
    raise RuntimeError(
        "Resolved LSL stream violates the configured stream contract. "
        f"mismatches={mismatches} expected={expected} resolved={resolved}"
    )


def _load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _counter_with_max(
    base: collections.Counter[str],
    extra: Optional[collections.Counter[Any]],
) -> collections.Counter[str]:
    merged: collections.Counter[str] = collections.Counter(
        {str(key): int(value) for key, value in base.items()}
    )
    if extra is None:
        return merged
    for key, value in extra.items():
        label = str(key)
        try:
            merged[label] = max(int(merged.get(label, 0)), int(value))
        except Exception:
            continue
    return merged


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
    *,
    cli_source_id: Optional[str] = None,
    env_source_id: Optional[str] = None,
    config_source_id: Optional[str] = None,
) -> LSLResolutionResult:
    if not LSL_AVAILABLE or StreamInlet is None or (
        resolve_streams is None and resolve_byprop is None
    ):
        raise RuntimeError("pylsl is required for live inference.")
    timeout_s = max(0.1, float(timeout_s))
    source_pref = resolve_source_id_preference(
        cli_source_id=cli_source_id,
        env_source_id=env_source_id,
        config_source_id=config_source_id,
    )
    desired_source_id = str(source_pref.requested_source_id or "").strip()
    logger.info(
        "Resolving LSL stream name=%s type=%s source_id=%s source=%s timeout=%.1fs",
        name,
        type_,
        desired_source_id or "-",
        source_pref.source,
        timeout_s,
    )
    deadline = time.monotonic() + timeout_s
    last_seen: list[dict[str, Any]] = []
    last_error: Optional[str] = None

    def _candidate_match(candidate: dict[str, Any], *, match_name: bool) -> bool:
        if type_ and str(candidate.get("type") or "") != str(type_):
            return False
        if match_name and name and str(candidate.get("name") or "") != str(name):
            return False
        return True

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
        candidate_rows: list[dict[str, Any]] = []
        for stream in all_streams:
            try:
                signature = dict(stream_signature(stream))
            except Exception:
                continue
            signature["_stream"] = stream
            candidate_rows.append(signature)
        last_seen = [dict(row) for row in candidate_rows]

        name_type_candidates = [
            row for row in candidate_rows if _candidate_match(row, match_name=True)
        ]
        selection_scope = "name_type"
        selection_candidates = list(name_type_candidates)
        if not selection_candidates and desired_source_id:
            type_only_candidates = [
                row for row in candidate_rows if _candidate_match(row, match_name=False)
            ]
            if type_only_candidates:
                selection_candidates = type_only_candidates
                selection_scope = "type_only_recovery"

        selection = None
        if selection_candidates:
            try:
                selection = select_stream_by_source_id(
                    selection_candidates,
                    requested_source_id=desired_source_id,
                    require_unique_when_unspecified=True,
                )
                last_error = None
            except (
                NoStreamFoundError,
                NoStreamMatchedError,
                MultipleStreamsMatchedError,
            ) as exc:
                last_error = str(exc)
        else:
            last_error = "No LSL stream candidates matched the requested name/type."

        if selection is not None:
            if desired_source_id and bool(selection.recovery_used):
                selected_source_id = str(selection.selected_source_id or "").strip() or "-"
                last_error = (
                    f"Requested LSL source_id={desired_source_id} was not found; "
                    f"refusing single-candidate recovery to source_id={selected_source_id}."
                )
                selection = None
        if selection is not None:
            chosen_candidate = dict(selection.selected)
            chosen = chosen_candidate.pop("_stream")
            inlet = StreamInlet(chosen, max_chunklen=64)
            resolved_stream = {
                key: value
                for key, value in chosen_candidate.items()
                if not str(key).startswith("_")
            }
            resolved_stream["channel_labels"] = _stream_labels(chosen)
            matched_exact_source = bool(
                desired_source_id
                and str(resolved_stream.get("source_id") or "") == desired_source_id
                and not bool(selection.recovery_used)
            )
            try:
                sample, ts = inlet.pull_sample(
                    timeout=min(0.25, max(0.05, remaining or 0.25))
                )
            except Exception:
                sample, ts = None, None
            resolution = {
                "requested_source_id": source_pref.requested_source_id,
                "selected_source_id": resolved_stream.get("source_id"),
                "source_id_source": source_pref.source,
                "source_id_match_mode": (
                    "exact_match"
                    if matched_exact_source
                    else (
                        "recovered_single_candidate"
                        if bool(selection.recovery_used)
                        else "unspecified"
                    )
                ),
                "selection_matched_by_source_id": bool(matched_exact_source),
                "recovery_used": bool(selection.recovery_used),
                "selection_scope": selection_scope,
                "candidate_count": int(len(selection_candidates)),
                "all_stream_count": int(len(candidate_rows)),
                "name": str(resolved_stream.get("name") or ""),
                "type": str(resolved_stream.get("type") or ""),
                "channel_count": int(resolved_stream.get("channel_count") or 0),
                "nominal_srate": float(resolved_stream.get("nominal_srate") or 0.0),
                "source_id": resolved_stream.get("source_id"),
                "uid": resolved_stream.get("uid"),
                "channel_labels": list(resolved_stream.get("channel_labels") or []),
            }
            if sample is not None and ts is not None:
                logger.info(
                    "Resolved LSL stream: %s source_match=%s recovery=%s scope=%s candidates=%s",
                    _format_lsl_stream(chosen),
                    resolution["source_id_match_mode"],
                    bool(selection.recovery_used),
                    selection_scope,
                    len(selection_candidates),
                )
                return LSLResolutionResult(inlet=inlet, resolution=resolution)
            logger.info(
                "LSL stream resolved but not yet producing samples; retrying: %s source_match=%s",
                _format_lsl_stream(chosen),
                resolution["source_id_match_mode"],
            )

        if remaining <= 0.0:
            break
        time.sleep(min(0.25, remaining))

    suffix = ""
    if last_seen:
        rendered = "; ".join(
            ", ".join(
                part
                for part in [
                    f"name={row.get('name')}",
                    f"type={row.get('type')}",
                    f"ch={row.get('channel_count')}",
                    f"rate={row.get('nominal_srate')}",
                    f"source_id={row.get('source_id')}" if row.get("source_id") else "",
                    f"uid={row.get('uid')}" if row.get("uid") else "",
                ]
                if part
            )
            for row in last_seen[:8]
        )
        suffix = f" Available streams: {rendered}"
    if last_error:
        suffix += f" Last selection error: {last_error}"
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
            (
                finger_logits_t,
                action_logits_t,
                applicability_logits_t,
                finger_probs_t,
                action_probs_t,
                applicability_prob_t,
            ) = direct_engine.forward_trace(x)
            action_logits_t = action_logits_t.squeeze(0)
            finger_logits_t = finger_logits_t.squeeze(0)
            applicability_logits_t = (
                applicability_logits_t.squeeze(0)
                if applicability_logits_t is not None
                else None
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
                finger_logits_t = finger_logits.squeeze(0)
                action_logits_t = action_logits.squeeze(0)
                applicability_logits_t = (
                    applicability_logits.squeeze(0)
                    if applicability_logits is not None
                    else None
                )
                if emit_viz:
                    live_viz_payload, hidden_mag = _build_live_viz_payload(model, x)
        return {
            "backend": "direct",
            "action_probs": action_probs_t.detach().cpu().numpy(),
            "finger_probs": finger_probs_t.detach().cpu().numpy(),
            "action_logits": action_logits_t.detach().cpu().numpy(),
            "finger_logits": finger_logits_t.detach().cpu().numpy(),
            "applicability_logit": (
                float(applicability_logits_t.detach().cpu().reshape(-1)[0].item())
                if applicability_logits_t is not None
                else None
            ),
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
        "action_logits": diagnostics.get("action_logits"),
        "finger_logits": diagnostics.get("finger_logits"),
        "applicability_logit": diagnostics.get("applicability_logit"),
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


def _compute_raw_channel_stats(
    raw_dir: Optional[Path],
    *,
    runtime_manifest_path: Optional[Path] = None,
) -> Optional[dict[str, Any]]:
    if raw_dir is None or not raw_dir.exists():
        return None
    from tools.analyze_live_raw_inputs import build_raw_channel_stats

    try:
        return build_raw_channel_stats(
            raw_dir=raw_dir,
            runtime_manifest_path=runtime_manifest_path,
        )
    except Exception as exc:
        return {
            "error": str(exc),
            "shard_count": int(len(sorted(raw_dir.glob("*.npy")))),
        }


def _is_non_rest_pair(action_id: Any, finger_id: Any) -> bool:
    try:
        return int(action_id) > 0 and int(finger_id) > 0
    except Exception:
        return False


def _build_non_rest_flow_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_top_non_rest_count = 0
    committed_valid_non_rest_count = 0
    non_rest_sent_count = 0
    suppressed_counts: collections.Counter[str] = collections.Counter()
    suppressed_reason_counts: collections.Counter[str] = collections.Counter()
    for row in records:
        if _is_non_rest_pair(row.get("raw_top_action_id"), row.get("raw_top_finger_id")):
            raw_top_non_rest_count += 1
        committed_valid_non_rest = bool(row.get("committed_pair_valid", True)) and _is_non_rest_pair(
            row.get("committed_action_id"),
            row.get("committed_finger_id"),
        )
        if not committed_valid_non_rest:
            continue
        committed_valid_non_rest_count += 1
        sent_non_rest = bool(row.get("actuation_sent")) and _is_non_rest_pair(
            row.get("actuation_target_action_id"),
            row.get("actuation_target_finger_id"),
        )
        if sent_non_rest:
            non_rest_sent_count += 1
            continue
        suppressed_reason = str(row.get("actuation_suppressed_reason") or "none")
        suppressed_reason_counts[suppressed_reason] += 1
        if suppressed_reason == "pair_stability":
            suppressed_counts["pair_stability"] += 1
        elif suppressed_reason == "quality_gate":
            suppressed_counts["quality"] += 1
        elif suppressed_reason == "latency_gate":
            suppressed_counts["latency"] += 1
        else:
            suppressed_counts["other"] += 1
    return {
        "raw_top_non_rest_count": int(raw_top_non_rest_count),
        "committed_valid_non_rest_count": int(committed_valid_non_rest_count),
        "non_rest_sent_count": int(non_rest_sent_count),
        "non_rest_suppressed_counts": _stringify_counter(suppressed_counts),
        "non_rest_suppressed_reason_counts": _stringify_counter(
            suppressed_reason_counts
        ),
    }


def _build_live_prediction_summary(
    *,
    pred_log_path: Path,
    summary_path: Path,
    raw_dir: Optional[Path],
    dropped_windows: int,
    dropped_nonfinite_samples: int,
    dropped_nonfinite_windows: int,
    segment_break_count: int,
    candidate_window_count: Optional[int] = None,
    accepted_window_count: Optional[int] = None,
    dropped_window_reason_counts: Optional[collections.Counter[Any]] = None,
    segment_break_reason_counts: Optional[collections.Counter[Any]] = None,
    window_audit_path: Optional[Path] = None,
    segment_break_path: Optional[Path] = None,
    runtime_manifest_path: Optional[Path] = None,
) -> None:
    from tools.analyze_live_predictions import summarize_records

    records = _load_prediction_records(pred_log_path)
    window_audit_rows = (
        _load_jsonl_records(window_audit_path)
        if window_audit_path is not None and window_audit_path.exists()
        else []
    )
    segment_break_rows = (
        _load_jsonl_records(segment_break_path)
        if segment_break_path is not None and segment_break_path.exists()
        else []
    )
    runtime_manifest = (
        load_json(str(runtime_manifest_path))
        if runtime_manifest_path is not None and runtime_manifest_path.exists()
        else {}
    )
    dropped_window_reason_counter = _counter_with_max(
        collections.Counter(
            str(row.get("drop_reason") or "none")
            for row in window_audit_rows
            if str(row.get("status") or "") == "dropped"
        ),
        dropped_window_reason_counts,
    )
    segment_break_reason_counter = _counter_with_max(
        collections.Counter(str(row.get("reason") or "none") for row in segment_break_rows),
        segment_break_reason_counts,
    )
    window_audit_candidate_count = int(len(window_audit_rows))
    window_audit_accepted_count = int(
        sum(str(row.get("status") or "") == "accepted" for row in window_audit_rows)
    )
    window_audit_dropped_count = int(
        sum(str(row.get("status") or "") == "dropped" for row in window_audit_rows)
    )
    candidate_window_count_value = max(
        int(candidate_window_count) if candidate_window_count is not None else 0,
        window_audit_candidate_count,
    )
    accepted_window_count_value = max(
        int(accepted_window_count) if accepted_window_count is not None else 0,
        window_audit_accepted_count,
    )
    segment_break_count_value = max(int(segment_break_count), int(len(segment_break_rows)))
    summary_valid_window_count = 0
    reconciliation = {
        "window_audit_candidate_count": int(window_audit_candidate_count),
        "window_audit_accepted_count": int(window_audit_accepted_count),
        "window_audit_dropped_count": int(window_audit_dropped_count),
        "passed_candidate_window_count": (
            int(candidate_window_count) if candidate_window_count is not None else None
        ),
        "passed_accepted_window_count": (
            int(accepted_window_count) if accepted_window_count is not None else None
        ),
        "passed_segment_break_count": int(segment_break_count),
        "segment_break_log_count": int(len(segment_break_rows)),
        "mismatches": [],
    }
    if candidate_window_count is not None and int(candidate_window_count) != window_audit_candidate_count:
        reconciliation["mismatches"].append("candidate_window_count_vs_window_audit")
    if accepted_window_count is not None and int(accepted_window_count) != window_audit_accepted_count:
        reconciliation["mismatches"].append("accepted_window_count_vs_window_audit")
    if int(segment_break_count) != int(len(segment_break_rows)):
        reconciliation["mismatches"].append("segment_break_count_vs_segment_break_log")
    if not records:
        payload = {
            "record_count": 0,
            "candidate_window_count": candidate_window_count_value,
            "accepted_window_count": accepted_window_count_value,
            "dropped_window_reason_counts": _stringify_counter(
                dropped_window_reason_counter
            ),
            "dropped_windows": int(dropped_windows),
            "dropped_nonfinite_samples": int(dropped_nonfinite_samples),
            "dropped_nonfinite_windows": int(dropped_nonfinite_windows),
            "segment_break_count": int(segment_break_count_value),
            "segment_break_reason_counts": _stringify_counter(
                segment_break_reason_counter
            ),
            "raw_channel_stats": _compute_raw_channel_stats(
                raw_dir,
                runtime_manifest_path=runtime_manifest_path,
            ),
            "runtime_manifest_path": (
                str(runtime_manifest_path) if runtime_manifest_path is not None else None
            ),
            "reconciliation": reconciliation,
        }
        if isinstance(runtime_manifest, dict) and runtime_manifest:
            payload["stream_resolution"] = runtime_manifest.get("stream_resolution")
            payload["artifact_provenance"] = runtime_manifest.get("artifacts")
            payload["stream_contract"] = runtime_manifest.get("stream_contract")
            payload["runtime_manifest_finalization"] = runtime_manifest.get("finalization")
        write_json(summary_path, payload)
        return

    summary_bundle = summarize_records(records)
    summary = dict(summary_bundle.get("summary", {}))
    segment_rows = list(summary_bundle.get("segments", []))
    summary_valid_window_count = int(summary.get("valid_window_count", 0) or 0)
    accepted_window_count_value = max(accepted_window_count_value, summary_valid_window_count)
    if window_audit_rows and (window_audit_candidate_count != (window_audit_accepted_count + window_audit_dropped_count)):
        reconciliation["mismatches"].append("window_audit_candidate_count_non_reconciling")
    if accepted_window_count is not None and int(accepted_window_count) != summary_valid_window_count:
        reconciliation["mismatches"].append("accepted_window_count_vs_predictions")
    if window_audit_rows and window_audit_accepted_count != summary_valid_window_count:
        reconciliation["mismatches"].append("window_audit_accepted_vs_predictions")

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
    for segment in segment_rows:
        if int(segment.get("action_id", 0) or 0) == 0:
            continue
        if int(segment.get("finger_id", 0) or 0) == 0:
            continue
        segments.append(
            {
                "finger_id": int(segment.get("finger_id", 0) or 0),
                "action_id": int(segment.get("action_id", 0) or 0),
                "frames": int(segment.get("window_count", 0) or 0),
                "start_s": float(segment.get("start_s", 0.0) or 0.0),
                "end_s": float(segment.get("end_s", 0.0) or 0.0),
            }
        )
    segments.sort(key=lambda item: int(item.get("frames", 0)), reverse=True)

    masked_channel_counts: collections.Counter[int] = collections.Counter()
    for row in records:
        for channel_id in row.get("masked_channel_ids", []) or []:
            masked_channel_counts[int(channel_id)] += 1

    summary.update(
        {
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
        "actuation_sent_pair_transition_rate": float(
            sent_transitions / max(1, len(sent_pairs) - 1)
        ),
        "longest_committed_non_rest_segments": segments[:10],
        "candidate_window_count": int(candidate_window_count_value),
        "accepted_window_count": int(accepted_window_count_value),
        "dropped_window_reason_counts": _stringify_counter(
            dropped_window_reason_counter
        ),
        "dropped_windows": int(dropped_windows),
        "dropped_nonfinite_samples": int(dropped_nonfinite_samples),
        "dropped_nonfinite_windows": int(dropped_nonfinite_windows),
        "segment_break_count": int(segment_break_count_value),
        "segment_break_reason_counts": _stringify_counter(
            segment_break_reason_counter
        ),
        "raw_channel_stats": _compute_raw_channel_stats(
            raw_dir,
            runtime_manifest_path=runtime_manifest_path,
        ),
        "runtime_manifest_path": (
            str(runtime_manifest_path) if runtime_manifest_path is not None else None
        ),
        "reconciliation": reconciliation,
        }
    )
    summary.update(_build_non_rest_flow_summary(records))
    if isinstance(runtime_manifest, dict) and runtime_manifest:
        summary["stream_resolution"] = runtime_manifest.get("stream_resolution")
        summary["artifact_provenance"] = runtime_manifest.get("artifacts")
        summary["stream_contract"] = runtime_manifest.get("stream_contract")
        summary["runtime_manifest_finalization"] = runtime_manifest.get("finalization")
    # Backward-compatible alias for consumers that still read the shortened key.
    if (
        "actuation_suppressed_reason_counts" in summary
        and "actuation_suppressed_counts" not in summary
    ):
        summary["actuation_suppressed_counts"] = dict(
            summary["actuation_suppressed_reason_counts"]
        )
    write_json(summary_path, summary)


def _summary_safe_finalization(finalization: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(finalization, dict):
        return {}
    return {
        "finalized_at": finalization.get("finalized_at"),
        "termination_reason": finalization.get("termination_reason"),
        "summary_path": finalization.get("summary_path"),
        "summary_write_error": finalization.get("summary_write_error"),
        "distribution_report_path": finalization.get("distribution_report_path"),
        "distribution_report_write_error": finalization.get(
            "distribution_report_write_error"
        ),
        "cleanup_errors": finalization.get("cleanup_errors"),
        "required_outputs_ok": finalization.get("required_outputs_ok"),
        "required_output_errors": finalization.get("required_output_errors"),
        "post_run_commands": finalization.get("post_run_commands"),
    }


def _sync_summary_finalization(
    *,
    summary_path: Optional[Path],
    runtime_manifest_finalization: dict[str, Any],
) -> None:
    if summary_path is None or not summary_path.exists():
        return
    payload = load_json(str(summary_path))
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Prediction summary is not a JSON object: {summary_path}"
        )
    payload["runtime_manifest_finalization"] = _summary_safe_finalization(
        runtime_manifest_finalization
    )
    write_json(summary_path, payload)


# -------------------- Main --------------------

def main() -> int:
    parser, defaults = _build_arg_parser()
    args = parser.parse_args()
    cli_lsl_source_id = getattr(args, "lsl_source_id", None)
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
    config_lsl_source_id = (
        config_settings.get("lsl_source_id") or config_settings.get("LSL_SOURCE_ID")
    )
    env_lsl_source_id = os.environ.get("LSL_SOURCE_ID")
    lsl_source_pref = resolve_source_id_preference(
        cli_source_id=cli_lsl_source_id,
        env_source_id=env_lsl_source_id,
        config_source_id=config_lsl_source_id,
    )
    lsl_source_id = lsl_source_pref.requested_source_id
    try:
        lsl_resolve_timeout_s = float(config_settings.get("LSL_RESOLVE_TIMEOUT", 25.0))
    except Exception:
        lsl_resolve_timeout_s = 25.0
    try:
        launch_plan = resolve_live_launch_plan(
            config_path=config_path,
            config_payload=config_payload,
            config_settings=config_settings,
            session_dir_override=args.session_dir,
            project_name_override=args.project_name,
            subject_id_override=args.subject_id,
            model_path_override=args.model_path,
            scaler_path_override=args.scaler_path,
            out_dir_override=args.out_dir,
            allow_outside_base=bool(args.allow_outside_base),
            no_file_io_override=(True if bool(args.no_file_io) else None),
        )
    except Exception as exc:
        print(f"Live launch planning failed: {exc}")
        return 2
    project_name = launch_plan.project_name
    subject_id = launch_plan.subject_id
    session_dir_inferred = bool(launch_plan.session_dir_inferred)
    selected_session_dir = launch_plan.selected_session_dir
    selection_source = str(launch_plan.selection_source)
    explicit_overrides = list(launch_plan.explicit_overrides)
    chosen_run_dir = launch_plan.chosen_run_dir
    model_path = str(launch_plan.model_path)
    scaler_path = str(launch_plan.scaler_path)
    out_dir_path = launch_plan.out_dir
    out_dir = str(out_dir_path)
    temperature_path = launch_plan.temperature_path
    no_file_io = bool(launch_plan.no_file_io)
    record_raw = bool(launch_plan.record_raw)

    if bool(args.parity_capture_enabled) and no_file_io:
        print(
            "Parity capture cannot be enabled when no_file_io is true. "
            "Disable no_file_io or disable parity capture."
        )
        return 2
    if int(args.parity_capture_max_windows) < 1:
        print("parity_capture_max_windows must be >= 1.")
        return 2
    if int(args.parity_capture_flush_every) < 1:
        print("parity_capture_flush_every must be >= 1.")
        return 2

    print(f"Session selection source: {selection_source}")
    if explicit_overrides:
        print(f"Explicit path overrides: {explicit_overrides}")
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
    runtime_manifest_path: Optional[Path] = None
    runtime_manifest: dict[str, Any] = {}
    window_audit_path: Optional[Path] = None
    window_audit_log = None
    window_audit_flush_every = 50
    window_audit_count = 0
    segment_break_path: Optional[Path] = None
    segment_break_log = None
    segment_break_flush_every = 10
    segment_break_log_count = 0
    parity_capture: Optional[LiveParityCapture] = None
    summary_path: Optional[Path] = None
    summary_write_error: Optional[str] = None
    distribution_report_path: Optional[Path] = None
    distribution_report_write_error: Optional[str] = None
    post_run_exit_code = 0
    source_pref_payload = {
        "cli_source_id": lsl_source_pref.cli_source_id,
        "env_source_id": lsl_source_pref.env_source_id,
        "config_source_id": lsl_source_pref.config_source_id,
        "requested_source_id": lsl_source_pref.requested_source_id,
        "source": lsl_source_pref.source,
    }
    if not no_file_io:
        pred_log_path = args.pred_log or str(Path(out_dir) / "predictions.jsonl")
        runtime_manifest_path = Path(out_dir) / "live_runtime_manifest.json"
        window_audit_path = Path(out_dir) / "window_audit.jsonl"
        segment_break_path = Path(out_dir) / "segment_breaks.jsonl"
        summary_path = Path(out_dir) / "live_prediction_summary.json"
        distribution_report_path = Path(out_dir) / "live_input_distribution_report.json"

    device = _select_device(args.device)
    logger.info("Using device=%s", device)
    resolved_model_path = Path(model_path).expanduser().resolve()
    resolved_scaler_path = Path(scaler_path).expanduser().resolve()
    deployment_run_dir = resolved_model_path.parent
    train_config = _load_train_config(deployment_run_dir)
    effective_target_fs, target_fs_info = _resolve_effective_target_fs(
        train_config=train_config,
        window_sec=float(args.window_sec),
        requested_target_fs=float(args.target_fs),
    )
    if abs(float(args.target_fs) - float(effective_target_fs)) > 1e-9:
        logger.warning(
            "Canonicalizing target_fs from %.6f Hz to %.6f Hz to preserve the trained "
            "model time axis over %.3f s windows.",
            float(args.target_fs),
            float(effective_target_fs),
            float(args.window_sec),
        )
        args.target_fs = float(effective_target_fs)
    expected_channel_labels, expected_channel_labels_source = (
        _resolve_expected_channel_labels(config_settings, deployment_run_dir)
    )
    try:
        expected_channel_labels = _require_expected_channel_labels(
            expected_channel_labels,
            expected_channel_labels_source,
        )
    except Exception as exc:
        _persist_manifest_error("expected_channel_labels_missing", exc)
        raise
    if expected_channel_labels:
        logger.info(
            "Expected live channel order=%s source=%s",
            expected_channel_labels,
            expected_channel_labels_source,
        )
    live_viz_enabled = bool(getattr(args, "LIVE_VIZ_ENABLED", False))
    live_viz_fps = float(getattr(args, "LIVE_VIZ_FPS", 0.0) or 0.0)
    if live_viz_fps <= 0.0:
        live_viz_enabled = False
    live_viz_interval = (1.0 / live_viz_fps) if live_viz_enabled else 0.0
    last_live_viz_emit = 0.0
    train_config_path = deployment_run_dir / "train_config.json"
    runtime_manifest = {
        "created_at": now_utc_iso(),
        "argv": list(sys.argv),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "config_created_at": config_payload.get("created_at"),
        "config_snapshot": dict(config_payload),
        "effective_args": {key: getattr(args, key) for key in sorted(vars(args))},
        "selection_source": str(selection_source),
        "session_dir": str(selected_session_dir) if selected_session_dir is not None else None,
        "session_dir_inferred": bool(session_dir_inferred),
        "explicit_overrides": list(explicit_overrides),
        "project_name": project_name,
        "subject_id": subject_id,
        "session_id": config_payload.get("session_id"),
        "out_dir": str(out_dir),
        "stream_resolution": None,
        "stream_contract": None,
        "stream_selection": {
            "stream_name": str(lsl_name),
            "stream_type": str(lsl_type),
            "resolve_timeout_s": float(lsl_resolve_timeout_s),
            "source_id_preference": source_pref_payload,
            "expected_channel_labels": list(expected_channel_labels),
            "expected_channel_labels_source": expected_channel_labels_source,
        },
        "artifacts": {
            "run_dir": (
                str(chosen_run_dir.resolve())
                if chosen_run_dir is not None
                else str(deployment_run_dir)
            ),
            "model_path": str(resolved_model_path),
            "model_sha256": sha256_file(resolved_model_path),
            "scaler_path": str(resolved_scaler_path),
            "scaler_sha256": sha256_file(resolved_scaler_path),
            "temperature_path": str(Path(temperature_path).expanduser().resolve()),
            "temperature_sha256": sha256_file(temperature_path),
            "train_config_path": (
                str(train_config_path.resolve()) if train_config_path.exists() else None
            ),
            "train_config_sha256": sha256_file(train_config_path),
            "model_input_time_samples": target_fs_info.get("model_input_time_samples"),
        },
        "runtime": {
            "device": str(device),
            "inference_backend": None,
            "no_file_io": bool(no_file_io),
            "record_raw": bool(record_raw),
            "allow_drop": bool(args.allow_drop),
            "log_every_s": float(args.log_every),
            "mc_passes": int(args.mc_passes),
            "uncertainty_base_threshold": float(args.uncertainty_base_threshold),
            "uncertainty_weight": float(args.uncertainty_weight),
            "postprocess_enabled": bool(postprocess_enabled),
            "postprocess_settings": asdict(post_settings),
            "latency_policy": str(args.latency_policy),
            "latency_threshold_ms": float(args.latency_threshold_ms),
            "window_sec": float(args.window_sec),
            "hop_sec": float(args.hop_sec),
            "target_fs": float(args.target_fs),
            "target_fs_requested": float(target_fs_info.get("requested_target_fs")),
            "target_fs_canonical": target_fs_info.get("canonical_target_fs"),
            "target_fs_adjusted": bool(target_fs_info.get("adjusted")),
            "target_fs_adjust_reason": target_fs_info.get("reason"),
            "alignment_internal_max_gap_s": max(
                float(args.alignment_internal_max_gap_s),
                (1.0 / float(args.target_fs) * 4.0),
            ),
            "alignment_edge_max_gap_s": (1.0 / float(args.target_fs) * 4.0),
            "live_quality_enabled": bool(args.live_quality_enabled),
            "quality_thresholds": {
                "input_clip_abs_z": float(args.input_clip_abs_z),
                "bad_channel_rms_z": float(args.bad_channel_rms_z),
                "bad_channel_abs_p95_z": float(args.bad_channel_abs_p95_z),
                "bad_channel_clipped_frac": float(args.bad_channel_clipped_frac),
                "bad_window_clipped_frac": float(args.bad_window_clipped_frac),
                "bad_window_max_masked_channels": int(
                    args.bad_window_max_masked_channels
                ),
            },
            "rest_bias": {
                "enabled": bool(args.rest_bias_correction_enabled),
                "strength": float(args.rest_bias_strength),
                "min_rest_windows": int(args.rest_bias_min_windows),
            },
            "actuation": {
                "enabled": bool(args.enable_actuation),
                "actuation_min_prob": float(args.actuation_min_prob),
                "actuation_stability": int(args.actuation_stability),
                "actuation_cooldown_ms": int(args.actuation_cooldown_ms),
                "actuation_repeat_ms": int(args.actuation_repeat_ms),
                "actuation_min_speed": float(args.actuation_min_speed),
                "modulate_actuation_speed": bool(args.modulate_actuation_speed),
                "actuation_speed_gamma": float(args.actuation_speed_gamma),
            },
            "parity_capture": {
                "enabled": bool(args.parity_capture_enabled and not no_file_io),
                "max_windows": int(args.parity_capture_max_windows),
                "flush_every": int(args.parity_capture_flush_every),
            },
        },
        "outputs": {
            "log_path": None if no_file_io else str(Path(out_dir) / "live_infer.log"),
            "prediction_log_path": str(pred_log_path) if pred_log_path is not None else None,
            "window_audit_path": (
                str(window_audit_path) if window_audit_path is not None else None
            ),
            "segment_break_path": (
                str(segment_break_path) if segment_break_path is not None else None
            ),
            "parity_capture_dir": (
                str(Path(out_dir) / "parity_capture") if not no_file_io else None
            ),
            "summary_path": str(summary_path) if summary_path is not None else None,
            "distribution_report_path": (
                str(distribution_report_path)
                if distribution_report_path is not None
                else None
            ),
        },
    }

    def _write_runtime_manifest() -> None:
        if runtime_manifest_path is None:
            return
        write_json(runtime_manifest_path, runtime_manifest)

    if runtime_manifest_path is not None:
        _write_runtime_manifest()
        logger.info("Runtime manifest: %s", runtime_manifest_path)

    def _persist_manifest_error(reason: str, exc: Exception) -> None:
        if runtime_manifest_path is None or not runtime_manifest:
            return
        runtime_manifest["finalization"] = {
            "finalized_at": now_utc_iso(),
            "termination_reason": str(reason),
            "error": str(exc),
        }
        _write_runtime_manifest()

    def _open_required_text_output(path: Path, label: str):
        try:
            handle = path.open("a", encoding="utf-8")
        except Exception as exc:
            _persist_manifest_error(f"{label}_open_error", exc)
            raise RuntimeError(f"Failed to open {label} {path}: {exc}") from exc
        logger.info("%s: %s", label.replace("_", " ").capitalize(), path)
        return handle

    if not no_file_io:
        assert pred_log_path is not None
        assert window_audit_path is not None
        assert segment_break_path is not None
        pred_log = _open_required_text_output(Path(pred_log_path), "prediction_log")
        window_audit_log = _open_required_text_output(window_audit_path, "window_audit_log")
        segment_break_log = _open_required_text_output(
            segment_break_path, "segment_break_log"
        )

    try:
        model, scaler = load_model_and_scaler(model_path, scaler_path, device=device)
    except Exception as exc:
        _persist_manifest_error("artifact_load_error", exc)
        raise
    model.eval()
    if not temperature_path.exists():
        exc = FileNotFoundError(f"Temperature scaling file not found: {temperature_path}")
        _persist_manifest_error("temperature_artifact_missing", exc)
        raise exc
    temperature_state = load_temperature_scaling(temperature_path)
    if temperature_state is None:
        exc = RuntimeError(f"Failed to load temperature scaling from {temperature_path}")
        _persist_manifest_error("temperature_artifact_load_error", exc)
        raise exc
    logger.info(
        "Temperature scaling loaded: action=%.4f finger=%.4f applicability=%.4f source=%s",
        float(temperature_state.action_temperature),
        float(temperature_state.finger_temperature),
        float(temperature_state.applicability_temperature),
        str(temperature_state.source),
    )
    inference_engine = _build_inference_engine(
        model, scaler, device, args, temperature_state
    )
    direct_inference_engine = (
        None
        if inference_engine is not None
        else _build_direct_inference_engine(model, scaler, device, temperature_state)
    )
    runtime_manifest["runtime"]["inference_backend"] = (
        "inference_engine" if inference_engine is not None else "direct"
    )
    runtime_manifest["artifacts"]["temperature_source"] = str(temperature_state.source)
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
        try:
            deploy_info = _require_deployable_run(deployment_run_dir)
        except Exception as exc:
            _persist_manifest_error("deployable_run_validation_error", exc)
            raise
        runtime_manifest["deployment"] = dict(deploy_info)
        logger.info(
            "Deployment model validated run_dir=%s active_finger_head=%s finger_applicability_head=%s n_fingers=%s n_actions=%s",
            deployment_run_dir,
            deploy_info.get("active_finger_head"),
            deploy_info.get("finger_applicability_head"),
            deploy_info.get("n_fingers"),
            deploy_info.get("n_actions"),
        )
    _write_runtime_manifest()

    try:
        lsl_result = _resolve_lsl_inlet(
            lsl_name,
            lsl_type,
            timeout_s=lsl_resolve_timeout_s,
            cli_source_id=cli_lsl_source_id,
            env_source_id=env_lsl_source_id,
            config_source_id=config_lsl_source_id,
        )
    except Exception as exc:
        _persist_manifest_error("lsl_resolution_error", exc)
        raise
    inlet = lsl_result.inlet
    info = inlet.info()
    sfreq = float(info.nominal_srate())
    ch = int(info.channel_count())
    logger.info(
        "Connected LSL stream name=%s type=%s sfreq=%s ch=%s",
        lsl_name,
        lsl_type,
        sfreq,
        ch,
    )
    stream_contract = _stream_contract_summary(
        config_settings=config_settings,
        expected_name=str(lsl_name),
        expected_type=str(lsl_type),
        source_id_preference=source_pref_payload,
        resolved_stream=lsl_result.resolution,
        expected_labels=expected_channel_labels,
        expected_rate=float(args.target_fs),
        expected_labels_source=expected_channel_labels_source,
    )
    channel_reorder = _build_channel_reorder(
        expected_channel_labels,
        lsl_result.resolution.get("channel_labels", []) or [],
    )
    channel_reorder_applied = bool(
        channel_reorder is not None
        and list(channel_reorder) != list(range(len(channel_reorder)))
    )
    stream_contract["resolved"]["channel_reorder_to_model_order"] = (
        list(channel_reorder) if channel_reorder is not None else None
    )
    stream_contract["resolved"]["channel_reorder_applied"] = bool(channel_reorder_applied)
    if channel_reorder_applied:
        logger.warning(
            "Reordering live stream channels into training order. expected=%s found=%s reorder=%s",
            expected_channel_labels,
            lsl_result.resolution.get("channel_labels", []) or [],
            list(channel_reorder or ()),
        )
    runtime_manifest["stream_resolution"] = lsl_result.resolution
    runtime_manifest["stream_contract"] = stream_contract
    if runtime_manifest_path is not None:
        _write_runtime_manifest()
    try:
        _require_stream_contract_ok(stream_contract)
    except Exception as exc:
        _persist_manifest_error("stream_contract_mismatch", exc)
        raise
    if expected_channel_labels and (lsl_result.resolution.get("channel_labels") or []) and channel_reorder is None:
        exc = RuntimeError(
            "Resolved stream labels passed the set check but could not be mapped into a "
            "deterministic model channel order."
        )
        _persist_manifest_error("stream_channel_reorder_error", exc)
        raise exc
    parity_capture = LiveParityCapture(
        root_dir=Path(out_dir),
        settings=ParityCaptureSettings(
            enabled=bool(args.parity_capture_enabled and not no_file_io),
            max_windows=int(args.parity_capture_max_windows),
            flush_every=int(args.parity_capture_flush_every),
        ),
        manifest_seed={
            "runtime_manifest_path": str(runtime_manifest_path)
            if runtime_manifest_path is not None
            else None,
            "config_sha256": runtime_manifest.get("config_sha256"),
            "stream_resolution": lsl_result.resolution,
            "stream_contract": stream_contract,
            "artifacts": runtime_manifest.get("artifacts"),
        },
    )

    # Session writer (raw shards, optional)
    session_writer = None
    raw_buffer: list[Packet] = []
    raw_flush_size = int(config_settings.get("raw_flush_size", 256))
    if record_raw:
        raw_shard_samples = int(config_settings.get("raw_shard_samples", 2048))
        try:
            session_writer = SessionWriter(
                out_dir=str(out_dir),
                channel_count=ch,
                shard_size_samples=raw_shard_samples,
            )
        except Exception as exc:
            _persist_manifest_error("session_writer_init_error", exc)
            raise
    else:
        logger.info("Raw recording disabled (no_file_io).")

    # Serial actuator
    actuator: Optional[SerialHandActuator] = None
    if args.enable_actuation:
        try:
            serial_port = args.serial_port or config_settings.get("serial_port")
            if not serial_port:
                serial_port = _autodetect_serial_port()
                logger.info("Actuation serial port auto-detected: %s", serial_port)
            actuator = SerialHandActuator(str(serial_port), baud=args.serial_baud)
            actuator.open()
            logger.info("Actuation enabled via serial port %s @ %s baud", serial_port, args.serial_baud)
            _warmup_actuation(actuator)
        except Exception as exc:
            _persist_manifest_error("actuator_init_error", exc)
            raise

    # Live buffers
    from collections import deque
    buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=int(max(5, args.window_sec * args.target_fs * 4)))
    latency_window: Deque[float] = deque(maxlen=200)

    stream_origin_mono: Optional[float] = None
    stream_origin_lsl: Optional[float] = None
    prev_lsl_mono: Optional[float] = None
    backwards_events_mono: Deque[float] = deque(maxlen=256)
    latest_sample_mono: Optional[float] = None
    latest_stream_time_s = 0.0
    last_buffer_time_s: Optional[float] = None
    dropped_windows = 0
    dropped_nonfinite_samples = 0
    dropped_nonfinite_windows = 0
    quality_bad_windows = 0
    quality_masked_windows = 0
    alignment_interpolated_windows = 0
    segment_break_count = 0
    candidate_window_count = 0
    accepted_window_count = 0
    segment_id = 0
    dropped_window_reason_counts: collections.Counter[str] = collections.Counter()
    segment_break_reason_counts: collections.Counter[str] = collections.Counter()
    masked_channel_counts: collections.Counter[int] = collections.Counter()
    last_masked_channel_warning: Optional[Tuple[int, int]] = None
    last_log = time.monotonic()

    next_window_start_s = 0.0
    strict_alignment_gap_s = 1.0 / float(args.target_fs) * 4.0
    alignment_internal_max_gap_s = max(
        float(strict_alignment_gap_s), float(args.alignment_internal_max_gap_s)
    )

    # Debounce state
    last_sent: Optional[Tuple[int, int]] = None
    last_send_ts = 0.0
    sample_seq = 0
    actuation_history: Deque[ActuationDecision] = deque(
        maxlen=max(1, int(args.actuation_stability))
    )
    actuation_command_shaper = _build_actuation_command_shaper(args)

    termination_reason = "ok"

    def _audit_window(
        *,
        candidate_index: int,
        segment_id_value: int,
        window_start_value: float,
        window_end_value: float,
        status: str,
        drop_reason: Optional[str] = None,
        **extra: Any,
    ) -> None:
        nonlocal window_audit_count
        if window_audit_log is None:
            return
        payload = {
            "ts_utc": time.time(),
            "candidate_index": int(candidate_index),
            "segment_id": int(segment_id_value),
            "window_start_s": float(window_start_value),
            "window_end_s": float(window_end_value),
            "status": str(status),
            "drop_reason": str(drop_reason) if drop_reason is not None else None,
        }
        payload.update(extra)
        write_jsonl_row(window_audit_log, payload)
        window_audit_count += 1
        if window_audit_count % window_audit_flush_every == 0:
            window_audit_log.flush()

    try:
        while True:
            # Pull a chunk from LSL
            chunk, timestamps = inlet.pull_chunk(timeout=0.0, max_samples=64)
            if timestamps:
                for sample, lsl_ts in zip(chunk, timestamps):
                    sample_mono = time.monotonic()
                    latest_sample_mono = float(sample_mono)
                    last_stream_time_before_sample = float(latest_stream_time_s)
                    prev_lsl_before = prev_lsl_mono
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
                    segment_break_reason: Optional[str] = None
                    segment_break_delta_s: Optional[float] = None
                    raw_lsl_ts = float(lsl_ts)
                    if (
                        np.isfinite(raw_lsl_ts)
                        and prev_lsl_before is not None
                        and raw_lsl_ts < float(prev_lsl_before)
                    ):
                        backwards_delta_s = float(prev_lsl_before - raw_lsl_ts)
                        if backwards_delta_s > 0.010 and should_segment_break_backwards(
                            backwards_events_mono,
                            float(sample_mono),
                            hard_backwards=backwards_delta_s >= 0.200,
                        ):
                            segment_break_reason = "backwards_lsl"
                            segment_break_delta_s = float(backwards_delta_s)
                            time_s = float(last_stream_time_before_sample)
                            stream_origin_lsl = float(raw_lsl_ts) - float(time_s)
                            stream_origin_mono = float(sample_mono) - float(time_s)
                            prev_lsl_mono = float(raw_lsl_ts)
                            lsl_ts_mono = float(raw_lsl_ts)
                    if (
                        segment_break_reason is None
                        and last_buffer_time_s is not None
                        and float(time_s) > float(last_buffer_time_s)
                        and is_gap(
                            float(time_s) - float(last_buffer_time_s),
                            1.0 / float(args.target_fs),
                        )
                    ):
                        segment_break_reason = "stream_gap"
                        segment_break_delta_s = float(time_s) - float(last_buffer_time_s)
                    if segment_break_reason is not None:
                        segment_break_reason_counts[str(segment_break_reason)] += 1
                        buffer_len_before = int(len(buffer))
                        actuation_history_len_before = int(len(actuation_history))
                        post_state_frames_before = int(post_state.frames_in_state)
                        post_action_len_before = int(len(post_state.action_ids))
                        segment_id += 1
                        segment_break_count += 1
                        if segment_break_log is not None:
                            write_jsonl_row(
                                segment_break_log,
                                {
                                    "ts_utc": time.time(),
                                    "reason": str(segment_break_reason),
                                    "delta_s": segment_break_delta_s,
                                    "new_segment_id": int(segment_id),
                                    "stream_time_s": float(time_s),
                                    "raw_lsl_ts": raw_lsl_ts,
                                    "prev_lsl_ts": prev_lsl_before,
                                    "buffer_len_before": buffer_len_before,
                                    "actuation_history_len_before": actuation_history_len_before,
                                    "post_state_frames_before": post_state_frames_before,
                                    "post_state_action_len_before": post_action_len_before,
                                    "next_window_start_s_before": float(next_window_start_s),
                                },
                            )
                            segment_break_log_count += 1
                            if segment_break_log_count % segment_break_flush_every == 0:
                                segment_break_log.flush()
                        buffer.clear()
                        latency_window.clear()
                        actuation_history.clear()
                        actuation_command_shaper.reset()
                        post_state.reset()
                        last_sent = None
                        last_send_ts = 0.0
                        next_window_start_s = float(time_s)
                        last_buffer_time_s = None
                        backwards_events_mono.clear()
                        logger.warning(
                            "Live stream segment break reason=%s new_segment_id=%s stream_time_s=%.3f raw_lsl_ts=%s prev_lsl_ts=%s",
                            segment_break_reason,
                            int(segment_id),
                            float(time_s),
                            raw_lsl_ts,
                            prev_lsl_before,
                        )
                    latest_stream_time_s = max(float(latest_stream_time_s), float(time_s))
                    vec = np.asarray(sample, dtype=np.float32)
                    if channel_reorder is not None and vec.ndim == 1:
                        vec = vec[np.asarray(channel_reorder, dtype=np.int64)]
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
                                segment_id=int(segment_id),
                                clamped=clamped,
                                raw_path=None,
                                segment_break_reason=segment_break_reason,
                            )
                        )
                        sample_seq += 1
                        if len(raw_buffer) >= raw_flush_size:
                            session_writer.append_packets(raw_buffer)
                            raw_buffer = []

                    if sample_flags & RAW_FLAG_NONFINITE:
                        continue

                    if (
                        last_buffer_time_s is not None
                        and float(time_s) <= float(last_buffer_time_s)
                    ):
                        actuation_command_shaper.note_valid(
                            timebase_ms=int(round(float(sample_mono) * 1000.0))
                        )
                        continue

                    buffer.append((time_s, vec))
                    last_buffer_time_s = float(time_s)
                    actuation_command_shaper.note_valid(
                        timebase_ms=int(round(float(sample_mono) * 1000.0))
                    )

            # Infer over available windows
            time_s = float(latest_stream_time_s)
            while (next_window_start_s + args.window_sec) <= time_s:
                candidate_window_count += 1
                candidate_index = int(candidate_window_count)
                window_start = next_window_start_s
                window_end = window_start + args.window_sec

                times = np.array([t for t, _ in buffer], dtype=float)
                values = np.array([v for _, v in buffer], dtype=float)

                if times.size < 2:
                    dropped_windows += 1
                    dropped_window_reason_counts["insufficient_times"] += 1
                    _audit_window(
                        candidate_index=candidate_index,
                        segment_id_value=segment_id,
                        window_start_value=window_start,
                        window_end_value=window_end,
                        status="dropped",
                        drop_reason="insufficient_times",
                        buffer_sample_count=int(times.size),
                    )
                    next_window_start_s += args.hop_sec
                    continue

                left_idx = max(
                    0, int(np.searchsorted(times, window_start, side="left")) - 1
                )
                right_idx = min(
                    int(times.size),
                    int(np.searchsorted(times, window_end, side="right")) + 1,
                )
                if (right_idx - left_idx) < 2:
                    dropped_windows += 1
                    dropped_window_reason_counts["insufficient_window_samples"] += 1
                    _audit_window(
                        candidate_index=candidate_index,
                        segment_id_value=segment_id,
                        window_start_value=window_start,
                        window_end_value=window_end,
                        status="dropped",
                        drop_reason="insufficient_window_samples",
                        buffer_sample_count=int(times.size),
                        left_idx=int(left_idx),
                        right_idx=int(right_idx),
                    )
                    next_window_start_s += args.hop_sec
                    continue

                window_times = times[left_idx:right_idx]
                window_values = values[left_idx:right_idx]
                if not np.all(np.isfinite(window_values)):
                    dropped_windows += 1
                    dropped_nonfinite_windows += 1
                    dropped_window_reason_counts["nonfinite_window_values"] += 1
                    _audit_window(
                        candidate_index=candidate_index,
                        segment_id_value=segment_id,
                        window_start_value=window_start,
                        window_end_value=window_end,
                        status="dropped",
                        drop_reason="nonfinite_window_values",
                        raw_window_sample_count=int(window_times.size),
                    )
                    next_window_start_s += args.hop_sec
                    continue

                alignment = verify_alignment(
                    window_times,
                    start_s=window_start,
                    end_s=window_end,
                    target_fs=args.target_fs,
                    max_gap_s=float(alignment_internal_max_gap_s),
                    max_edge_gap_s=float(strict_alignment_gap_s),
                )
                if not alignment.ok:
                    dropped_windows += 1
                    dropped_window_reason_counts[str(alignment.reason or "alignment_fail")] += 1
                    _audit_window(
                        candidate_index=candidate_index,
                        segment_id_value=segment_id,
                        window_start_value=window_start,
                        window_end_value=window_end,
                        status="dropped",
                        drop_reason=str(alignment.reason or "alignment_fail"),
                        raw_window_sample_count=int(window_times.size),
                        alignment_ok=False,
                        alignment_reason=alignment.reason,
                        alignment_window_size=int(alignment.window_size),
                        alignment_max_gap_s=alignment.max_gap_s,
                        alignment_start_gap_s=alignment.start_gap_s,
                        alignment_end_gap_s=alignment.end_gap_s,
                        alignment_monotonic=bool(alignment.monotonic),
                    )
                    if pred_log is not None:
                        payload = {
                            "ts_utc": time.time(),
                            "candidate_index": int(candidate_index),
                            "segment_id": int(segment_id),
                            "window_start_s": float(window_start),
                            "window_end_s": float(window_end),
                            "latency_ms": None,
                            "alignment_ok": False,
                            "alignment_reason": alignment.reason,
                            "alignment_window_size": int(alignment.window_size),
                            "alignment_max_gap_s": alignment.max_gap_s,
                            "alignment_start_gap_s": alignment.start_gap_s,
                            "alignment_end_gap_s": alignment.end_gap_s,
                            "alignment_monotonic": bool(alignment.monotonic),
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
                    dropped_window_reason_counts["resample_failed"] += 1
                    _audit_window(
                        candidate_index=candidate_index,
                        segment_id_value=segment_id,
                        window_start_value=window_start,
                        window_end_value=window_end,
                        status="dropped",
                        drop_reason="resample_failed",
                        raw_window_sample_count=int(window_times.size),
                        alignment_ok=True,
                        alignment_reason=None,
                        alignment_window_size=int(alignment.window_size),
                        alignment_max_gap_s=alignment.max_gap_s,
                        alignment_start_gap_s=alignment.start_gap_s,
                        alignment_end_gap_s=alignment.end_gap_s,
                        alignment_monotonic=bool(alignment.monotonic),
                    )
                    next_window_start_s += args.hop_sec
                    continue

                alignment_interpolated = bool(
                    alignment.max_gap_s is not None
                    and float(alignment.max_gap_s) > float(strict_alignment_gap_s)
                )
                if alignment_interpolated:
                    alignment_interpolated_windows += 1

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
                accepted_window_count += 1
                _audit_window(
                    candidate_index=candidate_index,
                    segment_id_value=segment_id,
                    window_start_value=window_start,
                    window_end_value=window_end,
                    status="accepted",
                    raw_window_sample_count=int(window_times.size),
                    resampled_shape=list(window.shape),
                    alignment_ok=True,
                    alignment_reason=None,
                    alignment_window_size=int(alignment.window_size),
                    alignment_max_gap_s=alignment.max_gap_s,
                    alignment_start_gap_s=alignment.start_gap_s,
                    alignment_end_gap_s=alignment.end_gap_s,
                    alignment_monotonic=bool(alignment.monotonic),
                    alignment_interpolated=bool(alignment_interpolated),
                    window_quality_bad=bool(quality.window_quality_bad),
                    quality_bad_reason=quality.quality_bad_reason,
                    masked_channel_count=int(len(quality.masked_channel_ids)),
                    masked_channel_ids=list(quality.masked_channel_ids),
                )

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
                action_logits_arr = (
                    np.asarray(inference_result["action_logits"], dtype=float)
                    if inference_result.get("action_logits") is not None
                    else None
                )
                model_raw_finger_probs = np.asarray(
                    inference_result["finger_probs"], dtype=float
                )
                finger_logits_arr = (
                    np.asarray(inference_result["finger_logits"], dtype=float)
                    if inference_result.get("finger_logits") is not None
                    else None
                )
                finger_applicable_prob = inference_result.get("finger_applicable_prob")
                applicability_logit = inference_result.get("applicability_logit")
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
                        shaper_timebase_ms = int(round(float(now) * 1000.0))
                        shaped_command = actuation_command_shaper.shape(
                            action_id=int(voted_decision.action_id),
                            finger_id=int(voted_decision.finger_id),
                            action_conf=float(voted_decision.prob),
                            speed_scalar_override=float(actuation_speed_scalar),
                            timestamp_stream_ms=int(round(window_center_stream_s * 1000.0)),
                            stability_ok=True,
                            timebase_ms=shaper_timebase_ms,
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
                            send_ok = _safe_send_actuation(
                                actuator,
                                finger_id=actuation_decision.finger_id,
                                action_id=actuation_decision.action_id,
                                speed_scalar=actuation_speed_scalar,
                            )
                            send_end = time.monotonic()
                            if send_ok:
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
                                actuator = None
                                actuation_suppressed_reason = "actuator_error"
                        else:
                            actuation_suppressed_reason = "cooldown_or_duplicate"

                if pred_log is not None:
                    payload = {
                        "ts_utc": time.time(),
                        "candidate_index": int(candidate_index),
                        "segment_id": int(segment_id),
                        "window_start_s": float(window_start),
                        "window_end_s": float(window_end),
                        "latency_ms": float(latency_ms),
                        "prediction_latency_ms": float(latency_ms),
                        "alignment_ok": True,
                        "alignment_window_size": int(alignment.window_size),
                        "alignment_max_gap_s": alignment.max_gap_s,
                        "alignment_start_gap_s": alignment.start_gap_s,
                        "alignment_end_gap_s": alignment.end_gap_s,
                        "alignment_monotonic": bool(alignment.monotonic),
                        "alignment_interpolated": bool(alignment_interpolated),
                        "action_probs": action_probs.tolist(),
                        "action_logits": (
                            action_logits_arr.tolist()
                            if action_logits_arr is not None
                            else None
                        ),
                        "model_raw_finger_probs": model_raw_finger_probs.tolist(),
                        "finger_logits": (
                            finger_logits_arr.tolist()
                            if finger_logits_arr is not None
                            else None
                        ),
                        "finger_probs": finger_probs.tolist(),
                        "applicability_logit": applicability_logit,
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
                        "masked_channel_count": int(len(quality.masked_channel_ids)),
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
                parity_capture.add(
                    {
                        "captured_at": now_utc_iso(),
                        "candidate_index": int(candidate_index),
                        "segment_id": int(segment_id),
                        "window_start_s": float(window_start),
                        "window_end_s": float(window_end),
                        "raw_window_times": window_times.tolist(),
                        "raw_window_values": window_values.tolist(),
                        "resampled_window": window.tolist(),
                        "prepared_window": quality.prepared_window.tolist(),
                        "latency_ms": float(latency_ms),
                        "prediction_latency_ms": float(latency_ms),
                        "alignment": {
                            "ok": True,
                            "reason": None,
                            "window_size": int(alignment.window_size),
                            "max_gap_s": alignment.max_gap_s,
                            "start_gap_s": alignment.start_gap_s,
                            "end_gap_s": alignment.end_gap_s,
                            "monotonic": bool(alignment.monotonic),
                            "interpolated": bool(alignment_interpolated),
                        },
                        "quality": {
                            "window_quality_bad": bool(quality.window_quality_bad),
                            "quality_bad_reason": quality.quality_bad_reason,
                            "masked_channel_ids": list(quality.masked_channel_ids),
                            "bad_channel_ids": list(quality.bad_channel_ids),
                            "masked_channel_count": int(len(quality.masked_channel_ids)),
                            "channel_rms_z": quality.channel_rms_z.tolist(),
                            "channel_abs_p95_z": quality.channel_abs_p95_z.tolist(),
                            "channel_clipped_frac": quality.channel_clipped_frac.tolist(),
                            "total_clipped_frac": float(quality.total_clipped_frac),
                        },
                        "inference": {
                            "backend": str(inference_result.get("backend", "direct")),
                            "action_logits": (
                                action_logits_arr.tolist()
                                if action_logits_arr is not None
                                else None
                            ),
                            "finger_logits": (
                                finger_logits_arr.tolist()
                                if finger_logits_arr is not None
                                else None
                            ),
                            "applicability_logit": applicability_logit,
                            "action_probs": action_probs.tolist(),
                            "model_raw_finger_probs": model_raw_finger_probs.tolist(),
                            "finger_probs": finger_probs.tolist(),
                            "finger_applicable_prob": finger_applicable_prob,
                            "action_uncertainty": float(action_uncertainty),
                            "finger_uncertainty": float(finger_uncertainty),
                            "applicability_uncertainty": applicability_uncertainty,
                            "adaptive_threshold": inference_result.get(
                                "adaptive_threshold"
                            ),
                        },
                        "decision": {
                            "raw_top_action_id": int(
                                decision_info.get("raw_top_action_id", 0)
                            ),
                            "raw_top_finger_id": int(
                                decision_info.get("raw_top_finger_id", 0)
                            ),
                            "model_raw_top_finger_id": int(model_raw_top_finger_id),
                            "smoothed_action_id": int(
                                decision_info.get("smoothed_action_id", 0)
                            ),
                            "smoothed_finger_id": int(
                                decision_info.get("smoothed_finger_id", 0)
                            ),
                            "committed_action_id": int(
                                decision_info.get("committed_action_id", 0)
                            ),
                            "committed_finger_id": int(
                                decision_info.get("committed_finger_id", 0)
                            ),
                            "action_conf": float(decision_info.get("action_conf", 0.0)),
                            "finger_conf": float(decision_info.get("finger_conf", 0.0)),
                            "decision_reason": str(
                                decision_info.get("decision_reason", "")
                            ),
                            "finger_gate_ok": bool(
                                decision_info.get("finger_gate_ok", True)
                            ),
                            "applicability_gate_ok": bool(
                                decision_info.get("applicability_gate_ok", True)
                            ),
                            "committed_pair_valid": bool(
                                decision_info.get("committed_pair_valid", True)
                            ),
                            "joint_conf": float(decision.prob),
                            "uncertainty_gate_ok": bool(uncertainty_gate_ok),
                        },
                        "actuation": {
                            "latency_gate_ok": bool(actuation_latency_gate_ok),
                            "vote_reason": str(actuation_vote.get("reason", "")),
                            "vote_finger_counts": actuation_vote.get("finger_votes", {}),
                            "vote_action_counts": actuation_vote.get("action_votes", {}),
                            "vote_pair_counts": actuation_vote.get("pair_votes", {}),
                            "target_action_id": int(actuation_target_action_id),
                            "target_finger_id": int(actuation_target_finger_id),
                            "speed_scalar": float(actuation_speed_scalar),
                            "suppressed_reason": actuation_suppressed_reason,
                            "sent": bool(actuation_sent),
                        },
                    }
                )

                next_window_start_s += args.hop_sec

            if args.enable_actuation and actuator is not None:
                watchdog_now_ms = int(round(time.monotonic() * 1000.0))
                watchdog_command = actuation_command_shaper.watchdog_command(
                    timebase_ms=watchdog_now_ms
                )
                if watchdog_command is not None:
                    if _safe_send_actuation(
                        actuator,
                        finger_id=int(watchdog_command.finger_id),
                        action_id=int(watchdog_command.action_id),
                        speed_scalar=float(watchdog_command.speed_scalar),
                    ):
                        last_sent = (
                            int(watchdog_command.finger_id),
                            int(watchdog_command.action_id),
                        )
                        last_send_ts = time.monotonic()
                        logger.warning(
                            "Actuation watchdog sent REST due to stalled valid input watchdog_ms=%s",
                            int(actuation_command_shaper.config.watchdog_ms),
                        )
                    else:
                        actuator = None

            # periodic status log
            now = time.monotonic()
            if now - last_log >= args.log_every:
                masked_snapshot = _top_counter_snapshot(masked_channel_counts, top_k=2)
                logger.info(
                    "buffer=%s dropped_windows=%s dropped_nonfinite_samples=%s dropped_nonfinite_windows=%s alignment_interpolated_windows=%s quality_bad_windows=%s quality_masked_windows=%s segment_breaks=%s masked_channels=%s rest_bias_ready=%s rest_bias_windows=%s",
                    len(buffer),
                    dropped_windows,
                    dropped_nonfinite_samples,
                    dropped_nonfinite_windows,
                    alignment_interpolated_windows,
                    quality_bad_windows,
                    quality_masked_windows,
                    int(segment_break_count),
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
        termination_reason = "interrupted"
        logger.info("Stopping live inference.")
    except Exception as exc:
        termination_reason = "error"
        logger.error("Live inference error: %s", exc)
        raise
    finally:
        cleanup_errors: list[str] = []
        if record_raw and session_writer is not None:
            try:
                if raw_buffer:
                    session_writer.append_packets(raw_buffer)
                session_writer.close()
            except Exception as exc:
                cleanup_errors.append(f"session_writer_close_error: {exc}")
        if parity_capture is not None:
            try:
                parity_capture.close()
            except Exception as exc:
                cleanup_errors.append(f"parity_capture_close_error: {exc}")
        if pred_log is not None:
            try:
                pred_log.flush()
                pred_log.close()
            except Exception as exc:
                cleanup_errors.append(f"prediction_log_close_error: {exc}")
        if window_audit_log is not None:
            try:
                window_audit_log.flush()
                window_audit_log.close()
            except Exception as exc:
                cleanup_errors.append(f"window_audit_log_close_error: {exc}")
        if segment_break_log is not None:
            try:
                segment_break_log.flush()
                segment_break_log.close()
            except Exception as exc:
                cleanup_errors.append(f"segment_break_log_close_error: {exc}")
        if not no_file_io:
            summary_path = Path(out_dir) / "live_prediction_summary.json"
        replay_cmd = (
            f"{sys.executable} tools/replay_live_capture.py --capture-dir "
            f"{Path(out_dir) / 'parity_capture'}"
        )
        audit_cmd = (
            f"{sys.executable} tools/audit_live_parity.py --live-dir {out_dir} "
            f"--parity-report {Path(out_dir) / 'parity_report.json'} "
            f"--distribution-report {Path(out_dir) / 'live_input_distribution_report.json'} "
            "--write-json --write-md"
        )
        if (
            not no_file_io
            and pred_log_path is not None
            and Path(pred_log_path).exists()
            and summary_path is not None
        ):
            try:
                _build_live_prediction_summary(
                    pred_log_path=Path(pred_log_path),
                    summary_path=summary_path,
                    raw_dir=(Path(out_dir) / "raw") if record_raw else None,
                    dropped_windows=dropped_windows,
                    dropped_nonfinite_samples=dropped_nonfinite_samples,
                    dropped_nonfinite_windows=dropped_nonfinite_windows,
                    segment_break_count=segment_break_count,
                    candidate_window_count=candidate_window_count,
                    accepted_window_count=accepted_window_count,
                    window_audit_path=window_audit_path,
                    segment_break_path=segment_break_path,
                    runtime_manifest_path=runtime_manifest_path,
                )
                logger.info("Prediction summary written: %s", summary_path)
            except Exception as exc:
                summary_write_error = str(exc)
                logger.error(
                    "Failed to write required prediction summary %s: %s",
                    summary_path,
                    exc,
                )
        if (
            not no_file_io
            and record_raw
            and distribution_report_path is not None
            and runtime_manifest_path is not None
        ):
            raw_dir = Path(out_dir) / "raw"
            offline_npz_candidates = []
            if selected_session_dir is not None:
                offline_npz_candidates.append(SessionLayout(selected_session_dir).windows_npz)
            offline_npz_candidates.append(deployment_run_dir.parent.parent / "eeg_windows.npz")
            offline_npz = next(
                (path for path in offline_npz_candidates if path is not None and path.exists()),
                None,
            )
            if raw_dir.exists() and offline_npz is not None:
                try:
                    from tools.analyze_live_raw_inputs import build_distribution_report

                    distribution_report = build_distribution_report(
                        raw_source=raw_dir,
                        run_dir=deployment_run_dir,
                        offline_npz=offline_npz,
                        runtime_manifest_path=runtime_manifest_path,
                        window_sec=float(args.window_sec),
                        hop_sec=float(args.hop_sec),
                        target_fs=float(args.target_fs),
                        relaxed_gap_s=max(
                            float(args.alignment_internal_max_gap_s),
                            (1.0 / float(args.target_fs) * 4.0),
                        ),
                        predictions_path=(
                            Path(pred_log_path)
                            if pred_log_path is not None and Path(pred_log_path).exists()
                            else None
                        ),
                    )
                    write_json(distribution_report_path, distribution_report)
                    logger.info(
                        "Live input distribution report written: %s",
                        distribution_report_path,
                    )
                except Exception as exc:
                    distribution_report_write_error = str(exc)
                    logger.error(
                        "Failed to write live input distribution report %s: %s",
                        distribution_report_path,
                        exc,
                    )
            else:
                distribution_report_write_error = (
                    f"offline_windows_npz_missing_for_distribution_report: raw_dir={raw_dir} offline_npz={offline_npz}"
                )
                logger.warning("%s", distribution_report_write_error)
        if actuator is not None:
            try:
                actuator.close()
            except Exception as exc:
                cleanup_errors.append(f"actuator_close_error: {exc}")
        if cleanup_errors:
            logger.error("Cleanup errors: %s", cleanup_errors)
        if not no_file_io:
            logger.info(
                "Live outputs: manifest=%s summary=%s distribution_report=%s",
                runtime_manifest_path,
                summary_path,
                distribution_report_path,
            )
            logger.info("Post-run replay: %s", replay_cmd)
            logger.info("Post-run audit: %s", audit_cmd)
        logger.info(
            "Shutdown complete (reason=%s, dropped_nonfinite_samples=%s, dropped_nonfinite_windows=%s, alignment_interpolated_windows=%s, quality_bad_windows=%s, quality_masked_windows=%s, segment_breaks=%s, masked_channels=%s, rest_bias_ready=%s, rest_bias_windows=%s).",
            termination_reason,
            dropped_nonfinite_samples,
            dropped_nonfinite_windows,
            alignment_interpolated_windows,
            quality_bad_windows,
            quality_masked_windows,
            int(segment_break_count),
            _top_counter_snapshot(masked_channel_counts, top_k=4) or None,
            bool(rest_bias.ready),
            int(rest_bias.rest_count),
        )
        logging.shutdown()
        output_hashes, required_output_errors = _collect_required_output_status(
            no_file_io=bool(no_file_io),
            out_dir=Path(out_dir),
            pred_log_path=(Path(pred_log_path) if pred_log_path is not None else None),
            window_audit_path=window_audit_path,
            segment_break_path=segment_break_path,
            summary_path=summary_path,
            parity_capture=parity_capture,
            parity_capture_required=bool(args.parity_capture_enabled and not no_file_io),
            cleanup_errors=cleanup_errors,
            summary_write_error=summary_write_error,
            distribution_report_path=distribution_report_path,
            distribution_report_write_error=distribution_report_write_error,
        )
        final_termination_reason = str(termination_reason)
        if required_output_errors and final_termination_reason == "ok":
            final_termination_reason = "required_output_error"
            post_run_exit_code = 2
        elif required_output_errors:
            post_run_exit_code = 2
        if runtime_manifest_path is not None and runtime_manifest:
            finalization_payload = {
                "finalized_at": now_utc_iso(),
                "termination_reason": final_termination_reason,
                "counters": {
                    "candidate_window_count": int(candidate_window_count),
                    "accepted_window_count": int(accepted_window_count),
                    "dropped_windows": int(dropped_windows),
                    "dropped_nonfinite_samples": int(dropped_nonfinite_samples),
                    "dropped_nonfinite_windows": int(dropped_nonfinite_windows),
                    "alignment_interpolated_windows": int(
                        alignment_interpolated_windows
                    ),
                    "quality_bad_windows": int(quality_bad_windows),
                    "quality_masked_windows": int(quality_masked_windows),
                    "segment_break_count": int(segment_break_count),
                    "masked_channel_counts": _top_counter_snapshot(
                        masked_channel_counts, top_k=8
                    )
                    or None,
                    "rest_bias_ready": bool(rest_bias.ready),
                    "rest_bias_window_count": int(rest_bias.rest_count),
                },
                "summary_path": str(summary_path) if summary_path is not None else None,
                "summary_write_error": (
                    str(summary_write_error) if summary_write_error is not None else None
                ),
                "distribution_report_path": (
                    str(distribution_report_path)
                    if distribution_report_path is not None
                    else None
                ),
                "distribution_report_write_error": (
                    str(distribution_report_write_error)
                    if distribution_report_write_error is not None
                    else None
                ),
                "cleanup_errors": cleanup_errors or None,
                "required_outputs_ok": bool(not required_output_errors),
                "required_output_errors": required_output_errors or None,
                "output_hashes": output_hashes,
                "post_run_commands": {
                    "replay": replay_cmd,
                    "audit": audit_cmd,
                },
            }
            try:
                _sync_summary_finalization(
                    summary_path=summary_path,
                    runtime_manifest_finalization=finalization_payload,
                )
            except Exception as exc:
                required_output_errors.append(
                    f"summary_finalization_sync_error: {exc}"
                )
                output_hashes["summary_sha256"] = sha256_file(summary_path)
                final_termination_reason = "required_output_error"
                post_run_exit_code = 2
                finalization_payload["termination_reason"] = final_termination_reason
                finalization_payload["required_outputs_ok"] = False
                finalization_payload["required_output_errors"] = required_output_errors
            else:
                output_hashes["summary_sha256"] = sha256_file(summary_path)
                finalization_payload["output_hashes"] = output_hashes
            runtime_manifest["finalization"] = finalization_payload
            write_json(runtime_manifest_path, runtime_manifest)

    return int(post_run_exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
