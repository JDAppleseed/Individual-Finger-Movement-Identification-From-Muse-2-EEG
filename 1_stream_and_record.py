"""
ISEF / Research Demo Pipeline — STEP 1 (FIXED TIMEBASE + RESUME-SAFE EVENTS)
Muse 2 EEG → LSL → Compression → ICA → Window Prep (cleaned)
Optional: CNN/LSTM inference + latency + MC-dropout uncertainty
Also: keyboard event marking (space=hold event), autosave events

FIXES (this version):
- Stream-relative absolute_v1 timebase with monotonic clamp:
    time_s = lsl_ts - stream_start_lsl_ts (clamped on backward jumps)
- Events aligned to the same stream timebase with clamp protection.
- Per-segment timebase fields anchored to EEG start:
    stream_start_lsl_ts, local_clock_at_start, clock_offset
- Resume gating retained; no silent overwrites.
- Session metadata/state sidecars updated to reflect timebase health counters.
"""

# Work around duplicate libomp on macOS (MKL/torch/scipy); must be set before imports.
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
import threading
import subprocess
import queue
from concurrent.futures import ThreadPoolExecutor
import time
import csv
import json
import shutil
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from scipy.signal import welch

from pylsl import StreamInlet, local_clock
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from utils.experiment_logger import (
    get_subject_id,
    generate_experiment_hash,
    log_experiment,
)

from utils.per_subject_calibration import record_prediction
from utils.online_calibration import OnlineCalibrator
from utils.label_schema import (
    ACTION_REST,
    ACTION_OPEN,
    ACTION_CLOSE,
    FINGER_NONE,
    ACTION_NAMES,
    FINGER_NAMES,
    event_type_for,
)
from utils.lsl_stream_select import (
    LSLStreamSelectError,
    MultipleStreamsMatchedError,
    NoStreamFoundError,
    NoStreamMatchedError,
    StreamSelector,
    extract_channel_labels,
    log_stream_signature,
    pick_stream,
    stream_signature,
)
from utils.stream_timebase import (
    gap_threshold_s,
    is_gap,
    summarize_gaps,
)
from utils.timebase_selfcheck import evaluate_timebase_alignment
from utils.ica_guard import guard_ica_fit, validate_ica_input
from utils.stream_health import RollingStreamHealthGate
from utils.stream_runtime import (
    FailedWriters,
    HardStopPolicy,
    HealthStopState,
    StreamRequirements,
)
from utils.output_paths import resolve_output_dir
from muse_streaming.io_paths import default_processed_dir, default_raw_dir

try:
    from pynput import keyboard
except Exception:
    keyboard = None

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# ===== CONFIG FLAGS ======
# =========================
TRAINING_MODE = False
DEMO_MODE = False
ENABLE_PLOT = True
SAVE_TO_DISK = True
SAVE_RAW = True

SAMPLING_RATE = 256
WINDOW_SEC = 0.25
WINDOW_HOP_SEC = 0.05
CHANNELS = 4
N_FINGERS = 6
N_ACTIONS = 3
TIMEBASE_VERSION = "absolute_v1"

MODEL_PATH = "models/finger_action_model.pt"
SCALER_PATH = "scaler.save"

# =========================
# ===== OUTPUT PATHS ======
# =========================
DEFAULT_PROCESSED_DIR = default_processed_dir()
DEFAULT_RAW_DIR = default_raw_dir()
PROCESSED_DIR = DEFAULT_PROCESSED_DIR
RAW_DIR = DEFAULT_RAW_DIR

# =========================
# ===== TIMEBASE JITTER =====
# =========================
SOFT_BACKWARDS_EPS_S = 0.010
HARD_BACKWARDS_S = 0.200
SOFT_BACKWARDS_LIMIT = 6
SOFT_BACKWARDS_WINDOW_S = 1.0
LSL_MONO_EPS_S = 1e-5

# =========================
# ===== SAFETY & UNCERTAINTY ======
# =========================
BASE_CONF_THRESH = 0.75
UNCERTAINTY_WEIGHT = 0.5
STABILITY_FRAMES = 3
ENABLE_ACTUATION = True
MC_DROPOUT_PASSES = 10

TIMEBASE_SELF_CHECK_WARN_S = 0.05
TIMEBASE_SELF_CHECK_ERROR_S = 0.2

# =========================
# ===== EVENT MARKING =====
# =========================
EVENT_MARKING_ENABLED = True
EVENTS_CSV_PATH = None
EVENTS_AUTOSAVE_PATH = None
EVENTS_CHANNEL = "n/a"

# =========================
# ===== STREAM SOURCE =====
# =========================
LSL_STREAM_NAME = None
LSL_STREAM_TYPE = None
CSV_OFFLINE_PATH = None
SESSION_ID_OVERRIDE = None

streamer_proc = None


def _stop_streamer_process(proc) -> None:
    if proc is None or not hasattr(proc, "poll"):
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                proc.kill()
    except (OSError, AttributeError) as exc:
        print(f"⚠️ Failed to stop streamer process: {exc}", file=sys.stderr)

# =========================
# ===== ICA SAFETY ========
# =========================
ENABLE_ICA = False
ICA_WARMUP_S = 10.0
ICA_MIN_SAMPLES = 256 * 5
ICA_MIN_VAR = 1e-8
ICA_FAIL_POLICY = "skip"
ICA_MAX_RETRIES_PER_SESSION = 1
LOG_ICA_DIAGNOSTICS = True

# =========================
# ===== STREAM HEALTH =====
# =========================
DATA_STREAM_TIMEOUT_S = 5.0
DATA_STREAM_CHECK_INTERVAL_S = 0.5
GAP_BREAK_S = 1.0
GAP_RESET_THRESHOLD_S = 0.5
STALL_S = 0.25
STALL_INPUT_S = 5.0
HEALTH_WINDOW_S = 2.0
MIN_WRITE_FRACTION = 0.90
MAX_QUEUE = 512
RECOVERY_S = 2.0
BACKWARDS_WINDOW_S = 1.0
BACKWARDS_LIMIT = 3
EVENT_MAX_LAG_S = 2.0
EVENT_MAX_LEAD_S = 0.5

DRAIN_MAX_SECONDS = 1.0
DRAIN_MAX_SAMPLES = int(0.5 * SAMPLING_RATE)
UNHEALTHY_WARN_S = 2.0
UNHEALTHY_SOFT_STOP_S = 10.0
UNHEALTHY_HARD_STOP_S = 20.0
DEBUG_SAMPLE_DECIMATE = 4
ACQ_MAX_BUFLEN_S = 60.0
ACQ_MAX_CHUNKLEN = 1024
ACQ_PULL_TIMEOUT_S = 0.05
RAW_QUEUE_MAXSIZE = 20000
PROCESSING_QUEUE_MAXSIZE = 20000
RAW_WRITE_BUFFER_BYTES = 1024 * 1024
RAW_WRITE_BATCH_MAX = 2000
BACKLOG_SECONDS_THRESHOLD = 2.0
BACKLOG_GRACE_S = 2.0
HEALTH_GAP_THRESHOLD_MULT = 1.5
HEALTH_GAP_MAX_EVENTS = 3
HEALTH_GAP_WINDOW_S = 2.0
INTEGRITY_GAP_TOLERANCE_MULT = 1.5

# =========================
# ===== SUBJECT INFO ======
# =========================
GENDER = "M"
AGE = 16
SUBJECT_ID_OVERRIDE = "8-M16"  # Set to None to use auto-increment registry
IMPORT_ONLY = os.environ.get("STREAM_IMPORT_ONLY") == "1"

# =========================
# ===== EARLY GLOBALS INIT =====
# =========================
INIT_ONLY = False
total_elapsed_s = 0.0
last_time_s = -1.0
BLOCK_ID = 0
features_path_state = None
events_path_state = None
raw_path_state = None
created_utc = None
time_s_clamped_count = 0
total_backward_timestamp_count = 0
total_gap_count = 0
event_clamped_count = 0
max_backwards_jump_s = 0.0
csv_file = None
csv_writer = None
raw_file = None
raw_writer = None
predictions_file = None
predictions_writer = None
ica = None
ica_scaler = None
lsl_stream_signature = None


@dataclass
class StreamState:
    subject_id: Optional[str]
    session_id: Optional[str]
    segment_id: int
    experiment_hash: Optional[str]
    features_path: Optional[str]
    events_path: Optional[str]
    predictions_path: Optional[str]
    raw_path: Optional[str]
    timebase_version: str
    stream_start_lsl_ts: Optional[float]
    local_clock_at_start: Optional[float]
    clock_offset: Optional[float]
    gap_count: int = 0
    gap_max_s: float = 0.0
    gap_p95_s: Optional[float] = None
    gap_p99_s: Optional[float] = None
    backward_timestamp_count: int = 0
    window_drop_count: int = 0
    window_gap_drop_count: int = 0
    window_incomplete_drop_count: int = 0
    window_health_drop_count: int = 0
    lstm_reset_count: int = 0
    lstm_reset_log: List[Dict[str, Any]] = field(default_factory=list)
    samples_seen: int = 0
    samples_written: int = 0
    action_pred_buffer: Deque[int] = field(
        default_factory=lambda: deque(maxlen=STABILITY_FRAMES)
    )
    sample_time_buffer: Deque[Tuple[float, np.ndarray]] = field(
        default_factory=lambda: deque(maxlen=512)
    )
    recent_sample_times: Deque[float] = field(
        default_factory=lambda: deque(maxlen=1024)
    )
    windows_processed: int = 0
    nearest_sample_delta_samples: List[Dict[str, Any]] = field(default_factory=list)
    ica_fitted: bool = False
    ica_fit_future: Any = None
    ica_transform_future: Any = None
    ica_fit_segment_id: Optional[int] = None
    ica_transform_segment_id: Optional[int] = None
    soft_stop_triggered: bool = False


RAW_FLAG_NONFINITE = 1


@dataclass(frozen=True)
class SamplePacket:
    lsl_ts_raw: float
    lsl_ts_mono: float
    local_ts: float
    sample: List[float]
    flags: int
    segment_id: int
    raw_path: Path
    clamped: bool
    segment_break_reason: Optional[str] = None

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
parser.add_argument(
    "--subject-id", type=str, default=None, help="Override subject ID for this run"
)
parser.add_argument(
    "--init-only",
    action="store_true",
    help="Initialize session and exit before LSL streaming",
)
parser.add_argument(
    "--force-new-session",
    action="store_true",
    help="Always start a new session (ignore resume state)",
)
parser.add_argument(
    "--processed-dir",
    type=str,
    default=None,
    help="Directory for processed outputs (features/events)",
)
parser.add_argument(
    "--raw-dir",
    type=str,
    default=None,
    help="Directory for raw EEG outputs",
)


def _load_config_payload(path: str):
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return payload


def _load_config_settings(path: str):
    payload = _load_config_payload(path)
    return payload.get("settings", payload)


def _apply_config(settings: dict):
    for key, val in settings.items():
        if key in globals():
            globals()[key] = val


def _apply_config_to_args(args_obj, settings: dict, defaults: dict):
    for key, default in defaults.items():
        if key in settings and getattr(args_obj, key) == default:
            setattr(args_obj, key, settings[key])


def _parse_label_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[str] = []
        for v in value:
            s = str(v).strip()
            if not s:
                continue
            # Configs/UI sometimes embed quotes inside the string, e.g. "'TP9'".
            # Normalize at the boundary so downstream label checks are stable.
            s = s.strip("\"'")
            out.append(s)
        return out
    if isinstance(value, str):
        out: List[str] = []
        for v in value.split(","):
            s = v.strip()
            if not s:
                continue
            s = s.strip("\"'")
            out.append(s)
        return out
    return [str(value).strip()]


def _evaluate_label_check(
    found_labels: Optional[List[str]], channel_count: int
) -> Dict[str, Any]:
    expected_n = int(stream_requirements.expected_channels)
    raw_required = list(stream_requirements.required_labels)
    found_labels = [str(x) for x in (found_labels or [])]

    def _norm(label: str) -> str:
        # Aggressive normalization to avoid false "label mismatch" due to
        # quoting/spacing differences across config/UI/LSL metadata.
        s = str(label).strip()
        # Remove any quoting characters wherever they appear (configs sometimes
        # contain "'TP9'" as a literal string).
        s = s.replace("\"", "").replace("'", "")
        # Normalize whitespace and remove it for stable comparisons.
        s = " ".join(s.split())
        s = s.replace(" ", "")
        return s.lower()

    invalid_required = [lab for lab in raw_required if _norm(lab) == "aux"]
    required_labels = [lab for lab in raw_required if _norm(lab) != "aux"]
    required_norm = [_norm(lab) for lab in required_labels]
    found_norm = [_norm(lab) for lab in found_labels]

    errors: list[str] = []
    missing: list[str] = []
    if invalid_required:
        errors.append("invalid_required_label_aux")

    if stream_requirements.require_exact_channels:
        if channel_count != expected_n:
            errors.append("channel_count_mismatch")
        if found_norm[:expected_n] != required_norm:
            errors.append("label_mismatch")
    else:
        if channel_count < expected_n:
            errors.append("channel_count_too_low")
        missing = [lab for lab in required_norm if lab not in found_norm]
        if missing:
            errors.append("missing_labels")

    status: Dict[str, Any] = {
        "ok": not errors,
        "reason": "ok" if not errors else ",".join(errors),
        "expected_labels": required_labels,
        "found_labels": found_labels[:expected_n]
        if stream_requirements.require_exact_channels
        else found_labels,
        "channel_count": int(channel_count),
        "expected_channel_count": expected_n,
        "require_exact_channels": bool(stream_requirements.require_exact_channels),
        "acknowledged": bool(label_check_acknowledged),
    }
    if missing:
        status["missing_labels"] = missing
    if invalid_required:
        status["invalid_required_labels"] = invalid_required
    return status


args, _ = parser.parse_known_args()
defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
config_payload = _load_config_payload(args.config) if args.config else {}
config_settings = config_payload.get("settings", config_payload)
_apply_config(config_settings)
_apply_config_to_args(args, config_settings, defaults)

if args.subject_id:
    SUBJECT_ID_OVERRIDE = args.subject_id

INIT_ONLY = bool(args.init_only)

subject_id = (
    SUBJECT_ID_OVERRIDE or ("unknown" if IMPORT_ONLY else get_subject_id(GENDER, AGE))
)
SESSION_ID_SEED = datetime.now().strftime("%Y%m%d_%H%M%S")
repo_root = Path(__file__).resolve().parent
session_id_hint = (
    config_settings.get("SESSION_ID_OVERRIDE")
    or config_payload.get("session_id")
    or config_settings.get("session_id")
)
session_id_hint = str(session_id_hint) if session_id_hint is not None else None
PROCESSED_DIR = resolve_output_dir(
    kind="processed",
    cli_value=args.processed_dir,
    config_settings=config_settings,
    config_payload=config_payload,
    config_path=Path(args.config).resolve() if args.config else None,
    repo_root=repo_root,
    subject_id=subject_id,
    session_id_hint=session_id_hint,
    default_base=DEFAULT_PROCESSED_DIR,
    session_id_seed=SESSION_ID_SEED,
)
RAW_DIR = resolve_output_dir(
    kind="raw",
    cli_value=args.raw_dir,
    config_settings=config_settings,
    config_payload=config_payload,
    config_path=Path(args.config).resolve() if args.config else None,
    repo_root=repo_root,
    subject_id=subject_id,
    session_id_hint=session_id_hint,
    default_base=DEFAULT_RAW_DIR,
    session_id_seed=SESSION_ID_SEED,
)
session_id = None
segment_id = 0
experiment_hash = None

stream_requirements = StreamRequirements(
    required_labels=_parse_label_list(
        config_settings.get("REQUIRED_LSL_LABELS", ["TP9", "AF7", "AF8", "TP10"])
    ),
    require_exact_channels=bool(
        config_settings.get("REQUIRE_EXACTLY_4_CHANNELS", True)
    ),
    expected_channels=int(config_settings.get("CHANNELS", CHANNELS)),
)
hard_stop_policy = HardStopPolicy(
    hard_stop_after_unhealthy_s=float(
        config_settings.get("HARD_STOP_AFTER_UNHEALTHY_S", UNHEALTHY_HARD_STOP_S)
    ),
    failed_write_window_s=float(config_settings.get("FAILED_WRITE_WINDOW_S", 5.0)),
    failed_dir=str(config_settings.get("FAILED_DIR", "data/failed")),
    hard_stop_exit_code=int(config_settings.get("HARD_STOP_EXIT_CODE", 73)),
)
health_state = HealthStopState(
    label_check_status=None,
)
failed_writers = FailedWriters()
label_check_acknowledged = bool(
    config_settings.get("LABEL_CHECK_ACKNOWLEDGED", False)
)
live_viz_enabled = bool(config_settings.get("LIVE_VIZ_ENABLED", False))
live_viz_fps = int(config_settings.get("LIVE_VIZ_FPS", 2))
streamer_internal = bool(config_settings.get("STREAMER_INTERNAL", True))
streamer_stream_name = config_settings.get("STREAMER_STREAM_NAME", "Muse2-EEG")
streamer_stream_type = config_settings.get("STREAMER_STREAM_TYPE", "EEG")

state = StreamState(
    subject_id=subject_id,
    session_id=session_id,
    segment_id=segment_id,
    experiment_hash=experiment_hash,
    features_path=None,
    events_path=None,
    predictions_path=None,
    raw_path=None,
    timebase_version=TIMEBASE_VERSION,
    stream_start_lsl_ts=None,
    local_clock_at_start=None,
    clock_offset=None,
)
state.soft_stop_triggered = False

experiment_config = {
    "sampling_rate": SAMPLING_RATE,
    "window_sec": WINDOW_SEC,
    "channels": CHANNELS,
    "model": "CNNLSTMFingerActionNet + ICA",
}


def utc_now_iso_z() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def local_now_iso() -> str:
    return datetime.now().astimezone().isoformat()

# =========================
# ===== SESSION STATE =====
# =========================
SESSION_STATE_DIR = Path("logs")
SESSION_STATE_PATH = SESSION_STATE_DIR / f"session_state_{subject_id}.json"

session_state = {}
if SESSION_STATE_PATH.exists():
    try:
        session_state = json.loads(SESSION_STATE_PATH.read_text())
    except Exception as e:
        print(f"⚠️ Failed to load session state: {e}")

# Defaults (initialized early for safe imports)
total_elapsed_s = 0.0  # session-continuous elapsed time (excludes downtime)
last_time_s = -1.0  # last written session-continuous time_s

ica_enabled_requested = bool(ENABLE_ICA)
ica_ran = False
ica_skipped_reason = None
ica_failed_exception = None
ica_disabled_due_to_error = False
ica_retries = 0

if not ENABLE_ICA:
    ica_skipped_reason = "disabled"

selected_channel_indices = None
selected_channel_variances = None

data_stream_active = False
data_stream_last_write_ts = None
data_stream_stalled_reason = None
stream_health_measured_fs = None
stream_health_write_rate = 0.0
stream_health_queue_size = 0
stream_health_backwards_count = 0
stream_health_last_received_lsl_ts = None
stream_health_last_written_lsl_ts = None
event_marking_allowed = False
last_health_decision = None

# Canonical timebase fields (absolute_v1, per segment)
stream_start_lsl_ts = None
local_clock_at_start = None
clock_offset = None
segment_start_lsl_ts = None
run_start_utc_iso = None
run_end_utc_iso = None
run_start_local_iso = None
run_end_local_iso = None


def _coerce_float(value, default=None):
    if value is None:
        return default
    try:
        return float(value)
    except Exception:
        return default


def _read_csv_header(path: Path):
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


def _csv_has_data_rows(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            _ = next(reader, None)
            for row in reader:
                if any(str(cell).strip() for cell in row):
                    return True
    except Exception:
        return False
    return False


def _header_has_columns(header, required_cols) -> bool:
    if not header:
        return False
    header_set = {str(h).strip() for h in header}
    return required_cols.issubset(header_set)


def _infer_session_id_from_path(path: Path, subject: str):
    if path is None:
        return None
    name = Path(path).name
    prefix = f"{subject}_"
    if not name.startswith(prefix):
        return None
    if name.endswith("_eeg_features.csv"):
        return name[len(prefix) : -len("_eeg_features.csv")]
    if name.endswith("_events.csv"):
        return name[len(prefix) : -len("_events.csv")]
    return None


def _session_has_existing_outputs(session: str, subject: str) -> bool:
    processed = PROCESSED_DIR
    raw = RAW_DIR
    patterns = [
        processed.glob(f"{subject}_{session}_*.csv"),
        raw.glob(f"{subject}_{session}_*.csv"),
        processed.glob(f"{subject}_{session}_*.json"),
    ]
    return any(any(pat) for pat in patterns)


def _unique_session_id(session: str, subject: str) -> str:
    if not _session_has_existing_outputs(session, subject):
        return session
    suffix = 1
    while True:
        candidate = f"{session}_{suffix:02d}"
        if not _session_has_existing_outputs(candidate, subject):
            print(
                f"⚠️ Session ID collision for {session}; using {candidate} to avoid overwrite."
            )
            return candidate
        suffix += 1


def _resolve_events_path(state_events_path, state_features_path, subject, session):
    if state_events_path:
        return Path(state_events_path)
    if state_features_path and session:
        return Path(state_features_path).parent / f"{subject}_{session}_events.csv"
    return None


def _infer_stream_start_lsl_ts(path: Path):
    # Legacy helper (still used only if you need to backfill diagnostics)
    if not path or not path.exists():
        return None
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row and "lsl_timestamp" in row:
                    return _coerce_float(row.get("lsl_timestamp"))
                break
    except Exception:
        return None
    return None


def _load_session_meta(path: Path):
    if not path or not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _to_jsonable(obj):
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        normalized = {}
        for key, value in obj.items():
            if isinstance(key, Path):
                normalized_key = str(key)
            elif isinstance(key, (np.integer, np.floating, np.bool_)):
                normalized_key = key.item()
            else:
                normalized_key = key
            normalized[normalized_key] = _to_jsonable(value)
        return normalized
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]
    return obj


def _write_json_atomic(path: Path, payload: dict):
    normalized_payload = _to_jsonable(payload)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(normalized_payload, indent=2))
    tmp.replace(path)


def _maybe_backup_path(path: Path, label: str):
    if path.exists():
        backup_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.name}.bak_{backup_stamp}")
        path.rename(backup_path)
        print(f"⚠️ {label} backed up to {backup_path}")
        return backup_path
    return None


def _merge_non_none(existing: dict, updates: dict) -> dict:
    merged = dict(existing)
    for key, value in updates.items():
        if value is not None:
            merged[key] = value
    return merged


resume_requested = not bool(args.force_new_session)
resume_blockers = []
state_subject_id = session_state.get("subject_id")
resume_subject_match = bool(state_subject_id == subject_id)
if not resume_requested:
    resume_blockers.append("forced_new_session")
if not resume_subject_match:
    resume_blockers.append("subject_id_mismatch")

state_features_path = session_state.get("features_path")
state_events_path = session_state.get("events_path")
state_raw_path = session_state.get("raw_path")
state_session_id = session_state.get("session_id")
state_timebase_version = session_state.get("timebase_version") or session_state.get(
    "timebase"
)

required_feature_cols = {"lsl_timestamp", "time_s"}
state_features_ok = False
state_features_header_ok = False
state_features = None
if resume_subject_match and state_features_path:
    state_features = Path(state_features_path)
    state_features_ok = _csv_has_data_rows(state_features)
    if state_features_ok:
        state_features_header_ok = _header_has_columns(
            _read_csv_header(state_features), required_feature_cols
        )
    if not state_features_ok:
        resume_blockers.append("features_missing_or_empty")
    elif not state_features_header_ok:
        resume_blockers.append("features_missing_required_columns")
else:
    if resume_subject_match:
        resume_blockers.append("features_missing")

resolved_events_path = _resolve_events_path(
    state_events_path, state_features_path, subject_id, state_session_id
)
events_path_safe = False
state_events_missing = False
if resume_subject_match and resolved_events_path:
    events_parent = resolved_events_path.parent
    features_parent = Path(state_features_path).parent if state_features_path else None
    same_dir = (features_parent is None) or (events_parent == features_parent)
    if same_dir and (resolved_events_path.exists() or events_parent.exists()):
        events_path_safe = True
        state_events_missing = not resolved_events_path.exists()
if not events_path_safe:
    resume_blockers.append("events_path_not_safe")

state_meta = {}
if state_session_id:
    meta_candidate = PROCESSED_DIR / f"{subject_id}_{state_session_id}_session_meta.json"
    state_meta = _load_session_meta(meta_candidate)
    if not state_timebase_version:
        state_timebase_version = state_meta.get("timebase_version") or state_meta.get(
            "timebase"
        )
if state_timebase_version and state_timebase_version != TIMEBASE_VERSION:
    resume_blockers.append(f"timebase_mismatch({state_timebase_version})")

resume_allowed = (
    resume_subject_match
    and state_features_ok
    and state_features_header_ok
    and events_path_safe
)
true_resume = (
    resume_requested
    and resume_allowed
    and not any(b for b in resume_blockers if b not in {"forced_new_session"})
)

if true_resume:
    session_id = state_session_id
    if not session_id:
        if state_features is not None:
            session_id = _infer_session_id_from_path(state_features, subject_id)
        if not session_id and resolved_events_path is not None:
            session_id = _infer_session_id_from_path(resolved_events_path, subject_id)
    BLOCK_ID = int(session_state.get("block_id", 0))
    segment_id = int(session_state.get("segment_id", -1)) + 1
    total_elapsed_s = float(session_state.get("total_elapsed_s", 0.0))
    last_time_s = float(session_state.get("last_time_s", -1.0))
    experiment_hash = session_state.get("experiment_hash")
    features_path_state = state_features_path
    events_path_state = (
        str(resolved_events_path) if resolved_events_path else state_events_path
    )
    raw_path_state = state_raw_path
    created_utc = session_state.get("created_utc") or state_meta.get("created_utc")
    state.session_id = session_id
    state.segment_id = segment_id
    state.experiment_hash = experiment_hash

    # Reset per-segment timebase; we intentionally DO NOT reuse old alignment across runs.
    stream_start_lsl_ts = None
    local_clock_at_start = None
    clock_offset = None
    state.stream_start_lsl_ts = None
    state.local_clock_at_start = None
    state.clock_offset = None
    run_start_utc_iso = None
    run_start_local_iso = None

else:
    session_id = SESSION_ID_SEED
    experiment_hash = generate_experiment_hash(subject_id, experiment_config)
    features_path_state = None
    events_path_state = None
    raw_path_state = None
    total_elapsed_s = 0.0
    last_time_s = -1.0
    BLOCK_ID = 0
    segment_id = 0
    created_utc = None
    state.session_id = session_id
    state.segment_id = segment_id
    state.experiment_hash = experiment_hash

# If no prior session_id, start fresh
if not session_id:
    session_id = SESSION_ID_SEED
if not true_resume and SESSION_ID_OVERRIDE:
    session_id = str(SESSION_ID_OVERRIDE)
if not true_resume and session_id:
    session_id = _unique_session_id(session_id, subject_id)
if experiment_hash is None:
    experiment_hash = generate_experiment_hash(subject_id, experiment_config)
state.session_id = session_id
state.segment_id = segment_id
state.experiment_hash = experiment_hash

if true_resume and state_features_path:
    resume_processed_dir = Path(state_features_path).expanduser().resolve().parent
    if resume_processed_dir != PROCESSED_DIR:
        print(
            "⚠️ Resume override: using existing processed_dir "
            f"{resume_processed_dir} (requested {PROCESSED_DIR})"
        )
    PROCESSED_DIR = resume_processed_dir
if true_resume and state_raw_path:
    resume_raw_dir = Path(state_raw_path).expanduser().resolve().parent
    if resume_raw_dir != RAW_DIR:
        print(
            "⚠️ Resume override: using existing raw_dir "
            f"{resume_raw_dir} (requested {RAW_DIR})"
        )
    RAW_DIR = resume_raw_dir
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_ARCHIVE_DIR = PROCESSED_DIR
RAW_ARCHIVE_DIR = RAW_DIR


def _segment_tag(seg_id: int) -> str:
    return f"SEG{int(seg_id):02d}"


def _segment_paths(seg_id: int):
    tag = _segment_tag(seg_id)
    features = FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_{tag}_eeg_features.csv"
    events = FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_{tag}_events.csv"
    autosave = (
        FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_{tag}_events_autosave.csv"
    )
    predictions = (
        FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_{tag}_predictions.csv"
    )
    raw = RAW_ARCHIVE_DIR / f"{subject_id}_{session_id}_{tag}_raw.csv"
    return features, events, autosave, predictions, raw


(
    FEATURES_ARCHIVE_PATH,
    EVENTS_ARCHIVE_PATH,
    EVENTS_AUTOSAVE_PATH_SEG,
    PREDICTIONS_ARCHIVE_PATH,
    RAW_ARCHIVE_PATH,
) = _segment_paths(segment_id)
EVENTS_AUTOSAVE_PATH = str(EVENTS_AUTOSAVE_PATH_SEG)
if EVENTS_CSV_PATH is None:
    EVENTS_CSV_PATH = str(EVENTS_ARCHIVE_PATH)
FEATURES_PATH = FEATURES_ARCHIVE_PATH
SESSION_META_PATH = (
    FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_session_meta.json"
)
ROOT_SESSION_META_PATH = Path("session_meta.json")
state.features_path = str(FEATURES_ARCHIVE_PATH)
state.events_path = str(EVENTS_ARCHIVE_PATH)
state.predictions_path = str(PREDICTIONS_ARCHIVE_PATH)
state.raw_path = str(RAW_ARCHIVE_PATH)


def _build_session_meta_payload(complete: bool):
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "experiment_hash": experiment_hash,
        "sampling_rate": SAMPLING_RATE,
        "window_sec": WINDOW_SEC,
        "channels": CHANNELS,
        "features_path": str(FEATURES_ARCHIVE_PATH),
        "events_path": str(EVENTS_ARCHIVE_PATH),
        "predictions_path": str(PREDICTIONS_ARCHIVE_PATH),
        "raw_path": str(RAW_ARCHIVE_PATH),
        "timebase_version": TIMEBASE_VERSION,
        "timebase": TIMEBASE_VERSION,
        # Canonical (segment-relative) bookkeeping
        "total_elapsed_s": float(total_elapsed_s),
        "last_time_s": float(last_time_s),
        "stream_start_lsl_ts": stream_start_lsl_ts,
        "local_clock_at_start": local_clock_at_start,
        "clock_offset": clock_offset,
        "run_start_utc_iso": run_start_utc_iso,
        "run_end_utc_iso": run_end_utc_iso,
        "run_start_local_iso": run_start_local_iso,
        "run_end_local_iso": run_end_local_iso,
        "segment_id": int(segment_id),
        "lsl_stream": lsl_stream_signature,
        "required_lsl_labels": stream_requirements.required_labels,
        "found_lsl_labels": (
            health_state.label_check_status.get("found_labels")
            if health_state.label_check_status
            else None
        ),
        # Legacy/diagnostic (kept)
        "complete": bool(complete),
        "created_utc": created_utc,
        "updated_utc": utc_now_iso_z(),
    }


def _update_session_meta(complete: bool, label: str):
    if INIT_ONLY:
        return
    payload = _build_session_meta_payload(complete)
    existing = _load_session_meta(SESSION_META_PATH)
    merged = _merge_non_none(existing, payload)
    msg = "Updating" if SESSION_META_PATH.exists() else "Writing"
    print(f"ℹ️ {msg} session meta ({label}): {SESSION_META_PATH}")
    _write_json_atomic(SESSION_META_PATH, merged)

    root_existing = _load_session_meta(ROOT_SESSION_META_PATH)
    root_merged = _merge_non_none(root_existing, payload)
    root_msg = "Updating" if ROOT_SESSION_META_PATH.exists() else "Writing"
    print(f"ℹ️ {root_msg} root session meta ({label}): {ROOT_SESSION_META_PATH}")
    _write_json_atomic(ROOT_SESSION_META_PATH, root_merged)


def _build_session_state_payload(
    state: StreamState,
    total_elapsed_override=None,
    last_time_override=None,
    block_id_override=None,
    segment_id_override=None,
    soft_stop_triggered: Optional[bool] = None,
):
    soft_stop = (
        bool(getattr(state, "soft_stop_triggered", False))
        if soft_stop_triggered is None
        else bool(soft_stop_triggered)
    )
    soft_stop_report = globals().get("soft_stop_report_path", None)
    timebase_health = globals().get("timebase_health_snapshot", None)
    segment_id_val = (
        int(segment_id_override)
        if segment_id_override is not None
        else state.segment_id
    )
    return {
        "subject_id": state.subject_id,
        "session_id": state.session_id,
        "experiment_hash": state.experiment_hash,
        "block_id": int(block_id_override if block_id_override is not None else BLOCK_ID),
        "segment_id": segment_id_val,
        # Canonical session-continuous time
        "total_elapsed_s": float(
            total_elapsed_override
            if total_elapsed_override is not None
            else total_elapsed_s
        ),
        "last_time_s": float(
            last_time_override if last_time_override is not None else last_time_s
        ),
        "features_path": state.features_path,
        "events_path": state.events_path,
        "predictions_path": state.predictions_path,
        "raw_path": state.raw_path,
        "created_utc": created_utc,
        "updated_utc": utc_now_iso_z(),
        "timebase_version": state.timebase_version,
        "timebase": state.timebase_version,
        "time_s_clamped_count": int(time_s_clamped_count),
        "event_clamped_count": int(event_clamped_count),
        "stream_start_lsl_ts": state.stream_start_lsl_ts,
        "local_clock_at_start": state.local_clock_at_start,
        "clock_offset": state.clock_offset,
        "gap_count": state.gap_count,
        "gap_max_s": state.gap_max_s if state.gap_count else None,
        "gap_p95_s": state.gap_p95_s,
        "gap_p99_s": state.gap_p99_s,
        "backward_timestamp_count": state.backward_timestamp_count,
        "backward_timestamp_count_total": int(total_backward_timestamp_count),
        "gap_count_total": int(total_gap_count),
        "window_drop_count": state.window_drop_count,
        "window_gap_drop_count": state.window_gap_drop_count,
        "window_incomplete_drop_count": state.window_incomplete_drop_count,
        "window_health_drop_count": state.window_health_drop_count,
        "lstm_reset_count": state.lstm_reset_count,
        "lstm_reset_log": list(state.lstm_reset_log),
        "ica_enabled_requested": bool(ica_enabled_requested),
        "ica_ran": bool(ica_ran),
        "ica_skipped_reason": ica_skipped_reason,
        "ica_failed_exception": ica_failed_exception,
        "selected_channel_indices": selected_channel_indices,
        "selected_channel_variances": selected_channel_variances,
        "label_check_status": health_state.label_check_status,
        "label_check_expected_labels": stream_requirements.required_labels,
        "label_check_found_labels": (
            health_state.label_check_status.get("found_labels")
            if health_state.label_check_status
            else None
        ),
        "label_check_acknowledged": bool(label_check_acknowledged),
        "hard_stop_triggered": bool(health_state.hard_stop_triggered),
        "hard_stop_report_path": str(health_state.hard_stop_report_path)
        if health_state.hard_stop_report_path
        else None,
        "soft_stop_triggered": soft_stop,
        "soft_stop_report_path": str(soft_stop_report)
        if soft_stop_report
        else None,
        "data_stream_active": bool(data_stream_active),
        "data_stream_last_write_ts": data_stream_last_write_ts,
        "data_stream_stalled_reason": data_stream_stalled_reason,
        "stream_health_measured_fs": stream_health_measured_fs,
        "stream_health_write_rate": float(stream_health_write_rate),
        "stream_health_queue_size": int(stream_health_queue_size),
        "stream_health_backwards_count": int(stream_health_backwards_count),
        "stream_health_last_received_lsl_ts": stream_health_last_received_lsl_ts,
        "stream_health_last_written_lsl_ts": stream_health_last_written_lsl_ts,
        "event_marking_allowed": bool(event_marking_allowed),
        "timebase_health": timebase_health,
        # Legacy/diagnostic
        "segment_start_lsl_ts": segment_start_lsl_ts,
    }


def _write_session_state(
    label: str,
    total_elapsed_override=None,
    last_time_override=None,
    block_id_override=None,
    segment_id_override=None,
):
    if INIT_ONLY:
        return
    try:
        soft_stop = bool(getattr(state, "soft_stop_triggered", False))
        payload = _build_session_state_payload(
            state,
            total_elapsed_override=total_elapsed_override,
            last_time_override=last_time_override,
            block_id_override=block_id_override,
            segment_id_override=segment_id_override,
            soft_stop_triggered=soft_stop,
        )
        SESSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
        msg = "Updating" if SESSION_STATE_PATH.exists() else "Writing"
        print(f"ℹ️ {msg} session state ({label}): {SESSION_STATE_PATH}")
        _write_json_atomic(SESSION_STATE_PATH, payload)
    except Exception as exc:
        print(
            f"⚠️ Failed to write session state ({label}): {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def _report_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_label_check_report(label_status: Dict[str, Any]) -> str:
    report_dir = Path("logs")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        f"label_check_{subject_id}_{session_id or 'UNKNOWN'}_{_report_stamp()}.json"
    )
    payload = {
        "subject_id": subject_id,
        "session_id": session_id,
        "timestamp_utc": utc_now_iso_z(),
        "acknowledged": bool(label_check_acknowledged),
        "lsl_stream": lsl_stream_signature,
    }
    payload.update(label_status)
    report_path.write_text(json.dumps(payload, indent=2))
    return str(report_path)


resume_flag = "YES" if true_resume else "NO"
resume_request_reason = "forced_new_session" if args.force_new_session else "auto"


def _fmt_time_value(value):
    if isinstance(value, (int, float)) and np.isfinite(value):
        return f"{float(value):.6f}"
    return "pending"


def _apply_channel_indices(
    sample: List[float], channel_indices: Optional[List[int]], channels: int
) -> List[float]:
    if channel_indices is None:
        return sample
    identity = list(range(channels))
    if max(channel_indices) >= len(sample):
        raise ValueError(
            f"LSL EEG sample has {len(sample)} channels; expected at least {max(channel_indices) + 1}."
        )
    if channel_indices != identity or len(sample) != channels:
        return [sample[i] for i in channel_indices]
    return sample


def _should_accept_ica_result(current_segment: int, result_segment: Optional[int]) -> bool:
    if result_segment is None:
        return True
    return int(result_segment) == int(current_segment)

channel_list = "TP9, AF7, AF8, TP10"

if not IMPORT_ONLY:
    print("-" * 50)
    print("🧠 EEG SESSION INITIALIZED")
    print("-" * 50)
    print(f"Subject ID        : {subject_id}")
    print(f"Session ID        : {session_id}")
    print(f"Experiment Hash   : {experiment_hash}")
    print("")
    print(
        f"Resume Requested  : {'YES' if resume_requested else 'NO'} ({resume_request_reason})"
    )
    print(f"Resume Decision   : {resume_flag}")
    if resume_blockers and not true_resume:
        print(f"Resume Blockers   : {', '.join(resume_blockers)}")
    print(f"Current Block ID  : {BLOCK_ID}")
    print(
        f"Total Elapsed Time: {total_elapsed_s:.2f} s (session-continuous across segments)"
    )
    print("")
    if true_resume and state_events_missing:
        print(
            f"⚠️ Resume note: events file missing at {resolved_events_path}; a new events file will be created."
        )
    print(f"EEG Channels      : {channel_list} ({CHANNELS})")
    print(f"Sampling Rate     : {SAMPLING_RATE} Hz")
    print(f"Window Length     : {WINDOW_SEC} s")
    print(f"Timebase Version  : {TIMEBASE_VERSION}")
    print(f"Stream Start LSL  : {_fmt_time_value(stream_start_lsl_ts)}")
    print(f"Stream Start Local: {_fmt_time_value(local_clock_at_start)}")
    print(f"Clock Offset      : {_fmt_time_value(clock_offset)}")
    print("")
    print("Modes:")
    print(f"  Event Marking   : {'ENABLED' if EVENT_MARKING_ENABLED else 'DISABLED'}")
    print(f"  Demo Mode       : {'ON' if DEMO_MODE else 'OFF'}")
    print(f"  Training Mode   : {'ON' if TRAINING_MODE else 'OFF'}")
    print(f"  Actuation       : {'ENABLED' if ENABLE_ACTUATION else 'DISABLED'}")
    print(f"  ICA             : {'ENABLED' if ENABLE_ICA else 'DISABLED'}")
    print("")
    print("Output Paths:")
    print(f"  Processed Dir   : {PROCESSED_DIR}")
    print(f"  Raw Dir         : {RAW_DIR}")
    print(f"  Features CSV    : {FEATURES_ARCHIVE_PATH}")
    print(f"  Events CSV      : {EVENTS_ARCHIVE_PATH}")
    print(f"  Predictions CSV : {PREDICTIONS_ARCHIVE_PATH}")
    print(f"  Raw EEG CSV     : {RAW_ARCHIVE_PATH}")
    print(f"  Session State   : {SESSION_STATE_PATH}")
    print("")
    print("Controls:")
    print("  SPACE  = hold event")
    print("  O/C/R  = OPEN / CLOSE / REST")
    print("  1–5    = assign finger")
    print("  A      = artifact")
    print("  N      = clear override")
    print("  Q or ESC = end stream safely")
    print("")
    print("Status:")
    print("  ✔ LSL EEG connected")
    print(f"  ✔ Time base aligned ({TIMEBASE_VERSION}, session-continuous)")
    print("  ✔ Resume-safe logging enabled")
    print("-" * 50)
    print("▶ Streaming started…")
    print("-" * 50)
    print("Type 'end_stream' into terminal OR press ESC/q to stop safely.")

    if INIT_ONLY:
        print("ℹ️ Init-only mode: exiting before LSL stream.")
        raise SystemExit(0)

    # Prepare output directories and metadata/state files.
    FEATURES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_RAW:
        RAW_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    if not true_resume:
        _maybe_backup_path(FEATURES_ARCHIVE_PATH, "Existing features file")
        _maybe_backup_path(EVENTS_ARCHIVE_PATH, "Existing events file")
        _maybe_backup_path(Path(EVENTS_AUTOSAVE_PATH), "Existing autosave events file")
        _maybe_backup_path(PREDICTIONS_ARCHIVE_PATH, "Existing predictions file")
        _maybe_backup_path(RAW_ARCHIVE_PATH, "Existing raw file")
        _maybe_backup_path(SESSION_META_PATH, "Existing session meta file")

    if not created_utc:
        created_utc = utc_now_iso_z()

    _update_session_meta(complete=False, label="init")
    _write_session_state(label="init")

    log_experiment(
        subject_id,
        experiment_hash,
        step="STEP_1_STREAM",
        notes="EEG collection + ICA + optional inference + event marking (session-continuous timebase)",
    )

# =========================
# ===== CALIBRATION STATE ==
# =========================
CALIBRATION_STATE_PATH = (
    Path("logs/calibration") / f"calibration_state_{subject_id}_{experiment_hash}.json"
)
CALIBRATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
calibrator = OnlineCalibrator()

if not IMPORT_ONLY and CALIBRATION_STATE_PATH.exists():
    try:
        state = json.loads(CALIBRATION_STATE_PATH.read_text())
        cfg = state.get("config", {})
        calibrator = OnlineCalibrator(
            init_threshold=state.get("threshold", cfg.get("init_threshold", 0.75)),
            min_threshold=cfg.get("min_threshold", 0.55),
            max_threshold=cfg.get("max_threshold", 0.90),
            ema_alpha=cfg.get("ema_alpha", 0.05),
        )
        print(f"🧠 Loaded calibration threshold: {calibrator.threshold:.2f}")
    except Exception as e:
        print(f"⚠️ Failed to load calibration state: {e}")

# =========================
# ===== CLEAN STOP ========
# =========================
stop_event = threading.Event()


def listen_for_exit():
    while not stop_event.is_set():
        try:
            if input().strip().lower() == "end_stream":
                print("\n🛑 end_stream received")
                state.soft_stop_triggered = True
                stop_event.set()
        except EOFError:
            break


if not IMPORT_ONLY:
    threading.Thread(target=listen_for_exit, daemon=True).start()


# =========================
# ===== TIMEBASE HELPERS ===
# =========================
def _lsl_now():
    """LSL-domain timestamp for 'now' using pylsl local_clock (already in LSL domain)."""
    if local_clock is None:
        return None
    return float(local_clock())


def _time_s_from_lsl(lsl_ts: float):
    """Stream-relative time_s from an LSL timestamp."""
    if stream_start_lsl_ts is None:
        return None
    return float(lsl_ts - stream_start_lsl_ts)


def validate_timestamp_sequence(
    timestamps, clamp_flags: Optional[Iterable[bool]] = None, epsilon: float = LSL_MONO_EPS_S
) -> Dict[str, Any]:
    ts = np.asarray(list(timestamps), dtype=float)
    if ts.size < 2:
        return {"monotonic_violations": 0, "clamp_count": 0, "fs_estimate": None}
    diffs = np.diff(ts)
    violations = int(np.sum(diffs <= 0))
    if clamp_flags is not None:
        clamp_count = int(np.sum([1 for flag in clamp_flags if flag]))
    else:
        clamp_count = int(np.sum(diffs <= float(epsilon)))
    pos = diffs[diffs > 0]
    fs_estimate = None
    if pos.size:
        median_dt = float(np.median(pos))
        if median_dt > 0:
            fs_estimate = float(1.0 / median_dt)
    return {
        "monotonic_violations": violations,
        "clamp_count": clamp_count,
        "fs_estimate": fs_estimate,
    }


def _raw_header(channel_count: int) -> List[str]:
    return [
        "lsl_timestamp_raw",
        "lsl_timestamp_mono",
        "local_timestamp",
        *[f"ch{idx + 1}" for idx in range(channel_count)],
        "flags",
    ]


def _raw_flags_for_sample(sample: Iterable[float]) -> int:
    try:
        if not np.all(np.isfinite(sample)):
            return RAW_FLAG_NONFINITE
    except Exception:
        return RAW_FLAG_NONFINITE
    return 0


def _build_raw_row(
    lsl_ts_raw: float,
    lsl_ts_mono: float,
    local_ts: float,
    sample: Iterable[float],
    flags: int,
) -> List[Any]:
    return [lsl_ts_raw, lsl_ts_mono, local_ts, *sample, int(flags)]


def _gap_tolerance_s(nominal_dt_s: float) -> float:
    return float(nominal_dt_s) * float(INTEGRITY_GAP_TOLERANCE_MULT)


def _estimate_missing_samples(
    dt_s: float, nominal_dt_s: float, gap_tolerance_s: float
) -> int:
    if nominal_dt_s <= 0 or dt_s <= gap_tolerance_s:
        return 0
    return max(0, int(round(float(dt_s) / float(nominal_dt_s))) - 1)


def analyze_lsl_timestamp_gaps(
    timestamps: Iterable[float], nominal_fs: float
) -> Dict[str, Any]:
    finite_ts = [float(ts) for ts in timestamps if ts is not None and np.isfinite(ts)]
    if len(finite_ts) < 2 or nominal_fs <= 0:
        return {
            "duration_s": 0.0,
            "expected_samples": 0,
            "gap_count": 0,
            "estimated_missing": 0,
            "max_gap_s": None,
        }
    nominal_dt = 1.0 / float(nominal_fs)
    tolerance = _gap_tolerance_s(nominal_dt)
    gap_count = 0
    missing_total = 0
    max_gap_s = 0.0
    prev = finite_ts[0]
    for ts in finite_ts[1:]:
        dt = float(ts - prev)
        if dt > tolerance:
            gap_count += 1
            missing_total += _estimate_missing_samples(dt, nominal_dt, tolerance)
            max_gap_s = max(max_gap_s, dt)
        prev = ts
    duration_s = float(finite_ts[-1] - finite_ts[0])
    expected_samples = int(round(float(nominal_fs) * duration_s))
    return {
        "duration_s": duration_s,
        "expected_samples": expected_samples,
        "gap_count": gap_count,
        "estimated_missing": missing_total,
        "max_gap_s": max_gap_s if gap_count else None,
    }


def _event_time_ok(event_time_s: float) -> bool:
    if event_time_s is None or not np.isfinite(event_time_s):
        return False
    if last_sample_time_s is None:
        return False
    if event_time_s < (last_sample_time_s - EVENT_MAX_LAG_S):
        return False
    if event_time_s > (last_sample_time_s + EVENT_MAX_LEAD_S):
        return False
    return True


def _record_nearest_sample_delta(event_time_s: float, label: str):
    if (
        event_time_s is None
        or len(state.nearest_sample_delta_samples) >= NEAREST_SAMPLE_MAX
    ):
        return
    if not state.recent_sample_times:
        return
    recent = list(state.recent_sample_times)
    nearest = min(recent, key=lambda t: abs(t - event_time_s))
    delta = float(event_time_s - nearest)
    state.nearest_sample_delta_samples.append(
        {
            "label": label,
            "event_time_s": float(event_time_s),
            "nearest_sample_s": float(nearest),
            "delta_s": float(delta),
        }
    )


def _timebase_report_payload():
    dt_median_ms = None
    dt_p95_ms = None
    if len(state.recent_sample_times) >= 2:
        diffs = np.diff(np.array(state.recent_sample_times, dtype=float))
        if diffs.size:
            dt_median_ms = float(np.median(diffs) * 1000.0)
            dt_p95_ms = float(np.percentile(diffs, 95) * 1000.0)

    delta_abs = [abs(s["delta_s"]) for s in state.nearest_sample_delta_samples]
    delta_abs_max = float(max(delta_abs)) if delta_abs else None
    delta_abs_mean = float(np.mean(delta_abs)) if delta_abs else None
    event_times = [s["event_time_s"] for s in state.nearest_sample_delta_samples]
    selfcheck = evaluate_timebase_alignment(
        state.recent_sample_times,
        event_times,
        warn_threshold_s=TIMEBASE_SELF_CHECK_WARN_S,
        error_threshold_s=TIMEBASE_SELF_CHECK_ERROR_S,
    )

    return {
        "ts_utc": utc_now_iso_z(),
        "subject_id": subject_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "timebase_version": TIMEBASE_VERSION,
        "stream_start_lsl_ts": stream_start_lsl_ts,
        "segment_start_lsl_ts": segment_start_lsl_ts,
        "stream_start_local": local_clock_at_start,
        "clock_offset": clock_offset,
        "gap_count": state.gap_count,
        "gap_max_s": state.gap_max_s if state.gap_count else None,
        "gap_p95_s": state.gap_p95_s,
        "gap_p99_s": state.gap_p99_s,
        "backward_timestamp_count": state.backward_timestamp_count,
        "samples_seen": state.samples_seen,
        "samples_written": state.samples_written,
        "time_s_clamped_count": int(time_s_clamped_count),
        "max_backwards_jump_s": float(max_backwards_jump_s),
        "event_stamps_count": int(event_stamps_count),
        "event_clamped_count": int(event_clamped_count),
        "nearest_sample_delta_s_abs_max": delta_abs_max,
        "nearest_sample_delta_s_abs_mean": delta_abs_mean,
        "nearest_sample_delta_s_samples": list(state.nearest_sample_delta_samples),
        "timebase_selfcheck": {
            "max_abs_delta_s": selfcheck.max_abs_delta_s,
            "mean_abs_delta_s": selfcheck.mean_abs_delta_s,
            "warn_threshold_s": selfcheck.warn_threshold_s,
            "error_threshold_s": selfcheck.error_threshold_s,
            "warn": selfcheck.warn,
            "error": selfcheck.error,
        },
        "dt_median_ms": dt_median_ms,
        "dt_p95_ms": dt_p95_ms,
        "first_time_s": first_time_s,
        "last_time_s": last_time_s_seen,
        "first_lsl_ts": first_lsl_ts,
        "last_lsl_ts": last_lsl_ts,
    }


def _write_timebase_report(label: str):
    report_path = (
        FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_timebase_report.json"
    )
    pointer_path = Path("reports") / "last_timebase_report.json"
    try:
        payload = _timebase_report_payload()
        payload["label"] = label
        selfcheck = payload.get("timebase_selfcheck", {})
        if selfcheck.get("error"):
            print(
                "⚠️ Timebase self-check error: event/sample alignment exceeds threshold."
            )
        elif selfcheck.get("warn"):
            print(
                "⚠️ Timebase self-check warning: event/sample alignment drift detected."
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(report_path, payload)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(pointer_path, payload)
    except Exception as exc:
        print(f"⚠️ Failed to write timebase report: {exc}")


def _maybe_write_timebase_report(force: bool = False, label: str = "periodic"):
    global last_report_time, last_report_samples_written
    now = time.time()
    if force or last_report_time is None:
        _write_timebase_report(label)
        _write_session_state(label=f"timebase_report:{label}")
        last_report_time = now
        last_report_samples_written = state.samples_written
        return
    if (now - last_report_time) >= 10.0 or (
        state.samples_written - last_report_samples_written
    ) >= 2500:
        _write_timebase_report(label)
        _write_session_state(label=f"timebase_report:{label}")
        last_report_time = now
        last_report_samples_written = state.samples_written


def _compute_timebase_health(now_monotonic: float) -> Dict[str, Any]:
    ts_list = list(recent_accepted_timestamps)
    clamp_flags = list(recent_accepted_clamped_flags)
    diffs = np.diff(np.asarray(ts_list, dtype=float)) if len(ts_list) >= 2 else np.array([])
    last_dt_ms = float(diffs[-1] * 1000.0) if diffs.size else None
    min_dt_ms = float(np.min(diffs) * 1000.0) if diffs.size else None
    max_dt_ms = float(np.max(diffs) * 1000.0) if diffs.size else None
    validation = validate_timestamp_sequence(
        ts_list, clamp_flags=clamp_flags, epsilon=LSL_MONO_EPS_S
    )
    lsl_gap_ms = (
        float((now_monotonic - last_any_sample_received_time) * 1000.0)
        if last_any_sample_received_time is not None
        else None
    )
    processing_lag_ms = None
    if last_lsl_ts_mono is not None:
        try:
            processing_lag_ms = float((local_clock() - last_lsl_ts_mono) * 1000.0)
        except Exception:
            processing_lag_ms = None

    return {
        "monotonic_ok": validation["monotonic_violations"] == 0,
        "clamped_samples_total": int(clamped_samples_total),
        "backwards_detected_total": int(total_backward_timestamp_count),
        "last_dt_ms": last_dt_ms,
        "min_dt_ms": min_dt_ms,
        "max_dt_ms": max_dt_ms,
        "lsl_gap_ms": lsl_gap_ms,
        "processing_lag_ms": processing_lag_ms,
        "write_rate_samples_per_s": float(stream_health_write_rate),
        "last_clamp_localtime": last_clamp_localtime,
        "last_clamp_lsl_ts": last_clamp_lsl_ts,
    }


def _maybe_write_timebase_health(now_monotonic: float, reason: str = "periodic") -> None:
    global last_timebase_health_write, timebase_health_snapshot
    timebase_health_snapshot = _compute_timebase_health(now_monotonic)
    if last_timebase_health_write is None:
        last_timebase_health_write = float(now_monotonic)
    if (now_monotonic - last_timebase_health_write) >= 1.0:
        _write_session_state(label=f"timebase_health:{reason}")
        last_timebase_health_write = float(now_monotonic)


def _write_debug_timebase_dump(reason: str, now_monotonic: float, decision) -> None:
    report_dir = Path("logs")
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp()
    report_path = report_dir / f"debug_timebase_{subject_id}_{session_id}_{stamp}.json"
    ts_list = list(recent_accepted_timestamps)
    clamp_flags = list(recent_accepted_clamped_flags)
    clamp_indices = [idx for idx, flag in enumerate(clamp_flags) if flag]
    validation = validate_timestamp_sequence(
        ts_list, clamp_flags=clamp_flags, epsilon=LSL_MONO_EPS_S
    )
    payload = {
        "reason": reason,
        "timestamp_utc": utc_now_iso_z(),
        "timestamp_monotonic": float(now_monotonic),
        "subject_id": subject_id,
        "session_id": session_id,
        "segment_id": int(segment_id),
        "last_accepted_timestamps": ts_list,
        "clamp_indices": clamp_indices,
        "tp9_values_decimated": list(recent_tp9_values),
        "measured_fs": decision.measured_fs if decision else None,
        "fs_estimate": validation.get("fs_estimate"),
        "queue_size": int(stream_health_queue_size),
        "write_rate": float(stream_health_write_rate),
        "samples_received": int(state.samples_seen),
        "samples_written": int(state.samples_written),
        "last_received_lsl_ts": last_received_lsl_ts,
        "last_written_lsl_ts": last_written_lsl_ts,
        "last_any_sample_received_time": last_any_sample_received_time,
        "last_sample_written_time": last_sample_written_time,
        "last_lsl_pull_monotonic_time": last_lsl_pull_monotonic_time,
        "clamped_samples_total": int(clamped_samples_total),
        "backwards_detected_total": int(total_backward_timestamp_count),
        "timebase_health": timebase_health_snapshot,
    }
    report_path.write_text(json.dumps(_to_jsonable(payload), indent=2))


def _raw_integrity_payload() -> Dict[str, Any]:
    duration_s = 0.0
    expected_samples = 0
    if first_received_lsl_ts is not None and last_received_lsl_ts is not None:
        duration_s = float(last_received_lsl_ts - first_received_lsl_ts)
        if duration_s < 0:
            duration_s = 0.0
        expected_samples = int(round(float(SAMPLING_RATE) * duration_s))
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "segment_id": int(segment_id),
        "timestamp_utc": utc_now_iso_z(),
        "nominal_fs": float(SAMPLING_RATE),
        "duration_s": duration_s,
        "expected_samples": expected_samples,
        "samples_received": int(state.samples_seen),
        "samples_written": int(state.samples_written),
        "gap_count": int(integrity_gap_count),
        "estimated_missing": int(integrity_missing_estimate),
        "max_gap_s": float(integrity_gap_max_s) if integrity_gap_count else None,
        "nonfinite_count": int(raw_nonfinite_count),
        "nonfinite_written": int(raw_written_nonfinite_count),
        "queue_overflow_count": int(
            raw_queue_overflow_count + processing_queue_overflow_count
        ),
        "raw_queue_overflow_count": int(raw_queue_overflow_count),
        "processing_queue_overflow_count": int(processing_queue_overflow_count),
        "max_queue_size_observed": int(max_queue_size_observed),
        "max_raw_queue_size_observed": int(max_raw_queue_size_observed),
        "max_processing_queue_size_observed": int(max_processing_queue_size_observed),
        "first_received_lsl_ts": first_received_lsl_ts,
        "last_received_lsl_ts": last_received_lsl_ts,
        "first_written_lsl_ts": first_written_lsl_ts,
        "last_written_lsl_ts": last_written_lsl_ts,
    }


def _write_raw_integrity_report() -> Optional[Path]:
    report_dir = Path("logs")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / (
        f"raw_integrity_{subject_id}_{session_id or 'UNKNOWN'}_seg{segment_id:02d}.json"
    )
    payload = _raw_integrity_payload()
    report_path.write_text(json.dumps(_to_jsonable(payload), indent=2))
    summary = (
        f"raw integrity: duration={payload['duration_s']:.2f}s "
        f"expected≈{payload['expected_samples']} "
        f"received={payload['samples_received']} "
        f"written={payload['samples_written']} "
        f"gaps={payload['gap_count']} "
        f"missing≈{payload['estimated_missing']} "
        f"nonfinite={payload['nonfinite_count']} "
        f"queue_overflows={payload['queue_overflow_count']}"
    )
    print(f"ℹ️ {summary} (report={report_path})")
    return report_path

# =========================
# ===== EVENT MARKING =====
# =========================
events: List[Dict[str, Any]] = []
events_lock = threading.Lock()
current_event: Optional[Dict[str, Any]] = None
last_event_index: Optional[int] = None
current_action_id = ACTION_REST
current_override = None

trial_id_counter = 0
last_written_time_s = float(last_time_s)
timebase_written = False
clock_offset_estimated = False
time_s_clamped_count = int(time_s_clamped_count)
event_clamped_count = int(event_clamped_count)
event_stamps_count = 0
last_sample_time_s: Optional[float] = None
last_sample_lsl_ts: Optional[float] = None
last_written_lsl_ts: Optional[float] = None
last_received_lsl_ts: Optional[float] = None
last_lsl_ts_mono: Optional[float] = None
last_lsl_ts_raw: Optional[float] = None
gap_durations_s: List[float] = []
first_time_s: Optional[float] = None
last_time_s_seen: Optional[float] = None
first_lsl_ts: Optional[float] = None
last_lsl_ts: Optional[float] = None
max_backwards_jump_s = 0.0
NEAREST_SAMPLE_MAX = 10
last_report_time: Optional[float] = None
last_report_samples_written = 0
timebase_report_initialized = False
last_live_viz_emit = 0.0
non_finite_sample_warned = False
stats_lock = threading.Lock()
health_gate_lock = threading.Lock()
failed_writers_lock = threading.Lock()
processing_queue: queue.Queue = queue.Queue(maxsize=int(PROCESSING_QUEUE_MAXSIZE))
raw_queue: queue.Queue = queue.Queue(maxsize=int(RAW_QUEUE_MAXSIZE))
processing_queue_overflow_count = 0
raw_queue_overflow_count = 0
max_processing_queue_size_observed = 0
max_raw_queue_size_observed = 0
max_queue_size_observed = 0
raw_nonfinite_count = 0
raw_written_count = 0
raw_written_nonfinite_count = 0
integrity_gap_count = 0
integrity_missing_estimate = 0
integrity_gap_max_s = 0.0
first_received_lsl_ts: Optional[float] = None
first_written_lsl_ts: Optional[float] = None
raw_writer_thread_error: Optional[Exception] = None
acquisition_thread_error: Optional[Exception] = None
backwards_events_monotonic: Deque[float] = deque(maxlen=256)
segment_break_hold_until: Optional[float] = None
last_health_warning_time = 0.0
nominal_dt_s = 1.0 / float(SAMPLING_RATE)
gap_threshold = gap_threshold_s(nominal_dt_s)
next_window_start_s: Optional[float] = None
lstm_state = None
last_pred_action = -1
last_pred_finger = -1
last_action_confidence = 0.0
last_action_uncertainty = 0.0
last_finger_confidence = 0.0
last_finger_uncertainty = 0.0
clamped_samples_total = 0
last_clamp_localtime: Optional[float] = None
last_clamp_lsl_ts: Optional[float] = None
last_lsl_pull_monotonic_time: Optional[float] = None
last_any_sample_received_time: Optional[float] = None
last_any_sample_received_lsl_ts: Optional[float] = None
last_sample_written_time: Optional[float] = None
recent_accepted_timestamps: Deque[float] = deque(maxlen=200)
recent_accepted_clamped_flags: Deque[bool] = deque(maxlen=200)
recent_tp9_values: Deque[float] = deque(maxlen=200)
debug_sample_counter = 0
timebase_health_snapshot: Dict[str, Any] = {}
last_timebase_health_write: Optional[float] = None
soft_stop_report_path: Optional[Path] = None
last_debug_dump_time: Optional[float] = None
last_debug_dump_reason: Optional[str] = None
last_received_lsl_ts_for_health: Optional[float] = None


def _apply_segment_paths(seg_id: int) -> None:
    global FEATURES_ARCHIVE_PATH, EVENTS_ARCHIVE_PATH, EVENTS_AUTOSAVE_PATH
    global EVENTS_CSV_PATH, FEATURES_PATH, RAW_ARCHIVE_PATH, PREDICTIONS_ARCHIVE_PATH
    features, events, autosave, predictions, raw = _segment_paths(seg_id)
    FEATURES_ARCHIVE_PATH = features
    EVENTS_ARCHIVE_PATH = events
    EVENTS_AUTOSAVE_PATH = str(autosave)
    EVENTS_CSV_PATH = str(events)
    FEATURES_PATH = FEATURES_ARCHIVE_PATH
    PREDICTIONS_ARCHIVE_PATH = predictions
    RAW_ARCHIVE_PATH = raw
    state.features_path = str(FEATURES_ARCHIVE_PATH)
    state.events_path = str(EVENTS_ARCHIVE_PATH)
    state.predictions_path = str(PREDICTIONS_ARCHIVE_PATH)
    state.raw_path = str(RAW_ARCHIVE_PATH)


def _reset_segment_state(state: StreamState) -> None:
    global segment_start_lsl_ts, last_written_lsl_ts, last_sample_time_s
    global last_sample_lsl_ts, first_time_s, last_time_s_seen
    global first_lsl_ts, last_lsl_ts, last_report_time
    global stream_start_lsl_ts, local_clock_at_start, clock_offset
    global last_lsl_ts_mono, last_lsl_ts_raw
    global gap_durations_s, next_window_start_s
    global backwards_events_monotonic
    global lstm_state
    global last_report_samples_written, timebase_report_initialized
    global last_live_viz_emit
    global ica_scaler, ica
    global recent_accepted_timestamps, recent_accepted_clamped_flags, recent_tp9_values
    global debug_sample_counter
    segment_start_lsl_ts = None
    stream_start_lsl_ts = None
    local_clock_at_start = None
    clock_offset = None
    state.stream_start_lsl_ts = None
    state.local_clock_at_start = None
    state.clock_offset = None
    last_written_lsl_ts = None
    last_sample_time_s = None
    last_sample_lsl_ts = None
    last_lsl_ts_mono = None
    last_lsl_ts_raw = None
    state.backward_timestamp_count = 0
    gap_durations_s = []
    backwards_events_monotonic.clear()
    state.gap_count = 0
    state.gap_max_s = 0.0
    state.gap_p95_s = None
    state.gap_p99_s = None
    first_time_s = None
    last_time_s_seen = None
    first_lsl_ts = None
    last_lsl_ts = None
    state.nearest_sample_delta_samples.clear()
    state.recent_sample_times.clear()
    state.action_pred_buffer.clear()
    state.sample_time_buffer.clear()
    next_window_start_s = None
    state.window_drop_count = 0
    state.window_gap_drop_count = 0
    state.window_incomplete_drop_count = 0
    state.window_health_drop_count = 0
    lstm_state = None
    state.lstm_reset_count = 0
    state.lstm_reset_log = []
    # Reset per-segment reporting/viz gates (must not leak across segments)
    last_report_time = 0.0
    last_report_samples_written = 0
    timebase_report_initialized = False
    last_live_viz_emit = 0.0
    recent_accepted_timestamps.clear()
    recent_accepted_clamped_flags.clear()
    recent_tp9_values.clear()
    debug_sample_counter = 0
    if state.ica_fit_future is not None:
        state.ica_fit_future.cancel()
        state.ica_fit_future = None
    if state.ica_transform_future is not None:
        state.ica_transform_future.cancel()
        state.ica_transform_future = None
    state.ica_fit_segment_id = None
    state.ica_transform_segment_id = None
    state.ica_fitted = False
    if ENABLE_ICA:
        ica_scaler = StandardScaler()
        ica = FastICA(n_components=CHANNELS, random_state=42)
    else:
        ica_scaler = None
        ica = None


def _resolve_events_output_path(
    now_monotonic: Optional[float], decision
) -> Optional[str]:
    if (
        not health_state.has_health_decision
        or decision is None
        or now_monotonic is None
    ):
        return None
    if (
        health_state.label_check_status
        and not health_state.label_check_status.get("ok")
        and not label_check_acknowledged
    ):
        return None
    if not decision.healthy:
        if (
            health_state.failed_write_until_mono is not None
            and now_monotonic <= health_state.failed_write_until_mono
        ):
            return str(failed_writers.events_path) if failed_writers.events_path else None
        return None
    if decision.event_allowed:
        if (
            health_state.failed_write_until_mono is None
            or now_monotonic >= health_state.failed_write_until_mono
        ):
            return EVENTS_CSV_PATH
    return None


def _flush_events_for_segment(now_monotonic: Optional[float] = None) -> None:
    global current_event, last_event_index
    if current_event is not None:
        current_event = None
    with events_lock:
        if events:
            output_path = _resolve_events_output_path(now_monotonic, last_health_decision)
            if output_path:
                save_events_csv(output_path, events)
                last_event_index = len(events) - 1


def _start_segment(reason: str, *, segment_id_override: Optional[int] = None) -> None:
    global segment_id, events, last_event_index
    global last_written_time_s, last_time_s, total_elapsed_s
    global segment_break_hold_until

    _flush_events_for_segment(time.monotonic())

    if last_written_time_s >= 0:
        total_elapsed_s += float(last_written_time_s)
    last_written_time_s = -1.0
    last_time_s = -1.0

    _close_segment_files()
    if segment_id_override is not None:
        segment_id = int(segment_id_override)
    else:
        segment_id += 1
    state.segment_id = segment_id
    _apply_segment_paths(segment_id)
    _reset_segment_state(state)
    failed_writers.close_failed_files()
    events = []
    last_event_index = None
    segment_break_hold_until = time.monotonic() + float(RECOVERY_S)
    _write_session_state(label=f"segment_break:{reason}", segment_id_override=segment_id)
    print(f"⚠️ Segment break ({reason}); now writing segment {segment_id:02d}")


def _load_existing_events(path: Path):
    """Load existing events to support resume without overwriting."""
    loaded: List[Dict[str, Any]] = []
    if not path.exists() or path.stat().st_size == 0:
        return loaded
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not row:
                    continue

                def _f(key, default=np.nan):
                    v = row.get(key, "")
                    if v is None or v == "":
                        return default
                    try:
                        return float(v)
                    except Exception:
                        return default

                def _i(key, default=0):
                    v = row.get(key, "")
                    if v is None or v == "":
                        return default
                    try:
                        return int(float(v))
                    except Exception:
                        return default

                e = dict(row)
                onset_lsl = _f("onset_lsl", np.nan)
                onset_s = _f("onset_s", np.nan)
                duration_s = _f("duration_s", 0.0)
                end_lsl = _f("end_lsl", np.nan)
                end_s = _f("end_s", np.nan)
                event_lsl_ts = _f("event_lsl_ts", np.nan)
                event_time_s = _f("event_time_s", np.nan)
                if np.isfinite(onset_s):
                    event_time_s = onset_s

                if duration_s < 0:
                    duration_s = 0.0
                if np.isfinite(onset_s) and not np.isfinite(end_s):
                    end_s = onset_s + duration_s
                if np.isfinite(onset_lsl) and not np.isfinite(end_lsl):
                    end_lsl = onset_lsl + duration_s

                # IMPORTANT: In stream-relative mode, we do NOT try to back-compute onset_lsl
                # from onset_s unless the file already contains onset_lsl.
                e["onset_lsl"] = onset_lsl
                e["onset_s"] = onset_s
                e["duration_s"] = duration_s
                e["end_lsl"] = end_lsl
                e["end_s"] = end_s
                e["event_lsl_ts"] = event_lsl_ts
                e["event_time_s"] = event_time_s

                if "onset_rel_s" in row:
                    e["onset_rel_s"] = _f("onset_rel_s", np.nan)
                if "end_rel_s" in row:
                    e["end_rel_s"] = _f("end_rel_s", np.nan)

                e["finger_id"] = _i("finger_id", 0)
                e["action_id"] = _i("action_id", 0)
                e["trial_id"] = _i("trial_id", 0)
                e["block_id"] = _i("block_id", 0)
                e["segment_id"] = _i("segment_id", segment_id)
                e["type"] = str(row.get("type", "")).strip()
                e["channel"] = str(row.get("channel", "n/a")).strip() or "n/a"
                e["source"] = str(row.get("source", "manual")).strip() or "manual"
                e["notes"] = str(row.get("notes", "")).strip()
                e["confidence"] = row.get("confidence", "")
                loaded.append(e)
    except Exception as e:
        print(f"⚠️ Could not load existing events from {path}: {e}")
        return []
    return loaded


# Resume-safe: load any existing events so we don't overwrite them on exit
if true_resume:
    loaded_events = _load_existing_events(EVENTS_ARCHIVE_PATH)
    if loaded_events:
        with events_lock:
            events.extend(loaded_events)
            last_event_index = len(events) - 1
        trial_id_counter = max(
            [int(e.get("trial_id", 0) or 0) for e in loaded_events] + [0]
        )
        print(
            f"🧾 Resumed events: loaded {len(loaded_events)} existing events (trial_id_counter={trial_id_counter})."
        )


def save_events_csv(path, items):
    """
    Writes events with a stable base header, plus any extra columns (debug) appended.
    Step 1b will ignore extra columns.
    """
    base_header = [
        "onset_lsl",
        "onset_s",
        "duration_s",
        "end_lsl",
        "end_s",
        "event_lsl_ts",
        "event_time_s",
        "type",
        "channel",
        "confidence",
        "notes",
        "finger_id",
        "action_id",
        "trial_id",
        "segment_id",
        "block_id",
        "source",
    ]

    extras = []
    extra_candidates = ["onset_rel_s", "end_rel_s"]
    for k in extra_candidates:
        if any((k in it) for it in items):
            extras.append(k)

    header = base_header + extras

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in items:

            def _f(val, default=np.nan):
                try:
                    if val is None or val == "":
                        return default
                    return float(val)
                except Exception:
                    return default

            onset_lsl = _f(item.get("onset_lsl"), np.nan)
            onset_s = _f(item.get("onset_s"), np.nan)
            duration_s = _f(item.get("duration_s"), 0.0)
            end_lsl = _f(item.get("end_lsl"), np.nan)
            end_s = _f(item.get("end_s"), np.nan)
            event_time_s = onset_s if np.isfinite(onset_s) else _f(item.get("event_time_s"), np.nan)

            if duration_s < 0:
                duration_s = 0.0
            if not np.isfinite(end_lsl) and np.isfinite(onset_lsl):
                end_lsl = onset_lsl + duration_s
            if not np.isfinite(end_s) and np.isfinite(onset_s):
                end_s = onset_s + duration_s

            row = [
                f"{float(onset_lsl):.6f}" if np.isfinite(onset_lsl) else "",
                f"{float(onset_s):.6f}" if np.isfinite(onset_s) else "",
                f"{float(duration_s):.6f}",
                f"{float(end_lsl):.6f}" if np.isfinite(end_lsl) else "",
                f"{float(end_s):.6f}" if np.isfinite(end_s) else "",
                f"{float(_f(item.get('event_lsl_ts'), np.nan)):.6f}"
                if np.isfinite(_f(item.get("event_lsl_ts"), np.nan))
                else "",
                f"{float(event_time_s):.6f}" if np.isfinite(event_time_s) else "",
                item.get("type", ""),
                item.get("channel", "n/a"),
                item.get("confidence", ""),
                item.get("notes", ""),
                int(item.get("finger_id", 0)),
                int(item.get("action_id", 0)),
                int(item.get("trial_id", 0)),
                int(item.get("segment_id", segment_id)),
                int(item.get("block_id", 0)),
                item.get("source", "manual"),
            ]
            for k in extras:
                v = item.get(k, "")
                if isinstance(v, (float, int)) and np.isfinite(v):
                    row.append(f"{float(v):.4f}")
                else:
                    row.append("" if v is None else str(v))
            writer.writerow(row)

    if EVENTS_ARCHIVE_PATH and str(path) != str(EVENTS_ARCHIVE_PATH):
        FEATURES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, EVENTS_ARCHIVE_PATH)


def finalize_event():
    global current_event, last_event_index, trial_id_counter, event_stamps_count
    if current_event is None:
        return

    duration_s = float(current_event.get("duration_s", 0.0))
    if duration_s < 0:
        duration_s = 0.0
    current_event["duration_s"] = duration_s

    onset_lsl = current_event.get("onset_lsl", np.nan)
    onset_s = current_event.get("onset_s", np.nan)
    if np.isfinite(onset_s):
        current_event["event_time_s"] = float(onset_s)

    if (
        current_event.get("end_lsl") is None
        or not np.isfinite(current_event.get("end_lsl", np.nan))
    ) and np.isfinite(onset_lsl):
        current_event["end_lsl"] = float(onset_lsl + duration_s)
    if (
        current_event.get("end_s") is None
        or not np.isfinite(current_event.get("end_s", np.nan))
    ) and np.isfinite(onset_s):
        current_event["end_s"] = float(onset_s + duration_s)

    current_event["type"] = event_type_for(
        int(current_event["action_id"]),
        int(current_event["finger_id"]),
        current_event.get("override_type"),
    )

    trial_id_counter += 1
    current_event["trial_id"] = int(trial_id_counter)
    current_event["block_id"] = int(BLOCK_ID)
    current_event["segment_id"] = int(segment_id)

    with events_lock:
        events.append(current_event)
        last_event_index = len(events) - 1
        save_events_csv(EVENTS_AUTOSAVE_PATH, events)

    event_stamps_count += 1
    _record_nearest_sample_delta(current_event.get("onset_s"), label="onset")

    current_event = None


def update_last_event_finger(finger_id):
    global last_event_index
    with events_lock:
        if last_event_index is None or last_event_index >= len(events):
            return
        event = events[last_event_index]
        event["finger_id"] = int(finger_id)
        event["override_type"] = None
        event["type"] = event_type_for(int(event["action_id"]), int(event["finger_id"]))
        save_events_csv(EVENTS_AUTOSAVE_PATH, events)


def on_key_press(key):
    global \
        current_event, \
        current_action_id, \
        current_override
    if not EVENT_MARKING_ENABLED:
        return

    try:
        key_char = key.char
    except AttributeError:
        key_char = None

    # Hold SPACE = event active
    if key == keyboard.Key.space:
        if not event_marking_allowed:
            return
        if current_event is None:
            # Refuse to stamp events until stream_start_lsl_ts exists,
            # because that's the anchor for the stream-relative timebase.
            if stream_start_lsl_ts is None or clock_offset is None:
                print(
                    "⚠️ Event ignored: timebase not initialized yet (waiting for first LSL sample)."
                )
                return

            event_local = local_clock()
            onset_lsl = _lsl_now()
            if onset_lsl is None:
                return

            if stream_start_lsl_ts is None:
                return
            onset_s = float(onset_lsl - stream_start_lsl_ts)
            if onset_s is None or not _event_time_ok(onset_s):
                return

            current_event = {
                "onset_lsl": float(onset_lsl),
                "onset_s": float(onset_s),
                "duration_s": 0.0,
                "end_lsl": np.nan,
                "end_s": np.nan,
                "event_lsl_ts": float(onset_lsl),
                "event_time_s": float(onset_s),
                "type": "",
                "channel": EVENTS_CHANNEL,
                "confidence": "",
                "notes": "",
                "finger_id": int(FINGER_NONE),
                "action_id": int(current_action_id),
                "override_type": current_override,
                "segment_id": int(segment_id),
                "source": "manual",
            }

            # Debug: local-clock relative to run start (NOT used by Step 1b)
            if local_clock_at_start is not None:
                current_event["onset_rel_s"] = float(
                    event_local - local_clock_at_start
                )
        return

    if key_char is None:
        return

    key_char = key_char.lower()

    # Quick stop hotkey
    if key == keyboard.Key.esc or key_char == "q":
        print("🛑 Stop requested (ESC/q).")
        state.soft_stop_triggered = True
        stop_event.set()
        return

    # Action modes
    if key_char in {"o", "c", "r"}:
        if key_char == "o":
            current_action_id = ACTION_OPEN
            current_override = None
        elif key_char == "c":
            current_action_id = ACTION_CLOSE
            current_override = None
        else:
            current_action_id = ACTION_REST
            current_override = "rest"
        print(f"📝 Action mode: {ACTION_NAMES[current_action_id]}")
        return

    # Overrides
    if key_char == "a":
        current_action_id = ACTION_REST
        current_override = "artifact"
        print("📝 Event override: artifact")
        return

    if key_char == "k":
        current_action_id = ACTION_REST
        current_override = "calibration"
        print("📝 Event override: calibration")
        return

    if key_char == "n":
        current_override = None
        print("📝 Event override cleared")
        return

    # Assign finger to last completed event (0-5)
    if key_char in {"0", "1", "2", "3", "4", "5"}:
        finger_id = int(key_char)
        update_last_event_finger(finger_id)
        print(f"📝 Finger assigned: {FINGER_NAMES.get(finger_id, 'UNKNOWN')}")


def on_key_release(key):
    global current_event
    if not EVENT_MARKING_ENABLED:
        return
    if key == keyboard.Key.space and current_event is not None:
        if not event_marking_allowed:
            current_event = None
            return
        if stream_start_lsl_ts is None or clock_offset is None:
            return

        end_local = local_clock()
        end_lsl = _lsl_now()
        if end_lsl is None:
            return

        onset_lsl = current_event.get("onset_lsl", np.nan)
        if not np.isfinite(onset_lsl):
            return

        if stream_start_lsl_ts is None:
            return
        end_s = float(end_lsl - stream_start_lsl_ts)
        if end_s is None or not _event_time_ok(end_s):
            return

        duration_s = float(end_s - current_event.get("onset_s", end_s))
        if duration_s < 0:
            duration_s = 0.0

        current_event["duration_s"] = duration_s
        current_event["end_lsl"] = float(end_lsl)
        current_event["end_s"] = float(end_s)

        if local_clock_at_start is not None:
            current_event["end_rel_s"] = float(end_local - local_clock_at_start)

        finalize_event()


if EVENT_MARKING_ENABLED and keyboard is None:
    print("⚠️ Event marking disabled (pynput not installed).")
    EVENT_MARKING_ENABLED = False

listener = None
event_marking_active = False


def start_event_listener() -> None:
    global listener, event_marking_active
    if (
        not EVENT_MARKING_ENABLED
        or keyboard is None
        or event_marking_active
        or not data_stream_active
        or not event_marking_allowed
    ):
        return
    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.daemon = True
    listener.start()
    event_marking_active = True


def stop_event_listener() -> None:
    global listener, event_marking_active
    if listener is None:
        return
    try:
        listener.stop()
    except Exception:
        pass
    listener = None
    event_marking_active = False


if EVENT_MARKING_ENABLED and not IMPORT_ONLY:
    start_event_listener()

# =========================
# ===== MODEL (MC DROPOUT) =====
# =========================
model = None
scaler = None


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


def mc_dropout_predict(model, x_BTC, passes: int):
    was_training = model.training
    model.train()
    probs_f, probs_a = [], []
    with torch.no_grad():
        for _ in range(passes):
            finger_logits, action_logits = model(x_BTC)
            probs_f.append(torch.softmax(finger_logits, dim=1))
            probs_a.append(torch.softmax(action_logits, dim=1))
    if not was_training:
        model.eval()
    probs_f = torch.stack(probs_f, dim=0)
    probs_a = torch.stack(probs_a, dim=0)
    return {
        "finger_mean": probs_f.mean(dim=0),
        "finger_std": probs_f.std(dim=0),
        "action_mean": probs_a.mean(dim=0),
        "action_std": probs_a.std(dim=0),
    }


model = None
scaler = None
if DEMO_MODE:
    # In lab workflows it's common to start streaming/recording before a
    # trained model exists. Do not hard-fail the entire session if the model
    # artifact is missing; instead disable inference-related features and
    # continue with raw/feature logging.
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Model file not found: {MODEL_PATH}")
        print("⚠️  Continuing without inference/actuation (recording still runs).")
        ENABLE_ACTUATION = False
        DEMO_MODE = False
        model = None
        scaler = None
    else:
        model = CNNLSTMFingerActionNet(
            n_channels=CHANNELS, n_fingers=N_FINGERS, n_actions=N_ACTIONS
        ).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.train()  # keep dropout active
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)


# =========================
# ===== LSL SETUP =========
# =========================
def _drain_inlet(
    inlet: StreamInlet,
    *,
    label: str,
    max_samples: Optional[int] = None,
    max_seconds: Optional[float] = None,
) -> int:
    drained = 0
    max_samples = int(DRAIN_MAX_SAMPLES if max_samples is None else max_samples)
    max_seconds = float(DRAIN_MAX_SECONDS if max_seconds is None else max_seconds)
    start = time.monotonic()
    idle_start = None
    stop_reason = "max_seconds"

    while True:
        now = time.monotonic()
        if drained >= max_samples:
            stop_reason = "max_samples"
            break
        if (now - start) >= max_seconds:
            stop_reason = "max_seconds"
            break
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            if drained > 0:
                if idle_start is None:
                    idle_start = now
                elif (now - idle_start) >= 0.05:
                    stop_reason = "completed"
                    break
            time.sleep(0.005)
            continue
        idle_start = None
        drained += 1

    print(f"🧹 Drained {drained} stale LSL samples ({label}, stop={stop_reason}).")
    return drained


def resolve_eeg_channel_indices(info, expected):
    try:
        ch = info.desc().child("channels").child("channel")
    except Exception:
        ch = None
    labels = []
    if ch is not None:
        for _ in range(info.channel_count()):
            labels.append(ch.child_value("label"))
            ch = ch.next_sibling()
    if labels:
        label_map = {label.lower(): idx for idx, label in enumerate(labels) if label}
        indices = []
        for name in expected:
            idx = label_map.get(name.lower())
            if idx is None:
                return None
            indices.append(idx)
        return indices
    return None


channel_indices = list(range(CHANNELS))
if not IMPORT_ONLY:
    print("🔍 Resolving EEG stream...")
    if CSV_OFFLINE_PATH:
        raise RuntimeError("CSV offline mode is not supported in 1_stream_and_record.py.")
    if streamer_internal and not LSL_STREAM_NAME:
        LSL_STREAM_NAME = streamer_stream_name
    if streamer_internal and not LSL_STREAM_TYPE:
        LSL_STREAM_TYPE = streamer_stream_type
    name_contains = LSL_STREAM_NAME or None
    type_equals = LSL_STREAM_TYPE or ("EEG" if not name_contains else None)
    selector = StreamSelector(
        name_contains=name_contains,
        type_equals=type_equals,
        min_channels=stream_requirements.expected_channels,
        require_unique=True,
    )
    try:
        eeg_stream = pick_stream(selector)
    except NoStreamMatchedError:
        if name_contains:
            raise
        eeg_stream = pick_stream(
            StreamSelector(
                name_contains="eeg",
                type_equals=None,
                min_channels=stream_requirements.expected_channels,
            )
        )
    except (NoStreamFoundError, MultipleStreamsMatchedError, LSLStreamSelectError):
        raise
    inlet = StreamInlet(
        eeg_stream,
        max_buflen=int(ACQ_MAX_BUFLEN_S),
        max_chunklen=int(ACQ_MAX_CHUNKLEN),
    )
    lsl_stream_signature = stream_signature(eeg_stream)
    info = inlet.info()
    found_labels = extract_channel_labels(info)
    lsl_stream_signature["labels"] = found_labels
    lsl_stream_signature["channel_labels"] = found_labels
    log_stream_signature(lsl_stream_signature)

    channel_count = int(info.channel_count())
    expected_labels = list(stream_requirements.required_labels)
    channel_indices = resolve_eeg_channel_indices(info, expected_labels)

    label_status = _evaluate_label_check(found_labels, channel_count)
    if channel_indices is None and label_status["ok"]:
        label_status["ok"] = False
        label_status["reason"] = "labels_missing_or_mismatched"
    health_state.label_check_status = label_status

    if not label_status["ok"]:
        expected_n = int(stream_requirements.expected_channels)
        report_path = _write_label_check_report(label_status)
        _write_session_state(label="label_check_failed")
        print(
            "🛑 LABEL CHECK FAILED: expected labels "
            f"{label_status['expected_labels']} ({expected_n}ch). "
            f"Found channels={channel_count}, labels={label_status['found_labels']}. "
            f"Report={report_path}"
        )
        if not label_check_acknowledged:
            print(
                "⚠️ Label mismatch not acknowledged; clean logging disabled until acknowledged."
            )
        else:
            print("⚠️ Label mismatch acknowledged by operator; proceeding with caution.")
    else:
        health_state.label_check_status = label_status

    if channel_indices is None:
        channel_indices = list(range(int(stream_requirements.expected_channels)))
    print(
        f"✅ EEG connected ({info.channel_count()} channels, using indices {channel_indices})"
    )
    selected_channel_indices = channel_indices
    _drain_inlet(inlet, label="startup")

# =========================
# ===== BUFFERS ===========
# =========================
buffer_len = int(WINDOW_SEC * SAMPLING_RATE)

# =========================
# ===== CSV OUTPUT ========
# =========================
csv_file = None
csv_writer = None
raw_file = None
raw_writer = None
predictions_file = None
predictions_writer = None

header = [
    "lsl_timestamp",
    "lsl_timestamp_mono",
    "time_s",
    "segment_id",
    "ch1",
    "ch2",
    "ch3",
    "ch4",
    "pred_action",
    "pred_finger",
    "action_confidence",
    "action_uncertainty",
    "finger_confidence",
    "finger_uncertainty",
    "velocity",
    "latency_ms",
]

def _close_segment_files() -> None:
    global csv_file, csv_writer, raw_file, raw_writer, predictions_file, predictions_writer
    if csv_file:
        csv_file.flush()
        csv_file.close()
    csv_file = None
    csv_writer = None
    if raw_file:
        raw_file.flush()
        raw_file.close()
    raw_file = None
    raw_writer = None
    if predictions_file:
        predictions_file.flush()
        predictions_file.close()
    predictions_file = None
    predictions_writer = None


def _failed_prefix(now_utc: Optional[str] = None) -> str:
    stamp = now_utc or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    sid = session_id or "UNKNOWN"
    return f"{subject_id}_{sid}_seg{segment_id:02d}_{stamp}_UNHEALTHY"


def _open_failed_files() -> None:
    prefix = _failed_prefix()
    failed_writers.open_failed_files(
        prefix,
        headers=header,
        save_raw=bool(SAVE_RAW),
        save_preds=bool(SAVE_TO_DISK),
        failed_dir=hard_stop_policy.failed_dir,
        prediction_header=[
            "prediction_time_s",
            "prediction_lsl_ts",
            "window_start_s",
            "window_end_s",
            "segment_id",
            "pred_action",
            "pred_finger",
            "action_confidence",
            "action_uncertainty",
            "finger_confidence",
            "finger_uncertainty",
            "inference_latency_ms",
            "experiment_hash",
            "model_version",
        ],
        raw_header=_raw_header(CHANNELS),
    )


def _open_segment_files() -> None:
    global csv_file, csv_writer, raw_file, raw_writer, predictions_file, predictions_writer
    if SAVE_TO_DISK:
        FEATURES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        features_exists = FEATURES_PATH.exists() and FEATURES_PATH.stat().st_size > 0
        csv_file = open(FEATURES_PATH, "a", newline="")
        csv_writer = csv.writer(csv_file)
        if not features_exists:
            csv_writer.writerow(header)
        predictions_exists = (
            PREDICTIONS_ARCHIVE_PATH.exists()
            and PREDICTIONS_ARCHIVE_PATH.stat().st_size > 0
        )
        predictions_file = open(PREDICTIONS_ARCHIVE_PATH, "a", newline="")
        predictions_writer = csv.writer(predictions_file)
        if not predictions_exists:
            predictions_writer.writerow(
                [
                    "prediction_time_s",
                    "prediction_lsl_ts",
                    "window_start_s",
                    "window_end_s",
                    "segment_id",
                    "pred_action",
                    "pred_finger",
                    "action_confidence",
                    "action_uncertainty",
                    "finger_confidence",
                    "finger_uncertainty",
                    "inference_latency_ms",
                    "experiment_hash",
                    "model_version",
                ]
            )


if not IMPORT_ONLY:
    _reset_segment_state(state)
    _open_segment_files()


backlog_queue_samples = max(1, int(float(BACKLOG_SECONDS_THRESHOLD) * SAMPLING_RATE))
health_gap_threshold_s = float(HEALTH_GAP_THRESHOLD_MULT) / float(SAMPLING_RATE)
health_gate = RollingStreamHealthGate(
    expected_fs=float(SAMPLING_RATE),
    health_window_s=float(HEALTH_WINDOW_S),
    stall_s=float(STALL_INPUT_S),
    max_queue=int(backlog_queue_samples),
    backlog_grace_s=float(BACKLOG_GRACE_S),
    recovery_s=float(RECOVERY_S),
    backwards_threshold=int(BACKWARDS_LIMIT),
    backwards_window_s=float(BACKWARDS_WINDOW_S),
    gap_threshold_s=float(health_gap_threshold_s),
    gap_count_threshold=int(HEALTH_GAP_MAX_EVENTS),
    gap_window_s=float(HEALTH_GAP_WINDOW_S),
)
last_stream_check = time.monotonic()


# =========================
# ===== ICA SETUP =========
# =========================
def is_artifact(ic_signal, fs):
    ic_signal = np.asarray(ic_signal)
    if ic_signal.size < max(64, int(fs * 0.25)):
        return False
    f, pxx = welch(ic_signal, fs=fs, nperseg=min(256, ic_signal.size))
    low = pxx[f < 4].sum()
    mid = pxx[(f >= 8) & (f <= 30)].sum()
    high = pxx[f > 35].sum()
    mid = max(mid, 1e-12)
    return (low > mid * 2.0) or (high > mid * 2.0)


def _ica_fit_worker(window_compressed: np.ndarray) -> Dict[str, Any]:
    result = guard_ica_fit(
        window_compressed,
        scaler=ica_scaler,
        ica=ica,
        min_samples=int(ICA_MIN_SAMPLES),
        min_var=float(ICA_MIN_VAR),
    )
    return {"ok": bool(result.ok), "reason": result.reason, "diagnostics": result.diagnostics}


def _ica_transform_worker(window: np.ndarray) -> Dict[str, Any]:
    reason, diag = validate_ica_input(window, min_samples=None, min_var=float(ICA_MIN_VAR))
    if reason is not None:
        return {"ok": False, "reason": reason, "diagnostics": diag}
    X_scaled = ica_scaler.transform(window)
    if not np.isfinite(X_scaled).all():
        raise ValueError("ICA scaling produced non-finite values.")
    S = ica.transform(X_scaled)
    for k in range(S.shape[1]):
        if is_artifact(S[:, k], fs=SAMPLING_RATE):
            S[:, k] *= ARTIFACT_ATTENUATION
    cleaned = ica.inverse_transform(S)
    return {"ok": True, "cleaned": cleaned}


ica = FastICA(n_components=CHANNELS, random_state=42) if ENABLE_ICA else None
ica_scaler = StandardScaler()
state.ica_fitted = False
ica_executor = ThreadPoolExecutor(max_workers=1) if ENABLE_ICA else None
state.ica_fit_future = None
state.ica_transform_future = None
ARTIFACT_ATTENUATION = 0.3

# =========================
# ===== PLOT ==============
# =========================
if ENABLE_PLOT:
    plt.ion()
    fig, ax = plt.subplots()
    lines = [ax.plot([], [])[0] for _ in range(CHANNELS)]
    latency_text = ax.text(0.01, 0.95, "", transform=ax.transAxes)
    info_text = ax.text(0.01, 0.90, "", transform=ax.transAxes)
    ax.set_ylim(-150, 150)
    ax.set_title("Cleaned EEG + Confidence Info")


def _label_check_blocked() -> bool:
    return bool(
        health_state.label_check_status
        and not health_state.label_check_status.get("ok")
        and not label_check_acknowledged
    )


def _is_true_input_stall(now_monotonic: float) -> bool:
    if last_any_sample_received_time is None:
        return True
    return (now_monotonic - last_any_sample_received_time) > float(STALL_INPUT_S)


def _write_hard_stop_report_and_exit(
    reason: str, now_monotonic: float, decision
) -> None:
    if health_state.hard_stop_triggered:
        raise SystemExit(hard_stop_policy.hard_stop_exit_code)
    report_dir = Path("logs")
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp()
    report_path = report_dir / (
        f"hard_stop_{subject_id}_{session_id or 'UNKNOWN'}_{stamp}.json"
    )
    payload = {
        "reason": reason,
        "subject_id": subject_id,
        "session_id": session_id,
        "segment_id": int(segment_id),
        "timestamp_utc": utc_now_iso_z(),
        "timestamp_monotonic": float(now_monotonic),
        "measured_fs": decision.measured_fs,
        "write_rate": float(decision.write_rate),
        "queue_size": int(decision.queue_size),
        "backwards_count": int(decision.backwards_count),
        "raw_queue_overflow_count": int(raw_queue_overflow_count),
        "processing_queue_overflow_count": int(processing_queue_overflow_count),
        "max_queue_size_observed": int(max_queue_size_observed),
        "nonfinite_count": int(raw_nonfinite_count),
        "integrity_gap_count": int(integrity_gap_count),
        "integrity_missing_estimate": int(integrity_missing_estimate),
        "last_received_lsl_ts": decision.last_received_lsl_ts,
        "last_written_lsl_ts": decision.last_written_lsl_ts,
        "lsl_stream": lsl_stream_signature,
        "label_check_status": health_state.label_check_status,
        "samples_received": int(state.samples_seen),
        "samples_written": int(state.samples_written),
        "first_received_lsl_ts": first_received_lsl_ts,
        "last_received_lsl_ts": last_received_lsl_ts,
        "first_written_lsl_ts": first_written_lsl_ts,
        "last_written_lsl_ts": last_written_lsl_ts,
        "windows_processed": int(state.windows_processed),
        "unhealthy_duration_s": float(
            now_monotonic - health_state.unhealthy_since_mono
            if health_state.unhealthy_since_mono is not None
            else 0.0
        ),
        "clean_paths": {
            "features": str(FEATURES_ARCHIVE_PATH),
            "predictions": str(PREDICTIONS_ARCHIVE_PATH),
            "raw": str(RAW_ARCHIVE_PATH),
            "events": str(EVENTS_CSV_PATH),
        },
        "failed_paths": {
            "features": str(failed_writers.features_file.name)
            if failed_writers.features_file
            else None,
            "predictions": str(failed_writers.preds_file.name)
            if failed_writers.preds_file
            else None,
            "raw": str(failed_writers.raw_file.name) if failed_writers.raw_file else None,
            "events": str(failed_writers.events_path)
            if failed_writers.events_path
            else None,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2))
    health_state.hard_stop_report_path = report_path
    health_state.hard_stop_triggered = True
    _write_session_state(label="hard_stop")
    print(
        "🛑 HARD STOP: "
        f"{reason} — wrote report: {report_path} — exiting (code {hard_stop_policy.hard_stop_exit_code})"
    )
    _close_segment_files()
    failed_writers.close_failed_files()
    raise SystemExit(hard_stop_policy.hard_stop_exit_code)


def _write_soft_stop_report(reason: str, now_monotonic: float, decision) -> None:
    global soft_stop_report_path
    if bool(getattr(state, "soft_stop_triggered", False)):
        return
    report_dir = Path("logs")
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = _report_stamp()
    report_path = report_dir / (
        f"soft_stop_{subject_id}_{session_id or 'UNKNOWN'}_{stamp}.json"
    )
    payload = {
        "reason": reason,
        "subject_id": subject_id,
        "session_id": session_id,
        "segment_id": int(segment_id),
        "timestamp_utc": utc_now_iso_z(),
        "timestamp_monotonic": float(now_monotonic),
        "measured_fs": decision.measured_fs if decision else None,
        "write_rate": float(stream_health_write_rate),
        "queue_size": int(stream_health_queue_size),
        "backwards_count": int(stream_health_backwards_count),
        "raw_queue_overflow_count": int(raw_queue_overflow_count),
        "processing_queue_overflow_count": int(processing_queue_overflow_count),
        "max_queue_size_observed": int(max_queue_size_observed),
        "nonfinite_count": int(raw_nonfinite_count),
        "integrity_gap_count": int(integrity_gap_count),
        "integrity_missing_estimate": int(integrity_missing_estimate),
        "last_received_lsl_ts": stream_health_last_received_lsl_ts,
        "last_written_lsl_ts": stream_health_last_written_lsl_ts,
        "lsl_stream": lsl_stream_signature,
        "label_check_status": health_state.label_check_status,
        "samples_received": int(state.samples_seen),
        "samples_written": int(state.samples_written),
        "first_received_lsl_ts": first_received_lsl_ts,
        "last_received_lsl_ts": last_received_lsl_ts,
        "first_written_lsl_ts": first_written_lsl_ts,
        "last_written_lsl_ts": last_written_lsl_ts,
        "windows_processed": int(state.windows_processed),
        "unhealthy_duration_s": float(
            now_monotonic - health_state.unhealthy_since_mono
            if health_state.unhealthy_since_mono is not None
            else 0.0
        ),
        "clean_paths": {
            "features": str(FEATURES_ARCHIVE_PATH),
            "predictions": str(PREDICTIONS_ARCHIVE_PATH),
            "raw": str(RAW_ARCHIVE_PATH),
            "events": str(EVENTS_CSV_PATH),
        },
        "failed_paths": {
            "features": str(failed_writers.features_file.name)
            if failed_writers.features_file
            else None,
            "predictions": str(failed_writers.preds_file.name)
            if failed_writers.preds_file
            else None,
            "raw": str(failed_writers.raw_file.name) if failed_writers.raw_file else None,
            "events": str(failed_writers.events_path)
            if failed_writers.events_path
            else None,
        },
    }
    report_path.write_text(json.dumps(_to_jsonable(payload), indent=2))
    soft_stop_report_path = report_path
    state.soft_stop_triggered = True
    _write_session_state(label="soft_stop")
    print(f"⚠️ SOFT STOP: {reason} — wrote report: {report_path}")


def _route_writers_for_health(now_mono: float, decision) -> None:
    if not health_state.has_health_decision:
        return
    if _label_check_blocked():
        health_state.unhealthy_since_mono = None
        health_state.failed_write_until_mono = None
        return

    if not decision.healthy:
        if health_state.unhealthy_since_mono is None:
            health_state.unhealthy_since_mono = now_mono
        health_state.failed_write_until_mono = max(
            health_state.failed_write_until_mono or 0.0,
            now_mono + float(hard_stop_policy.failed_write_window_s),
        )
        with failed_writers_lock:
            if not failed_writers.is_open():
                _open_failed_files()
        unhealthy_duration = float(now_mono - health_state.unhealthy_since_mono)
        if (
            unhealthy_duration >= float(UNHEALTHY_SOFT_STOP_S)
            and not bool(getattr(state, "soft_stop_triggered", False))
        ):
            stop_reason = decision.reason or "unhealthy"
            if _is_true_input_stall(now_mono):
                stop_reason = "soft_stop_input_stall"
            _write_soft_stop_report(stop_reason, now_mono, decision)
            stop_event.set()
        if (
            unhealthy_duration >= float(hard_stop_policy.hard_stop_after_unhealthy_s)
            and _is_true_input_stall(now_mono)
        ):
            hard_reason = decision.reason or "hard_stop_input_stall"
            if decision.reason == "lsl_starvation":
                hard_reason = "hard_stop_input_stall"
            _write_hard_stop_report_and_exit(hard_reason, now_mono, decision)
        return

    health_state.unhealthy_since_mono = None
    if (
        decision.event_allowed
        and health_state.failed_write_until_mono is not None
        and now_mono >= health_state.failed_write_until_mono
    ):
        health_state.failed_write_until_mono = None
    if decision.event_allowed and health_state.failed_write_until_mono is None:
        with failed_writers_lock:
            if failed_writers.is_open():
                failed_writers.close_failed_files()
        if SAVE_TO_DISK and (csv_writer is None or predictions_writer is None):
            _open_segment_files()


def _active_writer_for_mode(
    now_mono: float, clean_writer, failed_writer
) -> Optional[Any]:
    decision = last_health_decision
    if not health_state.has_health_decision or decision is None:
        return None
    if _label_check_blocked():
        return None
    if not decision.healthy:
        if (
            health_state.failed_write_until_mono is not None
            and now_mono <= health_state.failed_write_until_mono
        ):
            return failed_writer
        return None
    if decision.event_allowed:
        if (
            health_state.failed_write_until_mono is None
            or now_mono >= health_state.failed_write_until_mono
        ):
            return clean_writer
    return None


def _active_feature_writer(now_mono: float) -> Optional[Any]:
    return _active_writer_for_mode(now_mono, csv_writer, failed_writers.features_writer)


def _active_raw_writer(now_mono: float) -> Optional[Any]:
    return _active_writer_for_mode(now_mono, raw_writer, failed_writers.raw_writer)


def _active_predictions_writer(now_mono: float) -> Optional[Any]:
    return _active_writer_for_mode(
        now_mono, predictions_writer, failed_writers.preds_writer
    )


def _update_stream_health(now_monotonic: float) -> None:
    global data_stream_active, data_stream_stalled_reason, data_stream_last_write_ts
    global stream_health_measured_fs, stream_health_write_rate, stream_health_queue_size
    global stream_health_backwards_count, stream_health_last_received_lsl_ts
    global stream_health_last_written_lsl_ts, event_marking_allowed
    global segment_break_hold_until, last_health_warning_time
    global current_event
    global last_health_decision
    global last_debug_dump_time, last_debug_dump_reason

    prev_active = data_stream_active
    prev_event_allowed = event_marking_allowed
    with health_gate_lock:
        decision = health_gate.evaluate(now_monotonic)
    if decision.reason == "lsl_starvation" and not _is_true_input_stall(now_monotonic):
        decision.reason = "backpressure_processing"
    health_state.has_health_decision = True
    health_state.last_health_reason = decision.reason
    last_health_decision = decision
    stream_health_measured_fs = decision.measured_fs
    stream_health_write_rate = decision.write_rate
    stream_health_queue_size = decision.queue_size
    stream_health_backwards_count = decision.backwards_count
    stream_health_last_received_lsl_ts = decision.last_received_lsl_ts
    stream_health_last_written_lsl_ts = decision.last_written_lsl_ts

    data_stream_stalled_reason = decision.reason
    label_blocked = _label_check_blocked()
    if label_blocked:
        data_stream_stalled_reason = "label_check_unacknowledged"
    data_stream_active = decision.healthy and not label_blocked
    event_allowed = decision.event_allowed

    if segment_break_hold_until is not None and now_monotonic < segment_break_hold_until:
        event_allowed = False

    _route_writers_for_health(now_monotonic, decision)
    _maybe_write_timebase_health(now_monotonic, reason=decision.reason or "healthy")

    if not data_stream_active:
        unhealthy_duration = float(
            now_monotonic - health_state.unhealthy_since_mono
            if health_state.unhealthy_since_mono is not None
            else 0.0
        )
        should_dump = False
        if last_debug_dump_time is None or (now_monotonic - last_debug_dump_time) >= 2.0:
            should_dump = True
        if decision.reason != last_debug_dump_reason:
            should_dump = True
        if unhealthy_duration >= UNHEALTHY_WARN_S:
            should_dump = True
        if should_dump:
            _write_debug_timebase_dump(
                decision.reason or "unhealthy", now_monotonic, decision
            )
            last_debug_dump_time = float(now_monotonic)
            last_debug_dump_reason = decision.reason

        if (now_monotonic - last_health_warning_time) >= 2.0 and unhealthy_duration >= UNHEALTHY_WARN_S:
            reason = data_stream_stalled_reason or "unhealthy"
            if label_blocked:
                reason = "label_check_unacknowledged"
            print(
                f"⚠️ Stream unhealthy ({reason}); event marking disabled (unhealthy_for={unhealthy_duration:.1f}s)."
            )
            last_health_warning_time = now_monotonic
        if event_marking_active:
            stop_event_listener()
        if current_event is not None:
            current_event = None
        event_marking_allowed = False
        if prev_active or prev_event_allowed:
            _write_session_state(label="stream_unhealthy")
        return

    if event_allowed and not event_marking_allowed:
        event_marking_allowed = True
        print("✅ Stream healthy; event marking re-enabled.")
        if EVENT_MARKING_ENABLED:
            start_event_listener()
        if not prev_event_allowed:
            _write_session_state(label="stream_healthy")
    elif not event_allowed and event_marking_allowed:
        event_marking_allowed = False
        print("⚠️ Event marking disabled due to stalled writes.")
        stop_event_listener()
        if prev_event_allowed:
            _write_session_state(label="event_marking_disabled")


def _reset_lstm_state(
    reason: str, time_s: Optional[float], details: Optional[Dict[str, Any]] = None
) -> None:
    global lstm_state
    lstm_state = None
    state.lstm_reset_count += 1
    entry = {
        "time_s": float(time_s) if time_s is not None else None,
        "reason": str(reason),
        "ts_utc": utc_now_iso_z(),
    }
    if details:
        entry.update(details)
    state.lstm_reset_log.append(entry)
    print(f"⚠️ LSTM state reset ({reason}) at {entry['time_s']}")


def _extract_time_window(
    window_start_s: float,
    window_end_s: float,
) -> Tuple[Optional[np.ndarray], Optional[float], bool, Optional[float]]:
    if not state.sample_time_buffer:
        return None, None, False, None
    times = np.array([t for t, _ in state.sample_time_buffer], dtype=float)
    values = np.array([v for _, v in state.sample_time_buffer], dtype=float)
    mask = (times >= window_start_s) & (times <= window_end_s)
    if not np.any(mask):
        return None, None, False, None
    times = times[mask]
    values = values[mask]
    if times.size < 2:
        return None, None, False, None
    if not np.all(np.diff(times) > 0):
        uniq_times, uniq_idx = np.unique(times, return_index=True)
        times = uniq_times
        values = values[uniq_idx]
    if times.size < 2:
        return None, None, False, None
    if times[0] > (window_start_s + nominal_dt_s) or times[-1] < (
        window_end_s - nominal_dt_s
    ):
        return None, None, False, None
    diffs = np.diff(times)
    gap_mask = diffs > gap_threshold
    gap_flag = bool(np.any(gap_mask))
    gap_fraction = float(np.sum(diffs[gap_mask])) / float(WINDOW_SEC)
    max_gap = float(np.max(diffs[gap_mask])) if gap_flag else None
    grid = np.linspace(
        float(window_start_s),
        float(window_end_s),
        int(buffer_len),
        endpoint=False,
        dtype=float,
    )
    window = np.zeros((grid.size, values.shape[1]), dtype=float)
    for ch_idx in range(values.shape[1]):
        window[:, ch_idx] = np.interp(grid, times, values[:, ch_idx])
    return window, gap_fraction, gap_flag, max_gap


def _update_queue_metrics() -> None:
    global max_processing_queue_size_observed, max_raw_queue_size_observed
    global max_queue_size_observed
    processing_size = processing_queue.qsize()
    raw_size = raw_queue.qsize()
    max_processing_queue_size_observed = max(
        max_processing_queue_size_observed, processing_size
    )
    max_raw_queue_size_observed = max(max_raw_queue_size_observed, raw_size)
    max_queue_size_observed = max(max_queue_size_observed, processing_size, raw_size)
    with health_gate_lock:
        health_gate.set_queue_size(max(processing_size, raw_size))


def _enqueue_with_overflow(
    target_queue: queue.Queue, item: SamplePacket, *, label: str
) -> None:
    global processing_queue_overflow_count, raw_queue_overflow_count
    global last_health_warning_time
    now_mono = time.monotonic()
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        try:
            target_queue.get_nowait()
        except queue.Empty:
            pass
        if label == "raw":
            raw_queue_overflow_count += 1
        else:
            processing_queue_overflow_count += 1
        if (now_mono - last_health_warning_time) >= 2.0:
            print(f"⚠️ {label} queue overflow; dropping oldest sample.")
            last_health_warning_time = now_mono
        with health_gate_lock:
            health_gate.mark_backlog_overflow(now_mono)
        try:
            target_queue.put_nowait(item)
        except queue.Full:
            if label == "raw":
                raw_queue_overflow_count += 1
            else:
                processing_queue_overflow_count += 1
    _update_queue_metrics()


def _should_route_raw_to_failed(now_mono: float) -> bool:
    if _label_check_blocked():
        return True
    if (
        health_state.failed_write_until_mono is not None
        and now_mono <= health_state.failed_write_until_mono
    ):
        return True
    decision = last_health_decision
    if decision is None or not health_state.has_health_decision:
        return False
    return not decision.healthy


def _open_raw_csv(path: Path) -> tuple:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    file_obj = open(path, "a", newline="", buffering=int(RAW_WRITE_BUFFER_BYTES))
    writer = csv.writer(file_obj)
    if not exists:
        writer.writerow(_raw_header(CHANNELS))
    return file_obj, writer


def _write_raw_batch(
    target_kind: str,
    target_path: Optional[Path],
    rows: List[List[Any]],
    flags: List[int],
    lsl_ts_mono: List[float],
    clean_path: Optional[Path],
    clean_file,
    clean_writer,
) -> tuple[Optional[Path], Optional[Any], Optional[Any]]:
    global raw_written_count, raw_written_nonfinite_count
    global last_written_lsl_ts, first_written_lsl_ts, data_stream_last_write_ts
    global last_sample_written_time
    if not rows:
        return clean_path, clean_file, clean_writer
    writer = None
    if target_kind == "failed":
        with failed_writers_lock:
            if failed_writers.raw_writer is None:
                _open_failed_files()
            writer = failed_writers.raw_writer
    else:
        if target_path is None:
            return clean_path, clean_file, clean_writer
        if clean_path != target_path:
            if clean_file:
                try:
                    clean_file.flush()
                    clean_file.close()
                except Exception:
                    pass
            clean_path = target_path
            clean_file, clean_writer = _open_raw_csv(clean_path)
        writer = clean_writer
    if writer is None:
        return clean_path, clean_file, clean_writer
    write_mono = time.monotonic()
    writer.writerows(rows)
    data_stream_last_write_ts = utc_now_iso_z()
    last_sample_written_time = write_mono
    with stats_lock:
        raw_written_count += len(rows)
        state.samples_written = int(raw_written_count)
        raw_written_nonfinite_count += int(
            sum(1 for flag in flags if flag & RAW_FLAG_NONFINITE)
        )
    for lsl_ts_mono_value in lsl_ts_mono:
        last_written_lsl_ts = float(lsl_ts_mono_value)
        with health_gate_lock:
            health_gate.record_written(float(lsl_ts_mono_value), write_mono)
    if first_written_lsl_ts is None and lsl_ts_mono:
        first_written_lsl_ts = float(lsl_ts_mono[0])
    return clean_path, clean_file, clean_writer


def _raw_writer_worker() -> None:
    global raw_written_count, raw_written_nonfinite_count, raw_writer_thread_error
    global last_written_lsl_ts, first_written_lsl_ts, data_stream_last_write_ts
    global last_sample_written_time
    clean_path: Optional[Path] = None
    clean_file = None
    clean_writer = None
    batch_rows: List[List[Any]] = []
    batch_flags: List[int] = []
    batch_lsl_ts_mono: List[float] = []
    batch_target_kind: Optional[str] = None
    batch_target_path: Optional[Path] = None
    try:
        while not stop_event.is_set() or not raw_queue.empty():
            try:
                packet = raw_queue.get(timeout=0.1)
            except queue.Empty:
                packet = None
            _update_queue_metrics()
            if packet is None:
                if stop_event.is_set():
                    break
                continue
            now_mono = time.monotonic()
            route_failed = _should_route_raw_to_failed(now_mono)
            target_kind = "failed" if route_failed else "clean"
            target_path = None if route_failed else packet.raw_path
            if batch_target_kind is None:
                batch_target_kind = target_kind
                batch_target_path = target_path
            if (
                target_kind != batch_target_kind
                or target_path != batch_target_path
                or len(batch_rows) >= int(RAW_WRITE_BATCH_MAX)
            ):
                clean_path, clean_file, clean_writer = _write_raw_batch(
                    batch_target_kind,
                    batch_target_path,
                    batch_rows,
                    batch_flags,
                    batch_lsl_ts_mono,
                    clean_path,
                    clean_file,
                    clean_writer,
                )
                batch_rows = []
                batch_flags = []
                batch_lsl_ts_mono = []
                batch_target_kind = target_kind
                batch_target_path = target_path
            batch_rows.append(
                _build_raw_row(
                    packet.lsl_ts_raw,
                    packet.lsl_ts_mono,
                    packet.local_ts,
                    packet.sample,
                    packet.flags,
                )
            )
            batch_flags.append(int(packet.flags))
            batch_lsl_ts_mono.append(float(packet.lsl_ts_mono))
    except Exception as exc:
        raw_writer_thread_error = exc
        stop_event.set()
    finally:
        clean_path, clean_file, clean_writer = _write_raw_batch(
            batch_target_kind,
            batch_target_path,
            batch_rows,
            batch_flags,
            batch_lsl_ts_mono,
            clean_path,
            clean_file,
            clean_writer,
        )
        if clean_file:
            try:
                clean_file.flush()
                clean_file.close()
            except Exception:
                pass


def _acquisition_worker(inlet: StreamInlet) -> None:
    global non_finite_sample_warned, raw_nonfinite_count, acquisition_thread_error
    global last_lsl_pull_monotonic_time, last_any_sample_received_time
    global last_any_sample_received_lsl_ts, last_received_lsl_ts, first_received_lsl_ts
    global total_backward_timestamp_count, time_s_clamped_count, clamped_samples_total
    global max_backwards_jump_s, last_clamp_localtime, last_clamp_lsl_ts
    global integrity_gap_count, integrity_missing_estimate, integrity_gap_max_s
    global data_stream_active, data_stream_stalled_reason

    acq_segment_id = int(segment_id)
    last_mono: Optional[float] = None
    integrity_prev_ts: Optional[float] = None
    nominal_dt = 1.0 / float(SAMPLING_RATE)
    gap_tolerance = _gap_tolerance_s(nominal_dt)

    try:
        while not stop_event.is_set():
            last_lsl_pull_monotonic_time = time.monotonic()
            samples, timestamps = inlet.pull_chunk(
                timeout=float(ACQ_PULL_TIMEOUT_S),
                max_samples=int(ACQ_MAX_CHUNKLEN),
            )
            if not timestamps:
                continue
            for sample, lsl_ts in zip(samples, timestamps):
                if stop_event.is_set():
                    break
                try:
                    sample = _apply_channel_indices(sample, channel_indices, CHANNELS)
                except ValueError as exc:
                    data_stream_active = False
                    data_stream_stalled_reason = "channel_indices_out_of_range"
                    _write_session_state(label="stream_error:channel_indices")
                    print(f"⚠️ {exc}")
                    stop_event.set()
                    break
                lsl_ts_raw = float(lsl_ts)
                clamped = False
                if last_mono is not None and lsl_ts_raw <= last_mono:
                    backwards_delta = float(last_mono - lsl_ts_raw)
                    clamped = True
                    lsl_ts_mono = float(last_mono + LSL_MONO_EPS_S)
                    total_backward_timestamp_count += 1
                    time_s_clamped_count += 1
                    clamped_samples_total += 1
                    max_backwards_jump_s = max(max_backwards_jump_s, backwards_delta)
                    last_clamp_localtime = time.time()
                    last_clamp_lsl_ts = lsl_ts_mono
                else:
                    lsl_ts_mono = lsl_ts_raw
                segment_break_reason = None
                if last_mono is not None:
                    dt_mono = float(lsl_ts_mono - last_mono)
                    if dt_mono > GAP_BREAK_S:
                        acq_segment_id += 1
                        segment_break_reason = "gap"
                last_mono = lsl_ts_mono

                flags = _raw_flags_for_sample(sample)
                if flags & RAW_FLAG_NONFINITE:
                    raw_nonfinite_count += 1
                    if not non_finite_sample_warned:
                        print(
                            "⚠️ Non-finite EEG sample detected (NaN/Inf). Logging with flag."
                        )
                        non_finite_sample_warned = True

                sample_monotonic = time.monotonic()
                last_any_sample_received_time = sample_monotonic
                last_any_sample_received_lsl_ts = float(lsl_ts_raw)
                last_received_lsl_ts = float(lsl_ts_raw)
                if first_received_lsl_ts is None:
                    first_received_lsl_ts = float(lsl_ts_raw)

                if integrity_prev_ts is not None:
                    dt_integrity = float(lsl_ts_raw - integrity_prev_ts)
                    if dt_integrity > gap_tolerance:
                        integrity_gap_count += 1
                        integrity_missing_estimate += _estimate_missing_samples(
                            dt_integrity, nominal_dt, gap_tolerance
                        )
                        integrity_gap_max_s = max(integrity_gap_max_s, dt_integrity)
                integrity_prev_ts = float(lsl_ts_raw)

                with stats_lock:
                    state.samples_seen += 1

                local_ts = float(local_clock()) if local_clock else float(time.time())
                raw_path = _segment_paths(acq_segment_id)[4]
                packet = SamplePacket(
                    lsl_ts_raw=float(lsl_ts_raw),
                    lsl_ts_mono=float(lsl_ts_mono),
                    local_ts=float(local_ts),
                    sample=list(sample),
                    flags=int(flags),
                    segment_id=int(acq_segment_id),
                    raw_path=raw_path,
                    clamped=bool(clamped),
                    segment_break_reason=segment_break_reason,
                )
                _enqueue_with_overflow(processing_queue, packet, label="processing")
                if SAVE_RAW:
                    _enqueue_with_overflow(raw_queue, packet, label="raw")

                with health_gate_lock:
                    health_gate.record_received(float(lsl_ts_mono), sample_monotonic)
    except Exception as exc:
        acquisition_thread_error = exc
        stop_event.set()
    finally:
        try:
            processing_queue.put_nowait(None)
        except queue.Full:
            pass
        try:
            raw_queue.put_nowait(None)
        except queue.Full:
            pass

# =========================
# ===== MAIN LOOP =========
# =========================
if not IMPORT_ONLY:
    print("▶ Streaming — type 'end_stream' OR press q/ESC to stop")

    run_error = None
    acq_thread = None
    raw_thread = None
    try:
        acq_thread = threading.Thread(
            target=_acquisition_worker, args=(inlet,), daemon=True
        )
        acq_thread.start()
        if SAVE_RAW:
            raw_thread = threading.Thread(target=_raw_writer_worker, daemon=True)
            raw_thread.start()
        while not stop_event.is_set() or not processing_queue.empty():
            if acquisition_thread_error is not None:
                raise acquisition_thread_error
            if raw_writer_thread_error is not None:
                raise raw_writer_thread_error
            now_monotonic = time.monotonic()
            if now_monotonic - last_stream_check >= DATA_STREAM_CHECK_INTERVAL_S:
                _update_stream_health(now_monotonic)
                last_stream_check = now_monotonic

            try:
                packet = processing_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            _update_queue_metrics()
            if packet is None:
                if stop_event.is_set():
                    break
                continue

            now_mono = time.monotonic()
            if packet.segment_id != segment_id:
                candidate_time_s = (
                    float(packet.lsl_ts_mono - stream_start_lsl_ts)
                    if stream_start_lsl_ts is not None
                    else None
                )
                if last_lsl_ts_mono is not None:
                    dt_s = float(packet.lsl_ts_mono - last_lsl_ts_mono)
                    if dt_s > 0 and is_gap(dt_s, nominal_dt_s):
                        total_gap_count += 1
                        gap_durations_s.append(dt_s)
                        summary = summarize_gaps(gap_durations_s)
                        state.gap_count = summary.count
                        state.gap_max_s = summary.max_gap_s or 0.0
                        state.gap_p95_s = summary.p95_gap_s
                        state.gap_p99_s = summary.p99_gap_s
                    if dt_s > GAP_RESET_THRESHOLD_S:
                        _reset_lstm_state(
                            "gap_reset",
                            candidate_time_s,
                            {"gap_dt_s": dt_s, "lsl_ts_raw": packet.lsl_ts_raw},
                        )
                _start_segment(
                    packet.segment_break_reason or "gap",
                    segment_id_override=packet.segment_id,
                )

            # =========================
            # ===== TIMEBASE INIT ======
            # =========================
            lsl_ts_raw = float(packet.lsl_ts_raw)
            lsl_ts_mono = float(packet.lsl_ts_mono)
            candidate_time_s = (
                float(lsl_ts_mono - stream_start_lsl_ts)
                if stream_start_lsl_ts is not None
                else None
            )

            if packet.clamped:
                state.backward_timestamp_count += 1

            if last_lsl_ts_mono is not None:
                dt_s = float(lsl_ts_mono - last_lsl_ts_mono)
                if dt_s > 0 and is_gap(dt_s, nominal_dt_s):
                    total_gap_count += 1
                    gap_durations_s.append(dt_s)
                    summary = summarize_gaps(gap_durations_s)
                    state.gap_count = summary.count
                    state.gap_max_s = summary.max_gap_s or 0.0
                    state.gap_p95_s = summary.p95_gap_s
                    state.gap_p99_s = summary.p99_gap_s

            if stream_start_lsl_ts is None:
                stream_start_lsl_ts = float(lsl_ts_mono)
                state.stream_start_lsl_ts = stream_start_lsl_ts
                segment_start_lsl_ts = stream_start_lsl_ts
                local_clock_at_start = float(local_clock())
                clock_offset = float(stream_start_lsl_ts - local_clock_at_start)
                state.local_clock_at_start = local_clock_at_start
                state.clock_offset = clock_offset
                run_start_utc_iso = utc_now_iso_z()
                run_start_local_iso = local_now_iso()
                timebase_written = False
                timebase_report_initialized = False
                next_window_start_s = 0.0

            if clock_offset is None:
                clock_offset = float(lsl_ts_raw - local_clock())
                state.clock_offset = clock_offset
                if not clock_offset_estimated:
                    print("⚠️ clock_offset missing; estimating from current sample.")
                    clock_offset_estimated = True
                timebase_written = False

            if not timebase_written and stream_start_lsl_ts is not None:
                _update_session_meta(complete=False, label="timebase")
                _write_session_state(label="timebase")
                timebase_written = True
            if stream_start_lsl_ts is not None and not timebase_report_initialized:
                _write_timebase_report("init")
                timebase_report_initialized = True

            if segment_start_lsl_ts is None:
                segment_start_lsl_ts = float(lsl_ts_mono)
                _write_session_state(label="segment_start")

            time_s = float(lsl_ts_mono - stream_start_lsl_ts)
            latency_ms = (packet.local_ts - lsl_ts_raw) * 1000.0

            sample_arr = np.asarray(packet.sample, dtype=float)
            recent_accepted_timestamps.append(float(lsl_ts_mono))
            recent_accepted_clamped_flags.append(bool(packet.clamped))
            debug_sample_counter += 1
            if debug_sample_counter % DEBUG_SAMPLE_DECIMATE == 0:
                try:
                    if not (packet.flags & RAW_FLAG_NONFINITE):
                        recent_tp9_values.append(float(sample_arr[0]))
                except Exception:
                    pass

            last_lsl_ts_raw = lsl_ts_raw
            last_lsl_ts_mono = lsl_ts_mono

            if packet.flags & RAW_FLAG_NONFINITE:
                _maybe_write_timebase_report()
                continue

            state.sample_time_buffer.append((time_s, sample_arr))
            buffer_min_time = time_s - float(WINDOW_SEC) - 1.0
            while (
                state.sample_time_buffer
                and state.sample_time_buffer[0][0] < buffer_min_time
            ):
                state.sample_time_buffer.popleft()

            # ===== OPTIONAL ICA =====
            cleaned = None
            latest_sample = sample_arr
            latest_window = None
            if time_s >= WINDOW_SEC:
                latest_window, _, latest_gap_flag, _ = _extract_time_window(
                    time_s - WINDOW_SEC, time_s
                )
                if latest_window is not None and latest_gap_flag:
                    latest_window = None

            if ENABLE_ICA and ica is not None and not ica_disabled_due_to_error:
                ica_skipped_reason = None
                compressed = None
                if latest_window is None:
                    ica_skipped_reason = "window_unavailable"
                else:
                    diff_mask = np.any(np.diff(latest_window, axis=0) != 0, axis=1)
                    compressed = latest_window[1:][diff_mask]
                    if len(compressed) < CHANNELS:
                        ica_skipped_reason = "compressed_short"

                if state.ica_fit_future is not None and state.ica_fit_future.done():
                    fit_segment = state.ica_fit_segment_id
                    try:
                        result = state.ica_fit_future.result()
                        if not _should_accept_ica_result(segment_id, fit_segment):
                            if LOG_ICA_DIAGNOSTICS:
                                print(
                                    "⚠️ Discarding ICA fit from previous segment."
                                )
                        elif result.get("ok"):
                            state.ica_fitted = True
                            ica_ran = True
                        else:
                            ica_skipped_reason = result.get("reason")
                            if LOG_ICA_DIAGNOSTICS:
                                print(
                                    f"⚠️ ICA skipped: {result.get('reason')} {result.get('diagnostics')}"
                                )
                    except Exception as exc:
                        ica_failed_exception = repr(exc)
                        ica_retries += 1
                        ica_skipped_reason = "fit_exception"
                        print(f"⚠️ ICA fit failed: {exc}")
                        if ica_retries >= ICA_MAX_RETRIES_PER_SESSION:
                            ica_disabled_due_to_error = True
                            print(
                                "⚠️ ICA disabled for session due to repeated failures."
                            )
                    state.ica_fit_future = None
                    state.ica_fit_segment_id = None

                if (
                    state.ica_transform_future is not None
                    and state.ica_transform_future.done()
                ):
                    transform_segment = state.ica_transform_segment_id
                    try:
                        result = state.ica_transform_future.result()
                        if not _should_accept_ica_result(
                            segment_id, transform_segment
                        ):
                            if LOG_ICA_DIAGNOSTICS:
                                print(
                                    "⚠️ Discarding ICA transform from previous segment."
                                )
                        elif result.get("ok"):
                            cleaned = result.get("cleaned")
                            if cleaned is not None and len(cleaned):
                                latest_sample = cleaned[-1]
                                ica_ran = True
                        else:
                            ica_skipped_reason = result.get("reason")
                            if LOG_ICA_DIAGNOSTICS:
                                print(
                                    f"⚠️ ICA skipped: {result.get('reason')} {result.get('diagnostics')}"
                                )
                    except Exception as exc:
                        ica_failed_exception = repr(exc)
                        ica_retries += 1
                        ica_skipped_reason = "transform_exception"
                        print(f"⚠️ ICA transform failed: {exc}")
                        if ica_retries >= ICA_MAX_RETRIES_PER_SESSION:
                            ica_disabled_due_to_error = True
                            print(
                                "⚠️ ICA disabled for session due to repeated failures."
                            )
                    state.ica_transform_future = None
                    state.ica_transform_segment_id = None

                if (
                    not state.ica_fitted
                    and latest_window is not None
                    and ica_skipped_reason is None
                    and state.ica_fit_future is None
                ):
                    warmup_ready = state.samples_seen >= int(
                        ICA_WARMUP_S * SAMPLING_RATE
                    )
                    if not warmup_ready:
                        ica_skipped_reason = "warmup"
                    elif compressed is not None and ica_executor is not None:
                        state.ica_fit_segment_id = int(segment_id)
                        state.ica_fit_future = ica_executor.submit(
                            _ica_fit_worker, compressed.copy()
                        )

                if (
                    state.ica_fitted
                    and latest_window is not None
                    and ica_skipped_reason is None
                    and state.ica_transform_future is None
                    and ica_executor is not None
                ):
                    state.ica_transform_segment_id = int(segment_id)
                    state.ica_transform_future = ica_executor.submit(
                        _ica_transform_worker, latest_window.copy()
                    )
            if cleaned is None:
                cleaned = latest_window if latest_window is not None else None

            if latest_sample is None:
                continue

            # ===== DEFAULTS ===== (runs per sample)
            pred_action = int(last_pred_action)
            pred_finger = int(last_pred_finger)
            action_confidence = float(last_action_confidence)
            action_uncertainty = float(last_action_uncertainty)
            finger_confidence = float(last_finger_confidence)
            finger_uncertainty = float(last_finger_uncertainty)
            velocity = (
                action_confidence * (1.0 - action_uncertainty)
                if pred_action != ACTION_REST
                else 0.0
            )

            # ===== INFERENCE =====
            inference_enabled = DEMO_MODE and model is not None and scaler is not None
            if not inference_enabled and lstm_state is not None:
                _reset_lstm_state("inference_disabled", time_s)

            if inference_enabled and next_window_start_s is not None:
                max_windows_per_cycle = 5
                windows_processed = 0
                latest_time_s = float(time_s)

                while (next_window_start_s + WINDOW_SEC) <= latest_time_s:
                    if windows_processed >= max_windows_per_cycle:
                        backlog = int(
                            ((latest_time_s - WINDOW_SEC) - next_window_start_s)
                            // WINDOW_HOP_SEC
                        )
                        if backlog > 0:
                            state.window_drop_count += backlog
                            next_window_start_s += backlog * WINDOW_HOP_SEC
                        break

                    window_start_s = float(next_window_start_s)
                    window_end_s = float(window_start_s + WINDOW_SEC)
                    window_center_s = float(window_start_s + (WINDOW_SEC / 2.0))

                    if not data_stream_active:
                        state.window_health_drop_count += 1
                        _reset_lstm_state("stream_unhealthy", window_center_s)
                        next_window_start_s += WINDOW_HOP_SEC
                        continue

                    (
                        window_data,
                        gap_fraction,
                        gap_flag,
                        max_gap,
                    ) = _extract_time_window(window_start_s, window_end_s)
                    if window_data is None:
                        state.window_incomplete_drop_count += 1
                        next_window_start_s += WINDOW_HOP_SEC
                        continue
                    if gap_flag:
                        state.window_gap_drop_count += 1
                        if max_gap is not None and max_gap > GAP_RESET_THRESHOLD_S:
                            _reset_lstm_state("gap_reset", window_center_s)
                        next_window_start_s += WINDOW_HOP_SEC
                        continue

                    window_input = standardize_window_TxC(
                        window_data.astype(np.float32), scaler
                    )
                    x_BTC = (
                        torch.tensor(window_input, dtype=torch.float32)
                        .unsqueeze(0)
                        .to(DEVICE)
                    )
                    if hasattr(model, "forward_with_state"):
                        (
                            finger_logits,
                            action_logits,
                            lstm_state,
                        ) = model.forward_with_state(x_BTC, lstm_state)
                    else:
                        finger_logits, action_logits = model(x_BTC)
                        lstm_state = None

                    action_probs = torch.softmax(action_logits, dim=1).squeeze(0)
                    finger_probs = torch.softmax(finger_logits, dim=1).squeeze(0)
                    action_probs_np = action_probs.detach().cpu().numpy()
                    finger_probs_np = finger_probs.detach().cpu().numpy()

                    pred_action = int(np.argmax(action_probs_np))
                    action_confidence = float(action_probs_np[pred_action])
                    action_uncertainty = 0.0

                    pred_finger = int(np.argmax(finger_probs_np))
                    finger_confidence = float(finger_probs_np[pred_finger])
                    finger_uncertainty = 0.0

                    adaptive_thresh = min(
                        0.99,
                        max(
                            BASE_CONF_THRESH,
                            BASE_CONF_THRESH
                            + UNCERTAINTY_WEIGHT * action_uncertainty,
                        ),
                    )

                    state.action_pred_buffer.append(pred_action)
                    if pred_action != ACTION_REST:
                        velocity = action_confidence * (1.0 - action_uncertainty)

                    if ENABLE_ACTUATION and pred_action != ACTION_REST:
                        stable = (
                        len(state.action_pred_buffer) == STABILITY_FRAMES
                        and len(set(state.action_pred_buffer)) == 1
                        )
                        if (
                            stable
                            and calibrator.allow_actuation(
                                action_confidence, action_uncertainty
                            )
                            and action_confidence >= adaptive_thresh
                        ):
                            pass

                    prediction_time_s = window_center_s
                    prediction_lsl_ts = (
                        float(stream_start_lsl_ts + prediction_time_s)
                        if stream_start_lsl_ts is not None
                        else np.nan
                    )
                    if np.isfinite(prediction_lsl_ts):
                        inference_latency_ms = (
                            local_clock() - prediction_lsl_ts
                        ) * 1000.0
                    else:
                        inference_latency_ms = np.nan

                    predictions_target = _active_predictions_writer(now_mono)
                    if predictions_target is not None:
                        predictions_target.writerow(
                            [
                                prediction_time_s,
                                prediction_lsl_ts,
                                window_start_s,
                                window_end_s,
                                int(segment_id),
                                pred_action,
                                pred_finger,
                                action_confidence,
                                action_uncertainty,
                                finger_confidence,
                                finger_uncertainty,
                                inference_latency_ms,
                                experiment_hash,
                                MODEL_PATH,
                            ]
                        )

                    if (
                        live_viz_enabled
                        and lstm_state is not None
                        and live_viz_fps > 0
                    ):
                        now_live = time.monotonic()
                        min_interval = 1.0 / float(live_viz_fps)
                        if (now_live - last_live_viz_emit) >= min_interval:
                            try:
                                h_state = lstm_state[0] if isinstance(lstm_state, tuple) else lstm_state
                                hidden_vec = h_state[-1, 0].detach().cpu().numpy()
                                hidden_mag = float(np.linalg.norm(hidden_vec))
                                print(
                                    f"VIZ hidden_mag={hidden_mag:.6f} t={prediction_time_s:.3f}"
                                )
                                last_live_viz_emit = now_live
                            except Exception:
                                pass

                    last_pred_action = pred_action
                    last_pred_finger = pred_finger
                    last_action_confidence = action_confidence
                    last_action_uncertainty = action_uncertainty
                    last_finger_confidence = finger_confidence
                    last_finger_uncertainty = finger_uncertainty

                    state.windows_processed += 1
                    windows_processed += 1
                    next_window_start_s += WINDOW_HOP_SEC

            # ===== OPTIONAL ONLINE CALIBRATION FEEDBACK =====
            if DEMO_MODE and TRAINING_MODE:
                try:
                    true_action = int(
                        input("Action label (0=REST,1=OPEN,2=CLOSE): ")
                    )
                    true_finger = int(input("Finger label (0=NONE,1-5): "))

                    correct = (pred_action == true_action) and (
                        true_action == ACTION_REST or pred_finger == true_finger
                    )

                    calibrator.update(action_confidence, correct)
                    record_prediction(
                        subject_id=subject_id,
                        experiment_hash=experiment_hash,
                        confidence=action_confidence,
                        uncertainty=action_uncertainty,
                        correct=correct,
                        threshold=calibrator.threshold,
                    )

                    CALIBRATION_STATE_PATH.write_text(
                        json.dumps(
                            {
                                "threshold": calibrator.threshold,
                                "history": [],
                                "config": {
                                    "init_threshold": calibrator.threshold,
                                    "min_threshold": calibrator.min_threshold,
                                    "max_threshold": calibrator.max_threshold,
                                    "ema_alpha": calibrator.alpha,
                                },
                            },
                            indent=2,
                        )
                    )
                except Exception:
                    pass

            # ===== SAVE FEATURES =====
            feature_target = _active_feature_writer(now_mono)
            if feature_target is not None:
                if time_s is None or not np.isfinite(time_s):
                    continue

                row = [
                    lsl_ts_raw,
                    lsl_ts_mono,
                    time_s,
                    int(segment_id),
                    *latest_sample,
                    pred_action,
                    pred_finger,
                    action_confidence,
                    action_uncertainty,
                    finger_confidence,
                    finger_uncertainty,
                    velocity,
                    latency_ms,
                ]
                if len(row) != len(header):
                    raise RuntimeError(
                        f"Feature row length {len(row)} does not match header length {len(header)}"
                )
                feature_target.writerow(row)

            if feature_target is not None:
                if time_s is not None and np.isfinite(time_s):
                    last_written_time_s = float(time_s)
                    last_time_s = float(last_written_time_s)
                    last_sample_time_s = float(time_s)
                    last_sample_lsl_ts = float(lsl_ts_mono)
                    state.recent_sample_times.append(last_sample_time_s)
                    if first_time_s is None:
                        first_time_s = last_sample_time_s
                    last_time_s_seen = last_sample_time_s
                    if first_lsl_ts is None:
                        first_lsl_ts = float(lsl_ts_raw)
                    last_lsl_ts = float(lsl_ts_raw)

            # ===== PLOT =====
            if ENABLE_PLOT:
                plot_data = (
                    cleaned
                    if cleaned is not None
                    else np.asarray([latest_sample], dtype=float)
                )
                for i in range(CHANNELS):
                    lines[i].set_data(range(len(plot_data)), plot_data[:, i])
                ax.set_xlim(0, len(plot_data))
                ax.relim()
                ax.autoscale_view()
                ax.set_ylim(-100, 100)
                latency_text.set_text(
                    f"Latency: {latency_ms:.1f} ms" if DEMO_MODE else ""
                )
                info_text.set_text(
                    f"Act: {ACTION_NAMES.get(pred_action, '?')}  Conf: {action_confidence:.2f}  Unc: {action_uncertainty:.3f}  Vel: {velocity:.3f}"
                    if DEMO_MODE
                    else ""
                )
                plt.pause(0.001)
            _maybe_write_timebase_report()

    except KeyboardInterrupt:
        pass
    except Exception as exc:
        run_error = exc
        print(
            f"❌ Stream crashed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    finally:
        # =========================
        # ===== CLEANUP ===========
        # =========================
        print("\n🧹 Cleaning up")

        stop_event.set()
        if acq_thread is not None:
            acq_thread.join(timeout=2.0)
        if raw_thread is not None:
            raw_thread.join(timeout=5.0)
            if raw_thread.is_alive():
                print("⚠️ Raw writer thread did not stop cleanly.")

        _close_segment_files()
        with failed_writers_lock:
            failed_writers.close_failed_files()
        _stop_streamer_process(streamer_proc)

        if ENABLE_PLOT:
            plt.close("all")

        if EVENT_MARKING_ENABLED:
            if listener:
                listener.stop()
            with events_lock:
                output_path = _resolve_events_output_path(
                    time.monotonic(), last_health_decision
                )
                if output_path:
                    save_events_csv(output_path, events)
                    print(f"📝 Events saved to {output_path}")

        if ica_executor is not None:
            ica_executor.shutdown(wait=False)

        # Write final timebase report
        _maybe_write_timebase_report(force=True, label="final")

        # Write raw integrity report
        _write_raw_integrity_report()

        # Final run timestamps
        run_end_utc_iso = utc_now_iso_z()
        run_end_local_iso = local_now_iso()

        # Update session-continuous elapsed time at end of run
        last_time_s_updated = float(
            last_written_time_s if last_written_time_s >= 0 else last_time_s
        )
        segment_elapsed_s = last_time_s_updated if last_time_s_updated >= 0 else 0.0
        total_elapsed_s_updated = float(total_elapsed_s + segment_elapsed_s)

        # Commit updated times so next resume uses the correct anchor
        total_elapsed_s = total_elapsed_s_updated
        last_time_s = last_time_s_updated

        _update_session_meta(complete=True, label="final")
        _write_session_state(
            label="final",
            total_elapsed_override=total_elapsed_s_updated,
            last_time_override=last_time_s_updated,
            block_id_override=int(BLOCK_ID) + 1,
        )

        if run_error is None:
            print("✅ Stream terminated cleanly")
        else:
            print("❌ Stream terminated after error", file=sys.stderr)
            raise SystemExit(1)
