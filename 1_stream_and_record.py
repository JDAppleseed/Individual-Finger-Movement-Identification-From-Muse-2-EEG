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
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import csv
import json
import shutil
from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

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
from utils.session_timebase import compute_event_lsl_ts
from utils.lsl_stream_select import (
    LSLStreamSelectError,
    MultipleStreamsMatchedError,
    NoStreamFoundError,
    NoStreamMatchedError,
    StreamSelector,
    log_stream_signature,
    pick_stream,
    stream_signature,
)
from utils.stream_timebase import (
    clamp_lsl_timestamp,
    gap_threshold_s,
    is_gap,
    should_segment_break_backwards,
    summarize_gaps,
)
from utils.timebase_selfcheck import evaluate_timebase_alignment
from utils.ica_guard import guard_ica_fit, validate_ica_input
from utils.stream_health import RollingStreamHealthGate

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

MODEL_PATH = "finger_action_model.pt"
SCALER_PATH = "scaler.save"

# =========================
# ===== TIMEBASE JITTER =====
# =========================
SOFT_BACKWARDS_EPS_S = 0.010
HARD_BACKWARDS_S = 0.200
SOFT_BACKWARDS_LIMIT = 6
SOFT_BACKWARDS_WINDOW_S = 1.0

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
HEALTH_WINDOW_S = 2.0
MIN_WRITE_FRACTION = 0.90
MAX_QUEUE = 512
RECOVERY_S = 2.0
BACKWARDS_WINDOW_S = 1.0
BACKWARDS_LIMIT = 3
EVENT_MAX_LAG_S = 2.0
EVENT_MAX_LEAD_S = 0.5

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
    nearest_sample_delta_samples: List[Dict[str, Any]] = field(default_factory=list)
    ica_fitted: bool = False
    ica_fit_future: Any = None
    ica_transform_future: Any = None
    ica_fit_segment_id: Optional[int] = None
    ica_transform_segment_id: Optional[int] = None

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


def _load_config(path: str):
    if not path:
        return {}
    try:
        payload = json.loads(Path(path).read_text())
    except Exception:
        return {}
    return payload.get("settings", payload)


def _apply_config(settings: dict):
    for key, val in settings.items():
        if key in globals():
            globals()[key] = val


def _apply_config_to_args(args_obj, settings: dict, defaults: dict):
    for key, default in defaults.items():
        if key in settings and getattr(args_obj, key) == default:
            setattr(args_obj, key, settings[key])


args, _ = parser.parse_known_args()
defaults = {a.dest: a.default for a in parser._actions if hasattr(a, "dest")}
config_settings = _load_config(args.config) if args.config else {}
_apply_config(config_settings)
_apply_config_to_args(args, config_settings, defaults)

if args.subject_id:
    SUBJECT_ID_OVERRIDE = args.subject_id

INIT_ONLY = bool(args.init_only)

subject_id = (
    SUBJECT_ID_OVERRIDE or ("unknown" if IMPORT_ONLY else get_subject_id(GENDER, AGE))
)
session_id = None
segment_id = 0
experiment_hash = None

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
    meta_candidate = (
        Path("data/processed") / f"{subject_id}_{state_session_id}_session_meta.json"
    )
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
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if not true_resume and SESSION_ID_OVERRIDE:
    session_id = str(SESSION_ID_OVERRIDE)
if experiment_hash is None:
    experiment_hash = generate_experiment_hash(subject_id, experiment_config)
state.session_id = session_id
state.segment_id = segment_id
state.experiment_hash = experiment_hash

FEATURES_ARCHIVE_DIR = Path("data/processed")
RAW_ARCHIVE_DIR = Path("data/raw")


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
):
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
    payload = _build_session_state_payload(
        state,
        total_elapsed_override=total_elapsed_override,
        last_time_override=last_time_override,
        block_id_override=block_id_override,
        segment_id_override=segment_id_override,
    )
    SESSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
    msg = "Updating" if SESSION_STATE_PATH.exists() else "Writing"
    print(f"ℹ️ {msg} session state ({label}): {SESSION_STATE_PATH}")
    _write_json_atomic(SESSION_STATE_PATH, payload)


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
        Path("data/raw").mkdir(parents=True, exist_ok=True)

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
                stop_event.set()
        except EOFError:
            break


if not IMPORT_ONLY:
    threading.Thread(target=listen_for_exit, daemon=True).start()


# =========================
# ===== TIMEBASE HELPERS ===
# =========================
def _lsl_now():
    """LSL-domain timestamp for 'now' using local_clock and current clock_offset."""
    if clock_offset is None:
        return None
    return compute_event_lsl_ts(local_clock(), clock_offset)


def _time_s_from_lsl(lsl_ts: float):
    """Stream-relative time_s from an LSL timestamp."""
    if stream_start_lsl_ts is None:
        return None
    return float(lsl_ts - stream_start_lsl_ts)


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
non_finite_sample_warned = False
sample_queue: Deque[tuple[float, List[float]]] = deque()
queue_drop_count = 0
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
    global ica_scaler, ica
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
    last_report_time = None
    last_report_samples_written = 0
    timebase_report_initialized = False
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


def _flush_events_for_segment() -> None:
    global current_event, last_event_index
    if current_event is not None:
        current_event = None
    with events_lock:
        if events:
            save_events_csv(EVENTS_CSV_PATH, events)
            last_event_index = len(events) - 1


def _start_segment(reason: str) -> None:
    global segment_id, events, last_event_index
    global last_written_time_s, last_time_s, total_elapsed_s
    global segment_break_hold_until

    _flush_events_for_segment()

    if last_written_time_s >= 0:
        total_elapsed_s += float(last_written_time_s)
    last_written_time_s = -1.0
    last_time_s = -1.0

    _close_segment_files()
    segment_id += 1
    state.segment_id = segment_id
    _apply_segment_paths(segment_id)
    _reset_segment_state(state)
    _open_segment_files()
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


if DEMO_MODE:
    model = CNNLSTMFingerActionNet(
        n_channels=CHANNELS, n_fingers=N_FINGERS, n_actions=N_ACTIONS
    ).to(DEVICE)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.train()  # keep dropout active

    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)


# =========================
# ===== LSL SETUP =========
# =========================
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


def probe_channel_variances(
    inlet: StreamInlet,
    channel_count: int,
    *,
    probe_duration_s: float,
    min_samples: int = 10,
) -> Optional[np.ndarray]:
    samples: List[np.ndarray] = []
    start = time.monotonic()
    while time.monotonic() - start < probe_duration_s:
        sample, _ = inlet.pull_sample(timeout=0.0)
        if sample is None:
            continue
        if len(sample) < channel_count:
            continue
        sample_arr = np.asarray(sample[:channel_count], dtype=np.float64)
        if not np.all(np.isfinite(sample_arr)):
            continue
        samples.append(sample_arr)
    if len(samples) < min_samples:
        return None
    stacked = np.vstack(samples)
    return np.var(stacked, axis=0)


channel_indices = list(range(CHANNELS))
if not IMPORT_ONLY:
    print("🔍 Resolving EEG stream...")
    if CSV_OFFLINE_PATH:
        raise RuntimeError("CSV offline mode is not supported in 1_stream_and_record.py.")
    if LSL_STREAM_NAME:
        selector = StreamSelector(
            name_contains=LSL_STREAM_NAME, type_equals=None, min_channels=CHANNELS
        )
        eeg_stream = pick_stream(selector)
    elif LSL_STREAM_TYPE:
        selector = StreamSelector(
            name_contains=None, type_equals=LSL_STREAM_TYPE, min_channels=CHANNELS
        )
        eeg_stream = pick_stream(selector)
    else:
        selector = StreamSelector(
            name_contains=None, type_equals="EEG", min_channels=CHANNELS
        )
        try:
            eeg_stream = pick_stream(selector)
        except NoStreamMatchedError:
            eeg_stream = pick_stream(
                StreamSelector(
                    name_contains="eeg", type_equals=None, min_channels=CHANNELS
                )
            )
        except (NoStreamFoundError, MultipleStreamsMatchedError, LSLStreamSelectError):
            raise
    inlet = StreamInlet(eeg_stream, max_buflen=5)
    lsl_stream_signature = stream_signature(eeg_stream)
    log_stream_signature(lsl_stream_signature)
    info = inlet.info()
    expected_labels = ["TP9", "AF7", "AF8", "TP10"]
    channel_indices = resolve_eeg_channel_indices(info, expected_labels)
    if channel_indices is None:
        if info.channel_count() < CHANNELS:
            raise RuntimeError(
                f"LSL EEG stream has {info.channel_count()} channels; expected at least {CHANNELS}."
            )
        probe_duration = min(float(ICA_WARMUP_S), 2.0)
        probe_variances = probe_channel_variances(
            inlet, info.channel_count(), probe_duration_s=probe_duration
        )
        if probe_variances is None:
            channel_indices = list(range(CHANNELS))
            selected_channel_variances = None
            print(
                "⚠️ EEG channel labels not found; probe window insufficient. Using first four channels by index."
            )
        else:
            valid_mask = np.isfinite(probe_variances)
            valid_indices = np.where(valid_mask)[0].tolist()
            filtered = [
                (idx, float(probe_variances[idx]))
                for idx in valid_indices
                if probe_variances[idx] >= ICA_MIN_VAR
            ]
            if len(filtered) < CHANNELS:
                filtered = sorted(
                    [(idx, float(probe_variances[idx])) for idx in valid_indices],
                    key=lambda x: x[1],
                    reverse=True,
                )
            channel_indices = [idx for idx, _ in filtered[:CHANNELS]]
            selected_channel_variances = [var for _, var in filtered[:CHANNELS]]
            print(
                "⚠️ EEG channel labels not found; selected channels by variance probe window."
            )
            print(
                f"🔎 Selected indices: {channel_indices} with variances {selected_channel_variances}"
            )
    print(
        f"✅ EEG connected ({info.channel_count()} channels, using indices {channel_indices})"
    )
    selected_channel_indices = channel_indices
    drained_samples = _drain_inlet(inlet, drain_s=0.75)
    if drained_samples:
        print(f"🧹 Drained {drained_samples} stale LSL samples before start.")

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
    if SAVE_RAW:
        RAW_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        raw_exists = RAW_ARCHIVE_PATH.exists() and RAW_ARCHIVE_PATH.stat().st_size > 0
        raw_file = open(RAW_ARCHIVE_PATH, "a", newline="")
        raw_writer = csv.writer(raw_file)
        if not raw_exists:
            raw_writer.writerow(
                [
                    "lsl_timestamp_raw",
                    "lsl_timestamp_mono",
                    "ch1",
                    "ch2",
                    "ch3",
                    "ch4",
                ]
            )


if not IMPORT_ONLY:
    _reset_segment_state(state)
    _open_segment_files()


health_gate = RollingStreamHealthGate(
    expected_fs=float(SAMPLING_RATE),
    health_window_s=float(HEALTH_WINDOW_S),
    stall_s=float(STALL_S),
    min_write_fraction=float(MIN_WRITE_FRACTION),
    max_queue=int(MAX_QUEUE),
    recovery_s=float(RECOVERY_S),
    backwards_threshold=int(BACKWARDS_LIMIT),
    backwards_window_s=float(BACKWARDS_WINDOW_S),
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


def _update_stream_health(now_monotonic: float) -> None:
    global data_stream_active, data_stream_stalled_reason, data_stream_last_write_ts
    global stream_health_measured_fs, stream_health_write_rate, stream_health_queue_size
    global stream_health_backwards_count, stream_health_last_received_lsl_ts
    global stream_health_last_written_lsl_ts, event_marking_allowed
    global segment_break_hold_until, last_health_warning_time
    global current_event

    prev_active = data_stream_active
    prev_event_allowed = event_marking_allowed
    decision = health_gate.evaluate(now_monotonic)
    stream_health_measured_fs = decision.measured_fs
    stream_health_write_rate = decision.write_rate
    stream_health_queue_size = decision.queue_size
    stream_health_backwards_count = decision.backwards_count
    stream_health_last_received_lsl_ts = decision.last_received_lsl_ts
    stream_health_last_written_lsl_ts = decision.last_written_lsl_ts

    data_stream_stalled_reason = decision.reason
    data_stream_active = decision.healthy
    event_allowed = decision.event_allowed

    if segment_break_hold_until is not None and now_monotonic < segment_break_hold_until:
        event_allowed = False

    if not data_stream_active:
        if (now_monotonic - last_health_warning_time) >= 2.0:
            reason = data_stream_stalled_reason or "unhealthy"
            print(f"⚠️ Stream unhealthy ({reason}); event marking disabled.")
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

# =========================
# ===== MAIN LOOP =========
# =========================
if not IMPORT_ONLY:
    print("▶ Streaming — type 'end_stream' OR press q/ESC to stop")

    try:
        while not stop_event.is_set():
            now_monotonic = time.monotonic()
            if now_monotonic - last_stream_check >= DATA_STREAM_CHECK_INTERVAL_S:
                _update_stream_health(now_monotonic)
                last_stream_check = now_monotonic

            # Drain incoming samples into a bounded queue.
            while True:
                sample, lsl_ts = inlet.pull_sample(timeout=0.0)
                if sample is None:
                    break

                try:
                    sample = _apply_channel_indices(sample, channel_indices, CHANNELS)
                except ValueError as exc:
                    data_stream_active = False
                    data_stream_stalled_reason = "channel_indices_out_of_range"
                    _write_session_state(label="stream_error:channel_indices")
                    print(f"⚠️ {exc}")
                    stop_event.set()
                    sample_queue.clear()
                    break

                if not np.all(np.isfinite(sample)):
                    if not non_finite_sample_warned:
                        print(
                            "⚠️ Non-finite EEG sample detected (NaN/Inf). Skipping sample to keep ICA stable."
                        )
                        non_finite_sample_warned = True
                    continue

                state.samples_seen += 1
                last_received_lsl_ts = float(lsl_ts)
                sample_monotonic = time.monotonic()
                health_gate.record_received(float(lsl_ts), sample_monotonic)

                if len(sample_queue) >= MAX_QUEUE:
                    queue_drop_count += 1
                    health_gate.set_queue_size(len(sample_queue) + 1)
                    if (sample_monotonic - last_health_warning_time) >= 2.0:
                        print("⚠️ Sample queue overflow; dropping samples.")
                        last_health_warning_time = sample_monotonic
                    continue

                sample_queue.append((float(lsl_ts), sample))
                health_gate.set_queue_size(len(sample_queue))

            if not sample_queue:
                continue

            while sample_queue and not stop_event.is_set():
                lsl_ts, sample = sample_queue.popleft()
                health_gate.set_queue_size(len(sample_queue))

                # =========================
                # ===== TIMEBASE INIT ======
                # =========================
                lsl_ts_raw = float(lsl_ts)
                clamp_result = clamp_lsl_timestamp(
                    last_lsl_ts_mono,
                    lsl_ts_raw,
                    epsilon_s=SOFT_BACKWARDS_EPS_S,
                    hard_backwards_s=HARD_BACKWARDS_S,
                )
                lsl_ts_mono = clamp_result.mono_ts
                candidate_time_s = (
                    float(lsl_ts_mono - stream_start_lsl_ts)
                    if stream_start_lsl_ts is not None
                    else 0.0
                )

                segment_break_reason = None
                if clamp_result.clamped:
                    state.backward_timestamp_count += 1
                    total_backward_timestamp_count += 1
                    time_s_clamped_count += 1
                    max_backwards_jump_s = max(
                        max_backwards_jump_s, clamp_result.backwards_delta_s
                    )
                    clamp_details = {
                        "backwards_delta_s": clamp_result.backwards_delta_s,
                        "lsl_ts_raw": lsl_ts_raw,
                        "lsl_ts_prev": last_lsl_ts_mono,
                        "lsl_ts_mono": lsl_ts_mono,
                    }
                    if clamp_result.is_hard_backwards:
                        segment_break_reason = "backwards_hard"
                        _reset_lstm_state(
                            "backwards_timestamp",
                            candidate_time_s,
                            {**clamp_details, "backwards_kind": "hard"},
                        )
                    elif clamp_result.is_soft_backwards:
                        if should_segment_break_backwards(
                            backwards_events_monotonic,
                            now_monotonic,
                            soft_limit=SOFT_BACKWARDS_LIMIT,
                            window_s=SOFT_BACKWARDS_WINDOW_S,
                            hard_backwards=False,
                        ):
                            segment_break_reason = "backwards_burst"
                            _reset_lstm_state(
                                "backwards_timestamp",
                                candidate_time_s,
                                {**clamp_details, "backwards_kind": "burst"},
                            )
                elif last_lsl_ts_mono is not None:
                    dt_s = float(lsl_ts_mono - last_lsl_ts_mono)
                    if is_gap(dt_s, nominal_dt_s):
                        total_gap_count += 1
                        gap_durations_s.append(dt_s)
                        summary = summarize_gaps(gap_durations_s)
                        state.gap_count = summary.count
                        state.gap_max_s = summary.max_gap_s or 0.0
                        state.gap_p95_s = summary.p95_gap_s
                        state.gap_p99_s = summary.p99_gap_s
                    if dt_s > GAP_BREAK_S:
                        segment_break_reason = "gap"
                        if dt_s > GAP_RESET_THRESHOLD_S:
                            _reset_lstm_state(
                                "gap_reset",
                                candidate_time_s,
                                {"gap_dt_s": dt_s, "lsl_ts_raw": lsl_ts_raw},
                            )

                if segment_break_reason is not None:
                    _start_segment(segment_break_reason)
                    drained = _drain_inlet(inlet, drain_s=0.75)
                    if drained:
                        print(
                            f"🧹 Drained {drained} stale LSL samples after segment break."
                        )
                    sample_queue.clear()
                    continue

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
                latency_ms = (local_clock() - lsl_ts_raw) * 1000.0

                sample_arr = np.asarray(sample, dtype=float)
                state.sample_time_buffer.append((time_s, sample_arr))
                buffer_min_time = time_s - float(WINDOW_SEC) - 1.0
                while (
                    state.sample_time_buffer
                    and state.sample_time_buffer[0][0] < buffer_min_time
                ):
                    state.sample_time_buffer.popleft()

                if raw_writer:
                    raw_writer.writerow([lsl_ts_raw, lsl_ts_mono, *sample])

                last_lsl_ts_raw = lsl_ts_raw
                last_lsl_ts_mono = lsl_ts_mono

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
                        if clock_offset is not None and np.isfinite(prediction_lsl_ts):
                            inference_latency_ms = (
                                local_clock() - (prediction_lsl_ts - clock_offset)
                            ) * 1000.0
                        else:
                            inference_latency_ms = np.nan

                        if predictions_writer:
                            predictions_writer.writerow(
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

                        last_pred_action = pred_action
                        last_pred_finger = pred_finger
                        last_action_confidence = action_confidence
                        last_action_uncertainty = action_uncertainty
                        last_finger_confidence = finger_confidence
                        last_finger_uncertainty = finger_uncertainty

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
                if SAVE_TO_DISK and csv_writer:
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
                    csv_writer.writerow(row)
                    data_stream_last_write_ts = utc_now_iso_z()
                    health_gate.record_written(float(lsl_ts_raw), time.monotonic())
                    last_written_lsl_ts = float(lsl_ts_raw)
                    last_written_time_s = float(time_s)
                    last_time_s = float(
                        last_written_time_s
                    )  # keep session meta/state current
                    last_sample_time_s = float(time_s)
                    last_sample_lsl_ts = float(lsl_ts_raw)
                    state.recent_sample_times.append(last_sample_time_s)
                    if first_time_s is None:
                        first_time_s = last_sample_time_s
                    last_time_s_seen = last_sample_time_s
                    if first_lsl_ts is None:
                        first_lsl_ts = float(lsl_ts_raw)
                    last_lsl_ts = float(lsl_ts_raw)
                    state.samples_written += 1
                    _maybe_write_timebase_report()
                else:
                    data_stream_last_write_ts = utc_now_iso_z()
                    health_gate.record_written(float(lsl_ts_raw), time.monotonic())
                    last_written_lsl_ts = float(lsl_ts_raw)

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

    except KeyboardInterrupt:
        pass

    # =========================
    # ===== CLEANUP ===========
    # =========================
    print("\n🧹 Cleaning up")

    _close_segment_files()

    if ENABLE_PLOT:
        plt.close("all")

    if EVENT_MARKING_ENABLED:
        if listener:
            listener.stop()
        with events_lock:
            save_events_csv(EVENTS_CSV_PATH, events)
        print(f"📝 Events saved to {EVENTS_CSV_PATH}")

    if ica_executor is not None:
        ica_executor.shutdown(wait=False)

    # Write final timebase report
    _maybe_write_timebase_report(force=True, label="final")

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

    print("✅ Stream terminated cleanly")
