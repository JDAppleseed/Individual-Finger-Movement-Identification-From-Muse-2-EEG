"""
ISEF / Research Demo Pipeline — STEP 1 (FIXED TIMEBASE + RESUME-SAFE EVENTS)
Muse 2 EEG → LSL → Compression → ICA → Window Prep (cleaned)
Optional: CNN/LSTM inference + latency + MC-dropout uncertainty
Also: keyboard event marking (space=hold event), autosave events

FIXES (this version):
- Stream-relative absolute_v1 timebase with monotonic clamp:
    time_s = lsl_ts - stream_start_lsl_ts (clamped on backward jumps)
- Events aligned to the same stream timebase with clamp protection.
- Per-run timebase fields anchored per run:
    run_start_lsl_ts, run_start_local, clock_offset (in-memory only)
- Resume gating retained; no silent overwrites.
- Session metadata/state sidecars updated to reflect timebase health counters.
"""

# Work around duplicate libomp on macOS (MKL/torch/scipy); must be set before imports.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

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
CHANNELS = 4
N_FINGERS = 6
N_ACTIONS = 3
TIMEBASE_VERSION = "absolute_v1"

MODEL_PATH = "finger_action_model.pt"
SCALER_PATH = "scaler.save"

# =========================
# ===== SAFETY & UNCERTAINTY ======
# =========================
BASE_CONF_THRESH = 0.75
UNCERTAINTY_WEIGHT = 0.5
STABILITY_FRAMES = 3
ENABLE_ACTUATION = True
MC_DROPOUT_PASSES = 10

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
# ===== IMPORTS ===========
# =========================
import argparse
import threading
import time
import csv
import json
import shutil
from datetime import datetime
from collections import deque
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import joblib
from scipy.signal import welch

from pylsl import StreamInlet, resolve_streams, local_clock
from sklearn.decomposition import FastICA
from sklearn.preprocessing import StandardScaler

from utils.experiment_logger import (
    get_subject_id,
    generate_experiment_hash,
    log_experiment
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
from utils.timebase import clamp_monotonic_time

try:
    from pynput import keyboard
except Exception:
    keyboard = None

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# ===== SUBJECT INFO ======
# =========================
GENDER = "M"
AGE = 17
SUBJECT_ID_OVERRIDE = "1-M17"  # Set to None to use auto-increment registry

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
parser.add_argument("--subject-id", type=str, default=None, help="Override subject ID for this run")
parser.add_argument("--init-only", action="store_true", help="Initialize session and exit before LSL streaming")
parser.add_argument("--force-new-session", action="store_true", help="Always start a new session (ignore resume state)")

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

subject_id = SUBJECT_ID_OVERRIDE or get_subject_id(GENDER, AGE)

experiment_config = {
    "sampling_rate": SAMPLING_RATE,
    "window_sec": WINDOW_SEC,
    "channels": CHANNELS,
    "model": "CNNLSTMFingerActionNet + ICA",
}

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

# Defaults
session_id = None
segment_id = 0
total_elapsed_s = 0.0      # session-continuous elapsed time (excludes downtime)
last_time_s = -1.0         # last written session-continuous time_s
BLOCK_ID = 0
experiment_hash = None
features_path_state = None
events_path_state = None
raw_path_state = None
created_utc = None
time_s_clamped_count = 0
event_clamped_count = 0

# Legacy/diagnostic fields (kept for compatibility)
stream_start_lsl_ts = None
local_clock_at_start = None

# Canonical timebase fields (THIS FIX)
run_start_lsl_ts = None     # first LSL timestamp observed THIS run
run_start_local = None      # local_clock() at run start
clock_offset = None         # run_start_lsl_ts - run_start_local

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
        return name[len(prefix):-len("_eeg_features.csv")]
    if name.endswith("_events.csv"):
        return name[len(prefix):-len("_events.csv")]
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

def _write_json_atomic(path: Path, payload: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
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
state_timebase_version = session_state.get("timebase_version") or session_state.get("timebase")

required_feature_cols = {"lsl_timestamp", "time_s"}
state_features_ok = False
state_features_header_ok = False
state_features = None
if resume_subject_match and state_features_path:
    state_features = Path(state_features_path)
    state_features_ok = _csv_has_data_rows(state_features)
    if state_features_ok:
        state_features_header_ok = _header_has_columns(_read_csv_header(state_features), required_feature_cols)
    if not state_features_ok:
        resume_blockers.append("features_missing_or_empty")
    elif not state_features_header_ok:
        resume_blockers.append("features_missing_required_columns")
else:
    if resume_subject_match:
        resume_blockers.append("features_missing")

resolved_events_path = _resolve_events_path(state_events_path, state_features_path, subject_id, state_session_id)
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
    meta_candidate = Path("data/processed") / f"{subject_id}_{state_session_id}_session_meta.json"
    state_meta = _load_session_meta(meta_candidate)
    if not state_timebase_version:
        state_timebase_version = state_meta.get("timebase_version") or state_meta.get("timebase")
if state_timebase_version and state_timebase_version != TIMEBASE_VERSION:
    resume_blockers.append(f"timebase_mismatch({state_timebase_version})")

resume_allowed = resume_subject_match and state_features_ok and state_features_header_ok and events_path_safe
true_resume = resume_requested and resume_allowed and not any(
    b for b in resume_blockers if b not in {"forced_new_session"}
)

if true_resume:
    session_id = state_session_id
    if not session_id:
        session_id = _infer_session_id_from_path(state_features, subject_id) or _infer_session_id_from_path(
            resolved_events_path, subject_id
        )
    BLOCK_ID = int(session_state.get("block_id", 0))
    segment_id = int(session_state.get("segment_id", -1)) + 1
    total_elapsed_s = float(session_state.get("total_elapsed_s", 0.0))
    last_time_s = float(session_state.get("last_time_s", -1.0))
    experiment_hash = session_state.get("experiment_hash")
    features_path_state = state_features_path
    events_path_state = str(resolved_events_path) if resolved_events_path else state_events_path
    raw_path_state = state_raw_path
    created_utc = session_state.get("created_utc") or state_meta.get("created_utc")

    # Load any legacy timebase fields (diagnostics only)
    stream_start_lsl_ts = _coerce_float(session_state.get("stream_start_lsl_ts"))
    local_clock_at_start = _coerce_float(session_state.get("local_clock_at_start"))

    # We intentionally DO NOT reuse old per-run alignment across runs.
    clock_offset = None

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

# If no prior session_id, start fresh
if not session_id:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if not true_resume and SESSION_ID_OVERRIDE:
    session_id = str(SESSION_ID_OVERRIDE)
if experiment_hash is None:
    experiment_hash = generate_experiment_hash(subject_id, experiment_config)

FEATURES_ARCHIVE_DIR = Path("data/processed")

FEATURES_ARCHIVE_PATH = Path(features_path_state) if features_path_state else FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_eeg_features.csv"
EVENTS_ARCHIVE_PATH = Path(events_path_state) if events_path_state else FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_events.csv"
if EVENTS_AUTOSAVE_PATH is None:
    EVENTS_AUTOSAVE_PATH = str(FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_events_autosave.csv")
if EVENTS_CSV_PATH is None:
    EVENTS_CSV_PATH = str(EVENTS_ARCHIVE_PATH)
FEATURES_PATH = FEATURES_ARCHIVE_PATH

RAW_ARCHIVE_PATH = Path(raw_path_state) if raw_path_state else Path("data/raw") / f"{subject_id}_{session_id}_raw.csv"
SESSION_META_PATH = FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_session_meta.json"
ROOT_SESSION_META_PATH = Path("session_meta.json")

def _build_session_meta_payload(complete: bool):
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "experiment_hash": experiment_hash,
        "sampling_rate": SAMPLING_RATE,
        "window_sec": WINDOW_SEC,
        "channels": CHANNELS,
        "features_path": str(FEATURES_ARCHIVE_PATH),
        "events_path": str(EVENTS_ARCHIVE_PATH),
        "raw_path": str(RAW_ARCHIVE_PATH),
        "timebase_version": TIMEBASE_VERSION,
        "timebase": TIMEBASE_VERSION,

        # Canonical (session-continuous) bookkeeping
        "total_elapsed_s": float(total_elapsed_s),
        "last_time_s": float(last_time_s),

        # Legacy/diagnostic (kept)
        "stream_start_lsl_ts": stream_start_lsl_ts,
        "local_clock_at_start": local_clock_at_start,

        "complete": bool(complete),
        "created_utc": created_utc,
        "updated_utc": datetime.utcnow().isoformat() + "Z",
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
    total_elapsed_override=None,
    last_time_override=None,
    block_id_override=None,
    segment_id_override=None,
):
    return {
        "subject_id": subject_id,
        "session_id": session_id,
        "experiment_hash": experiment_hash,
        "block_id": int(block_id_override if block_id_override is not None else BLOCK_ID),
        "segment_id": int(segment_id_override if segment_id_override is not None else segment_id),

        # Canonical session-continuous time
        "total_elapsed_s": float(total_elapsed_override if total_elapsed_override is not None else total_elapsed_s),
        "last_time_s": float(last_time_override if last_time_override is not None else last_time_s),

        "features_path": str(FEATURES_ARCHIVE_PATH),
        "events_path": str(EVENTS_ARCHIVE_PATH),
        "raw_path": str(RAW_ARCHIVE_PATH),
        "created_utc": created_utc,
        "updated_utc": datetime.utcnow().isoformat() + "Z",
        "timebase_version": TIMEBASE_VERSION,
        "timebase": TIMEBASE_VERSION,
        "time_s_clamped_count": int(time_s_clamped_count),
        "event_clamped_count": int(event_clamped_count),

        # Legacy/diagnostic
        "stream_start_lsl_ts": stream_start_lsl_ts,
        "local_clock_at_start": local_clock_at_start,
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

channel_list = "TP9, AF7, AF8, TP10"

print("-" * 50)
print("🧠 EEG SESSION INITIALIZED")
print("-" * 50)
print(f"Subject ID        : {subject_id}")
print(f"Session ID        : {session_id}")
print(f"Experiment Hash   : {experiment_hash}")
print("")
print(f"Resume Requested  : {'YES' if resume_requested else 'NO'} ({resume_request_reason})")
print(f"Resume Decision   : {resume_flag}")
if resume_blockers and not true_resume:
    print(f"Resume Blockers   : {', '.join(resume_blockers)}")
print(f"Current Block ID  : {BLOCK_ID}")
print(f"Total Elapsed Time: {total_elapsed_s:.2f} s (stream-relative, monotonic-clamped)")
print("")
if true_resume and state_events_missing:
    print(f"⚠️ Resume note: events file missing at {resolved_events_path}; a new events file will be created.")
print(f"EEG Channels      : {channel_list} ({CHANNELS})")
print(f"Sampling Rate     : {SAMPLING_RATE} Hz")
print(f"Window Length     : {WINDOW_SEC} s")
print(f"Timebase Version  : {TIMEBASE_VERSION}")
print(f"Run Start LSL     : {_fmt_time_value(run_start_lsl_ts)}")
print(f"Run Start Local   : {_fmt_time_value(run_start_local)}")
print(f"Clock Offset      : {_fmt_time_value(clock_offset)}")
print("")
print("Modes:")
print(f"  Event Marking   : {'ENABLED' if EVENT_MARKING_ENABLED else 'DISABLED'}")
print(f"  Demo Mode       : {'ON' if DEMO_MODE else 'OFF'}")
print(f"  Training Mode   : {'ON' if TRAINING_MODE else 'OFF'}")
print(f"  Actuation       : {'ENABLED' if ENABLE_ACTUATION else 'DISABLED'}")
print("")
print("Output Paths:")
print(f"  Features CSV    : {FEATURES_ARCHIVE_PATH}")
print(f"  Events CSV      : {EVENTS_ARCHIVE_PATH}")
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
    _maybe_backup_path(RAW_ARCHIVE_PATH, "Existing raw file")
    _maybe_backup_path(SESSION_META_PATH, "Existing session meta file")

if not created_utc:
    created_utc = datetime.utcnow().isoformat() + "Z"

_update_session_meta(complete=False, label="init")
_write_session_state(label="init")

log_experiment(
    subject_id,
    experiment_hash,
    step="STEP_1_STREAM",
    notes="EEG collection + ICA + optional inference + event marking (session-continuous timebase)"
)

# =========================
# ===== CALIBRATION STATE ==
# =========================
CALIBRATION_STATE_PATH = Path("logs/calibration") / f"calibration_state_{subject_id}_{experiment_hash}.json"
CALIBRATION_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
calibrator = OnlineCalibrator()

if CALIBRATION_STATE_PATH.exists():
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

def _clamp_event_time(event_time_s: float):
    if event_time_s is None:
        return None, None, False
    if last_sample_time_s is None or stream_start_lsl_ts is None:
        return event_time_s, None, False
    if event_time_s < (last_sample_time_s - 2.0):
        clamped_time_s = float(last_sample_time_s)
        clamped_lsl = float(stream_start_lsl_ts + clamped_time_s)
        return clamped_time_s, clamped_lsl, True
    return event_time_s, None, False

def _record_nearest_sample_delta(event_time_s: float, label: str):
    if event_time_s is None or len(nearest_sample_delta_samples) >= NEAREST_SAMPLE_MAX:
        return
    if not recent_sample_times:
        return
    recent = list(recent_sample_times)
    nearest = min(recent, key=lambda t: abs(t - event_time_s))
    delta = float(event_time_s - nearest)
    nearest_sample_delta_samples.append({
        "label": label,
        "event_time_s": float(event_time_s),
        "nearest_sample_s": float(nearest),
        "delta_s": float(delta),
    })

def _timebase_report_payload():
    dt_median_ms = None
    dt_p95_ms = None
    if len(recent_sample_times) >= 2:
        diffs = np.diff(np.array(recent_sample_times, dtype=float))
        if diffs.size:
            dt_median_ms = float(np.median(diffs) * 1000.0)
            dt_p95_ms = float(np.percentile(diffs, 95) * 1000.0)

    delta_abs = [abs(s["delta_s"]) for s in nearest_sample_delta_samples]
    delta_abs_max = float(max(delta_abs)) if delta_abs else None
    delta_abs_mean = float(np.mean(delta_abs)) if delta_abs else None

    return {
        "ts_utc": datetime.utcnow().isoformat() + "Z",
        "subject_id": subject_id,
        "session_id": session_id,
        "segment_id": segment_id,
        "timebase_version": TIMEBASE_VERSION,
        "stream_start_lsl_ts": stream_start_lsl_ts,
        "stream_start_local": local_clock_at_start,
        "clock_offset": clock_offset,
        "samples_seen": int(samples_seen),
        "samples_written": int(samples_written),
        "time_s_clamped_count": int(time_s_clamped_count),
        "max_backwards_jump_s": float(max_backwards_jump_s),
        "event_stamps_count": int(event_stamps_count),
        "event_clamped_count": int(event_clamped_count),
        "nearest_sample_delta_s_abs_max": delta_abs_max,
        "nearest_sample_delta_s_abs_mean": delta_abs_mean,
        "nearest_sample_delta_s_samples": list(nearest_sample_delta_samples),
        "dt_median_ms": dt_median_ms,
        "dt_p95_ms": dt_p95_ms,
        "first_time_s": first_time_s,
        "last_time_s": last_time_s_seen,
        "first_lsl_ts": first_lsl_ts,
        "last_lsl_ts": last_lsl_ts,
    }

def _write_timebase_report(label: str):
    report_path = FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_timebase_report.json"
    pointer_path = Path("reports") / "last_timebase_report.json"
    try:
        payload = _timebase_report_payload()
        payload["label"] = label
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
        last_report_samples_written = samples_written
        return
    if (now - last_report_time) >= 10.0 or (samples_written - last_report_samples_written) >= 2500:
        _write_timebase_report(label)
        _write_session_state(label=f"timebase_report:{label}")
        last_report_time = now
        last_report_samples_written = samples_written

# =========================
# ===== EVENT MARKING =====
# =========================
events = []
events_lock = threading.Lock()
current_event = None
last_event_index = None
current_action_id = ACTION_REST
current_override = None

trial_id_counter = 0
last_written_time_s = float(last_time_s)
timebase_written = False
clock_offset_estimated = False
time_s_clamped_count = int(time_s_clamped_count)
event_clamped_count = int(event_clamped_count)
event_stamps_count = 0
samples_seen = 0
samples_written = 0
last_sample_time_s = None
last_sample_lsl_ts = None
first_time_s = None
last_time_s_seen = None
first_lsl_ts = None
last_lsl_ts = None
max_backwards_jump_s = 0.0
time_s_clamp_warned = False
event_clamp_warned = False
nearest_sample_delta_samples = []
NEAREST_SAMPLE_MAX = 10
recent_sample_times = deque(maxlen=512)
last_report_time = None
last_report_samples_written = 0
timebase_report_initialized = False

def _load_existing_events(path: Path):
    """Load existing events to support resume without overwriting."""
    loaded = []
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

                if duration_s < 0:
                    duration_s = 0.0
                if np.isfinite(onset_s) and not np.isfinite(end_s):
                    end_s = onset_s + duration_s
                if np.isfinite(onset_lsl) and not np.isfinite(end_lsl):
                    end_lsl = onset_lsl + duration_s

                # IMPORTANT: In session-continuous mode, we do NOT try to back-compute onset_lsl
                # from onset_s unless the file already contains onset_lsl. (Because downtime-free
                # session time can't be inverted without per-run mapping history.)
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
        trial_id_counter = max([int(e.get("trial_id", 0) or 0) for e in loaded_events] + [0])
        print(f"🧾 Resumed events: loaded {len(loaded_events)} existing events (trial_id_counter={trial_id_counter}).")

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
                f"{float(_f(item.get('event_lsl_ts'), np.nan)):.6f}" if np.isfinite(_f(item.get("event_lsl_ts"), np.nan)) else "",
                f"{float(_f(item.get('event_time_s'), np.nan)):.6f}" if np.isfinite(_f(item.get("event_time_s"), np.nan)) else "",
                item.get("type", ""),
                item.get("channel", "n/a"),
                item.get("confidence", ""),
                item.get("notes", ""),
                int(item.get("finger_id", 0)),
                int(item.get("action_id", 0)),
                int(item.get("trial_id", 0)),
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

    if (current_event.get("end_lsl") is None or not np.isfinite(current_event.get("end_lsl", np.nan))) and np.isfinite(onset_lsl):
        current_event["end_lsl"] = float(onset_lsl + duration_s)
    if (current_event.get("end_s") is None or not np.isfinite(current_event.get("end_s", np.nan))) and np.isfinite(onset_s):
        current_event["end_s"] = float(onset_s + duration_s)

    current_event["type"] = event_type_for(
        int(current_event["action_id"]),
        int(current_event["finger_id"]),
        current_event.get("override_type"),
    )

    trial_id_counter += 1
    current_event["trial_id"] = int(trial_id_counter)
    current_event["block_id"] = int(BLOCK_ID)

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
    global current_event, current_action_id, current_override, event_clamped_count, event_clamp_warned
    if not EVENT_MARKING_ENABLED:
        return

    try:
        key_char = key.char
    except AttributeError:
        key_char = None

    # Hold SPACE = event active
    if key == keyboard.Key.space:
        if current_event is None:
            # In this fixed design, we refuse to stamp events until run_start_lsl_ts exists,
            # because that's the anchor for session-continuous time.
            if run_start_lsl_ts is None or clock_offset is None:
                print("⚠️ Event ignored: timebase not initialized yet (waiting for first LSL sample).")
                return

            event_local = local_clock()
            onset_lsl = _lsl_now()
            if onset_lsl is None:
                return

            if stream_start_lsl_ts is None:
                return
            onset_s = float(onset_lsl - stream_start_lsl_ts)
            if onset_s is None:
                return
            clamped_onset_s, clamped_onset_lsl, clamped = _clamp_event_time(onset_s)
            if clamped:
                onset_s = clamped_onset_s
                onset_lsl = clamped_onset_lsl if clamped_onset_lsl is not None else onset_lsl
                event_clamped_count += 1
                if not event_clamp_warned:
                    print("⚠️ Event time behind stream; clamped.")
                    event_clamp_warned = True

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
                "source": "manual",
            }

            # Debug: local-clock relative to run start (NOT used by Step 1b)
            if run_start_local is not None:
                current_event["onset_rel_s"] = float(event_local - run_start_local)
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
    global event_clamped_count, event_clamp_warned
    if not EVENT_MARKING_ENABLED:
        return
    if key == keyboard.Key.space and current_event is not None:
        if run_start_lsl_ts is None or clock_offset is None:
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
        if end_s is None:
            return
        clamped_end_s, clamped_end_lsl, clamped = _clamp_event_time(end_s)
        if clamped:
            end_s = clamped_end_s
            end_lsl = clamped_end_lsl if clamped_end_lsl is not None else end_lsl
            event_clamped_count += 1
            if not event_clamp_warned:
                print("⚠️ Event time behind stream; clamped.")
                event_clamp_warned = True

        duration_s = float(end_s - current_event.get("onset_s", end_s))
        if duration_s < 0:
            duration_s = 0.0

        current_event["duration_s"] = duration_s
        current_event["end_lsl"] = float(end_lsl)
        current_event["end_s"] = float(end_s)

        if run_start_local is not None:
            current_event["end_rel_s"] = float(end_local - run_start_local)

        finalize_event()

if EVENT_MARKING_ENABLED and keyboard is None:
    print("⚠️ Event marking disabled (pynput not installed).")
    EVENT_MARKING_ENABLED = False

listener = None
if EVENT_MARKING_ENABLED:
    listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    listener.daemon = True
    listener.start()

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
        n_channels=CHANNELS,
        n_fingers=N_FINGERS,
        n_actions=N_ACTIONS
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

print("🔍 Resolving EEG stream...")
if CSV_OFFLINE_PATH:
    raise RuntimeError("CSV offline mode is not supported in 1_stream_and_record.py.")
streams = resolve_streams()
if not streams:
    raise RuntimeError("No LSL streams found.")
if LSL_STREAM_NAME:
    target = str(LSL_STREAM_NAME).lower()
    eeg_streams = [s for s in streams if target in s.name().lower()]
elif LSL_STREAM_TYPE:
    target = str(LSL_STREAM_TYPE).lower()
    eeg_streams = [s for s in streams if s.type().lower() == target]
else:
    eeg_streams = [s for s in streams if "eeg" in s.name().lower()]
if not eeg_streams:
    raise RuntimeError("No matching LSL EEG stream found.")
eeg_stream = eeg_streams[0]
inlet = StreamInlet(eeg_stream)
info = inlet.info()
expected_labels = ["TP9", "AF7", "AF8", "TP10"]
channel_indices = resolve_eeg_channel_indices(info, expected_labels)
if channel_indices is None:
    if info.channel_count() >= CHANNELS:
        channel_indices = list(range(CHANNELS))
        print("⚠️ EEG channel labels not found; using first four channels by index.")
    else:
        raise RuntimeError(
            f"LSL EEG stream has {info.channel_count()} channels; expected at least {CHANNELS}."
        )
print(f"✅ EEG connected ({info.channel_count()} channels, using indices {channel_indices})")

# =========================
# ===== BUFFERS ===========
# =========================
buffer_len = int(WINDOW_SEC * SAMPLING_RATE)
eeg_buffer = deque(maxlen=buffer_len)
action_pred_buffer = deque(maxlen=STABILITY_FRAMES)

# =========================
# ===== CSV OUTPUT ========
# =========================
csv_file = None
csv_writer = None
raw_file = None
raw_writer = None

header = [
    "lsl_timestamp",
    "time_s",
    "ch1", "ch2", "ch3", "ch4",
    "pred_action",
    "pred_finger",
    "action_confidence",
    "action_uncertainty",
    "finger_confidence",
    "finger_uncertainty",
    "velocity",
    "latency_ms"
]

if SAVE_TO_DISK:
    features_exists = FEATURES_PATH.exists() and FEATURES_PATH.stat().st_size > 0
    csv_file = open(FEATURES_PATH, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not features_exists:
        csv_writer.writerow(header)

if SAVE_RAW:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_ARCHIVE_PATH
    raw_exists = raw_path.exists() and raw_path.stat().st_size > 0
    raw_file = open(raw_path, "a", newline="")
    raw_writer = csv.writer(raw_file)
    if not raw_exists:
        raw_writer.writerow(["lsl_timestamp", "ch1", "ch2", "ch3", "ch4"])

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

ica = FastICA(n_components=CHANNELS, random_state=42)
ica_scaler = StandardScaler()
ica_fitted = False
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

# =========================
# ===== MAIN LOOP =========
# =========================
print("▶ Streaming — type 'end_stream' OR press q/ESC to stop")

try:
    while not stop_event.is_set():
        sample, lsl_ts = inlet.pull_sample(timeout=0.0)
        if sample is None:
            continue

        if len(sample) < max(channel_indices) + 1:
            raise RuntimeError(
                f"LSL EEG sample has {len(sample)} channels; expected at least {max(channel_indices)+1}."
            )
        if len(sample) != CHANNELS:
            sample = [sample[i] for i in channel_indices]

        # =========================
        # ===== TIMEBASE INIT ======
        # =========================
        # On the FIRST sample of THIS run, anchor run_start_lsl_ts and compute clock_offset.
        # This makes time_s continuous across resumes without including downtime.
        if run_start_lsl_ts is None:
            run_start_lsl_ts = float(lsl_ts)
            run_start_local = float(local_clock())
            clock_offset = float(run_start_lsl_ts - run_start_local)

            # Legacy diagnostics: keep stream_start_* set once if absent
            if stream_start_lsl_ts is None:
                stream_start_lsl_ts = run_start_lsl_ts
            if local_clock_at_start is None:
                local_clock_at_start = run_start_local

            timebase_written = False
            timebase_report_initialized = False

        # If clock_offset is missing (shouldn't happen after init), estimate from current sample
        if clock_offset is None:
            clock_offset = float(lsl_ts - local_clock())
            if not clock_offset_estimated:
                print("⚠️ clock_offset missing; estimating from current sample.")
                clock_offset_estimated = True
            timebase_written = False

        if not timebase_written and run_start_lsl_ts is not None and clock_offset is not None:
            _update_session_meta(complete=False, label="timebase")
            _write_session_state(label="timebase")
            timebase_written = True
        if run_start_lsl_ts is not None and not timebase_report_initialized:
            _write_timebase_report("init")
            timebase_report_initialized = True

        samples_seen += 1
        raw_time_s = None
        if stream_start_lsl_ts is not None:
            raw_time_s = float(lsl_ts - stream_start_lsl_ts)
        clamped_time_s, clamped = clamp_monotonic_time(last_sample_time_s, raw_time_s)
        if clamped and raw_time_s is not None and last_sample_time_s is not None:
            time_s_clamped_count += 1
            max_backwards_jump_s = max(max_backwards_jump_s, float(last_sample_time_s - raw_time_s))
            if not time_s_clamp_warned:
                print("⚠️ LSL time went backwards; time_s was clamped.")
                time_s_clamp_warned = True
        if clamped_time_s is not None and stream_start_lsl_ts is not None:
            last_sample_time_s = float(clamped_time_s)
            last_sample_lsl_ts = float(stream_start_lsl_ts + clamped_time_s)
            recent_sample_times.append(last_sample_time_s)
            if first_time_s is None:
                first_time_s = last_sample_time_s
            last_time_s_seen = last_sample_time_s
            if first_lsl_ts is None:
                first_lsl_ts = last_sample_lsl_ts
            last_lsl_ts = last_sample_lsl_ts

        latency_ms = (local_clock() - lsl_ts) * 1000.0

        eeg_buffer.append(sample)

        if raw_writer:
            raw_writer.writerow([lsl_ts, *sample])

        if len(eeg_buffer) < buffer_len:
            continue

        window = np.array(eeg_buffer)  # (T, C)

        # ===== Compression =====
        diff_mask = np.any(np.diff(window, axis=0) != 0, axis=1)
        compressed = window[1:][diff_mask]
        if len(compressed) < CHANNELS:
            continue

        # ===== ICA =====
        if not ica_fitted:
            X_scaled = ica_scaler.fit_transform(compressed)
            ica.fit(X_scaled)
            ica_fitted = True

        X_scaled = ica_scaler.transform(window)
        S = ica.transform(X_scaled)

        for k in range(S.shape[1]):
            if is_artifact(S[:, k], fs=SAMPLING_RATE):
                S[:, k] *= ARTIFACT_ATTENUATION

        cleaned = ica.inverse_transform(S)
        latest_sample = cleaned[-1] if len(cleaned) else window[-1]

        # ===== DEFAULTS =====
        pred_action = -1
        pred_finger = -1
        action_confidence = 0.0
        action_uncertainty = 0.0
        finger_confidence = 0.0
        finger_uncertainty = 0.0
        velocity = 0.0

        # ===== INFERENCE =====
        if DEMO_MODE and model is not None:
            t0 = time.perf_counter()

            window_input = cleaned.astype(np.float32)
            window_input = standardize_window_TxC(window_input, scaler)

            x_BTC = torch.tensor(window_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            if hasattr(model, "mc_forward"):
                mc = model.mc_forward(x_BTC, passes=MC_DROPOUT_PASSES)
            else:
                mc = mc_dropout_predict(model, x_BTC, passes=MC_DROPOUT_PASSES)

            action_mean = mc["action_mean"].squeeze(0).detach().cpu().numpy()
            action_std  = mc["action_std"].squeeze(0).detach().cpu().numpy()
            finger_mean = mc["finger_mean"].squeeze(0).detach().cpu().numpy()
            finger_std  = mc["finger_std"].squeeze(0).detach().cpu().numpy()

            pred_action = int(np.argmax(action_mean))
            action_confidence = float(action_mean[pred_action])
            action_uncertainty = float(np.mean(action_std))

            pred_finger = int(np.argmax(finger_mean))
            finger_confidence = float(finger_mean[pred_finger])
            finger_uncertainty = float(np.mean(finger_std))

            adaptive_thresh = min(
                0.99,
                max(BASE_CONF_THRESH, BASE_CONF_THRESH + UNCERTAINTY_WEIGHT * action_uncertainty)
            )

            action_pred_buffer.append(pred_action)

            if pred_action != ACTION_REST:
                velocity = action_confidence * (1.0 - action_uncertainty)

            if ENABLE_ACTUATION and pred_action != ACTION_REST:
                stable = (len(action_pred_buffer) == STABILITY_FRAMES and len(set(action_pred_buffer)) == 1)
                if stable and calibrator.allow_actuation(action_confidence, action_uncertainty) and action_confidence >= adaptive_thresh:
                    pass

            _ = time.perf_counter() - t0

        # ===== OPTIONAL ONLINE CALIBRATION FEEDBACK =====
        if DEMO_MODE and TRAINING_MODE:
            try:
                true_action = int(input("Action label (0=REST,1=OPEN,2=CLOSE): "))
                true_finger = int(input("Finger label (0=NONE,1-5): "))

                correct = (pred_action == true_action) and (true_action == ACTION_REST or pred_finger == true_finger)

                calibrator.update(action_confidence, correct)
                record_prediction(
                    subject_id=subject_id,
                    experiment_hash=experiment_hash,
                    confidence=action_confidence,
                    uncertainty=action_uncertainty,
                    correct=correct,
                    threshold=calibrator.threshold
                )

                CALIBRATION_STATE_PATH.write_text(json.dumps({
                    "threshold": calibrator.threshold,
                    "history": [],
                    "config": {
                        "init_threshold": calibrator.threshold,
                        "min_threshold": calibrator.min_threshold,
                        "max_threshold": calibrator.max_threshold,
                        "ema_alpha": calibrator.alpha,
                    }
                }, indent=2))
            except Exception:
                pass

        # ===== SAVE FEATURES =====
        if SAVE_TO_DISK and csv_writer:
            time_s = clamped_time_s
            if time_s is None:
                continue

            row = [
                lsl_ts,
                time_s,
                *latest_sample,
                pred_action,
                pred_finger,
                action_confidence,
                action_uncertainty,
                finger_confidence,
                finger_uncertainty,
                velocity,
                latency_ms
            ]
            if len(row) != len(header):
                raise RuntimeError(f"Feature row length {len(row)} does not match header length {len(header)}")
            csv_writer.writerow(row)
            last_written_time_s = float(time_s)
            last_time_s = float(last_written_time_s)  # keep session meta/state current
            samples_written += 1
            _maybe_write_timebase_report()

        # ===== PLOT =====
        if ENABLE_PLOT:
            for i in range(CHANNELS):
                lines[i].set_data(range(len(cleaned)), cleaned[:, i])
            ax.set_xlim(0, len(cleaned))
            ax.relim()
            ax.autoscale_view()
            ax.set_ylim(-100, 100)
            latency_text.set_text(f"Latency: {latency_ms:.1f} ms" if DEMO_MODE else "")
            info_text.set_text(
                f"Act: {ACTION_NAMES.get(pred_action, '?')}  Conf: {action_confidence:.2f}  Unc: {action_uncertainty:.3f}  Vel: {velocity:.3f}"
                if DEMO_MODE else ""
            )
            plt.pause(0.001)

except KeyboardInterrupt:
    pass

# =========================
# ===== CLEANUP ===========
# =========================
print("\n🧹 Cleaning up")

if csv_file:
    csv_file.flush()
    csv_file.close()

if raw_file:
    raw_file.flush()
    raw_file.close()

if ENABLE_PLOT:
    plt.close("all")

if EVENT_MARKING_ENABLED:
    if listener:
        listener.stop()
    with events_lock:
        save_events_csv(EVENTS_CSV_PATH, events)
    print(f"📝 Events saved to {EVENTS_CSV_PATH}")

# Write final timebase report
_maybe_write_timebase_report(force=True, label="final")

# Update session-continuous elapsed time at end of run
last_time_s_updated = float(last_written_time_s if last_written_time_s >= 0 else last_time_s)
total_elapsed_s_updated = float(last_time_s_updated if last_time_s_updated >= 0 else total_elapsed_s)

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
