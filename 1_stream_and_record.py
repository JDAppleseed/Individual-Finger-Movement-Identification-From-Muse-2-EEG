"""
ISEF / Research Demo Pipeline — STEP 1
Muse 2 EEG → LSL → Compression → ICA → Window Prep (cleaned)
Optional: CNN/LSTM inference + latency + MC-dropout uncertainty
Also: keyboard event marking (space=hold event), autosave events
"""

# Work around duplicate libomp on macOS (MKL/torch/scipy); must be set before imports.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# =========================
# ===== CONFIG FLAGS ======
# =========================
#                                          
TRAINING_MODE = False          # If True, allows optional interactive correctness feedback for calibrator
DEMO_MODE = False              # If True, loads model and runs inference/uncertainty
ENABLE_PLOT = True
SAVE_TO_DISK = True
SAVE_RAW = True

SAMPLING_RATE = 256
WINDOW_SEC = 0.25
CHANNELS = 4
N_FINGERS = 6
N_ACTIONS = 3

MODEL_PATH = "finger_action_model.pt"   # model weights file
SCALER_PATH = "scaler.save"             # Per-channel normalizer from Step 2 (optional for inference)

# =========================
# ===== SAFETY & UNCERTAINTY ======
# =========================

BASE_CONF_THRESH = 0.75        # base softmax threshold (lower bound)
UNCERTAINTY_WEIGHT = 0.5       # how much uncertainty increases threshold
STABILITY_FRAMES = 3           # consecutive agreement frames required
ENABLE_ACTUATION = True
MC_DROPOUT_PASSES = 10         # MC passes for uncertainty

# =========================
# ===== EVENT MARKING =====
# =========================

EVENT_MARKING_ENABLED = True
EVENTS_CSV_PATH = None
EVENTS_AUTOSAVE_PATH = None
EVENTS_CHANNEL = "n/a"

# =========================
# ===== IMPORTS ===========
# =========================

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

try:
    from pynput import keyboard
except Exception:
    keyboard = None

from models.cnn_lstm_finger_action_net import CNNLSTMFingerActionNet

# =========================
# ===== DEVICE ============
# =========================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# ===== SUBJECT INFO ======
# =========================

GENDER = "M"     # M / F / X
AGE = 16         # integer only
SUBJECT_ID_OVERRIDE = "3-M16"  # Set to None to use auto-increment registry

subject_id = SUBJECT_ID_OVERRIDE or get_subject_id(GENDER, AGE)

experiment_config = {
    "sampling_rate": SAMPLING_RATE,
    "window_sec": WINDOW_SEC,
    "channels": CHANNELS,
    "model": "CNNLSTMFingerActionNet + ICA",
}

SESSION_STATE_DIR = Path("logs")
SESSION_STATE_DIR.mkdir(parents=True, exist_ok=True)
SESSION_STATE_PATH = SESSION_STATE_DIR / f"session_state_{subject_id}.json"

session_state = {}
if SESSION_STATE_PATH.exists():
    try:
        session_state = json.loads(SESSION_STATE_PATH.read_text())
    except Exception as e:
        print(f"⚠️ Failed to load session state: {e}")

session_id = None
segment_id = 0
total_elapsed_s = 0.0
last_time_s = -1.0
BLOCK_ID = 0
experiment_hash = None
features_path_state = None
events_path_state = None
raw_path_state = None
created_utc = session_state.get("created_utc")

if session_state.get("subject_id") == subject_id:
    session_id = session_state.get("session_id")
    BLOCK_ID = int(session_state.get("block_id", 0))
    segment_id = int(session_state.get("segment_id", -1)) + 1
    total_elapsed_s = float(session_state.get("total_elapsed_s", 0.0))
    last_time_s = float(session_state.get("last_time_s", -1.0))
    experiment_hash = session_state.get("experiment_hash")
    features_path_state = session_state.get("features_path")
    events_path_state = session_state.get("events_path")
    raw_path_state = session_state.get("raw_path")

if not session_id:
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if experiment_hash is None:
    experiment_hash = generate_experiment_hash(subject_id, experiment_config)

FEATURES_ARCHIVE_DIR = Path("data/processed")
FEATURES_ARCHIVE_PATH = Path(features_path_state) if features_path_state else FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_eeg_features.csv"
EVENTS_ARCHIVE_PATH = Path(events_path_state) if events_path_state else FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_events.csv"
EVENTS_AUTOSAVE_PATH = str(FEATURES_ARCHIVE_DIR / f"{subject_id}_{session_id}_events_autosave.csv")
EVENTS_CSV_PATH = str(EVENTS_ARCHIVE_PATH)
FEATURES_PATH = FEATURES_ARCHIVE_PATH
RAW_ARCHIVE_PATH = Path(raw_path_state) if raw_path_state else Path("data/raw") / f"{subject_id}_{session_id}_raw.csv"

session_meta = {
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
}
Path("session_meta.json").write_text(json.dumps(session_meta, indent=2))

resume_session = bool(session_state.get("subject_id") == subject_id)
resume_flag = "YES" if resume_session else "NO"
channel_list = "TP9, AF7, AF8, TP10"

print("-" * 50)
print("🧠 EEG SESSION INITIALIZED")
print("-" * 50)
print(f"Subject ID        : {subject_id}")
print(f"Session ID        : {session_id}")
print(f"Experiment Hash   : {experiment_hash}")
print("")
print(f"Resume Session    : {resume_flag}")
print(f"Current Block ID  : {BLOCK_ID}")
print(f"Total Elapsed Time: {total_elapsed_s:.2f} s")
print("")
print(f"EEG Channels      : {channel_list} ({CHANNELS})")
print(f"Sampling Rate     : {SAMPLING_RATE} Hz")
print(f"Window Length     : {WINDOW_SEC} s")
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
print("  ✔ Time base aligned (event clock)")
print("  ✔ Resume-safe logging enabled")
print("-" * 50)
print("▶ Streaming started…")
print("-" * 50)
print("Type 'q' or press ESC to stop and safely close the session.")

log_experiment(
    subject_id,
    experiment_hash,
    step="STEP_1_STREAM",
    notes="EEG collection + ICA + optional inference + event marking"
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
# ===== EVENT MARKING =====
# =========================

events = []
events_lock = threading.Lock()
current_event = None
last_event_index = None
current_action_id = ACTION_REST
current_override = None
stream_start_ts = None
clock_offset = None
trial_id_counter = 0
TIME_EPS = 1e-4
last_event_clock_s = 0.0
last_written_time_s = float(last_time_s)
time_s_backwards_skips = 0

def save_events_csv(path, items):
    header = [
        "onset_s",
        "duration_s",
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
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for item in items:
            writer.writerow([
                f"{item['onset_s']:.4f}",
                f"{item['duration_s']:.4f}",
                item["type"],
                item["channel"],
                item.get("confidence", ""),
                item.get("notes", ""),
                int(item["finger_id"]),
                int(item["action_id"]),
                int(item.get("trial_id", 0)),
                int(item.get("block_id", 0)),
                item["source"],
            ])

    if EVENTS_ARCHIVE_PATH and str(path) != str(EVENTS_ARCHIVE_PATH):
        FEATURES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, EVENTS_ARCHIVE_PATH)

def finalize_event(duration_s):
    global current_event, last_event_index, trial_id_counter
    if current_event is None:
        return

    current_event["duration_s"] = max(0.0, float(duration_s))
    current_event["type"] = event_type_for(
        current_event["action_id"],
        current_event["finger_id"],
        current_event.get("override_type"),
    )
    trial_id_counter += 1
    current_event["trial_id"] = int(trial_id_counter)
    current_event["block_id"] = int(BLOCK_ID)

    with events_lock:
        events.append(current_event)
        last_event_index = len(events) - 1
        save_events_csv(EVENTS_AUTOSAVE_PATH, events)
    current_event = None

def update_last_event_finger(finger_id):
    global last_event_index
    with events_lock:
        if last_event_index is None or last_event_index >= len(events):
            return
        event = events[last_event_index]
        event["finger_id"] = int(finger_id)
        event["override_type"] = None
        event["type"] = event_type_for(event["action_id"], event["finger_id"])
        save_events_csv(EVENTS_AUTOSAVE_PATH, events)

def _event_time_seconds():
    if stream_start_ts is None or clock_offset is None:
        return None
    return (local_clock() + clock_offset) - stream_start_ts

def on_key_press(key):
    global current_event, current_action_id, current_override
    if not EVENT_MARKING_ENABLED:
        return

    try:
        key_char = key.char
    except AttributeError:
        key_char = None

    # Hold SPACE = event active
    if key == keyboard.Key.space:
        if current_event is None:
            onset = _event_time_seconds()
            if onset is None:
                return
            current_event = {
                "onset_s": float(onset),
                "duration_s": 0.0,
                "type": "",
                "channel": EVENTS_CHANNEL,
                "confidence": "",
                "notes": "",
                "finger_id": FINGER_NONE,
                "action_id": int(current_action_id),
                "override_type": current_override,
                "source": "manual",
            }
        return

    if key_char is None:
        return

    key_char = key_char.lower()

    # Quick stop hotkey (avoids stdin conflicts)
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
    if not EVENT_MARKING_ENABLED:
        return
    if key == keyboard.Key.space and current_event is not None:
        end_time = _event_time_seconds()
        if end_time is None:
            return
        duration = end_time - current_event["onset_s"]
        finalize_event(duration)

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
    """
    Standardize each channel over time using saved normalizer stats.
    Accepts dict {mean,std} or sklearn StandardScaler (mean_/scale_).
    window_TxC shape: (T, C)
    """
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
    """
    Generic MC dropout helper if your model doesn't implement mc_forward().
    Assumes forward returns (finger_logits, action_logits).
    """
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
    probs_f = torch.stack(probs_f, dim=0)  # [MC,B,F]
    probs_a = torch.stack(probs_a, dim=0)  # [MC,B,A]
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

    # IMPORTANT: keep dropout active for MC passes
    model.train()

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
streams = resolve_streams()
eeg_stream = next(s for s in streams if "EEG" in s.type() or "EEG" in s.name())
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

# stability gating based on ACTION only (since actuation is action-driven)
action_pred_buffer = deque(maxlen=STABILITY_FRAMES)

# =========================
# ===== CSV OUTPUT ========
# =========================

csv_file = None
csv_writer = None
archive_file = None
archive_writer = None
raw_file = None
raw_writer = None

if SAVE_TO_DISK:
    features_exists = FEATURES_PATH.exists() and FEATURES_PATH.stat().st_size > 0
    FEATURES_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

    csv_file = open(FEATURES_PATH, "a", newline="")
    csv_writer = csv.writer(csv_file)
    archive_writer = None

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

    if not features_exists:
        csv_writer.writerow(header)

if SAVE_RAW:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = RAW_ARCHIVE_PATH
    raw_exists = raw_path.exists()
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
ARTIFACT_ATTENUATION = 0.3  # 0.0 hard-zero, 1.0 no suppression

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

print("▶ Streaming — type 'q' to stop")

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

        # align local clock to stream time for event marking
        if stream_start_ts is None:
            stream_start_ts = lsl_ts
            clock_offset = stream_start_ts - local_clock()

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

        cleaned = ica.inverse_transform(S)  # (T', C)

        # ===== FEATURE LOGGING =====
        latest_sample = cleaned[-1] if len(cleaned) else window[-1]

        # ===== DEFAULTS =====
        pred_action = -1
        pred_finger = -1
        latency_ms = -1.0
        action_confidence = 0.0
        action_uncertainty = 0.0
        finger_confidence = 0.0
        finger_uncertainty = 0.0
        velocity = 0.0

        # ===== INFERENCE =====
        if DEMO_MODE and model is not None:
            t0 = time.perf_counter()

            window_input = cleaned.astype(np.float32)                 # (T, C)
            window_input = standardize_window_TxC(window_input, scaler)

            # Model expects (B, T, C)
            x_BTC = torch.tensor(window_input, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Use model.mc_forward if available, else generic MC helper
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

            # stability gate on action only
            action_pred_buffer.append(pred_action)

            # confidence-weighted velocity (only if moving)
            if pred_action != ACTION_REST:
                velocity = action_confidence * (1.0 - action_uncertainty)

            # actuation decision (your Arduino/Bluetooth send would live here)
            if ENABLE_ACTUATION and pred_action != ACTION_REST:
                stable = (len(action_pred_buffer) == STABILITY_FRAMES and len(set(action_pred_buffer)) == 1)
                if stable and calibrator.allow_actuation(action_confidence, action_uncertainty) and action_confidence >= adaptive_thresh:
                    # actuate=True (hook into robotics layer here)
                    pass

            latency_ms = (time.time() - lsl_ts) * 1000.0

            _ = time.perf_counter() - t0

        # ===== OPTIONAL ONLINE CALIBRATION FEEDBACK =====
        if DEMO_MODE and TRAINING_MODE:
            # only if you explicitly run with both True
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

        # ===== SAVE =====
        if SAVE_TO_DISK and csv_writer:
            event_clock_s = _event_time_seconds()
            if event_clock_s is None:
                continue
            last_event_clock_s = event_clock_s
            time_s = total_elapsed_s + event_clock_s
            if last_written_time_s >= 0 and time_s < (last_written_time_s - TIME_EPS):
                time_s_backwards_skips += 1
                if time_s_backwards_skips <= 5:
                    print(
                        f"⚠️ time_s went backwards ({time_s:.6f} < {last_written_time_s:.6f}); clamping."
                    )
                time_s = last_written_time_s + TIME_EPS

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
                raise RuntimeError(
                    f"Feature row length {len(row)} does not match header length {len(header)}"
                )
            csv_writer.writerow(row)
            last_written_time_s = time_s

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
                f"Act: {ACTION_NAMES.get(pred_action, '?')}  "
                f"Conf: {action_confidence:.2f}  Unc: {action_uncertainty:.3f}  Vel: {velocity:.3f}"
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

segment_duration = max(0.0, float(last_event_clock_s))
last_time_s_updated = last_written_time_s if last_written_time_s >= 0 else last_time_s
total_elapsed_s_updated = total_elapsed_s + segment_duration

SESSION_STATE_PATH.write_text(json.dumps({
    "subject_id": subject_id,
    "session_id": session_id,
    "experiment_hash": experiment_hash,
    "block_id": int(BLOCK_ID) + 1,
    "segment_id": int(segment_id),
    "total_elapsed_s": float(total_elapsed_s_updated),
    "last_time_s": float(last_time_s_updated),
    "features_path": str(FEATURES_ARCHIVE_PATH),
    "events_path": str(EVENTS_ARCHIVE_PATH),
    "raw_path": str(RAW_ARCHIVE_PATH),
    "created_utc": created_utc or datetime.utcnow().isoformat() + "Z",
    "updated_utc": datetime.utcnow().isoformat() + "Z",
}, indent=2))

print("✅ Stream terminated cleanly")
