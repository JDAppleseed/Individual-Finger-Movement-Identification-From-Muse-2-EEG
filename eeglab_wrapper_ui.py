from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
if sys.version_info[:2] != (3, 11):
    raise RuntimeError(f"Wrong Python. Expected 3.11, got {sys.version.split()[0]} at {sys.executable}")
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from PySide6.QtCore import QObject, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPen,
    QSyntaxHighlighter,
    QTextCharFormat,
    QPalette,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProxyStyle,
    QGraphicsDropShadowEffect,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QToolButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
    QSizePolicy,
    QStyle,
    QStyleOptionFrame,
    QStyleOptionViewItem,
    QStyledItemDelegate,
)

from app.config_model import (
    TIMEBASE_VERSION,
    SessionSnapshot,
    build_config,
    default_export_settings,
    default_infer_settings,
    default_preprocess_settings,
    default_step1_settings,
    default_step1b_settings,
    default_train_settings,
    write_json,
)
from app.autofill_utils import should_replace_autofilled_text
from app.paths import (
    SubjectInfo,
    ensure_project,
    ensure_session_dirs,
    ensure_subject_dirs,
    list_projects,
    list_subjects,
    next_available_path,
    session_backend_id,
    session_root,
    subject_meta_path,
    subject_root,
    ui_session_id,
)
from app.process_runner import ProcessRunner
from app.replay_path_utils import resolve_replay_artifact_paths
from app.repo_probe import discover_scripts
from app.ui_config_validation import validate_step_settings
from muse_streaming.config import DEFAULT_STREAM_NAME, DEFAULT_STREAM_TYPE
from muse_streaming.healthcheck import run_healthcheck
from utils.label_schema import ACTION_NAMES, FINGER_NAMES
from utils.session_layout import SessionLayout, resolve_latest_run_dir
from visualization.live_viz import parse_viz_line
from visualization.replay_viz import ReplayVisualizer

try:
    import pylsl

    LSL_AVAILABLE = True
except Exception:
    pylsl = None
    LSL_AVAILABLE = False

try:
    import pyqtgraph as pg

    PYQTGRAPH_AVAILABLE = True
except Exception:
    pg = None
    PYQTGRAPH_AVAILABLE = False


@dataclass
class StreamInfo:
    name: str
    stype: str


@dataclass
class ArgSpec:
    name: str
    flag: str
    kind: str
    description: str


def _latest_dir_by_mtime(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.iterdir() if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


class MuseConnectorController(QObject):
    log_line = Signal(str)
    status_changed = Signal(str)
    device_changed = Signal(str)
    stream_changed = Signal(str)
    process_exited = Signal(int)
    error = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process: Optional[subprocess.Popen[str]] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._status = "idle"
        self._device = "-"
        self._stream = "-"
        self._alive = True
        self.destroyed.connect(self._mark_dead)

    def _mark_dead(self, *_args: object) -> None:
        self._alive = False
        self._stop_event.set()

    def _safe_emit(self, signal: Signal, *args: object) -> bool:
        if not self._alive:
            return False
        try:
            signal.emit(*args)
            return True
        except RuntimeError:
            self._alive = False
            self._stop_event.set()
            return False

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def process_id(self) -> int:
        with self._lock:
            proc = self._process
        if proc is None:
            return 0
        try:
            return int(proc.pid or 0)
        except Exception:
            return 0

    def start(
        self,
        args: list[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        with self._lock:
            if self.is_running():
                self.error.emit("Connector already running.")
                return
            merged_env = os.environ.copy()
            merged_env["PYTHONUNBUFFERED"] = "1"
            if env:
                merged_env.update(env)
            self._stop_event.clear()
            try:
                self._process = subprocess.Popen(
                    args,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    env=merged_env,
                )
            except Exception as exc:
                self._process = None
                self._set_status("error")
                self._safe_emit(self.error, f"Failed to start connector: {exc}")
                return
            self._set_status("scanning")
            if self._process.stdout is not None:
                threading.Thread(
                    target=self._read_stream,
                    args=(self._process.stdout, ""),
                    daemon=True,
                ).start()
            if self._process.stderr is not None:
                threading.Thread(
                    target=self._read_stream,
                    args=(self._process.stderr, "[stderr] "),
                    daemon=True,
                ).start()
            threading.Thread(target=self._wait_for_exit, daemon=True).start()

    def stop(self) -> None:
        self.stop_staged()

    def stop_staged(
        self,
        *,
        sigint_timeout_s: float = 1.5,
        sigterm_timeout_s: float = 1.0,
        sigkill_timeout_s: float = 1.0,
    ) -> None:
        with self._lock:
            proc = self._process
        if proc is None:
            return

        def _stopper() -> None:
            try:
                if proc.poll() is not None:
                    return
                try:
                    os.kill(proc.pid, signal.SIGINT)
                    self._safe_emit(self.log_line, f"[connector] Sent SIGINT to PID {proc.pid}")
                except Exception as exc:
                    self._safe_emit(self.log_line, f"[connector] SIGINT failed: {exc}; using terminate()")
                    proc.terminate()
                deadline = time.monotonic() + float(sigint_timeout_s)
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        return
                    time.sleep(0.05)
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                    self._safe_emit(self.log_line, f"[connector] Sent SIGTERM to PID {proc.pid}")
                except Exception as exc:
                    self._safe_emit(self.log_line, f"[connector] SIGTERM failed: {exc}; using terminate()")
                    proc.terminate()
                deadline = time.monotonic() + float(sigterm_timeout_s)
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        return
                    time.sleep(0.05)
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                    self._safe_emit(self.log_line, f"[connector] Sent SIGKILL to PID {proc.pid}")
                except Exception as exc:
                    self._safe_emit(self.log_line, f"[connector] SIGKILL failed: {exc}; using kill()")
                    proc.kill()
                deadline = time.monotonic() + float(sigkill_timeout_s)
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        return
                    time.sleep(0.05)
            except Exception as exc:
                self._safe_emit(self.error, f"Connector stop failed: {exc}")

        threading.Thread(target=_stopper, daemon=True).start()

    def _set_status(self, status: str) -> None:
        if status == self._status:
            return
        self._status = status
        self._safe_emit(self.status_changed, status)

    def _read_stream(self, stream, prefix: str) -> None:
        try:
            for line in iter(stream.readline, ""):
                if self._stop_event.is_set() or not self._alive:
                    break
                clean = line.rstrip()
                if clean:
                    if not self._safe_emit(self.log_line, prefix + clean):
                        break
                    self._parse_status(clean)
        except Exception as exc:
            if self._alive:
                self._safe_emit(self.error, f"Connector stream read failed: {exc}")
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _parse_status(self, line: str) -> None:
        lowered = line.lower()
        if "scanning for muse 2" in lowered or "ble scan round" in lowered:
            self._set_status("scanning")
            return
        if "selected ble device" in lowered:
            self._set_status("connected")
            parts = line.split(":", 1)
            if len(parts) == 2:
                self._safe_emit(self.device_changed, parts[1].strip())
            return
        if "muse 2 connected" in lowered:
            self._set_status("connected")
            return
        if "lsl outlet started" in lowered or "simulated lsl outlet started" in lowered:
            self._set_status("streaming")
            name = self._extract_stream_name(line)
            if name:
                self._safe_emit(self.stream_changed, name)
            return
        if "no muse device found" in lowered or line.startswith("❌"):
            self._set_status("error")
            return

    def _extract_stream_name(self, line: str) -> Optional[str]:
        if "name=" not in line:
            return None
        try:
            after = line.split("name=", 1)[1]
            return after.split(",", 1)[0].strip()
        except Exception:
            return None

    def _wait_for_exit(self) -> None:
        proc = self._process
        if proc is None:
            return
        try:
            code = proc.wait()
        except Exception:
            code = -1
        finally:
            self._stop_event.set()
            with self._lock:
                self._process = None
            if code != 0:
                self._set_status("error")
            else:
                self._set_status("idle")
            self._safe_emit(self.process_exited, int(code))


TOOLTIPS: Dict[str, str] = {
    "MODE": "Runtime mode (train_record for lossless capture, live_infer for deployment).",
    "ALLOW_DROP": "Allow queue eviction in live_infer (never allowed in train_record).",
    "TRAINING_MODE": "Enable training capture mode (no live inference).",
    "DEMO_MODE": "Enable demo/inference mode during streaming.",
    "ENABLE_PLOT": "Show live plots during streaming.",
    "SAVE_TO_DISK": "Write features/events to disk during run.",
    "SAVE_RAW": "Write raw EEG CSV.",
    "ENABLE_FEATURES": "Write feature CSVs (disabled in train_record).",
    "ENABLE_INFERENCE": "Enable live inference (live_infer mode only).",
    "ENABLE_ICA": "Enable ICA preprocessing during streaming.",
    "ICA_WARMUP_S": "Seconds of warmup before ICA fitting.",
    "ICA_MIN_SAMPLES": "Minimum samples required for ICA fit.",
    "ICA_MIN_VAR": "Minimum per-channel variance required for ICA.",
    "ICA_FAIL_POLICY": "ICA failure policy (skip to keep streaming).",
    "ICA_MAX_RETRIES_PER_SESSION": "Max ICA retries before disabling.",
    "LOG_ICA_DIAGNOSTICS": "Log ICA diagnostics when skipped.",
    "DATA_STREAM_TIMEOUT_S": "Seconds before stream stall disables event marking.",
    "DATA_STREAM_CHECK_INTERVAL_S": "Stream health check interval in seconds.",
    "PROCESSING_QUEUE_MAXSIZE": "Max size of processing queue (advanced).",
    "RAW_QUEUE_MAXSIZE": "Max size of raw queue (advanced).",
    "MAX_BACKPRESSURE_S": "Seconds of sustained backpressure before abort (train_record).",
    "QUEUE_PUT_TIMEOUT_S": "Queue put timeout seconds (train_record).",
    "RAW_SHARD_SAMPLES": "Samples per raw shard file (train_record).",
    "SAMPLING_RATE": "Expected sampling rate (Hz).",
    "CHANNELS": "Number of EEG channels (read-only).",
    "TIMEBASE_VERSION": "Enforced absolute timebase (absolute_v1).",
    "WINDOW_SEC": "Window duration in seconds.",
    "N_FINGERS": "Number of finger classes.",
    "N_ACTIONS": "Number of action classes.",
    "MODEL_PATH": "Model weights path for inference.",
    "SCALER_PATH": "Scaler path for normalization.",
    "BASE_CONF_THRESH": "Base confidence threshold (0-1).",
    "UNCERTAINTY_WEIGHT": "Uncertainty weighting (0-1).",
    "STABILITY_FRAMES": "Frames required for stable prediction.",
    "MC_DROPOUT_PASSES": "MC dropout passes for uncertainty estimation.",
    "PLOT_SCALE_MODE": "Plot scaling mode for event marking (fixed or robust auto-scale).",
    "PLOT_FIXED_UV": "Fixed plot range in microvolts (± value, uV).",
    "PLOT_REFERENCE_LINES": "Show faint reference lines at ±25/±50/±100 uV in fixed mode.",
    "EVENT_MARKING_ENABLED": "Enable event stamping during streaming.",
    "EVENTS_CSV_PATH": "Events CSV output path.",
    "EVENTS_AUTOSAVE_PATH": "Autosave events CSV path.",
    "EVENTS_CHANNEL": "Event channel label.",
    "HARD_STOP_AFTER_UNHEALTHY_S": "Seconds of continuous unhealthy stream before hard stop.",
    "FAILED_WRITE_WINDOW_S": "Seconds to write failed debug files during unhealthy window.",
    "FAILED_DIR": "Directory for failed debug writes.",
    "REQUIRED_LSL_LABELS": "Required LSL channel labels (case-insensitive).",
    "REQUIRE_EXACTLY_4_CHANNELS": "Require exactly 4 EEG channels from LSL.",
    "LIVE_VIZ_ENABLED": "Emit Step 7 live model-view data for the Model Views window.",
    "LIVE_VIZ_FPS": "Step 7 live model-view update rate (Hz).",
    "STREAMER_INTERNAL": "Internal streamer enabled.",
    "STREAMER_STREAM_NAME": "Internal streamer LSL name.",
    "STREAMER_STREAM_TYPE": "Internal streamer LSL type.",
    "LABEL_CHECK_ACKNOWLEDGED": "Operator acknowledged label mismatch.",
    "model_path": "Model path for live inference.",
    "scaler_path": "Scaler path for live inference.",
    "out_dir": "Output directory override for live inference.",
    "stream_name": "LSL stream name for live inference.",
    "stream_type": "LSL stream type for live inference.",
    "hop_sec": "Window hop length in seconds.",
    "target_fs": "Target sampling rate for live inference.",
    "allow_drop": "Allow dropping windows in live inference.",
    "latency_threshold_ms": "Latency p95 threshold (ms).",
    "latency_policy": "Latency policy when threshold is exceeded.",
    "log_every": "Live inference log interval (s).",
    "enable_actuation": "Enable robot hand actuation (explicit opt-in, requires confirmation).",
    "bluetooth_target": "Bluetooth device name/address for actuation.",
    "no_file_io": "Disable file outputs during live inference (max performance).",
    "modulate_actuation_speed": "Modulate actuation speed from prediction confidence.",
    "actuation_speed_gamma": "Gamma curve for confidence-based actuation speed modulation.",
    "use_inference_engine": "Use utils.inference.InferenceEngine for MC-dropout mean probabilities and uncertainty-aware actuation gating.",
    "mc_passes": "Monte Carlo dropout passes for live inference when the inference engine backend is enabled.",
    "uncertainty_base_threshold": "Base action confidence threshold before uncertainty adjustment.",
    "uncertainty_weight": "Additional adaptive threshold weight applied to action uncertainty.",
    "infer_subject_override": "Subject override for Step 7 (defaults to current subject).",
    "project_name": "Project name for auto-resolving latest session.",
    "subject_id": "Subject ID for auto-resolving latest session.",
    "raw_dir": "Session root for raw recording.",
    "session_id": "Session ID for raw recording.",
    "finger_weights": "Per-finger loss weights (CSV or JSON). Example: 1,1,1,1,1,0.4 or {\"pinky\":0.4}",
    "loss_action_weight": "Weight applied to the finger loss term.",
    "rest_weight": "Class weight for REST actions (0 = ignore).",
    "action_weights": "Per-action loss weights in REST,OPEN,CLOSE order (CSV or JSON). Overrides rest_weight when set.",
    "rest_balance_mode": "How REST windows are reweighted across source sessions during training.",
    "rest_finger_loss_weight": "Additional finger-head loss weight applied on REST windows toward NONE.",
    "test_size": "Fraction of windows held out for testing.",
    "split_mode": "Split strategy (group_trial or holdout_session).",
    "calibration_size": "Fraction of the training split reserved for post-hoc temperature scaling (0 disables).",
    "window_preprocess": "Per-window preprocessing before channel normalization (none, center, center_detrend).",
    "purge_seconds": "Purge training windows within this many seconds of any test window.",
    "hop_seconds": "Window hop override in seconds (0 = auto).",
    "window_idx_leak_threshold": "Warn if window_idx-only classifier exceeds this accuracy.",
    "strict_leakage": "Fail training if leakage checks exceed thresholds.",
    "device": "Training device (auto/cpu/cuda/mps).",
    "num_workers": "DataLoader worker processes (0 = main process).",
    "pin_memory": "Pin DataLoader memory (useful for CUDA).",
    "save_preds": "Output path for test predictions.",
    "save_temperature": "Output path for fitted post-hoc temperature scaling parameters.",
    "run_dir": "Explicit output directory for training run.",
}

EEGLAB_STYLE = """
QMainWindow { background: rgb(220, 225, 235); }
QMenuBar { background: rgb(80, 100, 130); color: white; padding: 4px; font-weight: 600; }
QMenuBar::item:selected { background: rgb(110, 130, 170); }
QMenu { background: rgb(80, 100, 130); color: white; border: 1px solid #7a8fb8; }
QMenu::item:selected { background: rgb(110, 130, 170); }

QDockWidget::title { color: white; background: rgb(70, 90, 120); padding: 6px 8px; font-weight: 700; }
QDockWidget { font-weight: 600; }

QWidget#Sidebar { background-color: rgb(80, 100, 130); }
QWidget#CentralWorkspace { background-color: rgb(220, 225, 235); }
QWidget#BottomBar { background-color: rgb(45, 45, 45); }

QScrollArea { border: none; }
QScrollArea#Sidebar { background-color: rgb(80, 100, 130); }
QScrollArea#CentralWorkspace { background-color: rgb(220, 225, 235); }
QScrollArea#BottomBar { background-color: rgb(45, 45, 45); }
QScrollArea::viewport { background: transparent; }

QGroupBox {
  background: rgb(95, 110, 135);
  border: 1px solid #8ba0c7;
  border-radius: 6px;
  margin-top: 18px;
}
QGroupBox::title {
  subcontrol-origin: margin;
  subcontrol-position: top left;
  left: 10px;
  top: 2px;
  padding: 0 6px;
  color: white;
  font-weight: 700;
}

QToolButton#InfoButton {
  background: rgb(80, 100, 130);
  color: white;
  border: 1px solid #8ba0c7;
  border-radius: 7px;
  padding: 0px 4px;
  min-width: 14px;
  min-height: 14px;
  font-weight: 700;
}
QToolButton#InfoButton:hover { background: rgb(110, 130, 170); }
QToolButton#InfoButton:pressed { background: rgb(65, 80, 105); }

QPushButton {
  background: rgb(110, 130, 170);
  color: white;
  border-radius: 6px;
  padding: 7px 10px;
  font-weight: 600;
}
QPushButton:hover { background: rgb(130, 150, 190); }
QPushButton:disabled { background: rgb(105, 120, 150); color: rgba(255,255,255,180); }

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
  background: rgb(65, 80, 105);
  color: white;
  border: 1px solid #7a8fb8;
  border-radius: 6px;
  padding: 4px 6px;
}
QAbstractItemView, QListWidget {
  background: rgb(65, 80, 105);
  color: white;
  border: 1px solid #7a8fb8;
}

QLabel { color: white; font-weight: 600; }
QCheckBox, QRadioButton { color: white; font-weight: 600; }
QToolTip { background-color: rgb(80, 100, 130); color: white; border: 1px solid #7a8fb8; }
QTabBar::tab { background: rgb(80, 100, 130); color: white; padding: 6px 10px; }
QTabBar::tab:selected { background: rgb(110, 130, 170); }
"""

_BATT_RE = re.compile(r"(?:BATTERY|Battery)\s*[:=]\s*(\d{1,3})\s*%?", re.IGNORECASE)


class OutlineStyle(QProxyStyle):
    def drawItemText(self, painter, rect, flags, pal, enabled, text, textRole):
        if not text:
            return super().drawItemText(
                painter, rect, flags, pal, enabled, text, textRole
            )
        painter.save()
        painter.setPen(Qt.black)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                painter.drawText(rect.translated(dx, dy), flags, text)
        painter.restore()
        super().drawItemText(painter, rect, flags, pal, enabled, text, textRole)


class OutlineItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index) -> None:
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        text = opt.text
        opt.text = ""
        style = opt.widget.style() if opt.widget else QApplication.style()
        style.drawControl(QStyle.CE_ItemViewItem, opt, painter, opt.widget)
        if not text:
            return
        text_rect = style.subElementRect(QStyle.SE_ItemViewItemText, opt, opt.widget)
        text = opt.fontMetrics.elidedText(text, opt.textElideMode, text_rect.width())
        align = opt.displayAlignment | Qt.TextSingleLine
        if not (align & Qt.AlignVertical_Mask):
            align |= Qt.AlignVCenter
        text_color = opt.palette.color(
            QPalette.HighlightedText if opt.state & QStyle.State_Selected else QPalette.Text
        )
        painter.save()
        painter.setFont(opt.font)
        painter.setPen(Qt.black)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                painter.drawText(text_rect.translated(dx, dy), align, text)
        painter.setPen(text_color)
        painter.drawText(text_rect, align, text)
        painter.restore()


class OutlineTextHighlighter(QSyntaxHighlighter):
    def __init__(self, document) -> None:
        super().__init__(document)
        self._outline_pen = QPen(Qt.black, 1)

    def highlightBlock(self, text: str) -> None:
        if not text:
            return
        fmt = QTextCharFormat()
        fmt.setTextOutline(self._outline_pen)
        self.setFormat(0, len(text), fmt)


class OutlineLineEdit(QLineEdit):
    """Line edit with *no* custom painting.

    We previously attempted to draw an outline around the editable text to
    improve contrast. On macOS (and some retina/HiDPI configurations) this can
    produce visible "ghost" glyphs/overdraw in the central forms.

    Keeping this subclass preserves drop-in compatibility with the rest of the
    file, while reverting to Qt's native text rendering for clean, crisp inputs.
    """

    # Intentionally do not override paintEvent.
    pass


class OutlinedLabel(QLabel):
    """QLabel that paints white text with a thin black outline.

    This avoids global style overrides that can cause duplicated/overlapping
    text on some platforms.
    """

    def __init__(
        self,
        text: str = "",
        parent: Optional[QWidget] = None,
        *,
        outline_width: float = 1.1,
        outline_color: QColor = QColor(0, 0, 0, 255),
        fill_color: QColor = QColor(255, 255, 255, 255),
    ) -> None:
        super().__init__(text, parent)
        self._outline_width = float(outline_width)
        self._outline_color = QColor(outline_color)
        self._fill_color = QColor(fill_color)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAutoFillBackground(False)

    def paintEvent(self, event) -> None:
        # Let Qt handle background/frames.
        super().paintEvent(event)
        text = self.text()
        if not text:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        painter.setFont(self.font())

        rect = self.contentsRect()
        flags = int(self.alignment())
        if not (flags & int(Qt.AlignVertical_Mask)):
            flags |= int(Qt.AlignVCenter)
        flags |= int(Qt.TextSingleLine)

        fm = QFontMetrics(self.font())
        elided = fm.elidedText(text, Qt.ElideRight, rect.width())

        # Compute a baseline position inside rect for path drawing.
        # We use boundingRect to position the text, then convert to baseline.
        br = fm.boundingRect(rect, flags, elided)
        x = br.left()
        y = br.top() + fm.ascent()

        path = QPainterPath()
        path.addText(x, y, self.font(), elided)

        # Stroke outline, then fill.
        painter.setPen(QPen(self._outline_color, self._outline_width))
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self._fill_color)
        painter.drawPath(path)


class OutlinePlainTextEdit(QPlainTextEdit):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._outline_highlighter = OutlineTextHighlighter(self.document())


class OutlineTextEdit(QTextEdit):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._outline_highlighter = OutlineTextHighlighter(self.document())


class OutlineSpinBox(QSpinBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setLineEdit(OutlineLineEdit(self))
        self.lineEdit().setAlignment(self.alignment())


class OutlineDoubleSpinBox(QDoubleSpinBox):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setLineEdit(OutlineLineEdit(self))
        self.lineEdit().setAlignment(self.alignment())


class FloatSlider(QWidget):
    def __init__(
        self,
        min_val: float,
        max_val: float,
        value: float,
        decimals: int = 2,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._factor = 10**decimals
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(min_val * self._factor), int(max_val * self._factor))
        self.spin = OutlineDoubleSpinBox()
        self.spin.setRange(min_val, max_val)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(0.01)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.slider)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._sync_from_slider)
        self.spin.valueChanged.connect(self._sync_from_spin)
        self.setValue(value)

    def _sync_from_slider(self, raw: int) -> None:
        self.spin.blockSignals(True)
        self.spin.setValue(raw / self._factor)
        self.spin.blockSignals(False)

    def _sync_from_spin(self, val: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(val * self._factor)))
        self.slider.blockSignals(False)

    def value(self) -> float:
        return float(self.spin.value())

    def setValue(self, val: float) -> None:
        self.spin.setValue(float(val))
        self.slider.setValue(int(round(float(val) * self._factor)))


class SubjectDialog(QDialog):
    def __init__(self, parent: QWidget, info: Optional[SubjectInfo] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Subject")
        self._info = info or SubjectInfo(subject_id="")

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.subject_id = OutlineLineEdit(self._info.subject_id)
        self.handedness = QComboBox()
        self.handedness.addItems(["Right", "Left", "Ambidextrous", "Unknown"])
        if self._info.handedness:
            idx = self.handedness.findText(self._info.handedness)
            if idx >= 0:
                self.handedness.setCurrentIndex(idx)
        self.age = OutlineSpinBox()
        self.age.setRange(0, 120)
        if self._info.age is not None:
            self.age.setValue(int(self._info.age))
        else:
            self.age.setValue(0)
        self.notes = OutlineTextEdit(self._info.notes)
        self.notes.setFixedHeight(100)

        form.addRow("Subject ID*", self.subject_id)
        form.addRow("Handedness", self.handedness)
        form.addRow("Age", self.age)
        form.addRow("Notes", self.notes)
        layout.addLayout(form)

        button_row = QHBoxLayout()
        save_btn = QPushButton("Save")
        cancel_btn = QPushButton("Cancel")
        save_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(save_btn)
        button_row.addWidget(cancel_btn)
        layout.addLayout(button_row)

    def info(self) -> SubjectInfo:
        age_val = int(self.age.value())
        return SubjectInfo(
            subject_id=self.subject_id.text().strip(),
            handedness=self.handedness.currentText(),
            age=age_val if age_val > 0 else None,
            notes=self.notes.toPlainText().strip(),
        )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("EEGLAB Wrapper UI")
        self.resize(1200, 780)
        self.setMinimumSize(1400, 900)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet(EEGLAB_STYLE)

        self.repo_root = Path(__file__).resolve().parent
        self.scripts = discover_scripts(self.repo_root)

        self.current_project: Optional[str] = None
        self.current_subject: Optional[str] = None
        self.current_session_ui: Optional[str] = None
        self.current_session_backend: Optional[str] = None
        self._auto_session_dir_value: Optional[str] = None
        self._auto_field_values: Dict[str, str] = {}

        self.fields: Dict[str, Dict[str, QWidget]] = {}
        self.defaults: Dict[str, Dict[str, Any]] = {}
        self.step_status: Dict[str, QLabel] = {}
        self.step_checklists: Dict[str, QListWidget] = {}
        self.step_script_key: Dict[str, str] = {}
        self.active_settings: Dict[str, Any] = {}
        self.step_arg_specs = self._build_step_arg_specs()
        self.step_arg_widgets: Dict[str, Dict[str, QWidget]] = {}
        self.step_arg_includes: Dict[str, Dict[str, QCheckBox]] = {}
        self.eval_fields: Dict[str, Dict[str, QWidget]] = {}

        self.runner = ProcessRunner(self)
        self.runner.line_ready.connect(self._append_log)
        self.runner.started.connect(self._on_process_started)
        self.runner.finished.connect(self._on_process_finished)
        self.runner.failed.connect(self._append_log)
        self.active_step: Optional[str] = None

        self.muse_connector = MuseConnectorController(self)
        self.muse_connector.log_line.connect(self._on_connector_log)
        self.muse_connector.status_changed.connect(self._on_connector_status)
        self.muse_connector.device_changed.connect(self._on_connector_device)
        self.muse_connector.stream_changed.connect(self._on_connector_stream)
        self.muse_connector.process_exited.connect(self._on_connector_finished)
        self.muse_connector.error.connect(
            lambda msg: self._append_log(f"[connector] {msg}")
        )

        self.live_stream_ready = False
        self.live_label_acknowledged = False
        self.live_label_details: Dict[str, Any] = {}
        self.live_stream_name = DEFAULT_STREAM_NAME
        self.live_stream_type = DEFAULT_STREAM_TYPE
        self.live_lsl_source_id: Optional[str] = None
        self.hard_stop_locked = False
        self._legacy_warnings: set[str] = set()
        self._auto_scan_active = False
        self._auto_scan_wants_healthcheck = False
        self._healthcheck_pending = False
        self._auto_scan_timer = QTimer(self)
        self._auto_scan_timer.setInterval(1500)
        self._auto_scan_timer.timeout.connect(self._auto_scan_lsl_streams)

        self.live_feature_view = None
        self.live_hidden_plot = None
        self.live_saliency_view = None
        self.live_pred_finger_plot = None
        self.live_pred_action_plot = None
        self.live_pred_label: Optional[QLabel] = None
        self.live_viz_tab_index: Optional[int] = None
        self.live_viz_status_label: Optional[QLabel] = None
        self._latest_live_viz_payload: Optional[Dict[str, Any]] = None
        self._last_live_viz_mono = 0.0
        self._live_viz_status_timer = QTimer(self)
        self._live_viz_status_timer.setInterval(750)
        self._live_viz_status_timer.timeout.connect(self._update_live_viz_status)
        self._live_viz_status_timer.start()
        self.replay_viz: Optional[ReplayVisualizer] = None
        self.replay_pred_finger_plot = None
        self.replay_pred_action_plot = None
        self.replay_auto_checkbox: Optional[QCheckBox] = None
        self.replay_auto_interval: Optional[QSpinBox] = None
        self.replay_pred_label: Optional[QLabel] = None
        self._replay_auto_timer = QTimer(self)
        self._replay_auto_timer.setInterval(500)
        self._replay_auto_timer.timeout.connect(self._advance_replay_window)
        self.model_views_window: Optional[QDialog] = None
        self._model_views_root: Optional[QWidget] = None
        self.log_entries: list[str] = []
        self._stop_requested = False
        self._stop_waiting_runner = False
        self._stop_waiting_connector = False
        self._stop_step_id: Optional[str] = None
        self._eval_queue: list[str] = []
        self._eval_queue_active = False

        self._build_ui()

    def _build_ui(self) -> None:
        self._build_menu()

        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(8, 8, 8, 8)

        status_bar = self._build_status_bar()
        main_layout.addWidget(status_bar)

        splitter = QSplitter()
        splitter.setOrientation(Qt.Horizontal)

        self.workflow_list = QListWidget()
        self.workflow_list.setItemDelegate(OutlineItemDelegate(self.workflow_list))
        self.workflow_list.setObjectName("Sidebar")
        self.workflow_list.setMinimumWidth(220)
        self.workflow_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.workflow_list.setStyleSheet(
            "QListWidget { background: rgb(80, 100, 130); color: white; font-weight: 700; } "
            "QListWidget::item:selected { background: rgb(110,130,170); }"
        )
        for item in [
            "Pipeline Overview",
            "1) Record (Lossless)",
            "Events: Mark/Edit (Optional)",
            "Validate Session (Tool)",
            "1b) Extract Windows",
            "2) Train Model",
            "3+) Evaluate / Reports",
            "7) Live Infer + Actuate",
            "Logs & Diagnostics",
            "Projects",
            "Stream Setup",
            "Export",
        ]:
            QListWidgetItem(item, self.workflow_list)
        self.workflow_list.currentRowChanged.connect(self._switch_page)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._wrap_scroll(self._build_pipeline_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_step1_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_event_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_session_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_step1b_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_train_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_evaluate_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_infer_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_logs_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_projects_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_stream_page(), "CentralWorkspace"))
        self.stack.addWidget(self._wrap_scroll(self._build_export_page(), "CentralWorkspace"))
        self.stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        splitter.addWidget(self.workflow_list)
        splitter.addWidget(self.stack)
        splitter.setSizes([200, 900])

        main_layout.addWidget(splitter, 1)
        self.setCentralWidget(main)

        self._build_log_dock()
        self._build_control_docks()
        self._wire_status_updates()
        self._refresh_status_summary()
        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self._refresh_health_indicator)
        self.health_timer.start(1000)
        self._set_live_buttons_state()
        self._set_connector_stream(self.live_stream_name)
        self.workflow_list.setCurrentRow(0)

    def _build_step_arg_specs(self) -> Dict[str, list[ArgSpec]]:
        return {
            "step1": [
                ArgSpec("subject_id", "--subject-id", "text", "Override subject ID."),
                ArgSpec("init_only", "--init-only", "bool", "Initialize session then exit."),
                ArgSpec(
                    "force_new_session",
                    "--force-new-session",
                    "bool",
                    "Force a new session (ignore resume).",
                ),
            ],
            "infer": [
                ArgSpec("model_path", "--model-path", "text", "Model path."),
                ArgSpec("scaler_path", "--scaler-path", "text", "Scaler path."),
                ArgSpec("out_dir", "--out-dir", "text", "Output directory override."),
                ArgSpec("stream_name", "--stream-name", "text", "LSL stream name."),
                ArgSpec("stream_type", "--stream-type", "text", "LSL stream type."),
                ArgSpec("window_sec", "--window-sec", "float", "Window length (s)."),
                ArgSpec("hop_sec", "--hop-sec", "float", "Window hop (s)."),
                ArgSpec("target_fs", "--target-fs", "float", "Target FS for resampling."),
                ArgSpec(
                    "use_inference_engine",
                    "--use-inference-engine",
                    "bool",
                    "Use MC-dropout inference backend.",
                ),
                ArgSpec(
                    "mc_passes",
                    "--mc-passes",
                    "int",
                    "Monte Carlo dropout passes.",
                ),
                ArgSpec(
                    "uncertainty_base_threshold",
                    "--uncertainty-base-threshold",
                    "float",
                    "Base action threshold for uncertainty gating.",
                ),
                ArgSpec(
                    "uncertainty_weight",
                    "--uncertainty-weight",
                    "float",
                    "Adaptive threshold weight for action uncertainty.",
                ),
                ArgSpec("allow_drop", "--allow-drop", "bool", "Allow dropping windows."),
                ArgSpec(
                    "latency_threshold_ms",
                    "--latency-threshold-ms",
                    "float",
                    "Latency p95 threshold (ms).",
                ),
                ArgSpec(
                    "latency_policy",
                    "--latency-policy",
                    "text",
                    "Latency policy (warn/drop/degrade).",
                ),
                ArgSpec("log_every", "--log-every", "float", "Log interval (s)."),
                ArgSpec(
                    "enable_actuation",
                    "--enable-actuation",
                    "bool",
                    "Enable robot hand actuation.",
                ),
                ArgSpec(
                    "modulate_actuation_speed",
                    "--modulate-actuation-speed",
                    "bool",
                    "Modulate actuation speed from prediction confidence.",
                ),
                ArgSpec(
                    "actuation_speed_gamma",
                    "--actuation-speed-gamma",
                    "float",
                    "Gamma for confidence-based actuation speed.",
                ),
                ArgSpec(
                    "bluetooth_target",
                    "--bluetooth-target",
                    "text",
                    "Bluetooth target name/address.",
                ),
                ArgSpec("no_file_io", "--no_file_io", "bool", "Disable file outputs."),
                ArgSpec("subject_id", "--subject-id", "text", "Subject ID (auto-resolve latest session)."),
                ArgSpec("project_name", "--project-name", "text", "Project name (auto-resolve latest session)."),
            ],
            "step1b": [
                ArgSpec(
                    "session_dir",
                    "--session-dir",
                    "text",
                    "Session Directory (sessions/<session_id>).",
                ),
                ArgSpec(
                    "features",
                    "--features",
                    "text",
                    "Legacy features CSV path (optional).",
                ),
                ArgSpec(
                    "events",
                    "--events",
                    "text",
                    "Legacy events CSV path (optional).",
                ),
                ArgSpec("subject_id", "--subject-id", "text", "Subject ID override."),
                ArgSpec("target_fs", "--target-fs", "float", "Target resample rate."),
                ArgSpec("allow_gaps", "--allow-gaps", "bool", "Allow gaps in windows."),
                ArgSpec(
                    "allow_partial",
                    "--allow-partial",
                    "bool",
                    "Allow partial sessions (skip strict manifest validation).",
                ),
                ArgSpec(
                    "ignore_misalignment",
                    "--ignore-misalignment",
                    "bool",
                    "Continue if events are out of range.",
                ),
                ArgSpec("seed", "--seed", "int", "Seed for REST subsampling."),
            ],
            "train": [
                ArgSpec(
                    "session_dir",
                    "--session-dir",
                    "text",
                    "Session directory containing processed windows.",
                ),
                ArgSpec(
                    "run_dir",
                    "--run-dir",
                    "text",
                    "Explicit output directory for this run.",
                ),
                ArgSpec("npz", "--npz", "text", "Window dataset path."),
                ArgSpec("subject_id", "--subject-id", "text", "Filter by subject ID."),
                ArgSpec("epochs", "--epochs", "int", "Training epochs."),
                ArgSpec("batch_size", "--batch-size", "int", "Training batch size."),
                ArgSpec("lr", "--lr", "float", "Learning rate."),
                ArgSpec("seed", "--seed", "int", "Random seed."),
                ArgSpec("device", "--device", "text", "Training device (auto/cpu/cuda/mps)."),
                ArgSpec("num_workers", "--num-workers", "int", "DataLoader workers."),
                ArgSpec("pin_memory", "--pin-memory", "bool", "Pin DataLoader memory."),
                ArgSpec(
                    "loss_action_weight",
                    "--loss-action-weight",
                    "float",
                    "Action loss weight.",
                ),
                ArgSpec("rest_weight", "--rest-weight", "float", "REST class weight."),
                ArgSpec(
                    "action_weights",
                    "--action-weights",
                    "text",
                    "Per-action loss weights (REST,OPEN,CLOSE; CSV/JSON).",
                ),
                ArgSpec(
                    "rest_balance_mode",
                    "--rest-balance-mode",
                    "text",
                    "REST reweighting mode (none, session_equalized, core_event_equalized).",
                ),
                ArgSpec(
                    "rest_finger_loss_weight",
                    "--rest-finger-loss-weight",
                    "float",
                    "Additional finger loss on REST windows toward NONE.",
                ),
                ArgSpec(
                    "finger_weights",
                    "--finger-weights",
                    "text",
                    "Per-finger loss weights (CSV/JSON).",
                ),
                ArgSpec(
                    "window_preprocess",
                    "--window-preprocess",
                    "text",
                    "Window preprocessing (none, center, center_detrend).",
                ),
                ArgSpec("test_size", "--test-size", "float", "Test split fraction."),
                ArgSpec(
                    "split_mode",
                    "--split-mode",
                    "text",
                    "Split strategy (group_trial or holdout_session).",
                ),
                ArgSpec(
                    "purge_seconds",
                    "--purge-seconds",
                    "float",
                    "Purge train windows near test windows (s).",
                ),
                ArgSpec(
                    "hop_seconds",
                    "--hop-seconds",
                    "float",
                    "Window hop override (s).",
                ),
                ArgSpec("non_rest_only", "--non-rest-only", "bool", "Train on non-REST only."),
                ArgSpec(
                    "window_idx_leak_threshold",
                    "--window-idx-leak-threshold",
                    "float",
                    "Leakage warning threshold.",
                ),
                ArgSpec(
                    "strict_leakage",
                    "--strict-leakage",
                    "bool",
                    "Fail training on leakage checks.",
                ),
                ArgSpec("save_model", "--save-model", "text", "Model output path."),
                ArgSpec("save_scaler", "--save-scaler", "text", "Scaler output path."),
                ArgSpec("save_preds", "--save-preds", "text", "Predictions output path."),
            ],
        }

    def _wrap_scroll(self, widget: QWidget, object_name: Optional[str] = None) -> QWidget:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        if object_name:
            area.setObjectName(object_name)
            area.viewport().setObjectName(object_name)
            widget.setObjectName(object_name)
        area.setWidget(widget)
        return area

    def _apply_text_outline_effect(
        self, widget: QWidget, *, radius: float = 1.5, dx: int = 1, dy: int = 1
    ) -> None:
        effect = QGraphicsDropShadowEffect(widget)
        effect.setBlurRadius(radius)
        effect.setOffset(dx, dy)
        effect.setColor(QColor(0, 0, 0, 230))
        widget.setGraphicsEffect(effect)

    def _set_status_semantic(self, label: QLabel, state: str, text: str) -> None:
        label.setText(text)
        if state == "green":
            label.setStyleSheet("color: rgb(130, 255, 130); font-weight: 800;")
        elif state == "yellow":
            label.setStyleSheet("color: rgb(255, 230, 120); font-weight: 800;")
        elif state == "red":
            label.setStyleSheet("color: rgb(255, 120, 120); font-weight: 900;")
        else:
            label.setStyleSheet("color: white; font-weight: 700;")

    def _build_dropdown_button(self, title: str, panel: QWidget) -> QToolButton:
        button = QToolButton()
        button.setText(title)
        button.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(button)
        menu.setMinimumWidth(360)
        action = QWidgetAction(menu)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(340)
        scroll.setMinimumHeight(260)
        scroll.setWidget(panel)
        action.setDefaultWidget(scroll)
        menu.addAction(action)
        button.setMenu(menu)
        return button

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction(
            "Projects", lambda: self.workflow_list.setCurrentRow(9)
        )
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        edit_menu = menu.addMenu("Edit")
        edit_menu.addAction(
            "Validate Session", lambda: self.workflow_list.setCurrentRow(3)
        )

        tools_menu = menu.addMenu("Tools")
        tools_menu.addAction(
            "Stream Setup", lambda: self.workflow_list.setCurrentRow(10)
        )
        tools_menu.addAction(
            "Validate Session", lambda: self.workflow_list.setCurrentRow(3)
        )
        tools_menu.addAction("Event Review", lambda: self.workflow_list.setCurrentRow(2))
        tools_menu.addAction("Diagnostics", lambda: self.workflow_list.setCurrentRow(8))
        model_views_action = tools_menu.addAction("Model Views", self._open_model_views_window)
        model_views_action.setShortcut("Ctrl+M")

        plot_menu = menu.addMenu("Plot")
        plot_menu.addAction(
            "Record (Lossless)", lambda: self.workflow_list.setCurrentRow(1)
        )

        study_menu = menu.addMenu("Study")
        study_menu.addAction("Windowing", lambda: self.workflow_list.setCurrentRow(4))
        study_menu.addAction("Training", lambda: self.workflow_list.setCurrentRow(5))
        study_menu.addAction("Evaluate", lambda: self.workflow_list.setCurrentRow(6))

        datasets_menu = menu.addMenu("Datasets")
        datasets_menu.addAction("Export", lambda: self.workflow_list.setCurrentRow(11))

        run_menu = menu.addMenu("Run")
        run_menu.addAction(
            "Run Record (Lossless)", lambda: self._run_step("step1", "step1")
        )
        run_menu.addAction(
            "Run Extract Windows", lambda: self._run_step("step1b", "step1b")
        )
        run_menu.addAction("Run Train Model", lambda: self._run_step("train", "train"))
        run_menu.addAction("Run Evaluate", self._run_evaluate_all)
        run_menu.addAction(
            "Run Live Infer + Actuate", lambda: self._run_step("infer", "live_infer")
        )

        help_menu = menu.addMenu("Help")
        help_menu.addAction("Logs", lambda: self.workflow_list.setCurrentRow(8))
        help_menu.addAction("Open README.md", lambda: self._open_doc("README.md"))
        help_menu.addAction(
            "Open DATA_CONTRACT.md",
            lambda: self._open_doc("docs/spec/DATA_CONTRACT.md"),
        )

    def _build_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("StatusBarFrame")
        bar.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(14)

        # Use outlined labels (painted inside their own rect) to avoid the
        # QGraphicsEffect halo overlapping adjacent widgets.
        self.project_label = OutlinedLabel("Project: -")
        self.subject_label = OutlinedLabel("Subject: -")
        self.session_label = OutlinedLabel("Session: -")
        self.stream_state_label = OutlinedLabel("Stream: idle")
        self.ica_state_label = OutlinedLabel("ICA: off")
        self.events_state_label = OutlinedLabel("Events: off")
        self.battery_label = OutlinedLabel("Battery: N/A")

        for label in (
            self.project_label,
            self.subject_label,
            self.session_label,
            self.stream_state_label,
            self.ica_state_label,
            self.events_state_label,
            self.battery_label,
        ):
            label.setWordWrap(False)
            label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            label.setMinimumWidth(0)

        def _sep() -> QFrame:
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            line.setStyleSheet("color: rgba(0,0,0,120);")
            return line

        layout.addWidget(self.project_label)
        layout.addWidget(_sep())
        layout.addWidget(self.subject_label)
        layout.addWidget(_sep())
        layout.addWidget(self.session_label)
        layout.addWidget(_sep())
        layout.addWidget(self.stream_state_label)
        layout.addWidget(_sep())
        layout.addWidget(self.ica_state_label)
        layout.addWidget(_sep())
        layout.addWidget(self.events_state_label)
        layout.addStretch(1)
        layout.addWidget(self.battery_label)

        return bar

    def _build_log_dock(self) -> None:
        self.log_console = OutlinePlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(10000)
        container = QWidget()
        container.setObjectName("BottomBar")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)
        filter_row = QHBoxLayout()
        filter_label = QLabel("Filter")
        self.log_filter_combo = QComboBox()
        self.log_filter_combo.addItems(["All", "Warnings", "Errors"])
        self.log_filter_combo.currentTextChanged.connect(self._refresh_log_display)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_logs)
        filter_row.addWidget(filter_label)
        filter_row.addWidget(self.log_filter_combo)
        filter_row.addStretch(1)
        filter_row.addWidget(clear_btn)
        layout.addLayout(filter_row)
        self.hard_stop_banner = QLabel("HARD STOP — Stream Unhealthy")
        self.hard_stop_banner.setStyleSheet(
            "background-color: #b71c1c; color: white; font-weight: 700; padding: 6px;"
        )
        self.hard_stop_banner.setVisible(False)
        layout.addWidget(self.hard_stop_banner)
        layout.addWidget(self.log_console)
        dock = QDockWidget("Log Console", self)
        dock.setWidget(self._wrap_scroll(container, "BottomBar"))
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _build_control_docks(self) -> None:
        self.stream_status_dock = QLabel("Stream status: idle")
        stream_widget = QWidget()
        stream_widget.setObjectName("Sidebar")
        stream_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        stream_layout = QVBoxLayout(stream_widget)
        stream_layout.addWidget(self.stream_status_dock)
        self.health_indicator = QLabel("Health: unknown")
        stream_layout.addWidget(self.health_indicator)
        connector_header = QLabel("Muse Connector")
        self._apply_text_outline_effect(connector_header)
        stream_layout.addWidget(connector_header)
        self.connector_status_dock = QLabel("Connector: idle")
        self.connector_device_dock = QLabel("Muse device: -")
        self.connector_stream_dock = QLabel("LSL stream: -")
        self.connector_log_dock = QLabel("Last connector log: -")
        self.connector_log_dock.setWordWrap(True)
        for label in (
            self.connector_status_dock,
            self.connector_device_dock,
            self.connector_stream_dock,
            self.connector_log_dock,
        ):
            self._apply_text_outline_effect(label)
            stream_layout.addWidget(label)
        self.live_connect_btn = QPushButton("Connect Muse (BLE → LSL)")
        self.live_connect_btn.clicked.connect(self._connect_muse)
        self.live_disconnect_btn = QPushButton("Disconnect Muse")
        self.live_disconnect_btn.clicked.connect(self._disconnect_muse)

        # Keep Stream Control buttons from stretching vertically when dock content changes.
        for _btn in (self.live_connect_btn, self.live_disconnect_btn):
            _btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            _btn.setMinimumHeight(34)
            _btn.setMaximumHeight(34)
        stream_layout.addWidget(self.live_connect_btn)
        stream_layout.addWidget(self.live_disconnect_btn)
        # Quick Actions removed (use Stream Setup page for diagnostics)
        self._apply_text_outline_effect(self.stream_status_dock)
        self._apply_text_outline_effect(self.health_indicator)
        stream_layout.addStretch(1)
        stream_dock = QDockWidget("Stream Control", self)
        stream_dock.setWidget(self._wrap_scroll(stream_widget, "Sidebar"))
        stream_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        stream_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.LeftDockWidgetArea, stream_dock)

        pipeline_widget = QWidget()
        pipeline_widget.setObjectName("Sidebar")
        pipeline_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        pipeline_layout = QVBoxLayout(pipeline_widget)
        pipeline_header = QLabel("Pipeline Controls")
        self._apply_text_outline_effect(pipeline_header)
        pipeline_layout.addWidget(pipeline_header)
        run_step1_btn = QPushButton("Run Record (Lossless)")
        run_step1_btn.clicked.connect(lambda: self._run_step("step1", "step1"))
        pipeline_layout.addWidget(run_step1_btn)
        open_events_btn = QPushButton("Open Events: Mark/Edit (Optional)")
        open_events_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(2))
        pipeline_layout.addWidget(open_events_btn)
        validate_btn = QPushButton("Validate Session")
        validate_btn.clicked.connect(self._run_validate_session)
        pipeline_layout.addWidget(validate_btn)
        run_step1b_btn = QPushButton("Run Extract Windows")
        run_step1b_btn.clicked.connect(lambda: self._run_step("step1b", "step1b"))
        pipeline_layout.addWidget(run_step1b_btn)
        run_train_btn = QPushButton("Run Train Model")
        run_train_btn.clicked.connect(lambda: self._run_step("train", "train"))
        pipeline_layout.addWidget(run_train_btn)
        run_eval_btn = QPushButton("Run Evaluate")
        run_eval_btn.clicked.connect(self._run_evaluate_all)
        pipeline_layout.addWidget(run_eval_btn)
        run_infer_btn = QPushButton("Run Live Infer + Actuate")
        run_infer_btn.clicked.connect(lambda: self._run_step("infer", "live_infer"))
        pipeline_layout.addWidget(run_infer_btn)
        self.dry_run_checkbox = QCheckBox("Dry run (print CLI only)")
        pipeline_layout.addWidget(self.dry_run_checkbox)
        stop_btn = QPushButton("Stop Active Run")
        stop_btn.clicked.connect(self._stop_process)
        pipeline_layout.addWidget(stop_btn)
        pipeline_layout.addStretch(1)
        pipeline_dock = QDockWidget("Pipeline", self)
        pipeline_dock.setWidget(self._wrap_scroll(pipeline_widget, "Sidebar"))
        pipeline_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        pipeline_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.LeftDockWidgetArea, pipeline_dock)

        event_widget = QWidget()
        event_widget.setObjectName("Sidebar")
        event_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        event_layout = QVBoxLayout(event_widget)
        event_header = QLabel("Event Marking")
        self._apply_text_outline_effect(event_header)
        event_layout.addWidget(event_header)
        self.event_toggle_dock = QCheckBox("Enable event marking")
        self._bind_checkbox(self.event_toggle_dock, "step1", "EVENT_MARKING_ENABLED")
        event_layout.addWidget(self.event_toggle_dock)
        event_review_btn = QPushButton("Event Review")
        event_review_btn.clicked.connect(self._run_event_review)
        event_layout.addWidget(event_review_btn)
        event_validate_btn = QPushButton("Event Validate")
        event_validate_btn.clicked.connect(self._run_event_validate)
        event_layout.addWidget(event_validate_btn)
        open_event_btn = QPushButton("Open Event Tools")
        open_event_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(2))
        event_layout.addWidget(open_event_btn)
        event_layout.addStretch(1)
        event_dock = QDockWidget("Events", self)
        event_dock.setWidget(self._wrap_scroll(event_widget, "Sidebar"))
        event_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        event_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.RightDockWidgetArea, event_dock)

        model_widget = QWidget()
        model_widget.setObjectName("Sidebar")
        model_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        model_layout = QVBoxLayout(model_widget)
        model_header = QLabel("Model & Preprocess")
        self._apply_text_outline_effect(model_header)
        model_layout.addWidget(model_header)
        self.ica_toggle_dock = QCheckBox("Enable ICA")
        self._bind_checkbox(self.ica_toggle_dock, "step1", "ENABLE_ICA")
        model_layout.addWidget(self.ica_toggle_dock)
        self.plot_toggle_dock = QCheckBox("Enable plot")
        self._bind_checkbox(self.plot_toggle_dock, "step1", "ENABLE_PLOT")
        model_layout.addWidget(self.plot_toggle_dock)
        open_infer_btn = QPushButton("Open Live Inference")
        open_infer_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(7))
        model_layout.addWidget(open_infer_btn)
        open_model_views_btn = QPushButton("Open Model Views")
        open_model_views_btn.clicked.connect(self._open_model_views_window)
        model_layout.addWidget(open_model_views_btn)
        model_layout.addStretch(1)
        model_dock = QDockWidget("Model", self)
        model_dock.setWidget(self._wrap_scroll(model_widget, "Sidebar"))
        model_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        model_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.RightDockWidgetArea, model_dock)

        session_widget = QWidget()
        session_widget.setObjectName("Sidebar")
        session_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        session_layout = QVBoxLayout(session_widget)
        session_header = QLabel("Session Overview")
        self._apply_text_outline_effect(session_header)
        session_layout.addWidget(session_header)
        self.project_label_dock = QLabel("Project: -")
        self.subject_label_dock = QLabel("Subject: -")
        self.session_label_dock = QLabel("Session: -")
        for label in (
            self.project_label_dock,
            self.subject_label_dock,
            self.session_label_dock,
        ):
            label.setWordWrap(False)
            self._apply_text_outline_effect(label)
        session_layout.addWidget(self.project_label_dock)
        session_layout.addWidget(self.subject_label_dock)
        session_layout.addWidget(self.session_label_dock)
        projects_btn = QPushButton("Projects")
        projects_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(9))
        session_layout.addWidget(projects_btn)
        session_layout.addStretch(1)
        session_dock = QDockWidget("Session", self)
        session_dock.setWidget(self._wrap_scroll(session_widget, "Sidebar"))
        session_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        session_dock.setMinimumWidth(260)
        self.addDockWidget(Qt.RightDockWidgetArea, session_dock)

        self.resizeDocks(
            [event_dock, model_dock, session_dock],
            [320, 320, 320],
            Qt.Vertical,
        )

    def _build_model_views_widget(self) -> QWidget:
        widget = QWidget()
        widget.setObjectName("CentralWorkspace")
        widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)

        model_views_header = QLabel("Model Views")
        self._apply_text_outline_effect(model_views_header)
        layout.addWidget(model_views_header)

        self.live_viz_checkbox = QCheckBox("Emit Step 7 live model views")
        layout.addWidget(self.live_viz_checkbox)
        self.live_viz_fps_spin = QSpinBox()
        self.live_viz_fps_spin.setRange(1, 10)
        self.live_viz_fps_spin.setValue(2)
        fps_row = QHBoxLayout()
        fps_label = QLabel("Step 7 live viz FPS")
        self._apply_text_outline_effect(fps_label)
        fps_row.addWidget(fps_label)
        fps_row.addWidget(self.live_viz_fps_spin)
        fps_row.addStretch(1)
        layout.addLayout(fps_row)
        self.live_viz_status_label = QLabel("Live Model View: Step 7 inactive")
        self._apply_text_outline_effect(self.live_viz_status_label)
        layout.addWidget(self.live_viz_status_label)

        if not PYQTGRAPH_AVAILABLE:
            pg_note = QLabel("pyqtgraph not available; model visualizations disabled.")
            self._apply_text_outline_effect(pg_note)
            layout.addWidget(pg_note)
            return widget

        self.model_view_tabs = QTabWidget()

        replay_tab = QWidget()
        replay_layout = QVBoxLayout(replay_tab)
        replay_form = QFormLayout()
        self.replay_npz_path = OutlineLineEdit()
        self.replay_model_path = OutlineLineEdit()
        self.replay_scaler_path = OutlineLineEdit()
        replay_form.addRow("Windows NPZ", self.replay_npz_path)
        replay_form.addRow("Model Path", self.replay_model_path)
        replay_form.addRow("Scaler Path", self.replay_scaler_path)
        replay_layout.addLayout(replay_form)
        self.replay_window_index = QSpinBox()
        self.replay_window_index.setRange(0, 1000000)
        self.replay_layer_index = QSpinBox()
        self.replay_layer_index.setRange(0, 10)
        replay_controls = QHBoxLayout()
        replay_controls.addWidget(QLabel("Window idx"))
        replay_controls.addWidget(self.replay_window_index)
        replay_controls.addWidget(QLabel("Conv layer idx"))
        replay_controls.addWidget(self.replay_layer_index)
        replay_controls.addStretch(1)
        replay_layout.addLayout(replay_controls)

        replay_auto_row = QHBoxLayout()
        self.replay_auto_checkbox = QCheckBox("Auto-advance window")
        self.replay_auto_interval = QSpinBox()
        self.replay_auto_interval.setRange(50, 5000)
        self.replay_auto_interval.setValue(500)
        self.replay_auto_interval.setSuffix(" ms")
        self.replay_auto_checkbox.toggled.connect(self._toggle_replay_auto)
        self.replay_auto_interval.valueChanged.connect(self._set_replay_auto_interval)
        replay_auto_row.addWidget(self.replay_auto_checkbox)
        replay_auto_row.addWidget(self.replay_auto_interval)
        replay_auto_row.addStretch(1)
        replay_layout.addLayout(replay_auto_row)
        self.replay_pred_label = QLabel("Current prediction: -")
        self._apply_text_outline_effect(self.replay_pred_label)
        replay_layout.addWidget(self.replay_pred_label)
        replay_btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Replay Data")
        load_btn.clicked.connect(self._load_replay_data)
        refresh_btn = QPushButton("Refresh Views")
        refresh_btn.clicked.connect(self._refresh_replay_views)
        replay_btn_row.addWidget(load_btn)
        replay_btn_row.addWidget(refresh_btn)
        replay_btn_row.addStretch(1)
        replay_layout.addLayout(replay_btn_row)
        self.replay_feature_view = pg.ImageView()
        try:
            self.replay_feature_view.getView().setAspectLocked(False)
        except Exception:
            pass
        self.replay_hidden_plot = pg.PlotWidget()
        self.replay_saliency_view = pg.ImageView()
        self._add_section_header(
            replay_layout,
            "Feature Maps",
            "Convolutional feature maps for the selected window/layer.",
        )
        replay_layout.addWidget(self.replay_feature_view)
        self._add_section_header(
            replay_layout,
            "Hidden Magnitude",
            "LSTM hidden-state magnitude over time for the selected window.",
        )
        replay_layout.addWidget(self.replay_hidden_plot)
        self._add_section_header(
            replay_layout,
            "Prediction Timeline (Fingers)",
            "Per-timestep finger probabilities from the model.",
        )
        self.replay_pred_finger_plot = pg.PlotWidget()
        replay_layout.addWidget(self.replay_pred_finger_plot)
        self._add_section_header(
            replay_layout,
            "Prediction Timeline (Actions)",
            "Per-timestep action probabilities from the model.",
        )
        self.replay_pred_action_plot = pg.PlotWidget()
        replay_layout.addWidget(self.replay_pred_action_plot)
        self._add_section_header(
            replay_layout,
            "Saliency",
            "Input saliency (absolute gradient) for the predicted action.",
        )
        replay_layout.addWidget(self.replay_saliency_view)

        live_tab = QWidget()
        live_layout = QVBoxLayout(live_tab)
        live_notice = QLabel("Available only while Step 7 Live Infer + Actuate is actively running.")
        live_notice.setWordWrap(True)
        self._apply_text_outline_effect(live_notice)
        live_layout.addWidget(live_notice)
        self.live_pred_label = QLabel("Current prediction: -")
        self._apply_text_outline_effect(self.live_pred_label)
        live_layout.addWidget(self.live_pred_label)
        self.live_feature_view = pg.ImageView()
        try:
            self.live_feature_view.getView().setAspectLocked(False)
        except Exception:
            pass
        self.live_hidden_plot = pg.PlotWidget()
        self.live_saliency_view = pg.ImageView()
        self._add_section_header(
            live_layout,
            "Feature Maps",
            "Latest convolutional feature maps from the active Step 7 inference window.",
        )
        live_layout.addWidget(self.live_feature_view)
        self._add_section_header(
            live_layout,
            "Hidden Magnitude",
            "Streaming LSTM hidden-state magnitude from the active Step 7 inference window.",
        )
        live_layout.addWidget(self.live_hidden_plot)
        self._add_section_header(
            live_layout,
            "Prediction Timeline (Fingers)",
            "Per-timestep finger probabilities for the latest Step 7 inference window.",
        )
        self.live_pred_finger_plot = pg.PlotWidget()
        live_layout.addWidget(self.live_pred_finger_plot)
        self._add_section_header(
            live_layout,
            "Prediction Timeline (Actions)",
            "Per-timestep action probabilities for the latest Step 7 inference window.",
        )
        self.live_pred_action_plot = pg.PlotWidget()
        live_layout.addWidget(self.live_pred_action_plot)
        self._add_section_header(
            live_layout,
            "Saliency",
            "Input saliency for the latest Step 7 inference window.",
        )
        live_layout.addWidget(self.live_saliency_view)

        self.model_view_tabs.addTab(replay_tab, "Replay")
        self.live_viz_tab_index = self.model_view_tabs.addTab(live_tab, "Step 7 Live")
        layout.addWidget(self.model_view_tabs)
        self._bind_checkbox(self.live_viz_checkbox, "infer", "LIVE_VIZ_ENABLED")
        field = self.fields.get("infer", {}).get("LIVE_VIZ_FPS")
        if isinstance(field, QSpinBox):
            self.live_viz_fps_spin.setValue(field.value())
            self.live_viz_fps_spin.valueChanged.connect(field.setValue)
            field.valueChanged.connect(self.live_viz_fps_spin.setValue)
        self.live_viz_checkbox.toggled.connect(lambda _val: self._update_live_viz_status())
        self._update_live_viz_status()
        self._autofill_replay_paths()
        return widget

    def _open_model_views_window(self) -> None:
        if self.model_views_window and self.model_views_window.isVisible():
            self.model_views_window.raise_()
            self.model_views_window.activateWindow()
            return
        if self.model_views_window is None and self._model_views_root is not None:
            self._model_views_root = None
        content = self._build_model_views_widget()
        dialog = QDialog(self)
        dialog.setWindowTitle("Model Views")
        dialog.setMinimumSize(980, 740)
        dialog.setModal(False)
        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(6, 6, 6, 6)
        scroll = self._wrap_scroll(content, "CentralWorkspace")
        dialog_layout.addWidget(scroll)
        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        close_row.addWidget(close_btn)
        dialog_layout.addLayout(close_row)
        dialog.finished.connect(self._on_model_views_window_closed)
        self.model_views_window = dialog
        self._model_views_root = content
        dialog.show()

    def _make_info_button(self, title: str, body: str) -> QToolButton:
        btn = QToolButton()
        btn.setText("i")
        btn.setToolTip("Info")
        btn.setFixedSize(18, 18)
        btn.setStyleSheet(
            "QToolButton { border: 1px solid #6b7c92; border-radius: 9px; "
            "background: #c9d2df; color: #2c3e50; font-weight: 700; }"
            "QToolButton:hover { background: #d7dee8; }"
        )
        btn.clicked.connect(lambda: QMessageBox.information(self, title, body))
        return btn

    def _add_section_header(self, layout: QVBoxLayout, title: str, body: str) -> None:
        row = QHBoxLayout()
        label = QLabel(title)
        self._apply_text_outline_effect(label)
        row.addWidget(label)
        row.addWidget(self._make_info_button(title, body))
        row.addStretch(1)
        layout.addLayout(row)

    def _on_model_views_window_closed(self, *_args) -> None:
        self._replay_auto_timer.stop()
        self.model_views_window = None
        self._model_views_root = None

    def _live_viz_enabled(self) -> bool:
        field = self.fields.get("infer", {}).get("LIVE_VIZ_ENABLED")
        if isinstance(field, QCheckBox):
            return field.isChecked()
        return False

    def _live_model_view_active(self) -> bool:
        return bool(self.runner.is_running() and self.active_step == "infer")

    def _update_model_view_access(self) -> None:
        if not PYQTGRAPH_AVAILABLE:
            return
        if not hasattr(self, "model_view_tabs") or self.live_viz_tab_index is None:
            return
        live_enabled = self._live_model_view_active()
        self.model_view_tabs.setTabEnabled(self.live_viz_tab_index, live_enabled)
        if not live_enabled and self.model_view_tabs.currentIndex() == self.live_viz_tab_index:
            self.model_view_tabs.setCurrentIndex(0)

    def _render_model_visualization(
        self,
        *,
        feature_view: Any,
        hidden_plot: Any,
        pred_label: Optional[QLabel],
        pred_finger_plot: Any,
        pred_action_plot: Any,
        saliency_view: Any,
        feature_map: Optional[np.ndarray],
        hidden_mag: Optional[np.ndarray],
        finger_probs: Optional[np.ndarray],
        action_probs: Optional[np.ndarray],
        saliency: Optional[np.ndarray],
    ) -> None:
        if not PYQTGRAPH_AVAILABLE:
            return
        if feature_view is not None:
            if feature_map is not None and feature_map.size:
                feature_view.setImage(feature_map, autoLevels=True)
            else:
                try:
                    feature_view.clear()
                except Exception:
                    pass
        if hidden_plot is not None:
            hidden_plot.clear()
            if hidden_mag is not None and hidden_mag.size:
                hidden_plot.plot(hidden_mag)
        if pred_label is not None:
            finger_text = "-"
            action_text = "-"
            if finger_probs is not None and finger_probs.size:
                finger_last = finger_probs[-1]
                finger_idx = int(np.argmax(finger_last))
                finger_prob = float(finger_last[finger_idx])
                finger_name = FINGER_NAMES.get(finger_idx, f"finger_{finger_idx}")
                finger_text = f"{finger_name} ({finger_prob:.2f})"
            if action_probs is not None and action_probs.size:
                action_last = action_probs[-1]
                action_idx = int(np.argmax(action_last))
                action_prob = float(action_last[action_idx])
                action_name = ACTION_NAMES.get(action_idx, f"action_{action_idx}")
                action_text = f"{action_name} ({action_prob:.2f})"
            pred_label.setText(
                f"Current prediction: Finger {finger_text}, Action {action_text}"
            )
        self._plot_prediction_timeline(
            pred_finger_plot,
            finger_probs if finger_probs is not None else np.array([]),
            FINGER_NAMES,
            title="Finger Prob",
        )
        self._plot_prediction_timeline(
            pred_action_plot,
            action_probs if action_probs is not None else np.array([]),
            ACTION_NAMES,
            title="Action Prob",
        )
        if saliency_view is not None:
            if saliency is not None and saliency.size:
                saliency_view.setImage(saliency, autoLevels=True)
            else:
                try:
                    saliency_view.clear()
                except Exception:
                    pass

    def _update_live_viz_status(self) -> None:
        if self.live_viz_status_label is None:
            return
        self._update_model_view_access()
        if not self._live_model_view_active():
            self.live_viz_status_label.setText("Live Model View: Step 7 inactive")
            return
        if not self._live_viz_enabled():
            self.live_viz_status_label.setText("Live Model View: Disabled")
            return
        now = time.monotonic()
        if self._last_live_viz_mono and (now - self._last_live_viz_mono) <= 2.5:
            self.live_viz_status_label.setText("Live Model View: Active")
        else:
            self.live_viz_status_label.setText("Live Model View: Waiting for Step 7 payloads")

    def _load_replay_data(self) -> None:
        if not PYQTGRAPH_AVAILABLE:
            self._append_log(
                "⚠️ Replay views require pyqtgraph in the active Python environment."
            )
            return
        session_dir = self._resolve_effective_session_dir(step_id=None)
        sessions_root = None
        if self.current_project and self.current_subject:
            sessions_root = subject_root(self.current_project, self.current_subject) / "sessions"
        npz_path, model_path, scaler_path = resolve_replay_artifact_paths(
            session_dir=session_dir,
            sessions_root=sessions_root,
            npz_text=self.replay_npz_path.text().strip(),
            model_text=self.replay_model_path.text().strip(),
            scaler_text=self.replay_scaler_path.text().strip(),
        )
        if npz_path is None or model_path is None or scaler_path is None:
            missing = []
            if npz_path is None:
                missing.append("windows NPZ")
            if model_path is None:
                missing.append("model")
            if scaler_path is None:
                missing.append("scaler")
            details = ", ".join(missing)
            session_text = str(session_dir) if session_dir is not None else "(none)"
            self._append_log(
                f"⚠️ Failed to resolve replay data paths for: {details}. Session dir: {session_text}"
            )
            return
        self.replay_npz_path.setText(str(npz_path))
        self.replay_model_path.setText(str(model_path))
        self.replay_scaler_path.setText(str(scaler_path))
        try:
            self.replay_viz = ReplayVisualizer(
                npz_path=str(npz_path),
                model_path=str(model_path),
                scaler_path=str(scaler_path),
            )
            self.replay_window_index.setMaximum(max(0, self.replay_viz.window_count - 1))
            self._refresh_replay_views()
            self._append_log("✅ Replay data loaded for model views.")
        except Exception as exc:
            self._append_log(f"⚠️ Failed to load replay data: {exc}")

    def _refresh_replay_views(self) -> None:
        if not PYQTGRAPH_AVAILABLE or not self.replay_viz:
            return
        idx = int(self.replay_window_index.value())
        layer_idx = int(self.replay_layer_index.value())
        try:
            feature_map = self.replay_viz.feature_map(idx, layer_idx)
            hidden_mag = self.replay_viz.hidden_magnitude(idx)
            saliency = self.replay_viz.saliency(idx)
            timeline = self.replay_viz.prediction_timeline(idx)
            finger_probs = timeline[0] if timeline is not None else None
            action_probs = timeline[1] if timeline is not None else None
            self._render_model_visualization(
                feature_view=self.replay_feature_view,
                hidden_plot=self.replay_hidden_plot,
                pred_label=self.replay_pred_label,
                pred_finger_plot=self.replay_pred_finger_plot,
                pred_action_plot=self.replay_pred_action_plot,
                saliency_view=self.replay_saliency_view,
                feature_map=feature_map,
                hidden_mag=hidden_mag,
                finger_probs=finger_probs,
                action_probs=action_probs,
                saliency=saliency,
            )
        except Exception as exc:
            self._append_log(f"⚠️ Replay view refresh failed: {exc}")

    def _refresh_live_model_views(self) -> None:
        if not PYQTGRAPH_AVAILABLE or not self._live_model_view_active():
            return
        payload = self._latest_live_viz_payload
        if not payload:
            return

        def _as_array(name: str) -> Optional[np.ndarray]:
            value = payload.get(name)
            if value is None:
                return None
            arr = np.asarray(value, dtype=float)
            return arr if arr.size else None

        hidden_mag = _as_array("hidden_timeline")
        if hidden_mag is None:
            hidden_scalar = payload.get("hidden_mag")
            if hidden_scalar is not None:
                hidden_mag = np.asarray([float(hidden_scalar)], dtype=float)

        self._render_model_visualization(
            feature_view=self.live_feature_view,
            hidden_plot=self.live_hidden_plot,
            pred_label=self.live_pred_label,
            pred_finger_plot=self.live_pred_finger_plot,
            pred_action_plot=self.live_pred_action_plot,
            saliency_view=self.live_saliency_view,
            feature_map=_as_array("feature_map"),
            hidden_mag=hidden_mag,
            finger_probs=_as_array("finger_probs"),
            action_probs=_as_array("action_probs"),
            saliency=_as_array("saliency"),
        )

    def _toggle_replay_auto(self, enabled: bool) -> None:
        if enabled:
            self._replay_auto_timer.start()
            self._advance_replay_window()
        else:
            self._replay_auto_timer.stop()

    def _set_replay_auto_interval(self, value: int) -> None:
        self._replay_auto_timer.setInterval(int(value))
        if self._replay_auto_timer.isActive():
            self._replay_auto_timer.start()

    def _advance_replay_window(self) -> None:
        if not self.replay_window_index:
            return
        if not self.replay_viz:
            return
        current = int(self.replay_window_index.value())
        maximum = int(self.replay_window_index.maximum())
        if maximum <= 0:
            return
        next_idx = 0 if current >= maximum else current + 1
        self.replay_window_index.setValue(next_idx)
        self._refresh_replay_views()

    def _plot_prediction_timeline(
        self,
        plot_widget: Optional["pg.PlotWidget"],
        probs: np.ndarray,
        label_map: Dict[int, str],
        *,
        title: str,
    ) -> None:
        if plot_widget is None:
            return
        plot_item = plot_widget.getPlotItem()
        if plot_item.legend is not None:
            try:
                plot_item.legend.scene().removeItem(plot_item.legend)
            except Exception:
                pass
            plot_item.legend = None
        plot_item.clear()
        plot_item.addLegend(offset=(10, 10))
        plot_item.setLabel("left", title)
        plot_item.setLabel("bottom", "t")
        if probs is None or probs.size == 0:
            return

        steps = int(probs.shape[0])
        classes = int(probs.shape[1]) if probs.ndim == 2 else 0
        if classes <= 0 or steps <= 0:
            return
        x = np.arange(steps, dtype=float)
        for idx in range(classes):
            label = label_map.get(idx, f"class_{idx}")
            color = pg.intColor(idx, hues=max(6, classes))
            plot_item.plot(x, probs[:, idx], pen=pg.mkPen(color, width=1), name=label)

    def _bind_checkbox(self, dock_cb: QCheckBox, step_id: str, key: str) -> None:
        field_cb = self.fields.get(step_id, {}).get(key)
        if not isinstance(field_cb, QCheckBox):
            dock_cb.setEnabled(False)
            return
        dock_cb.setChecked(field_cb.isChecked())
        dock_cb.toggled.connect(field_cb.setChecked)
        field_cb.toggled.connect(dock_cb.setChecked)

    def _sync_infer_inference_engine_controls(self) -> None:
        infer_fields = self.fields.get("infer", {})
        enabled = False
        cb = infer_fields.get("use_inference_engine")
        if isinstance(cb, QCheckBox):
            enabled = cb.isChecked()
        for key in (
            "mc_passes",
            "uncertainty_base_threshold",
            "uncertainty_weight",
        ):
            widget = infer_fields.get(key)
            if isinstance(widget, QWidget):
                widget.setEnabled(enabled)

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def _set_stream_status(self, text: str) -> None:
        if hasattr(self, "stream_status") and self.stream_status is not None:
            self.stream_status.setText(text)
        if hasattr(self, "stream_status_dock") and self.stream_status_dock is not None:
            self.stream_status_dock.setText(text)

    def _set_connector_status(self, status: str) -> None:
        text = f"Connector: {status}"
        for label in (
            getattr(self, "connector_status_dock", None),
            getattr(self, "connector_status_page", None),
        ):
            if isinstance(label, QLabel):
                label.setText(text)

    def _set_connector_device(self, device: str) -> None:
        text = f"Muse device: {device}"
        for label in (
            getattr(self, "connector_device_dock", None),
            getattr(self, "connector_device_page", None),
        ):
            if isinstance(label, QLabel):
                label.setText(text)

    def _set_connector_stream(self, stream_name: str) -> None:
        text = f"LSL stream: {stream_name}"
        for label in (
            getattr(self, "connector_stream_dock", None),
            getattr(self, "connector_stream_page", None),
        ):
            if isinstance(label, QLabel):
                label.setText(text)

    def _set_connector_log(self, line: str) -> None:
        text = f"Last connector log: {line}"
        for label in (
            getattr(self, "connector_log_dock", None),
            getattr(self, "connector_log_page", None),
        ):
            if isinstance(label, QLabel):
                label.setText(text)

    def _set_project_label(self, text: str) -> None:
        self.project_label.setText(text)
        if hasattr(self, "project_label_dock") and self.project_label_dock is not None:
            self.project_label_dock.setText(text)
        self._refresh_eval_context()

    def _set_subject_label(self, text: str) -> None:
        self.subject_label.setText(text)
        if hasattr(self, "subject_label_dock") and self.subject_label_dock is not None:
            self.subject_label_dock.setText(text)
        self._refresh_eval_context()

    def _set_session_label(self, text: str) -> None:
        self.session_label.setText(text)
        if hasattr(self, "session_label_dock") and self.session_label_dock is not None:
            self.session_label_dock.setText(text)
        self._refresh_eval_context()

    def _refresh_eval_context(self) -> None:
        if not hasattr(self, "eval_context_label"):
            return
        project = self.current_project or "-"
        subject = self.current_subject or "-"
        session = self.current_session_ui or "-"
        session_dir = "-"
        if self.current_project and self.current_subject:
            subject_dir = subject_root(self.current_project, self.current_subject)
            resolved = self._resolve_session_dir_for_current(subject_dir)
            if resolved:
                session_dir = str(resolved)
        context = f"Project: {project} | Subject: {subject} | Session: {session}\nSession dir: {session_dir}"
        self.eval_context_label.setText(context)

    def _wire_status_updates(self) -> None:
        for step_id in ("step1", "infer"):
            for key in ("ENABLE_ICA", "EVENT_MARKING_ENABLED"):
                widget = self.fields.get(step_id, {}).get(key)
                if isinstance(widget, QCheckBox):
                    widget.toggled.connect(self._refresh_status_summary)
        self.input_source.currentTextChanged.connect(self._refresh_status_summary)

    def _refresh_status_summary(self) -> None:
        stream_label = self.input_source.currentText()
        self._set_status_semantic(self.stream_state_label, "neutral", f"Stream: {stream_label}")
        ica_enabled = False
        event_enabled = False
        for step_id in ("step1", "infer"):
            if isinstance(
                self.fields.get(step_id, {}).get("ENABLE_ICA"), QCheckBox
            ) and self.fields[step_id]["ENABLE_ICA"].isChecked():
                ica_enabled = True
            if isinstance(
                self.fields.get(step_id, {}).get("EVENT_MARKING_ENABLED"), QCheckBox
            ) and self.fields[step_id]["EVENT_MARKING_ENABLED"].isChecked():
                event_enabled = True
        self._set_status_semantic(
            self.ica_state_label,
            "green" if ica_enabled else "yellow",
            f"ICA: {'on' if ica_enabled else 'off'}",
        )
        runtime_note = ""
        payload = self._read_session_state_payload()
        if payload:
            runtime_allowed = payload.get("event_marking_allowed")
            if runtime_allowed is not None:
                runtime_note = f" (runtime: {'on' if runtime_allowed else 'off'})"
        self._set_status_semantic(
            self.events_state_label,
            "green" if event_enabled else "yellow",
            f"Events: {'on' if event_enabled else 'off'}{runtime_note}",
        )

    def _refresh_health_indicator(self) -> None:
        if not hasattr(self, "health_indicator") or self.health_indicator is None:
            return
        payload = self._read_session_state_payload()
        if not payload:
            self._set_status_semantic(self.health_indicator, "yellow", "Health: unknown")
            return
        if payload.get("hard_stop_triggered"):
            self._set_status_semantic(self.health_indicator, "red", "Health: HARD STOP")
            return
        active = payload.get("data_stream_active")
        if active is True:
            self._set_status_semantic(self.health_indicator, "green", "Health: healthy")
        elif active is False:
            self._set_status_semantic(self.health_indicator, "red", "Health: unhealthy")
        else:
            self._set_status_semantic(self.health_indicator, "yellow", "Health: unknown")

    def _build_project_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        form = QFormLayout()
        self.project_combo = QComboBox()
        self._refresh_projects()
        self.project_combo.currentTextChanged.connect(self._open_project)
        self.project_name_input = OutlineLineEdit()

        open_project_label = QLabel("Open Project")
        open_project_label.setStyleSheet("color: white;")
        form.addRow(open_project_label, self.project_combo)
        form.addRow("New Project", self.project_name_input)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        create_btn = QPushButton("Create / Open")
        create_btn.clicked.connect(self._create_project)
        btn_row.addWidget(create_btn)
        layout.addLayout(btn_row)
        layout.addStretch(1)
        return page

    def _build_subject_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        form = QFormLayout()
        self.subject_combo = QComboBox()
        self.subject_combo.currentTextChanged.connect(self._select_subject)
        form.addRow("Subject", self.subject_combo)
        layout.addLayout(form)

        edit_btn = QPushButton("Create / Edit Subject")
        edit_btn.clicked.connect(self._edit_subject)
        layout.addWidget(edit_btn)
        layout.addStretch(1)
        return page


    def _build_projects_page(self) -> QWidget:
        """
        Combined Project + Subject management page.

        Top: project selection / creation.
        Bottom: subject selection / create/edit (within selected project).
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        header_row = QHBoxLayout()
        header = QLabel("Projects & Subjects")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        header_row.addWidget(
            self._make_info_button(
                "Projects & Subjects",
                "Create/open a project, then create/edit subjects within it. "
                "Subjects are used to organize sessions and model runs.",
            )
        )
        header_row.addStretch(1)
        layout.addLayout(header_row)

        # --- Project controls ---
        project_box = QGroupBox("Project")
        project_layout = QVBoxLayout(project_box)
        project_form = QFormLayout()

        self.project_combo = QComboBox()
        # Populate without triggering open until the user selects
        self._refresh_projects()
        self.project_combo.currentTextChanged.connect(self._open_project)

        self.project_name_input = OutlineLineEdit()

        open_project_label = QLabel("Open Project")
        open_project_label.setStyleSheet("color: white;")
        project_form.addRow(open_project_label, self.project_combo)
        project_form.addRow("New Project", self.project_name_input)
        project_layout.addLayout(project_form)

        project_btn_row = QHBoxLayout()
        create_btn = QPushButton("Create / Open")
        create_btn.clicked.connect(self._create_project)
        project_btn_row.addWidget(create_btn)
        project_btn_row.addStretch(1)
        project_layout.addLayout(project_btn_row)

        layout.addWidget(project_box)

        # --- Subject controls ---
        subject_box = QGroupBox("Subject")
        subject_layout = QVBoxLayout(subject_box)
        subject_form = QFormLayout()

        self.subject_combo = QComboBox()
        self.subject_combo.currentTextChanged.connect(self._select_subject)
        subject_form.addRow("Subject", self.subject_combo)
        self.projects_selected_session_value = QLabel("(none)")
        self.projects_selected_session_value.setWordWrap(True)
        subject_form.addRow("Selected Session", self.projects_selected_session_value)
        subject_layout.addLayout(subject_form)

        edit_btn = QPushButton("Create / Edit Subject")
        edit_btn.clicked.connect(self._edit_subject)
        subject_layout.addWidget(edit_btn)

        layout.addWidget(subject_box)

        # Ensure subject list reflects current project selection (if any)
        self._refresh_subjects()

        layout.addStretch(1)
        return page

    def _build_stream_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header_row = QHBoxLayout()
        header = QLabel("Stream Setup")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        header_row.addWidget(
            self._make_info_button(
                "Stream Setup",
                "Connect Muse via BLE→LSL, select an LSL stream, or use CSV offline mode. "
                "Use this page to verify the stream and sampling rate before recording.",
            )
        )
        header_row.addStretch(1)
        layout.addLayout(header_row)

        self.stream_status = QLabel("")
        connector_box = QGroupBox("Muse Connector (BLE → LSL)")
        connector_layout = QFormLayout(connector_box)
        self.stream_name_input = OutlineLineEdit()
        self.stream_name_input.setPlaceholderText("Auto (Muse2-EEG-<subject>)")
        self.stream_name_input.textChanged.connect(self._on_stream_name_input)
        self.live_connect_btn_page = QPushButton("Connect Muse (BLE → LSL)")
        self.live_connect_btn_page.clicked.connect(self._connect_muse)
        self.live_disconnect_btn_page = QPushButton("Disconnect Muse")
        self.live_disconnect_btn_page.clicked.connect(self._disconnect_muse)
        connector_buttons = QHBoxLayout()
        connector_buttons.addWidget(self.live_connect_btn_page)
        connector_buttons.addWidget(self.live_disconnect_btn_page)
        connector_buttons_widget = QWidget()
        connector_buttons_widget.setLayout(connector_buttons)
        self.connector_status_page = QLabel("Connector: idle")
        self.connector_device_page = QLabel("Muse device: -")
        self.connector_stream_page = QLabel("LSL stream: -")
        self.connector_log_page = QLabel("Last connector log: -")
        self.connector_log_page.setWordWrap(True)
        connector_layout.addRow("Stream name", self.stream_name_input)
        connector_layout.addRow("Controls", connector_buttons_widget)
        connector_layout.addRow(self.connector_status_page)
        connector_layout.addRow(self.connector_device_page)
        connector_layout.addRow(self.connector_stream_page)
        connector_layout.addRow(self.connector_log_page)
        layout.addWidget(connector_box)
        form = QFormLayout()

        self.input_source = QComboBox()
        self.input_source.addItems(["Muse 2 (LSL)", "Any LSL Stream", "CSV Offline"])
        self.input_source.currentTextChanged.connect(self._update_stream_controls)

        self.lsl_combo = QComboBox()
        self.lsl_combo.currentTextChanged.connect(self._on_lsl_stream_changed)
        self.detect_btn = QPushButton("Scan LSL Streams")
        self.detect_btn.clicked.connect(self._detect_lsl_streams)

        self.csv_path = OutlineLineEdit()
        csv_btn = QPushButton("Browse")
        csv_btn.clicked.connect(
            lambda: self._browse_path(
                self.csv_path, "CSV (*.csv)", "Select CSV", mode="open"
            )
        )
        csv_row = QHBoxLayout()
        csv_row.addWidget(self.csv_path)
        csv_row.addWidget(csv_btn)
        csv_widget = QWidget()
        csv_widget.setLayout(csv_row)

        self.sample_rate_display = QLabel("-")
        self.test_btn = QPushButton("Test Connection")
        self.test_btn.clicked.connect(self._test_lsl)

        form.addRow("Input Source", self.input_source)
        form.addRow("LSL Stream", self.lsl_combo)
        form.addRow("CSV Offline", csv_widget)
        form.addRow("Sampling Rate", self.sample_rate_display)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.detect_btn)
        btn_row.addWidget(self.test_btn)
        layout.addLayout(btn_row)
        layout.addWidget(self.stream_status)

        if not LSL_AVAILABLE:
            self.lsl_combo.hide()
            self.detect_btn.hide()
            self.test_btn.hide()
            self.input_source.clear()
            self.input_source.addItems(["CSV Offline"])
            self.input_source.setCurrentIndex(0)
            self.csv_path.setEnabled(True)
            self.live_connect_btn_page.setEnabled(False)
            self.live_disconnect_btn_page.setEnabled(False)
            self._set_stream_status(
                "pylsl not installed; LSL controls hidden (CSV offline only)."
            )
        else:
            self._update_stream_controls()

        layout.addStretch(1)
        return page

    def _build_pipeline_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header_row = QHBoxLayout()
        header = QLabel("Pipeline Overview")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        header_row.addWidget(
            self._make_info_button(
                "Pipeline Overview",
                "High-level map of the end-to-end pipeline. Use the left navigation to "
                "move through steps in order.",
            )
        )
        header_row.addStretch(1)
        layout.addLayout(header_row)
        intro = QLabel(
            "Use the navigation on the left to walk through the lossless pipeline. "
            "Record (lossless) captures raw shards + events only (no inference). "
            "Window extraction and training are offline, and live inference is a separate script."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        for label, row in [
            ("1) Record (Lossless)", 1),
            ("Events: Mark/Edit (Optional)", 2),
            ("Validate Session (Tool)", 3),
            ("1b) Extract Windows", 4),
            ("2) Train Model", 5),
            ("3+) Evaluate / Reports", 6),
            ("7) Live Infer + Actuate", 7),
        ]:
            box = QGroupBox(label)
            box_layout = QVBoxLayout(box)
            go_btn = QPushButton(f"Open {label}")
            go_btn.clicked.connect(lambda _=None, r=row: self.workflow_list.setCurrentRow(r))
            box_layout.addWidget(go_btn)
            layout.addWidget(box)

        layout.addStretch(1)
        return page

    def _build_session_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header_row = QHBoxLayout()
        header = QLabel("Validate Session (Tool)")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        header_row.addWidget(
            self._make_info_button(
                "Validate Session",
                "Checks session integrity (manifest continuity, missing shards, timebase ranges) "
                "and summarizes metadata before window extraction.",
            )
        )
        header_row.addStretch(1)
        layout.addLayout(header_row)

        form = QFormLayout()
        self.session_root_input = OutlineLineEdit()
        root_btn = QPushButton("Browse")
        root_btn.clicked.connect(
            lambda: self._browse_dir(self.session_root_input, "Select Session Root")
        )
        root_row = QHBoxLayout()
        root_row.setContentsMargins(0, 0, 0, 0)
        root_row.addWidget(self.session_root_input)
        root_row.addWidget(root_btn)
        root_widget = QWidget()
        root_widget.setLayout(root_row)
        form.addRow("Session Root", root_widget)

        self.session_dir_input = OutlineLineEdit()
        self.session_dir_input.textChanged.connect(self._on_session_dir_changed)
        session_btn = QPushButton("Browse")
        session_btn.clicked.connect(
            lambda: self._browse_dir(self.session_dir_input, "Select Session Directory")
        )
        session_row = QHBoxLayout()
        session_row.setContentsMargins(0, 0, 0, 0)
        session_row.addWidget(self.session_dir_input)
        session_row.addWidget(session_btn)
        session_widget = QWidget()
        session_widget.setLayout(session_row)
        form.addRow("Session Dir", session_widget)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        new_btn = QPushButton("Create New Session")
        new_btn.clicked.connect(self._create_new_session)
        load_btn = QPushButton("Load Session")
        load_btn.clicked.connect(self._load_session_summary)
        validate_btn = QPushButton("Validate Session")
        validate_btn.clicked.connect(self._run_validate_session)
        self.allow_partial_checkbox = QCheckBox("Allow partial validation")
        btn_row.addWidget(new_btn)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(validate_btn)
        btn_row.addWidget(self.allow_partial_checkbox)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        summary = QGroupBox("Session Summary")
        summary_layout = QFormLayout(summary)
        self.session_summary_labels = {
            "session_id": QLabel("-"),
            "created_at": QLabel("-"),
            "mode": QLabel("-"),
            "termination_reason": QLabel("-"),
            "seq_range": QLabel("-"),
            "missing_seq": QLabel("-"),
            "shards": QLabel("-"),
            "total_samples": QLabel("-"),
            "timebase_ranges": QLabel("-"),
            "events": QLabel("-"),
        }
        summary_layout.addRow("Session ID", self.session_summary_labels["session_id"])
        summary_layout.addRow("Created at", self.session_summary_labels["created_at"])
        summary_layout.addRow("Mode", self.session_summary_labels["mode"])
        summary_layout.addRow(
            "Termination", self.session_summary_labels["termination_reason"]
        )
        summary_layout.addRow("Seq range", self.session_summary_labels["seq_range"])
        summary_layout.addRow("Missing seq", self.session_summary_labels["missing_seq"])
        summary_layout.addRow("Shards", self.session_summary_labels["shards"])
        summary_layout.addRow("Total samples", self.session_summary_labels["total_samples"])
        summary_layout.addRow(
            "Timebase ranges", self.session_summary_labels["timebase_ranges"]
        )
        summary_layout.addRow("Event count", self.session_summary_labels["events"])
        layout.addWidget(summary)

        layout.addStretch(1)
        return page

    def _build_evaluate_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header_row = QHBoxLayout()
        header = QLabel("Step 3+: Evaluate / Reports")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        header_row.addWidget(
            self._make_info_button(
                "Evaluate / Reports",
                "Runs model evaluation, Deepchecks diagnostics, paper figures, "
                "and reports for the selected session. Recommended order: "
                "Step 3 → 3b → 3c → 4.",
            )
        )
        header_row.addStretch(1)
        layout.addLayout(header_row)
        note = QLabel(
            "Run evaluation on the selected session. Recommended order: "
            "Step 3 → 3b → 3c → 4. Outputs write under "
            "`sessions/<id>/processed/` by default."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        target_box = QGroupBox("Evaluation Target")
        target_layout = QFormLayout(target_box)
        self.eval_context_label = QLabel("")
        self.eval_context_label.setWordWrap(True)
        target_layout.addRow("Current selection", self.eval_context_label)
        self.eval_fields.setdefault("evaluate_common", {})
        common_fields = self.eval_fields["evaluate_common"]
        run_dir_override = OutlineLineEdit()
        run_dir_override.setPlaceholderText(
            "Optional: processed/models/<run_id> or full run dir path"
        )
        target_layout.addRow("Run dir override", run_dir_override)
        common_fields["run_dir"] = run_dir_override
        if hasattr(self, "session_dir_input"):
            session_widget = self._clone_bound_widget(
                self.session_dir_input, "session_dir"
            )
            if isinstance(session_widget, QLineEdit):
                session_widget.setPlaceholderText(
                    "Use Validate Session selection or browse here."
                )
            target_layout.addRow("Session Dir (recommended)", session_widget)
        layout.addWidget(target_box)

        full_btn_row = QHBoxLayout()
        full_btn = QPushButton("Run Full Evaluation (3 → 3b → 3c → 4)")
        full_btn.clicked.connect(self._run_evaluate_all)
        full_btn_row.addWidget(full_btn)
        full_btn_row.addStretch(1)
        layout.addLayout(full_btn_row)

        layout.addWidget(self._build_eval_step3_box())
        layout.addWidget(self._build_eval_deepchecks_box())
        layout.addWidget(self._build_eval_figures_box())
        layout.addWidget(self._build_eval_reports_box())
        layout.addStretch(1)
        self._refresh_eval_context()
        return page

    def _build_eval_step3_box(self) -> QWidget:
        box = QGroupBox("Step 3: Evaluate Model + Calibration (3_evaluate_model.py)")
        layout = QVBoxLayout(box)
        desc = QLabel(
            "Core evaluation: action/finger/joint accuracy, raw invalid-pair rate, "
            "confusion matrices, calibration curves, and cached predictions. "
            "Uses the latest model under the selected session."
        )
        desc.setWordWrap(True)
        desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        desc_row = QHBoxLayout()
        desc_row.addWidget(desc)
        desc_row.addWidget(
            self._make_info_button(
                "Step 3: Evaluate Model",
                "Deterministic evaluation of the latest model using the session's "
                "window dataset. Applies saved temperature scaling when available and writes "
                "an eval manifest and plots under "
                "`processed/reports/<run_id>/`.",
            )
        )
        desc_row.addStretch(1)
        layout.addLayout(desc_row)

        self.eval_fields.setdefault("evaluate", {})
        fields = self.eval_fields["evaluate"]

        basic = QGroupBox("Basic Overrides")
        basic_layout = QFormLayout(basic)

        max_samples = OutlineSpinBox()
        max_samples.setRange(0, 10_000_000)
        max_samples.setSpecialValueText("Auto")
        max_samples.setValue(0)
        basic_layout.addRow("Max samples (auto)", max_samples)
        fields["max_samples"] = max_samples

        batch_size = OutlineSpinBox()
        batch_size.setRange(1, 8192)
        batch_size.setValue(256)
        basic_layout.addRow("Batch size", batch_size)
        fields["batch_size"] = batch_size

        split_seed = OutlineSpinBox()
        split_seed.setRange(0, 1_000_000)
        split_seed.setValue(42)
        basic_layout.addRow("Split seed", split_seed)
        fields["split_seed"] = split_seed

        export_preds = QCheckBox("Export cached test predictions")
        basic_layout.addRow("Export test preds", export_preds)
        fields["export_test_pred"] = export_preds

        no_manifest = QCheckBox("Disable manifest output")
        basic_layout.addRow("No manifest", no_manifest)
        fields["no_manifest"] = no_manifest

        save_manifest = OutlineLineEdit()
        save_manifest.setPlaceholderText("Optional manifest path override")
        basic_layout.addRow("Save manifest", save_manifest)
        fields["save_manifest"] = save_manifest

        disable_det = QCheckBox("Disable deterministic eval (not recommended)")
        basic_layout.addRow("Determinism", disable_det)
        fields["disable_deterministic"] = disable_det

        layout.addWidget(basic)

        post = QGroupBox("Postprocess Overrides")
        post_layout = QFormLayout(post)

        smooth = QCheckBox("Enable smoothing")
        post_layout.addRow("Smooth", smooth)
        fields["smooth"] = smooth

        smooth_action_only = QCheckBox("Smooth action only")
        post_layout.addRow("Smooth action only", smooth_action_only)
        fields["smooth_action_only"] = smooth_action_only

        smooth_method = QComboBox()
        smooth_method.addItems(["vote", "ema"])
        smooth_method.setCurrentText("vote")
        post_layout.addRow("Smooth method", smooth_method)
        fields["smooth_method"] = smooth_method

        smooth_window = OutlineSpinBox()
        smooth_window.setRange(1, 200)
        smooth_window.setValue(5)
        post_layout.addRow("Smooth window", smooth_window)
        fields["smooth_window"] = smooth_window

        hysteresis = QCheckBox("Enable hysteresis")
        post_layout.addRow("Hysteresis", hysteresis)
        fields["hysteresis"] = hysteresis

        hysteresis_frames = OutlineSpinBox()
        hysteresis_frames.setRange(1, 50)
        hysteresis_frames.setValue(3)
        post_layout.addRow("Hysteresis frames", hysteresis_frames)
        fields["hysteresis_frames"] = hysteresis_frames

        threshold_action = OutlineDoubleSpinBox()
        threshold_action.setRange(0.0, 1.0)
        threshold_action.setDecimals(2)
        threshold_action.setSingleStep(0.01)
        threshold_action.setValue(0.20)
        post_layout.addRow("Threshold action", threshold_action)
        fields["threshold_action"] = threshold_action

        threshold_finger = OutlineDoubleSpinBox()
        threshold_finger.setRange(0.0, 1.0)
        threshold_finger.setDecimals(2)
        threshold_finger.setSingleStep(0.01)
        threshold_finger.setValue(0.20)
        post_layout.addRow("Threshold finger", threshold_finger)
        fields["threshold_finger"] = threshold_finger

        adjacency = QCheckBox("Enable adjacency assist")
        post_layout.addRow("Adjacency", adjacency)
        fields["adjacency"] = adjacency

        layout.addWidget(post)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run Step 3 Evaluate")
        run_btn.clicked.connect(lambda: self._run_eval_script("evaluate"))
        btn_row.addWidget(run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        if "evaluate" not in self.scripts:
            run_btn.setEnabled(False)
        return box

    def _build_eval_deepchecks_box(self) -> QWidget:
        box = QGroupBox("Step 3b: Deepchecks Evaluation (3b_deepchecks_evaluate.py)")
        layout = QVBoxLayout(box)
        desc = QLabel(
            "Dataset integrity and model evaluation checks. Aligns to the same session/model "
            "resolution as Step 3."
        )
        desc.setWordWrap(True)
        desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        desc_row = QHBoxLayout()
        desc_row.addWidget(desc)
        desc_row.addWidget(
            self._make_info_button(
                "Step 3b: Deepchecks",
                "Runs Deepchecks suites (data integrity, train/test validation, "
                "model evaluation) using the same split config as training unless overridden.",
            )
        )
        desc_row.addStretch(1)
        layout.addLayout(desc_row)

        self.eval_fields.setdefault("evaluate_deepchecks", {})
        fields = self.eval_fields["evaluate_deepchecks"]

        form = QFormLayout()
        max_samples = OutlineSpinBox()
        max_samples.setRange(0, 10_000_000)
        max_samples.setSpecialValueText("Auto")
        max_samples.setValue(0)
        form.addRow("Max samples (auto)", max_samples)
        fields["max_samples"] = max_samples

        batch_size = OutlineSpinBox()
        batch_size.setRange(1, 8192)
        batch_size.setValue(256)
        form.addRow("Batch size", batch_size)
        fields["batch_size"] = batch_size

        split_mode = QComboBox()
        split_mode.addItems(["Auto (train_config)", "group_trial", "holdout_session"])
        split_mode.setCurrentText("Auto (train_config)")
        form.addRow("Split mode", split_mode)
        fields["split_mode"] = split_mode

        purge_seconds = OutlineDoubleSpinBox()
        purge_seconds.setRange(0.0, 60.0)
        purge_seconds.setDecimals(2)
        purge_seconds.setSingleStep(0.25)
        purge_seconds.setValue(0.0)
        form.addRow("Purge seconds", purge_seconds)
        fields["purge_seconds"] = purge_seconds

        hop_seconds = OutlineDoubleSpinBox()
        hop_seconds.setRange(0.0, 10.0)
        hop_seconds.setDecimals(2)
        hop_seconds.setSingleStep(0.05)
        hop_seconds.setSpecialValueText("Auto")
        hop_seconds.setValue(0.0)
        form.addRow("Hop seconds (auto)", hop_seconds)
        fields["hop_seconds"] = hop_seconds

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run 3b Deepchecks")
        run_btn.clicked.connect(lambda: self._run_eval_script("evaluate_deepchecks"))
        btn_row.addWidget(run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        if "evaluate_deepchecks" not in self.scripts:
            run_btn.setEnabled(False)
        return box

    def _build_eval_figures_box(self) -> QWidget:
        box = QGroupBox("Step 3c: Paper Figures (3c_live_paper_figures.py)")
        layout = QVBoxLayout(box)
        desc = QLabel(
            "Generates reliability and confidence figures for reports/paper. "
            "Defaults: MC_SAMPLES=30, SEED=42."
        )
        desc.setWordWrap(True)
        desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        desc_row = QHBoxLayout()
        desc_row.addWidget(desc)
        desc_row.addWidget(
            self._make_info_button(
                "Step 3c: Paper Figures",
                "Produces MC-dropout reliability/uncertainty plots saved under "
                "`processed/reports/<run_id>/`.",
            )
        )
        desc_row.addStretch(1)
        layout.addLayout(desc_row)

        self.eval_fields.setdefault("evaluate_figures", {})
        fields = self.eval_fields["evaluate_figures"]

        form = QFormLayout()
        show_plots = QCheckBox("Show interactive plots (sets SHOW_PLOTS=1)")
        form.addRow("Live plots", show_plots)
        fields["show_plots"] = show_plots
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run 3c Paper Figures")
        run_btn.clicked.connect(lambda: self._run_eval_script("evaluate_figures"))
        btn_row.addWidget(run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        if "evaluate_figures" not in self.scripts:
            run_btn.setEnabled(False)
        return box

    def _build_eval_reports_box(self) -> QWidget:
        box = QGroupBox("Step 4: Generate Reports (4_generate_reports.py)")
        layout = QVBoxLayout(box)
        desc = QLabel(
            "Produces per-run HTML/summary reports. Uses the latest model run under the "
            "selected session unless you override the run directory."
        )
        desc.setWordWrap(True)
        desc.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        desc_row = QHBoxLayout()
        desc_row.addWidget(desc)
        desc_row.addWidget(
            self._make_info_button(
                "Step 4: Generate Reports",
                "Builds HTML reports from model artifacts (metrics, predictions, "
                "confusion matrices, calibration figures).",
            )
        )
        desc_row.addStretch(1)
        layout.addLayout(desc_row)

        self.eval_fields.setdefault("evaluate_reports", {})
        fields = self.eval_fields["evaluate_reports"]

        form = QFormLayout()
        run_dir = OutlineLineEdit()
        run_dir.setPlaceholderText("Optional: override run dir")
        form.addRow("Run dir override", run_dir)
        fields["run_dir"] = run_dir

        exp_hash = OutlineLineEdit()
        exp_hash.setPlaceholderText("Optional: legacy exp hash")
        form.addRow("Exp hash", exp_hash)
        fields["exp_hash"] = exp_hash

        subject_id = OutlineLineEdit()
        subject_id.setPlaceholderText("Optional: legacy subject id")
        form.addRow("Subject ID", subject_id)
        fields["subject_id"] = subject_id

        layout.addLayout(form)

        btn_row = QHBoxLayout()
        run_btn = QPushButton("Run 4 Generate Reports")
        run_btn.clicked.connect(lambda: self._run_eval_script("evaluate_reports"))
        btn_row.addWidget(run_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        if "evaluate_reports" not in self.scripts:
            run_btn.setEnabled(False)
        return box

    def _build_step1_page(self) -> QWidget:
        lossless_banner = QLabel(
            "LOSSLESS RECORD — raw EEG + events only. No inference. No drops allowed."
        )
        lossless_banner.setStyleSheet(
            "background: #263238; color: #fff; padding: 6px; font-weight: 700;"
        )
        return self._build_step_page(
            step_id="step1",
            title="Step 1: Record (Lossless)",
            defaults=default_step1_settings(),
            script_key="step1",
            include_event_tools=True,
            custom_controls=lossless_banner,
        )

    def _build_event_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        header_row = QHBoxLayout()
        header = QLabel("Events: Mark/Edit (Optional)")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        header_row.addWidget(
            self._make_info_button(
                "Events: Mark/Edit",
                "Review or repair event labels after capture. Use this when you need to "
                "clean up timestamps or correct labels.",
            )
        )
        header_row.addStretch(1)
        layout.addLayout(header_row)
        info = QLabel("Post-hoc event review/edit tools for the current session.")
        layout.addWidget(info)
        note = QLabel(
            "Live graph + event labeling run inside Step 1: Record (Lossless) "
            "(1_stream_and_record.py). "
            "This page is for review/repair after capture."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self.event_session_dir_override = OutlineLineEdit()
        self.event_session_dir_override.setPlaceholderText(
            "Optional override for the auto-selected session"
        )
        self.event_session_dir_override.textChanged.connect(
            lambda _text: self._update_checklist("event_tools")
        )
        event_session_btn = QPushButton("Browse")
        event_session_btn.clicked.connect(
            lambda: self._browse_dir(
                self.event_session_dir_override, "Select Event Review Session Directory"
            )
        )
        event_session_row = QHBoxLayout()
        event_session_row.setContentsMargins(0, 0, 0, 0)
        event_session_row.addWidget(self.event_session_dir_override)
        event_session_row.addWidget(event_session_btn)
        event_session_widget = QWidget()
        event_session_widget.setLayout(event_session_row)
        form.addRow("Session Dir Override", event_session_widget)
        self.event_features_path = OutlineLineEdit()
        self.event_events_path = OutlineLineEdit()
        form.addRow("Legacy Features CSV", self.event_features_path)
        form.addRow("Events JSONL", self.event_events_path)
        layout.addLayout(form)

        advanced = QGroupBox("Validation Options")
        adv_layout = QFormLayout(advanced)
        self.event_apply_fix = QCheckBox()
        self.event_strict = QCheckBox()
        self.event_json_report = OutlineLineEdit()
        json_btn = QPushButton("Browse")
        json_btn.clicked.connect(
            lambda: self._browse_path(
                self.event_json_report,
                "JSON (*.json);;All Files (*)",
                "Save JSON Report",
                mode="save",
            )
        )
        json_row = QHBoxLayout()
        json_row.setContentsMargins(0, 0, 0, 0)
        json_row.addWidget(self.event_json_report)
        json_row.addWidget(json_btn)
        json_widget = QWidget()
        json_widget.setLayout(json_row)
        adv_layout.addRow("Apply fixes", self.event_apply_fix)
        adv_layout.addRow("Strict mode", self.event_strict)
        adv_layout.addRow("JSON report", json_widget)
        layout.addWidget(advanced)

        btn_row = QHBoxLayout()
        review_btn = QPushButton("Launch Event Review (5_review_events.py)")
        review_btn.clicked.connect(self._run_event_review)
        validate_btn = QPushButton("Validate Events (5_validate_events.py)")
        validate_btn.clicked.connect(self._run_event_validate)
        finalize_btn = QPushButton("Finalize Save & Close")
        finalize_btn.clicked.connect(self._finalize_event_review)
        btn_row.addWidget(review_btn)
        btn_row.addWidget(validate_btn)
        btn_row.addWidget(finalize_btn)
        layout.addLayout(btn_row)

        self._build_checklist("event_tools", layout)
        layout.addStretch(1)
        return page

    def _build_step1b_page(self) -> QWidget:
        note = QLabel(
            "Extract windows from a lossless session directory (raw/ + events.jsonl). "
            "Use the Session Directory (sessions/<session_id>) field or select a session on the Validate Session page. "
            "Step 1b rejects OPEN/CLOSE events labeled with finger NONE, so fix or prune those events before extraction."
        )
        note.setWordWrap(True)
        return self._build_step_page(
            step_id="step1b",
            title="Step 1b: Extract Windows",
            defaults=default_step1b_settings(),
            script_key="step1b",
            include_event_tools=False,
            custom_controls=note,
        )

    def _build_preprocess_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        msg = QLabel(
            "No standalone preprocess/ICA script found. ICA runs inside Step 1."
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)
        layout.addStretch(1)
        return page

    def _build_train_page(self) -> QWidget:
        note = QLabel(
            "Step 2 trains the model, then fits post-hoc temperature scaling on a held-out "
            "calibration subset of the training split. The run now saves model, scaler, "
            "test predictions, and `temperature_scaling.json` together."
        )
        note.setWordWrap(True)
        return self._build_step_page(
            step_id="train",
            title="Step 2: Train Model",
            defaults=default_train_settings(),
            script_key="train",
            include_event_tools=False,
            custom_controls=note,
        )

    def _build_infer_page(self) -> QWidget:
        live_controls = self._build_live_infer_controls()
        return self._build_step_page(
            step_id="infer",
            title="Step 7: Live Infer + Actuate",
            defaults=default_infer_settings(),
            script_key="live_infer",
            include_event_tools=False,
            include_run_controls=True,
            custom_controls=live_controls,
        )

    def _build_live_infer_controls(self) -> QWidget:
        box = QGroupBox("Live Inference Notes")
        layout = QVBoxLayout(box)
        note = QLabel(
            "Live inference runs in 7_live_infer_and_actuate.py. "
            "When a session (or subject/project) is selected, the latest trained run "
            "is auto-resolved; model/scaler fields act as explicit overrides. "
            "Outputs default to processed/live_infer and auto-version if the folder exists. "
            "Disable file outputs to run inference-only for max performance. "
            "Use the inference subject dropdown to target a different subject for Step 7. "
            "Actuation is opt-in and requires confirmation before running. "
            "Enable the MC-dropout inference engine to use uncertainty-aware mean probabilities "
            "and adaptive actuation gating from utils/inference.py. "
            "Saved run-specific temperature scaling is auto-loaded and applied before softmax. "
            "Actuation speed is confidence-modulated by default unless you disable it."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        return box

    def _build_export_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        header_row = QHBoxLayout()
        header = QLabel("Export")
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        header_row.addWidget(
            self._make_info_button(
                "Export",
                "Export utilities (e.g., EEGLAB .set/.mat) would appear here when available.",
            )
        )
        header_row.addStretch(1)
        layout.addLayout(header_row)

        msg = QLabel("Export to EEGLAB (.set/.mat) not found in repo; export disabled.")
        msg.setWordWrap(True)
        layout.addWidget(msg)
        layout.addStretch(1)
        return page

    def _build_logs_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        msg_row = QHBoxLayout()
        msg = QLabel("Diagnostics for timebase alignment and event coverage.")
        msg_row.addWidget(msg)
        msg_row.addWidget(
            self._make_info_button(
                "Diagnostics",
                "Runs time alignment checks between raw EEG and events, and reports "
                "gaps or timebase inconsistencies.",
            )
        )
        msg_row.addStretch(1)
        layout.addLayout(msg_row)

        form = QFormLayout()
        self.diag_features_path = OutlineLineEdit()
        self.diag_events_path = OutlineLineEdit()
        form.addRow("Raw/Features CSV", self.diag_features_path)
        form.addRow("Events JSONL", self.diag_events_path)
        layout.addLayout(form)

        advanced = QGroupBox("Advanced")
        adv_layout = QFormLayout(advanced)
        self.diag_session_meta = OutlineLineEdit()
        meta_btn = QPushButton("Browse")
        meta_btn.clicked.connect(
            lambda: self._browse_path(
                self.diag_session_meta,
                "JSON (*.json);;All Files (*)",
                "Select Session Meta",
            )
        )
        meta_row = QHBoxLayout()
        meta_row.setContentsMargins(0, 0, 0, 0)
        meta_row.addWidget(self.diag_session_meta)
        meta_row.addWidget(meta_btn)
        meta_widget = QWidget()
        meta_widget.setLayout(meta_row)

        self.diag_target_fs = OutlineDoubleSpinBox()
        self.diag_target_fs.setRange(1, 4096)
        self.diag_target_fs.setValue(256.0)
        self.diag_target_fs.setDecimals(2)
        self.diag_self_test = QCheckBox()
        adv_layout.addRow("Session meta", meta_widget)
        adv_layout.addRow("Target FS", self.diag_target_fs)
        adv_layout.addRow("Self-test", self.diag_self_test)
        layout.addWidget(advanced)

        btn_row = QHBoxLayout()
        diag_btn = QPushButton("Run Time Alignment Check")
        diag_btn.clicked.connect(self._run_alignment_check)
        btn_row.addWidget(diag_btn)
        layout.addLayout(btn_row)

        self._build_checklist("diagnostics", layout)
        layout.addStretch(1)
        return page

    def _step_description(self, step_id: str) -> str:
        descriptions = {
            "step1": (
                "Record raw EEG + events into a session directory. No inference is run. "
                "Outputs live under `sessions/<id>/raw/` and `sessions/<id>/events/`."
            ),
            "step1b": (
                "Extract fixed windows from a session directory and generate `eeg_windows.npz`. "
                "Performs manifest continuity validation by default and rejects OPEN/CLOSE "
                "labels that use finger NONE."
            ),
            "train": (
                "Train the CNN+LSTM model from `eeg_windows.npz` and write model/scaler artifacts "
                "under `sessions/<id>/processed/models/<run_id>/`, including post-hoc "
                "temperature scaling."
            ),
            "infer": (
                "Run live inference on an LSL stream or CSV input. Optional actuation is opt-in "
                "with safety confirmation, latency logging, and auto-loaded run-specific "
                "temperature scaling."
            ),
        }
        return descriptions.get(step_id, "")

    def _build_step_page(
        self,
        step_id: str,
        title: str,
        defaults: Dict[str, Any],
        script_key: str,
        include_event_tools: bool,
        include_run_controls: bool = True,
        custom_controls: Optional[QWidget] = None,
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.step_script_key[step_id] = script_key

        description = self._step_description(step_id)
        header_row = QHBoxLayout()
        header = QLabel(title)
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        header_row.addWidget(header)
        if description:
            header_row.addWidget(self._make_info_button(title, description))
        header_row.addStretch(1)
        layout.addLayout(header_row)
        if description:
            desc_label = QLabel(description)
            desc_label.setWordWrap(True)
            layout.addWidget(desc_label)

        status_row = QHBoxLayout()
        status_label = QLabel("Status: Idle")
        self.step_status[step_id] = status_label
        status_row.addWidget(status_label)
        status_row.addStretch(1)
        layout.addLayout(status_row)

        if step_id == "step1":
            resume_row = QHBoxLayout()
            self.resume_status_label = QLabel("Resume available: -")
            self.resume_checkbox = QCheckBox("Resume last session")
            self.resume_checkbox.setEnabled(False)
            resume_row.addWidget(self.resume_status_label)
            resume_row.addWidget(self.resume_checkbox)
            resume_row.addStretch(1)
            layout.addLayout(resume_row)
            live_box = QGroupBox("Live Graph + Event Labeling")
            live_layout = QVBoxLayout(live_box)
            live_note = QLabel(
                "The live plot and keyboard event labeling run inside Step 1 "
                "(1_stream_and_record.py). Keep 'Enable plot' and 'Event marking' checked "
                "for live capture. Controls: Space=hold event, o/c/r=mode, a/k/n=override, "
                "1-5=assign finger, q or ESC=stop."
            )
            live_note.setWordWrap(True)
            live_layout.addWidget(live_note)
            layout.addWidget(live_box)

        self.fields[step_id] = {}
        self.defaults[step_id] = defaults

        form = QFormLayout()
        self._populate_basic_fields(step_id, form)
        layout.addLayout(form)

        if step_id == "infer":
            warning = QLabel(
                "⚠️ Actuation enabled. Confirm the hand is safe and clear before running."
            )
            warning.setStyleSheet("color: #f5d76e; font-weight: 700;")
            warning.setVisible(False)
            enable_widget = self.fields.get(step_id, {}).get("enable_actuation")
            if isinstance(enable_widget, QCheckBox):
                warning.setVisible(enable_widget.isChecked())
                enable_widget.toggled.connect(warning.setVisible)
            layout.addWidget(warning)

        advanced_group = QGroupBox("Advanced")
        advanced_group.setCheckable(True)
        advanced_group.setChecked(False)
        adv_layout = QFormLayout()
        self._populate_advanced_fields(step_id, adv_layout)
        advanced_group.setLayout(adv_layout)
        layout.addWidget(advanced_group)

        dropdown_row = QHBoxLayout()
        dropdown_row.addWidget(
            self._build_dropdown_button(
                "Config Flags",
                self._build_config_flags_panel(step_id),
            )
        )
        dropdown_row.addWidget(
            self._build_dropdown_button(
                "Passable Args",
                self._build_step_args_panel(step_id),
            )
        )
        dropdown_row.addStretch(1)
        layout.addLayout(dropdown_row)

        if custom_controls is not None:
            layout.addWidget(custom_controls)

        if include_run_controls:
            buttons = QHBoxLayout()
            run_btn = QPushButton("Run")
            stop_btn = QPushButton("Stop")
            reset_btn = QPushButton("Reset to defaults")
            run_btn.clicked.connect(lambda: self._run_step(step_id, script_key))
            stop_btn.clicked.connect(self._stop_process)
            reset_btn.clicked.connect(lambda: self._reset_step(step_id))
            buttons.addWidget(run_btn)
            buttons.addWidget(stop_btn)
            buttons.addWidget(reset_btn)
            layout.addLayout(buttons)

            if script_key not in self.scripts:
                run_btn.setEnabled(False)
                status_label.setText("Status: Missing script")

        self._build_checklist(step_id, layout)

        if include_event_tools:
            hint = QLabel(
                "Event marking runs inside Step 1 when enabled. Event review available on Events page."
            )
            hint.setWordWrap(True)
            layout.addWidget(hint)

        layout.addStretch(1)
        return page

    def _build_checklist(self, step_id: str, layout: QVBoxLayout) -> None:
        box = QGroupBox("Output Checklist")
        box_layout = QVBoxLayout(box)
        checklist = QListWidget()
        box_layout.addWidget(checklist)
        layout.addWidget(box)
        self.step_checklists[step_id] = checklist

    def _build_config_flags_panel(self, step_id: str) -> QWidget:
        panel = QWidget()
        layout = QFormLayout(panel)
        layout.setLabelAlignment(Qt.AlignRight)
        fields = self.fields.get(step_id, {})
        for key in sorted(fields.keys()):
            source_widget = fields[key]
            proxy = self._clone_bound_widget(source_widget, key)
            label = QLabel(self._friendly_label(step_id, key))
            label.setMinimumWidth(160)
            label.setWordWrap(True)
            layout.addRow(label, proxy)
        return panel

    def _build_step_args_panel(self, step_id: str) -> QWidget:
        panel = QWidget()
        layout = QFormLayout(panel)
        layout.setLabelAlignment(Qt.AlignRight)
        specs = self.step_arg_specs.get(step_id, [])
        self.step_arg_widgets.setdefault(step_id, {})
        self.step_arg_includes.setdefault(step_id, {})
        for spec in specs:
            if spec.kind == "bool":
                widget = QCheckBox()
                self._apply_tooltip(widget, spec.name, spec.description)
                source_widget = self.fields.get(step_id, {}).get(spec.name)
                if isinstance(source_widget, QCheckBox):
                    widget.setChecked(source_widget.isChecked())
                    widget.toggled.connect(source_widget.setChecked)
                    source_widget.toggled.connect(widget.setChecked)
                layout.addRow(QLabel(spec.flag), widget)
                self.step_arg_widgets[step_id][spec.name] = widget
                continue

            include_cb = QCheckBox("Include")
            include_cb.setChecked(False)
            self.step_arg_includes[step_id][spec.name] = include_cb
            source_widget = self.fields.get(step_id, {}).get(spec.name)
            if source_widget is not None:
                widget = self._clone_bound_widget(source_widget, spec.name)
            else:
                widget = OutlineLineEdit()
                widget.setPlaceholderText("Value")
            self._apply_tooltip(widget, spec.name, spec.description)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(include_cb)
            row_layout.addWidget(widget)
            layout.addRow(QLabel(spec.flag), row)
            self.step_arg_widgets[step_id][spec.name] = widget
        return panel

    def _clone_bound_widget(self, source_widget: QWidget, key: str) -> QWidget:
        if isinstance(source_widget, QCheckBox):
            target = QCheckBox()
            target.setChecked(source_widget.isChecked())
            target.toggled.connect(
                lambda val: self._sync_checkbox(source_widget, val)
            )
            source_widget.toggled.connect(
                lambda val: self._sync_checkbox(target, val)
            )
            self._apply_tooltip(target, key)
            return target
        if isinstance(source_widget, QSpinBox):
            target = OutlineSpinBox()
            target.setRange(source_widget.minimum(), source_widget.maximum())
            target.setValue(source_widget.value())
            target.valueChanged.connect(
                lambda val: self._sync_spinbox(source_widget, val)
            )
            source_widget.valueChanged.connect(
                lambda val: self._sync_spinbox(target, val)
            )
            self._apply_tooltip(target, key)
            return target
        if isinstance(source_widget, QDoubleSpinBox):
            target = OutlineDoubleSpinBox()
            target.setRange(source_widget.minimum(), source_widget.maximum())
            target.setDecimals(source_widget.decimals())
            target.setSingleStep(source_widget.singleStep())
            target.setValue(source_widget.value())
            target.valueChanged.connect(
                lambda val: self._sync_spinbox(source_widget, val)
            )
            source_widget.valueChanged.connect(
                lambda val: self._sync_spinbox(target, val)
            )
            self._apply_tooltip(target, key)
            return target
        if isinstance(source_widget, QLineEdit):
            target = OutlineLineEdit()
            target.setText(source_widget.text())
            target.textChanged.connect(
                lambda text: self._sync_line_edit(source_widget, text)
            )
            source_widget.textChanged.connect(
                lambda text: self._sync_line_edit(target, text)
            )
            self._apply_tooltip(target, key)
            return target
        if isinstance(source_widget, QTextEdit):
            target = OutlineTextEdit()
            target.setPlainText(source_widget.toPlainText())
            target.textChanged.connect(
                lambda: self._sync_text_edit(source_widget, target.toPlainText())
            )
            source_widget.textChanged.connect(
                lambda: self._sync_text_edit(target, source_widget.toPlainText())
            )
            self._apply_tooltip(target, key)
            return target
        if isinstance(source_widget, QComboBox):
            target = QComboBox()
            for idx in range(source_widget.count()):
                target.addItem(source_widget.itemText(idx))
            target.setCurrentText(source_widget.currentText())
            target.currentTextChanged.connect(
                lambda text: self._sync_combo(source_widget, text)
            )
            source_widget.currentTextChanged.connect(
                lambda text: self._sync_combo(target, text)
            )
            self._apply_tooltip(target, key)
            return target
        fallback = QLabel("Unsupported")
        return fallback

    def _sync_checkbox(self, checkbox: QCheckBox, value: bool) -> None:
        if checkbox.isChecked() == value:
            return
        checkbox.blockSignals(True)
        checkbox.setChecked(value)
        checkbox.blockSignals(False)

    def _sync_spinbox(self, spinbox: QWidget, value: float) -> None:
        if isinstance(spinbox, (QSpinBox, QDoubleSpinBox)):
            if spinbox.value() == value:
                return
            spinbox.blockSignals(True)
            spinbox.setValue(value)
            spinbox.blockSignals(False)

    def _sync_line_edit(self, widget: QLineEdit, text: str) -> None:
        if widget.text() == text:
            return
        widget.blockSignals(True)
        widget.setText(text)
        widget.blockSignals(False)

    def _sync_text_edit(self, widget: QTextEdit, text: str) -> None:
        if widget.toPlainText() == text:
            return
        widget.blockSignals(True)
        widget.setPlainText(text)
        widget.blockSignals(False)

    def _sync_combo(self, combo: QComboBox, text: str) -> None:
        if combo.currentText() == text:
            return
        combo.blockSignals(True)
        combo.setCurrentText(text)
        combo.blockSignals(False)

    def _populate_basic_fields(self, step_id: str, form: QFormLayout) -> None:
        defaults = self.defaults[step_id]
        if step_id == "step1":
            self._add_text(step_id, form, "MODE", "Mode", defaults, read_only=True)
            self._add_checkbox(step_id, form, "ENABLE_PLOT", "Enable plot", defaults)
            self._add_choice_dropdown(
                step_id,
                form,
                "PLOT_SCALE_MODE",
                "Plot scale mode",
                defaults,
                ["fixed", "robust_auto"],
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_FIXED_UV",
                "Fixed plot range (±uV)",
                defaults,
                10,
                5000,
                is_float=True,
            )
            self._add_checkbox(
                step_id,
                form,
                "PLOT_REFERENCE_LINES",
                "Reference overlay (±25/50/100 uV)",
                defaults,
            )
            self._add_checkbox(step_id, form, "SAVE_RAW", "Save raw", defaults)
            save_raw_widget = self.fields[step_id].get("SAVE_RAW")
            if isinstance(save_raw_widget, QCheckBox):
                save_raw_widget.setEnabled(False)
            self._add_checkbox(
                step_id, form, "EVENT_MARKING_ENABLED", "Event marking", defaults
            )
            self._add_spin(
                step_id, form, "SAMPLING_RATE", "Sampling rate", defaults, 1, 4096
            )
            self._add_spin(
                step_id, form, "CHANNELS", "Channels", defaults, 1, 64, read_only=True
            )
        elif step_id == "infer":
            infer_subject_combo = QComboBox()
            infer_subject_combo.addItem("(current)")
            if self.current_project:
                infer_subject_combo.addItems(list_subjects(self.current_project))
            infer_subject_combo.setCurrentText("(current)")
            infer_subject_combo.currentTextChanged.connect(self._on_infer_subject_changed)
            self._apply_tooltip(infer_subject_combo, "infer_subject_override")
            form.addRow("Inference subject", infer_subject_combo)
            self.fields[step_id]["infer_subject_override"] = infer_subject_combo
            self.infer_subject_combo = infer_subject_combo
            self._add_file_picker(
                step_id,
                form,
                "model_path",
                "Model path",
                defaults,
                "Model (*.pt *.pth);;All Files (*)",
            )
            self._add_file_picker(
                step_id,
                form,
                "scaler_path",
                "Scaler path",
                defaults,
                "Scaler (*.save *.pkl);;All Files (*)",
            )
            self._add_text(step_id, form, "stream_name", "Stream name", defaults)
            self._add_text(step_id, form, "stream_type", "Stream type", defaults)
            self._add_spin(
                step_id,
                form,
                "window_sec",
                "Window sec",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "hop_sec",
                "Hop sec",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "target_fs",
                "Target FS",
                defaults,
                1,
                4096,
                is_float=True,
            )
            self._add_checkbox(
                step_id,
                form,
                "use_inference_engine",
                "Use MC-dropout inference engine",
                defaults,
            )
            self._add_spin(
                step_id,
                form,
                "mc_passes",
                "MC passes",
                defaults,
                1,
                100,
            )
            infer_engine_widget = self.fields[step_id].get("use_inference_engine")
            if isinstance(infer_engine_widget, QCheckBox):
                infer_engine_widget.toggled.connect(
                    lambda _checked: self._sync_infer_inference_engine_controls()
                )
            self._add_checkbox(step_id, form, "allow_drop", "Allow drop", defaults)
            self._add_checkbox(
                step_id,
                form,
                "LIVE_VIZ_ENABLED",
                "Emit Step 7 live model views",
                defaults,
            )
            self._add_spin(
                step_id,
                form,
                "LIVE_VIZ_FPS",
                "Step 7 live viz FPS",
                defaults,
                1,
                10,
                is_float=False,
            )
            self._add_checkbox(
                step_id,
                form,
                "enable_actuation",
                "Enable Robot Hand Actuation (DANGEROUS)",
                defaults,
            )
            self._add_checkbox(
                step_id,
                form,
                "modulate_actuation_speed",
                "Modulate actuation speed from confidence",
                defaults,
            )
            self._add_checkbox(
                step_id,
                form,
                "no_file_io",
                "Disable file outputs (max performance)",
                defaults,
            )
            self._sync_infer_inference_engine_controls()
        elif step_id == "step1b":
            self._add_spin(
                step_id,
                form,
                "WINDOW_SEC",
                "Window sec",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_text(
                step_id,
                form,
                "session_dir",
                "Session Directory (sessions/<session_id>)",
                defaults,
            )
            self._add_file_picker(
                step_id,
                form,
                "features",
                "Legacy features CSV",
                defaults,
                "CSV (*.csv);;All Files (*)",
            )
            self._add_file_picker(
                step_id,
                form,
                "events",
                "Legacy events CSV",
                defaults,
                "CSV (*.csv);;All Files (*)",
            )
            self._add_text(step_id, form, "subject_id", "Subject ID", defaults)
            self._add_spin(
                step_id,
                form,
                "target_fs",
                "Target FS",
                defaults,
                1,
                4096,
                is_float=True,
            )
            self._add_checkbox(step_id, form, "allow_gaps", "Allow gaps", defaults)
            self._add_checkbox(
                step_id, form, "allow_partial", "Allow partial sessions", defaults
            )
            self._add_checkbox(
                step_id, form, "ignore_misalignment", "Ignore misalignment", defaults
            )
        elif step_id == "train":
            self.train_session_dir_input = OutlineLineEdit()
            self.train_session_dir_input.setPlaceholderText("")
            self.train_session_dir_input.textChanged.connect(
                self._on_train_session_dir_changed
            )
            form.addRow("Session Dir (Step 2 override)", self.train_session_dir_input)
            self._add_file_picker(
                step_id,
                form,
                "npz",
                "Window NPZ",
                defaults,
                "NPZ (*.npz);;All Files (*)",
            )
            self._add_text(step_id, form, "subject_id", "Subject ID", defaults)
            self._add_spin(step_id, form, "epochs", "Epochs", defaults, 1, 1000)
            self._add_spin(step_id, form, "batch_size", "Batch size", defaults, 1, 1024)
            self._add_spin(
                step_id,
                form,
                "lr",
                "Learning rate",
                defaults,
                0,
                1,
                is_float=True,
                decimals=6,
            )
            self._add_file_picker(
                step_id,
                form,
                "save_model",
                "Save model",
                defaults,
                "Model (*.pt *.pth);;All Files (*)",
                mode="save",
            )
            self._add_file_picker(
                step_id,
                form,
                "save_scaler",
                "Save scaler",
                defaults,
                "Scaler (*.save *.pkl);;All Files (*)",
                mode="save",
            )
        else:
            for key, val in defaults.items():
                if isinstance(val, bool):
                    self._add_checkbox(step_id, form, key, key, defaults)
                elif isinstance(val, (int, float)):
                    self._add_spin(
                        step_id,
                        form,
                        key,
                        key,
                        defaults,
                        0,
                        100000,
                        is_float=isinstance(val, float),
                    )
                else:
                    self._add_text(step_id, form, key, key, defaults)

    def _populate_advanced_fields(self, step_id: str, form: QFormLayout) -> None:
        defaults = self.defaults[step_id]
        if step_id == "step1":
            self._add_spin(
                step_id,
                form,
                "RAW_SHARD_SAMPLES",
                "Raw shard samples",
                defaults,
                256,
                16384,
            )
            self._add_spin(
                step_id,
                form,
                "PROCESSING_QUEUE_MAXSIZE",
                "Processing queue max",
                defaults,
                128,
                100000,
            )
            self._add_spin(
                step_id,
                form,
                "RAW_QUEUE_MAXSIZE",
                "Raw queue max",
                defaults,
                128,
                100000,
            )
            self._add_spin(
                step_id,
                form,
                "RAW_SHARD_FLUSH_INTERVAL_S",
                "Raw shard flush interval (s)",
                defaults,
                0,
                30,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "MAX_BACKPRESSURE_S",
                "Backpressure abort (s)",
                defaults,
                0,
                60,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "QUEUE_PUT_TIMEOUT_S",
                "Queue put timeout (s)",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "LSL_RESOLVE_TIMEOUT",
                "LSL resolve timeout (s)",
                defaults,
                0,
                30,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "LSL_INLET_MAX_BUFLEN_SEC",
                "LSL inlet max buflen (s)",
                defaults,
                0,
                60,
                is_float=False,
            )
            self._add_spin(
                step_id,
                form,
                "LSL_INLET_MAX_CHUNKLEN",
                "LSL inlet max chunklen",
                defaults,
                0,
                256,
                is_float=False,
            )
            self._add_spin(
                step_id,
                form,
                "HEARTBEAT_INTERVAL_S",
                "Heartbeat interval (s)",
                defaults,
                0,
                60,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "NO_SAMPLE_TIMEOUT_S",
                "No sample timeout (s)",
                defaults,
                0,
                60,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "WRITE_STALL_TIMEOUT_S",
                "Write stall timeout (s)",
                defaults,
                0,
                60,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "WARMUP_SAMPLE_COUNT",
                "Warmup sample count",
                defaults,
                0,
                1000,
            )
            self._add_spin(
                step_id,
                form,
                "WARMUP_TIMEOUT_S",
                "Warmup timeout (s)",
                defaults,
                0,
                60,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "EVENT_FLUSH_INTERVAL_S",
                "Event flush interval (s)",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_checkbox(step_id, form, "ENABLE_ICA", "Enable ICA", defaults)
            self._add_spin(
                step_id,
                form,
                "ICA_WARMUP_S",
                "ICA warmup (s)",
                defaults,
                0,
                120,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "ICA_MIN_SAMPLES",
                "ICA min samples",
                defaults,
                0,
                100000,
            )
            self._add_spin(
                step_id,
                form,
                "ICA_MIN_VAR",
                "ICA min var",
                defaults,
                0,
                1,
                is_float=True,
                decimals=8,
            )
            self._add_text(
                step_id, form, "ICA_FAIL_POLICY", "ICA fail policy", defaults
            )
            self._add_spin(
                step_id,
                form,
                "ICA_MAX_RETRIES_PER_SESSION",
                "ICA max retries",
                defaults,
                0,
                10,
            )
            self._add_checkbox(
                step_id, form, "LOG_ICA_DIAGNOSTICS", "Log ICA diagnostics", defaults
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_FPS",
                "Plot FPS",
                defaults,
                1,
                120,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_DISPLAY_FS",
                "Plot display FS",
                defaults,
                1,
                512,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_WINDOW_SEC",
                "Plot window sec",
                defaults,
                0,
                30,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_ROBUST_WINDOW_SEC",
                "Plot robust window sec",
                defaults,
                0,
                30,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_ROBUST_EMA",
                "Plot robust EMA",
                defaults,
                0,
                1,
                is_float=True,
                decimals=3,
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_STARTUP_TIMEOUT_S",
                "Plot startup timeout (s)",
                defaults,
                0,
                30,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "PLOT_CHANNEL_SPACING_UV",
                "Plot channel spacing (uV)",
                defaults,
                0,
                1000,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "HARD_STOP_AFTER_UNHEALTHY_S",
                "Hard stop after unhealthy (s)",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_text(
                step_id,
                form,
                "REQUIRED_LSL_LABELS",
                "Required labels (CSV)",
                defaults,
            )
            self._add_checkbox(
                step_id,
                form,
                "REQUIRE_EXACTLY_4_CHANNELS",
                "Require exactly 4 channels",
                defaults,
            )
            self._add_text(step_id, form, "EVENT_KEYMAP", "Event keymap", defaults)
            self._add_checkbox(step_id, form, "init_only", "Init only", defaults)
        elif step_id == "infer":
            self._add_spin(
                step_id,
                form,
                "latency_threshold_ms",
                "Latency threshold (ms)",
                defaults,
                0,
                5000,
                is_float=True,
            )
            self._add_text(step_id, form, "latency_policy", "Latency policy", defaults)
            self._add_spin(
                step_id,
                form,
                "log_every",
                "Log every (s)",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_text(
                step_id, form, "bluetooth_target", "Bluetooth target", defaults
            )
            self._add_spin(
                step_id,
                form,
                "actuation_speed_gamma",
                "Actuation speed gamma",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_text(step_id, form, "subject_id", "Subject ID", defaults)
            self._add_text(step_id, form, "session_id", "Session ID", defaults)
            self._add_checkbox(step_id, form, "postprocess", "Enable postprocess", defaults)
            self._add_checkbox(step_id, form, "smoothing_enabled", "Smoothing enabled", defaults)
            self._add_choice_dropdown(
                step_id,
                form,
                "smoothing_method",
                "Smoothing method",
                defaults,
                ["vote", "ema"],
            )
            self._add_spin(
                step_id,
                form,
                "smoothing_window",
                "Smoothing window",
                defaults,
                1,
                50,
            )
            self._add_checkbox(step_id, form, "hysteresis_enabled", "Hysteresis enabled", defaults)
            self._add_spin(
                step_id,
                form,
                "hysteresis_frames",
                "Hysteresis frames",
                defaults,
                1,
                20,
            )
            self._add_slider(
                step_id,
                form,
                "threshold_action",
                "Action threshold",
                defaults,
                0,
                1,
                decimals=2,
            )
            self._add_slider(
                step_id,
                form,
                "threshold_finger",
                "Finger threshold",
                defaults,
                0,
                1,
                decimals=2,
            )
            self._add_checkbox(step_id, form, "adjacency_enabled", "Adjacency assist", defaults)
            self._add_slider(
                step_id,
                form,
                "hysteresis_margin",
                "Hysteresis margin",
                defaults,
                0,
                1,
                decimals=2,
            )
            self._add_slider(
                step_id,
                form,
                "finger_delta",
                "Finger delta",
                defaults,
                0,
                1,
                decimals=2,
            )
            self._add_choice_dropdown(
                step_id,
                form,
                "finger_mode",
                "Finger mode",
                defaults,
                ["raw", "smooth"],
            )
            self._add_slider(
                step_id,
                form,
                "uncertainty_base_threshold",
                "Uncertainty base threshold",
                defaults,
                0,
                1,
                decimals=2,
            )
            self._add_slider(
                step_id,
                form,
                "uncertainty_weight",
                "Uncertainty weight",
                defaults,
                0,
                2,
                decimals=2,
            )
            self._add_file_picker(
                step_id,
                form,
                "pred_log",
                "Prediction log (JSONL)",
                defaults,
                "JSONL (*.jsonl);;All Files (*)",
                mode="save",
            )
            self._sync_infer_inference_engine_controls()
        elif step_id == "step1b":
            self._add_spin(
                step_id,
                form,
                "WINDOW_SEC_DEFAULT",
                "Window sec (default)",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id, form, "STEP_SEC", "Step sec", defaults, 0, 10, is_float=True
            )
            self._add_spin(
                step_id, form, "PAD_SEC", "Pad sec", defaults, 0, 10, is_float=True
            )
            self._add_spin(
                step_id,
                form,
                "GAP_THRESHOLD_SEC",
                "Gap threshold",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_text(step_id, form, "DEDUP_POLICY", "Dedupe policy", defaults)
            self._add_text(
                step_id, form, "INTERPOLATION_POLICY", "Interpolation policy", defaults
            )
            self._add_checkbox(
                step_id,
                form,
                "LABEL_GATED",
                "Label gated (OPEN/CLOSE+NONE invalid)",
                defaults,
            )
            self._add_choice_dropdown(
                step_id,
                form,
                "REST_POLICY",
                "REST policy",
                defaults,
                ["label_gated", "rest_by_exclusion"],
            )
            self._add_spin(
                step_id,
                form,
                "KEEP_BASELINE_REST_EVENTS",
                "Keep baseline rest",
                defaults,
                0,
                50,
            )
            self._add_spin(
                step_id,
                form,
                "MIN_OVERLAP_RATIO",
                "Min overlap ratio",
                defaults,
                0,
                1,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "GUARD_BAND_SEC",
                "Guard band sec",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "ARTIFACT_MIN_OVERLAP_FRAC",
                "Artifact overlap",
                defaults,
                0,
                1,
                is_float=True,
            )
            self._add_text(step_id, form, "OUT_FILE", "Output CSV", defaults)
            self._add_text(step_id, form, "OUT_NPZ", "Output NPZ", defaults)
        elif step_id == "train":
            self._add_spin(step_id, form, "seed", "Seed", defaults, 0, 1_000_000)
            self._add_spin(
                step_id,
                form,
                "loss_action_weight",
                "Finger loss weight",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "rest_weight",
                "REST class weight",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_text(
                step_id,
                form,
                "action_weights",
                "Action weights (CSV/JSON)",
                defaults,
            )
            self._add_spin(
                step_id,
                form,
                "rest_finger_loss_weight",
                "REST finger loss weight",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_text(
                step_id,
                form,
                "finger_weights",
                "Finger weights (CSV/JSON)",
                defaults,
            )
            self._add_choice_dropdown(
                step_id,
                form,
                "rest_balance_mode",
                "REST balance mode",
                defaults,
                ["none", "session_equalized", "core_event_equalized"],
            )
            self._add_spin(
                step_id,
                form,
                "test_size",
                "Test split size",
                defaults,
                0,
                1,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "calibration_size",
                "Calibration holdout",
                defaults,
                0,
                0.99,
                is_float=True,
            )
            self._add_choice_dropdown(
                step_id,
                form,
                "split_mode",
                "Split mode",
                defaults,
                ["group_trial", "holdout_session"],
            )
            self._add_spin(
                step_id,
                form,
                "purge_seconds",
                "Purge seconds",
                defaults,
                0,
                60,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "hop_seconds",
                "Hop seconds (auto=0)",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "window_idx_leak_threshold",
                "Window index leak threshold",
                defaults,
                0,
                1,
                is_float=True,
            )
            self._add_checkbox(
                step_id, form, "strict_leakage", "Strict leakage", defaults
            )
            self._add_checkbox(
                step_id, form, "non_rest_only", "Train non-REST only", defaults
            )
            self._add_choice_dropdown(
                step_id,
                form,
                "device",
                "Device",
                defaults,
                ["auto", "cpu", "cuda", "mps"],
            )
            self._add_spin(
                step_id, form, "num_workers", "DataLoader workers", defaults, 0, 64
            )
            self._add_checkbox(step_id, form, "pin_memory", "Pin memory", defaults)
            self._add_text(step_id, form, "run_dir", "Run dir override", defaults)
            self._add_text(
                step_id, form, "save_preds", "Save predictions", defaults
            )
            self._add_text(
                step_id,
                form,
                "save_temperature",
                "Save temperature scaling",
                defaults,
            )
        else:
            for key, val in defaults.items():
                if key in self.fields.get(step_id, {}):
                    continue
                if isinstance(val, bool):
                    self._add_checkbox(step_id, form, key, key, defaults)
                elif isinstance(val, (int, float)):
                    self._add_spin(
                        step_id,
                        form,
                        key,
                        key,
                        defaults,
                        0,
                        100000,
                        is_float=isinstance(val, float),
                    )
                else:
                    self._add_text(step_id, form, key, key, defaults)
        self._add_dynamic_fields(step_id, form)

    def _add_dynamic_fields(self, step_id: str, form: QFormLayout) -> None:
        script_key = self.step_script_key.get(step_id)
        info = self.scripts.get(script_key) if script_key else None
        if not info:
            return
        configured_keys = set(self.defaults.get(step_id, {}).keys())
        ignored = {
            "DEVICE",
            "ROOT_DIR",
            "ROOT",
            "PROJECT_ROOT",
            "SESSION_STATE_DIR",
            "config",
            "ENABLE_ACTUATION",
        }
        if step_id == "train":
            ignored.update(
                {
                    "BATCH_SIZE",
                    "EPOCHS",
                    "LR",
                    "SEED",
                    "LOSS_ACTION_WEIGHT",
                    "REST_WEIGHT",
                    "DEFAULT_MODEL",
                    "DEFAULT_SCALER",
                    "DEFAULT_PREDS",
                    "DEFAULT_NPZ",
                    "MAX_SEARCH_DEPTH",
                }
            )
        if step_id == "step1b":
            ignored.update(
                {
                    "DEFAULT_SUBJECT_ID",
                    "LEGACY_EVENT_FILE",
                    "LEGACY_RAW_FILE",
                    "REST_SUBSAMPLE_PROB",
                    "REST_SUBSAMPLE_SEED",
                    "SEED",
                    "SOURCE_FS_DEFAULT",
                    "TARGET_FS_DEFAULT",
                }
            )
        if step_id == "step1":
            ignored.update(
                {
                    "DEFAULT_SUBJECT_ID",
                    "DEFAULT_EVENT_KEYMAP",
                    "GENDER",
                    "AGE",
                    "PLOT_FPS",
                    "PLOT_DISPLAY_FS",
                    "PLOT_FIXED_YLIM",
                    "PLOT_ROBUST_WINDOW_SEC",
                    "PLOT_ROBUST_EMA",
                    "PLOT_REFERENCE_OVERLAY",
                    "PLOT_WINDOW_SEC",
                    "PLOT_STARTUP_TIMEOUT_S",
                    "PLOT_CHANNEL_SPACING_UV",
                    "LSL_RESOLVE_TIMEOUT",
                    "LSL_INLET_MAX_BUFLEN_SEC",
                    "LSL_INLET_MAX_CHUNKLEN",
                    "RAW_SHARD_FLUSH_INTERVAL_S",
                    "HEARTBEAT_INTERVAL_S",
                    "NO_SAMPLE_TIMEOUT_S",
                    "WRITE_STALL_TIMEOUT_S",
                    "WARMUP_SAMPLE_COUNT",
                    "WARMUP_TIMEOUT_S",
                    "EVENT_FLUSH_INTERVAL_S",
                    "RAW_FLAG_NONFINITE",
                    "INTEGRITY_GAP_TOLERANCE_MULT",
                    "ALLOW_DROP",
                    "subject_id",
                    "force_new_session",
                    "SESSION_ID_OVERRIDE",
                    "session_id",
                }
            )
        existing = self.fields.get(step_id, {})
        for name, value in sorted(info.constants.items()):
            if name in ignored or name in existing:
                continue
            if not isinstance(name, str) or not name.isupper():
                continue
            self.defaults[step_id].setdefault(name, value)
            if isinstance(value, bool):
                self._add_checkbox(
                    step_id, form, name, f"{name} (auto)", self.defaults[step_id]
                )
            elif isinstance(value, (int, float)):
                self._add_spin(
                    step_id,
                    form,
                    name,
                    f"{name} (auto)",
                    self.defaults[step_id],
                    0,
                    100000,
                    is_float=isinstance(value, float),
                )
            else:
                self._add_text(
                    step_id, form, name, f"{name} (auto)", self.defaults[step_id]
                )

        for arg in info.args:
            dest = arg.dest
            if dest not in configured_keys:
                continue
            if not dest or dest in ignored or dest in existing:
                continue
            if dest == "config":
                continue
            default = self.defaults[step_id].get(dest, arg.default)
            if default is None and arg.action in {"store_true", "store_false"}:
                default = False if arg.action == "store_true" else True
            self.defaults[step_id].setdefault(dest, default)
            label = f"{dest} (auto)"

            if arg.choices:
                self._add_choice_dropdown(
                    step_id, form, dest, label, self.defaults[step_id], arg.choices
                )
            elif arg.action in {"store_true", "store_false"} or isinstance(
                default, bool
            ):
                self._add_checkbox(step_id, form, dest, label, self.defaults[step_id])
            elif arg.arg_type in {"int"} or isinstance(default, int):
                self._add_spin(
                    step_id,
                    form,
                    dest,
                    label,
                    self.defaults[step_id],
                    0,
                    100000,
                    is_float=False,
                )
            elif arg.arg_type in {"float"} or isinstance(default, float):
                self._add_spin(
                    step_id,
                    form,
                    dest,
                    label,
                    self.defaults[step_id],
                    0,
                    100000,
                    is_float=True,
                )
            else:
                self._add_text(step_id, form, dest, label, self.defaults[step_id])
            widget = self.fields[step_id].get(dest)
            if widget:
                self._apply_tooltip(widget, dest, arg.help)

    def _friendly_label(self, step_id: str, key: str) -> str:
        if step_id == "infer":
            labels = {
                "model_path": "Model path",
                "scaler_path": "Scaler path",
                "out_dir": "Output directory",
                "stream_name": "Stream name",
                "stream_type": "Stream type",
                "window_sec": "Window sec",
                "hop_sec": "Hop sec",
                "target_fs": "Target FS",
                "use_inference_engine": "Use MC-dropout inference engine",
                "mc_passes": "MC passes",
                "uncertainty_base_threshold": "Uncertainty base threshold",
                "uncertainty_weight": "Uncertainty weight",
                "allow_drop": "Allow drop",
                "LIVE_VIZ_ENABLED": "Emit Step 7 live model views",
                "LIVE_VIZ_FPS": "Step 7 live viz FPS",
                "latency_threshold_ms": "Latency threshold (ms)",
                "latency_policy": "Latency policy",
                "log_every": "Log every (s)",
                "enable_actuation": "Enable robot hand actuation",
                "modulate_actuation_speed": "Modulate actuation speed from confidence",
                "actuation_speed_gamma": "Actuation speed gamma",
                "bluetooth_target": "Bluetooth target",
                "no_file_io": "Disable file outputs (max performance)",
                "subject_id": "Subject ID",
                "session_id": "Session ID",
                "postprocess": "Enable postprocess",
                "smoothing_enabled": "Smoothing enabled",
                "smoothing_method": "Smoothing method",
                "smoothing_window": "Smoothing window",
                "hysteresis_enabled": "Hysteresis enabled",
                "hysteresis_frames": "Hysteresis frames",
                "threshold_action": "Action threshold",
                "threshold_finger": "Finger threshold",
                "adjacency_enabled": "Adjacency assist",
                "hysteresis_margin": "Hysteresis margin",
                "finger_delta": "Finger delta",
                "finger_mode": "Finger mode",
                "pred_log": "Prediction log (JSONL)",
            }
            return labels.get(key, key)
        if step_id == "train":
            labels = {
                "session_dir": "Session Dir (Step 2 override)",
                "npz": "Window NPZ",
                "subject_id": "Subject ID",
                "epochs": "Epochs",
                "batch_size": "Batch size",
                "lr": "Learning rate",
                "save_model": "Save model",
                "save_scaler": "Save scaler",
                "seed": "Seed",
                "loss_action_weight": "Finger loss weight",
                "rest_weight": "REST class weight",
                "action_weights": "Action weights (CSV/JSON)",
                "rest_finger_loss_weight": "REST finger loss weight",
                "finger_weights": "Finger weights (CSV/JSON)",
                "rest_balance_mode": "REST balance mode",
                "test_size": "Test split size",
                "calibration_size": "Calibration holdout",
                "split_mode": "Split mode",
                "purge_seconds": "Purge seconds",
                "hop_seconds": "Hop seconds (auto=0)",
                "window_idx_leak_threshold": "Window index leak threshold",
                "strict_leakage": "Strict leakage",
                "non_rest_only": "Train non-REST only",
                "device": "Device",
                "num_workers": "DataLoader workers",
                "pin_memory": "Pin memory",
                "run_dir": "Run dir override",
                "save_preds": "Save predictions",
                "save_temperature": "Save temperature scaling",
            }
            return labels.get(key, key)
        if step_id == "step1":
            labels = {
                "MODE": "Mode",
                "ENABLE_PLOT": "Enable plot",
                "PLOT_SCALE_MODE": "Plot scale mode",
                "PLOT_FIXED_UV": "Fixed plot range (±uV)",
                "PLOT_REFERENCE_LINES": "Reference overlay (±25/50/100 uV)",
                "SAVE_RAW": "Save raw",
                "EVENT_MARKING_ENABLED": "Event marking",
                "EVENT_KEYMAP": "Event keymap",
                "SAMPLING_RATE": "Sampling rate",
                "CHANNELS": "Channels",
                "RAW_SHARD_SAMPLES": "Raw shard samples",
                "RAW_SHARD_FLUSH_INTERVAL_S": "Raw shard flush interval (s)",
                "PROCESSING_QUEUE_MAXSIZE": "Processing queue max",
                "RAW_QUEUE_MAXSIZE": "Raw queue max",
                "MAX_BACKPRESSURE_S": "Backpressure abort (s)",
                "QUEUE_PUT_TIMEOUT_S": "Queue put timeout (s)",
                "LSL_RESOLVE_TIMEOUT": "LSL resolve timeout (s)",
                "LSL_INLET_MAX_BUFLEN_SEC": "LSL inlet max buflen (s)",
                "LSL_INLET_MAX_CHUNKLEN": "LSL inlet max chunklen",
                "HEARTBEAT_INTERVAL_S": "Heartbeat interval (s)",
                "NO_SAMPLE_TIMEOUT_S": "No sample timeout (s)",
                "WRITE_STALL_TIMEOUT_S": "Write stall timeout (s)",
                "WARMUP_SAMPLE_COUNT": "Warmup sample count",
                "WARMUP_TIMEOUT_S": "Warmup timeout (s)",
                "EVENT_FLUSH_INTERVAL_S": "Event flush interval (s)",
                "ENABLE_ICA": "Enable ICA",
                "ICA_WARMUP_S": "ICA warmup (s)",
                "ICA_MIN_SAMPLES": "ICA min samples",
                "ICA_MIN_VAR": "ICA min var",
                "ICA_FAIL_POLICY": "ICA fail policy",
                "ICA_MAX_RETRIES_PER_SESSION": "ICA max retries",
                "LOG_ICA_DIAGNOSTICS": "Log ICA diagnostics",
                "HARD_STOP_AFTER_UNHEALTHY_S": "Hard stop after unhealthy (s)",
                "REQUIRED_LSL_LABELS": "Required labels (CSV)",
                "REQUIRE_EXACTLY_4_CHANNELS": "Require exactly 4 channels",
                "init_only": "Init only",
                "PLOT_FPS": "Plot FPS",
                "PLOT_DISPLAY_FS": "Plot display FS",
                "PLOT_WINDOW_SEC": "Plot window sec",
                "PLOT_ROBUST_WINDOW_SEC": "Plot robust window sec",
                "PLOT_ROBUST_EMA": "Plot robust EMA",
                "PLOT_STARTUP_TIMEOUT_S": "Plot startup timeout (s)",
                "PLOT_CHANNEL_SPACING_UV": "Plot channel spacing (uV)",
            }
            return labels.get(key, key)
        if step_id == "step1b":
            labels = {
                "session_dir": "Session Directory (sessions/<session_id>)",
                "features": "Legacy features CSV",
                "events": "Legacy events CSV",
                "subject_id": "Subject ID",
                "target_fs": "Target FS",
                "allow_gaps": "Allow gaps",
                "allow_partial": "Allow partial sessions",
                "ignore_misalignment": "Ignore misalignment",
                "WINDOW_SEC": "Window sec",
                "WINDOW_SEC_DEFAULT": "Window sec (default)",
                "STEP_SEC": "Step sec",
                "PAD_SEC": "Pad sec",
                "GAP_THRESHOLD_SEC": "Gap threshold",
                "DEDUP_POLICY": "Dedupe policy",
                "INTERPOLATION_POLICY": "Interpolation policy",
                "LABEL_GATED": "Label gated (OPEN/CLOSE+NONE invalid)",
                "REST_POLICY": "REST policy",
                "KEEP_BASELINE_REST_EVENTS": "Keep baseline rest",
                "MIN_OVERLAP_RATIO": "Min overlap ratio",
                "GUARD_BAND_SEC": "Guard band sec",
                "ARTIFACT_MIN_OVERLAP_FRAC": "Artifact overlap",
                "OUT_FILE": "Output CSV",
                "OUT_NPZ": "Output NPZ",
            }
            return labels.get(key, key)
        return key

    def _add_checkbox(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
    ) -> None:
        cb = QCheckBox()
        cb.setChecked(bool(defaults.get(key, False)))
        self._apply_tooltip(cb, key)
        form.addRow(label, cb)
        self.fields[step_id][key] = cb

    def _add_spin(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
        min_val: float,
        max_val: float,
        is_float: bool = False,
        read_only: bool = False,
        decimals: int = 3,
    ) -> None:
        if is_float:
            box: QWidget = OutlineDoubleSpinBox()
            box.setProperty("is_float", True)
            box.setDecimals(decimals)
            box.setSingleStep(0.01)
        else:
            box = OutlineSpinBox()
        if isinstance(box, QDoubleSpinBox):
            box.setRange(float(min_val), float(max_val))
            box.setValue(float(defaults.get(key, 0.0) or 0.0))
        else:
            box.setRange(int(min_val), int(max_val))
            box.setValue(int(defaults.get(key, 0) or 0))
        box.setEnabled(not read_only)
        self._apply_tooltip(box, key)
        form.addRow(label, box)
        self.fields[step_id][key] = box

    def _add_text(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
        read_only: bool = False,
    ) -> None:
        line = OutlineLineEdit()
        val = defaults.get(key, "")
        line.setText("" if val is None else str(val))
        line.setReadOnly(read_only)
        self._apply_tooltip(line, key)
        form.addRow(label, line)
        self.fields[step_id][key] = line

    def _add_file_picker(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
        pattern: str,
        mode: str = "open",
    ) -> None:
        line = OutlineLineEdit()
        val = defaults.get(key, "")
        line.setText("" if val is None else str(val))
        btn = QPushButton("Browse")
        btn.clicked.connect(lambda: self._browse_path(line, pattern, label, mode=mode))
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(line)
        row.addWidget(btn)
        container = QWidget()
        container.setLayout(row)
        self._apply_tooltip(line, key)
        form.addRow(label, container)
        self.fields[step_id][key] = line

    def _add_slider(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
        min_val: float,
        max_val: float,
        decimals: int = 2,
    ) -> None:
        val = float(defaults.get(key, 0.0) or 0.0)
        slider = FloatSlider(min_val, max_val, val, decimals=decimals, parent=self)
        self._apply_tooltip(slider, key)
        form.addRow(label, slider)
        self.fields[step_id][key] = slider

    def _add_int_dropdown(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
        values: list[int],
    ) -> None:
        combo = QComboBox()
        combo.setProperty("value_type", "int")
        for val in values:
            combo.addItem(str(val))
        default = defaults.get(key)
        if default is not None:
            if combo.findText(str(default)) < 0:
                combo.addItem(str(default))
            combo.setCurrentText(str(default))
        self._apply_tooltip(combo, key)
        form.addRow(label, combo)
        self.fields[step_id][key] = combo

    def _add_choice_dropdown(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
        choices: list[Any],
    ) -> None:
        combo = QComboBox()
        choice_values = []
        for val in choices:
            choice_values.append(val)
            combo.addItem(str(val))
        default = defaults.get(key)
        if default is not None:
            if combo.findText(str(default)) < 0:
                combo.addItem(str(default))
            combo.setCurrentText(str(default))
        value_type = None
        for val in choice_values:
            if isinstance(val, bool):
                value_type = "bool"
                break
            if isinstance(val, int):
                value_type = "int"
                break
            if isinstance(val, float):
                value_type = "float"
                break
        if value_type:
            combo.setProperty("value_type", value_type)
        self._apply_tooltip(combo, key)
        form.addRow(label, combo)
        self.fields[step_id][key] = combo

    def _add_timebase_dropdown(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
    ) -> None:
        combo = QComboBox()
        combo.addItem(TIMEBASE_VERSION)
        combo.setCurrentText(
            str(defaults.get(key, TIMEBASE_VERSION) or TIMEBASE_VERSION)
        )
        combo.setEnabled(False)
        self._apply_tooltip(combo, key)
        form.addRow(label, combo)
        self.fields[step_id][key] = combo

    def _add_editable_combo(
        self,
        step_id: str,
        form: QFormLayout,
        key: str,
        label: str,
        defaults: Dict[str, Any],
        options: list[str],
    ) -> None:
        combo = QComboBox()
        combo.setEditable(True)
        for option in options:
            combo.addItem(option)
        default = defaults.get(key)
        if default:
            if combo.findText(str(default)) < 0:
                combo.addItem(str(default))
            combo.setCurrentText(str(default))
        self._apply_tooltip(combo, key)
        form.addRow(label, combo)
        self.fields[step_id][key] = combo

    def _apply_tooltip(
        self, widget: QWidget, key: str, fallback: Optional[str] = None
    ) -> None:
        tip = TOOLTIPS.get(key) or fallback
        if not tip:
            return
        if isinstance(widget, FloatSlider):
            widget.setToolTip(tip)
            widget.slider.setToolTip(tip)
            widget.spin.setToolTip(tip)
            return
        widget.setToolTip(tip)

    def _refresh_projects(self) -> None:
        self.project_combo.blockSignals(True)
        self.project_combo.clear()
        self.project_combo.addItems(["-"] + list_projects())
        self.project_combo.blockSignals(False)

    def _create_project(self) -> None:
        name = self.project_name_input.text().strip()
        if not name:
            return
        ensure_project(name)
        self.project_name_input.clear()
        self._refresh_projects()
        self._open_project(name)

    def _open_project(self, name: str) -> None:
        if name == "-" or not name:
            return
        self.current_project = name
        self._set_project_label(f"Project: {name}")
        self._refresh_subjects()

    def _refresh_subjects(self) -> None:
        self.subject_combo.blockSignals(True)
        self.subject_combo.clear()
        if self.current_project:
            self.subject_combo.addItems(["-"] + list_subjects(self.current_project))
        else:
            self.subject_combo.addItem("-")
        self.subject_combo.blockSignals(False)
        self._refresh_infer_subjects()

    def _refresh_infer_subjects(self) -> None:
        if not hasattr(self, "infer_subject_combo"):
            return
        combo = self.infer_subject_combo
        current = combo.currentText().strip()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("(current)")
        if self.current_project:
            combo.addItems(list_subjects(self.current_project))
        if current and combo.findText(current) >= 0:
            combo.setCurrentText(current)
        else:
            combo.setCurrentText("(current)")
        combo.blockSignals(False)

    def _infer_subject_override(self) -> Optional[str]:
        combo = getattr(self, "infer_subject_combo", None)
        if not isinstance(combo, QComboBox):
            return None
        value = combo.currentText().strip()
        if not value or value in {"(current)", "-"}:
            return None
        return value

    def _latest_session_for_subject(self, subject_id: str) -> Optional[Path]:
        if not self.current_project:
            return None
        subject_dir = subject_root(self.current_project, subject_id)
        sessions_root = subject_dir / "sessions"
        return _latest_dir_by_mtime(sessions_root)

    def _on_infer_subject_changed(self, _value: str) -> None:
        subject_id = self._infer_subject_override() or self.current_subject
        if not self.current_project or not subject_id:
            return
        subject_dir = subject_root(self.current_project, subject_id)
        session_dir = self._latest_session_for_subject(subject_id)
        run_dir = resolve_latest_run_dir(session_dir) if session_dir else None
        if not run_dir:
            return
        infer_model_widget = self.fields.get("infer", {}).get("model_path")
        infer_scaler_widget = self.fields.get("infer", {}).get("scaler_path")
        self._maybe_autofill_text(
            infer_model_widget,
            str(run_dir / "finger_action_model.pt"),
            key="infer.model_path",
            legacy_values={"finger_action_model.pt", "models/finger_action_model.pt"},
        )
        self._maybe_autofill_text(
            infer_scaler_widget,
            str(run_dir / "scaler.npz"),
            key="infer.scaler_path",
            legacy_values={"scaler.npz"},
        )

    def _select_subject(self, subject_id: str) -> None:
        if subject_id == "-" or not subject_id:
            return
        self.current_subject = subject_id
        self._set_subject_label(f"Subject: {subject_id}")
        if self.current_project:
            subject_dir = subject_root(self.current_project, subject_id)
            ensure_subject_dirs(subject_dir)
            self._ensure_default_configs(subject_dir)
        self._auto_select_latest_session_for_subject()
        self._auto_fill_paths()
        self._seed_stream_name_input()

    def _edit_subject(self) -> None:
        if not self.current_project:
            QMessageBox.warning(self, "Project Required", "Select a project first.")
            return
        subject_id = self.subject_combo.currentText()
        subject_dir = (
            subject_root(self.current_project, subject_id)
            if subject_id and subject_id != "-"
            else None
        )
        existing_info = None
        if subject_dir and subject_meta_path(subject_dir).exists():
            try:
                payload = json.loads(subject_meta_path(subject_dir).read_text())
                existing_info = SubjectInfo(**payload)
            except Exception:
                existing_info = SubjectInfo(subject_id=subject_id)
        dialog = SubjectDialog(self, existing_info)
        if dialog.exec() != QDialog.Accepted:
            return
        info = dialog.info()
        if not info.subject_id:
            QMessageBox.warning(self, "Missing Subject", "Subject ID is required.")
            return
        subject_dir = subject_root(self.current_project, info.subject_id)
        ensure_subject_dirs(subject_dir)
        subject_meta_path(subject_dir).write_text(json.dumps(info.__dict__, indent=2))
        self._refresh_subjects()
        self._select_subject(info.subject_id)

    def _ensure_default_configs(self, subject_dir: Path) -> None:
        if not self.current_project or not self.current_subject:
            return
        defaults = {
            "step1": default_step1_settings(),
            "step1b": default_step1b_settings(),
            "preprocess": default_preprocess_settings(),
            "train": default_train_settings(),
            "infer": default_infer_settings(),
            "export": default_export_settings(),
        }
        for name, settings in defaults.items():
            path = subject_dir / "config" / f"{name}.json"
            if path.exists():
                continue
            config = build_config(
                project_name=self.current_project,
                subject_id=self.current_subject,
                session_id="PENDING",
                settings=settings,
                timebase_version=TIMEBASE_VERSION,
            )
            write_json(path, config.to_dict())

    def _auto_fill_paths(self) -> None:
        if not self.current_project or not self.current_subject:
            return
        subject_dir = subject_root(self.current_project, self.current_subject)
        # Default session root to the canonical project layout so Step 1 never falls back to data/raw.
        sessions_root = subject_dir / "sessions"
        if getattr(self, "session_root_input", None) and not self.session_root_input.text().strip():
            self.session_root_input.setText(str(sessions_root))
        events_dir = subject_dir / "events"
        features_dir = subject_dir / "features"
        preferred_events = None
        preferred_features = None
        session_dir_value = (
            self.session_dir_input.text().strip()
            if getattr(self, "session_dir_input", None)
            else ""
        )
        session_dir = Path(session_dir_value) if session_dir_value else None
        if session_dir and session_dir.exists():
            candidate_events = session_dir / "events" / "events.jsonl"
            if not candidate_events.exists():
                candidate_events = session_dir / "events" / "events.json"
            if candidate_events.exists():
                preferred_events = candidate_events
        if self.current_session_backend:
            candidate_events_jsonl = (
                events_dir
                / f"{self.current_subject}_{self.current_session_backend}_events.jsonl"
            )
            candidate_events_csv = (
                events_dir
                / f"{self.current_subject}_{self.current_session_backend}_events.csv"
            )
            candidate_features = (
                features_dir
                / f"{self.current_subject}_{self.current_session_backend}_eeg_features.csv"
            )
            if candidate_events_jsonl.exists():
                preferred_events = candidate_events_jsonl
            elif candidate_events_csv.exists():
                preferred_events = candidate_events_csv
            if candidate_features.exists():
                preferred_features = candidate_features

        latest_events = preferred_events or self._latest_subject_file(
            events_dir, f"{self.current_subject}_*_events.jsonl"
        )
        if latest_events is None:
            latest_events = self._latest_subject_file(
                events_dir, f"{self.current_subject}_*_events.csv"
            )
        latest_features = preferred_features or self._latest_subject_file(
            features_dir, f"{self.current_subject}_*_eeg_features.csv"
        )
        self.event_events_path.setText(str(latest_events) if latest_events else "")
        self.event_features_path.setText(str(latest_features) if latest_features else "")
        self.diag_events_path.setText(
            str(latest_events) if latest_events else str(events_dir)
        )
        self.diag_features_path.setText(
            str(latest_features) if latest_features else str(features_dir)
        )
        self._update_resume_ui()
        self._autofill_replay_paths()

    def _latest_subject_file(self, base: Path, pattern: str) -> Optional[Path]:
        if not base.exists():
            return None
        candidates = sorted(base.glob(pattern))
        return candidates[-1] if candidates else None


    def _resolve_effective_session_dir(self, step_id: Optional[str] = None) -> Optional[Path]:
        override_value = ""
        if step_id == "train" and hasattr(self, "train_session_dir_input"):
            override_value = self.train_session_dir_input.text().strip()
        value = override_value or (
            self.session_dir_input.text().strip() if hasattr(self, "session_dir_input") else ""
        )
        if not value:
            return None
        p = Path(value).expanduser()
        if p.exists():
            try:
                return p.resolve()
            except Exception:
                return p
        return p

    def _existing_file_path(self, value: str) -> Optional[Path]:
        text = str(value or "").strip()
        if not text:
            return None
        path = Path(text).expanduser()
        try:
            if path.exists():
                path = path.resolve()
        except Exception:
            pass
        if path.exists() and path.is_file():
            return path
        return None

    def _resolve_event_tools_session_dir(self) -> Optional[Path]:
        override = ""
        if hasattr(self, "event_session_dir_override"):
            override = self.event_session_dir_override.text().strip()
        if override:
            path = Path(override).expanduser()
            try:
                if path.exists():
                    return path.resolve()
            except Exception:
                pass
            return path
        session_dir = self._resolve_effective_session_dir(step_id=None)
        if session_dir is not None and session_dir.exists():
            return session_dir
        if self.current_project and self.current_subject:
            subject_dir = subject_root(self.current_project, self.current_subject)
            return self._resolve_session_dir_for_current(subject_dir)
        return None

    def _event_file_for_session(self, session_dir: Optional[Path]) -> Optional[Path]:
        if session_dir is None:
            return None
        for rel in ("events/events.jsonl", "events/events.json", "events/events.csv"):
            candidate = session_dir / rel
            if candidate.exists() and candidate.is_file():
                return candidate
        return None

    def _propagate_session_dir_autofill(
        self, prev_value: Optional[str], new_value: str
    ) -> None:
        if not hasattr(self, "train_session_dir_input"):
            return
        current = self.train_session_dir_input.text().strip()
        if not current or (prev_value and current == prev_value):
            self.train_session_dir_input.setText(new_value)

    def _maybe_autofill_text(
        self,
        widget: Optional[QLineEdit],
        value: str,
        *,
        key: str,
        legacy_values: set[str],
    ) -> None:
        if not isinstance(widget, QLineEdit):
            return
        current = widget.text().strip()
        previous_auto = self._auto_field_values.get(key)
        if should_replace_autofilled_text(current, previous_auto, legacy_values):
            widget.setText(value)
            self._auto_field_values[key] = value
            return
        if current == value:
            self._auto_field_values[key] = value

    def _auto_select_latest_session_for_subject(self) -> None:
        if not self.current_project or not self.current_subject:
            return
        if not hasattr(self, "session_dir_input"):
            return
        subject_dir = subject_root(self.current_project, self.current_subject)
        sessions_root = subject_dir / "sessions"
        latest = _latest_dir_by_mtime(sessions_root)
        if latest is None:
            self._auto_session_dir_value = None
            self._update_projects_selected_session_label(force_none=True)
            return
        prev_auto = self._auto_session_dir_value
        self.session_dir_input.setText(str(latest))
        self._auto_session_dir_value = str(latest)
        self._propagate_session_dir_autofill(prev_value=prev_auto, new_value=str(latest))
        self._update_projects_selected_session_label()
        self._autofill_dependent_paths_from_session_dir()

    def _update_projects_selected_session_label(self, force_none: bool = False) -> None:
        if not hasattr(self, "projects_selected_session_value"):
            return
        if force_none:
            self.projects_selected_session_value.setText("(none)")
            return
        effective = self._resolve_effective_session_dir(step_id="train")
        if not effective:
            self.projects_selected_session_value.setText("(none)")
            return
        display_path = effective.expanduser()
        if display_path.exists():
            try:
                display_path = display_path.resolve()
            except Exception:
                pass
        elif not display_path.is_absolute():
            try:
                display_path = display_path.absolute()
            except Exception:
                pass
        self.projects_selected_session_value.setText(str(display_path))

    def _autofill_dependent_paths_from_session_dir(self) -> None:
        global_session = self._resolve_effective_session_dir(step_id=None)
        train_session = self._resolve_effective_session_dir(step_id="train")

        train_source = None
        if train_session and train_session.exists():
            train_source = train_session
        elif global_session and global_session.exists():
            train_source = global_session

        if train_source:
            train_layout = SessionLayout(train_source)
            train_npz = str(train_layout.windows_npz)
            train_npz_widget = self.fields.get("train", {}).get("npz")
            self._maybe_autofill_text(
                train_npz_widget,
                train_npz,
                key="train.npz",
                legacy_values={"eeg_windows.npz"},
            )

        if not global_session or not global_session.exists():
            self._autofill_replay_paths()
            return

        infer_model_widget = self.fields.get("infer", {}).get("model_path")
        infer_scaler_widget = self.fields.get("infer", {}).get("scaler_path")

        run_dir = resolve_latest_run_dir(global_session)
        if run_dir:
            model_path = str(run_dir / "finger_action_model.pt")
            scaler_path = str(run_dir / "scaler.npz")
            self._maybe_autofill_text(
                infer_model_widget,
                model_path,
                key="infer.model_path",
                legacy_values={"finger_action_model.pt", "models/finger_action_model.pt"},
            )
            self._maybe_autofill_text(
                infer_scaler_widget,
                scaler_path,
                key="infer.scaler_path",
                legacy_values={"scaler.npz"},
            )
        self._autofill_replay_paths(session_dir_override=global_session)

    def _autofill_replay_paths(
        self, *, session_dir_override: Optional[Path] = None
    ) -> None:
        if not hasattr(self, "replay_npz_path"):
            return
        if not self.current_project or not self.current_subject:
            return
        subject_dir = subject_root(self.current_project, self.current_subject)

        npz_path: Optional[Path] = None
        if session_dir_override and session_dir_override.exists():
            layout = SessionLayout(session_dir_override)
            if layout.windows_npz.exists():
                npz_path = layout.windows_npz
            else:
                legacy = session_dir_override / "windows" / "eeg_windows.npz"
                if legacy.exists():
                    npz_path = legacy
        if npz_path is None:
            npz_path = self._resolve_windows_npz_for_current(subject_dir)
        if npz_path is not None:
            self._maybe_autofill_text(
                self.replay_npz_path,
                str(npz_path),
                key="replay.npz",
                legacy_values={"eeg_windows.npz"},
            )

        _exp_hash, model_path, scaler_path = self._resolve_latest_model_artifacts(
            subject_dir, session_dir_override=session_dir_override
        )
        if model_path is not None:
            self._maybe_autofill_text(
                self.replay_model_path,
                str(model_path),
                key="replay.model_path",
                legacy_values={"finger_action_model.pt"},
            )
        if scaler_path is not None:
            self._maybe_autofill_text(
                self.replay_scaler_path,
                str(scaler_path),
                key="replay.scaler_path",
                legacy_values={"scaler.npz"},
            )
        if model_path is None:
            infer_model_widget = self.fields.get("infer", {}).get("model_path")
            if isinstance(infer_model_widget, QLineEdit):
                text = infer_model_widget.text().strip()
                if text:
                    self._maybe_autofill_text(
                        self.replay_model_path,
                        text,
                        key="replay.model_path",
                        legacy_values={"finger_action_model.pt"},
                    )
        if scaler_path is None:
            infer_scaler_widget = self.fields.get("infer", {}).get("scaler_path")
            if isinstance(infer_scaler_widget, QLineEdit):
                text = infer_scaler_widget.text().strip()
                if text:
                    self._maybe_autofill_text(
                        self.replay_scaler_path,
                        text,
                        key="replay.scaler_path",
                        legacy_values={"scaler.npz"},
                    )

    def _on_session_dir_changed(self, _text: str) -> None:
        self._update_projects_selected_session_label()
        self._autofill_dependent_paths_from_session_dir()

    def _on_train_session_dir_changed(self, _text: str) -> None:
        self._update_projects_selected_session_label()
        self._autofill_dependent_paths_from_session_dir()

    def _resolve_session_dir_for_current(self, subject_dir: Path) -> Optional[Path]:
        """Best-effort resolve the session directory that the UI is currently targeting."""
        # 1) Explicit textbox / override
        p = self._resolve_effective_session_dir(step_id=None)
        if p and p.exists():
            return p
        # 2) Current session label (ui session id)
        if getattr(self, "current_session_ui", None):
            p = session_root(subject_dir, self.current_session_ui)
            if p.exists():
                return p
        # 3) Latest session under subject (best-effort default)
        sessions_root = subject_dir / "sessions"
        latest = _latest_dir_by_mtime(sessions_root)
        if latest and latest.exists():
            return latest
        return None

    def _infer_session_dir_from_run_dir(self, run_dir: str) -> Optional[Path]:
        p = Path(run_dir).expanduser()
        try:
            if p.exists():
                p = p.resolve()
        except Exception:
            pass
        if p.name == "models":
            processed_dir = p.parent
            if processed_dir.name == "processed":
                return processed_dir.parent
            return None
        if p.parent.name != "models":
            return None
        processed_dir = p.parent.parent
        if processed_dir.name != "processed":
            return None
        return processed_dir.parent

    def _resolve_windows_npz_for_current(self, subject_dir: Path) -> Optional[Path]:
        """Resolve the correct windows NPZ for the currently selected session/subject."""
        # Preferred: canonical session-local windows (sessions/<id>/processed/eeg_windows.npz)
        sdir = self._resolve_session_dir_for_current(subject_dir)
        if sdir:
            p = sdir / "processed" / "eeg_windows.npz"
            if p.exists():
                return p
            legacy = sdir / "windows" / "eeg_windows.npz"
            if legacy.exists():
                return legacy
        # Fallback: per-subject aggregated windows file (legacy)
        if getattr(self, "current_session_backend", None):
            p = subject_dir / "windows" / f"{self.current_subject}_{self.current_session_backend}_eeg_windows.npz"
            if p.exists():
                return p
        # Fallback: most recent NPZ in subject/windows matching subject prefix
        latest = self._latest_subject_file(subject_dir / "windows", f"{self.current_subject}_*_eeg_windows.npz")
        return latest

    def _resolve_latest_model_artifacts(
        self,
        subject_dir: Path,
        *,
        session_dir_override: Optional[Path] = None,
        subject_id_override: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[Path], Optional[Path]]:
        """Resolve latest (exp_hash, model_path, scaler_path) for the selected subject."""
        # Preferred: session-local model runs (sessions/<id>/processed/models/<run_id>/)
        sdir = session_dir_override or self._resolve_session_dir_for_current(subject_dir)
        if sdir:
            models_root = sdir / "processed" / "models"
            if models_root.exists():
                runs = [p for p in models_root.iterdir() if p.is_dir()]
                if runs:
                    runs.sort(key=lambda p: p.stat().st_mtime)
                    run_dir = runs[-1]
                    run_id = run_dir.name
                    model_path = run_dir / "finger_action_model.pt"
                    scaler_path = run_dir / "scaler.npz"
                    return run_id, (model_path if model_path.exists() else None), (scaler_path if scaler_path.exists() else None)

        sessions_root = subject_dir / "sessions"
        if sessions_root.exists():
            best: Optional[Tuple[float, Path, Path]] = None
            for candidate_session in sessions_root.iterdir():
                if not candidate_session.is_dir():
                    continue
                models_root = candidate_session / "processed" / "models"
                if not models_root.exists():
                    continue
                runs = [p for p in models_root.iterdir() if p.is_dir()]
                if not runs:
                    continue
                runs.sort(key=lambda p: p.stat().st_mtime)
                run_dir = runs[-1]
                try:
                    score = run_dir.stat().st_mtime
                except Exception:
                    score = 0.0
                if best is None or score > best[0]:
                    best = (score, candidate_session, run_dir)
            if best is not None:
                run_dir = best[2]
                run_id = run_dir.name
                model_path = run_dir / "finger_action_model.pt"
                scaler_path = run_dir / "scaler.npz"
                return run_id, (model_path if model_path.exists() else None), (scaler_path if scaler_path.exists() else None)

        # Legacy fallback: repo-level models (data/models/<subject>/<exp_hash>/)
        subject_id = subject_id_override or self.current_subject or "UNKNOWN"
        models_root = self.repo_root / "data" / "models" / str(subject_id)
        if not models_root.exists():
            return None, None, None
        runs = [p for p in models_root.iterdir() if p.is_dir()]
        if not runs:
            return None, None, None
        runs.sort(key=lambda p: p.stat().st_mtime)
        run_dir = runs[-1]
        exp_hash = run_dir.name
        model_path = run_dir / "finger_action_model.pt"
        scaler_path = run_dir / "scaler.npz"
        return exp_hash, (model_path if model_path.exists() else None), (scaler_path if scaler_path.exists() else None)

    def _update_resume_ui(self) -> None:
        if not hasattr(self, "resume_status_label") or not hasattr(
            self, "resume_checkbox"
        ):
            return
        ok, reason = self._resume_available()
        self.resume_status_label.setText(f"Resume available: {'Yes' if ok else 'No'}")
        self.resume_status_label.setToolTip(reason)
        self.resume_checkbox.setEnabled(ok)
        if not ok:
            self.resume_checkbox.setChecked(False)

    def _resume_available(self) -> Tuple[bool, str]:
        if not self.current_subject:
            return False, "Select a subject to evaluate resume."
        state = self._read_session_state_payload()
        if not state:
            return False, "No session state found."
        if state.get("subject_id") and state.get("subject_id") != self.current_subject:
            return False, "Session state subject mismatch."
        tb = state.get("timebase_version") or state.get("timebase")
        if tb and tb != TIMEBASE_VERSION:
            return False, f"Timebase mismatch: {tb}"

        # Preferred (current) layout: session_dir/raw contains eeg_raw_shard_*.npy
        session_dir = Path(state.get("session_dir", "")) if state.get("session_dir") else None
        raw_dir = None
        if session_dir and session_dir.exists():
            cand = session_dir / "raw"
            if cand.exists():
                raw_dir = cand

        # Backward compatibility: some older states may only store a 'features_path'
        features_path = Path(state.get("features_path", "")) if state.get("features_path") else None

        has_raw_shards = False
        if raw_dir and raw_dir.exists():
            try:
                has_raw_shards = any(raw_dir.glob("eeg_raw_shard_*.npy"))
            except Exception:
                has_raw_shards = False

        has_legacy_features_csv = False
        if features_path and features_path.exists():
            # Legacy requirement: basic schema check
            if self._csv_has_data_rows(features_path):
                header = self._read_csv_header(features_path)
                required = {"lsl_timestamp", "time_s", "ch1", "ch2", "ch3", "ch4"}
                has_legacy_features_csv = required.issubset(set(header))

        if not has_raw_shards and not has_legacy_features_csv:
            return False, "No raw shards found (session_dir/raw) and no valid legacy features CSV."

        events_path = Path(state.get("events_path", "")) if state.get("events_path") else None
        if events_path and events_path.exists():
            return True, "Resume OK."

        # If events are missing, it's still safe to resume: we will create a new events file in the session.
        if session_dir and session_dir.exists():
            return True, "Events missing; will create new events file."
        if features_path and features_path.parent.exists():
            return True, "Events missing; will create new events file."
        return False, "Events path not safe."


    def _csv_has_data_rows(self, path: Path) -> bool:
        try:
            with path.open("r", newline="") as f:
                header = f.readline()
                if not header:
                    return False
                for line in f:
                    if line.strip():
                        return True
        except Exception:
            return False
        return False

    def _read_csv_header(self, path: Path) -> list[str]:
        try:
            with path.open("r", newline="") as f:
                line = f.readline()
                if not line:
                    return []
                return [h.strip() for h in line.strip().split(",") if h.strip()]
        except Exception:
            return []

    def _detect_lsl_streams(self) -> None:
        if not LSL_AVAILABLE:
            return
        streams = pylsl.resolve_streams() if pylsl else []
        items = []
        previous = self._selected_stream_name()
        for stream in streams:
            try:
                items.append(StreamInfo(name=stream.name(), stype=stream.type()))
            except Exception:
                continue
        self.lsl_combo.clear()
        self.lsl_combo.addItem("-")
        if not items:
            self._set_stream_status("No LSL streams detected.")
            return
        for info in items:
            self.lsl_combo.addItem(f"{info.name} ({info.stype})")
        if previous:
            idx = self._find_stream_index(previous)
            if idx is not None:
                self.lsl_combo.setCurrentIndex(idx)
        self._set_stream_status(f"Detected {len(items)} LSL stream(s).")

    def _on_lsl_stream_changed(self, _text: str) -> None:
        name = self._selected_stream_name()
        stype = self._selected_stream_type()
        if name:
            self.live_stream_name = name
            self._set_connector_stream(name)
        if stype:
            self.live_stream_type = stype
        if self._auto_scan_active and name:
            self._stop_auto_scan()
            if self._auto_scan_wants_healthcheck:
                self._auto_scan_wants_healthcheck = False
                self._schedule_healthcheck()

    def _on_stream_name_input(self, text: str) -> None:
        cleaned = text.strip()
        if cleaned:
            self.live_stream_name = cleaned
            self._set_connector_stream(cleaned)

    def _default_stream_name(self) -> str:
        if self.current_subject:
            return f"Muse2-EEG-{self.current_subject}"
        return DEFAULT_STREAM_NAME

    def _seed_stream_name_input(self) -> None:
        if not getattr(self, "stream_name_input", None):
            return
        if self.stream_name_input.text().strip():
            return
        seeded = self._default_stream_name()
        self.stream_name_input.setText(seeded)
        self.live_stream_name = seeded
        self._set_connector_stream(seeded)

    def _effective_stream_name(self) -> str:
        if getattr(self, "stream_name_input", None):
            text = self.stream_name_input.text().strip()
            if text:
                return text
        return self._default_stream_name()

    def _start_auto_scan(self, wants_healthcheck: bool = False) -> None:
        if not LSL_AVAILABLE:
            return
        self._auto_scan_active = True
        self._auto_scan_wants_healthcheck = wants_healthcheck
        if not self._auto_scan_timer.isActive():
            self._auto_scan_timer.start()
        self._auto_scan_lsl_streams()

    def _stop_auto_scan(self) -> None:
        self._auto_scan_active = False
        self._auto_scan_wants_healthcheck = False
        if self._auto_scan_timer.isActive():
            self._auto_scan_timer.stop()

    def _auto_scan_lsl_streams(self) -> None:
        if not self._auto_scan_active or not LSL_AVAILABLE:
            return
        if self.input_source.currentText() == "CSV Offline":
            self._stop_auto_scan()
            return
        self._detect_lsl_streams()
        if self._selected_stream_name():
            self._stop_auto_scan()
            if self._auto_scan_wants_healthcheck:
                self._auto_scan_wants_healthcheck = False
                self._schedule_healthcheck()
            return
        expected = self._effective_stream_name()
        idx = self._find_stream_index(expected)
        if idx is None and expected:
            idx = self._find_stream_index(None)
        if idx is not None:
            self.lsl_combo.setCurrentIndex(idx)

    def _find_stream_index(self, expected_name: Optional[str]) -> Optional[int]:
        expected = expected_name.lower().strip() if expected_name else ""
        for i in range(self.lsl_combo.count()):
            raw = self.lsl_combo.itemText(i)
            if not raw or raw == "-":
                continue
            name = raw.split("(")[0].strip()
            stype = ""
            if "(" in raw and ")" in raw:
                stype = raw.split("(", 1)[1].split(")", 1)[0].strip()
            if expected:
                if name.lower() == expected or expected in name.lower():
                    return i
                continue
            if "muse" in name.lower() and stype.lower() == "eeg":
                return i
        return None

    def _update_stream_controls(self) -> None:
        if not LSL_AVAILABLE:
            return
        source = self.input_source.currentText()
        csv_mode = source == "CSV Offline"
        self.lsl_combo.setEnabled(not csv_mode)
        self.detect_btn.setEnabled(not csv_mode)
        self.test_btn.setEnabled(not csv_mode)
        self.csv_path.setEnabled(csv_mode)
        if csv_mode:
            self._stop_auto_scan()
        self._refresh_status_summary()

    def _test_lsl(self) -> None:
        if not LSL_AVAILABLE:
            return
        choice = self.lsl_combo.currentText()
        if not choice or choice == "-":
            self._set_stream_status("No LSL stream selected.")
            return
        name = choice.split("(")[0].strip()
        streams = pylsl.resolve_streams() if pylsl else []
        match = None
        for stream in streams:
            if stream.name() == name:
                match = stream
                break
        if match is None:
            self._set_stream_status("Selected stream not found.")
            return
        try:
            inlet = pylsl.StreamInlet(match)
            inlet.pull_sample(timeout=0.0)
            self._set_stream_status(f"Connected to {match.name()} ({match.type()})")
            try:
                srate = match.nominal_srate()
                self.sample_rate_display.setText(str(srate))
            except Exception:
                self.sample_rate_display.setText("-")
        except Exception as exc:
            self._set_stream_status(f"Failed to connect: {exc}")

    def _browse_path(
        self, widget: QLineEdit, pattern: str, title: str, mode: str = "open"
    ) -> None:
        if mode == "save":
            path, _ = QFileDialog.getSaveFileName(self, title, "", pattern)
        else:
            path, _ = QFileDialog.getOpenFileName(self, title, "", pattern)
        if path:
            widget.setText(path)

    def _browse_dir(self, widget: QLineEdit, title: str) -> None:
        path = QFileDialog.getExistingDirectory(self, title, "")
        if path:
            widget.setText(path)

    def _open_doc(self, rel_path: str) -> None:
        path = self.repo_root / rel_path
        if not path.exists():
            self._append_log(f"Doc not found: {path}")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _run_step(self, step_id: str, script_key: str) -> None:
        if self.hard_stop_locked:
            self._show_blocking_notice(
                "HARD STOP — Acknowledgement Required",
                "You must acknowledge the last hard stop report before restarting steps.",
            )
            return
        if self._eval_queue_active:
            self._eval_queue_active = False
            self._eval_queue = []
        if not self.current_project or not self.current_subject:
            QMessageBox.warning(
                self, "Project/Subject Required", "Select a project and subject first."
            )
            return
        if self.runner.is_running():
            QMessageBox.warning(self, "Busy", "Another step is still running.")
            return
        script_info = self.scripts.get(script_key)
        if not script_info:
            QMessageBox.warning(
                self, "Missing Script", f"Script for {step_id} not found."
            )
            return

        infer_subject_override = None
        subject_for_step = self.current_subject
        if step_id == "infer":
            infer_subject_override = self._infer_subject_override()
            if infer_subject_override:
                subject_for_step = infer_subject_override

        subject_dir = subject_root(self.current_project, subject_for_step)
        ensure_subject_dirs(subject_dir)

        settings = self._collect_settings(step_id)
        settings["TIMEBASE_VERSION"] = TIMEBASE_VERSION
        if step_id == "infer":
            settings.pop("infer_subject_override", None)
        if step_id == "step1":
            settings["MODE"] = "train_record"
            settings["ALLOW_DROP"] = False
            settings["SAVE_RAW"] = True
            settings["ENABLE_FEATURES"] = False
            settings["ENABLE_INFERENCE"] = False
            # Step 1 always writes into a canonical session directory under Projects/<project>/subjects/<subject>/sessions/.
            session_dir_path = self._resolve_effective_session_dir(step_id=None)
            if session_dir_path:
                settings["session_dir"] = str(session_dir_path)
        infer_session_dir: Optional[Path] = None
        if step_id == "infer":
            stream_name = self._selected_stream_name() or self.live_stream_name
            stream_type = self._selected_stream_type() or self.live_stream_type
            if stream_name:
                self.live_stream_name = stream_name
            if stream_type:
                self.live_stream_type = stream_type
            if script_key == "live_infer":
                settings["stream_name"] = self.live_stream_name
                settings["stream_type"] = self.live_stream_type
                if getattr(self, "live_lsl_source_id", None):
                    settings["lsl_source_id"] = str(self.live_lsl_source_id)
            else:
                settings["STREAMER_INTERNAL"] = self.muse_connector.is_running()
                settings["STREAMER_STREAM_NAME"] = self.live_stream_name
                settings["STREAMER_STREAM_TYPE"] = self.live_stream_type
                settings["LSL_STREAM_NAME"] = self.live_stream_name
                settings["LSL_STREAM_TYPE"] = self.live_stream_type
            settings["LABEL_CHECK_ACKNOWLEDGED"] = self.live_label_acknowledged
            settings["LABEL_CHECK_FOUND_LABELS"] = self.live_label_details.get("labels")
            settings["LABEL_CHECK_EXPECTED_LABELS"] = settings.get("REQUIRED_LSL_LABELS")
            if not settings.get("session_id") and self.current_session_backend and not infer_subject_override:
                settings["session_id"] = self.current_session_backend
            if infer_subject_override and infer_subject_override != self.current_subject:
                infer_session_dir = self._latest_session_for_subject(subject_for_step)
            else:
                infer_session_dir = self._resolve_effective_session_dir(step_id=None)
                if infer_session_dir is None:
                    infer_session_dir = self._latest_session_for_subject(subject_for_step)
            if infer_session_dir:
                settings["session_dir"] = str(infer_session_dir)
        if (
            step_id in {"step1", "infer"}
            and self.input_source.currentText() == "CSV Offline"
        ):
            self._append_log(
                "CSV Offline selected; backend does not support offline replay in Step 1."
            )
        if step_id == "step1":
            if not settings.get("ENABLE_PLOT", True):
                self._append_log(
                    "Note: ENABLE_PLOT is disabled; no live graph will appear."
                )
            if not settings.get("EVENT_MARKING_ENABLED", True):
                self._append_log(
                    "Note: EVENT_MARKING_ENABLED is disabled; live labeling is off."
                )

        backend_session = self.current_session_backend
        if step_id == "step1":
            settings["subject_id"] = self.current_subject
            resume_requested = bool(
                getattr(self, "resume_checkbox", None)
                and self.resume_checkbox.isChecked()
            )
            settings["force_new_session"] = not resume_requested
            backend_session = self._prepare_session_id(step_id, settings)
        elif step_id == "infer":
            settings["subject_id"] = subject_for_step
            if infer_subject_override:
                backend_session = None
            else:
                backend_session = self._prepare_session_id(step_id, settings)
        elif step_id == "step1b" and not backend_session:
            backend_session = self._guess_backend_session_id()
        elif step_id == "train" and not backend_session:
            backend_session = self._guess_backend_session_id()

        if step_id in {"train", "infer"}:
            if step_id == "train":
                session_dir_path = self._resolve_effective_session_dir(step_id="train")
                if not session_dir_path:
                    QMessageBox.warning(
                        self,
                        "Session Dir Required",
                        "Missing session dir. Step 2 requires the session folder that contains processed EEG windows\n"
                        "(produced by Step 1b). Select subjects/<id>/sessions/<session_id>.",
                    )
                    return
            else:
                session_dir_path = infer_session_dir
                if not session_dir_path:
                    QMessageBox.warning(
                        self,
                        "Session Dir Required",
                        "Missing session dir. Select the session folder under subjects/<id>/sessions/<session_id> "
                        "or ensure the subject has at least one session.",
                    )
                    return
        if step_id == "step1b":
            settings["subject_id"] = settings.get("subject_id") or self.current_subject
            session_dir_path = self._resolve_effective_session_dir(step_id=None)
            session_dir_value = str(session_dir_path) if session_dir_path else ""
            if session_dir_value:
                settings["session_dir"] = session_dir_value
            if settings.get("WINDOW_SEC") is not None:
                settings["WINDOW_SEC_DEFAULT"] = settings.get("WINDOW_SEC")
            # Legacy mode: only guess CSV paths when no session_dir is provided.
            if not session_dir_value:
                if not settings.get("features"):
                    latest_features = self._latest_subject_file(
                        subject_dir / "features",
                        f"{self.current_subject}_*_eeg_features.csv",
                    )
                    if latest_features:
                        settings["features"] = str(latest_features)
                if not settings.get("events"):
                    latest_events = self._latest_subject_file(
                        subject_dir / "events",
                        f"{self.current_subject}_*_events.jsonl",
                    )
                    if not latest_events:
                        latest_events = self._latest_subject_file(
                            subject_dir / "events",
                            f"{self.current_subject}_*_events.csv",
                        )
                    if latest_events:
                        settings["events"] = str(latest_events)
        if step_id == "train":
            settings["subject_id"] = settings.get("subject_id") or self.current_subject
            session_dir_path = self._resolve_effective_session_dir(step_id="train")
            session_dir_value = str(session_dir_path) if session_dir_path else ""
            if session_dir_value:
                settings["session_dir"] = session_dir_value
                settings["npz"] = "eeg_windows.npz"
            # Legacy mode: only guess an aggregated subject-level NPZ when no session_dir is selected.
            if not session_dir_value and not settings.get("npz"):
                latest_npz = self._latest_subject_file(
                    subject_dir / "windows",
                    f"{self.current_subject}_*_eeg_windows.npz",
                )
                if latest_npz:
                    settings["npz"] = str(latest_npz)

        validation = validate_step_settings(step_id, settings)
        if not validation.ok:
            QMessageBox.warning(
                self,
                "Invalid Settings",
                "\n".join(validation.errors),
            )
            return
        for warning in validation.warnings:
            self._append_log(f"⚠️ {warning}")

        if step_id == "infer" and settings.get("enable_actuation"):
            if not self._confirm_actuation():
                self._append_log("Actuation confirmation cancelled; run aborted.")
                return

        reserve_session_dir = not (
            step_id == "step1" and bool(settings.get("force_new_session", True))
        )
        if backend_session:
            self.current_session_backend = backend_session
            self.current_session_ui = ui_session_id(
                self.current_subject, backend_session
            )
            self._set_session_label(f"Session: {self.current_session_ui}")
            session_dir = session_root(subject_dir, self.current_session_ui)
            # For a brand-new Step 1 run, do not pre-create the session dir.
            # The recorder owns final session allocation and would otherwise
            # detect this placeholder as a collision, creating an unnecessary
            # `_01` sibling session.
            if reserve_session_dir:
                ensure_session_dirs(session_dir)
            self.session_dir_input.setText(str(session_dir))
            settings["session_dir"] = str(session_dir)

        
        # Ensure downstream steps that rely on model/scaler defaults don't accidentally pick up
        # stale root-level files. Prefer latest artifacts for the selected subject.
        if step_id in {"infer"}:
            def _uses_placeholder_path(value: Optional[str], filename: str) -> bool:
                if not value:
                    return True
                normalized = str(value).strip().replace("\\", "/")
                return normalized in {
                    filename,
                    f"models/{filename}",
                    f"./{filename}",
                    f"./models/{filename}",
                }

            exp_hash, model_path, scaler_path = self._resolve_latest_model_artifacts(
                subject_dir,
                session_dir_override=infer_session_dir,
                subject_id_override=subject_for_step,
            )
            if model_path and _uses_placeholder_path(settings.get("model_path"), "finger_action_model.pt"):
                settings["model_path"] = str(model_path)
            if scaler_path and _uses_placeholder_path(settings.get("scaler_path"), "scaler.npz"):
                settings["scaler_path"] = str(scaler_path)
        config_path = subject_dir / "config" / f"{step_id}.json"
        session_id_value = self.current_session_ui or "UNKNOWN"
        if step_id == "infer" and infer_session_dir is not None:
            session_id_value = infer_session_dir.name
        config = build_config(
            project_name=self.current_project,
            subject_id=subject_for_step,
            session_id=session_id_value,
            settings=settings,
            timebase_version=TIMEBASE_VERSION,
        )
        write_json(config_path, config.to_dict())

        if reserve_session_dir:
            self._write_session_snapshot(subject_dir, config.to_dict(), step_id)

        args = [str(script_info.path), "--config", str(config_path)]
        if step_id == "step1":
            args.extend(["--mode", "train_record"])
            if settings.get("session_dir"):
                args.extend(["--session-dir", str(settings["session_dir"])])
        if step_id == "step1b":
            session_dir_path = self._resolve_effective_session_dir(step_id=None)
            session_dir_value = str(session_dir_path) if session_dir_path else ""
            # If the user hasn't manually provided a session dir, default to the currently selected session.
            if not session_dir_value and self.current_session_ui:
                session_dir_value = str(session_root(subject_dir, self.current_session_ui))
                self.session_dir_input.setText(session_dir_value)
            if session_dir_value:
                args.extend(["--session-dir", session_dir_value])
                if self.allow_partial_checkbox.isChecked():
                    args.append("--allow-partial")
        if step_id == "infer":
            session_dir_value = str(infer_session_dir) if infer_session_dir else ""
            if session_dir_value:
                args.extend(["--session-dir", session_dir_value])
        if step_id == "train":
            session_dir_path = self._resolve_effective_session_dir(step_id="train")
            session_dir_value = str(session_dir_path) if session_dir_path else ""
            if session_dir_value:
                args.extend(["--session-dir", session_dir_value])
        # Enforce correct handoff between Step 1b → Step 2:
        # - always train on the selected subject (avoids argparse default filtering to an unrelated subject)
        # - prefer the windows NPZ produced for the current session to avoid stale ./eeg_windows.npz
        if step_id == "train":
            args.extend(["--subject-id", str(self.current_subject)])

        extra_args = self._collect_step_args(step_id)
        if step_id == "train":
            cleaned: list[str] = []
            skip_next = False
            for item in extra_args:
                if skip_next:
                    skip_next = False
                    continue
                if item == "--npz":
                    skip_next = True
                    continue
                cleaned.append(item)
            extra_args = cleaned
        args.extend(extra_args)

        cwd = str(self.repo_root)
        if step_id == "step1b":
            session_dir_path = self._resolve_effective_session_dir(step_id=None)
            session_dir_value = str(session_dir_path) if session_dir_path else ""
            if session_dir_value:
                cwd = str(Path(session_dir_value))

        self.active_step = step_id
        self.active_settings = dict(settings)
        self._set_step_status(step_id, "Running")
        self._append_log(f"Running: {args} (cwd={cwd})")
        if getattr(self, "dry_run_checkbox", None) and self.dry_run_checkbox.isChecked():
            self._append_log("Dry run enabled; command not executed.")
            return
        # Export LSL source-id (captured from the connector) so downstream steps can resolve
        # the exact stream instance even if multiple similarly-named streams are present.
        if getattr(self, "live_lsl_source_id", None):
            os.environ["LSL_SOURCE_ID"] = str(self.live_lsl_source_id)
        # Also export stream name/type as a convenience for scripts that support env overrides.
        if getattr(self, "live_stream_name", None):
            os.environ["LSL_STREAM_NAME"] = str(self.live_stream_name)
        if getattr(self, "live_stream_type", None):
            os.environ["LSL_STREAM_TYPE"] = str(self.live_stream_type)

        self.runner.start(sys.executable, args, cwd=cwd)

    def _write_session_snapshot(
        self, subject_dir: Path, step_payload: Dict[str, Any], step_id: str
    ) -> None:
        if not self.current_session_ui:
            return
        session_dir = session_root(subject_dir, self.current_session_ui)
        ensure_session_dirs(session_dir)
        snapshot_path = session_dir / "session_config.json"
        if snapshot_path.exists():
            existing = json.loads(snapshot_path.read_text())
        else:
            existing = SessionSnapshot(
                schema_version=step_payload.get("schema_version", "1.0"),
                created_at=step_payload.get("created_at", ""),
                project_name=step_payload.get("project_name", ""),
                subject_id=step_payload.get("subject_id", ""),
                session_id=step_payload.get("session_id", ""),
                timebase_version=step_payload.get("TIMEBASE_VERSION", TIMEBASE_VERSION),
                steps={},
            ).to_dict()
        existing.setdefault("steps", {})[step_id] = step_payload
        write_json(snapshot_path, existing)

    def _collect_settings(self, step_id: str) -> Dict[str, Any]:
        defaults = dict(self.defaults.get(step_id, {}))
        fields = self.fields.get(step_id, {})
        for key, widget in fields.items():
            defaults[key] = self._widget_value(widget)
        if step_id == "train":
            if defaults.get("hop_seconds") == 0.0:
                defaults["hop_seconds"] = None
        if "REQUIRED_LSL_LABELS" in defaults:
            defaults["REQUIRED_LSL_LABELS"] = self._parse_label_field(
                defaults.get("REQUIRED_LSL_LABELS")
            )
        if self.input_source.currentText() == "CSV Offline":
            defaults["LSL_STREAM_NAME"] = None
            defaults["LSL_STREAM_TYPE"] = None
        else:
            defaults["LSL_STREAM_NAME"] = self._selected_stream_name()
            defaults["LSL_STREAM_TYPE"] = self._selected_stream_type()
        defaults["CSV_OFFLINE_PATH"] = self.csv_path.text().strip() or None
        return self._migrate_legacy_settings(step_id, defaults)

    def _migrate_legacy_settings(
        self, step_id: str, settings: Dict[str, Any]
    ) -> Dict[str, Any]:
        if "record_raw" in settings and "no_file_io" not in settings:
            settings["no_file_io"] = not bool(settings.get("record_raw"))
            settings.pop("record_raw", None)
        if "ENABLE_ACTUATION" in settings:
            if "ENABLE_ACTUATION" not in self._legacy_warnings:
                self._append_log(
                    "⚠️ Legacy key ENABLE_ACTUATION detected; mapping to enable_actuation."
                )
                self._legacy_warnings.add("ENABLE_ACTUATION")
            if "enable_actuation" not in settings:
                settings["enable_actuation"] = bool(settings.get("ENABLE_ACTUATION"))
            settings.pop("ENABLE_ACTUATION", None)
        return settings

    def _create_new_session(self) -> None:
        if not self.current_subject:
            QMessageBox.warning(self, "Subject Required", "Select a subject first.")
            return
        session_root_value = self.session_root_input.text().strip()
        if not session_root_value:
            QMessageBox.warning(self, "Session Root Required", "Select a session root.")
            return
        session_id = session_backend_id()
        session_dir = Path(session_root_value) / f"{self.current_subject}_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)
        self.current_session_backend = session_id
        self.current_session_ui = ui_session_id(self.current_subject, session_id)
        self.session_dir_input.setText(str(session_dir))
        self._set_session_label(f"Session: {self.current_session_ui}")
        self._append_log(f"Created session dir: {session_dir}")
        self._load_session_summary()

    def _load_session_summary(self) -> None:
        session_dir_value = self.session_dir_input.text().strip()
        if not session_dir_value:
            return
        session_dir = Path(session_dir_value)
        manifest_path = session_dir / "manifest.json"
        meta_path = session_dir / "meta.json"
        events_jsonl_path = session_dir / "events" / "events.jsonl"
        events_json_path = session_dir / "events" / "events.json"
        timebase_path = session_dir / "timebase_report.json"
        manifest = {}
        meta = {}
        timebase = {}
        try:
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text())
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
            if timebase_path.exists():
                timebase = json.loads(timebase_path.read_text())
        except Exception as exc:
            self._append_log(f"Failed to read session summary: {exc}")
            return
        event_count = 0
        if events_jsonl_path.exists():
            event_count = len(
                [ln for ln in events_jsonl_path.read_text().splitlines() if ln.strip()]
            )
        elif events_json_path.exists():
            try:
                payload = json.loads(events_json_path.read_text())
                if isinstance(payload, dict) and isinstance(payload.get("events"), list):
                    payload = payload.get("events")
                if isinstance(payload, list):
                    event_count = len([ev for ev in payload if isinstance(ev, dict)])
            except Exception:
                event_count = 0
        seq_min = manifest.get("seq_min")
        seq_max = manifest.get("seq_max")
        seq_range = f"{seq_min}..{seq_max}" if seq_min is not None else "-"
        shard_list = manifest.get("shard_list") or []
        total_samples = manifest.get("actual_sample_count")
        timebase_ranges = timebase.get("ranges") if isinstance(timebase, dict) else None
        self.session_summary_labels["session_id"].setText(str(meta.get("session_id", "-")))
        self.session_summary_labels["created_at"].setText(
            str(meta.get("created_at_utc", "-"))
        )
        self.session_summary_labels["mode"].setText(str(meta.get("mode", "-")))
        self.session_summary_labels["termination_reason"].setText(
            str(manifest.get("termination_reason", "-"))
        )
        self.session_summary_labels["seq_range"].setText(seq_range)
        self.session_summary_labels["missing_seq"].setText(
            str(manifest.get("missing_seq_count", "-"))
        )
        self.session_summary_labels["shards"].setText(str(len(shard_list)))
        self.session_summary_labels["total_samples"].setText(str(total_samples or "-"))
        self.session_summary_labels["timebase_ranges"].setText(
            str(len(timebase_ranges or []))
        )
        self.session_summary_labels["events"].setText(str(event_count))

    def _run_validate_session(self) -> None:
        if self.runner.is_running():
            self._append_log("Another process is running; stop it before validation.")
            return
        session_dir_value = self.session_dir_input.text().strip()
        if not session_dir_value:
            self._append_log("No session directory selected for validation.")
            return
        args = [
            "-m",
            "muse_streaming.validate_session",
            "--session",
            session_dir_value,
        ]
        if self.allow_partial_checkbox.isChecked():
            args.append("--allow-partial")
        self.active_step = "validate_session"
        self._append_log(f"Validating session: {args}")
        if getattr(self, "dry_run_checkbox", None) and self.dry_run_checkbox.isChecked():
            self._append_log("Dry run enabled; validation command not executed.")
            return
        self.runner.start(sys.executable, args, cwd=str(self.repo_root))

    def _run_evaluate_all(self) -> None:
        if self._eval_queue_active:
            self._append_log("Evaluate pipeline already running.")
            return
        self._eval_queue = [
            "evaluate",
            "evaluate_deepchecks",
            "evaluate_figures",
            "evaluate_reports",
        ]
        self._eval_queue_active = True
        self._append_log(
            "Run evaluate pipeline: Step 3 → 3b → 3c → 4 (full battery)."
        )
        started = self._run_eval_script(self._eval_queue.pop(0), from_queue=True)
        if not started:
            self._eval_queue_active = False
            self._eval_queue = []

    def _run_eval_script(self, script_key: str, *, from_queue: bool = False) -> bool:
        if self.runner.is_running():
            self._append_log("Another process is running; stop it before evaluation.")
            return False
        if not from_queue and self._eval_queue_active:
            self._eval_queue_active = False
            self._eval_queue = []
        script_info = self.scripts.get(script_key)
        if not script_info:
            self._append_log(f"Evaluation script not found: {script_key}")
            return False

        subject_dir = subject_root(self.current_project, self.current_subject)
        session_dir = self._resolve_session_dir_for_current(subject_dir)
        if not session_dir:
            QMessageBox.warning(
                self,
                "Session Dir Required",
                "Missing session dir. Select the session folder under subjects/<id>/sessions/<session_id>.",
            )
            return False
        npz_path = self._resolve_windows_npz_for_current(subject_dir)
        exp_hash, model_path, scaler_path = self._resolve_latest_model_artifacts(subject_dir)

        common_fields = self.eval_fields.get("evaluate_common", {})
        run_dir_override: Optional[str] = None
        run_dir_widget = common_fields.get("run_dir")
        if isinstance(run_dir_widget, QLineEdit):
            run_dir_override = run_dir_widget.text().strip() or None
        if run_dir_override:
            inferred_session_dir = self._infer_session_dir_from_run_dir(run_dir_override)
            if inferred_session_dir is not None:
                try:
                    inferred_resolved = inferred_session_dir.resolve()
                except Exception:
                    inferred_resolved = inferred_session_dir
                try:
                    session_resolved = session_dir.resolve()
                except Exception:
                    session_resolved = session_dir
                if str(inferred_resolved) != str(session_resolved):
                    self._append_log(
                        "WARNING: Run dir override appears to belong to a different session."
                    )
                    QMessageBox.warning(
                        self,
                        "Run Dir Override Mismatch",
                        "The run dir override appears to belong to a different session.\n\n"
                        f"Selected session dir:\n{session_resolved}\n\n"
                        f"Run dir session:\n{inferred_resolved}\n\n"
                        "Evaluation will use the selected session's windows with the "
                        "overridden model artifacts.",
                    )

        args = [str(script_info.path)]

        # Preferred: session_dir contract (scripts auto-resolve latest run under processed/models)
        if session_dir:
            args += ["--session-dir", str(session_dir)]
        args += self._collect_eval_args(script_key)
        env_overrides = self._collect_eval_env(script_key)

        self.active_step = script_key
        self._append_log(f"Running: {args} (cwd={self.repo_root})")
        if getattr(self, "dry_run_checkbox", None) and self.dry_run_checkbox.isChecked():
            self._append_log("Dry run enabled; command not executed.")
            return False
        self.runner.start(
            sys.executable,
            args,
            cwd=str(self.repo_root),
            env=env_overrides or None,
        )
        return True

    def _collect_step_args(self, step_id: str) -> list[str]:
        specs = self.step_arg_specs.get(step_id, [])
        args: list[str] = []
        widgets = self.step_arg_widgets.get(step_id, {})
        includes = self.step_arg_includes.get(step_id, {})
        for spec in specs:
            widget = widgets.get(spec.name)
            if widget is None:
                continue
            value = self._widget_value(widget)
            if spec.kind == "bool":
                if bool(value):
                    args.append(spec.flag)
                continue
            include_cb = includes.get(spec.name)
            if include_cb is None or not include_cb.isChecked():
                continue
            if value is None:
                continue
            args.extend([spec.flag, str(value)])
        return args

    def _collect_eval_args(self, script_key: str) -> list[str]:
        args: list[str] = []
        fields = self.eval_fields.get(script_key, {})

        def _spin_value(key: str) -> Optional[float]:
            widget = fields.get(key)
            if isinstance(widget, QSpinBox):
                return float(widget.value())
            if isinstance(widget, QDoubleSpinBox):
                return float(widget.value())
            return None

        def _text_value(key: str) -> Optional[str]:
            widget = fields.get(key)
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                return text or None
            if isinstance(widget, QTextEdit):
                text = widget.toPlainText().strip()
                return text or None
            return None

        common_fields = self.eval_fields.get("evaluate_common", {})

        def _text_value_common(key: str) -> Optional[str]:
            widget = common_fields.get(key)
            if isinstance(widget, QLineEdit):
                text = widget.text().strip()
                return text or None
            if isinstance(widget, QTextEdit):
                text = widget.toPlainText().strip()
                return text or None
            return None

        def _is_checked(key: str) -> bool:
            widget = fields.get(key)
            return bool(isinstance(widget, QCheckBox) and widget.isChecked())

        common_run_dir = _text_value_common("run_dir")

        if script_key == "evaluate":
            if common_run_dir:
                args += ["--run-dir", common_run_dir]
            max_samples = _spin_value("max_samples")
            if max_samples and max_samples > 0:
                args += ["--max-samples", str(int(max_samples))]
            batch_size = _spin_value("batch_size")
            if batch_size:
                args += ["--batch-size", str(int(batch_size))]
            split_seed = _spin_value("split_seed")
            if split_seed is not None:
                args += ["--split-seed", str(int(split_seed))]
            save_manifest = _text_value("save_manifest")
            if save_manifest:
                args += ["--save-manifest", save_manifest]
            if _is_checked("no_manifest"):
                args.append("--no-manifest")
            if _is_checked("export_test_pred"):
                args.append("--export-test-pred")
            if _is_checked("disable_deterministic"):
                args.append("--no-deterministic")
            if _is_checked("smooth"):
                args.append("--smooth")
            if _is_checked("smooth_action_only"):
                args.append("--smooth-action-only")
            smooth_method = fields.get("smooth_method")
            if isinstance(smooth_method, QComboBox):
                args += ["--smooth-method", smooth_method.currentText()]
            smooth_window = _spin_value("smooth_window")
            if smooth_window:
                args += ["--smooth-window", str(int(smooth_window))]
            if _is_checked("hysteresis"):
                args.append("--hysteresis")
            hysteresis_frames = _spin_value("hysteresis_frames")
            if hysteresis_frames:
                args += ["--hysteresis-frames", str(int(hysteresis_frames))]
            threshold_action = _spin_value("threshold_action")
            if threshold_action is not None:
                args += ["--threshold-action", f"{threshold_action:.2f}"]
            threshold_finger = _spin_value("threshold_finger")
            if threshold_finger is not None:
                args += ["--threshold-finger", f"{threshold_finger:.2f}"]
            if _is_checked("adjacency"):
                args.append("--adjacency")
        elif script_key == "evaluate_deepchecks":
            if common_run_dir:
                args += ["--run-dir", common_run_dir]
            max_samples = _spin_value("max_samples")
            if max_samples and max_samples > 0:
                args += ["--max-samples", str(int(max_samples))]
            batch_size = _spin_value("batch_size")
            if batch_size:
                args += ["--batch-size", str(int(batch_size))]
            split_mode = fields.get("split_mode")
            if isinstance(split_mode, QComboBox):
                mode_text = split_mode.currentText().strip()
                if not mode_text.lower().startswith("auto"):
                    args += ["--split-mode", mode_text]
            purge_seconds = _spin_value("purge_seconds")
            if purge_seconds and purge_seconds > 0:
                args += ["--purge-seconds", f"{purge_seconds:.2f}"]
            hop_seconds = _spin_value("hop_seconds")
            if hop_seconds and hop_seconds > 0:
                args += ["--hop-seconds", f"{hop_seconds:.2f}"]
        elif script_key == "evaluate_figures":
            if common_run_dir:
                args += ["--run-dir", common_run_dir]
        elif script_key == "evaluate_reports":
            run_dir = _text_value("run_dir") or common_run_dir
            if run_dir:
                args += ["--run-dir", run_dir]
            exp_hash = _text_value("exp_hash")
            if exp_hash:
                args += ["--exp-hash", exp_hash]
            subject_id = _text_value("subject_id")
            if subject_id:
                args += ["--subject-id", subject_id]
        return args

    def _collect_eval_env(self, script_key: str) -> Dict[str, str]:
        env: Dict[str, str] = {}
        fields = self.eval_fields.get(script_key, {})
        if script_key == "evaluate_figures":
            show_plots = fields.get("show_plots")
            if isinstance(show_plots, QCheckBox) and show_plots.isChecked():
                env["SHOW_PLOTS"] = "1"
        return env

    def _widget_value(self, widget: QWidget) -> Any:
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        if isinstance(widget, FloatSlider):
            return widget.value()
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QLineEdit):
            text = widget.text().strip()
            return text if text else None
        if isinstance(widget, QTextEdit):
            text = widget.toPlainText().strip()
            return text if text else None
        if isinstance(widget, QComboBox):
            text = widget.currentText().strip()
            if not text:
                return None
            value_type = widget.property("value_type")
            if value_type == "int":
                try:
                    return int(text)
                except Exception:
                    return text
            if value_type == "float":
                try:
                    return float(text)
                except Exception:
                    return text
            if value_type == "bool":
                return text.lower() in {"1", "true", "yes", "y"}
            return text
        return None

    def _parse_label_field(self, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned.startswith("[") and cleaned.endswith("]"):
                cleaned = cleaned[1:-1]
            parts = [p.strip() for p in cleaned.split(",")]
            return [p for p in parts if p]
        return [str(value).strip()]

    def _reset_step(self, step_id: str) -> None:
        defaults = self.defaults.get(step_id, {})
        for key, widget in self.fields.get(step_id, {}).items():
            val = defaults.get(key)
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, FloatSlider):
                widget.setValue(float(val or 0.0))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val or 0.0))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(val or 0))
            elif isinstance(widget, QLineEdit):
                widget.setText("" if val is None else str(val))
            elif isinstance(widget, QTextEdit):
                widget.setPlainText("" if val is None else str(val))
            elif isinstance(widget, QComboBox):
                text_val = "" if val is None else str(val)
                if widget.findText(text_val) < 0:
                    widget.addItem(text_val)
                widget.setCurrentText(text_val)
        if step_id == "infer":
            self._sync_infer_inference_engine_controls()

    def _stop_process(self) -> None:
        runner_pid = self.runner.process_id() if self.runner.is_running() else 0
        connector_pid = self.muse_connector.process_id() if self.muse_connector.is_running() else 0
        self._append_log(
            f"Stop requested: runner={runner_pid or '-'} connector={connector_pid or '-'} step={self.active_step or '-'}"
        )
        stopping_any = False
        if self.active_step:
            self._set_step_status(self.active_step, "Stopping...")
        self._set_stream_status("Stopping...")
        if self.muse_connector.is_running():
            self._set_connector_status("stopping")
        if self.runner.is_running():
            stopping_any = True
            self._append_log("Stopping process...")
            self._stop_requested = True
            self._stop_waiting_runner = True
            self._stop_waiting_connector = self.muse_connector.is_running()
            self._stop_step_id = self.active_step
            self.runner.stop_staged()
        if self.muse_connector.is_running():
            stopping_any = True
            self._append_log("Stopping connector...")
            self._stop_requested = True
            self._stop_waiting_connector = True
            if self._stop_step_id is None:
                self._stop_step_id = self.active_step
            self.muse_connector.stop_staged()
        if stopping_any:
            self._set_live_buttons_state()

    def _stop_live_hard(self) -> None:
        if self.runner.is_running():
            self._append_log("⚠️ Hard stop requested for live recording...")
            self.runner.stop_staged()
        self._set_live_buttons_state()

    def _connect_muse(self) -> None:
        if self.hard_stop_locked:
            self._show_blocking_notice(
                "HARD STOP — Acknowledgement Required",
                "You must acknowledge the last hard stop report before restarting live steps.",
            )
            return
        if self.muse_connector.is_running():
            self._append_log("Connector already running.")
            return
        self.live_stream_ready = False
        self.live_label_acknowledged = False
        self.live_label_details = {}
        self._set_connector_status("scanning")
        labels, rate, _ = self._current_live_config()
        stream_name = self._effective_stream_name()
        self.live_stream_name = stream_name
        if getattr(self, "stream_name_input", None) and not self.stream_name_input.text().strip():
            self.stream_name_input.setText(stream_name)
        self._set_connector_stream(stream_name)
        # Source of truth for Muse BLE -> LSL is muse_streaming/cli.py (start-streamer),
        # which wraps muse_streaming/muse_lsl_streamer.py.
        args = [
            sys.executable,
            "-u",
            "-m",
            "muse_streaming.cli",
            "start-streamer",
            "--stream-name",
            stream_name,
            "--type",
            self.live_stream_type,
            "--rate",
            str(rate),
            "--labels",
            ",".join(labels),
        ]
        self.muse_connector.start(args, cwd=str(self.repo_root))
        self._set_live_buttons_state()

    def _disconnect_muse(self) -> None:
        if self.muse_connector.is_running():
            self._append_log("Disconnecting Muse connector...")
            self.muse_connector.stop_staged()
        self.live_stream_ready = False
        self._stop_auto_scan()
        self._set_live_buttons_state()

    def _on_connector_log(self, line: str) -> None:
        # Capture exported LSL source-id from the streamer (printed as `LSL_SOURCE_ID=...`).
        # We persist it and export it into this UI process environment so any subsequently
        # launched scripts inherit it automatically.
        m = re.search(r"LSL_SOURCE_ID=([A-Za-z0-9_.:-]+)", line)
        if m:
            self.live_lsl_source_id = m.group(1)
            try:
                os.environ["LSL_SOURCE_ID"] = self.live_lsl_source_id
            except Exception:
                pass
            self._append_log(f"[connector] {line}")
            self._append_log(f"[connector] captured LSL_SOURCE_ID={self.live_lsl_source_id}")
            # Still surface the line in the compact connector status readout.
            self._set_connector_log(line)
            return

        self._append_log(f"[connector] {line}")
        clean = line.replace("[stderr] ", "")
        self._set_connector_log(clean)
    def _on_connector_status(self, status: str) -> None:
        self._set_connector_status(status)
        if status == "streaming":
            self._start_auto_scan(wants_healthcheck=True)
        elif status in {"idle", "error"}:
            self._stop_auto_scan()
            self.live_stream_ready = False
        self._set_live_buttons_state()

    def _on_connector_device(self, device: str) -> None:
        if device:
            self._set_connector_device(device)

    def _on_connector_stream(self, stream_name: str) -> None:
        if not stream_name:
            return
        self.live_stream_name = stream_name
        self._set_connector_stream(stream_name)
        if getattr(self, "stream_name_input", None) and not self.stream_name_input.text().strip():
            self.stream_name_input.setText(stream_name)

    def _schedule_healthcheck(self, delay_ms: int = 1500) -> None:
        if self._healthcheck_pending:
            return
        self._healthcheck_pending = True
        QTimer.singleShot(delay_ms, self._run_scheduled_healthcheck)

    def _run_scheduled_healthcheck(self) -> None:
        self._healthcheck_pending = False
        self._run_stream_healthcheck()

    def _run_stream_healthcheck(self) -> None:
        if self.hard_stop_locked:
            return
        labels, _rate, require_exact = self._current_live_config()
        stream_name = self._selected_stream_name() or self.live_stream_name
        stream_type = self._selected_stream_type() or self.live_stream_type
        if stream_name:
            self.live_stream_name = stream_name
        if stream_type:
            self.live_stream_type = stream_type
        try:
            result = run_healthcheck(
                stream_name=stream_name,
                stype=stream_type,
                required_labels=labels,
                require_exact_channels=require_exact,
            )
        except Exception as exc:
            self._append_log(f"⚠️ Healthcheck failed: {exc}")
            self._show_blocking_notice(
                "Healthcheck Failed",
                f"Unable to run LSL healthcheck: {exc}",
            )
            self._update_live_status("Live status: healthcheck failed")
            self._set_live_buttons_state()
            return

        if result.ok:
            self.live_stream_ready = True
            self.live_label_details = result.to_dict()
            self._append_log("✅ LSL healthcheck passed.")
            self._update_live_status("Live status: healthy")
            self._set_live_buttons_state()
            return

        if result.reason in {"label_mismatch", "channel_count_mismatch"}:
            message = (
                f"Expected labels: {labels} ({'exact' if require_exact else 'min'}).\n"
                f"Found: channels={result.channel_count}, labels={result.labels}.\n"
                "Proceeding may cause incorrect labeling."
            )
            acknowledged = self._show_blocking_ack(
                "Label/Channel Mismatch", message, result.to_dict()
            )
            if acknowledged:
                self.live_stream_ready = True
                self.live_label_acknowledged = True
                self.live_label_details = result.to_dict()
                self._append_log(
                    "⚠️ Label mismatch acknowledged; enabling Start Recording."
                )
                self._update_live_status("Live status: acknowledged mismatch")
            else:
                self._update_live_status("Live status: mismatch not acknowledged")
            self._set_live_buttons_state()
            return

        self._append_log(f"⚠️ Healthcheck failed: {result.reason}")
        self._show_blocking_notice(
            "Healthcheck Failed",
            f"No valid samples received ({result.reason}). "
            "Fix the stream before starting recording.",
        )
        self._update_live_status("Live status: unhealthy")
        self._set_live_buttons_state()

    def _start_live_recording(self) -> None:
        if self.hard_stop_locked:
            self._show_blocking_notice(
                "HARD STOP — Acknowledgement Required",
                "You must acknowledge the last hard stop report before restarting live steps.",
            )
            return
        if not self.live_stream_ready:
            self._show_blocking_notice(
                "Stream Not Ready",
                "Connect to Muse 2 and complete the healthcheck first.",
            )
            return
        self._run_step("infer", "step1")

    def _on_connector_finished(self, exit_code: int) -> None:
        self._append_log(f"[connector] process finished with code {exit_code}")
        if exit_code != 0:
            self._set_connector_status("error")
        else:
            self._set_connector_status("idle")
        self.live_stream_ready = False
        self._stop_auto_scan()
        self._set_live_buttons_state()
        self._stop_waiting_connector = False
        self._maybe_finalize_stop()

    def _set_live_buttons_state(self) -> None:
        connector_running = self.muse_connector.is_running()
        connect_enabled = not self.hard_stop_locked and not connector_running and LSL_AVAILABLE
        disconnect_enabled = connector_running
        start_enabled = (
            not self.hard_stop_locked and self.live_stream_ready and not self.runner.is_running()
        )
        stop_enabled = self.runner.is_running()
        for btn in (
            getattr(self, "live_connect_btn", None),
            getattr(self, "live_connect_btn_page", None),
        ):
            if isinstance(btn, QPushButton):
                btn.setEnabled(connect_enabled)
        for btn in (
            getattr(self, "live_disconnect_btn", None),
            getattr(self, "live_disconnect_btn_page", None),
        ):
            if isinstance(btn, QPushButton):
                btn.setEnabled(disconnect_enabled)
        for btn in (
            getattr(self, "live_start_btn", None),
            getattr(self, "live_start_btn_page", None),
        ):
            if isinstance(btn, QPushButton):
                btn.setEnabled(start_enabled)
        for btn in (
            getattr(self, "live_stop_btn", None),
            getattr(self, "live_stop_btn_page", None),
        ):
            if isinstance(btn, QPushButton):
                btn.setEnabled(stop_enabled)

    def _current_live_config(self) -> tuple[list[str], int, bool]:
        settings = self._collect_settings("infer")
        labels = self._parse_label_field(settings.get("REQUIRED_LSL_LABELS"))
        rate = int(settings.get("SAMPLING_RATE") or 256)
        require_exact = bool(settings.get("REQUIRE_EXACTLY_4_CHANNELS", True))
        return labels, rate, require_exact

    def _update_live_status(self, text: str) -> None:
        if hasattr(self, "live_status_label") and self.live_status_label is not None:
            self.live_status_label.setText(text)

    def _confirm_actuation(self) -> bool:
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Confirm Actuation")
        dialog.setText("Actuation enabled. Confirm the hand is safe and clear.")
        dialog.setIcon(QMessageBox.Warning)
        dialog.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        return dialog.exec() == QMessageBox.Ok

    def _show_blocking_notice(self, title: str, message: str) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setIcon(QMessageBox.Warning)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()

    def _show_blocking_ack(
        self, title: str, message: str, details: Optional[Dict[str, Any]] = None
    ) -> bool:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        summary = QLabel(message)
        summary.setWordWrap(True)
        layout.addWidget(summary)
        detail_view = QTextEdit()
        detail_view.setReadOnly(True)
        if details:
            detail_view.setPlainText(json.dumps(details, indent=2))
        layout.addWidget(detail_view)
        btn = QPushButton("I understand")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        return dialog.exec() == QDialog.Accepted

    def _show_info_dialog(self, title: str, message: str) -> None:
        dialog = QMessageBox(self)
        dialog.setWindowTitle(title)
        dialog.setText(message)
        dialog.setIcon(QMessageBox.Information)
        dialog.setStandardButtons(QMessageBox.Ok)
        dialog.exec()

    def _make_info_button(self, title: str, message: str) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("InfoButton")
        btn.setText("i")
        btn.setCursor(Qt.PointingHandCursor)
        btn.setToolTip("Info")
        btn.clicked.connect(lambda: self._show_info_dialog(title, message))
        return btn

    def _on_process_started(self) -> None:
        self._stop_requested = False
        self._stop_waiting_runner = False
        self._stop_step_id = None
        if self.active_step:
            self._set_step_status(self.active_step, "Running")
        if self.active_step == "step1":
            self.stream_state_label.setText("Stream: running")
        if self.active_step == "infer":
            self._latest_live_viz_payload = None
            self._last_live_viz_mono = 0.0
        self._update_live_viz_status()
        self._set_live_buttons_state()

    def _on_process_finished(self, exit_code: int, exit_status: int) -> None:
        step = self.active_step
        if not step:
            return
        self._stop_waiting_runner = False
        if self._stop_requested:
            status = "Stopping..." if self.muse_connector.is_running() else "Stopped"
        elif exit_status != 0:
            status = "Crashed"
        else:
            status = "Success" if exit_code == 0 else f"Failed ({exit_code})"
        self._set_step_status(step, status)
        if exit_status != 0:
            self._append_log(
                f"Process crashed/aborted (exit_status={exit_status}, code={exit_code})"
            )
        elif exit_code == 0:
            self._append_log("Process completed")
        else:
            self._append_log(f"Process finished with code {exit_code}")
        if exit_code == 73:
            self._handle_hard_stop_detected()
        if exit_code == 0:
            self._sync_outputs(step)
        self._update_checklist(step)
        if step == "step1":
            self._update_resume_ui()
            self._refresh_status_summary()
        self.active_step = None
        self._update_live_viz_status()
        self._set_live_buttons_state()
        self._maybe_continue_eval_queue(step, exit_code, exit_status)
        self._maybe_finalize_stop()

    def _maybe_continue_eval_queue(
        self, step: str, exit_code: int, exit_status: int
    ) -> None:
        if not self._eval_queue_active:
            return
        if step not in {
            "evaluate",
            "evaluate_deepchecks",
            "evaluate_figures",
            "evaluate_reports",
        }:
            return
        if exit_status != 0 or exit_code != 0:
            self._append_log(
                "Evaluate pipeline aborted due to non-zero exit code/status."
            )
            self._eval_queue_active = False
            self._eval_queue = []
            return
        if not self._eval_queue:
            self._append_log("✅ Evaluate pipeline complete.")
            self._eval_queue_active = False
            return
        next_step = self._eval_queue.pop(0)
        QTimer.singleShot(
            150,
            lambda: (
                self._run_eval_script(next_step, from_queue=True)
                or self._finalize_eval_queue_abort()
            ),
        )

    def _finalize_eval_queue_abort(self) -> None:
        self._append_log("Evaluate pipeline aborted (failed to start next step).")
        self._eval_queue_active = False
        self._eval_queue = []

    def _maybe_finalize_stop(self) -> None:
        if not self._stop_requested:
            return
        if self.runner.is_running() or self.muse_connector.is_running():
            self._set_stream_status("Stopping...")
            return
        self._stop_requested = False
        self._stop_waiting_runner = False
        self._stop_waiting_connector = False
        if self._stop_step_id:
            self._set_step_status(self._stop_step_id, "Stopped")
        self._stop_step_id = None
        self._set_stream_status("Stopped")
        self._append_log("Stopped")

    def _set_step_status(self, step_id: str, text: str) -> None:
        label = self.step_status.get(step_id)
        if label:
            label.setText(f"Status: {text}")

    def _append_log(self, line: str) -> None:
        payload = parse_viz_line(line)
        if payload:
            self._latest_live_viz_payload = payload
            self._last_live_viz_mono = time.monotonic()
            self._update_live_viz_status()
            self._refresh_live_model_views()
            return
        self.log_entries.append(line)
        self._append_log_line_to_console(line)
        if line.startswith("🛑 HARD STOP"):
            self._handle_hard_stop_detected()

    def _log_is_at_bottom(self) -> bool:
        if not hasattr(self, "log_console"):
            return True
        bar = self.log_console.verticalScrollBar()
        return bar.value() >= (bar.maximum() - 2)

    def _restore_log_scroll(self, prev_value: int, was_at_bottom: bool) -> None:
        if not hasattr(self, "log_console"):
            return
        bar = self.log_console.verticalScrollBar()
        if was_at_bottom:
            bar.setValue(bar.maximum())
        else:
            bar.setValue(min(prev_value, bar.maximum()))

    def _append_log_line_to_console(self, line: str) -> None:
        if not hasattr(self, "log_console"):
            return
        filter_mode = (
            self.log_filter_combo.currentText()
            if hasattr(self, "log_filter_combo")
            else "All"
        )
        if filter_mode == "Errors" and "ERROR" not in line:
            return
        if filter_mode == "Warnings" and not any(
            token in line for token in ["WARN", "WARNING", "ERROR"]
        ):
            return
        bar = self.log_console.verticalScrollBar()
        was_at_bottom = self._log_is_at_bottom()
        prev_value = bar.value()
        self.log_console.appendPlainText(line)
        self._restore_log_scroll(prev_value, was_at_bottom)

    def _refresh_log_display(self) -> None:
        if not hasattr(self, "log_console"):
            return
        filter_mode = (
            self.log_filter_combo.currentText()
            if hasattr(self, "log_filter_combo")
            else "All"
        )
        lines: list[str] = []
        for line in self.log_entries:
            if filter_mode == "Errors" and "ERROR" not in line:
                continue
            if filter_mode == "Warnings" and not any(
                token in line for token in ["WARN", "WARNING", "ERROR"]
            ):
                continue
            lines.append(line)
        bar = self.log_console.verticalScrollBar()
        was_at_bottom = self._log_is_at_bottom()
        prev_value = bar.value()
        self.log_console.setPlainText("\n".join(lines))
        self._restore_log_scroll(prev_value, was_at_bottom)

    def _clear_logs(self) -> None:
        self.log_entries = []
        if hasattr(self, "log_console"):
            self.log_console.clear()

    def _handle_hard_stop_detected(self) -> None:
        if self.hard_stop_locked:
            return
        self.hard_stop_locked = True
        QApplication.beep()
        if hasattr(self, "hard_stop_banner"):
            self.hard_stop_banner.setVisible(True)
        self._set_live_buttons_state()
        report_path = self._find_latest_hard_stop_report()
        self._show_hard_stop_modal(report_path)
        self.hard_stop_locked = False
        if hasattr(self, "hard_stop_banner"):
            self.hard_stop_banner.setVisible(False)
        self._set_live_buttons_state()

    def _find_latest_hard_stop_report(self) -> Optional[Path]:
        report_dir = self.repo_root / "logs"
        candidates = sorted(report_dir.glob("hard_stop_*.json"))
        if not candidates:
            return None
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return candidates[0]

    def _show_hard_stop_modal(self, report_path: Optional[Path]) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("HARD STOP — Stream Unhealthy")
        layout = QVBoxLayout(dialog)
        summary = QLabel(
            "The live stream stopped due to an unhealthy signal. "
            "Review the diagnostics below before restarting."
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)
        detail_view = QTextEdit()
        detail_view.setReadOnly(True)
        if report_path and report_path.exists():
            try:
                payload = json.loads(report_path.read_text())
                detail_view.setPlainText(json.dumps(payload, indent=2))
            except Exception as exc:
                detail_view.setPlainText(f"Failed to read report: {exc}")
        else:
            detail_view.setPlainText("Hard stop report not found.")
        layout.addWidget(detail_view)
        btn = QPushButton("I understand")
        btn.clicked.connect(dialog.accept)
        layout.addWidget(btn)
        dialog.exec()

    def _safe_copy(
        self, src: Path, dest: Path, allow_overwrite: bool
    ) -> Optional[Path]:
        if not src.exists():
            return None
        try:
            if src.resolve() == dest.resolve():
                return dest
        except Exception:
            pass
        dest_path = dest
        if dest_path.exists() and not allow_overwrite:
            dest_path = next_available_path(dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest_path)
        return dest_path

    def _safe_copy_dir(
        self, src: Path, dest: Path, allow_overwrite: bool
    ) -> Optional[Path]:
        if not src.exists():
            return None
        try:
            if src.resolve() == dest.resolve():
                return dest
        except Exception:
            pass
        dest_path = dest
        if dest_path.exists() and not allow_overwrite:
            dest_path = next_available_path(dest_path)
        if dest_path.exists() and allow_overwrite:
            shutil.rmtree(dest_path)
        shutil.copytree(src, dest_path)
        return dest_path

    def _sync_outputs(self, step_id: str) -> None:
        if step_id == "step1":
            self._sync_step1_outputs()
        elif step_id == "step1b":
            self._sync_step1b_outputs()
        elif step_id == "train":
            self._sync_train_outputs()
        elif step_id == "event_tools":
            self._sync_event_outputs()

    def _log_session_artifacts(self, session_dir: Path) -> None:
        if not session_dir:
            return
        self._append_log(f"Session artifacts saved to: {session_dir}")
        self._append_log(f"raw shards: {session_dir / 'raw'}")
        self._append_log(f"events.jsonl: {session_dir / 'events' / 'events.jsonl'}")

    def _sync_step1_outputs(self) -> None:
        if (
            not self.current_project
            or not self.current_subject
            or not self.current_session_backend
        ):
            return
        session_dir_value = self.session_dir_input.text().strip()
        if session_dir_value:
            self._log_session_artifacts(Path(session_dir_value))
            self._auto_fill_paths()
            return
        subject_dir = subject_root(self.current_project, self.current_subject)
        session_dir = session_root(
            subject_dir,
            self.current_session_ui
            or ui_session_id(self.current_subject, self.current_session_backend),
        )
        ensure_session_dirs(session_dir)
        allow_overwrite = not bool(self.active_settings.get("force_new_session", True))

        subject = self.current_subject
        session = self.current_session_backend
        state_src = None
        for candidate in self._session_state_candidates():
            if candidate.exists():
                state_src = candidate
                break
        state_payload = self._read_session_state_payload()
        state_session = state_payload.get("session_id") if state_payload else None

        features_src = None
        events_src = None
        raw_src = None
        if state_payload and state_session == session:
            features_path = state_payload.get("features_path")
            events_path = state_payload.get("events_path")
            raw_path = state_payload.get("raw_path")
            if features_path:
                features_src = Path(features_path)
            if events_path:
                events_src = Path(events_path)
            if raw_path:
                raw_src = Path(raw_path)

        if features_src is None:
            features_src = (
                self.repo_root
                / "data"
                / "processed"
                / f"{subject}_{session}_eeg_features.csv"
            )
        if events_src is None:
            candidate_jsonl = session_dir / "events" / "events.jsonl"
            candidate_json = session_dir / "events" / "events.json"
            if candidate_jsonl.exists():
                events_src = candidate_jsonl
            elif candidate_json.exists():
                events_src = candidate_json
            else:
                events_src = (
                    self.repo_root
                    / "data"
                    / "processed"
                    / f"{subject}_{session}_events.csv"
                )
        if raw_src is None:
            raw_dir = session_dir / "raw"
            if raw_dir.exists() and any(raw_dir.glob("eeg_raw_shard_*.npy")):
                raw_src = raw_dir
            else:
                raw_src = self.repo_root / "data" / "raw" / f"{subject}_{session}_raw.csv"

        autosave_src = None
        if events_src.suffix.lower() == ".csv":
            autosave_src = events_src.with_name(
                events_src.name.replace("_events.csv", "_events_autosave.csv")
            )
        meta_src = features_src.parent / f"{subject}_{session}_session_meta.json"

        self._safe_copy(
            features_src, subject_dir / "features" / features_src.name, allow_overwrite
        )
        self._safe_copy(
            events_src, subject_dir / "events" / events_src.name, allow_overwrite
        )
        if autosave_src is not None:
            self._safe_copy(
                autosave_src,
                subject_dir / "events" / autosave_src.name,
                allow_overwrite,
            )
        if raw_src.is_dir():
            self._safe_copy_dir(raw_src, subject_dir / "raw", allow_overwrite)
        else:
            self._safe_copy(raw_src, subject_dir / "raw" / raw_src.name, allow_overwrite)
        if state_src:
            self._safe_copy(
                state_src, subject_dir / "logs" / state_src.name, allow_overwrite
            )
        if meta_src.exists():
            self._safe_copy(
                meta_src, session_dir / "session_meta.json", allow_overwrite
            )

        self._safe_copy(
            features_src, session_dir / "features" / features_src.name, allow_overwrite
        )
        self._safe_copy(
            events_src, session_dir / "events" / events_src.name, allow_overwrite
        )
        if autosave_src is not None:
            self._safe_copy(
                autosave_src,
                session_dir / "events" / autosave_src.name,
                allow_overwrite,
            )
        if raw_src.is_dir():
            self._safe_copy_dir(raw_src, session_dir / "raw", allow_overwrite)
        else:
            self._safe_copy(raw_src, session_dir / "raw" / raw_src.name, allow_overwrite)
        if state_src:
            self._safe_copy(
                state_src, session_dir / "logs" / state_src.name, allow_overwrite
            )
        self._auto_fill_paths()
        self._log_session_artifacts(session_dir)

    def _sync_step1b_outputs(self) -> None:
        if (
            not self.current_project
            or not self.current_subject
            or not self.current_session_ui
            or not self.current_session_backend
        ):
            return
        subject_dir = subject_root(self.current_project, self.current_subject)
        session_dir_value = self.session_dir_input.text().strip()
        session_dir = (
            Path(session_dir_value)
            if session_dir_value
            else session_root(subject_dir, self.current_session_ui)
        )
        # Canonical contract: Step 1b writes outputs into <session_dir>/processed/.
        processed_dir = session_dir / "processed"
        if (processed_dir / "eeg_windows.npz").exists() or (processed_dir / "eeg_windows.csv").exists():
            # Avoid duplicating artifacts into subject-level legacy folders.
            return

        # Legacy mode: older sessions may still write into <session_dir>/windows/.
        windows_dir = session_dir / "windows"
        if not windows_dir.exists():
            return
        subject = self.current_subject
        session = self.current_session_backend
        csv_src = windows_dir / "eeg_windows.csv"
        npz_src = windows_dir / "eeg_windows.npz"
        if csv_src.exists():
            dest_csv = subject_dir / "windows" / f"{subject}_{session}_eeg_windows.csv"
            self._safe_copy(csv_src, dest_csv, allow_overwrite=False)
        if npz_src.exists():
            dest_npz = subject_dir / "windows" / f"{subject}_{session}_eeg_windows.npz"
            self._safe_copy(npz_src, dest_npz, allow_overwrite=False)

    def _sync_train_outputs(self) -> None:
        if not self.current_project or not self.current_subject:
            return
        session_dir_value = self.session_dir_input.text().strip()
        if session_dir_value:
            session_dir = Path(session_dir_value)
            models_root = session_dir / "processed" / "models"
            if models_root.exists() and any(p.is_dir() for p in models_root.iterdir()):
                # Canonical contract keeps model artifacts session-local.
                return

        src_root = self.repo_root / "data" / "models" / self.current_subject
        if not src_root.exists():
            return
        run_dirs = [p for p in src_root.iterdir() if p.is_dir()]
        if not run_dirs:
            return
        latest = max(run_dirs, key=lambda p: p.stat().st_mtime)
        subject_dir = subject_root(self.current_project, self.current_subject)
        dest_dir = subject_dir / "models" / latest.name
        self._safe_copy_dir(latest, dest_dir, allow_overwrite=False)

    def _sync_event_outputs(self) -> None:
        if not self.current_project or not self.current_subject:
            return
        events_path = self._existing_file_path(self.event_events_path.text())
        if events_path is None:
            return
        subject_dir = subject_root(self.current_project, self.current_subject)
        self._safe_copy(
            events_path, subject_dir / "events" / events_path.name, allow_overwrite=True
        )
        if self.current_session_ui:
            session_dir = session_root(subject_dir, self.current_session_ui)
            ensure_session_dirs(session_dir)
            self._safe_copy(
                events_path,
                session_dir / "events" / events_path.name,
                allow_overwrite=True,
            )

    def _selected_stream_name(self) -> Optional[str]:
        choice = self.lsl_combo.currentText().strip()
        if not choice or choice == "-":
            return None
        return choice.split("(")[0].strip()

    def _selected_stream_type(self) -> Optional[str]:
        choice = self.lsl_combo.currentText().strip()
        if "(" in choice and ")" in choice:
            return choice.split("(", 1)[1].split(")", 1)[0].strip()
        return None

    def _prepare_session_id(self, step_id: str, settings: Dict[str, Any]) -> str:
        if step_id == "step1" and settings.get("force_new_session") is False:
            existing = self._read_session_state()
            if existing:
                return existing
        if self.current_session_backend and self.session_dir_input.text().strip():
            settings["SESSION_ID_OVERRIDE"] = self.current_session_backend
            widget = self.fields.get(step_id, {}).get("SESSION_ID_OVERRIDE")
            if isinstance(widget, QLineEdit):
                widget.setText(self.current_session_backend)
            return self.current_session_backend
        backend_id = session_backend_id()
        settings["SESSION_ID_OVERRIDE"] = backend_id
        widget = self.fields.get(step_id, {}).get("SESSION_ID_OVERRIDE")
        if isinstance(widget, QLineEdit):
            widget.setText(backend_id)
        return backend_id

    def _session_state_candidates(self) -> list[Path]:
        candidates: list[Path] = []
        session_dir = self._resolve_effective_session_dir(step_id=None)
        if session_dir is None and self.current_project and self.current_subject:
            subject_dir = subject_root(self.current_project, self.current_subject)
            session_dir = self._resolve_session_dir_for_current(subject_dir)
        if session_dir:
            candidates.append(session_dir / "logs" / "session_state.json")
        if self.current_subject:
            candidates.append(
                self.repo_root / "logs" / f"session_state_{self.current_subject}.json"
            )
        return candidates

    def _read_session_state(self) -> Optional[str]:
        if not self.current_subject:
            return None
        for state_path in self._session_state_candidates():
            if not state_path.exists():
                continue
            try:
                data = json.loads(state_path.read_text())
                return data.get("session_id")
            except Exception:
                continue
        return None

    def _read_session_state_payload(self) -> Optional[Dict[str, Any]]:
        if not self.current_subject:
            return None
        for state_path in self._session_state_candidates():
            if not state_path.exists():
                continue
            try:
                return json.loads(state_path.read_text())
            except Exception:
                continue
        return None

    def _guess_backend_session_id(self) -> Optional[str]:
        if not self.current_project or not self.current_subject:
            return None
        subject_dir = subject_root(self.current_project, self.current_subject)
        features_dir = subject_dir / "features"
        if not features_dir.exists():
            return None
        candidates = sorted(
            features_dir.glob(f"{self.current_subject}_*_eeg_features.csv")
        )
        if not candidates:
            return None
        latest = candidates[-1]
        name = latest.name
        prefix = f"{self.current_subject}_"
        if name.startswith(prefix) and name.endswith("_eeg_features.csv"):
            return name[len(prefix) : -len("_eeg_features.csv")]
        return None

    def _update_checklist(self, step_id: str) -> None:
        checklist = self.step_checklists.get(step_id)
        if not checklist:
            return
        checklist.clear()
        items = []
        if step_id in {"step1", "infer"}:
            items = self._expected_step1_outputs()
        elif step_id == "step1b":
            items = self._expected_step1b_outputs()
        elif step_id == "train":
            items = self._expected_train_outputs()
        elif step_id == "event_tools":
            items = self._expected_event_outputs()
        elif step_id == "diagnostics":
            items = []
        for label, path in items:
            item = QListWidgetItem(f"{label}: {path}")
            item.setCheckState(Qt.Checked if Path(path).exists() else Qt.Unchecked)
            checklist.addItem(item)

    def _expected_step1_outputs(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if not self.current_subject or not self.current_session_backend:
            return outputs
        subject = self.current_subject
        session = self.current_session_backend
        session_dir = None
        session_dir_value = self.session_dir_input.text().strip()
        if session_dir_value:
            session_dir = Path(session_dir_value)
        else:
            session_root_value = self.session_root_input.text().strip()
            if session_root_value:
                if self.current_session_ui:
                    session_dir = Path(session_root_value) / str(self.current_session_ui)
                else:
                    session_dir = Path(session_root_value) / f"{subject}_{session}"
        if session_dir:
            outputs.append(("Session dir", str(session_dir)))
            outputs.append(("Manifest", str(session_dir / "manifest.json")))
            outputs.append(("Meta", str(session_dir / "meta.json")))
            outputs.append(("Timebase report", str(session_dir / "timebase_report.json")))
            outputs.append(("Step1 log", str(session_dir / "logs" / "step1.log")))
            outputs.append(("Session state", str(session_dir / "logs" / "session_state.json")))
            outputs.append(
                ("Resolved settings", str(session_dir / "logs" / "resolved_settings.json"))
            )
            outputs.append(("Raw shards", str(session_dir / "raw")))
            outputs.append(("Events JSONL", str(session_dir / "events" / "events.jsonl")))
        return outputs

    def _expected_step1b_outputs(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        session_dir_value = self.session_dir_input.text().strip()
        if not session_dir_value:
            return outputs
        session_dir = Path(session_dir_value)
        outputs.append(("Window CSV", str(session_dir / "processed" / "eeg_windows.csv")))
        outputs.append(("Window NPZ", str(session_dir / "processed" / "eeg_windows.npz")))
        outputs.append(
            (
                "Extraction report",
                str(session_dir / "processed" / "extraction_report.json"),
            )
        )
        return outputs

    def _expected_train_outputs(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        session_dir_value = self.session_dir_input.text().strip()
        if session_dir_value:
            session_dir = Path(session_dir_value)
            outputs.append(("Model runs", str(session_dir / "processed" / "models")))
            outputs.append(("Reports", str(session_dir / "processed" / "reports")))
        elif self.current_project and self.current_subject:
            subject_dir = subject_root(self.current_project, self.current_subject)
            outputs.append(("Models (legacy)", str(subject_dir / "models")))
        return outputs

    def _expected_event_outputs(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        session_dir = self._resolve_event_tools_session_dir()
        if session_dir is not None:
            outputs.append(("Event session", str(session_dir)))
            outputs.append(("Session events", str(session_dir / "events")))
        elif self.current_project and self.current_subject:
            subject_dir = subject_root(self.current_project, self.current_subject)
            outputs.append(("Events dir", str(subject_dir / "events")))
        return outputs

    def _run_event_review(self) -> None:
        if self.runner.is_running():
            self._append_log(
                "Another process is running; stop it before launching event review."
            )
            return
        script_info = self.scripts.get("event_review")
        if not script_info:
            self._append_log("Event review script not found.")
            return
        args = [str(script_info.path)]
        session_dir = self._resolve_event_tools_session_dir()
        if session_dir:
            args += ["--session-dir", str(session_dir)]
        events_override = self._existing_file_path(self.event_events_path.text())
        features_override = self._existing_file_path(self.event_features_path.text())
        session_events = self._event_file_for_session(session_dir)
        if events_override is not None:
            args += ["--events", str(events_override)]
        if features_override is not None:
            args += ["--features", str(features_override)]
        if session_dir and session_events is None and events_override is None:
            QMessageBox.warning(
                self,
                "No Events In Session",
                "The selected session does not contain events/events.jsonl.\n\n"
                "Choose a recorded source session with raw/ and events/, not a combined processed session.",
            )
            return
        if not session_dir and events_override is None:
            QMessageBox.warning(
                self,
                "Session Dir Required",
                "Select a session directory or provide an explicit events file before launching event review.",
            )
            return
        self.active_step = "event_tools"
        self._append_log(f"Running: {args} (cwd={self.repo_root})")
        self.runner.start(sys.executable, args, cwd=str(self.repo_root))

    def _run_event_validate(self) -> None:
        if self.runner.is_running():
            self._append_log(
                "Another process is running; stop it before validating events."
            )
            return
        script_info = self.scripts.get("event_validate")
        if not script_info:
            self._append_log("Event validation script not found.")
            return
        args = [str(script_info.path)]
        session_dir = self._resolve_event_tools_session_dir()
        if session_dir:
            args += ["--session-dir", str(session_dir)]
        events_override = self._existing_file_path(self.event_events_path.text())
        features_override = self._existing_file_path(self.event_features_path.text())
        session_events = self._event_file_for_session(session_dir)
        if events_override is not None:
            args += ["--events", str(events_override)]
        if features_override is not None:
            args += ["--features", str(features_override)]
        if self.event_apply_fix.isChecked():
            args.append("--apply")
        if self.event_strict.isChecked():
            args.append("--strict")
        if self.event_json_report.text().strip():
            args += ["--json-report", self.event_json_report.text().strip()]
        if session_dir and session_events is None and events_override is None:
            QMessageBox.warning(
                self,
                "No Events In Session",
                "The selected session does not contain events/events.jsonl.\n\n"
                "Choose a recorded source session with raw/ and events/, not a combined processed session.",
            )
            return
        if not session_dir and events_override is None:
            QMessageBox.warning(
                self,
                "Session Dir Required",
                "Select a session directory or provide an explicit events file before validating events.",
            )
            return
        self.active_step = "event_tools"
        self._append_log(f"Running: {args} (cwd={self.repo_root})")
        self.runner.start(sys.executable, args, cwd=str(self.repo_root))

    def _finalize_event_review(self) -> None:
        self._append_log(
            "Finalize: use 's' in the review window to save, then 'q' to quit."
        )

    def _run_alignment_check(self) -> None:
        if self.runner.is_running():
            self._append_log("Another process is running; stop it before diagnostics.")
            return
        script_info = self.scripts.get("diagnostics")
        if not script_info:
            self._append_log("Diagnostics script not found.")
            return
        args = [str(script_info.path)]
        if self.diag_features_path.text().strip():
            args += ["--features", self.diag_features_path.text().strip()]
        if self.diag_events_path.text().strip():
            args += ["--events", self.diag_events_path.text().strip()]
        if hasattr(self, "diag_session_meta") and self.diag_session_meta.text().strip():
            args += ["--session-meta", self.diag_session_meta.text().strip()]
        if hasattr(self, "diag_target_fs"):
            args += ["--target-fs", str(float(self.diag_target_fs.value()))]
        if hasattr(self, "diag_self_test") and self.diag_self_test.isChecked():
            args.append("--self-test")
        self.active_step = "diagnostics"
        self.runner.start(sys.executable, args, cwd=str(self.repo_root))


def main() -> None:
    app = QApplication(sys.argv)
    # NOTE: Avoid a global proxy style that outlines *all* text.
    # It can cause glyph overdraw/"ghosting" on form labels and controls.
    # We instead outline only key labels via OutlinedLabel / delegates.
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
