from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import (
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
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QSlider,
    QSplitter,
    QStackedWidget,
    QToolButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWidgetAction,
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
from app.repo_probe import discover_scripts

try:
    import pylsl

    LSL_AVAILABLE = True
except Exception:
    pylsl = None
    LSL_AVAILABLE = False


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


TOOLTIPS: Dict[str, str] = {
    "TRAINING_MODE": "Enable training capture mode (no live inference).",
    "DEMO_MODE": "Enable demo/inference mode during streaming.",
    "ENABLE_PLOT": "Show live plots during streaming.",
    "SAVE_TO_DISK": "Write features/events to disk during run.",
    "SAVE_RAW": "Write raw EEG CSV.",
    "ENABLE_ICA": "Enable ICA preprocessing during streaming.",
    "ICA_WARMUP_S": "Seconds of warmup before ICA fitting.",
    "ICA_MIN_SAMPLES": "Minimum samples required for ICA fit.",
    "ICA_MIN_VAR": "Minimum per-channel variance required for ICA.",
    "ICA_FAIL_POLICY": "ICA failure policy (skip to keep streaming).",
    "ICA_MAX_RETRIES_PER_SESSION": "Max ICA retries before disabling.",
    "LOG_ICA_DIAGNOSTICS": "Log ICA diagnostics when skipped.",
    "DATA_STREAM_TIMEOUT_S": "Seconds before stream stall disables event marking.",
    "DATA_STREAM_CHECK_INTERVAL_S": "Stream health check interval in seconds.",
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
    "ENABLE_ACTUATION": "Allow actuation output.",
    "MC_DROPOUT_PASSES": "MC dropout passes for uncertainty estimation.",
    "EVENT_MARKING_ENABLED": "Enable event stamping during streaming.",
    "EVENTS_CSV_PATH": "Events CSV output path.",
    "EVENTS_AUTOSAVE_PATH": "Autosave events CSV path.",
    "EVENTS_CHANNEL": "Event channel label.",
}

EEGLAB_STYLE = """
QMainWindow {
    background: #b7c7e6;
}
QMenuBar {
    background: #e4e8f2;
    color: white;
    padding: 4px;
    font-weight: 600;
}
QMenuBar::item:selected {
    background: #6c86c7;
    color: white;
}
QMenu {
    background: #f2f4fb;
    color: white;
    border: 1px solid #7a8fb8;
}
QMenu::item:selected {
    background: #6c86c7;
    color: white;
}
QDockWidget {
    titlebar-close-icon: url(none);
    titlebar-normal-icon: url(none);
    font-weight: 600;
}
QDockWidget::title {
    color: white;
    background: #dbe3f4;
    padding: 4px;
}
QGroupBox {
    background: #e6edf9;
    border: 1px solid #8ba0c7;
    border-radius: 6px;
    margin-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 8px;
    top: -8px;
    padding: 0 4px;
    color: white;
}
QPushButton {
    background: #6c86c7;
    color: white;
    border-radius: 4px;
    padding: 4px 10px;
}
QPushButton:disabled {
    background: #9fb0d8;
}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background: white;
    border: 1px solid #7a8fb8;
    border-radius: 4px;
    padding: 2px 4px;
}
QListWidget {
    background: #f7f9ff;
    border: 1px solid #7a8fb8;
}
QLabel {
    color: white;
}
QToolButton {
    background: #6c86c7;
    color: white;
    border-radius: 4px;
    padding: 4px 10px;
}
QToolTip {
    background-color: #f2f4fb;
    color: white;
    border: 1px solid #7a8fb8;
}
QFrame#StatusBarFrame {
    background: #e4e8f2;
    border: 1px solid #7a8fb8;
    border-radius: 6px;
}
"""


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
    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        text = self.displayText()
        if not text:
            text = self.placeholderText()
            if not text:
                return
        option = QStyleOptionFrame()
        option.initFrom(self)
        option.rect = self.rect()
        contents = self.style().subElementRect(QStyle.SE_LineEditContents, option, self)
        margins = self.textMargins()
        contents = contents.adjusted(
            margins.left(), margins.top(), -margins.right(), -margins.bottom()
        )
        flags = self.alignment() | Qt.TextSingleLine
        if not (flags & Qt.AlignVertical_Mask):
            flags |= Qt.AlignVCenter
        fm = QFontMetrics(self.font())
        text_rect = fm.boundingRect(contents, flags, text)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setClipRect(contents)
        path = QPainterPath()
        path.addText(text_rect.left(), text_rect.top() + fm.ascent(), self.font(), text)
        painter.strokePath(path, QPen(Qt.black, 1))


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
        self.setStyleSheet(EEGLAB_STYLE)

        self.repo_root = Path(__file__).resolve().parent
        self.scripts = discover_scripts(self.repo_root)

        self.current_project: Optional[str] = None
        self.current_subject: Optional[str] = None
        self.current_session_ui: Optional[str] = None
        self.current_session_backend: Optional[str] = None

        self.fields: Dict[str, Dict[str, QWidget]] = {}
        self.defaults: Dict[str, Dict[str, Any]] = {}
        self.step_status: Dict[str, QLabel] = {}
        self.step_checklists: Dict[str, QListWidget] = {}
        self.step_script_key: Dict[str, str] = {}
        self.active_settings: Dict[str, Any] = {}
        self.step_arg_specs = self._build_step_arg_specs()
        self.step_arg_widgets: Dict[str, Dict[str, QWidget]] = {}
        self.step_arg_includes: Dict[str, Dict[str, QCheckBox]] = {}

        self.runner = ProcessRunner(self)
        self.runner.line_ready.connect(self._append_log)
        self.runner.started.connect(self._on_process_started)
        self.runner.finished.connect(self._on_process_finished)
        self.runner.failed.connect(self._append_log)
        self.active_step: Optional[str] = None

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
        for item in [
            "Project",
            "Subject",
            "Stream Setup",
            "Step 1: Record + Events",
            "Events: Mark/Edit",
            "Step 1b: Windowing",
            "Step 3: Train",
            "Step 4: Live Inference",
            "Export",
            "Logs & Diagnostics",
        ]:
            QListWidgetItem(item, self.workflow_list)
        self.workflow_list.currentRowChanged.connect(self._switch_page)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._wrap_scroll(self._build_project_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_subject_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_stream_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_step1_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_event_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_step1b_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_train_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_infer_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_export_page()))
        self.stack.addWidget(self._wrap_scroll(self._build_logs_page()))

        splitter.addWidget(self.workflow_list)
        splitter.addWidget(self.stack)
        splitter.setSizes([200, 900])

        main_layout.addWidget(splitter)
        self.setCentralWidget(main)

        self._build_log_dock()
        self._build_control_docks()
        self._wire_status_updates()
        self._refresh_status_summary()
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
                ArgSpec("subject_id", "--subject-id", "text", "Override subject ID."),
                ArgSpec("init_only", "--init-only", "bool", "Initialize session then exit."),
                ArgSpec(
                    "force_new_session",
                    "--force-new-session",
                    "bool",
                    "Force a new session (ignore resume).",
                ),
            ],
            "step1b": [
                ArgSpec("features", "--features", "text", "Override features path."),
                ArgSpec("events", "--events", "text", "Override events path."),
                ArgSpec("subject_id", "--subject-id", "text", "Subject ID override."),
                ArgSpec("target_fs", "--target-fs", "float", "Target resample rate."),
                ArgSpec("allow_gaps", "--allow-gaps", "bool", "Allow gaps in windows."),
                ArgSpec(
                    "ignore_misalignment",
                    "--ignore-misalignment",
                    "bool",
                    "Continue if events are out of range.",
                ),
                ArgSpec("seed", "--seed", "int", "Seed for REST subsampling."),
            ],
            "train": [
                ArgSpec("npz", "--npz", "text", "Window dataset path."),
                ArgSpec("subject_id", "--subject-id", "text", "Filter by subject ID."),
                ArgSpec("epochs", "--epochs", "int", "Training epochs."),
                ArgSpec("batch_size", "--batch-size", "int", "Training batch size."),
                ArgSpec("lr", "--lr", "float", "Learning rate."),
                ArgSpec("seed", "--seed", "int", "Random seed."),
                ArgSpec(
                    "loss_action_weight",
                    "--loss-action-weight",
                    "float",
                    "Action loss weight.",
                ),
                ArgSpec("rest_weight", "--rest-weight", "float", "REST class weight."),
                ArgSpec("test_size", "--test-size", "float", "Test split fraction."),
                ArgSpec("non_rest_only", "--non-rest-only", "bool", "Train on non-REST only."),
                ArgSpec("save_model", "--save-model", "text", "Model output path."),
                ArgSpec("save_scaler", "--save-scaler", "text", "Scaler output path."),
                ArgSpec("save_preds", "--save-preds", "text", "Predictions output path."),
            ],
        }

    def _wrap_scroll(self, widget: QWidget) -> QWidget:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QFrame.NoFrame)
        area.setWidget(widget)
        return area

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
        file_menu.addAction("Project Manager", lambda: self.workflow_list.setCurrentRow(0))
        file_menu.addAction("Subject Manager", lambda: self.workflow_list.setCurrentRow(1))
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        edit_menu = menu.addMenu("Edit")
        edit_menu.addAction("Session Settings", lambda: self.workflow_list.setCurrentRow(2))

        tools_menu = menu.addMenu("Tools")
        tools_menu.addAction("Stream Setup", lambda: self.workflow_list.setCurrentRow(2))
        tools_menu.addAction("Event Review", lambda: self.workflow_list.setCurrentRow(4))
        tools_menu.addAction("Diagnostics", lambda: self.workflow_list.setCurrentRow(9))

        plot_menu = menu.addMenu("Plot")
        plot_menu.addAction("Live Plot (Step 1)", lambda: self.workflow_list.setCurrentRow(3))

        study_menu = menu.addMenu("Study")
        study_menu.addAction("Windowing", lambda: self.workflow_list.setCurrentRow(5))
        study_menu.addAction("Training", lambda: self.workflow_list.setCurrentRow(6))

        datasets_menu = menu.addMenu("Datasets")
        datasets_menu.addAction("Export", lambda: self.workflow_list.setCurrentRow(8))

        run_menu = menu.addMenu("Run")
        run_menu.addAction("Run Step 1", lambda: self._run_step("step1", "step1"))
        run_menu.addAction("Run Step 1b", lambda: self._run_step("step1b", "step1b"))
        run_menu.addAction("Run Train", lambda: self._run_step("train", "train"))
        run_menu.addAction("Run Inference", lambda: self._run_step("infer", "step1"))

        help_menu = menu.addMenu("Help")
        help_menu.addAction("Logs", lambda: self.workflow_list.setCurrentRow(9))

    def _build_status_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("StatusBarFrame")
        bar.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(6, 4, 6, 4)

        self.project_label = QLabel("Project: -")
        self.subject_label = QLabel("Subject: -")
        self.session_label = QLabel("Session: -")
        self.stream_state_label = QLabel("Stream: idle")
        self.ica_state_label = QLabel("ICA: off")
        self.events_state_label = QLabel("Events: off")
        for label in (
            self.project_label,
            self.subject_label,
            self.session_label,
            self.stream_state_label,
            self.ica_state_label,
            self.events_state_label,
        ):
            label.setMinimumWidth(140)
            label.setWordWrap(False)

        layout.addWidget(self.project_label)
        layout.addWidget(self.subject_label)
        layout.addWidget(self.session_label)
        layout.addWidget(self.stream_state_label)
        layout.addWidget(self.ica_state_label)
        layout.addWidget(self.events_state_label)
        layout.addStretch(1)

        return bar

    def _build_log_dock(self) -> None:
        self.log_console = OutlinePlainTextEdit()
        self.log_console.setReadOnly(True)
        self.log_console.setMaximumBlockCount(10000)
        dock = QDockWidget("Log Console", self)
        dock.setWidget(self.log_console)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)

    def _build_control_docks(self) -> None:
        self.stream_status_dock = QLabel("Stream status: idle")
        stream_widget = QWidget()
        stream_layout = QVBoxLayout(stream_widget)
        stream_layout.addWidget(self.stream_status_dock)
        stream_layout.addWidget(QLabel("Quick Actions"))
        detect_btn = QPushButton("Detect LSL Streams")
        detect_btn.clicked.connect(self._detect_lsl_streams)
        test_btn = QPushButton("Test Connection")
        test_btn.clicked.connect(self._test_lsl)
        stream_layout.addWidget(detect_btn)
        stream_layout.addWidget(test_btn)
        open_stream_btn = QPushButton("Open Stream Setup")
        open_stream_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(2))
        stream_layout.addWidget(open_stream_btn)
        stream_layout.addStretch(1)
        stream_dock = QDockWidget("Stream Control", self)
        stream_dock.setWidget(stream_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, stream_dock)

        pipeline_widget = QWidget()
        pipeline_layout = QVBoxLayout(pipeline_widget)
        pipeline_layout.addWidget(QLabel("Pipeline Controls"))
        run_step1_btn = QPushButton("Run Step 1")
        run_step1_btn.clicked.connect(lambda: self._run_step("step1", "step1"))
        pipeline_layout.addWidget(run_step1_btn)
        run_step1b_btn = QPushButton("Run Step 1b")
        run_step1b_btn.clicked.connect(lambda: self._run_step("step1b", "step1b"))
        pipeline_layout.addWidget(run_step1b_btn)
        run_train_btn = QPushButton("Run Train")
        run_train_btn.clicked.connect(lambda: self._run_step("train", "train"))
        pipeline_layout.addWidget(run_train_btn)
        run_infer_btn = QPushButton("Run Inference")
        run_infer_btn.clicked.connect(lambda: self._run_step("infer", "step1"))
        pipeline_layout.addWidget(run_infer_btn)
        stop_btn = QPushButton("Stop Active Run")
        stop_btn.clicked.connect(self._stop_process)
        pipeline_layout.addWidget(stop_btn)
        pipeline_layout.addStretch(1)
        pipeline_dock = QDockWidget("Pipeline", self)
        pipeline_dock.setWidget(pipeline_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, pipeline_dock)

        event_widget = QWidget()
        event_layout = QVBoxLayout(event_widget)
        event_layout.addWidget(QLabel("Event Marking"))
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
        open_event_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(4))
        event_layout.addWidget(open_event_btn)
        event_layout.addStretch(1)
        event_dock = QDockWidget("Events", self)
        event_dock.setWidget(event_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, event_dock)

        model_widget = QWidget()
        model_layout = QVBoxLayout(model_widget)
        model_layout.addWidget(QLabel("Model & Preprocess"))
        self.ica_toggle_dock = QCheckBox("Enable ICA")
        self._bind_checkbox(self.ica_toggle_dock, "step1", "ENABLE_ICA")
        model_layout.addWidget(self.ica_toggle_dock)
        self.demo_toggle_dock = QCheckBox("Demo mode")
        self._bind_checkbox(self.demo_toggle_dock, "step1", "DEMO_MODE")
        model_layout.addWidget(self.demo_toggle_dock)
        self.training_toggle_dock = QCheckBox("Training mode")
        self._bind_checkbox(self.training_toggle_dock, "step1", "TRAINING_MODE")
        model_layout.addWidget(self.training_toggle_dock)
        self.plot_toggle_dock = QCheckBox("Enable plot")
        self._bind_checkbox(self.plot_toggle_dock, "step1", "ENABLE_PLOT")
        model_layout.addWidget(self.plot_toggle_dock)
        open_infer_btn = QPushButton("Open Live Inference")
        open_infer_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(7))
        model_layout.addWidget(open_infer_btn)
        model_layout.addStretch(1)
        model_dock = QDockWidget("Model", self)
        model_dock.setWidget(model_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, model_dock)

        session_widget = QWidget()
        session_layout = QVBoxLayout(session_widget)
        session_layout.addWidget(QLabel("Session Overview"))
        self.project_label_dock = QLabel("Project: -")
        self.subject_label_dock = QLabel("Subject: -")
        self.session_label_dock = QLabel("Session: -")
        for label in (
            self.project_label_dock,
            self.subject_label_dock,
            self.session_label_dock,
        ):
            label.setWordWrap(False)
        session_layout.addWidget(self.project_label_dock)
        session_layout.addWidget(self.subject_label_dock)
        session_layout.addWidget(self.session_label_dock)
        project_btn = QPushButton("Project Page")
        project_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(0))
        session_layout.addWidget(project_btn)
        subject_btn = QPushButton("Subject Page")
        subject_btn.clicked.connect(lambda: self.workflow_list.setCurrentRow(1))
        session_layout.addWidget(subject_btn)
        session_layout.addStretch(1)
        session_dock = QDockWidget("Session", self)
        session_dock.setWidget(session_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, session_dock)

    def _bind_checkbox(self, dock_cb: QCheckBox, step_id: str, key: str) -> None:
        field_cb = self.fields.get(step_id, {}).get(key)
        if not isinstance(field_cb, QCheckBox):
            dock_cb.setEnabled(False)
            return
        dock_cb.setChecked(field_cb.isChecked())
        dock_cb.toggled.connect(field_cb.setChecked)
        field_cb.toggled.connect(dock_cb.setChecked)

    def _switch_page(self, index: int) -> None:
        self.stack.setCurrentIndex(index)

    def _set_stream_status(self, text: str) -> None:
        if hasattr(self, "stream_status") and self.stream_status is not None:
            self.stream_status.setText(text)
        if hasattr(self, "stream_status_dock") and self.stream_status_dock is not None:
            self.stream_status_dock.setText(text)

    def _set_project_label(self, text: str) -> None:
        self.project_label.setText(text)
        if hasattr(self, "project_label_dock") and self.project_label_dock is not None:
            self.project_label_dock.setText(text)

    def _set_subject_label(self, text: str) -> None:
        self.subject_label.setText(text)
        if hasattr(self, "subject_label_dock") and self.subject_label_dock is not None:
            self.subject_label_dock.setText(text)

    def _set_session_label(self, text: str) -> None:
        self.session_label.setText(text)
        if hasattr(self, "session_label_dock") and self.session_label_dock is not None:
            self.session_label_dock.setText(text)

    def _wire_status_updates(self) -> None:
        for step_id in ("step1", "infer"):
            for key in ("ENABLE_ICA", "EVENT_MARKING_ENABLED"):
                widget = self.fields.get(step_id, {}).get(key)
                if isinstance(widget, QCheckBox):
                    widget.toggled.connect(self._refresh_status_summary)
        self.input_source.currentTextChanged.connect(self._refresh_status_summary)

    def _refresh_status_summary(self) -> None:
        stream_label = self.input_source.currentText()
        self.stream_state_label.setText(f"Stream: {stream_label}")
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
        self.ica_state_label.setText(f"ICA: {'on' if ica_enabled else 'off'}")
        self.events_state_label.setText(f"Events: {'on' if event_enabled else 'off'}")

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

    def _build_stream_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.stream_status = QLabel("")
        form = QFormLayout()

        self.input_source = QComboBox()
        self.input_source.addItems(["Muse 2 (LSL)", "Any LSL Stream", "CSV Offline"])
        self.input_source.currentTextChanged.connect(self._update_stream_controls)

        self.lsl_combo = QComboBox()
        self.detect_btn = QPushButton("Detect LSL Streams")
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
            self._set_stream_status(
                "pylsl not installed; LSL controls hidden (CSV offline only)."
            )
        else:
            self._update_stream_controls()

        layout.addStretch(1)
        return page

    def _build_step1_page(self) -> QWidget:
        return self._build_step_page(
            step_id="step1",
            title="Step 1: Record + Events",
            defaults=default_step1_settings(),
            script_key="step1",
            include_event_tools=True,
        )

    def _build_event_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        info = QLabel("Post-hoc event review/edit tools for the current session.")
        layout.addWidget(info)
        note = QLabel(
            "Live graph + event labeling run inside Step 1 (1_stream_and_record.py). "
            "This page is for review/repair after capture."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self.event_features_path = OutlineLineEdit()
        self.event_events_path = OutlineLineEdit()
        form.addRow("Features CSV", self.event_features_path)
        form.addRow("Events CSV", self.event_events_path)
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
        return self._build_step_page(
            step_id="step1b",
            title="Step 1b: Windowing",
            defaults=default_step1b_settings(),
            script_key="step1b",
            include_event_tools=False,
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
        return self._build_step_page(
            step_id="train",
            title="Step 3: Train",
            defaults=default_train_settings(),
            script_key="train",
            include_event_tools=False,
        )

    def _build_infer_page(self) -> QWidget:
        return self._build_step_page(
            step_id="infer",
            title="Step 4: Live Inference",
            defaults=default_infer_settings(),
            script_key="step1",
            include_event_tools=False,
        )

    def _build_export_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        msg = QLabel("Export to EEGLAB (.set/.mat) not found in repo; export disabled.")
        msg.setWordWrap(True)
        layout.addWidget(msg)
        layout.addStretch(1)
        return page

    def _build_logs_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        msg = QLabel("Diagnostics for timebase alignment and event coverage.")
        layout.addWidget(msg)

        form = QFormLayout()
        self.diag_features_path = OutlineLineEdit()
        self.diag_events_path = OutlineLineEdit()
        form.addRow("Features CSV", self.diag_features_path)
        form.addRow("Events CSV", self.diag_events_path)
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

    def _build_step_page(
        self,
        step_id: str,
        title: str,
        defaults: Dict[str, Any],
        script_key: str,
        include_event_tools: bool,
    ) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        self.step_script_key[step_id] = script_key

        header = QLabel(title)
        header.setStyleSheet("font-weight: 600; font-size: 16px;")
        layout.addWidget(header)

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
            label = QLabel(key)
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
        if step_id in {"step1", "infer"}:
            self._add_checkbox(
                step_id, form, "TRAINING_MODE", "Training mode", defaults
            )
            self._add_checkbox(step_id, form, "DEMO_MODE", "Demo mode", defaults)
            self._add_checkbox(step_id, form, "ENABLE_PLOT", "Enable plot", defaults)
            self._add_checkbox(step_id, form, "SAVE_TO_DISK", "Save to disk", defaults)
            self._add_checkbox(step_id, form, "SAVE_RAW", "Save raw", defaults)
            self._add_spin(
                step_id, form, "SAMPLING_RATE", "Sampling rate", defaults, 1, 4096
            )
            self._add_spin(
                step_id, form, "CHANNELS", "Channels", defaults, 1, 64, read_only=True
            )
            self._add_file_picker(
                step_id,
                form,
                "MODEL_PATH",
                "Model path",
                defaults,
                "Model (*.pt *.pth);;All Files (*)",
            )
            self._add_file_picker(
                step_id,
                form,
                "SCALER_PATH",
                "Scaler path",
                defaults,
                "Scaler (*.save *.pkl);;All Files (*)",
            )
            self._add_checkbox(
                step_id, form, "EVENT_MARKING_ENABLED", "Event marking", defaults
            )
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
            self._add_file_picker(
                step_id,
                form,
                "features",
                "Features path",
                defaults,
                "CSV (*.csv);;All Files (*)",
            )
            self._add_file_picker(
                step_id,
                form,
                "events",
                "Events path",
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
                step_id, form, "ignore_misalignment", "Ignore misalignment", defaults
            )
        elif step_id == "train":
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
        if step_id in {"step1", "infer"}:
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
            self._add_spin(step_id, form, "N_FINGERS", "N fingers", defaults, 1, 50)
            self._add_spin(step_id, form, "N_ACTIONS", "N actions", defaults, 1, 50)
            self._add_timebase_dropdown(
                step_id, form, "TIMEBASE_VERSION", "Timebase", defaults
            )
            self._add_slider(
                step_id,
                form,
                "BASE_CONF_THRESH",
                "Base conf thresh",
                defaults,
                0,
                1,
                decimals=2,
            )
            self._add_slider(
                step_id,
                form,
                "UNCERTAINTY_WEIGHT",
                "Uncertainty weight",
                defaults,
                0,
                1,
                decimals=2,
            )
            self._add_int_dropdown(
                step_id,
                form,
                "STABILITY_FRAMES",
                "Stability frames",
                defaults,
                [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20],
            )
            self._add_checkbox(
                step_id, form, "ENABLE_ACTUATION", "Enable actuation", defaults
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
                "DATA_STREAM_TIMEOUT_S",
                "Stream timeout (s)",
                defaults,
                0,
                60,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "DATA_STREAM_CHECK_INTERVAL_S",
                "Stream check interval (s)",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_int_dropdown(
                step_id,
                form,
                "MC_DROPOUT_PASSES",
                "MC dropout passes",
                defaults,
                [1, 2, 3, 5, 8, 10, 15, 20, 30, 50],
            )
            self._add_file_picker(
                step_id,
                form,
                "EVENTS_CSV_PATH",
                "Events CSV",
                defaults,
                "CSV (*.csv);;All Files (*)",
                mode="save",
            )
            self._add_file_picker(
                step_id,
                form,
                "EVENTS_AUTOSAVE_PATH",
                "Events autosave",
                defaults,
                "CSV (*.csv);;All Files (*)",
                mode="save",
            )
            self._add_editable_combo(
                step_id,
                form,
                "EVENTS_CHANNEL",
                "Events channel",
                defaults,
                ["n/a", "ch1", "ch2", "ch3", "ch4"],
            )
            self._add_text(step_id, form, "subject_id", "Subject ID", defaults)
            self._add_checkbox(
                step_id, form, "force_new_session", "Force new session", defaults
            )
            self._add_checkbox(step_id, form, "init_only", "Init only", defaults)
            self._add_text(
                step_id,
                form,
                "SESSION_ID_OVERRIDE",
                "Session ID override",
                defaults,
                read_only=True,
            )
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
            self._add_checkbox(step_id, form, "LABEL_GATED", "Label gated", defaults)
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
                "Loss action weight",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id,
                form,
                "rest_weight",
                "REST weight",
                defaults,
                0,
                10,
                is_float=True,
            )
            self._add_spin(
                step_id, form, "test_size", "Test size", defaults, 0, 1, is_float=True
            )
            self._add_checkbox(
                step_id, form, "non_rest_only", "Non-REST only", defaults
            )
            self._add_text(step_id, form, "save_preds", "Save predictions", defaults)
            self._add_spin(step_id, form, "N_FINGERS", "N fingers", defaults, 1, 50)
            self._add_spin(step_id, form, "N_ACTIONS", "N actions", defaults, 1, 50)
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
        ignored = {
            "DEVICE",
            "ROOT_DIR",
            "ROOT",
            "PROJECT_ROOT",
            "SESSION_STATE_DIR",
            "config",
        }
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

    def _select_subject(self, subject_id: str) -> None:
        if subject_id == "-" or not subject_id:
            return
        self.current_subject = subject_id
        self._set_subject_label(f"Subject: {subject_id}")
        if self.current_project:
            subject_dir = subject_root(self.current_project, subject_id)
            ensure_subject_dirs(subject_dir)
            self._ensure_default_configs(subject_dir)
        self._auto_fill_paths()

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
        events_dir = subject_dir / "events"
        features_dir = subject_dir / "features"
        preferred_events = None
        preferred_features = None
        if self.current_session_backend:
            candidate_events = (
                events_dir
                / f"{self.current_subject}_{self.current_session_backend}_events.csv"
            )
            candidate_features = (
                features_dir
                / f"{self.current_subject}_{self.current_session_backend}_eeg_features.csv"
            )
            if candidate_events.exists():
                preferred_events = candidate_events
            if candidate_features.exists():
                preferred_features = candidate_features

        latest_events = preferred_events or self._latest_subject_file(
            events_dir, f"{self.current_subject}_*_events.csv"
        )
        latest_features = preferred_features or self._latest_subject_file(
            features_dir, f"{self.current_subject}_*_eeg_features.csv"
        )
        self.event_events_path.setText(
            str(latest_events) if latest_events else str(events_dir)
        )
        self.event_features_path.setText(
            str(latest_features) if latest_features else str(features_dir)
        )
        self.diag_events_path.setText(
            str(latest_events) if latest_events else str(events_dir)
        )
        self.diag_features_path.setText(
            str(latest_features) if latest_features else str(features_dir)
        )
        self._update_resume_ui()

    def _latest_subject_file(self, base: Path, pattern: str) -> Optional[Path]:
        if not base.exists():
            return None
        candidates = sorted(base.glob(pattern))
        return candidates[-1] if candidates else None

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
        state_path = (
            self.repo_root / "logs" / f"session_state_{self.current_subject}.json"
        )
        if not state_path.exists():
            return False, "No session state found."
        try:
            state = json.loads(state_path.read_text())
        except Exception:
            return False, "Failed to read session state."
        if state.get("subject_id") and state.get("subject_id") != self.current_subject:
            return False, "Session state subject mismatch."
        tb = state.get("timebase_version") or state.get("timebase")
        if tb and tb != TIMEBASE_VERSION:
            return False, f"Timebase mismatch: {tb}"
        features_path = (
            Path(state.get("features_path", "")) if state.get("features_path") else None
        )
        if not features_path or not features_path.exists():
            return False, "Features file missing."
        if not self._csv_has_data_rows(features_path):
            return False, "Features file empty."
        header = self._read_csv_header(features_path)
        required = {"lsl_timestamp", "time_s", "ch1", "ch2", "ch3", "ch4"}
        if not required.issubset(set(header)):
            return False, "Features file missing required columns."
        events_path = (
            Path(state.get("events_path", "")) if state.get("events_path") else None
        )
        if events_path and events_path.exists():
            return True, "Resume OK."
        if features_path.parent.exists():
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
        for stream in streams:
            try:
                items.append(StreamInfo(name=stream.name(), stype=stream.type()))
            except Exception:
                continue
        self.lsl_combo.clear()
        if not items:
            self.lsl_combo.addItem("-")
            self._set_stream_status("No LSL streams detected.")
            return
        for info in items:
            self.lsl_combo.addItem(f"{info.name} ({info.stype})")
        self._set_stream_status(f"Detected {len(items)} LSL stream(s).")

    def _update_stream_controls(self) -> None:
        if not LSL_AVAILABLE:
            return
        source = self.input_source.currentText()
        csv_mode = source == "CSV Offline"
        self.lsl_combo.setEnabled(not csv_mode)
        self.detect_btn.setEnabled(not csv_mode)
        self.test_btn.setEnabled(not csv_mode)
        self.csv_path.setEnabled(csv_mode)
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

    def _run_step(self, step_id: str, script_key: str) -> None:
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

        subject_dir = subject_root(self.current_project, self.current_subject)
        ensure_subject_dirs(subject_dir)

        settings = self._collect_settings(step_id)
        settings["TIMEBASE_VERSION"] = TIMEBASE_VERSION
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
            settings["subject_id"] = self.current_subject
            backend_session = self._prepare_session_id(step_id, settings)
        elif step_id == "step1b" and not backend_session:
            backend_session = self._guess_backend_session_id()
        elif step_id == "train" and not backend_session:
            backend_session = self._guess_backend_session_id()

        if step_id == "step1b":
            settings["subject_id"] = settings.get("subject_id") or self.current_subject
            if settings.get("WINDOW_SEC") is not None:
                settings["WINDOW_SEC_DEFAULT"] = settings.get("WINDOW_SEC")
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
                    f"{self.current_subject}_*_events.csv",
                )
                if latest_events:
                    settings["events"] = str(latest_events)
        if step_id == "train":
            settings["subject_id"] = settings.get("subject_id") or self.current_subject
            if not settings.get("npz"):
                latest_npz = self._latest_subject_file(
                    subject_dir / "windows",
                    f"{self.current_subject}_*_eeg_windows.npz",
                )
                if latest_npz:
                    settings["npz"] = str(latest_npz)

        if backend_session:
            self.current_session_backend = backend_session
            self.current_session_ui = ui_session_id(
                self.current_subject, backend_session
            )
            self._set_session_label(f"Session: {self.current_session_ui}")

        config_path = subject_dir / "config" / f"{step_id}.json"
        config = build_config(
            project_name=self.current_project,
            subject_id=self.current_subject,
            session_id=self.current_session_ui or "UNKNOWN",
            settings=settings,
            timebase_version=TIMEBASE_VERSION,
        )
        write_json(config_path, config.to_dict())

        self._write_session_snapshot(subject_dir, config.to_dict(), step_id)

        args = [str(script_info.path), "--config", str(config_path)]
        args.extend(self._collect_step_args(step_id))
        cwd = str(self.repo_root)
        if step_id == "step1b" and self.current_session_ui:
            session_dir = session_root(subject_dir, self.current_session_ui)
            ensure_session_dirs(session_dir)
            windows_dir = session_dir / "windows"
            windows_dir.mkdir(parents=True, exist_ok=True)
            cwd = str(windows_dir)

        self.active_step = step_id
        self.active_settings = dict(settings)
        self._set_step_status(step_id, "Running")
        self._append_log(f"Running: {args} (cwd={cwd})")
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
        if self.input_source.currentText() == "CSV Offline":
            defaults["LSL_STREAM_NAME"] = None
            defaults["LSL_STREAM_TYPE"] = None
        else:
            defaults["LSL_STREAM_NAME"] = self._selected_stream_name()
            defaults["LSL_STREAM_TYPE"] = self._selected_stream_type()
        defaults["CSV_OFFLINE_PATH"] = self.csv_path.text().strip() or None
        return defaults

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

    def _stop_process(self) -> None:
        if self.runner.is_running():
            self._append_log("Stopping process...")
            self.runner.stop()

    def _on_process_started(self) -> None:
        if self.active_step:
            self._set_step_status(self.active_step, "Running")
        if self.active_step == "step1":
            self.stream_state_label.setText("Stream: running")

    def _on_process_finished(self, exit_code: int, _exit_status: int) -> None:
        step = self.active_step
        if not step:
            return
        status = "Success" if exit_code == 0 else f"Failed ({exit_code})"
        self._set_step_status(step, status)
        self._append_log(f"Process finished with code {exit_code}")
        if exit_code == 0:
            self._sync_outputs(step)
        self._update_checklist(step)
        if step == "step1":
            self._update_resume_ui()
            self._refresh_status_summary()
        self.active_step = None

    def _set_step_status(self, step_id: str, text: str) -> None:
        label = self.step_status.get(step_id)
        if label:
            label.setText(f"Status: {text}")

    def _append_log(self, line: str) -> None:
        self.log_console.appendPlainText(line)

    def _safe_copy(
        self, src: Path, dest: Path, allow_overwrite: bool
    ) -> Optional[Path]:
        if not src.exists():
            return None
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

    def _sync_step1_outputs(self) -> None:
        if (
            not self.current_project
            or not self.current_subject
            or not self.current_session_backend
        ):
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
        features_src = (
            self.repo_root
            / "data"
            / "processed"
            / f"{subject}_{session}_eeg_features.csv"
        )
        events_src = (
            self.repo_root / "data" / "processed" / f"{subject}_{session}_events.csv"
        )
        autosave_src = (
            self.repo_root
            / "data"
            / "processed"
            / f"{subject}_{session}_events_autosave.csv"
        )
        raw_src = self.repo_root / "data" / "raw" / f"{subject}_{session}_raw.csv"
        meta_src = (
            self.repo_root
            / "data"
            / "processed"
            / f"{subject}_{session}_session_meta.json"
        )
        state_src = self.repo_root / "logs" / f"session_state_{subject}.json"

        self._safe_copy(
            features_src, subject_dir / "features" / features_src.name, allow_overwrite
        )
        self._safe_copy(
            events_src, subject_dir / "events" / events_src.name, allow_overwrite
        )
        self._safe_copy(
            autosave_src, subject_dir / "events" / autosave_src.name, allow_overwrite
        )
        self._safe_copy(raw_src, subject_dir / "raw" / raw_src.name, allow_overwrite)
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
        self._safe_copy(
            autosave_src, session_dir / "events" / autosave_src.name, allow_overwrite
        )
        self._safe_copy(raw_src, session_dir / "raw" / raw_src.name, allow_overwrite)
        self._safe_copy(
            state_src, session_dir / "logs" / state_src.name, allow_overwrite
        )
        self._auto_fill_paths()

    def _sync_step1b_outputs(self) -> None:
        if (
            not self.current_project
            or not self.current_subject
            or not self.current_session_ui
            or not self.current_session_backend
        ):
            return
        subject_dir = subject_root(self.current_project, self.current_subject)
        session_dir = session_root(subject_dir, self.current_session_ui)
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
        events_path = Path(self.event_events_path.text().strip())
        if not events_path.exists():
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
        backend_id = session_backend_id()
        settings["SESSION_ID_OVERRIDE"] = backend_id
        widget = self.fields.get(step_id, {}).get("SESSION_ID_OVERRIDE")
        if isinstance(widget, QLineEdit):
            widget.setText(backend_id)
        return backend_id

    def _read_session_state(self) -> Optional[str]:
        if not self.current_subject:
            return None
        state_path = (
            self.repo_root / "logs" / f"session_state_{self.current_subject}.json"
        )
        if not state_path.exists():
            return None
        try:
            data = json.loads(state_path.read_text())
            return data.get("session_id")
        except Exception:
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
        outputs.append(
            (
                "Features",
                str(
                    self.repo_root
                    / "data"
                    / "processed"
                    / f"{subject}_{session}_eeg_features.csv"
                ),
            )
        )
        outputs.append(
            (
                "Events",
                str(
                    self.repo_root
                    / "data"
                    / "processed"
                    / f"{subject}_{session}_events.csv"
                ),
            )
        )
        outputs.append(
            (
                "Events autosave",
                str(
                    self.repo_root
                    / "data"
                    / "processed"
                    / f"{subject}_{session}_events_autosave.csv"
                ),
            )
        )
        outputs.append(
            (
                "Session meta",
                str(
                    self.repo_root
                    / "data"
                    / "processed"
                    / f"{subject}_{session}_session_meta.json"
                ),
            )
        )
        outputs.append(
            (
                "Raw",
                str(self.repo_root / "data" / "raw" / f"{subject}_{session}_raw.csv"),
            )
        )
        outputs.append(
            (
                "Session state",
                str(self.repo_root / "logs" / f"session_state_{subject}.json"),
            )
        )
        if self.current_project:
            subject_dir = subject_root(self.current_project, subject)
            outputs.append(
                (
                    "Project features",
                    str(
                        subject_dir
                        / "features"
                        / f"{subject}_{session}_eeg_features.csv"
                    ),
                )
            )
            outputs.append(
                (
                    "Project events",
                    str(subject_dir / "events" / f"{subject}_{session}_events.csv"),
                )
            )
            outputs.append(
                (
                    "Project raw",
                    str(subject_dir / "raw" / f"{subject}_{session}_raw.csv"),
                )
            )
            if self.current_session_ui:
                session_dir = session_root(subject_dir, self.current_session_ui)
                outputs.append(
                    (
                        "Session events",
                        str(session_dir / "events" / f"{subject}_{session}_events.csv"),
                    )
                )
                outputs.append(("Session meta", str(session_dir / "session_meta.json")))
        return outputs

    def _expected_step1b_outputs(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if (
            not self.current_project
            or not self.current_subject
            or not self.current_session_ui
        ):
            return outputs
        subject_dir = subject_root(self.current_project, self.current_subject)
        session_dir = session_root(subject_dir, self.current_session_ui)
        outputs.append(("Window CSV", str(session_dir / "windows" / "eeg_windows.csv")))
        outputs.append(("Window NPZ", str(session_dir / "windows" / "eeg_windows.npz")))
        if self.current_session_backend:
            outputs.append(
                (
                    "Project window CSV",
                    str(
                        subject_dir
                        / "windows"
                        / f"{self.current_subject}_{self.current_session_backend}_eeg_windows.csv"
                    ),
                )
            )
            outputs.append(
                (
                    "Project window NPZ",
                    str(
                        subject_dir
                        / "windows"
                        / f"{self.current_subject}_{self.current_session_backend}_eeg_windows.npz"
                    ),
                )
            )
        return outputs

    def _expected_train_outputs(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if not self.current_project or not self.current_subject:
            return outputs
        subject_dir = subject_root(self.current_project, self.current_subject)
        outputs.append(("Models", str(subject_dir / "models")))
        return outputs

    def _expected_event_outputs(self) -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        if not self.current_project or not self.current_subject:
            return outputs
        subject_dir = subject_root(self.current_project, self.current_subject)
        outputs.append(("Events dir", str(subject_dir / "events")))
        if self.current_session_ui:
            outputs.append(
                (
                    "Session events",
                    str(session_root(subject_dir, self.current_session_ui) / "events"),
                )
            )
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
        if self.current_subject:
            args += ["--subject-id", self.current_subject]
        if self.event_events_path.text().strip():
            args += ["--events", self.event_events_path.text().strip()]
        if self.event_features_path.text().strip():
            args += ["--features", self.event_features_path.text().strip()]
        self.active_step = "event_tools"
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
        if self.current_subject:
            args += ["--subject-id", self.current_subject]
        if self.event_events_path.text().strip():
            args += ["--events", self.event_events_path.text().strip()]
        if self.event_features_path.text().strip():
            args += ["--features", self.event_features_path.text().strip()]
        if self.event_apply_fix.isChecked():
            args.append("--apply")
        if self.event_strict.isChecked():
            args.append("--strict")
        if self.event_json_report.text().strip():
            args += ["--json-report", self.event_json_report.text().strip()]
        self.active_step = "event_tools"
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
    app.setStyle(OutlineStyle(app.style()))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
