from __future__ import annotations

from typing import Any, Dict, Optional

import os
import signal
import logging

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


logger = logging.getLogger(__name__)


def _normalize_exit_status(exit_status: Any) -> int:
    """
    Normalize Qt QProcess.ExitStatus to an int for PySide/PyQt compatibility.
    Returns:
      0 = NormalExit
      1 = CrashExit (or non-normal)
    """
    status_val: int
    try:
        if hasattr(exit_status, "value"):
            status_val = int(getattr(exit_status, "value"))
        else:
            status_val = int(exit_status)
    except Exception:
        status_val = 1

    try:
        if exit_status == QProcess.NormalExit:
            status_val = 0
        elif exit_status == QProcess.CrashExit:
            status_val = 1
    except Exception:
        pass

    return 0 if status_val == 0 else 1


class ProcessRunner(QObject):
    line_ready = Signal(str)
    started = Signal()
    finished = Signal(int, int)
    failed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._process = QProcess(self)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.started.connect(self.started.emit)
        self._process.finished.connect(self._handle_finished)
        self._stdout_buffer = ""
        self._stderr_buffer = ""

    def is_running(self) -> bool:
        return self._process.state() != QProcess.NotRunning

    def start(
        self,
        program: str,
        args: list[str],
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        if self.is_running():
            self.failed.emit("Process already running")
            return
        if cwd:
            self._process.setWorkingDirectory(cwd)
        if env:
            process_env = self._process.processEnvironment()
            for key, val in env.items():
                process_env.insert(key, val)
            self._process.setProcessEnvironment(process_env)
        self._process.setProgram(program)
        self._process.setArguments(args)
        self._process.start()

    def stop(self, timeout_ms: int = 2000) -> None:
        if not self.is_running():
            return
        self._process.terminate()
        QTimer.singleShot(timeout_ms, self._kill_if_running)

    def stop_hard(
        self, sigint_timeout_ms: int = 1500, terminate_timeout_ms: int = 1500
    ) -> None:
        if not self.is_running():
            return
        pid = int(self._process.processId() or 0)
        sent_sigint = False
        if os.name != "nt" and pid:
            try:
                os.kill(pid, signal.SIGINT)
                sent_sigint = True
                self.line_ready.emit(f"⚠️ Sent SIGINT to PID {pid} (hard stop).")
            except Exception as exc:
                self.line_ready.emit(
                    f"⚠️ Failed to send SIGINT to PID {pid}: {exc}. Falling back to terminate()."
                )
        if not sent_sigint:
            self._process.terminate()
            self.line_ready.emit("⚠️ Terminate requested (hard stop).")
        QTimer.singleShot(
            sigint_timeout_ms,
            lambda: self._terminate_if_running(terminate_timeout_ms),
        )

    def _kill_if_running(self) -> None:
        if self.is_running():
            self._process.kill()

    def _terminate_if_running(self, terminate_timeout_ms: int) -> None:
        if self.is_running():
            self._process.terminate()
            self.line_ready.emit("⚠️ Terminate requested after SIGINT timeout.")
            QTimer.singleShot(terminate_timeout_ms, self._kill_if_running)

    def _read_stdout(self) -> None:
        data = bytes(self._process.readAllStandardOutput()).decode(errors="replace")
        self._stdout_buffer += data
        self._emit_lines(from_stderr=False)

    def _read_stderr(self) -> None:
        data = bytes(self._process.readAllStandardError()).decode(errors="replace")
        self._stderr_buffer += data
        self._emit_lines(from_stderr=True)

    def _emit_lines(self, from_stderr: bool) -> None:
        buffer = self._stderr_buffer if from_stderr else self._stdout_buffer
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            prefix = "[stderr] " if from_stderr else ""
            self.line_ready.emit(prefix + line.rstrip("\r"))
        if from_stderr:
            self._stderr_buffer = buffer
        else:
            self._stdout_buffer = buffer

    def _handle_finished(self, exit_code: int, exit_status: Any) -> None:
        if self._stdout_buffer.strip():
            self.line_ready.emit(self._stdout_buffer.rstrip("\r\n"))
            self._stdout_buffer = ""
        if self._stderr_buffer.strip():
            self.line_ready.emit("[stderr] " + self._stderr_buffer.rstrip("\r\n"))
            self._stderr_buffer = ""
        exit_status_int = _normalize_exit_status(exit_status)
        logger.info(
            "Process finished: exit_code=%s exit_status=%r normalized=%s",
            exit_code,
            exit_status,
            exit_status_int,
        )
        self.finished.emit(int(exit_code), int(exit_status_int))
