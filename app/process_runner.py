from __future__ import annotations

from typing import Dict, Optional

from PySide6.QtCore import QObject, QProcess, QTimer, Signal


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

    def _kill_if_running(self) -> None:
        if self.is_running():
            self._process.kill()

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

    def _handle_finished(self, exit_code: int, exit_status: int) -> None:
        if self._stdout_buffer.strip():
            self.line_ready.emit(self._stdout_buffer.rstrip("\r\n"))
            self._stdout_buffer = ""
        if self._stderr_buffer.strip():
            self.line_ready.emit("[stderr] " + self._stderr_buffer.rstrip("\r\n"))
            self._stderr_buffer = ""
        self.finished.emit(exit_code, exit_status)

