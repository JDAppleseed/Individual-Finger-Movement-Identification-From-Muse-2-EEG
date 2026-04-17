from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Any, Optional, Sequence, Tuple

import numpy as np

DEFAULT_PLOT_FPS = 20.0
DEFAULT_PLOT_DISPLAY_FS = 64.0
DEFAULT_PLOT_QUEUE_MAXSIZE = 512
DEFAULT_PLOT_SCALE_MODE = "fixed"
DEFAULT_PLOT_FIXED_YLIM = (-200.0, 200.0)
DEFAULT_PLOT_ROBUST_WINDOW_SEC = 5.0
DEFAULT_PLOT_ROBUST_EMA = 0.2
DEFAULT_PLOT_REFERENCE_OVERLAY = False
DEFAULT_PLOT_WINDOW_SEC = 5.0
DEFAULT_PLOT_STARTUP_TIMEOUT_S = 0.75
DEFAULT_PLOT_CHANNEL_SPACING_UV = 120.0


def force_interactive_matplotlib_backend(logger: logging.Logger) -> None:
    """Force a GUI-capable Matplotlib backend."""
    try:
        import matplotlib  # noqa: WPS433
    except Exception as exc:
        logger.warning("[plot] Matplotlib not available: %s", exc)
        return

    env_backend = os.environ.get("MPLBACKEND")
    preferred_backend = os.environ.get("MUSE_PLOT_BACKEND") or os.environ.get(
        "PLOT_BACKEND"
    )
    logger.info("[plot] env MPLBACKEND=%s", env_backend)
    if preferred_backend:
        logger.info("[plot] preferred backend override=%s", preferred_backend)

    def _has_qt() -> bool:
        try:
            import PyQt5  # noqa: F401

            return True
        except Exception:
            pass
        try:
            import PySide6  # noqa: F401

            return True
        except Exception:
            return False

    def _has_tk() -> bool:
        try:
            import tkinter  # noqa: F401

            return True
        except Exception:
            return False

    non_interactive = {"agg", "pdf", "ps", "svg", "cairo", "template"}
    try:
        current_backend = matplotlib.get_backend()
    except Exception:
        current_backend = None

    if current_backend and str(current_backend).strip().lower() not in non_interactive:
        try:
            matplotlib.interactive(True)
        except Exception:
            pass
        try:
            logger.info("[plot] Using matplotlib backend=%s", matplotlib.get_backend())
        except Exception:
            logger.info("[plot] Using matplotlib backend=(unknown)")
        return
    if env_backend and env_backend.strip().lower() not in non_interactive:
        try:
            matplotlib.interactive(True)
        except Exception:
            pass
        try:
            logger.info("[plot] Using matplotlib backend=%s", matplotlib.get_backend())
        except Exception:
            logger.info("[plot] Using matplotlib backend=(unknown)")
        return

    chosen = None
    if preferred_backend:
        chosen = preferred_backend
    elif _has_tk():
        chosen = "TkAgg"
    elif _has_qt():
        chosen = "QtAgg"
    elif sys.platform == "darwin":
        chosen = "MacOSX"

    if chosen:
        try:
            matplotlib.use(chosen, force=True)
        except Exception as exc:
            logger.warning("[plot] Failed to set backend=%s (%s).", chosen, exc)

    try:
        matplotlib.interactive(True)
    except Exception:
        pass

    try:
        logger.info("[plot] Using matplotlib backend=%s", matplotlib.get_backend())
    except Exception:
        logger.info("[plot] Using matplotlib backend=(unknown)")


def apply_plot_lines(
    lines: list[Any],
    t_arr: np.ndarray,
    y_arr: np.ndarray,
    offsets: np.ndarray,
    plot_channels: int,
) -> None:
    for idx in range(plot_channels):
        lines[idx].set_data(t_arr, y_arr[:, idx] + offsets[idx])
    for idx in range(plot_channels, len(lines)):
        lines[idx].set_data([], [])


def resolve_plot_fixed_ylim(
    value: Optional[Sequence[float]],
    *,
    default_ylim: tuple[float, float] = DEFAULT_PLOT_FIXED_YLIM,
) -> tuple[float, float]:
    if not value or len(value) != 2:
        return float(default_ylim[0]), float(default_ylim[1])
    low = float(value[0])
    high = float(value[1])
    if low == high:
        if low == 0:
            return -200.0, 200.0
        return low - abs(low), low + abs(low)
    return (min(low, high), max(low, high))


def normalize_scale_mode(value: str) -> str:
    val = (value or "").strip().lower()
    if val in {"robust", "robust_auto", "auto"}:
        return "robust"
    return "fixed"


def plot_process_main(
    *,
    sample_queue: mp.Queue,
    stop_flag: mp.Event,
    channel_labels: list[str],
    expected_channels: int,
    plot_window_sec: float,
    plot_fps: float,
    plot_fixed_ylim: tuple[float, float],
    plot_scale: str,
    plot_robust_ema: float,
    plot_reference_overlay: bool,
    plot_channel_spacing_uv: float,
    title: str,
) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler()],
    )
    plot_logger = logging.getLogger("live_eeg_plot")
    try:
        force_interactive_matplotlib_backend(plot_logger)
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception as exc:
        plot_logger.error("[plot] Plot process failed to import matplotlib: %s", exc)
        return

    channel_count = len(channel_labels)
    expected_channels = int(expected_channels or channel_count)
    plot_scale = normalize_scale_mode(plot_scale)
    plot_window_sec = float(plot_window_sec)
    plot_fps = float(plot_fps)
    plot_robust_ema = float(plot_robust_ema)
    plot_fixed_ylim = resolve_plot_fixed_ylim(list(plot_fixed_ylim))
    plot_channel_spacing_uv = float(plot_channel_spacing_uv)

    plt.ion()
    fig, ax = plt.subplots()
    try:
        fig.canvas.manager.set_window_title(title)
    except Exception:
        pass
    lines: list[Any] = []
    for _ in range(channel_count):
        line, = ax.plot([], [], lw=1)
        lines.append(line)
    plot_logger.info("[plot] Created %d line(s) for channels=%s", len(lines), channel_labels)
    spacing_seed = (
        plot_channel_spacing_uv
        if plot_channel_spacing_uv > 0 and np.isfinite(plot_channel_spacing_uv)
        else 120.0
    )
    plot_offsets = np.arange(channel_count, dtype=float) * float(spacing_seed)
    try:
        ax.set_yticks(plot_offsets.tolist())
        ax.set_yticklabels([str(label) for label in channel_labels])
    except Exception:
        pass
    ax.set_title("EEG (uV)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (uV)")

    overlay_lines = []
    if plot_reference_overlay:
        for off in plot_offsets:
            overlay_lines.append(
                ax.axhline(float(off), color="#888888", alpha=0.2, linewidth=0.6)
            )

    base_half = max(abs(plot_fixed_ylim[0]), abs(plot_fixed_ylim[1]))
    ax.set_ylim(float(plot_offsets[0] - base_half), float(plot_offsets[-1] + base_half))
    ax.set_xlim(-plot_window_sec, 0.0)

    times: deque[float] = deque()
    values: deque[np.ndarray] = deque()
    plot_ylim_ema: Optional[Tuple[np.ndarray, np.ndarray]] = None
    last_draw = 0.0
    last_diag = 0.0
    warned_shape = False
    warned_channel_mismatch = False

    def _trim(now_s: float) -> None:
        while times and (now_s - float(times[0])) > plot_window_sec:
            times.popleft()
            values.popleft()

    def _draw(now_s: float) -> None:
        nonlocal plot_ylim_ema, last_draw, plot_offsets, warned_shape
        nonlocal warned_channel_mismatch, last_diag
        if plot_fps > 0 and (now_s - last_draw) < (1.0 / plot_fps):
            return
        last_draw = now_s
        if not times:
            return
        t_arr = np.asarray(times, dtype=float)
        v_arr = np.asarray(values, dtype=float)
        if v_arr.ndim == 3 and v_arr.shape[-1] == 1:
            v_arr = v_arr[:, :, 0]
        if v_arr.ndim == 1:
            v_arr = v_arr[None, :]
        if (
            v_arr.ndim == 2
            and v_arr.shape[0] == expected_channels
            and v_arr.shape[1] != expected_channels
        ):
            v_arr = v_arr.T
        if v_arr.ndim != 2:
            if not warned_shape:
                plot_logger.warning("[plot] Unexpected y shape=%s; skipping draw.", v_arr.shape)
                warned_shape = True
            return

        if v_arr.shape[0] != t_arr.shape[0]:
            n = int(min(v_arr.shape[0], t_arr.shape[0]))
            if n <= 0:
                return
            v_arr = v_arr[:n, :]
            t_arr = t_arr[:n]

        actual_channels = int(v_arr.shape[1])
        if actual_channels != expected_channels and not warned_channel_mismatch:
            plot_logger.warning(
                "[plot] Channel mismatch: expected=%d actual=%d (plotting min=%d).",
                expected_channels,
                actual_channels,
                min(expected_channels, actual_channels),
            )
            warned_channel_mismatch = True

        plot_channels = min(channel_count, actual_channels)
        if plot_channels <= 0:
            return
        if plot_channel_spacing_uv > 0 and np.isfinite(plot_channel_spacing_uv):
            spacing_uv = float(plot_channel_spacing_uv)
        else:
            stds = np.nanstd(v_arr[:, :plot_channels], axis=0)
            median_std = float(np.nanmedian(stds)) if stds.size else 0.0
            spacing_uv = max(120.0, 6.0 * median_std)
            spacing_uv = float(max(80.0, min(400.0, spacing_uv)))

        plot_offsets = np.arange(channel_count, dtype=float) * spacing_uv
        try:
            ax.set_yticks(plot_offsets.tolist())
            ax.set_yticklabels([str(label) for label in channel_labels])
        except Exception:
            pass

        t0 = float(t_arr[-1])
        x = t_arr - t0

        apply_plot_lines(lines, x, v_arr, plot_offsets, plot_channels)
        ax.set_xlim(-plot_window_sec, 0.0)

        if plot_scale == "robust":
            lows = np.nanpercentile(v_arr[:, :plot_channels], 5, axis=0)
            highs = np.nanpercentile(v_arr[:, :plot_channels], 95, axis=0)
            if plot_ylim_ema is None:
                plot_ylim_ema = (lows, highs)
            else:
                alpha = max(0.0, min(1.0, plot_robust_ema))
                plot_ylim_ema = (
                    (1.0 - alpha) * plot_ylim_ema[0] + alpha * lows,
                    (1.0 - alpha) * plot_ylim_ema[1] + alpha * highs,
                )
            low_off = plot_ylim_ema[0][:plot_channels] + plot_offsets[:plot_channels]
            high_off = plot_ylim_ema[1][:plot_channels] + plot_offsets[:plot_channels]
            ax.set_ylim(float(np.min(low_off)), float(np.max(high_off)))
        else:
            lo, hi = float(plot_fixed_ylim[0]), float(plot_fixed_ylim[1])
            ax.set_ylim(float(lo + plot_offsets[0]), float(hi + plot_offsets[-1]))

        if overlay_lines:
            try:
                for idx, line in enumerate(overlay_lines):
                    if idx < len(plot_offsets):
                        line.set_ydata([plot_offsets[idx], plot_offsets[idx]])
                    line.set_alpha(0.2)
            except Exception:
                pass

        if (now_s - last_diag) >= 5.0:
            plot_logger.info(
                "[plot] y_shape=%s channels=%d offsets=%d",
                v_arr.shape,
                plot_channels,
                len(plot_offsets),
            )
            last_diag = now_s

        fig.canvas.draw_idle()
        plt.pause(0.001)

    while not stop_flag.is_set():
        try:
            item = sample_queue.get(timeout=0.1)
        except queue.Empty:
            item = None
        if item is None:
            if not plt.fignum_exists(fig.number):
                break
            continue
        try:
            now_s, sample = item
            now_s = float(now_s)
            sample_arr = np.asarray(sample, dtype=float)
            if sample_arr.ndim != 1 or sample_arr.size != channel_count:
                continue
            times.append(now_s)
            values.append(sample_arr)
            _trim(now_s)
            _draw(now_s)
        except Exception:
            continue


class PlotProcess:
    def __init__(
        self,
        *,
        enabled: bool,
        channel_labels: list[str],
        expected_channels: int,
        plot_window_sec: float,
        plot_fps: float,
        plot_fixed_ylim: tuple[float, float],
        plot_scale: str,
        plot_robust_ema: float,
        plot_reference_overlay: bool,
        plot_channel_spacing_uv: float,
        title: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.dropped = 0
        self._queue: Optional[mp.Queue] = None
        self._stop: Optional[mp.Event] = None
        self._proc: Optional[mp.Process] = None
        if not self.enabled:
            return
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue(maxsize=int(DEFAULT_PLOT_QUEUE_MAXSIZE))
        self._stop = ctx.Event()
        self._proc = ctx.Process(
            target=plot_process_main,
            kwargs={
                "sample_queue": self._queue,
                "stop_flag": self._stop,
                "channel_labels": list(channel_labels),
                "expected_channels": int(expected_channels or len(channel_labels)),
                "plot_window_sec": float(plot_window_sec),
                "plot_fps": float(plot_fps),
                "plot_fixed_ylim": tuple(plot_fixed_ylim),
                "plot_scale": str(plot_scale),
                "plot_robust_ema": float(plot_robust_ema),
                "plot_reference_overlay": bool(plot_reference_overlay),
                "plot_channel_spacing_uv": float(plot_channel_spacing_uv),
                "title": str(title),
            },
            daemon=True,
        )
        self._proc.start()

    def push(self, *, now_s: float, sample: np.ndarray) -> None:
        if not self._queue or not self._proc or not self._proc.is_alive():
            return
        try:
            self._queue.put_nowait((float(now_s), np.asarray(sample, dtype=np.float32)))
        except queue.Full:
            self.dropped += 1
        except Exception:
            return

    def stop(self) -> None:
        if not self._stop or not self._proc:
            return
        try:
            self._stop.set()
        except Exception:
            pass
        try:
            self._proc.join(timeout=1.0)
        except Exception:
            pass
        if self._proc.is_alive():
            try:
                self._proc.terminate()
            except Exception:
                pass

    def is_alive(self) -> bool:
        return bool(self._proc and self._proc.is_alive())


class PlotRingBuffer:
    def __init__(self, maxlen: int) -> None:
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=max(1, int(maxlen)))
        self._lock = threading.Lock()
        self.dropped = 0

    def append(self, now_s: float, sample: np.ndarray) -> None:
        with self._lock:
            if len(self._buf) == self._buf.maxlen:
                self.dropped += 1
            self._buf.append((float(now_s), np.asarray(sample, dtype=np.float32)))

    def pop_left(self) -> Optional[tuple[float, np.ndarray]]:
        with self._lock:
            if not self._buf:
                return None
            return self._buf.popleft()

    def __len__(self) -> int:
        with self._lock:
            return len(self._buf)


class PlotController:
    def __init__(
        self,
        *,
        enabled: bool,
        startup_timeout_s: float,
        channel_labels: list[str],
        expected_channels: int,
        plot_window_sec: float,
        plot_fps: float,
        plot_fixed_ylim: tuple[float, float],
        plot_scale: str,
        plot_robust_ema: float,
        plot_reference_overlay: bool,
        plot_channel_spacing_uv: float,
        title: str,
    ) -> None:
        self.enabled = bool(enabled)
        self.startup_timeout_s = float(startup_timeout_s)
        self.dropped = 0
        self._plotter: Optional[PlotProcess] = None
        self._start_thread: Optional[threading.Thread] = None
        self._start_time: Optional[float] = None
        self._start_done = threading.Event()
        self._disabled_reason: Optional[str] = None
        self._disable_logged = False
        self._lock = threading.Lock()
        self._kwargs = {
            "enabled": True,
            "channel_labels": list(channel_labels),
            "expected_channels": int(expected_channels or len(channel_labels)),
            "plot_window_sec": float(plot_window_sec),
            "plot_fps": float(plot_fps),
            "plot_fixed_ylim": tuple(plot_fixed_ylim),
            "plot_scale": str(plot_scale),
            "plot_robust_ema": float(plot_robust_ema),
            "plot_reference_overlay": bool(plot_reference_overlay),
            "plot_channel_spacing_uv": float(plot_channel_spacing_uv),
            "title": str(title),
        }

    def request_start(self) -> None:
        if not self.enabled or self._start_thread is not None:
            return
        self._start_time = time.monotonic()
        self._start_thread = threading.Thread(target=self._start_worker, daemon=True)
        self._start_thread.start()

    def _start_worker(self) -> None:
        try:
            plotter = PlotProcess(**self._kwargs)
            with self._lock:
                if self._disabled_reason:
                    plotter.stop()
                    return
                self._plotter = plotter
        except Exception as exc:
            with self._lock:
                self._disabled_reason = f"startup_error:{exc}"
        finally:
            self._start_done.set()

    def check_startup_timeout(self, now_mono: float, logger: logging.Logger) -> None:
        if not self.enabled:
            return
        if self._disabled_reason:
            self._log_disabled_once(logger)
            return
        if self._start_time is None or self._start_done.is_set():
            return
        if (now_mono - self._start_time) < self.startup_timeout_s:
            return
        with self._lock:
            if not self._disabled_reason:
                self._disabled_reason = "startup_timeout"
        self._start_done.set()
        self._log_disabled_once(logger)

    def _log_disabled_once(self, logger: logging.Logger) -> None:
        if self._disable_logged:
            return
        reason = self._disabled_reason or "unknown"
        logger.warning("[plot] PLOT DISABLED (%s).", reason)
        self._disable_logged = True

    def get_plotter(self) -> Optional[PlotProcess]:
        if not self.enabled or self._disabled_reason:
            return None
        return self._plotter

    def stop(self) -> None:
        with self._lock:
            self._disabled_reason = self._disabled_reason or "stop"
        if self._plotter is not None:
            self._plotter.stop()


class LiveEEGPlotRuntime:
    def __init__(
        self,
        *,
        enabled: bool,
        nominal_srate: float,
        channel_labels: list[str],
        expected_channels: int,
        title: str,
        plot_display_fs: float = DEFAULT_PLOT_DISPLAY_FS,
        plot_window_sec: float = DEFAULT_PLOT_WINDOW_SEC,
        plot_fps: float = DEFAULT_PLOT_FPS,
        plot_fixed_ylim: tuple[float, float] = DEFAULT_PLOT_FIXED_YLIM,
        plot_scale: str = DEFAULT_PLOT_SCALE_MODE,
        plot_robust_ema: float = DEFAULT_PLOT_ROBUST_EMA,
        plot_reference_overlay: bool = DEFAULT_PLOT_REFERENCE_OVERLAY,
        plot_channel_spacing_uv: float = DEFAULT_PLOT_CHANNEL_SPACING_UV,
        startup_timeout_s: float = DEFAULT_PLOT_STARTUP_TIMEOUT_S,
    ) -> None:
        self.enabled = bool(enabled)
        self.channel_labels = list(channel_labels)
        self.expected_channels = int(expected_channels or len(channel_labels))
        self.plot_display_fs = float(plot_display_fs)
        self.plot_window_sec = float(plot_window_sec)
        self.plot_fps = float(plot_fps)
        self.plot_fixed_ylim = tuple(plot_fixed_ylim)
        self.plot_scale = str(plot_scale)
        self.plot_robust_ema = float(plot_robust_ema)
        self.plot_reference_overlay = bool(plot_reference_overlay)
        self.plot_channel_spacing_uv = float(plot_channel_spacing_uv)
        self.startup_timeout_s = float(startup_timeout_s)
        self.title = str(title)
        self.plot_buffer_len = max(
            16, int(round(self.plot_window_sec * self.plot_display_fs * 2.0))
        )
        self.plot_buffer = (
            PlotRingBuffer(self.plot_buffer_len) if self.enabled else None
        )
        self.plot_controller = (
            PlotController(
                enabled=True,
                startup_timeout_s=self.startup_timeout_s,
                channel_labels=self.channel_labels,
                expected_channels=self.expected_channels,
                plot_window_sec=self.plot_window_sec,
                plot_fps=self.plot_fps,
                plot_fixed_ylim=self.plot_fixed_ylim,
                plot_scale=self.plot_scale,
                plot_robust_ema=self.plot_robust_ema,
                plot_reference_overlay=self.plot_reference_overlay,
                plot_channel_spacing_uv=self.plot_channel_spacing_uv,
                title=self.title,
            )
            if self.enabled
            else None
        )
        self.plot_decim = 1
        if self.enabled and self.plot_display_fs > 0:
            try:
                self.plot_decim = max(
                    1, int(round(float(nominal_srate) / float(self.plot_display_fs)))
                )
            except Exception:
                self.plot_decim = 1
        self._feeder_stop = threading.Event()
        self._feeder_thread: Optional[threading.Thread] = None
        self._external_stop_event: Optional[threading.Event] = None
        self._plot_start_requested = False

    @property
    def plot_start_requested(self) -> bool:
        return self._plot_start_requested

    def start(self, *, external_stop_event: Optional[threading.Event] = None) -> None:
        if not self.enabled or self.plot_buffer is None or self._feeder_thread is not None:
            return
        self._external_stop_event = external_stop_event

        def _plot_feeder() -> None:
            while not self._feeder_stop.is_set():
                if self._external_stop_event is not None and self._external_stop_event.is_set():
                    return
                item = self.plot_buffer.pop_left()
                if item is None:
                    time.sleep(0.01)
                    continue
                plotter = (
                    self.plot_controller.get_plotter()
                    if self.plot_controller is not None
                    else None
                )
                if plotter is None or not plotter.is_alive():
                    continue
                now_s, sample = item
                plotter.push(now_s=float(now_s), sample=np.asarray(sample, dtype=np.float32))

        self._feeder_thread = threading.Thread(target=_plot_feeder, daemon=True)
        self._feeder_thread.start()

    def request_start(self) -> None:
        if not self.enabled or self.plot_controller is None or self._plot_start_requested:
            return
        self.plot_controller.request_start()
        self._plot_start_requested = True

    def append_sample(self, *, sample_index: int, now_s: float, sample: np.ndarray) -> None:
        if (
            not self.enabled
            or self.plot_buffer is None
            or self.plot_decim <= 0
            or (int(sample_index) % int(self.plot_decim)) != 0
        ):
            return
        self.plot_buffer.append(now_s=float(now_s), sample=np.asarray(sample, dtype=np.float32))

    def check_startup_timeout(self, now_mono: float, logger: logging.Logger) -> None:
        if self.plot_controller is not None:
            self.plot_controller.check_startup_timeout(now_mono, logger)

    def get_plotter(self) -> Optional[PlotProcess]:
        if self.plot_controller is None:
            return None
        return self.plot_controller.get_plotter()

    def buffer_depth(self) -> int:
        if self.plot_buffer is None:
            return 0
        return len(self.plot_buffer)

    def dropped_count(self) -> int:
        plotter = self.get_plotter()
        return int(getattr(plotter, "dropped", 0)) + int(
            getattr(self.plot_buffer, "dropped", 0) if self.plot_buffer is not None else 0
        )

    def stop(self) -> None:
        self._feeder_stop.set()
        if self._feeder_thread is not None:
            try:
                self._feeder_thread.join(timeout=0.5)
            except Exception:
                pass
        if self.plot_controller is not None:
            self.plot_controller.stop()
