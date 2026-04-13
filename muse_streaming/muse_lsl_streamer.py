from __future__ import annotations

import argparse
import asyncio
import os
import logging
import signal
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from muse_streaming.config import DEFAULT_LABELS
from utils.channel_labels import (
    append_lsl_channel_metadata,
    normalize_channel_label,
    normalize_channel_labels,
)

import numpy as np
from bitstring import Bits

try:
    from bleak import BleakClient, BleakScanner

    BLEAK_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    BleakClient = None
    BleakScanner = None
    BLEAK_AVAILABLE = False

try:
    from pylsl import StreamInfo, StreamOutlet, local_clock

    LSL_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    StreamInfo = None
    StreamOutlet = None
    local_clock = None
    LSL_AVAILABLE = False


# UUIDs derived from uvicMUSE/Musey constants
MUSE_GATT_ATTR_STREAM_TOGGLE = "273e0001-4c4d-454d-96be-f03bac821358"
MUSE_GATT_ATTR_TP9 = "273e0003-4c4d-454d-96be-f03bac821358"
MUSE_GATT_ATTR_AF7 = "273e0004-4c4d-454d-96be-f03bac821358"
MUSE_GATT_ATTR_AF8 = "273e0005-4c4d-454d-96be-f03bac821358"
MUSE_GATT_ATTR_TP10 = "273e0006-4c4d-454d-96be-f03bac821358"

HARD_ERR_S = 0.25
MAX_FORWARD_SNAP_S = 0.05
MAX_BACKWARD_SNAP_S = 0.05
FUTURE_TOL_S = 0.05


@dataclass
class StreamConfig:
    name: str
    stype: str
    rate: float
    labels: List[str]


class MuseLslStreamer:
    """Stream Muse 2 EEG over BLE and publish to LSL.

    Notes on differences vs UVicMUSE:
    - UVicMUSE streams 5 EEG channels (adds Right AUX). This streamer currently streams the 4
      primary EEG channels (TP9, AF7, AF8, TP10) only.
    - Packet decoding & scaling matches UVicMUSE/Musey for EEG packets.
    """

    def __init__(
        self,
        *,
        name: str = "Muse2-EEG",
        stype: str = "EEG",
        rate: float = 256.0,
        labels: Optional[Iterable[str]] = None,
        device_name: Optional[str] = None,
        mac_address: Optional[str] = None,
        start_delay_s: float = 0.5,
        connect_first: bool = True,
        throttle_logs: bool = True,
        log_fn: Optional[Callable[[str], None]] = None,
        logger: Optional[logging.Logger] = None,
        simulate: bool = False,
        max_pending_packets: int = 128,
    ) -> None:
        self.config = StreamConfig(
            name=name,
            stype=stype,
            rate=float(rate),
            labels=list(labels) if labels is not None else list(DEFAULT_LABELS),
        )
        self._source_id = f"muse2-{os.getpid()}-{int(time.time() * 1000)}"
        self.device_name = device_name
        self.mac_address = mac_address
        self.log_fn = log_fn or (lambda msg: print(msg, flush=True))
        self.logger = logger
        self.simulate = simulate
        self.start_delay_s = float(start_delay_s)
        self.connect_first = bool(connect_first)
        self.throttle_logs = bool(throttle_logs)

        self._client: Optional[BleakClient] = None
        self._outlet: Optional[StreamOutlet] = None
        self._packet_buffer: Dict[int, Dict[str, np.ndarray]] = {}
        self._stop_requested = asyncio.Event()
        self._last_packet_index: Optional[int] = None
        self._max_pending_packets = max(16, int(max_pending_packets))
        self._packets_dropped_overflow = 0
        self._packet_arrival_order = deque()
        self._notify_uuids: List[str] = []
        self._notify_active: Dict[str, bool] = {}
        self._device_ref = None
        self._restart_lock = asyncio.Lock()
        self._restart_in_progress = False
        self._restart_seq = 0
        self._notify_error_last_log = 0.0
        self._last_lsl_backpressure_log = 0.0
        self._startup_notify_grace_s = 5.0
        self._startup_notify_deadline_mono: Optional[float] = None
        self._log_throttle_interval_s = 1.0
        self._log_throttle_state: Dict[str, Dict[str, float]] = {}
        self._timebase_discontinuities = 0

        # Normalized label list used everywhere after startup.
        self._labels_clean: List[str] = []

        # Monotonic timebase + stats
        self._lsl_offset: float = 0.0
        if local_clock is not None:
            try:
                self._lsl_offset = float(local_clock()) - time.monotonic()
            except Exception:
                self._lsl_offset = 0.0
        self._last_pushed_ts: Optional[float] = None
        self._monotonic_epsilon: float = 1e-6
        self._ts_alpha: float = 0.01
        self._timebase_t0_mono: Optional[float] = None
        self._sample_index: int = 0
        self._t0_adjust_total: float = 0.0
        self._last_time_err_s: Optional[float] = None

        # Packet resilience
        self._packet_first_seen: Dict[int, float] = {}
        self._packet_deadline_s: float = 0.030
        self._packet_flush_cap: int = 5

        # Instrumentation
        self._packets_seen_total = 0
        self._packets_dropped_partial = 0
        self._chunks_pushed_total = 0
        self._monotonic_clamps_total = 0
        self._last_push_wallclock: Optional[float] = None
        self._last_push_monotonic: Optional[float] = None
        self._last_push_ts: Optional[float] = None
        self._recent_dt_stats = deque(maxlen=50)
        self._last_notify_monotonic: Optional[float] = None
        self._last_reconnect_attempt = 0.0
        self._reconnect_cooldown_s = 30.0
        self._notify_stall_s = 3.0
        self._heartbeat_interval_s = 1.0
        self._last_heartbeat_monotonic = 0.0
        self._device_rssi: Optional[float] = None
        self._monitor_task: Optional[asyncio.Task] = None

    # -----------------------------
    # Lifecycle / control
    # -----------------------------

    def request_stop(self) -> None:
        self._stop_requested.set()

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger.info(message)
        elif self.log_fn:
            self.log_fn(message)

    def _log_throttled(self, key: str, message: str, *, interval_s: float) -> None:
        if not self.throttle_logs or interval_s <= 0:
            self._log(message)
            return
        now = time.monotonic()
        state = self._log_throttle_state.setdefault(
            key, {"last": 0.0, "count": 0.0}
        )
        state["count"] = float(state.get("count", 0.0) + 1.0)
        if (now - float(state.get("last", 0.0))) < interval_s:
            return
        count = int(state.get("count", 0.0))
        state["count"] = 0.0
        state["last"] = float(now)
        suffix = f" (x{count})" if count > 1 else ""
        self._log(f"{message}{suffix}")

    def _log_kv(self, event: str, **fields: object) -> None:
        parts = [f"event={event}"]
        for key, value in fields.items():
            parts.append(f"{key}={value}")
        self._log("[streamer] " + " ".join(parts))

    def _start_outlet(self) -> None:
        self._outlet = self._build_outlet(self.config)
        print(f"LSL_SOURCE_ID={self._source_id}", flush=True)
        self._last_push_wallclock = time.time()
        self._last_push_monotonic = time.monotonic()
        self._log(
            f"✅ LSL outlet started: name={self.config.name}, type={self.config.stype}, "
            f"ch={len(self.config.labels)}, rate={self.config.rate}"
        )

    def _eeg_channels(self) -> Dict[str, str]:
        return {
            "TP9": MUSE_GATT_ATTR_TP9,
            "AF7": MUSE_GATT_ATTR_AF7,
            "AF8": MUSE_GATT_ATTR_AF8,
            "TP10": MUSE_GATT_ATTR_TP10,
        }

    def _notify_targets(self) -> List[tuple[str, str]]:
        channels = self._eeg_channels()
        targets: List[tuple[str, str]] = []
        for label in self.config.labels:
            uuid = channels.get(label)
            if uuid is None:
                raise RuntimeError(
                    f"Unsupported EEG label: {label}. Supported: {sorted(channels.keys())}"
                )
            targets.append((label, uuid))
            self._notify_active.setdefault(uuid, False)
        return targets

    def _reset_notify_state(self) -> None:
        self._notify_active = {}
        self._notify_uuids = []

    async def run(self) -> None:
        if self.simulate:
            await self._run_simulated()
            return
        if not BLEAK_AVAILABLE:
            raise RuntimeError("Bleak is required for Muse 2 BLE streaming.")
        if not LSL_AVAILABLE:
            raise RuntimeError("pylsl is required for Muse 2 LSL streaming.")

        # Normalize labels once, so downstream logic can't be broken by UI serialization.
        self._normalize_labels()

        if not self.connect_first:
            self._start_outlet()

        self._device_ref = await self._resolve_device()
        self._client = BleakClient(self._device_ref)
        self._log_kv("connect_begin", stage="initial")
        await self._client.connect()
        self._log_kv("connect_ok", stage="initial")
        self._startup_notify_deadline_mono = (
            time.monotonic() + self._startup_notify_grace_s
        )
        self._log("✅ Muse 2 connected")

        if self.connect_first:
            if self.start_delay_s > 0:
                await asyncio.sleep(self.start_delay_s)
            self._start_outlet()
        elif self.start_delay_s > 0:
            await asyncio.sleep(self.start_delay_s)

        await self._start_streaming()
        await self._subscribe_eeg_channels()

        self._monitor_task = asyncio.create_task(self._monitor_loop())
        await self._wait_until_stop()
        await self._shutdown()

    async def _run_simulated(self) -> None:
        if not LSL_AVAILABLE:
            raise RuntimeError("pylsl is required for simulated streaming.")

        self._normalize_labels()
        self._outlet = self._build_outlet(self.config)
        print(f"LSL_SOURCE_ID={self._source_id}", flush=True)
        self._log(
            f"🧪 Simulated LSL outlet started: name={self.config.name}, type={self.config.stype}, "
            f"ch={len(self.config.labels)}, rate={self.config.rate}"
        )

        sample_dt = 1.0 / float(self.config.rate)
        rng = np.random.default_rng(7)
        last_ts: Optional[float] = None

        try:
            while not self._stop_requested.is_set():
                now = float(local_clock()) if local_clock is not None else time.time()
                if last_ts is None:
                    last_ts = now
                ts = [last_ts + (i + 1) * sample_dt for i in range(12)]
                last_ts = ts[-1]
                chunk = rng.normal(0, 1, size=(12, len(self.config.labels))).astype(
                    np.float32
                )
                self._outlet.push_chunk(chunk.tolist(), ts)
                await asyncio.sleep(sample_dt * 12)
        finally:
            self._log("ℹ️ Simulated Muse 2 streamer stopped")

    async def _wait_until_stop(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(0.1)

    async def _shutdown(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        if self._client and getattr(self._client, "is_connected", False):
            await self._stop_notifications(reason="shutdown")
            await self._safe_disconnect(reason="shutdown")
        if self._outlet is not None:
            close_fn = getattr(self._outlet, "close_stream", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    pass
            self._outlet = None
        self._log("ℹ️ Muse 2 streamer stopped")

    # -----------------------------
    # BLE discovery / selection
    # -----------------------------

    def _dev_name_guess(self, dev) -> str:
        """Best-effort name retrieval across Bleak versions/platforms."""
        name = getattr(dev, "name", None)
        if name:
            return str(name)

        md = getattr(dev, "metadata", {}) or {}
        ln = md.get("local_name") or md.get("name")
        if ln:
            return str(ln)

        return "None"

    def _is_muse_candidate(self, dev, want: str) -> bool:
        """Return True if BLE device matches target criteria."""
        name_guess = (self._dev_name_guess(dev) or "").lower()
        addr = (getattr(dev, "address", "") or "").lower()

        if self.device_name:
            # Explicit match: allow either name or address partial match.
            return want in name_guess or want in addr

        # Default: accept any muse-ish name.
        if "muse" in name_guess:
            return True

        # Very weak fallback (kept only for compatibility)
        if "muse" in addr:
            return True

        return False

    async def _resolve_device(self):
        if self.mac_address:
            # Allow directly passing the address/UUID string to BleakClient.
            return self.mac_address

        if BleakScanner is None:
            raise RuntimeError("BleakScanner unavailable; cannot scan for Muse 2.")

        want = (self.device_name or "muse").lower()

        # UVicMUSE-style repeated scanning. Multiple short rounds tend to be more reliable on macOS.
        self._log("🔎 Scanning for Muse 2 over BLE...")
        rounds = 5
        round_timeout = 2.0

        seen: Dict[str, object] = {}  # address -> BLEDevice

        for i in range(rounds):
            devices = await BleakScanner.discover(timeout=round_timeout)
            self._log(f"🔎 BLE scan round {i+1}/{rounds} found {len(devices)} devices")

            for d in devices:
                addr = (getattr(d, "address", "") or "").strip()
                if addr:
                    seen[addr] = d

            # Early exit if we already have a match; do one extra quick scan to improve name visibility.
            if any(self._is_muse_candidate(dev, want) for dev in seen.values()):
                devices2 = await BleakScanner.discover(timeout=1.0)
                for d in devices2:
                    addr = (getattr(d, "address", "") or "").strip()
                    if addr:
                        seen[addr] = d
                break

        all_devices = list(seen.values())
        self._log(f"🔎 BLE combined results: {len(all_devices)} unique devices")
        for d in all_devices[:30]:
            self._log(f"  - {self._dev_name_guess(d)} | {getattr(d, 'address', None)}")

        candidates = [d for d in all_devices if self._is_muse_candidate(d, want)]
        if not candidates:
            raise RuntimeError(
                "No Muse device found over BLE.\n"
                "- Ensure the Muse is on and not connected to another app.\n"
                "- On macOS, device names can appear as None intermittently.\n"
                "- If this persists, pass mac_address from your Bleak scan output."
            )

        def _safe_rssi(dev) -> float:
            # RSSI placement varies by Bleak version/platform.
            rssi = getattr(dev, "rssi", None)
            if isinstance(rssi, (int, float)):
                return float(rssi)
            md = getattr(dev, "metadata", {}) or {}
            rssi2 = md.get("rssi", None)
            return float(rssi2) if isinstance(rssi2, (int, float)) else -999.0

        candidates.sort(key=_safe_rssi, reverse=True)
        chosen = candidates[0]
        self._device_rssi = _safe_rssi(chosen)
        self._log(
            f"✅ Selected BLE device: {self._dev_name_guess(chosen)} | {getattr(chosen,'address',None)}"
        )
        return chosen

    # -----------------------------
    # Muse protocol
    # -----------------------------

    async def _start_streaming(self) -> None:
        if not self._client:
            raise RuntimeError("Muse client not connected.")
        # Matches uvicMUSE resume(): [0x02, 0x64, 0x0A]
        start_cmd = bytearray([0x02, 0x64, 0x0A])
        await self._client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, start_cmd)

    def _normalize_labels(self) -> None:
        raw = list(self.config.labels) if self.config.labels is not None else []
        if not raw:
            self._log(
                f"⚠️ [streamer] labels empty; using DEFAULT_LABELS: {DEFAULT_LABELS}"
            )
            raw = list(DEFAULT_LABELS)
        deduped = normalize_channel_labels(raw, dedupe=True)
        if not deduped:
            self._log(
                f"⚠️ [streamer] normalized labels empty; using DEFAULT_LABELS: {DEFAULT_LABELS}"
            )
            deduped = normalize_channel_labels(DEFAULT_LABELS, dedupe=True)

        self._labels_clean = deduped
        self._log(f"🧾 Raw labels: {raw}")
        self._log(f"🧾 Normalized labels: {self._labels_clean}")

        # IMPORTANT: overwrite config.labels so the rest of the pipeline (outlet + buffering)
        # uses normalized labels.
        self.config.labels = list(self._labels_clean)

    async def _subscribe_eeg_channels(self) -> None:
        if not self._client:
            raise RuntimeError("Muse client not connected.")

        targets = self._notify_targets()
        self._notify_uuids = [uuid for _, uuid in targets]
        started = 0
        skipped = 0
        for label, uuid in targets:
            if self._notify_active.get(uuid, False):
                skipped += 1
                self._log_kv(
                    "notify_start_skip",
                    uuid=uuid,
                    label=label,
                    reason="already_active",
                )
                continue
            self._log_kv("notify_start", uuid=uuid, label=label)
            await self._client.start_notify(uuid, self._make_notify_handler(label))
            self._notify_active[uuid] = True
            started += 1
            self._log_kv("notify_started", uuid=uuid, label=label)

        if started:
            self._log("✅ EEG notifications enabled")
        elif skipped:
            self._log("✅ EEG notifications already active")

    def _make_notify_handler(self, label: str):
        def _handler(_sender: int, data: bytearray) -> None:
            try:
                self._handle_eeg_packet(label, data)
            except Exception:
                now = time.monotonic()
                if (now - self._notify_error_last_log) >= 1.0:
                    self._notify_error_last_log = now
                    self._log(
                        "⚠️ [streamer] notify handler error:\n"
                        + traceback.format_exc()
                    )

        return _handler

    async def _stop_notifications(self, *, reason: str) -> None:
        if not self._client:
            return
        try:
            targets = self._notify_targets()
        except Exception:
            return
        for label, uuid in targets:
            if not self._notify_active.get(uuid, False):
                self._log_kv(
                    "notify_stop_skip",
                    uuid=uuid,
                    label=label,
                    reason="not_active",
                )
                continue
            try:
                self._log_kv(
                    "notify_stop",
                    uuid=uuid,
                    label=label,
                    reason=reason,
                )
                await self._client.stop_notify(uuid)
                self._notify_active[uuid] = False
                self._log_kv(
                    "notify_stopped",
                    uuid=uuid,
                    label=label,
                    reason=reason,
                )
            except Exception as exc:
                self._log_kv(
                    "notify_stop_error",
                    uuid=uuid,
                    label=label,
                    reason=reason,
                    error=str(exc),
                    exc_type=type(exc).__name__,
                )

    async def _safe_disconnect(self, *, reason: str) -> None:
        if not self._client:
            return
        try:
            if getattr(self._client, "is_connected", False):
                self._log_kv("disconnect_begin", reason=reason)
                await self._client.disconnect()
                self._log_kv("disconnect_ok", reason=reason)
            else:
                self._log_kv("disconnect_skip", reason=reason, detail="not_connected")
        except Exception as exc:
            self._log_kv(
                "disconnect_error",
                reason=reason,
                error=str(exc),
                exc_type=type(exc).__name__,
            )

    async def _reconnect_client(self, *, reason: str) -> None:
        device = self._device_ref
        if device is None:
            device = await self._resolve_device()
            self._device_ref = device
        self._log_kv("connect_begin", reason=reason, stage="reconnect")
        self._client = BleakClient(device)
        self._reset_notify_state()
        await self._client.connect()
        self._log_kv("connect_ok", reason=reason, stage="reconnect")
        self._startup_notify_deadline_mono = (
            time.monotonic() + self._startup_notify_grace_s
        )

    def _handle_eeg_packet(self, label: str, data: bytearray) -> None:
        if self._outlet is None:
            return

        packet_index, samples = _decode_eeg_packet(data)
        self._last_notify_monotonic = time.monotonic()
        self._packets_seen_total += 1
        self._ingest_decoded_packet(label, packet_index, samples)

    def _ingest_decoded_packet(
        self, label: str, packet_index: int, samples: np.ndarray
    ) -> None:
        if self._outlet is None:
            return
        label = normalize_channel_label(label)
        slot = self._packet_buffer.setdefault(packet_index, {})
        slot[label] = samples
        if packet_index not in self._packet_first_seen:
            now = float(local_clock()) if local_clock is not None else time.time()
            self._packet_first_seen[packet_index] = now
            self._packet_arrival_order.append(packet_index)

        if len(self._packet_buffer) > self._max_pending_packets:
            if self._packet_arrival_order:
                idx_to_drop = self._packet_arrival_order.popleft()
                # Packet might have been flushed already, so check for existence.
                if idx_to_drop in self._packet_buffer:
                    self._packet_buffer.pop(idx_to_drop)
                    self._packet_first_seen.pop(idx_to_drop, None)
                    self._packets_dropped_overflow += 1
                    self._log(
                        f"⚠️ [streamer] packet buffer overflow; dropped packet {idx_to_drop}, total dropped={self._packets_dropped_overflow}"
                    )

        # Wait until we have all channels for this packet index.
        if len(slot) < len(self.config.labels):
            self._flush_stale_packets()
            return

        self._flush_packet(packet_index, slot, partial=False)
        self._flush_stale_packets()

    def _release_packet(self, packet_index: int) -> None:
        n = 12
        self._packet_buffer.pop(packet_index, None)
        self._packet_first_seen.pop(packet_index, None)
        if self._packet_arrival_order:
            if self._packet_arrival_order[0] == packet_index:
                self._packet_arrival_order.popleft()
            while (
                self._packet_arrival_order
                and self._packet_arrival_order[0] not in self._packet_buffer
            ):
                self._packet_arrival_order.popleft()
        self._last_packet_index = packet_index

    def _drop_partial_packet(
        self, packet_index: int, slot: Dict[str, np.ndarray], missing: List[str]
    ) -> None:
        n = 12
        if slot:
            n = int(next(iter(slot.values())).shape[0])
        # Preserve stream timing even when a packet is discarded.
        self._build_timestamps(n)
        self._packets_dropped_partial += 1
        self._log_throttled(
            "partial_packet",
            (
                "⚠️ [streamer] partial packet dropped "
                f"index={packet_index} missing={missing}"
            ),
            interval_s=self._log_throttle_interval_s,
        )
        self._release_packet(packet_index)

    def _flush_packet(
        self, packet_index: int, slot: Dict[str, np.ndarray], partial: bool
    ) -> None:
        if self._outlet is None:
            return

        n = 12
        if slot:
            n = int(next(iter(slot.values())).shape[0])

        missing = [lab for lab in self.config.labels if lab not in slot]
        if partial or missing:
            self._drop_partial_packet(packet_index, slot, missing)
            return

        ordered = []
        for i in range(n):
            ordered.append([slot[ch][i] for ch in self.config.labels])

        ts = self._build_timestamps(n)
        ts = self._enforce_monotonic(ts)
        prev_last_ts = self._last_push_ts
        self._outlet.push_chunk(ordered, ts)
        self._chunks_pushed_total += 1
        self._last_push_wallclock = time.time()
        self._last_push_monotonic = time.monotonic()
        self._last_push_ts = ts[-1] if ts else self._last_push_ts
        self._record_dt_stats(ts, prev_last_ts)
        self._release_packet(packet_index)

    def _flush_stale_packets(self, now_time: Optional[float] = None) -> None:
        if not self._packet_first_seen:
            return
        now = (
            float(now_time)
            if now_time is not None
            else (float(local_clock()) if local_clock is not None else time.time())
        )
        stale = [
            (idx, first_seen)
            for idx, first_seen in self._packet_first_seen.items()
            if (now - first_seen) > self._packet_deadline_s
        ]
        if not stale:
            return
        stale.sort(key=lambda x: x[1])
        flushed = 0
        for packet_index, _ in stale:
            if flushed >= self._packet_flush_cap:
                break
            slot = self._packet_buffer.get(packet_index)
            if slot is None:
                self._packet_first_seen.pop(packet_index, None)
                continue
            self._flush_packet(packet_index, slot, partial=True)
            flushed += 1

    def _now_mono(self) -> float:
        if local_clock is None:
            return time.time()
        return time.monotonic()

    def _mono_to_lsl(self, t_mono: float) -> float:
        if local_clock is None:
            return float(t_mono)
        return float(t_mono + self._lsl_offset)

    def _apply_timebase_adjust(self, delta_s: float) -> None:
        if self._timebase_t0_mono is None:
            return
        if delta_s == 0.0:
            return
        self._timebase_t0_mono = float(self._timebase_t0_mono + delta_s)
        self._t0_adjust_total += float(delta_s)

    def _build_timestamps(self, n: int) -> List[float]:
        if n <= 0:
            return []
        fs = float(self.config.rate)
        now_mono = float(self._now_mono())
        if self._timebase_t0_mono is None:
            self._timebase_t0_mono = now_mono - ((self._sample_index + n - 1) / fs)
            self._last_time_err_s = 0.0
        else:
            expected_end = self._timebase_t0_mono + (
                (self._sample_index + n - 1) / fs
            )
            err = float(now_mono - expected_end)
            self._last_time_err_s = err
            if abs(err) > HARD_ERR_S:
                if err > 0:
                    snap = min(err, MAX_FORWARD_SNAP_S)
                    self._apply_timebase_adjust(snap)
                    self._timebase_discontinuities += 1
                    self._log_throttled(
                        "timebase_discontinuity",
                        (
                            "⚠️ [streamer] timebase discontinuity forward "
                            f"snap={snap:.4f}s err={err:.4f}s"
                        ),
                        interval_s=self._log_throttle_interval_s,
                    )
                else:
                    snap = max(err, -MAX_BACKWARD_SNAP_S)
                    self._apply_timebase_adjust(snap)
                    self._timebase_discontinuities += 1
                    self._log_throttled(
                        "timebase_discontinuity",
                        (
                            "⚠️ [streamer] timebase discontinuity backward "
                            f"snap={snap:.4f}s err={err:.4f}s"
                        ),
                        interval_s=self._log_throttle_interval_s,
                    )
            elif err < -FUTURE_TOL_S:
                snap = max(err, -MAX_BACKWARD_SNAP_S)
                self._apply_timebase_adjust(snap)
            else:
                self._apply_timebase_adjust(self._ts_alpha * err)

        ts_mono = [
            self._timebase_t0_mono + ((self._sample_index + i) / fs) for i in range(n)
        ]
        self._sample_index += n
        return [self._mono_to_lsl(t) for t in ts_mono]

    def _enforce_monotonic(self, ts: List[float]) -> List[float]:
        if not ts:
            return ts
        if self._last_pushed_ts is None:
            self._last_pushed_ts = float(ts[-1])
            return ts
        out: List[float] = []
        prev = float(self._last_pushed_ts)
        for t in ts:
            t = float(t)
            if t <= prev:
                t = prev + self._monotonic_epsilon
                self._monotonic_clamps_total += 1
            out.append(t)
            prev = t
        self._last_pushed_ts = prev
        return out

    def _record_dt_stats(self, ts: List[float], prev_last_ts: Optional[float]) -> None:
        if not ts:
            return
        diffs = []
        if prev_last_ts is not None:
            diffs.append(float(ts[0] - prev_last_ts))
        for i in range(1, len(ts)):
            diffs.append(float(ts[i] - ts[i - 1]))
        if not diffs:
            return
        self._recent_dt_stats.append((min(diffs), max(diffs)))

    async def _monitor_loop(self) -> None:
        while not self._stop_requested.is_set():
            now_mono = time.monotonic()
            self._emit_heartbeat(now_mono)
            await self._maybe_reconnect(now_mono)
            await asyncio.sleep(0.2)

    async def _maybe_reconnect(self, now_mono: float) -> None:
        notify_stall = False
        push_stall = False
        last_notify_age_s = None
        last_push_age_s = None
        if self._last_notify_monotonic is not None:
            last_notify_age_s = float(now_mono - self._last_notify_monotonic)
            notify_stall = last_notify_age_s >= self._notify_stall_s
        if self._last_push_monotonic is not None:
            last_push_age_s = float(now_mono - self._last_push_monotonic)
            push_stall = last_push_age_s >= self._notify_stall_s
        if (
            self._last_notify_monotonic is None
            and self._startup_notify_deadline_mono is not None
            and now_mono >= self._startup_notify_deadline_mono
        ):
            notify_stall = True
        if not notify_stall and not push_stall:
            return
        if push_stall and not notify_stall:
            if (
                now_mono - self._last_lsl_backpressure_log
            ) >= self._log_throttle_interval_s:
                self._last_lsl_backpressure_log = now_mono
                self._log(
                    "⚠️ [streamer] LSL push stall while BLE notifications active; "
                    "skipping BLE restart"
                )
                self._log_kv(
                    "lsl_backpressure",
                    last_notify_age_s=(
                        f"{last_notify_age_s:.3f}"
                        if last_notify_age_s is not None
                        else "n/a"
                    ),
                    last_push_age_s=(
                        f"{last_push_age_s:.3f}" if last_push_age_s is not None else "n/a"
                    ),
                )
            return
        if (now_mono - self._last_reconnect_attempt) < self._reconnect_cooldown_s:
            return
        if self._restart_lock.locked():
            return
        self._last_reconnect_attempt = now_mono
        reason = "notify_stall" if notify_stall else "push_stall"
        if notify_stall:
            self._log("⚠️ [streamer] no BLE notifications; attempting restart")
        else:
            self._log("⚠️ [streamer] no LSL pushes; attempting restart")
        async with self._restart_lock:
            self._restart_in_progress = True
            self._restart_seq += 1
            seq = self._restart_seq
            self._log_kv("restart_begin", reason=reason, seq=seq)
            try:
                await self._restart_streaming(reason=reason, seq=seq)
            finally:
                self._restart_in_progress = False

    async def _restart_streaming(self, *, reason: str, seq: int) -> None:
        if self._client is None:
            self._log_kv("restart_skip", reason=reason, seq=seq, detail="no_client")
            return
        await self._stop_notifications(reason=reason)
        await self._safe_disconnect(reason=reason)
        stage = "reconnect"
        try:
            await self._reconnect_client(reason=reason)
            stage = "start_streaming"
            self._log_kv("stream_start_begin", reason=reason, seq=seq)
            await self._start_streaming()
            self._log_kv("stream_start_ok", reason=reason, seq=seq)
            stage = "subscribe"
            await self._subscribe_eeg_channels()
            self._log_kv("restart_complete", reason=reason, seq=seq)
        except Exception as exc:
            if stage == "reconnect":
                self._log(f"⚠️ [streamer] reconnect failed: {exc}")
                self._device_ref = None
            else:
                self._log(f"⚠️ [streamer] restart failed: {exc}")
            self._log_kv(
                "restart_error",
                reason=reason,
                seq=seq,
                stage=stage,
                error=str(exc),
                exc_type=type(exc).__name__,
            )

    def _emit_heartbeat(self, now_mono: float) -> None:
        if (now_mono - self._last_heartbeat_monotonic) < self._heartbeat_interval_s:
            return
        self._last_heartbeat_monotonic = now_mono
        dt_min_ms = None
        dt_max_ms = None
        if self._recent_dt_stats:
            mins = [m for m, _ in self._recent_dt_stats]
            maxs = [m for _, m in self._recent_dt_stats]
            dt_min_ms = float(min(mins) * 1000.0)
            dt_max_ms = float(max(maxs) * 1000.0)

        last_notify_age_s = None
        if self._last_notify_monotonic is not None:
            last_notify_age_s = float(now_mono - self._last_notify_monotonic)

        no_push_for = None
        if self._last_push_monotonic is not None:
            no_push_for = float(now_mono - self._last_push_monotonic)

        last_push_age_s = None
        if self._last_push_monotonic is not None:
            last_push_age_s = float(now_mono - self._last_push_monotonic)

        is_connected = bool(self._client and getattr(self._client, "is_connected", False))
        notify_active_count = sum(1 for active in self._notify_active.values() if active)
        self._log_kv(
            "heartbeat",
            is_connected=is_connected,
            restart_in_progress=self._restart_in_progress,
            last_notify_age_s=(
                f"{last_notify_age_s:.3f}" if last_notify_age_s is not None else "n/a"
            ),
            last_push_age_s=(
                f"{last_push_age_s:.3f}" if last_push_age_s is not None else "n/a"
            ),
            notify_active_count=notify_active_count,
            restart_seq=self._restart_seq,
        )

        rssi = self._device_rssi
        last_ts = self._last_push_ts
        time_err_s = self._last_time_err_s
        time_err_fmt = f"{time_err_s:.4f}" if time_err_s is not None else "n/a"
        msg = (
            "[streamer] heartbeat: "
            f"chunks={self._chunks_pushed_total} "
            f"packets={self._packets_seen_total} "
            f"partial_dropped={self._packets_dropped_partial} "
            f"clamps={self._monotonic_clamps_total} "
            f"dropped={self._packets_dropped_overflow} "
            f"last_ts={last_ts if last_ts is not None else 'n/a'} "
            f"dt_min_ms={dt_min_ms if dt_min_ms is not None else 'n/a'} "
            f"dt_max_ms={dt_max_ms if dt_max_ms is not None else 'n/a'} "
            f"time_err_s={time_err_fmt} "
            f"t0_adj_total={self._t0_adjust_total:.4f} "
            f"rssi={rssi if rssi is not None else 'n/a'}"
        )
        if no_push_for is not None and no_push_for > self._heartbeat_interval_s:
            msg += f" no_push_for={no_push_for:.2f}s"
        self._log(msg)

    # -----------------------------
    # LSL outlet
    # -----------------------------

    def _build_outlet(self, config: StreamConfig) -> StreamOutlet:
        if not config.labels:
            raise RuntimeError(
                "No EEG labels available; refusing to create ch=0 LSL stream. "
                "Check CLI/config overlay."
            )
        info = StreamInfo(
            config.name,
            config.stype,
            len(config.labels),
            config.rate,
            "float32",
            self._source_id,
        )
        desc = info.desc()
        append_lsl_channel_metadata(desc, config.labels, unit="microvolts", channel_type="EEG")
        return StreamOutlet(info, chunk_size=12)


def _decode_eeg_packet(packet: bytearray) -> tuple[int, np.ndarray]:
    """Decode a Muse EEG packet.

    Adapted from uvicMUSE/Musey packet decoding logic.

    Each packet encodes a 16-bit packet index followed by 12 samples at 12-bit resolution.
    """
    bits = Bits(bytes=packet)
    pattern = (
        "uint:16,uint:12,uint:12,uint:12,uint:12,uint:12,uint:12,"
        "uint:12,uint:12,uint:12,uint:12,uint:12,uint:12"
    )
    values = bits.unpack(pattern)
    packet_index = int(values[0])
    data = np.asarray(values[1:], dtype=np.float32)
    # 12-bit values on 2mVpp range (Muse scaling)
    data = 0.48828125 * (data - 2048)
    return packet_index, data


def install_signal_handlers(streamer: MuseLslStreamer) -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    for sig in (signal.SIGINT, signal.SIGTERM):
        if loop is not None:
            try:
                loop.add_signal_handler(sig, streamer.request_stop)
                continue
            except NotImplementedError:
                pass
        try:
            signal.signal(sig, lambda _sig, _frame: streamer.request_stop())
        except Exception:
            # e.g. on Windows or when signal handlers are unavailable
            pass


def _self_test_monotonic() -> None:
    class _DummyOutlet:
        def __init__(self):
            self.ts = []

        def push_chunk(self, _data, ts):
            self.ts.extend(ts)

    streamer = MuseLslStreamer(log_fn=lambda _msg: None)
    streamer._outlet = _DummyOutlet()
    labels = list(streamer.config.labels)
    rng = np.random.default_rng(42)
    for packet_index in range(100):
        missing_label = None
        if rng.random() < 0.15:
            missing_label = rng.choice(labels)
        for label in labels:
            if label == missing_label:
                continue
            samples = rng.normal(0, 1, size=(12,)).astype(np.float32)
            streamer._ingest_decoded_packet(label, packet_index, samples)
        if missing_label is not None:
            first_seen = streamer._packet_first_seen.get(packet_index, time.time())
            streamer._flush_stale_packets(
                now_time=first_seen + streamer._packet_deadline_s + 0.01
            )

    ts = streamer._outlet.ts
    monotonic_ok = all(ts[i] > ts[i - 1] for i in range(1, len(ts)))
    print(f"monotonic_ok={'true' if monotonic_ok else 'false'}")


def _configure_cli_logger(level: str) -> logging.Logger:
    logger = logging.getLogger("muse_streaming.streamer")
    logger.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    logger.propagate = False
    return logger


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Muse 2 BLE -> LSL streamer")
    parser.add_argument(
        "--name",
        "--stream-name",
        dest="name",
        type=str,
        default="Muse2-EEG",
        help="LSL stream name",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=256.0,
        help="Nominal sampling rate (Hz)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="TP9,AF7,AF8,TP10",
        help="Comma-separated channel labels",
    )
    parser.add_argument(
        "--device-name",
        type=str,
        default="Muse",
        help="Optional BLE device name hint",
    )
    parser.add_argument("--mac-address", type=str, default=None, help="Optional BLE MAC")
    parser.add_argument(
        "--start-delay-s",
        type=float,
        default=0.5,
        help="Delay after BLE connect before opening LSL outlet",
    )
    parser.add_argument(
        "--connect-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Connect BLE before starting the LSL outlet",
    )
    parser.add_argument(
        "--throttle-logs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rate-limit high-frequency logs (heartbeat/timebase/partials)",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--sim", action="store_true", help="Run simulated stream")
    args = parser.parse_args(argv)

    if os.environ.get("MUSE_LSL_SELF_TEST") == "1":
        _self_test_monotonic()
        return 0

    logger = _configure_cli_logger(args.log_level)
    labels = [label.strip() for label in args.labels.split(",") if label.strip()]
    labels = labels if labels else None
    streamer = MuseLslStreamer(
        name=args.name,
        stype="EEG",
        rate=float(args.rate),
        labels=labels,
        device_name=args.device_name or None,
        mac_address=args.mac_address,
        start_delay_s=float(args.start_delay_s),
        connect_first=bool(args.connect_first),
        throttle_logs=bool(args.throttle_logs),
        simulate=bool(args.sim),
        logger=logger,
    )

    async def _run() -> None:
        install_signal_handlers(streamer)
        await streamer.run()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logger.error(f"❌ Muse streamer failed: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
