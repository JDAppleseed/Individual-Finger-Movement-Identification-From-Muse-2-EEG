from __future__ import annotations

import asyncio
import os
import logging
import signal
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

from muse_streaming.config import DEFAULT_LABELS

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


def normalize_label(label: str) -> str:
    """Normalize labels from UI/config to stable uppercase tokens."""
    s = (label or "").strip()
    # Strip multiple layers, e.g. "'TP9'" or "\"TP9\""
    while len(s) >= 2 and ((s[0] == s[-1] == '"') or (s[0] == s[-1] == "'")):
        s = s[1:-1].strip()
    return s.upper()


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
        self.device_name = device_name
        self.mac_address = mac_address
        self.log_fn = log_fn or (lambda msg: print(msg, flush=True))
        self.logger = logger
        self.simulate = simulate

        self._client: Optional[BleakClient] = None
        self._outlet: Optional[StreamOutlet] = None
        self._packet_buffer: Dict[int, Dict[str, np.ndarray]] = {}
        self._stop_requested = asyncio.Event()
        self._last_packet_index: Optional[int] = None
        self._max_pending_packets = max(16, int(max_pending_packets))
        self._packets_dropped_overflow = 0

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
        self._packets_flushed_partial = 0
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

        self._outlet = self._build_outlet(self.config)
        self._last_push_wallclock = time.time()
        self._last_push_monotonic = time.monotonic()
        self._log(
            f"✅ LSL outlet started: name={self.config.name}, type={self.config.stype}, "
            f"ch={len(self.config.labels)}, rate={self.config.rate}"
        )

        device = await self._resolve_device()
        self._client = BleakClient(device)
        await self._client.connect()
        self._log("✅ Muse 2 connected")

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
        if self._client and getattr(self._client, "is_connected", False):
            await self._client.disconnect()
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
        raw = list(self.config.labels)
        clean = [normalize_label(x) for x in raw]
        # Preserve order but dedupe.
        seen = set()
        deduped: List[str] = []
        for c in clean:
            if c and c not in seen:
                deduped.append(c)
                seen.add(c)

        self._labels_clean = deduped
        self._log(f"🧾 Raw labels: {raw}")
        self._log(f"🧾 Normalized labels: {self._labels_clean}")

        # IMPORTANT: overwrite config.labels so the rest of the pipeline (outlet + buffering)
        # uses normalized labels.
        self.config.labels = list(self._labels_clean)

    async def _subscribe_eeg_channels(self) -> None:
        if not self._client:
            raise RuntimeError("Muse client not connected.")

        channels = {
            "TP9": MUSE_GATT_ATTR_TP9,
            "AF7": MUSE_GATT_ATTR_AF7,
            "AF8": MUSE_GATT_ATTR_AF8,
            "TP10": MUSE_GATT_ATTR_TP10,
        }

        for label in self.config.labels:
            uuid = channels.get(label)
            if uuid is None:
                raise RuntimeError(
                    f"Unsupported EEG label: {label}. Supported: {sorted(channels.keys())}"
                )
            await self._client.start_notify(uuid, self._make_notify_handler(label))

        self._log("✅ EEG notifications enabled")

    def _make_notify_handler(self, label: str):
        def _handler(_sender: int, data: bytearray) -> None:
            self._handle_eeg_packet(label, data)

        return _handler

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
        label = normalize_label(label)
        slot = self._packet_buffer.setdefault(packet_index, {})
        slot[label] = samples
        if packet_index not in self._packet_first_seen:
            now = float(local_clock()) if local_clock is not None else time.time()
            self._packet_first_seen[packet_index] = now

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

    def _flush_packet(
        self, packet_index: int, slot: Dict[str, np.ndarray], partial: bool
    ) -> None:
        if self._outlet is None:
            return

        n = 12
        if slot:
            n = int(next(iter(slot.values())).shape[0])

        missing = [lab for lab in self.config.labels if lab not in slot]
        if missing:
            fill = np.full((n,), np.nan, dtype=np.float32)
            for lab in missing:
                slot[lab] = fill

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

        if partial or missing:
            self._packets_flushed_partial += 1
            self._log(
                f"⚠️ [streamer] partial packet flushed index={packet_index} missing={missing}"
            )

        self._packet_buffer.pop(packet_index, None)
        self._packet_first_seen.pop(packet_index, None)
        self._last_packet_index = packet_index

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
                    self._log(
                        f"⚠️ [streamer] timebase discontinuity forward snap={snap:.4f}s err={err:.4f}s"
                    )
                else:
                    snap = max(err, -MAX_BACKWARD_SNAP_S)
                    self._apply_timebase_adjust(snap)
                    self._log(
                        f"⚠️ [streamer] timebase discontinuity backward snap={snap:.4f}s err={err:.4f}s"
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
        if self._last_notify_monotonic is None:
            return
        if (now_mono - self._last_notify_monotonic) < self._notify_stall_s:
            return
        if (now_mono - self._last_reconnect_attempt) < self._reconnect_cooldown_s:
            return
        self._last_reconnect_attempt = now_mono
        self._log("⚠️ [streamer] no BLE notifications; attempting restart")
        if self._client is None:
            return
        try:
            await self._start_streaming()
            await self._subscribe_eeg_channels()
            self._log("✅ [streamer] streaming restart attempted")
            return
        except Exception as exc:
            self._log(f"⚠️ [streamer] restart failed: {exc}")
        try:
            if getattr(self._client, "is_connected", False):
                await self._client.disconnect()
            await self._client.connect()
            await self._start_streaming()
            await self._subscribe_eeg_channels()
            self._log("✅ [streamer] reconnected after notification stall")
        except Exception as exc:
            self._log(f"⚠️ [streamer] reconnect failed: {exc}")

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

        no_push_for = None
        if self._last_push_monotonic is not None:
            no_push_for = float(now_mono - self._last_push_monotonic)

        rssi = self._device_rssi
        last_ts = self._last_push_ts
        time_err_s = self._last_time_err_s
        time_err_fmt = f"{time_err_s:.4f}" if time_err_s is not None else "n/a"
        msg = (
            "[streamer] heartbeat: "
            f"chunks={self._chunks_pushed_total} "
            f"packets={self._packets_seen_total} "
            f"partial={self._packets_flushed_partial} "
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
        info = StreamInfo(
            config.name,
            config.stype,
            len(config.labels),
            config.rate,
            "float32",
            "muse2_internal",
        )
        desc = info.desc()
        channels = desc.append_child("channels")
        for label in config.labels:
            ch = channels.append_child("channel")
            ch.append_child_value("label", label)
            ch.append_child_value("unit", "microvolts")
            ch.append_child_value("type", "EEG")
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
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, streamer.request_stop)
        except NotImplementedError:
            # e.g. on Windows or when loop doesn't support signal handlers
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


if __name__ == "__main__":
    if os.environ.get("MUSE_LSL_SELF_TEST") == "1":
        _self_test_monotonic()
