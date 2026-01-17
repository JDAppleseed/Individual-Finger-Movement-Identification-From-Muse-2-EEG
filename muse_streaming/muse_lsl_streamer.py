from __future__ import annotations

import asyncio
import signal
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

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

DEFAULT_LABELS = ["TP9", "AF7", "AF8", "TP10"]


@dataclass
class StreamConfig:
    name: str
    stype: str
    rate: float
    labels: List[str]


class MuseLslStreamer:
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
        self._client: Optional[BleakClient] = None
        self._outlet: Optional[StreamOutlet] = None
        self._packet_buffer: Dict[int, Dict[str, np.ndarray]] = {}
        self._stop_requested = asyncio.Event()
        self._last_packet_index: Optional[int] = None

    def request_stop(self) -> None:
        self._stop_requested.set()

    def _log(self, message: str) -> None:
        if self.log_fn:
            self.log_fn(message)

    async def run(self) -> None:
        if not BLEAK_AVAILABLE:
            raise RuntimeError("Bleak is required for Muse 2 BLE streaming.")
        if not LSL_AVAILABLE:
            raise RuntimeError("pylsl is required for Muse 2 LSL streaming.")

        self._outlet = self._build_outlet(self.config)
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

        await self._wait_until_stop()
        await self._shutdown()

    async def _wait_until_stop(self) -> None:
        while not self._stop_requested.is_set():
            await asyncio.sleep(0.1)

    async def _resolve_device(self):
        if self.mac_address:
            return self.mac_address
        if BleakScanner is None:
            raise RuntimeError("BleakScanner unavailable; cannot scan for Muse 2.")
        self._log("🔎 Scanning for Muse 2 over BLE...")
        devices = await BleakScanner.discover(timeout=6.0)
        candidates = []
        for dev in devices:
            name = (dev.name or "").lower()
            if self.device_name and self.device_name.lower() in name:
                candidates.append(dev)
            elif not self.device_name and "muse" in name:
                candidates.append(dev)
        if not candidates:
            raise RuntimeError("No Muse device found over BLE.")
        candidates.sort(key=lambda d: d.rssi or -999, reverse=True)
        return candidates[0]

    async def _start_streaming(self) -> None:
        if not self._client:
            raise RuntimeError("Muse client not connected.")
        start_cmd = bytearray([0x02, 0x64, 0x0A])
        await self._client.write_gatt_char(MUSE_GATT_ATTR_STREAM_TOGGLE, start_cmd)

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
            uuid = channels.get(label.upper())
            if uuid is None:
                raise RuntimeError(f"Unsupported EEG label: {label}")
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
        slot = self._packet_buffer.setdefault(packet_index, {})
        slot[label.upper()] = samples
        if len(slot) < len(self.config.labels):
            return
        ordered = []
        for idx in range(samples.shape[0]):
            ordered.append([slot[label.upper()][idx] for label in self.config.labels])
        now = float(local_clock()) if local_clock is not None else time.time()
        ts = [now - (len(ordered) - 1 - i) / self.config.rate for i in range(len(ordered))]
        self._outlet.push_chunk(ordered, ts)
        self._packet_buffer.pop(packet_index, None)
        self._last_packet_index = packet_index

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

    async def _shutdown(self) -> None:
        if self._client and self._client.is_connected:
            await self._client.disconnect()
        self._log("ℹ️ Muse 2 streamer stopped")


def _decode_eeg_packet(packet: bytearray) -> tuple[int, np.ndarray]:
    """
    Decode a Muse EEG packet.

    Adapted from uvicMUSE/Musey packet decoding logic.
    """
    bits = Bits(bytes=packet)
    pattern = (
        "uint:16,uint:12,uint:12,uint:12,uint:12,uint:12,uint:12,"
        "uint:12,uint:12,uint:12,uint:12,uint:12,uint:12"
    )
    values = bits.unpack(pattern)
    packet_index = int(values[0])
    data = np.asarray(values[1:], dtype=np.float32)
    data = 0.48828125 * (data - 2048)
    return packet_index, data


def install_signal_handlers(streamer: MuseLslStreamer) -> None:
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, streamer.request_stop)
        except NotImplementedError:
            pass
