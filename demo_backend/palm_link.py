from __future__ import annotations

from dataclasses import dataclass
import socket
import struct
from typing import Optional


MAGIC = 0xBC
PKT_VER = 1

FLAG_HOLD = 0x01
FLAG_WATCHDOG = 0x02
FLAG_THERMAL = 0x04
FLAG_LIMIT = 0x08


def encode_packet(
    seq: int,
    timestamp_ms: int,
    finger_id: int,
    action_id: int,
    speed: float,
    flags: int,
) -> bytes:
    speed_u8 = int(max(0, min(255, round(speed * 255.0))))
    return struct.pack(
        "<BBH I BBBB",
        MAGIC,
        PKT_VER,
        seq & 0xFFFF,
        timestamp_ms & 0xFFFFFFFF,
        finger_id & 0xFF,
        action_id & 0xFF,
        speed_u8 & 0xFF,
        flags & 0xFF,
    )


def crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
    crc = init
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ poly) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def append_crc16(packet: bytes) -> bytes:
    crc = crc16_ccitt(packet)
    return packet + struct.pack("<H", crc)


@dataclass
class Telemetry:
    servo_rail_v: Optional[float] = None
    rail_current_a: Optional[float] = None
    thermal_c: Optional[float] = None
    fingertip_contact: Optional[bool] = None


class PalmControllerLink:
    def connect(self) -> None:
        return None

    def close(self) -> None:
        return None

    def send(self, payload: bytes) -> None:
        return None

    def read_telemetry(self) -> Optional[Telemetry]:
        return None


class NullLink(PalmControllerLink):
    def __init__(self) -> None:
        self.connected = True


class SerialUARTLink(PalmControllerLink):
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 0.0) -> None:
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._serial = None

    def connect(self) -> None:
        if self._serial is not None:
            return
        try:
            import serial  # type: ignore
        except Exception as exc:
            raise RuntimeError("pyserial is required for SerialUARTLink") from exc
        self._serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self) -> None:
        if self._serial is not None:
            self._serial.close()
            self._serial = None

    def send(self, payload: bytes) -> None:
        if self._serial is None:
            self.connect()
        if self._serial is not None:
            self._serial.write(payload)


class UDPSimLink(PalmControllerLink):
    def __init__(self, host: str = "127.0.0.1", port: int = 9010) -> None:
        self.host = host
        self.port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass

    def send(self, payload: bytes) -> None:
        self._sock.sendto(payload, (self.host, self.port))
