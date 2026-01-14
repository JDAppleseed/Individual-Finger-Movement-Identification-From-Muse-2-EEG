import struct

from demo_backend.palm_link import MAGIC, PKT_VER, append_crc16, encode_packet


def test_encode_packet_format():
    payload = encode_packet(
        seq=0x1234,
        timestamp_ms=0x89ABCDEF,
        finger_id=2,
        action_id=1,
        speed=0.5,
        flags=0xA5,
    )
    magic, ver, seq, ts, finger, action, speed_u8, flags = struct.unpack(
        "<BBH I BBBB", payload
    )
    assert magic == MAGIC
    assert ver == PKT_VER
    assert seq == 0x1234
    assert ts == 0x89ABCDEF
    assert finger == 2
    assert action == 1
    assert speed_u8 == int(round(0.5 * 255.0))
    assert flags == 0xA5


def test_append_crc16():
    payload = encode_packet(
        seq=1,
        timestamp_ms=2,
        finger_id=3,
        action_id=4,
        speed=1.0,
        flags=0,
    )
    with_crc = append_crc16(payload)
    assert len(with_crc) == len(payload) + 2
