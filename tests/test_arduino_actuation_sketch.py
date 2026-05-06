from pathlib import Path


def test_arduino_sketch_uses_per_finger_hold_and_nonblocking_updates() -> None:
    sketch = (
        Path(__file__).resolve().parents[1]
        / "hardware"
        / "arduino"
        / "blue_hand_receive_upload"
        / "blue_hand_receive_upload.ino"
    ).read_text()

    assert "lastCommandMs[N_SERVOS]" in sketch
    assert "holdingCommand[N_SERVOS]" in sketch
    assert "updateServo(i, now)" in sketch
    assert "while (Serial.available())" in sketch
    assert "lastCommandMs = millis()" not in sketch
    assert "bool atRest" not in sketch
