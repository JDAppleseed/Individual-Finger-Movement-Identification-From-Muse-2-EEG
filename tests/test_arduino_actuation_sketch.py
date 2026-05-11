from pathlib import Path


def test_arduino_sketch_uses_per_finger_hold_and_nonblocking_updates() -> None:
    sketch = (
        Path(__file__).resolve().parents[1]
        / "hardware"
        / "arduino"
        / "blue_hand_receive_upload"
        / "blue_hand_receive_upload.ino"
    ).read_text()

    assert "MotionPhase motionPhase[N_SERVOS]" in sketch
    assert "PHASE_MOVING_TO_EXTREME" in sketch
    assert "returnToRestQueued[N_SERVOS]" in sketch
    assert "updateMotionState(i, now)" in sketch
    assert "while (Serial.available())" in sketch
    assert "lastCommandMs = millis()" not in sketch
    assert "bool atRest" not in sketch


def test_arduino_sketch_defers_rest_until_endpoint_is_reached() -> None:
    sketch = (
        Path(__file__).resolve().parents[1]
        / "hardware"
        / "arduino"
        / "blue_hand_receive_upload"
        / "blue_hand_receive_upload.ino"
    ).read_text()

    assert "requestRest(idx)" in sketch
    assert "returnToRestQueued[idx] = true;" in sketch
    assert "currentAngle[idx] == extremeAngle[idx]" in sketch
    assert "targetAngle[idx] == extremeAngle[idx]" in sketch
    assert "beginReturnToRest(idx);" in sketch
