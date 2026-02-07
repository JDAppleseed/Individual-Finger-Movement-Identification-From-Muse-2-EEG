/*
  uHand UNO Serial Receiver (Drop-in)
  -----------------------------------
  Purpose:
    Make the Hiwonder uHand UNO respond to simple serial commands from a host
    computer (e.g., 7_live_infer_and_actuate.py) for live inference actuation.

  Wiring (matches common uHand UNO wiring in Hiwonder examples):
    Thumb  -> D7
    Index  -> D6
    Middle -> D5
    Ring   -> D4
    Pinky  -> D3
    Wrist  -> D2 (optional; ignored unless commanded)

  Serial Protocol (9600 baud, newline-terminated):
    "<finger_id>,<action_id>\n"

    finger_id:
      0 = none (NO-OP)
      1 = thumb
      2 = index
      3 = middle
      4 = ring
      5 = pinky
      6 = wrist (optional)

    action_id:
      0 = rest (no move; optional "hold" behavior)
      1 = open
      2 = close

  Examples:
    "0,1\n" -> NO-OP (never actuate)
    "3,2\n" -> close middle finger
    "6,1\n" -> wrist open (if used)

  Notes:
    - This is intentionally minimal: no IMU/glove/BLE logic, just serial control.
    - If your servos move the wrong direction, swap OPEN_ANGLE/CLOSE_ANGLE
      for that channel in the arrays below.
    - Invariant: finger_id=0 is NONE and is always a no-op; never actuate hardware.

  Manual test (serial):
    - Send "0,1\n" -> should do nothing (no-op).
    - Send "1,1\n" -> should open thumb.
    - Send "1,2\n" -> should close thumb.

*/

#include <Servo.h>

// ------------------------- Config -------------------------
const long BAUD = 9600;

// Servo channel order: thumb, index, middle, ring, pinky, wrist
static const uint8_t N_SERVOS = 6;

// Common uHand UNO servo pins (adjust if your wiring differs)
static const uint8_t SERVO_PINS[N_SERVOS] = {7, 6, 5, 4, 3, 2};

// Per-servo open/close angles (degrees). Tune as needed.
static const uint8_t OPEN_ANGLE[N_SERVOS]  = { 20,  20,  20,  20,  20,  90};   // wrist at ~90 neutral
static const uint8_t CLOSE_ANGLE[N_SERVOS] = {160, 160, 160, 160, 160,  90};   // keep wrist neutral by default

// How aggressively to move (ms between steps). Larger = slower/smoother.
static const uint16_t STEP_DELAY_MS = 8;

// ------------------------- State -------------------------
Servo servos[N_SERVOS];
uint8_t currentAngle[N_SERVOS];

// ------------------------- Helpers -------------------------
void attachServos() {
  for (uint8_t i = 0; i < N_SERVOS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    currentAngle[i] = OPEN_ANGLE[i];
    servos[i].write(currentAngle[i]);
    delay(50);
  }
}

void moveServoTo(uint8_t idx, uint8_t target) {
  if (idx >= N_SERVOS) return;
  uint8_t cur = currentAngle[idx];
  if (cur == target) return;

  int8_t step = (cur < target) ? 1 : -1;
  while (cur != target) {
    cur = (uint8_t)(cur + step);
    servos[idx].write(cur);
    delay(STEP_DELAY_MS);
  }
  currentAngle[idx] = target;
}

void commandFinger(uint8_t finger_id, uint8_t action_id) {
  // action: 0=rest, 1=open, 2=close
  // Invariant: finger_id=0 is NONE and is always a no-op; never actuate hardware.
  if (finger_id == 0) {
    return;
  }
  if (action_id == 0) {
    // "rest" = do nothing (hold last position)
    return;
  }

  auto do_one = [&](uint8_t idx) {
    if (idx >= N_SERVOS) return;
    uint8_t target = (action_id == 1) ? OPEN_ANGLE[idx] : CLOSE_ANGLE[idx];
    moveServoTo(idx, target);
  };

  // Map ids to indices (1..6 -> 0..5)
  if (finger_id >= 1 && finger_id <= 6) {
    do_one((uint8_t)(finger_id - 1));
  }
}

// Parse line formats like "3,2" or "3 2"
bool parseCommand(String line, uint8_t &finger_id, uint8_t &action_id) {
  line.trim();
  if (line.length() == 0) return false;

  int sep = line.indexOf(',');
  if (sep < 0) sep = line.indexOf(' ');
  if (sep < 0) return false;

  String a = line.substring(0, sep);
  String b = line.substring(sep + 1);
  a.trim(); b.trim();
  if (a.length() == 0 || b.length() == 0) return false;

  int f = a.toInt();
  int act = b.toInt();
  if (f < 0 || f > 6) return false;
  if (act < 0 || act > 2) return false;

  finger_id = (uint8_t)f;
  action_id = (uint8_t)act;
  return true;
}

// ------------------------- Arduino -------------------------
void setup() {
  Serial.begin(BAUD);
  attachServos();
  Serial.println("uHand Serial Receiver ready (protocol: finger,action)");
}

void loop() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  uint8_t finger_id = 0, action_id = 0;
  if (!parseCommand(line, finger_id, action_id)) {
    // ignore malformed lines
    return;
  }
  commandFinger(finger_id, action_id);
}
