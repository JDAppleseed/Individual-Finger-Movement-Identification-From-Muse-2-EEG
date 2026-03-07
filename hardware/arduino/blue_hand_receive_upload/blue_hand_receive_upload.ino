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
    "<finger_id>,<action_id>,<speed_u8>\n"

    finger_id:
      0 = none (NO-OP)
      1 = thumb
      2 = index
      3 = middle
      4 = ring
      5 = pinky
      6 = wrist (optional)

    action_id:
      0 = rest (move to midpoint between open/close; delayed 0.5s if coming from open/close)
      1 = open (no extra delay)
      2 = close (no extra delay)

    speed_u8:
      Optional 0-255 speed scalar. Higher values move faster.

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
// If no valid command arrives within this window, return to midpoint/rest.
const unsigned long IDLE_TIMEOUT_MS = 250;

// Servo channel order: thumb, index, middle, ring, pinky, wrist
static const uint8_t N_SERVOS = 6;

// Common uHand UNO servo pins (adjust if your wiring differs)
static const uint8_t SERVO_PINS[N_SERVOS] = {7, 6, 5, 4, 3, 2};

// Per-servo open/close angles (degrees). Tune as needed.
static const uint8_t OPEN_ANGLE[N_SERVOS]  = { 20,  20,  20,  20,  20,  90};   // wrist at ~90 neutral
static const uint8_t CLOSE_ANGLE[N_SERVOS] = {160, 160, 160, 160, 160,  90};   // keep wrist neutral by default

// Movement behavior
// Speed-modulated ramp motion for open/close commands.

// ------------------------- State -------------------------
Servo servos[N_SERVOS];
uint8_t currentAngle[N_SERVOS];
unsigned long lastCommandMs = 0;
bool atRest = false;

// ------------------------- Helpers -------------------------
void attachServos() {
  for (uint8_t i = 0; i < N_SERVOS; i++) {
    servos[i].attach(SERVO_PINS[i]);
    uint8_t rest = (uint8_t)((uint16_t)OPEN_ANGLE[i] + (uint16_t)CLOSE_ANGLE[i]) / 2;
    currentAngle[i] = rest;
    servos[i].write(currentAngle[i]);
    delay(50);
  }
  atRest = true;
}

void moveServoTo(uint8_t idx, uint8_t target, uint8_t speed_u8 = 255) {
  if (idx >= N_SERVOS) return;
  if (currentAngle[idx] == target) return;

  uint8_t current = currentAngle[idx];
  if (speed_u8 >= 250) {
    servos[idx].write(target);
    currentAngle[idx] = target;
    return;
  }

  unsigned long stepDelayMs = map((long)(255 - speed_u8), 0, 255, 2, 18);
  int direction = (target > current) ? 1 : -1;
  while (current != target) {
    current = (uint8_t)((int)current + direction);
    servos[idx].write(current);
    delay(stepDelayMs);
  }
  currentAngle[idx] = target;
}

void commandFinger(uint8_t finger_id, uint8_t action_id, uint8_t speed_u8 = 255) {
  // action: 0=rest, 1=open, 2=close
  // Invariant: finger_id=0 is NONE and is always a no-op; never actuate hardware.
  if (finger_id == 0) {
    return;
  }

  auto do_one = [&](uint8_t idx) {
    if (idx >= N_SERVOS) return;
    if (action_id == 0) {
      uint8_t rest = (uint8_t)((uint16_t)OPEN_ANGLE[idx] + (uint16_t)CLOSE_ANGLE[idx]) / 2;
      moveServoTo(idx, rest);
      return;
    }

    // action_id 1 or 2: no extra delay, override any pending rest.
    uint8_t target = (action_id == 1) ? OPEN_ANGLE[idx] : CLOSE_ANGLE[idx];
    moveServoTo(idx, target, speed_u8);
  };

  // Map ids to indices (1..6 -> 0..5)
  if (finger_id >= 1 && finger_id <= 6) {
    do_one((uint8_t)(finger_id - 1));
  }

  // Any valid command resets the idle timer and marks the hand as active.
  lastCommandMs = millis();
  atRest = false;
}

// Parse line formats like "3,2", "3 2", or "3,2,180"
bool parseCommand(String line, uint8_t &finger_id, uint8_t &action_id, uint8_t &speed_u8) {
  line.trim();
  if (line.length() == 0) return false;

  int sep = line.indexOf(',');
  if (sep < 0) sep = line.indexOf(' ');
  if (sep < 0) return false;

  int sep2 = line.indexOf(',', sep + 1);
  if (sep2 < 0) sep2 = line.indexOf(' ', sep + 1);

  String a = line.substring(0, sep);
  String b = (sep2 < 0) ? line.substring(sep + 1) : line.substring(sep + 1, sep2);
  String c = (sep2 < 0) ? String("") : line.substring(sep2 + 1);
  a.trim(); b.trim(); c.trim();
  if (a.length() == 0 || b.length() == 0) return false;

  int f = a.toInt();
  int act = b.toInt();
  if (f < 0 || f > 6) return false;
  if (act < 0 || act > 2) return false;

  int speed = 255;
  if (c.length() > 0) {
    speed = c.toInt();
    if (speed < 0 || speed > 255) return false;
  }

  finger_id = (uint8_t)f;
  action_id = (uint8_t)act;
  speed_u8 = (uint8_t)speed;
  return true;
}

// ------------------------- Arduino -------------------------
void setup() {
  Serial.begin(BAUD);
  attachServos();
  lastCommandMs = millis();
  Serial.println("uHand Serial Receiver ready (protocol: finger,action[,speed_u8])");
}

void loop() {
  if (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    uint8_t finger_id = 0, action_id = 0, speed_u8 = 255;
    if (parseCommand(line, finger_id, action_id, speed_u8)) {
      commandFinger(finger_id, action_id, speed_u8);
    }
  }

  // Safety idle timeout: if no commands arrive, return all servos to rest.
  unsigned long now = millis();
  if (!atRest && (now - lastCommandMs) >= IDLE_TIMEOUT_MS) {
    for (uint8_t i = 0; i < N_SERVOS; i++) {
      uint8_t rest = (uint8_t)((uint16_t)OPEN_ANGLE[i] + (uint16_t)CLOSE_ANGLE[i]) / 2;
      moveServoTo(i, rest);
    }
    atRest = true;
  }

}
