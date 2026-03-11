# Blue Hand Arduino Firmware

This folder contains the Arduino sketch used for USB/serial actuation.

Notes:
- Open `blue_hand_receive_upload.ino` in the Arduino IDE.
- Select the correct board and serial port for your controller.
- Upload the sketch before enabling `--enable_actuation` in Step 7.
- Step 7 can auto-detect common USB serial Arduino ports when `--serial_port` is omitted.
- If multiple serial devices are connected, pass `--serial_port` explicitly.
