# 🚁 Raspberry Pi Drone Setup Guide

## Architecture
```
[Pi + Camera] --WiFi/RTSP--> [Your PC - AI Processing] --WiFi--> [Pi - Drone Control]
     |                              |                                    |
   Stream video              YOLO + LSTM detection              Hover/RTL/Alert
```

## 📁 Files Created
- `raspberry_pi/drone_streamer.py` - Run this ON the Raspberry Pi
- Updated `realtime_pose_fight_detection.py` - Run this on your PC

---

## 🍓 Raspberry Pi Setup

### 1. Install Dependencies (on Pi)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python packages
pip3 install flask opencv-python-headless dronekit

# For Pi Camera
sudo apt install -y python3-picamera2

# For RTSP streaming (optional)
sudo apt install -y v4l2rtspserver ffmpeg
```

### 2. Enable Camera
```bash
sudo raspi-config
# -> Interface Options -> Camera -> Enable
# -> Reboot
```

### 3. Copy the Script to Pi
```bash
# From your PC, copy via SCP:
scp raspberry_pi/drone_streamer.py pi@<PI_IP>:~/drone_streamer.py
```

### 4. Run on Pi
```bash
python3 drone_streamer.py
```

You'll see:
```
📡 Pi IP Address: 192.168.x.x
📺 STREAM URLS:
   HTTP:  http://192.168.x.x:5000/video_feed
   RTSP:  rtsp://192.168.x.x:8554/live
```

---

## 💻 PC Configuration

### Update your PC script with Pi's IP:
In `realtime_pose_fight_detection.py`:
```python
# Camera source
RECAMERA_IP = '192.168.x.x'  # Your Pi's IP
RTSP_URL = f'rtsp://{RECAMERA_IP}:8554/live'  # or use 5000 for HTTP

# Drone alerts
PI_DRONE_IP = '192.168.x.x'  # Same as above
PI_DRONE_PORT = 5000
PI_FIGHTING_ACTION = 'hover'  # or 'return_home'
```

---

## 🎮 Autopilot Connection (Optional)

If connecting Pi to a flight controller (Pixhawk, ArduPilot):

### Wiring
```
Pi GPIO 14 (TX) --> FC RX (TELEM2)
Pi GPIO 15 (RX) --> FC TX (TELEM2)
Pi GND         --> FC GND
```

### Enable Serial on Pi
```bash
sudo raspi-config
# -> Interface Options -> Serial -> No console, Yes serial port
```

### Update drone_streamer.py
```python
AUTOPILOT_ENABLED = True
AUTOPILOT_CONNECTION = '/dev/ttyAMA0'  # or /dev/serial0
AUTOPILOT_BAUD = 57600
```

---

## 🚨 How It Works

1. **Pi streams video** → Your PC receives via RTSP/HTTP
2. **PC runs YOLO + LSTM** → Detects fighting (95.98% accuracy)
3. **PC sends alert to Pi** → `POST /alert` with action
4. **Pi executes action**:
   - `'alert'` - Just trigger buzzer/LED
   - `'hover'` - Switch to LOITER mode (hold position)
   - `'return_home'` - Switch to RTL mode (return to launch)

---

## 📡 API Endpoints (on Pi)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status page with live preview |
| `/video_feed` | GET | MJPEG video stream |
| `/status` | GET | JSON status (camera, drone, GPS) |
| `/alert` | POST | Receive fighting alert from PC |
| `/command` | POST | Send drone command (hover, return_home) |

---

## 🔧 Testing

### 1. Test camera stream
Open browser: `http://<PI_IP>:5000/`

### 2. Test from PC
```bash
# Check status
curl http://<PI_IP>:5000/status

# Send test alert
curl -X POST http://<PI_IP>:5000/alert \
  -H "Content-Type: application/json" \
  -d '{"event":"fighting_detected","confidence":95,"action":"hover"}'
```

### 3. Run full system
```bash
# On Pi:
python3 drone_streamer.py

# On PC:
python realtime_pose_fight_detection.py
```

---

## ⚠️ Notes

- Pi streams video, PC does AI (Pi is too slow for YOLO)
- WiFi range limits drone distance (~100m typical)
- For longer range, use 4G/LTE module or dedicated video transmitter
- Battery: Pi 4 draws ~3W, add this to flight time calculations
