#!/usr/bin/env python3
"""
Raspberry Pi Drone - Camera Streamer + Autopilot Interface
Streams camera to PC for AI processing, receives commands back

Setup on Pi:
1. pip3 install picamera2 flask dronekit
2. Enable camera: sudo raspi-config -> Interface -> Camera
3. Run: python3 drone_streamer.py

Your PC connects to:
- RTSP stream: rtsp://<PI_IP>:8554/live
- Or HTTP stream: http://<PI_IP>:5000/video_feed
"""

import cv2
GitHub: Sign in
import socket
import subprocess
import threading
import time
from flask import Flask, Response, jsonify, request
import json
from ultralytics import YOLO
import torch
import torch.nn as nn
import pickle
import numpy as np

# ============== CONFIGURATION ==============
PI_IP = '0.0.0.0'  # Listen on all interfaces
HTTP_PORT = 5000
RTSP_PORT = 8554
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# Camera mode
USE_LOCAL_CAMERA = False  # Set False to use reCamera instead
RECAMERA_IP = '192.168.42.1'  # reCamera IP
RECAMERA_RTSP = f'rtsp://{RECAMERA_IP}:554/live'

# Autopilot settings (ArduPilot/PX4)
AUTOPILOT_ENABLED = False  # Set True if connected to flight controller
AUTOPILOT_CONNECTION = '/dev/ttyAMA0'  # Serial port to flight controller
AUTOPILOT_BAUD = 57600

POSE_MODEL_PATH = 'runs/pose/human_pose_detector/weights/best.pt'
LSTM_MODEL_PATH = 'models/fighting_lstm_v2.pt'
TEMPORAL_SEQUENCE_LENGTH = 16

# Load YOLO pose model
pose_model = YOLO(POSE_MODEL_PATH)
# Load LSTM model
lstm_model = torch.load(LSTM_MODEL_PATH, map_location='cpu')
lstm_model.eval()

# ============== GLOBAL STATE ==============
app = Flask(__name__)
camera = None
latest_frame = None
frame_lock = threading.Lock()
drone = None  # DroneKit vehicle

# Alert state from PC
alert_state = {
    'fighting_detected': False,
    'confidence': 0,
    'last_update': 0,
    'action': 'none'  # none, hover, return_home, alert
}

pose_sequence = []


def get_ip_address():
    """Get Pi's IP address"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "127.0.0.1"


def init_camera():
    """Initialize camera (local or reCamera RTSP)"""
    global camera
    
    # Option 1: Use reCamera via RTSP (no local camera needed)
    if not USE_LOCAL_CAMERA:
        print(f"📹 Connecting to reCamera: {RECAMERA_RTSP}")
        camera = cv2.VideoCapture(RECAMERA_RTSP)
        if camera.isOpened():
            print(f"✅ reCamera connected!")
            return True
        else:
            print(f"⚠️ reCamera not available, trying local camera...")
    
    # Option 2: Try Pi Camera with picamera2
    try:
        from picamera2 import Picamera2
        camera = Picamera2()
        config = camera.create_video_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT), "format": "RGB888"}
        )
        camera.configure(config)
        camera.start()
        print(f"✅ Pi Camera initialized with picamera2 ({CAMERA_WIDTH}x{CAMERA_HEIGHT})")
        return True
    except Exception as e:
        print(f"⚠️ picamera2 not available: {e}")
    
    # Option 3: Try USB camera
    for device_id in [0, 1, 2]:
        try:
            camera = cv2.VideoCapture(device_id)
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 160)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 120)
            camera.set(cv2.CAP_PROP_FPS, 30)
            if camera.isOpened():
                ret, frame = camera.read()
                if ret and frame is not None:
                    print(f"✅ USB Camera on /dev/video{device_id}")
                    return True
                camera.release()
        except:
            pass
    
    print("⚠️ No camera - running in ALERT RECEIVER MODE only")
    return False


def capture_frames():
    """Continuously capture frames"""
    global latest_frame, camera, pose_sequence
    
    if camera is None:
        return  # No camera, just run as alert receiver
    
    while True:
        try:
            if hasattr(camera, 'capture_array'):
                # Pi Camera
                frame = camera.capture_array()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frame = cv2.resize(frame, (160, 120))
            else:
                # USB Camera
                ret, frame = camera.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                frame = cv2.resize(frame, (160, 120))
            
            # --- Fighting Detection ---
            results = pose_model(frame)
            keypoints = results[0].keypoints.xy.cpu().numpy() if results[0].keypoints is not None else None
            if keypoints is not None:
                pose_sequence.append(keypoints)
                if len(pose_sequence) > TEMPORAL_SEQUENCE_LENGTH:
                    pose_sequence = pose_sequence[-TEMPORAL_SEQUENCE_LENGTH:]
                if len(pose_sequence) == TEMPORAL_SEQUENCE_LENGTH:
                    # Flatten and preprocess for LSTM
                    sequence = np.array(pose_sequence)
                    sequence = sequence.reshape(TEMPORAL_SEQUENCE_LENGTH, -1)
                    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0)
                    with torch.no_grad():
                        output = lstm_model(sequence_tensor)
                        fighting_prob = torch.sigmoid(output).item()
                        if fighting_prob > 0.5:
                            print(f"FIGHTING DETECTED! Confidence: {fighting_prob:.2f}")
            # --- End Fighting Detection ---
            with frame_lock:
                latest_frame = frame
            
            time.sleep(0.01)
        except Exception as e:
            print(f"⚠️ Frame capture error: {e}")
            time.sleep(0.5)


def generate_mjpeg():
    """Generate MJPEG stream for HTTP"""
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame = latest_frame.copy()
        
        # Encode as JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 40])
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        
        time.sleep(0.01)


def start_rtsp_server():
    """Start RTSP server using ffmpeg (optional, more compatible)"""
    try:
        # This requires ffmpeg and v4l2rtspserver
        # Install: sudo apt install ffmpeg v4l2rtspserver
        cmd = f"v4l2rtspserver -W {CAMERA_WIDTH} -H {CAMERA_HEIGHT} -F {CAMERA_FPS} -P {RTSP_PORT} /dev/video0"
        subprocess.Popen(cmd.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ RTSP server started on port {RTSP_PORT}")
    except:
        print("⚠️ RTSP server not available (install v4l2rtspserver)")


# ============== AUTOPILOT FUNCTIONS ==============
def init_autopilot():
    """Connect to flight controller via DroneKit"""
    global drone
    
    if not AUTOPILOT_ENABLED:
        print("ℹ️ Autopilot disabled")
        return False
    
    try:
        from dronekit import connect
        print(f"🔗 Connecting to autopilot on {AUTOPILOT_CONNECTION}...")
        drone = connect(AUTOPILOT_CONNECTION, baud=AUTOPILOT_BAUD, wait_ready=True)
        print(f"✅ Connected! Mode: {drone.mode.name}")
        return True
    except Exception as e:
        print(f"❌ Autopilot connection failed: {e}")
        return False


def hover_in_place():
    """Command drone to hold position"""
    if drone is None:
        return False
    try:
        from dronekit import VehicleMode
        drone.mode = VehicleMode("LOITER")
        print("🚁 Drone hovering in place")
        return True
    except:
        return False


def return_to_home():
    """Command drone to return home"""
    if drone is None:
        return False
    try:
        from dronekit import VehicleMode
        drone.mode = VehicleMode("RTL")
        print("🏠 Drone returning to home")
        return True
    except:
        return False


# ============== HTTP ENDPOINTS ==============
@app.route('/')
def index():
    """Status page"""
    ip = get_ip_address()
    return f"""
    <h1>🚁 Drone Camera Streamer</h1>
    <h2>Streams:</h2>
    <ul>
        <li>HTTP MJPEG: <a href="http://{ip}:{HTTP_PORT}/video_feed">http://{ip}:{HTTP_PORT}/video_feed</a></li>
        <li>RTSP: rtsp://{ip}:{RTSP_PORT}/live</li>
    </ul>
    <h2>API Endpoints:</h2>
    <ul>
        <li>GET /status - Drone status</li>
        <li>POST /alert - Receive fighting alert from PC</li>
        <li>POST /command - Send drone command</li>
    </ul>
    <h2>Live Preview:</h2>
    <img src="/video_feed" width="640" height="480">
    """


@app.route('/video_feed')
def video_feed():
    """MJPEG video stream"""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    """Get drone/camera status"""
    status_data = {
        'camera': latest_frame is not None,
        'ip': get_ip_address(),
        'http_stream': f"http://{get_ip_address()}:{HTTP_PORT}/video_feed",
        'rtsp_stream': f"rtsp://{get_ip_address()}:{RTSP_PORT}/live",
        'alert_state': alert_state,
        'autopilot_connected': drone is not None
    }
    
    if drone:
        status_data['drone'] = {
            'mode': drone.mode.name,
            'armed': drone.armed,
            'battery': drone.battery.voltage if drone.battery else None,
            'gps': {
                'lat': drone.location.global_frame.lat if drone.location.global_frame else None,
                'lon': drone.location.global_frame.lon if drone.location.global_frame else None,
                'alt': drone.location.global_frame.alt if drone.location.global_frame else None
            }
        }
    
    return jsonify(status_data)


@app.route('/alert', methods=['POST'])
def receive_alert():
    """Receive fighting alert from PC detection system"""
    global alert_state
    
    data = request.json
    alert_state = {
        'fighting_detected': data.get('event') == 'fighting_detected',
        'confidence': data.get('confidence', 0),
        'last_update': time.time(),
        'action': data.get('action', 'alert')
    }
    
    print(f"🚨 ALERT RECEIVED: Fighting {alert_state['confidence']}% confidence")
    
    # Auto-respond based on action
    action = data.get('action', 'alert')
    if action == 'hover' and drone:
        hover_in_place()
    elif action == 'return_home' and drone:
        return_to_home()
    
    # Trigger local alarm (buzzer, LED, etc.)
    trigger_local_alarm(alert_state['fighting_detected'])
    
    return jsonify({'status': 'received', 'action_taken': action})


@app.route('/command', methods=['POST'])
def drone_command():
    """Receive drone command from PC"""
    data = request.json
    cmd = data.get('command')
    
    if cmd == 'hover':
        success = hover_in_place()
    elif cmd == 'return_home':
        success = return_to_home()
    else:
        return jsonify({'error': 'Unknown command'}), 400
    
    return jsonify({'command': cmd, 'success': success})


def trigger_local_alarm(active):
    """Trigger local alarm (buzzer/LED on GPIO)"""
    try:
        import RPi.GPIO as GPIO
        BUZZER_PIN = 18
        LED_PIN = 17
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUZZER_PIN, GPIO.OUT)
        GPIO.setup(LED_PIN, GPIO.OUT)
        
        GPIO.output(BUZZER_PIN, active)
        GPIO.output(LED_PIN, active)
        
        if active:
            print("🔔 LOCAL ALARM TRIGGERED")
    except:
        pass  # GPIO not available


# ============== MAIN ==============
def main():
    print("=" * 50)
    print("🚁 DRONE ALERT RECEIVER & CONTROLLER")
    print("=" * 50)
    
    # Get IP
    ip = get_ip_address()
    print(f"📡 Pi IP Address: {ip}")
    
    # Initialize camera (optional - can run without)
    has_camera = init_camera()
    
    if has_camera:
        # Start frame capture thread
        capture_thread = threading.Thread(target=capture_frames, daemon=True)
        capture_thread.start()
        
        # Start RTSP server
        start_rtsp_server()
        print("📺 Camera streaming enabled")
    else:
        print("⚠️ Running in ALERT RECEIVER MODE (no camera)")
        print("   Video comes from reCamera directly to PC")
    
    # Initialize autopilot (optional)
    init_autopilot()
    
    # Print connection info
    print("=" * 50)
    print("🔧 AVAILABLE ENDPOINTS:")
    print(f"   POST http://{ip}:{HTTP_PORT}/alert    - Receive fighting alerts")
    print(f"   POST http://{ip}:{HTTP_PORT}/command  - Send drone commands")
    print(f"   GET  http://{ip}:{HTTP_PORT}/status   - Get system status")
    if has_camera:
        print(f"   GET  http://{ip}:{HTTP_PORT}/video_feed - Video stream")
        print(f"   RTSP: rtsp://{ip}:{RTSP_PORT}/live")
    print("=" * 50)
    print("🔧 For your PC script, set:")
    print(f"   PI_DRONE_IP = '{ip}'")
    print("=" * 50)
    
    # Start Flask server
    print("🚀 Starting Flask server...")
    app.run(host=PI_IP, port=HTTP_PORT, threaded=True)


if __name__ == '__main__':
    main()
