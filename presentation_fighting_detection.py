"""
PRESENTATION MODE: Fighting Detection via USB Camera (Low Latency)
Uses laptop USB camera with 160x120 resolution for fast inference
Includes YOLO pose + LSTM temporal fighting detection
"""

import cv2
from flask import Flask, Response
import threading
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import pickle
import torch
import torch.nn as nn

app = Flask(__name__)
latest_frame = None
frame_lock = threading.Lock()

# ============== LOW LATENCY SETTINGS ==============
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120
CAMERA_FPS = 30
JPEG_QUALITY = 40  # Low quality for speed

# Model paths
POSE_MODEL_PATH = 'runs/pose/human_pose_detector/weights/best.pt'
TEMPORAL_MODEL_PATH = 'fighting_temporal_model_v2/fighting_lstm_v2.pt'

# Temporal model settings
TEMPORAL_SEQUENCE_LENGTH = 24
TEMPORAL_FEATURE_DIM = 68

# Performance settings
FRAME_SKIP = 1  # Process every frame
INFERENCE_SIZE = 256
PERSON_CONF = 0.40
MIN_BOX_SIZE = 30

# Motion detection - DISABLED
MOTION_THRESHOLD = 0
MOTION_SENSITIVITY = 25

# Fighting detection parameters
CLOSE_PROXIMITY_RATIO = 0.12
OVERLAP_THRESHOLD = 0.20
ML_FIGHTING_THRESHOLD = 0.85
LSTM_FIGHTING_THRESHOLD = 0.90

ARM_RAISED_THRESHOLD = 0.3
AGGRESSIVE_POSE_SCORE = 0.6

# Keypoint indices (COCO format)
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

DEBUG_MODE = True
HISTORY_LENGTH = 12
MIN_SUSPICIOUS_FRAMES = 5

# ============== IMPROVED SPATIO-TEMPORAL LSTM MODEL ==============
class FightingLSTM(nn.Module):
    """Improved LSTM model with multi-head attention (95.98% accuracy)"""
    def __init__(self, input_size=TEMPORAL_FEATURE_DIM, hidden_size=256, num_layers=3, dropout=0.4, num_heads=4):
        super(FightingLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        
        # Input projection with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Bidirectional LSTM with more layers
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Layer norm after LSTM
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Second attention layer for refinement
        self.attention2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        
        # Deep classifier with residual
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        
        # Multi-head self-attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Combine with residual
        combined = lstm_out + attn_out
        
        # Second attention for weighting
        attn_weights = nn.functional.softmax(self.attention2(combined), dim=1)
        context = torch.sum(attn_weights * combined, dim=1)
        
        # Classification
        return self.classifier(context).squeeze(-1)


class TemporalFightingDetector:
    """Real-time temporal fighting detector using LSTM"""
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = FightingLSTM().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=TEMPORAL_SEQUENCE_LENGTH)
        self.prob_history = deque(maxlen=5)
    
    def extract_temporal_features(self, keypoints_list, frame_width, frame_height):
        """Extract enhanced spatial features for one frame (68-dim for improved model)"""
        if len(keypoints_list) < 2:
            return None
        
        CONF_THRESH = 0.25
        features = []
        
        def normalize(kpts):
            kpts = kpts.copy().astype(np.float32)
            kpts[:, 0] /= frame_width
            kpts[:, 1] /= frame_height
            return kpts
        
        def get_kpt(kpts, idx):
            if idx >= len(kpts):
                return None
            kpt = kpts[idx]
            if len(kpt) >= 3 and kpt[2] < CONF_THRESH:
                return None
            return np.array(kpt[:2], dtype=np.float32)
        
        def calc_center(kpts, indices):
            valid = [get_kpt(kpts, i) for i in indices if get_kpt(kpts, i) is not None]
            return np.mean(valid, axis=0) if len(valid) >= 2 else None
        
        def calc_angle(p1, p2, p3):
            """Calculate angle at p2"""
            if any(x is None for x in [p1, p2, p3]):
                return 0.5
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            return (np.arccos(np.clip(cos_angle, -1, 1)) / np.pi)
        
        pairs_done = 0
        for i in range(min(len(keypoints_list), 4)):
            for j in range(i + 1, min(len(keypoints_list), 4)):
                if pairs_done >= 4:
                    break
                
                kpts1 = normalize(keypoints_list[i])
                kpts2 = normalize(keypoints_list[j])
                pair_feat = []
                
                # 1. Center distance
                c1 = calc_center(kpts1, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])
                c2 = calc_center(kpts2, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])
                dist = np.linalg.norm(c1 - c2) if c1 is not None and c2 is not None else 1.0
                pair_feat.append(dist)
                
                # 2. Head distance
                head1 = get_kpt(kpts1, NOSE)
                head2 = get_kpt(kpts2, NOSE)
                head_dist = np.linalg.norm(head1 - head2) if head1 is not None and head2 is not None else 1.0
                pair_feat.append(head_dist)
                
                # 3. Arm raised indicators
                for kpts in [kpts1, kpts2]:
                    for side in [(LEFT_SHOULDER, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_WRIST)]:
                        s, w = get_kpt(kpts, side[0]), get_kpt(kpts, side[1])
                        if s is not None and w is not None:
                            pair_feat.append(1.0 if w[1] < s[1] else 0.0)
                            pair_feat.append(s[1] - w[1])
                        else:
                            pair_feat.extend([0.0, 0.0])
                
                # 4. Wrist distances
                w1s = [get_kpt(kpts1, LEFT_WRIST), get_kpt(kpts1, RIGHT_WRIST)]
                w2s = [get_kpt(kpts2, LEFT_WRIST), get_kpt(kpts2, RIGHT_WRIST)]
                wdists = [np.linalg.norm(a - b) for a in w1s for b in w2s if a is not None and b is not None]
                pair_feat.append(min(wdists) if wdists else 1.0)
                pair_feat.append(np.mean(wdists) if wdists else 1.0)
                
                # 5. Elbow angles
                for kpts in [kpts1, kpts2]:
                    for side in [(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)]:
                        s, e, w = [get_kpt(kpts, idx) for idx in side]
                        pair_feat.append(calc_angle(s, e, w))
                
                # 6. Arm extension
                for kpts in [kpts1, kpts2]:
                    for side in [(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)]:
                        s, e, w = [get_kpt(kpts, idx) for idx in side]
                        if all(x is not None for x in [s, e, w]):
                            ext = np.linalg.norm(w - s) / (np.linalg.norm(e - s) + 1e-6)
                            pair_feat.append(min(ext, 3.0) / 3.0)
                        else:
                            pair_feat.append(0.5)
                
                # 7. Body orientation
                for kpts in [kpts1, kpts2]:
                    ls, rs = get_kpt(kpts, LEFT_SHOULDER), get_kpt(kpts, RIGHT_SHOULDER)
                    if ls is not None and rs is not None:
                        pair_feat.append(ls[0] - rs[0])
                    else:
                        pair_feat.append(0.0)
                
                features.extend(pair_feat)
                pairs_done += 1
        
        while len(features) < TEMPORAL_FEATURE_DIM:
            features.append(0.0)
        
        return np.array(features[:TEMPORAL_FEATURE_DIM], dtype=np.float32)
    
    def detect(self, keypoints_list, frame_width, frame_height):
        """Detect fighting from keypoint sequence"""
        features = self.extract_temporal_features(keypoints_list, frame_width, frame_height)
        if features is None:
            return 0.0
        
        self.frame_buffer.append(features)
        
        if len(self.frame_buffer) < TEMPORAL_SEQUENCE_LENGTH:
            return 0.0
        
        try:
            seq = np.array(list(self.frame_buffer), dtype=np.float32)
            seq_tensor = torch.from_numpy(seq).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                prob = self.model(seq_tensor).cpu().item()
            
            self.prob_history.append(prob)
            avg_prob = np.mean(self.prob_history)
            return avg_prob
        except Exception as e:
            print(f"⚠️ LSTM inference error: {e}")
            return 0.0


# Helper functions
def calculate_distance(box1, box2):
    """Calculate distance between two bounding boxes"""
    x1_c = (box1[0] + box1[2]) / 2
    y1_c = (box1[1] + box1[3]) / 2
    x2_c = (box2[0] + box2[2]) / 2
    y2_c = (box2[1] + box2[3]) / 2
    return np.sqrt((x1_c - x2_c)**2 + (y1_c - y2_c)**2)


def get_box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def calculate_motion(prev_frame, curr_frame):
    """Calculate motion between frames"""
    if prev_frame is None:
        return 0
    
    try:
        diff = cv2.absdiff(prev_frame, curr_frame)
        motion = np.mean(diff)
        return motion
    except:
        return 0


def draw_skeleton(frame, keypoints, color, thickness):
    """Draw pose skeleton on frame"""
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16)
    ]
    
    for connection in connections:
        pt1 = tuple(map(int, keypoints[connection[0]][:2]))
        pt2 = tuple(map(int, keypoints[connection[1]][:2]))
        cv2.line(frame, pt1, pt2, color, thickness)
    
    for point in keypoints:
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 2, color, -1)


@app.route('/')
def index():
    return '<h1>Fighting Detection Stream</h1><img src="/video_feed" width="800">'


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with frame_lock:
                if latest_frame is None:
                    time.sleep(0.01)
                    continue
                _, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.03)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


def main():
    """Main detection loop"""
    global latest_frame
    
    # Load models
    print("🔄 Loading YOLO pose model...")
    pose_model = YOLO(POSE_MODEL_PATH)
    
    print("🔄 Loading LSTM temporal model...")
    temporal_detector = TemporalFightingDetector(TEMPORAL_MODEL_PATH)
    
    # Initialize reCamera via RTSP
    RTSP_URL = 'rtsp://192.168.42.1:554/live'
    print(f"📷 Opening reCamera via RTSP: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print("❌ Error: Could not connect to reCamera")
        print(f"Make sure reCamera is at {RTSP_URL}")
        return
    
    print(f"✅ Camera opened! Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}, FPS: {CAMERA_FPS}")
    print(f"🥊 Starting fighting detection...")
    
    frame_count = 0
    prev_frame = None
    fight_history = deque(maxlen=HISTORY_LENGTH)
    global_status = 'Normal'
    global_color = (0, 255, 0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break
        
        frame_count += 1
        display = frame.copy()
        
        motion = calculate_motion(prev_frame, frame)
        motion_detected = motion > MOTION_THRESHOLD
        prev_frame = frame.copy()
        
        persons_data = []
        temporal_prob = 0.0
        fighting_detected = False
        fighting_confidence = 0.0
        
        if frame_count % FRAME_SKIP == 0:
            try:
                pose_results = pose_model.predict(
                    source=frame,
                    conf=PERSON_CONF,
                    imgsz=INFERENCE_SIZE,
                    verbose=False
                )
                
                if len(pose_results[0].boxes) > 0:
                    boxes = pose_results[0].boxes
                    keypoints_list = []
                    
                    for idx, box in enumerate(boxes):
                        coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                        x1, y1, x2, y2 = map(int, coords)
                        
                        if hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None:
                            kpts = pose_results[0].keypoints.data[idx]
                            if hasattr(kpts, 'cpu'):
                                kpts = kpts.cpu().numpy()
                            keypoints_list.append(kpts)
                    
                    # Temporal detection
                    if len(keypoints_list) >= 2:
                        temporal_prob = temporal_detector.detect(keypoints_list, CAMERA_WIDTH, CAMERA_HEIGHT)
                        
                        if temporal_prob >= LSTM_FIGHTING_THRESHOLD:
                            fighting_detected = True
                            fighting_confidence = temporal_prob
                    
                    # Draw persons
                    for idx, box in enumerate(boxes):
                        coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                        x1, y1, x2, y2 = map(int, coords)
                        
                        color = (0, 0, 255) if fighting_detected else (0, 255, 0)
                        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display, f"P{idx+1}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            except Exception as e:
                print(f"⚠️ Error: {e}")
        
        # Update status
        fight_history.append(fighting_detected)
        if fighting_detected:
            global_status = f'🥊 FIGHTING DETECTED ({fighting_confidence:.0%})'
            global_color = (0, 0, 255)
        else:
            global_status = 'Normal'
            global_color = (0, 255, 0)
        
        # Draw status
        cv2.rectangle(display, (5, 5), (300, 50), global_color, -1)
        cv2.putText(display, global_status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, f"LSTM: {temporal_prob:.0%}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        with frame_lock:
            latest_frame = display.copy()
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        t = threading.Thread(target=main, daemon=True)
        t.start()
        print("\n" + "="*70)
        print("[INFO] Fighting Detection Stream RUNNING")
        print("[INFO] Open browser: http://localhost:5000")
        print("="*70 + "\n")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True, use_reloader=False)
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
