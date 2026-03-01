import threading
# ============== THREADED FRAME GRABBER ==============
class ThreadedCamera:
    def __init__(self, cap):
        self.cap = cap
        self.lock = threading.Lock()
        self.frame = None
        self.stopped = False
        self.last_read_time = time.time()
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                    self.last_read_time = time.time()
            else:
                # If failed, sleep briefly to avoid busy loop
                time.sleep(0.05)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1)
        self.cap.release()

    def is_opened(self):
        return self.cap.isOpened()

    def time_since_last_frame(self):
        return time.time() - self.last_read_time
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import pickle
import os
import torch
import torch.nn as nn

# ============== SOURCE SELECTION ==============
USE_RECAMERA = False  # Using reCamera RTSP stream
USE_VIDEO_FILE = True  # Use video file instead of camera

# Configuration
RECAMERA_IP = '192.168.42.1'
# Use TCP transport for more reliable RTSP streaming (prevents packet loss disconnects)
RTSP_URL = f'rtsp://{RECAMERA_IP}:554/live?tcp'
RTSP_URL_FFMPEG = f'rtsp://{RECAMERA_IP}:554/live'  # For FFmpeg with explicit options
WEBCAM_ID = 0
VIDEO_FILE_PATH = 'surveillanceVideos/1657Pri_OutFG_C1.mp4'  # Fighting video - camera 1 angle
# Model paths
POSE_MODEL_PATH = 'runs/pose/human_pose_detector/weights/best.pt'  # Trained pose model
ACTIVITY_MODEL_PATH = 'runs/train/activity_cls_improved/weights/best.pt'  # Activity classifier
FIGHTING_CLASSIFIER_PATH = 'fighting_pose_dataset/fighting_classifier.pkl'  # ML fighting classifier
# Use the 91% accuracy LSTM model for better performance
TEMPORAL_MODEL_PATH = 'fighting_temporal_model/fighting_lstm_91.pt'  # LSTM temporal model (91% accuracy)

TEMPORAL_SEQUENCE_LENGTH = 8  # Lowered for faster detection
TEMPORAL_FEATURE_DIM = 68  # Updated for improved model

# Performance settings - OPTIMIZED FOR SPEED
FRAME_SKIP = 8          # Process every 8th frame (max speed)
INFERENCE_SIZE = 256    # Even smaller for max speed
PERSON_CONF = 0.30      # Lower threshold = more detections (was 0.40)
MIN_BOX_SIZE = 35       # Skip small detections

# Motion detection - DISABLED for testing
MOTION_THRESHOLD = 0    # 0 = always detect (no motion filter)
MOTION_SENSITIVITY = 25

# ============== FIGHTING DETECTION PARAMETERS ==============
# Distance thresholds (in pixels, relative to image size)
CLOSE_PROXIMITY_RATIO = 0.12  # People closer than 12% of frame width = close
OVERLAP_THRESHOLD = 0.20  # 20% box overlap = physical contact
ML_FIGHTING_THRESHOLD = 0.70  # ML threshold (backup, was 0.85)
# Ultra-sensitive for 91% model
LSTM_FIGHTING_THRESHOLD = 0.20  # LSTM threshold (primary, ultra-high sensitivity)

# Pose-based fighting indicators
ARM_RAISED_THRESHOLD = 0.3  # Wrist above shoulder level = raised arm
AGGRESSIVE_POSE_SCORE = 0.6  # Threshold for aggressive pose detection

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

# Activity classes
SUSPICIOUS_ACTIVITIES = ['fighting_group', 'robbery_knife', 'aggressive_activity']
MILD_SUSPICIOUS = ['leaving_package', 'passing_out', 'person_pushing']
NORMAL_ACTIVITIES = ['walking', 'person_running']

# Thresholds
ALERT_THRESHOLD = 0.40
MILD_THRESHOLD = 0.50
DEBUG_MODE = True

# Temporal (add short persistence to reduce flicker)
HISTORY_LENGTH = 6  # Smoothing window (6 frames)
MIN_SUSPICIOUS_FRAMES = 1  # Only need 1 positive in window
FIGHTING_COOLDOWN_FRAMES = 6  # Hold 'FIGHTING DETECTED' for 6 frames after last detection


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
    

    def update(self, keypoints_list, frame_width, frame_height):
        """Add frame and return (fighting_prob, buffer_fill_ratio)"""
        if len(keypoints_list) >= 2:
            features = self.extract_temporal_features(keypoints_list, frame_width, frame_height)
            if features is None:
                features = np.zeros(TEMPORAL_FEATURE_DIM, dtype=np.float32)
        else:
            features = np.zeros(TEMPORAL_FEATURE_DIM, dtype=np.float32)

        # If buffer is empty, pad with the first frame's features
        if len(self.frame_buffer) == 0:
            for _ in range(TEMPORAL_SEQUENCE_LENGTH - 1):
                self.frame_buffer.append(features)

        self.frame_buffer.append(features)

        if len(self.frame_buffer) < TEMPORAL_SEQUENCE_LENGTH:
            return 0.0, len(self.frame_buffer) / TEMPORAL_SEQUENCE_LENGTH

        sequence = np.array(list(self.frame_buffer))
        seq_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.model(seq_tensor).item()

        self.prob_history.append(prob)
        smoothed = np.mean(self.prob_history)

        # Debug: print LSTM input and output
        if DEBUG_MODE:
            print(f"[LSTM] Features: {features[:8]}... Prob: {prob:.2f} Smoothed: {smoothed:.2f}")

        return smoothed, 1.0
    
    def reset(self):
        self.frame_buffer.clear()
        self.prob_history.clear()


def calculate_motion(prev_frame, curr_frame):
    """Calculate motion between frames"""
    if prev_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(cv2.GaussianBlur(prev_gray, (21, 21), 0),
                       cv2.GaussianBlur(curr_gray, (21, 21), 0))
    _, thresh = cv2.threshold(diff, MOTION_SENSITIVITY, 255, cv2.THRESH_BINARY)
    return np.sum(thresh > 0)


def get_box_center(box):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def calculate_distance(box1, box2):
    """Calculate distance between centers of two boxes"""
    c1 = get_box_center(box1)
    c2 = get_box_center(box2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)


def boxes_overlap(box1, box2, threshold=0.2):
    """Check if two boxes overlap significantly"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return False, 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    iou = intersection / min(area1, area2) if min(area1, area2) > 0 else 0
    return iou > threshold, iou


def analyze_pose_aggression(keypoints):
    """
    Analyze pose keypoints for aggressive postures.
    Returns aggression score (0-1) and indicators.
    """
    if keypoints is None or len(keypoints) < 17:
        return 0.0, []
    
    indicators = []
    aggression_score = 0.0
    
    # Get keypoint coordinates (x, y, confidence)
    kpts = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
    
    # Check if we have valid keypoints
    if len(kpts.shape) == 1:
        kpts = kpts.reshape(-1, 3) if len(kpts) == 51 else kpts.reshape(-1, 2)
    
    # Helper to get keypoint if confident enough
    def get_kpt(idx, min_conf=0.3):
        if idx >= len(kpts):
            return None
        kpt = kpts[idx]
        if len(kpt) >= 3 and kpt[2] < min_conf:
            return None
        return kpt[:2]
    
    # Check for raised arms (wrists above shoulders)
    left_shoulder = get_kpt(LEFT_SHOULDER)
    right_shoulder = get_kpt(RIGHT_SHOULDER)
    left_wrist = get_kpt(LEFT_WRIST)
    right_wrist = get_kpt(RIGHT_WRIST)
    left_elbow = get_kpt(LEFT_ELBOW)
    right_elbow = get_kpt(RIGHT_ELBOW)
    
    # Raised left arm
    if left_shoulder is not None and left_wrist is not None:
        if left_wrist[1] < left_shoulder[1]:  # y decreases upward
            aggression_score += 0.25
            indicators.append("LEFT_ARM_RAISED")
    
    # Raised right arm
    if right_shoulder is not None and right_wrist is not None:
        if right_wrist[1] < right_shoulder[1]:
            aggression_score += 0.25
            indicators.append("RIGHT_ARM_RAISED")
    
    # Both arms raised = very aggressive
    if "LEFT_ARM_RAISED" in indicators and "RIGHT_ARM_RAISED" in indicators:
        aggression_score += 0.2
        indicators.append("BOTH_ARMS_UP")
    
    # Check for extended arms (punching motion)
    if left_shoulder is not None and left_wrist is not None and left_elbow is not None:
        shoulder_to_wrist = np.sqrt((left_wrist[0] - left_shoulder[0])**2 + 
                                     (left_wrist[1] - left_shoulder[1])**2)
        shoulder_to_elbow = np.sqrt((left_elbow[0] - left_shoulder[0])**2 + 
                                     (left_elbow[1] - left_shoulder[1])**2)
        if shoulder_to_elbow > 0 and shoulder_to_wrist / shoulder_to_elbow > 1.5:
            aggression_score += 0.15
            indicators.append("LEFT_ARM_EXTENDED")
    
    if right_shoulder is not None and right_wrist is not None and right_elbow is not None:
        shoulder_to_wrist = np.sqrt((right_wrist[0] - right_shoulder[0])**2 + 
                                     (right_wrist[1] - right_shoulder[1])**2)
        shoulder_to_elbow = np.sqrt((right_elbow[0] - right_shoulder[0])**2 + 
                                     (right_elbow[1] - right_shoulder[1])**2)
        if shoulder_to_elbow > 0 and shoulder_to_wrist / shoulder_to_elbow > 1.5:
            aggression_score += 0.15
            indicators.append("RIGHT_ARM_EXTENDED")
    
    return min(1.0, aggression_score), indicators


def detect_fighting_from_poses(persons, frame_width, frame_height):
    """
    Detect fighting between 2+ people using pose analysis.
    Returns: (is_fighting, confidence, reason)
    """
    if len(persons) < 2:
        return False, 0.0, "Need 2+ people"
    
    fighting_score = 0.0
    reasons = []
    
    # Check all pairs of people
    for i in range(len(persons)):
        for j in range(i + 1, len(persons)):
            p1 = persons[i]
            p2 = persons[j]
            
            # 1. Check proximity
            distance = calculate_distance(p1['box'], p2['box'])
            proximity_threshold = frame_width * CLOSE_PROXIMITY_RATIO
            
            if distance < proximity_threshold:
                fighting_score += 0.3
                reasons.append(f"CLOSE_PROXIMITY({distance:.0f}px)")
            
            # 2. Check overlap (physical contact)
            overlapping, iou = boxes_overlap(p1['box'], p2['box'], OVERLAP_THRESHOLD)
            if overlapping:
                fighting_score += 0.4
                reasons.append(f"PHYSICAL_CONTACT(IoU:{iou:.0%})")
            
            # 3. Check aggressive poses
            if p1.get('aggression_score', 0) > 0.4:
                fighting_score += 0.2
                reasons.append(f"P1_AGGRESSIVE({p1['aggression_score']:.0%})")
            
            if p2.get('aggression_score', 0) > 0.4:
                fighting_score += 0.2
                reasons.append(f"P2_AGGRESSIVE({p2['aggression_score']:.0%})")
            
            # 4. Both have raised arms while close = likely fighting
            if (distance < proximity_threshold and 
                p1.get('aggression_score', 0) > 0.3 and 
                p2.get('aggression_score', 0) > 0.3):
                fighting_score += 0.3
                reasons.append("MUTUAL_AGGRESSION")
    
    is_fighting = fighting_score >= 0.5
    return is_fighting, min(1.0, fighting_score), "; ".join(reasons[:3])


def extract_ml_features(keypoints_list, frame_width, frame_height):
    """
    Extract features for ML fighting classifier.
    Same format as training script.
    """
    if len(keypoints_list) < 2:
        return None
    
    KEYPOINT_CONF_THRESHOLD = 0.3
    features = []
    
    def normalize(kpts):
        if kpts is None:
            return None
        kpts = kpts.copy()
        kpts[:, 0] /= frame_width
        kpts[:, 1] /= frame_height
        return kpts
    
    def get_kpt(kpts, idx, min_conf=KEYPOINT_CONF_THRESHOLD):
        if idx >= len(kpts):
            return None
        kpt = kpts[idx]
        if len(kpt) >= 3 and kpt[2] < min_conf:
            return None
        return np.array(kpt[:2])
    
    def calc_center(kpts, indices):
        valid_pts = []
        for idx in indices:
            pt = get_kpt(kpts, idx)
            if pt is not None:
                valid_pts.append(pt)
        if len(valid_pts) >= 2:
            return np.mean(valid_pts, axis=0)
        return None
    
    # Process each pair of people (max 3 pairs)
    pairs_processed = 0
    for i in range(len(keypoints_list)):
        for j in range(i + 1, len(keypoints_list)):
            if pairs_processed >= 3:
                break
            
            kpts1 = normalize(keypoints_list[i])
            kpts2 = normalize(keypoints_list[j])
            
            if kpts1 is None or kpts2 is None:
                continue
            
            pair_features = []
            
            # 1. Center distance
            center1 = calc_center(kpts1, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])
            center2 = calc_center(kpts2, [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP])
            
            if center1 is not None and center2 is not None:
                pair_features.append(np.linalg.norm(center1 - center2))
            else:
                pair_features.append(1.0)
            
            # 2. Arm raised for each person
            for kpts in [kpts1, kpts2]:
                left_shoulder = get_kpt(kpts, LEFT_SHOULDER)
                right_shoulder = get_kpt(kpts, RIGHT_SHOULDER)
                left_wrist = get_kpt(kpts, LEFT_WRIST)
                right_wrist = get_kpt(kpts, RIGHT_WRIST)
                
                if left_shoulder is not None and left_wrist is not None:
                    pair_features.append(1.0 if left_wrist[1] < left_shoulder[1] else 0.0)
                else:
                    pair_features.append(0.0)
                
                if right_shoulder is not None and right_wrist is not None:
                    pair_features.append(1.0 if right_wrist[1] < right_shoulder[1] else 0.0)
                else:
                    pair_features.append(0.0)
            
            # 3. Wrist distances
            lw1, rw1 = get_kpt(kpts1, LEFT_WRIST), get_kpt(kpts1, RIGHT_WRIST)
            lw2, rw2 = get_kpt(kpts2, LEFT_WRIST), get_kpt(kpts2, RIGHT_WRIST)
            
            wrist_distances = []
            for w1 in [lw1, rw1]:
                for w2 in [lw2, rw2]:
                    if w1 is not None and w2 is not None:
                        wrist_distances.append(np.linalg.norm(w1 - w2))
            
            if wrist_distances:
                pair_features.append(min(wrist_distances))
                pair_features.append(np.mean(wrist_distances))
            else:
                pair_features.extend([1.0, 1.0])
            
            # 4. Arm extension
            for kpts in [kpts1, kpts2]:
                for side in [(LEFT_SHOULDER, LEFT_ELBOW, LEFT_WRIST), 
                            (RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_WRIST)]:
                    shoulder = get_kpt(kpts, side[0])
                    elbow = get_kpt(kpts, side[1])
                    wrist = get_kpt(kpts, side[2])
                    
                    if all(x is not None for x in [shoulder, elbow, wrist]):
                        s_to_w = np.linalg.norm(wrist - shoulder)
                        s_to_e = np.linalg.norm(elbow - shoulder)
                        extension = s_to_w / (s_to_e + 1e-6)
                        pair_features.append(min(extension, 3.0) / 3.0)
                    else:
                        pair_features.append(0.5)
            
            features.extend(pair_features)
            pairs_processed += 1
    
    # Pad to fixed size (42 features)
    FEATURE_SIZE = 42
    if len(features) < FEATURE_SIZE:
        features.extend([0.0] * (FEATURE_SIZE - len(features)))
    else:
        features = features[:FEATURE_SIZE]
    
    return np.array(features, dtype=np.float32).reshape(1, -1)


def draw_skeleton(frame, keypoints, color=(0, 255, 255), thickness=2):
    """Draw pose skeleton on frame"""
    if keypoints is None:
        return
    
    kpts = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
    
    if len(kpts.shape) == 1:
        kpts = kpts.reshape(-1, 3) if len(kpts) == 51 else kpts.reshape(-1, 2)
    
    # Skeleton connections
    skeleton = [
        (NOSE, LEFT_EYE), (NOSE, RIGHT_EYE),
        (LEFT_EYE, LEFT_EAR), (RIGHT_EYE, RIGHT_EAR),
        (NOSE, LEFT_SHOULDER), (NOSE, RIGHT_SHOULDER),
        (LEFT_SHOULDER, RIGHT_SHOULDER),
        (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
        (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
        (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP),
        (LEFT_HIP, RIGHT_HIP),
        (LEFT_HIP, LEFT_KNEE), (LEFT_KNEE, LEFT_ANKLE),
        (RIGHT_HIP, RIGHT_KNEE), (RIGHT_KNEE, RIGHT_ANKLE),
    ]
    
    # Draw skeleton lines
    for start_idx, end_idx in skeleton:
        if start_idx >= len(kpts) or end_idx >= len(kpts):
            continue
        
        start = kpts[start_idx]
        end = kpts[end_idx]
        
        # Check confidence if available
        start_conf = start[2] if len(start) >= 3 else 1.0
        end_conf = end[2] if len(end) >= 3 else 1.0
        
        if start_conf > 0.3 and end_conf > 0.3:
            pt1 = (int(start[0]), int(start[1]))
            pt2 = (int(end[0]), int(end[1]))
            if pt1[0] > 0 and pt1[1] > 0 and pt2[0] > 0 and pt2[1] > 0:
                cv2.line(frame, pt1, pt2, color, thickness)
    
    # Draw keypoints
    for idx, kpt in enumerate(kpts):
        conf = kpt[2] if len(kpt) >= 3 else 1.0
        if conf > 0.3:
            pt = (int(kpt[0]), int(kpt[1]))
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(frame, pt, 4, color, -1)


def main():
    # Load models
    print("Loading models...")
    pose_model = YOLO(POSE_MODEL_PATH)  # Pose estimation
    activity_model = YOLO(ACTIVITY_MODEL_PATH)  # Activity classification (backup)
    
    # Load TEMPORAL LSTM model (95.98% accuracy - improved)
    temporal_detector = None
    if os.path.exists(TEMPORAL_MODEL_PATH):
        print(f"Loading TEMPORAL LSTM model from {TEMPORAL_MODEL_PATH}...")
        temporal_detector = TemporalFightingDetector(TEMPORAL_MODEL_PATH)
        print("✅ TEMPORAL LSTM model loaded! (95.98% accuracy - improved)")
    else:
        print("⚠️ Temporal model not found. Run: python train_temporal_fighting.py")
    
    # Load ML fighting classifier (backup/fallback)
    fighting_classifier = None
    if os.path.exists(FIGHTING_CLASSIFIER_PATH):
        print(f"Loading ML fighting classifier from {FIGHTING_CLASSIFIER_PATH}...")
        with open(FIGHTING_CLASSIFIER_PATH, 'rb') as f:
            fighting_classifier = pickle.load(f)
        print("✅ ML classifier loaded!")
    else:
        print("⚠️ ML classifier not found, using heuristic detection only")
    
    print("=" * 70)
    print("🥊 SPATIO-TEMPORAL FIGHTING DETECTION (IMPROVED MODEL)")
    print("=" * 70)
    print(f"Pose model: {POSE_MODEL_PATH}")
    print(f"Temporal model: {TEMPORAL_MODEL_PATH}")
    print("Detection methods:")
    print(f"  1. LSTM Temporal Model (analyzes {TEMPORAL_SEQUENCE_LENGTH}-frame sequences)")
    print("  2. Heuristic (proximity, overlap, aggressive poses)")
    print("  3. ML Classifier (fallback)")
    print("=" * 70)
    
    def connect_stream():
        if USE_VIDEO_FILE:
            print(f"🎬 Using VIDEO FILE: {VIDEO_FILE_PATH}")
            cap = cv2.VideoCapture(VIDEO_FILE_PATH)
        elif USE_RECAMERA:
            print(f"📹 Connecting to reCamera: {RTSP_URL}")
            import os
            # Set only rtsp_transport;tcp for classic stability
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            ffmpeg_url = f"rtsp://{RECAMERA_IP}:554/live"
            cap = cv2.VideoCapture(ffmpeg_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
        else:
            print(f"📷 Using WEBCAM (ID: {WEBCAM_ID})")
            cap = cv2.VideoCapture(WEBCAM_ID)
        return cap
    
    cap = connect_stream()
    cam = ThreadedCamera(cap)
    
    start_time = time.time()
    timeout = 15 if USE_RECAMERA else 5  # Longer timeout for reCamera
    while not cam.is_opened() and (time.time() - start_time) < timeout:
        print("Waiting for camera...")
        time.sleep(1)
        cam.stop()
        cap = connect_stream()
        cam = ThreadedCamera(cap)

    if not cam.is_opened():
        print('❌ Error: Could not open camera/video')
        return

    print('✅ Camera connected! Press Q to quit.')

    frame_count = 0
    prev_frame = None
    persons_data = []
    global_status = 'Normal'
    global_color = (0, 255, 0)
    fight_history = deque(maxlen=HISTORY_LENGTH)
    consecutive_failures = 0
    MAX_FAILURES = 10  # More tolerance before reconnecting
    MAX_TOTAL_FAILURES = 50  # Hard exit after 50 consecutive failures
    RECONNECT_DELAY = 3  # Seconds to wait before reconnecting

    # Temporal model state
    temporal_prob = 0.0
    temporal_buffer_fill = 0.0
    last_frame_pos = 0  # Track video position to detect restart

    fighting_cooldown = 0
    while True:
        # Detect video restart/loop
        if USE_VIDEO_FILE:
            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            if current_pos < last_frame_pos - 10:  # Video restarted
                print("🔄 Video restarted - resetting detection")
                if temporal_detector:
                    temporal_detector.reset()
                fight_history.clear()
                temporal_prob = 0.0
                global_status = 'Normal'
                global_color = (0, 255, 0)
            last_frame_pos = current_pos
        
        # FIXED: Wrap frame reading in try-except to catch RTSP errors
        try:
            frame = cam.read()
            # If no frame for 2 seconds, reconnect
            if frame is None or cam.time_since_last_frame() > 2.0:
                consecutive_failures += 1
                if consecutive_failures % 5 == 1:
                    print(f"⚠️ Frame grab stalled ({consecutive_failures}/{MAX_FAILURES})")
                if consecutive_failures >= MAX_FAILURES:
                    print(f"🔄 Reconnecting to camera (after {consecutive_failures} failures)...")
                    cam.stop()
                    time.sleep(RECONNECT_DELAY)
                    cap = connect_stream()
                    cam = ThreadedCamera(cap)
                    consecutive_failures = 0
                    # Wait for reconnection
                    reconnect_start = time.time()
                    while not cam.is_opened() and (time.time() - reconnect_start) < 15:
                        print("[DEBUG] Waiting for camera reconnect...")
                        time.sleep(1)
                    if not cam.is_opened():
                        print("❌ Failed to reconnect. Retrying...")
                        time.sleep(2)
                        cap = connect_stream()
                        cam = ThreadedCamera(cap)
                        if not cam.is_opened():
                            print("❌ Second reconnect failed. Exiting...")
                            break
                if consecutive_failures >= MAX_TOTAL_FAILURES:
                    print(f"❌ Too many consecutive failures ({consecutive_failures}). Exiting script.")
                    break
                time.sleep(0.1)
                continue
            # Validate frame dimensions
            if frame.shape[0] == 0 or frame.shape[1] == 0:
                print("⚠️ Invalid frame dimensions, skipping...")
                consecutive_failures += 1
                if consecutive_failures >= MAX_TOTAL_FAILURES:
                    print(f"❌ Too many consecutive failures ({consecutive_failures}). Exiting script.")
                    break
                continue
            # DEBUG: Print frame shape
            print(f"[DEBUG] Frame shape: {frame.shape}")
        except Exception as e:
            print(f"❌ Frame read exception: {e}")
            consecutive_failures += 1
            if consecutive_failures >= MAX_FAILURES:
                print(f"🔄 Reconnecting due to exception (attempt after {consecutive_failures} failures)...")
                cam.stop()
                time.sleep(RECONNECT_DELAY)
                cap = connect_stream()
                cam = ThreadedCamera(cap)
                consecutive_failures = 0
                # Wait for reconnection
                reconnect_start = time.time()
                while not cam.is_opened() and (time.time() - reconnect_start) < 15:
                    print("[DEBUG] Waiting for camera reconnect...")
                    time.sleep(1)
                if not cam.is_opened():
                    print("❌ Failed to reconnect. Retrying...")
                    time.sleep(2)
                    cap = connect_stream()
                    cam = ThreadedCamera(cap)
                    if not cam.is_opened():
                        print("❌ Second reconnect failed. Exiting...")
                        break
            if consecutive_failures >= MAX_TOTAL_FAILURES:
                print(f"❌ Too many consecutive failures ({consecutive_failures}). Exiting script.")
                break
            time.sleep(0.1)
            continue
        # Frame successfully read
        consecutive_failures = 0
        frame_count += 1
        
        # FIXED: Wrap resize in try-except to catch corrupted frames
        try:
            display = cv2.resize(frame, (640, 480))
            scale_x, scale_y = 640 / frame.shape[1], 480 / frame.shape[0]
        except Exception as e:
            print(f"⚠️ Resize error: {e}, skipping frame")
            continue
        
        motion = calculate_motion(prev_frame, frame)
        motion_detected = motion > MOTION_THRESHOLD
        prev_frame = frame.copy()
        
        if frame_count % FRAME_SKIP == 0:
            persons_data = []
            fighting_detected = False
            fighting_confidence = 0.0
            fighting_reason = ""
            
            if motion_detected:
                # Run pose detection
                try:
                    pose_results = pose_model.predict(
                        source=frame, 
                        conf=PERSON_CONF, 
                        imgsz=INFERENCE_SIZE, 
                        verbose=False
                    )
                except Exception as e:
                    print(f"⚠️ YOLO inference error: {e}")
                    continue
                # DEBUG: Print number of detected people
                num_boxes = len(pose_results[0].boxes)
                print(f"[DEBUG] Detected {num_boxes} people in frame")
                if num_boxes > 0:
                    boxes = pose_results[0].boxes
                    keypoints_all = pose_results[0].keypoints if hasattr(pose_results[0], 'keypoints') else None
                    for idx, box in enumerate(boxes):
                        # Safety check for NaN values
                        coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else box.xyxy[0]
                        if np.any(np.isnan(coords)):
                            continue  # Skip boxes with NaN values
                        x1, y1, x2, y2 = map(int, coords)
                        conf = float(box.conf[0])
                        box_w, box_h = x2 - x1, y2 - y1
                        if box_w < MIN_BOX_SIZE or box_h < MIN_BOX_SIZE:
                            continue
                        # Get keypoints for this person
                        kpts = None
                        aggression_score = 0.0
                        aggression_indicators = []
                        if keypoints_all is not None and idx < len(keypoints_all.data):
                            kpts = keypoints_all.data[idx]
                            # Analyze pose for aggression
                            aggression_score, aggression_indicators = analyze_pose_aggression(kpts)
                        persons_data.append({
                            'box': (x1, y1, x2, y2),
                            'conf': conf,
                            'keypoints': kpts,
                            'aggression_score': aggression_score,
                            'aggression_indicators': aggression_indicators
                        })
                    # DEBUG: Print persons with keypoints
                    num_with_kpts = sum(1 for p in persons_data if p['keypoints'] is not None)
                    print(f"[DEBUG] Persons with keypoints: {num_with_kpts}")
                    # ========== FIGHTING DETECTION (3 methods) ========== 
                    if len(persons_data) >= 2:
                        # Get keypoints list for ML/temporal models
                        keypoints_list = []
                        for p in persons_data:
                            if p['keypoints'] is not None:
                                kpts_np = p['keypoints'].cpu().numpy() if hasattr(p['keypoints'], 'cpu') else p['keypoints']
                                if len(kpts_np.shape) == 1:
                                    kpts_np = kpts_np.reshape(-1, 3) if len(kpts_np) == 51 else kpts_np.reshape(-1, 2)
                                keypoints_list.append(kpts_np)
                        # METHOD 1: TEMPORAL LSTM (most reliable - 91% accuracy)
                        lstm_detected = False
                        if temporal_detector is not None and len(keypoints_list) >= 2:
                            temporal_prob, temporal_buffer_fill = temporal_detector.update(
                                keypoints_list, frame.shape[1], frame.shape[0]
                            )
                            if temporal_prob >= LSTM_FIGHTING_THRESHOLD:
                                lstm_detected = True
                                fighting_detected = True
                                fighting_confidence = max(fighting_confidence, temporal_prob)
                                fighting_reason = f"LSTM:{temporal_prob:.0%}"
                        # METHOD 2: Heuristic detection (proximity + aggression)
                        heuristic_fight, heuristic_conf, heuristic_reason = detect_fighting_from_poses(
                            persons_data, frame.shape[1], frame.shape[0]
                        )
                        # METHOD 3: ML classifier
                        ml_confidence = 0.0
                        if fighting_classifier is not None and len(keypoints_list) >= 2:
                            ml_features = extract_ml_features(keypoints_list, frame.shape[1], frame.shape[0])
                            if ml_features is not None:
                                ml_prob = fighting_classifier.predict_proba(ml_features)[0]
                                ml_confidence = ml_prob[1]
                        # COMBINED DETECTION LOGIC (Research-backed):
                        # Trust LSTM model - it's trained specifically for temporal fighting patterns
                        # Heuristic is supplementary, not required
                        if lstm_detected:
                            # LSTM is primary detector - trust it when confident
                            fighting_detected = True
                            fighting_confidence = temporal_prob
                            fighting_reason = f"LSTM:{temporal_prob:.0%}"
                        elif ml_confidence >= 0.80:
                            # Trust ML classifier if very confident
                            fighting_detected = True
                            fighting_confidence = ml_confidence
                            fighting_reason = f"ML:{ml_confidence:.0%}"
                        elif heuristic_conf >= 0.90:
                            # High heuristic only (strong physical contact + mutual aggression)
                            fighting_detected = True
                            fighting_confidence = heuristic_conf
                            fighting_reason = f"Heur:{heuristic_conf:.0%}"
                        else:
                            fighting_detected = False
                        if DEBUG_MODE and (fighting_confidence > 0.3 or ml_confidence > 0.5):
                            print(f"[FIGHT] LSTM:{temporal_prob:.0%} Heur:{heuristic_conf:.0%} ML:{ml_confidence:.0%} -> {'FIGHT' if fighting_detected else 'normal'}")
            
            # Update history
            fight_history.append({
                'fighting': fighting_detected,
                'confidence': fighting_confidence,
                'reason': fighting_reason,
                'num_persons': len(persons_data)
            })
            
            # TEMPORAL PERSISTENCE: Require multiple non-fighting frames to clear
            # This prevents flickering when heuristic fluctuates
            fight_count = sum(1 for h in fight_history if h['fighting'])
            if DEBUG_MODE:
                print(f"[DEBUG] LSTM fight_count: {fight_count}/{HISTORY_LENGTH} | confidences: {[round(h['confidence'],2) for h in fight_history]}")

            if fighting_detected or fight_count >= MIN_SUSPICIOUS_FRAMES:
                global_status = f' FIGHTING DETECTED ({fighting_confidence:.0%})'
                global_color = (0, 0, 255)
                fighting_cooldown = FIGHTING_COOLDOWN_FRAMES
            elif fighting_cooldown > 0:
                # Hold 'FIGHTING DETECTED' for cooldown frames
                global_status = f' FIGHTING DETECTED (cooldown)'
                global_color = (0, 0, 200)
                fighting_cooldown -= 1
            else:
                global_status = 'Normal'
                global_color = (0, 255, 0)
        
        # ===== DRAW =====
        # Draw persons with pose
        for idx, person in enumerate(persons_data):
            x1, y1, x2, y2 = person['box']
            dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
            dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)
            
            aggression = person.get('aggression_score', 0)
            is_aggressive = aggression > 0.4
            
            # Color based on aggression
            if is_aggressive:
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            # Draw bounding box
            cv2.rectangle(display, (dx1, dy1), (dx2, dy2), color, 2)
            
            # Draw skeleton if available
            if person.get('keypoints') is not None:
                # Scale keypoints for display
                kpts = person['keypoints'].cpu().numpy() if hasattr(person['keypoints'], 'cpu') else person['keypoints']
                if len(kpts.shape) == 1:
                    kpts = kpts.reshape(-1, 3) if len(kpts) == 51 else kpts.reshape(-1, 2)
                
                scaled_kpts = kpts.copy()
                scaled_kpts[:, 0] *= scale_x
                scaled_kpts[:, 1] *= scale_y
                
                skeleton_color = (0, 255, 255) if not is_aggressive else (0, 128, 255)
                draw_skeleton(display, scaled_kpts, skeleton_color, 2)
            
            # Label
            label = f"P{idx+1}"
            if aggression > 0:
                label += f" Aggr:{aggression:.0%}"
            
            indicators = person.get('aggression_indicators', [])
            if indicators:
                label += f" [{','.join(indicators[:2])}]"
            
            cv2.putText(display, label, (dx1, dy1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw lines between close people
        for i in range(len(persons_data)):
            for j in range(i + 1, len(persons_data)):
                p1, p2 = persons_data[i], persons_data[j]
                c1 = get_box_center(p1['box'])
                c2 = get_box_center(p2['box'])
                
                distance = calculate_distance(p1['box'], p2['box'])
                proximity_threshold = frame.shape[1] * CLOSE_PROXIMITY_RATIO
                
                if distance < proximity_threshold:
                    # Scale centers for display
                    dc1 = (int(c1[0] * scale_x), int(c1[1] * scale_y))
                    dc2 = (int(c2[0] * scale_x), int(c2[1] * scale_y))
                    cv2.line(display, dc1, dc2, (0, 0, 255), 2)
                    
                    # Show distance
                    mid = ((dc1[0] + dc2[0]) // 2, (dc1[1] + dc2[1]) // 2)
                    cv2.putText(display, f"{distance:.0f}px", mid, 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        # Status bar
        cv2.rectangle(display, (10, 10), (400, 55), global_color, -1)
        cv2.putText(display, global_status, (15, 42), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Info panel
        cv2.putText(display, f"Persons: {len(persons_data)}", 
                   (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display, f"Motion: {'YES' if motion_detected else 'NO'}", 
                   (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # LSTM Temporal model indicator
        if temporal_detector is not None:
            cv2.putText(display, f"LSTM Prob: {temporal_prob:.0%}", 
                       (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                       (0, 0, 255) if temporal_prob >= LSTM_FIGHTING_THRESHOLD else (0, 255, 0), 1)
            
            # Buffer fill bar
            bar_x, bar_y, bar_w, bar_h = 20, 130, 150, 15
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100, 100, 100), -1)
            fill_w = int(bar_w * temporal_buffer_fill)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 255, 0), -1)
            cv2.putText(display, f"Buffer: {temporal_buffer_fill:.0%}", (bar_x + bar_w + 10, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        else:
            # Fighting detection info (fallback)
            if fight_history:
                latest = fight_history[-1]
                cv2.putText(display, f"Fight Score: {latest['confidence']:.0%}", 
                           (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           (0, 0, 255) if latest['confidence'] > 0.5 else (200, 200, 200), 1)
        
        # Method indicator
        method_text = "LSTM+Heuristic+ML" if temporal_detector else "Heuristic+ML"
        cv2.putText(display, f"Mode: {method_text}", (600, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # FIXED: Connection status indicator
        status_color = (0, 255, 0) if consecutive_failures == 0 else (0, 165, 255)
        cv2.circle(display, (780, 30), 8, status_color, -1)
        
        cv2.imshow('Spatio-Temporal Fighting Detection', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r') and temporal_detector:
            temporal_detector.reset()
            print("🔄 Temporal buffer reset")
    
    cam.stop()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()