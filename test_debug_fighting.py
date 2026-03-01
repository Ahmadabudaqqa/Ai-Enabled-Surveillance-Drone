"""
DEBUG 2-STAGE SURVEILLANCE TEST
With detailed pose extraction logging
"""

import cv2
import sys
import os
import torch
import torch.nn as nn
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
from datetime import datetime
from intrusion_detector import IntrusionDetector
import torch.nn.functional as F

# ============== SETTINGS ==============
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120

POSE_MODEL_PATH = 'runs/pose/human_pose_detector/weights/best.pt'
TEMPORAL_MODEL_PATH = 'fighting_temporal_model_v2/fighting_lstm_v2.pt'

TEMPORAL_SEQUENCE_LENGTH = 8
TEMPORAL_FEATURE_DIM = 68

LSTM_FIGHTING_THRESHOLD = 0.20

# Keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12

# ============== LSTM MODEL ==============
class FightingLSTM(nn.Module):
    def __init__(self, input_size=TEMPORAL_FEATURE_DIM, hidden_size=256, num_layers=3, dropout=0.4, num_heads=4):
        super(FightingLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        self.lstm_norm = nn.LayerNorm(hidden_size * 2)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.attention2 = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
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
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        lstm_out = self.lstm_norm(lstm_out)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        combined = lstm_out + attn_out
        attn_weights = F.softmax(self.attention2(combined), dim=1)
        context = torch.sum(attn_weights * combined, dim=1)
        return self.classifier(context).squeeze(-1)

# ============== FIGHTING DETECTOR ==============
class FightingDetector:
    def __init__(self, pose_model_path, temporal_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.pose_model = YOLO(pose_model_path)
        
        self.temporal_model = FightingLSTM().to(self.device)
        checkpoint = torch.load(temporal_model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            self.temporal_model.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.temporal_model.load_state_dict(checkpoint)
        self.temporal_model.eval()
        print(f"✓ Loaded temporal model from {temporal_model_path}")
        
        self.person_sequences = {}
        self.frame_count = 0
    
    def detect_fighting(self, frame, persons, frame_width, frame_height):
        """DEBUG: Detailed pose extraction logging"""
        self.frame_count += 1
        
        fighting_results = {
            "fighting_detected": False,
            "persons_analyzed": 0,
            "avg_probability": 0.0,
            "sequences_ready": 0
        }
        
        if self.temporal_model is None or not persons or len(persons) == 0:
            return fighting_results
        
        # ===== DEBUG: Pose Extraction =====
        print(f"\n[Frame {self.frame_count}] DEBUG:")
        print(f"  Input: {len(persons)} person boxes detected")
        
        # Extract poses
        pose_results = self.pose_model(frame, verbose=False)
        print(f"  Pose model output: {len(pose_results)} results")
        
        if not pose_results:
            print(f"  → No pose results!")
            return fighting_results
        
        if not hasattr(pose_results[0], 'keypoints'):
            print(f"  → No keypoints attribute!")
            return fighting_results
        
        if pose_results[0].keypoints is None:
            print(f"  → Keypoints is None!")
            return fighting_results
        
        print(f"  Keypoints found: {len(pose_results[0].keypoints)}")
        
        keypoints_list = []
        for i, keypoints_obj in enumerate(pose_results[0].keypoints):
            if keypoints_obj is None:
                print(f"    Person {i}: No keypoints")
                continue
            
            kpts = keypoints_obj.xy
            if kpts is None or len(kpts) == 0:
                print(f"    Person {i}: Empty keypoints")
                continue
            
            # keypoints_obj.xy is [1, 17, 2] (batch, joints, xy)
            kpts_np = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
            print(f"    Person {i}: Shape {kpts_np.shape}")
            
            # Remove batch dimension if present
            if len(kpts_np.shape) == 3 and kpts_np.shape[0] == 1:
                kpts_np = kpts_np[0]  # Shape becomes (17, 2)
            
            if len(kpts_np) >= 17:
                # Flatten to 34 dims (17 joints × 2 coords), or pad to 68
                flat = kpts_np.flatten()[:34]
                # Pad to 68 dimensions
                while len(flat) < 68:
                    flat = np.append(flat, 0.0)
                keypoints_list.append(flat[:68])
                print(f"      ✓ Valid keypoints added (flattened to 68 dims)")
        
        print(f"  Total valid keypoints: {len(keypoints_list)} people")
        fighting_results["persons_analyzed"] = len(keypoints_list)
        
        # ===== DEBUG: Sequence Tracking =====
        print(f"  Active sequences: {len(self.person_sequences)}")
        
        # Simple index-based tracking (limited but works)
        for idx, kpts in enumerate(keypoints_list):
            if idx not in self.person_sequences:
                self.person_sequences[idx] = deque(maxlen=TEMPORAL_SEQUENCE_LENGTH)
            
            self.person_sequences[idx].append(kpts)
            seq_len = len(self.person_sequences[idx])
            print(f"    Person {idx}: sequence length {seq_len}/{TEMPORAL_SEQUENCE_LENGTH}")
        
        # Check for complete sequences
        sequences_ready = sum(1 for seq in self.person_sequences.values() if len(seq) == TEMPORAL_SEQUENCE_LENGTH)
        fighting_results["sequences_ready"] = sequences_ready
        print(f"  Ready sequences: {sequences_ready}")
        
        # Analyze complete sequences
        fighting_probs = []
        for person_id, seq in self.person_sequences.items():
            if len(seq) == TEMPORAL_SEQUENCE_LENGTH:
                print(f"    Analyzing person {person_id}...")
                seq_tensor = torch.FloatTensor(np.array(list(seq))).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logits = self.temporal_model(seq_tensor)
                    prob = logits.item()
                    fighting_probs.append(prob)
                    print(f"      Fighting probability: {prob:.4f}")
        
        avg_prob = np.mean(fighting_probs) if fighting_probs else 0
        fighting_results["avg_probability"] = avg_prob
        fighting_results["fighting_detected"] = any(p > LSTM_FIGHTING_THRESHOLD for p in fighting_probs)
        
        if fighting_results["fighting_detected"]:
            print(f"  🔴 FIGHTING DETECTED: {avg_prob:.4f}")
        
        return fighting_results

# ============== MAIN TEST ==============
def test_debug(video_path):
    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        return
    
    # Initialize detectors
    intrusion_detector = IntrusionDetector()
    fighting_detector = FightingDetector(POSE_MODEL_PATH, TEMPORAL_MODEL_PATH)
    yolo_detection = YOLO('yolo11n.pt')
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("✗ Failed to open video")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video: {frame_width}x{frame_height} @ {fps} FPS ({total_frames} frames)")
    print(f"Duration: {total_frames / fps:.1f} seconds\n")
    print("Processing first 100 frames with DEBUG output...\n")
    
    frame_count = 0
    start_time = time.time()
    max_debug_frames = 100
    
    while frame_count < max_debug_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Stage 1: Intrusion
        yolo_results = yolo_detection(frame, verbose=False)
        intrusions, alerts = intrusion_detector.detect_intrusions(
            yolo_results[0] if yolo_results else None,
            frame_width,
            frame_height
        )
        
        persons_in_zone = sum(intr["persons_count"] for intr in intrusions.values())
        
        if persons_in_zone > 0:
            print(f"\n[Frame {frame_count}] 🔴 INTRUSION: {persons_in_zone} person(s)")
            
            # Stage 2: Fighting
            all_persons_in_zones = []
            for intr in intrusions.values():
                all_persons_in_zones.extend(intr["persons"])
            
            fighting_results = fighting_detector.detect_fighting(
                frame,
                all_persons_in_zones,
                frame_width,
                frame_height
            )
    
    cap.release()
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"Processed: {frame_count} frames in {elapsed:.1f}s")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python test_debug_fighting.py <video_path>")
        sys.exit(1)
    
    test_debug(sys.argv[1])
