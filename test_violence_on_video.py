"""
Test violence detection on a video file
Simple version without camera dependencies
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

# ============== SETTINGS ==============
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240

POSE_MODEL_PATH = 'runs/pose/human_pose_detector/weights/best.pt'
TEMPORAL_MODEL_PATH = 'fighting_temporal_model_v2/fighting_lstm_v2.pt'

TEMPORAL_SEQUENCE_LENGTH = 8
TEMPORAL_FEATURE_DIM = 68

LSTM_FIGHTING_THRESHOLD = 0.15

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
        attn_weights = torch.nn.functional.softmax(self.attention2(combined), dim=1)
        context = torch.sum(attn_weights * combined, dim=1)
        return self.classifier(context).squeeze(-1)


class ViolenceDetector:
    def __init__(self, pose_model_path, temporal_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"✓ Using device: {self.device}")
        
        # Load pose model
        print(f"✓ Loading pose model from {pose_model_path}...")
        self.pose_model = YOLO(pose_model_path)
        
        # Load temporal model
        print(f"✓ Loading temporal model from {temporal_model_path}...")
        self.temporal_model = FightingLSTM().to(self.device)
        checkpoint = torch.load(temporal_model_path, map_location=self.device, weights_only=False)
        # Handle both raw weights and checkpoint with model_state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.temporal_model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.temporal_model.load_state_dict(checkpoint)
        self.temporal_model.eval()
        
        # Tracking
        self.person_sequences = {}
        self.violence_detections = []
    
    def detect_violence(self, frame):
        """Detect violence in frame"""
        try:
            # Pose detection
            results = self.pose_model(frame, verbose=False)
            
            if not results or len(results) == 0:
                return {"violence": False, "confidence": 0, "persons": 0}
            
            keypoints_list = []
            
            # Extract keypoints
            for result in results:
                if result.keypoints is None:
                    continue
                
                for person_idx, keypoints_obj in enumerate(result.keypoints):
                    if keypoints_obj is None or keypoints_obj.xy is None:
                        continue
                    
                    kpts = keypoints_obj.xy
                    kpts_np = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
                    
                    # Remove batch dimension if present
                    if len(kpts_np.shape) == 3 and kpts_np.shape[0] == 1:
                        kpts_np = kpts_np[0]
                    
                    if len(kpts_np) >= 17:
                        # Flatten to 68 dimensions
                        flat = kpts_np.flatten()[:34]
                        while len(flat) < 68:
                            flat = np.append(flat, 0.0)
                        keypoints_list.append(flat[:68])
            
            # Track sequences
            for idx, kpts in enumerate(keypoints_list):
                if idx not in self.person_sequences:
                    self.person_sequences[idx] = deque(maxlen=TEMPORAL_SEQUENCE_LENGTH)
                self.person_sequences[idx].append(kpts)
            
            # Analyze complete sequences
            violence_detected = False
            max_confidence = 0
            
            with torch.no_grad():
                for person_id, seq in self.person_sequences.items():
                    if len(seq) == TEMPORAL_SEQUENCE_LENGTH:
                        seq_tensor = torch.FloatTensor(np.array(list(seq))).unsqueeze(0).to(self.device)
                        pred = self.temporal_model(seq_tensor)
                        confidence = float(pred.cpu().numpy()[0])
                        max_confidence = max(max_confidence, confidence)
                        
                        if confidence > LSTM_FIGHTING_THRESHOLD:
                            violence_detected = True
            
            return {
                "violence": violence_detected,
                "confidence": max_confidence,
                "persons": len(keypoints_list)
            }
        
        except Exception as e:
            print(f"Error in violence detection: {e}")
            return {"violence": False, "confidence": 0, "persons": 0}


def test_on_video(video_path):
    """Test violence detection on video"""
    
    if not os.path.exists(video_path):
        print(f"❌ Video not found: {video_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"Violence Detection Test")
    print(f"{'='*70}")
    print(f"Video: {video_path}")
    print(f"Video size: {os.path.getsize(video_path) / (1024*1024):.2f} MB")
    
    # Initialize detector
    detector = ViolenceDetector(POSE_MODEL_PATH, TEMPORAL_MODEL_PATH)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Failed to open video")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✓ Resolution: {frame_width}x{frame_height}")
    print(f"✓ FPS: {fps}")
    print(f"✓ Total frames: {total_frames}")
    print(f"✓ Duration: {total_frames/fps:.1f}s")
    print(f"\n{'='*70}")
    print(f"Processing video...")
    print(f"{'='*70}\n")
    
    frame_count = 0
    violence_frames = []
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process at full resolution for better detection
            result = detector.detect_violence(frame)
            
            # Debug: Show person detection every 30 frames
            if frame_count % 30 == 1 and result["persons"] > 0:
                print(f"  Frame {frame_count}: {result['persons']} person(s) detected, max confidence: {result['confidence']:.2%}")
            
            # Show ALL high confidence detections
            if result["confidence"] > 0.05:
                print(f"  ⚠ Frame {frame_count}: Confidence {result['confidence']:.2%} (persons: {result['persons']})")
            
            if result["violence"]:
                violence_frames.append({
                    "frame": frame_count,
                    "confidence": result["confidence"],
                    "persons": result["persons"],
                    "time": f"{(frame_count/fps):.1f}s"
                })
                print(f"⚠️  VIOLENCE DETECTED at frame {frame_count} ({result['time']}) - "
                      f"Confidence: {result['confidence']:.2%}, Persons: {result['persons']}")
            
            # Progress
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps_actual = frame_count / elapsed
                print(f"  Processed {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%) "
                      f"- {fps_actual:.1f} FPS")
    
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
    
    finally:
        cap.release()
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total frames processed: {frame_count}")
    print(f"Total time: {elapsed:.1f}s")
    print(f"Processing speed: {frame_count/elapsed:.1f} FPS")
    print(f"Violence detections: {len(violence_frames)}")
    
    if violence_frames:
        print(f"\nViolence frames detected:")
        for vf in violence_frames[:20]:  # Show first 20
            print(f"  - Frame {vf['frame']} ({vf['time']}) - Conf: {vf['confidence']:.2%}")
        if len(violence_frames) > 20:
            print(f"  ... and {len(violence_frames)-20} more")
    else:
        print(f"No violence detected")
    
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Default to first violence video if no argument provided
    if len(sys.argv) < 2:
        video_path = r'rwf2000_download/Violence/V_1.mp4'
    else:
        video_path = sys.argv[1]
    
    test_on_video(video_path)
