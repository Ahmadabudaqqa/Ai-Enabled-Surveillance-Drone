"""
Test the updated two_stage_surveillance.py with smart heuristics
"""

import cv2
import sys
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
import time

# Import from two_stage_surveillance
import sys
sys.path.insert(0, '.')

# Import the FightingDetector
from two_stage_surveillance import (
    FightingDetector, 
    POSE_MODEL_PATH, 
    TEMPORAL_MODEL_PATH,
    extract_temporal_features,
    check_aggressive_poses,
    distance_between_centroids,
    draw_skeleton,
    LSTM_FIGHTING_THRESHOLD,
    CLOSE_PROXIMITY_RATIO
)

def test_on_video(video_path, output_path):
    """Test 2-stage surveillance on video"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video info
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n{'='*60}")
    print(f"TESTING: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps:.1f} FPS ({total_frames} frames)")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Initialize detectors
    print("🔄 Loading models...")
    yolo_detector = YOLO('yolo11n.pt')
    fighting_detector = FightingDetector(POSE_MODEL_PATH, TEMPORAL_MODEL_PATH)
    print("✓ Models loaded\n")
    
    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    fighting_frames = 0
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_display = frame.copy()
        
        # Stage 1: Intrusion Detection
        yolo_results = yolo_detector(frame, verbose=False)
        persons = yolo_results[0].boxes if yolo_results[0].boxes is not None else []
        
        intrusion_count = 0
        for box in persons:
            if box.conf[0] > 0.5:
                intrusion_count += 1
        
        # Stage 2: Fighting Detection (only if persons detected)
        if intrusion_count >= 2:
            fighting_results = fighting_detector.detect_fighting(frame, persons, width, height)
            
            # Extract and draw skeletons
            pose_results = fighting_detector.pose_model(frame, verbose=False)
            if pose_results and hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None:
                for pose in pose_results:
                    if pose.keypoints is None:
                        continue
                    for keypoints_obj in pose.keypoints:
                        kpts = keypoints_obj.xy
                        if kpts is not None and len(kpts) > 0:
                            kpts_np = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
                            if len(kpts_np.shape) == 3 and kpts_np.shape[0] == 1:
                                kpts_np = kpts_np[0]
                            if len(kpts_np) >= 17:
                                # Red skeleton if fighting, green otherwise
                                skeleton_color = (0, 0, 255) if fighting_results["fighting_detected"] else (0, 255, 0)
                                draw_skeleton(frame_display, kpts_np, skeleton_color, 2)
            
            if fighting_results["fighting_detected"]:
                fighting_frames += 1
                text = f"🚨 FIGHTING! Conf: {fighting_results['confidence']:.1%}"
                cv2.putText(frame_display, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.rectangle(frame_display, (5, 5), (width-5, height-5), (0, 0, 255), 3)
            else:
                if fighting_results['confidence'] > 0:
                    text = f"⚠ Possible fight: {fighting_results['confidence']:.1%}"
                    cv2.putText(frame_display, text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        
        # Display frame info
        status = "🟢 OK" if intrusion_count < 2 else (f"🔴 INTRUSION: {intrusion_count} people" if fighting_results.get("fighting_detected") is False else "🚨 FIGHTING")
        cv2.putText(frame_display, f"Frame {frame_count}/{total_frames} - {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        out.write(frame_display)
        
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            fps_actual = frame_count / elapsed
            print(f"[Frame {frame_count:4d}] Fighting detected: {fighting_frames:3d} frames ({100*fighting_frames/frame_count:.1f}%) | {fps_actual:.1f} FPS")
    
    cap.release()
    out.release()
    
    elapsed = time.time() - start_time
    fps_actual = frame_count / elapsed
    
    print(f"\n{'='*60}")
    print(f"✓ Processing complete!")
    print(f"Total frames: {frame_count}")
    print(f"Fighting detected: {fighting_frames} frames ({100*fighting_frames/frame_count:.1f}%)")
    print(f"Processing speed: {fps_actual:.1f} FPS")
    print(f"Output saved: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_2stage_surveillance.py <video_path> [output_path]")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "test_2stage_output.mp4"
    
    test_on_video(video_path, output_path)
