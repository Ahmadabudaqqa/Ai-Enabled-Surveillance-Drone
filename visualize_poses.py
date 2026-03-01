"""
POSE VISUALIZATION WITH KEYPOINTS & SKELETON
Shows 17 COCO keypoints and body skeleton on video frames
"""

import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO

# Keypoint names (COCO format - 17 joints)
KEYPOINT_NAMES = [
    'NOSE', 'L_EYE', 'R_EYE', 'L_EAR', 'R_EAR',
    'L_SHOULDER', 'R_SHOULDER', 'L_ELBOW', 'R_ELBOW',
    'L_WRIST', 'R_WRIST', 'L_HIP', 'R_HIP',
    'L_KNEE', 'R_KNEE', 'L_ANKLE', 'R_ANKLE'
]

# Keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

# Skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head
    (5, 6),  # Shoulders
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (5, 11), (6, 12), (11, 12),  # Torso
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16)  # Right leg
]

# Colors for visualization
COLORS = {
    "keypoint": (0, 255, 0),  # Green
    "skeleton": (255, 0, 0),  # Blue
    "raised_arm": (0, 0, 255),  # Red (aggressive pose)
    "label": (255, 255, 255)  # White
}

POSE_MODEL_PATH = 'runs/pose/human_pose_detector/weights/best.pt'

def draw_pose(frame, keypoints, person_id=0, show_labels=True, check_aggressive=False):
    """Draw pose skeleton and keypoints on frame"""
    
    # Filter valid keypoints
    valid_kpts = []
    for i, kpt in enumerate(keypoints):
        if len(kpt) >= 2 and kpt[0] > 0 and kpt[1] > 0:
            valid_kpts.append((i, int(kpt[0]), int(kpt[1])))
    
    if not valid_kpts:
        return
    
    # Draw skeleton connections
    for connection in SKELETON_CONNECTIONS:
        kpt1_idx = connection[0]
        kpt2_idx = connection[1]
        
        kpt1 = None
        kpt2 = None
        
        for idx, x, y in valid_kpts:
            if idx == kpt1_idx:
                kpt1 = (x, y)
            if idx == kpt2_idx:
                kpt2 = (x, y)
        
        if kpt1 and kpt2:
            cv2.line(frame, kpt1, kpt2, COLORS["skeleton"], 2)
    
    # Check for aggressive pose (arms raised)
    aggressive = False
    if len(keypoints) >= 17:
        left_wrist = keypoints[LEFT_WRIST]
        right_wrist = keypoints[RIGHT_WRIST]
        left_shoulder = keypoints[LEFT_SHOULDER]
        right_shoulder = keypoints[RIGHT_SHOULDER]
        
        if (left_wrist[1] < left_shoulder[1] or 
            right_wrist[1] < right_shoulder[1]):
            aggressive = True
    
    # Draw keypoints
    for idx, x, y in valid_kpts:
        color = COLORS["raised_arm"] if (aggressive and idx in [LEFT_WRIST, RIGHT_WRIST]) else COLORS["keypoint"]
        
        cv2.circle(frame, (x, y), 4, color, -1)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)
        
        if show_labels and idx < len(KEYPOINT_NAMES):
            label = KEYPOINT_NAMES[idx]
            cv2.putText(frame, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, COLORS["label"], 1)
    
    # Draw aggressive pose indicator
    if aggressive and check_aggressive:
        y_offset = 50 + person_id * 25
        cv2.putText(frame, f"Person {person_id}: ARMS RAISED!", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS["raised_arm"], 2)

def visualize_poses(video_path, output_path=None, show_labels=True):
    """Visualize poses in video"""
    
    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        return
    
    # Load pose model
    print("🔄 Loading YOLO pose model...")
    pose_model = YOLO(POSE_MODEL_PATH)
    print(f"✓ Pose model loaded")
    
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
    
    # Setup output
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_count = 0
    max_people_per_frame = 0
    
    print("Processing frames with pose visualization...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        frame_display = frame.copy()
        
        # Extract poses
        pose_results = pose_model(frame, verbose=False)
        
        if pose_results and hasattr(pose_results[0], 'keypoints'):
            person_id = 0
            for pose in pose_results:
                if pose.keypoints is None or len(pose.keypoints) == 0:
                    continue
                
                for keypoints_obj in pose.keypoints:
                    kpts = keypoints_obj.xy
                    if kpts is None or len(kpts) == 0:
                        continue
                    
                    kpts_np = kpts.cpu().numpy() if hasattr(kpts, 'cpu') else kpts
                    
                    # Remove batch dimension if present
                    if len(kpts_np.shape) == 3 and kpts_np.shape[0] == 1:
                        kpts_np = kpts_np[0]
                    
                    if len(kpts_np) >= 17:
                        draw_pose(frame_display, kpts_np, person_id, show_labels=True, check_aggressive=True)
                        person_id += 1
            
            max_people_per_frame = max(max_people_per_frame, person_id)
        
        # Add frame info
        info_text = f"Frame: {frame_count}/{total_frames}"
        cv2.putText(frame_display, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add keypoint legend
        cv2.putText(frame_display, "Legend:", (frame_width - 200, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(frame_display, (frame_width - 50, 50), 3, COLORS["keypoint"], -1)
        cv2.putText(frame_display, "Keypoint", (frame_width - 200, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["keypoint"], 1)
        
        cv2.line(frame_display, (frame_width - 200, 75), (frame_width - 150, 75), COLORS["skeleton"], 2)
        cv2.putText(frame_display, "Skeleton", (frame_width - 200, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["skeleton"], 1)
        
        cv2.circle(frame_display, (frame_width - 50, 100), 3, COLORS["raised_arm"], -1)
        cv2.putText(frame_display, "Aggressive", (frame_width - 200, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["raised_arm"], 1)
        
        if out:
            out.write(frame_display)
        
        if frame_count % 50 == 0:
            print(f"  → {frame_count}/{total_frames} frames processed")
    
    cap.release()
    if out:
        out.release()
    
    print(f"\n{'='*70}")
    print(f"POSE VISUALIZATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total frames: {frame_count}")
    print(f"Max people in single frame: {max_people_per_frame}")
    if output_path:
        print(f"Output video: {output_path}")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python visualize_poses.py <video> [output]")
        sys.exit(1)
    
    visualize_poses(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
