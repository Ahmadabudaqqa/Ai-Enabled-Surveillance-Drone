"""
STAGE 1 ONLY TEST: Intrusion Detection on Video/Image
Tests only the intrusion detection (person detection in zone)
"""

import cv2
import sys
import os
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from intrusion_detector import IntrusionDetector
import time

def test_on_video(video_path, output_path=None):
    """Test intrusion detection on video file"""
    print(f"\n{'='*70}")
    print(f"STAGE 1: INTRUSION DETECTION TEST")
    print(f"Testing on VIDEO: {video_path}")
    print(f"{'='*70}")
    
    if not os.path.exists(video_path):
        print(f"✗ Video not found: {video_path}")
        return
    
    # Initialize detectors
    intrusion_detector = IntrusionDetector("intrusion_test_log.json")
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
    
    print(f"Video info: {frame_width}x{frame_height} @ {fps} FPS ({total_frames} frames)")
    print(f"Duration: {total_frames / fps:.1f} seconds")
    
    # Setup output video if requested
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        print(f"Output: {output_path}")
    
    # Processing
    frame_count = 0
    intrusion_count = 0
    max_persons = 0
    start_time = time.time()
    
    print(f"\nProcessing frames...\n")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # ========== STAGE 1: INTRUSION DETECTION ==========
        yolo_results = yolo_detection(frame, verbose=False)
        intrusions, alerts = intrusion_detector.detect_intrusions(
            yolo_results[0] if yolo_results else None,
            frame_width,
            frame_height
        )
        
        # Count persons
        total_persons_in_zone = sum(intr["persons_count"] for intr in intrusions.values())
        if total_persons_in_zone > max_persons:
            max_persons = total_persons_in_zone
        
        if alerts:
            for alert in alerts:
                print(f"[Frame {frame_count:4d}] {alert['type']:20s} | {alert['zone_name']:25s} | {alert.get('persons_count', 0)} persons")
                if alert['type'] == 'INTRUSION':
                    intrusion_count += 1
                intrusion_detector.log_alert(alert)
        
        # Draw zones and info on frame
        frame_display = intrusion_detector.draw_zones(frame)
        
        # Draw info text
        y_offset = 30
        cv2.putText(frame_display, f"Frame: {frame_count}/{total_frames}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y_offset += 30
        
        if total_persons_in_zone > 0:
            cv2.putText(frame_display, f"INTRUSION: {total_persons_in_zone} person(s) in zone!", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 3)
            y_offset += 30
        
        # Write frame
        if out:
            out.write(frame_display)
        
        # Display progress
        if frame_count % 50 == 0:
            elapsed = time.time() - start_time
            print(f"  → Processed {frame_count}/{total_frames} frames ({frame_count/elapsed:.1f} FPS)")
    
    # Cleanup
    cap.release()
    if out:
        out.release()
    
    # Summary
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"SUMMARY - STAGE 1 (Intrusion Detection)")
    print(f"{'='*70}")
    print(f"Processed: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    print(f"Intrusion alerts: {intrusion_count}")
    print(f"Max persons in zone: {max_persons}")
    if output_path:
        print(f"Output video: {output_path}")
    print(f"Logs: intrusion_test_log.json")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    print("\n" + "="*70)
    print("STAGE 1 INTRUSION DETECTION TEST (Video/Image)")
    print("="*70)
    print("Usage:")
    print("  python test_intrusion_only.py <video_path> [output_path]")
    print("\nExample:")
    print("  python test_intrusion_only.py surveillanceVideos/1657Pri_OutFG_C1.mp4 output.mp4")
    print("="*70 + "\n")
    
    if len(sys.argv) < 2:
        print("No file provided. Using test video...\n")
        test_video = "surveillanceVideos/1657Pri_OutFG_C1.mp4"
        
        if os.path.exists(test_video):
            output_video = "test_intrusion_output.mp4"
            test_on_video(test_video, output_video)
        else:
            print(f"✗ Test video not found: {test_video}")
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        test_on_video(input_file, output_file)
