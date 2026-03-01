"""
Collect Custom Training Data from reCamera
Records clips and saves frames for each activity class
This will help improve model accuracy for YOUR specific environment
"""
import cv2
import os
import time
from datetime import datetime

# reCamera settings
RECAMERA_IP = '192.168.42.1'
RTSP_URL = f'rtsp://{RECAMERA_IP}:554/live'

# Output directory
OUTPUT_DIR = 'custom_training_data'

# Activity classes to collect
ACTIVITIES = [
    'walking',           # Press 1 - Normal walking
    'person_running',    # Press 2 - Running
    'fighting_group',    # Press 3 - Two people fighting
    'aggressive_activity', # Press 4 - Aggressive gestures/behavior  
    'person_pushing',    # Press 5 - Pushing someone
    'leaving_package',   # Press 6 - Leaving a bag/package
    'passing_out',       # Press 7 - Falling down/collapsing
    'robbery_knife',     # Press 8 - Robbery with weapon
]

# How many frames to save per recording
FRAMES_PER_RECORDING = 30
FRAME_INTERVAL = 3  # Save every 3rd frame

def create_directories():
    """Create output directories for each activity"""
    for activity in ACTIVITIES:
        path = os.path.join(OUTPUT_DIR, activity)
        os.makedirs(path, exist_ok=True)
    print(f"✅ Created directories in {OUTPUT_DIR}/")

def get_frame_count(activity):
    """Count existing frames for an activity"""
    path = os.path.join(OUTPUT_DIR, activity)
    if os.path.exists(path):
        return len([f for f in os.listdir(path) if f.endswith('.jpg')])
    return 0

def show_instructions():
    """Display recording instructions"""
    print("\n" + "=" * 60)
    print("📹 CUSTOM TRAINING DATA COLLECTOR")
    print("=" * 60)
    print("\nPress a NUMBER KEY to start recording that activity:")
    print("-" * 40)
    for i, activity in enumerate(ACTIVITIES, 1):
        count = get_frame_count(activity)
        print(f"  [{i}] {activity:20s} ({count} frames collected)")
    print("-" * 40)
    print("  [S] Show current frame counts")
    print("  [Q] Quit")
    print("=" * 60)
    print("\n💡 TIPS:")
    print("  - For 'walking': Walk normally in front of camera")
    print("  - For 'fighting': Have 2 people simulate fighting")
    print("  - For 'running': Run across camera view")
    print("  - For 'passing_out': Simulate falling/collapsing")
    print("  - Collect at least 50-100 frames per activity")
    print("=" * 60)

def record_activity(cap, activity_name, activity_idx):
    """Record frames for a specific activity"""
    print(f"\n🔴 RECORDING: {activity_name}")
    print(f"   Perform the activity now! Recording {FRAMES_PER_RECORDING} frames...")
    print("   Press SPACE to stop early, ESC to cancel\n")
    
    output_path = os.path.join(OUTPUT_DIR, activity_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    frame_count = 0
    saved_count = 0
    
    while saved_count < FRAMES_PER_RECORDING:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Frame read failed, retrying...")
            continue
        
        frame_count += 1
        
        # Save every Nth frame
        if frame_count % FRAME_INTERVAL == 0:
            # Add some variation - slight crops and flips
            h, w = frame.shape[:2]
            
            # Random crop (95-100% of frame)
            crop_pct = 0.95 + (saved_count % 10) * 0.005
            new_h, new_w = int(h * crop_pct), int(w * crop_pct)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            cropped = frame[y_off:y_off+new_h, x_off:x_off+new_w]
            cropped = cv2.resize(cropped, (224, 224))
            
            # Save frame
            filename = f"{activity_name}_{timestamp}_{saved_count:04d}.jpg"
            filepath = os.path.join(output_path, filename)
            cv2.imwrite(filepath, cropped)
            saved_count += 1
            
            # Show progress
            progress = "█" * (saved_count * 20 // FRAMES_PER_RECORDING)
            progress += "░" * (20 - len(progress))
            print(f"\r   [{progress}] {saved_count}/{FRAMES_PER_RECORDING}", end="")
        
        # Show live preview with recording indicator
        display = frame.copy()
        cv2.putText(display, f"RECORDING: {activity_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(display, f"Frames: {saved_count}/{FRAMES_PER_RECORDING}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add red recording dot
        cv2.circle(display, (display.shape[1] - 30, 30), 15, (0, 0, 255), -1)
        
        cv2.imshow('Recording', display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space - stop early
            print(f"\n   ⏹️ Stopped early with {saved_count} frames")
            break
        elif key == 27:  # ESC - cancel
            print(f"\n   ❌ Cancelled recording")
            return 0
    
    print(f"\n   ✅ Saved {saved_count} frames to {output_path}/")
    return saved_count

def main():
    create_directories()
    show_instructions()
    
    print(f"\n📹 Connecting to reCamera: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Wait for connection
    start_time = time.time()
    while not cap.isOpened() and (time.time() - start_time) < 10:
        print("   Waiting for camera...")
        time.sleep(0.5)
        cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
    
    if not cap.isOpened():
        print("❌ Could not connect to reCamera!")
        print("   Make sure reCamera is connected via USB")
        return
    
    print("✅ Camera connected!")
    
    total_collected = 0
    
    while True:
        ret, frame = cap.read()
        if ret:
            # Show live preview
            display = frame.copy()
            cv2.putText(display, "Press 1-8 to record activity, Q to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show activity options
            for i, activity in enumerate(ACTIVITIES[:4], 1):
                cv2.putText(display, f"[{i}] {activity}", (10, 60 + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            for i, activity in enumerate(ACTIVITIES[4:], 5):
                cv2.putText(display, f"[{i}] {activity}", (300, 60 + (i-4)*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow('Recording', display)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Number keys 1-8 for activities
        if ord('1') <= key <= ord('8'):
            idx = key - ord('1')
            if idx < len(ACTIVITIES):
                collected = record_activity(cap, ACTIVITIES[idx], idx)
                total_collected += collected
        
        elif key == ord('s') or key == ord('S'):
            show_instructions()
        
        elif key == ord('q') or key == ord('Q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print(f"📊 SESSION SUMMARY")
    print("=" * 60)
    print(f"Total frames collected this session: {total_collected}")
    print("\nFrames per activity:")
    for activity in ACTIVITIES:
        count = get_frame_count(activity)
        status = "✅" if count >= 50 else "⚠️ Need more"
        print(f"  {activity:20s}: {count:4d} frames {status}")
    
    print("\n💡 Next steps:")
    print("  1. Collect at least 50-100 frames per activity")
    print("  2. Run: python retrain_with_custom_data.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
