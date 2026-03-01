# DRONE SURVEILLANCE SYSTEM WITH FIGHTING DETECTION

## Overview

This system provides **complete drone-based surveillance** with:

1. **Intrusion Detection** - Detects when persons enter protected zones
2. **Automatic Tracking** - Drone automatically follows detected intruders
3. **Fighting Detection** - Real-time fighting analysis while tracking
4. **Visual Feedback** - Live HUD showing drone status and alerts

## System Architecture

```
Video Input
    ↓
[Stage 1] YOLO11n Person Detection
    ↓
[Stage 2] Intrusion Zone Check
    ├─→ No intrusion → IDLE (GREEN)
    └─→ Intrusion → Activate tracking
         ↓
    [Stage 2b] Drone Tracking & Movement
    ├─→ Lock onto person
    ├─→ Calculate movement vectors
    ├─→ Simulate drone following
         ↓
    [Stage 3] Fighting Detection
    ├─→ Extract pose keypoints
    ├─→ Build temporal features (68-dim)
    ├─→ Run LSTM model
    ├─→ Check proximity + LSTM threshold
    └─→ Output alert (YELLOW/RED)
         ↓
    [Visualization] Annotate frame with:
    ├─→ Intrusion zones (blue polygons)
    ├─→ Detected persons (green/red boxes)
    ├─→ Drone HUD (crosshair, velocity, lock status)
    ├─→ Fighting indicator (confidence %)
    └─→ Alert level (GREEN/YELLOW/RED)
         ↓
Output Video
```

## File Structure

```
fyp detection/
├── drone_surveillance.py              # Basic drone tracking module
├── drone_surveillance_advanced.py     # Advanced system with fighting detection
├── test_drone_surveillance.py         # Quick start test script
├── DRONE_SURVEILLANCE_GUIDE.md        # This file
│
├── zone_config.py                     # Define protected zones
├── two_stage_surveillance.py          # Main surveillance system
│
├── models/
│   ├── yolo11n.pt                     # YOLO person detection
│   ├── runs/pose/.../best.pt          # Pose keypoint extraction
│   └── fighting_temporal_model_v2/
│       └── fighting_lstm_v2.pt        # Fighting detection LSTM
│
└── output/
    ├── drone_surveillance_output.mp4
    ├── drone_output_v1.mp4
    └── drone_output_field.mp4
```

## Key Components

### 1. DroneConfig
Defines drone movement parameters:
```python
class DroneConfig:
    FRAME_WIDTH = 1920
    FRAME_HEIGHT = 1080
    MAX_SPEED = 50              # pixels per frame
    ACCELERATION = 5            # pixel/frame²
    SMOOTH_FACTOR = 0.7         # tracking smoothness
    DEAD_ZONE = 150             # center region before movement
    TARGET_LOCK_FRAMES = 5      # frames to lock target
```

### 2. DroneTracker
Handles tracking state and movement calculation:
```python
tracker = DroneTracker()
tracker.update_target(person_positions)  # Track person
movement = tracker.calculate_movement()   # Get drone movement
info = tracker.get_tracking_info()       # Get HUD data
```

### 3. DroneSurveillanceSystem
Main surveillance engine:
```python
system = DroneSurveillanceSystem(
    intrusion_zones=ZONES,
    video_source='video.mp4'
)
system.run_surveillance(
    output_path='output.mp4',
    display=True
)
```

### 4. AdvancedDroneSurveillance
Extended system with fighting detection:
```python
surveillance = AdvancedDroneSurveillance(
    intrusion_zones=ZONES,
    video_source='video.mp4'
)
surveillance.run_surveillance(
    output_path='output.mp4',
    display=True
)
```

## Defining Intrusion Zones

### Basic Zone Definition

```python
INTRUSION_ZONES = {
    'Zone Name': {
        'points': [
            [x1, y1],  # Top-left
            [x2, y2],  # Top-right
            [x3, y3],  # Bottom-right
            [x4, y4]   # Bottom-left
        ],
        'alert_distance': 100  # Safety perimeter
    }
}
```

### Example: Multi-Zone Setup

```python
INTRUSION_ZONES = {
    'Main Gate': {
        'points': [[100, 100], [600, 100], [600, 500], [100, 500]],
        'alert_distance': 50
    },
    'Parking Area': {
        'points': [[800, 200], [1500, 200], [1500, 700], [800, 700]],
        'alert_distance': 50
    },
    'Building Entrance': {
        'points': [[1600, 300], [1900, 300], [1900, 800], [1600, 800]],
        'alert_distance': 50
    }
}
```

## Usage

### Quick Start

```python
from drone_surveillance_advanced import AdvancedDroneSurveillance

ZONES = {
    'Protected Area': {
        'points': [[300, 200], [900, 200], [900, 600], [300, 600]],
        'alert_distance': 100
    }
}

system = AdvancedDroneSurveillance(
    intrusion_zones=ZONES,
    video_source='video.mp4'
)

system.run_surveillance(
    output_path='output.mp4',
    display=True
)
```

### Run Tests

```bash
# Run all test cases
python test_drone_surveillance.py

# Or run on single video
python -c "
from test_drone_surveillance import run_drone_surveillance
run_drone_surveillance('input.mp4', 'output.mp4', display=True)
"
```

### With Different Models

```python
surveillance = AdvancedDroneSurveillance(
    intrusion_zones=ZONES,
    video_source='video.mp4',
    pose_model_path='custom_pose_model.pt',
    lstm_model_path='custom_fighting_model.pt'
)
```

## Output Visualization

### HUD Elements

1. **Drone Crosshair** (Yellow)
   - Center of drone tracking focus
   - Circle shows dead zone (no movement threshold)

2. **Tracking Box** (Left side)
   - Lock confidence (0-100%)
   - Current speed (pixels/frame)
   - Zoom level
   - Tracking duration

3. **Fighting Indicator** (Bottom left)
   - Status: DETECTED/CLEAR
   - Confidence percentage
   - Number of people detected

4. **Alert Level** (Top right)
   - GREEN: No intrusion
   - YELLOW: Intrusion detected
   - RED: Fighting detected

5. **Velocity Vector** (Green arrow from crosshair)
   - Direction and magnitude of drone movement

6. **Movement Trail** (Fading line)
   - Historical drone path (last 30 frames)

### Zone Visualization

- **Blue polygons**: Intrusion zones
- **Green boxes**: Normal persons
- **Red boxes**: Intruders (in zone)
- **Zone labels**: Zone names

## Drone Tracking Algorithm

### Target Acquisition
1. Detect all persons in frame
2. Find closest person to drone center
3. Lock confidence increases with each frame
4. Full lock after 5 consecutive frames

### Tracking Phase
1. Compare current person positions with last tracked position
2. Find closest match (tracking continuity)
3. Update drone target if distance < 200px
4. Calculate movement vector to center target

### Target Loss Recovery
1. Lost frames counter increments when target not found
2. System searches for new target for 30 frames
3. If target not found, return to IDLE state
4. Reset lock confidence and tracking state

### Drone Movement

```
Distance to Target > Dead Zone?
    ├─ NO → Smooth deceleration to stop
    └─ YES → Accelerate toward target
         ├─ Max acceleration: 5 px/frame²
         ├─ Max velocity: 50 px/frame
         └─ Smooth velocity adjustment
```

## Fighting Detection Pipeline

### Stage 1: Pose Extraction
- YOLO pose model extracts 17 keypoints per person
- Keypoints normalized to frame dimensions
- Confidence filtering (threshold 0.25)

### Stage 2: Feature Engineering
- 68-dimensional pairwise feature vector
- Compares each pair of people (max 4 pairs)
- Features include:
  - Center distance (normalized)
  - Head distance
  - Arm raised indicators
  - Wrist distances
  - Elbow angles
  - Torso orientation

### Stage 3: LSTM Inference
- 24-frame temporal sequence buffer
- BiLSTM with multi-head attention
- Output: Fighting probability (0-1)

### Stage 4: Decision Logic
```python
fighting = (LSTM_confidence > 0.70) AND (proximity_detected)
```

## Performance Metrics

### Real-time Performance
- Detection: ~30 FPS (YOLO11n)
- Pose extraction: ~15 FPS
- Fighting LSTM: ~11-14 FPS
- Overall: ~11-14 FPS

### Accuracy (Validated)
- Non-fighting videos: 0% false positives ✓
- Single-person surveillance: 0% false positives ✓
- Crowd fighting: 60-62% detection rate ✓
- Peak LSTM confidence: 96-99% ✓

### Drone Tracking
- Acquisition time: 5 frames (~167ms)
- Tracking smoothness: 70% (SMOOTH_FACTOR)
- Max pursuit speed: 50 pixels/frame
- Reacquisition window: 30 frames (~1 second)

## Customization

### Adjust Drone Speed

```python
# In DroneConfig class
MAX_SPEED = 100           # Faster tracking
ACCELERATION = 10         # Snappier response
SMOOTH_FACTOR = 0.5       # Less smooth = faster

# Or dynamically
surveillance.drone_tracker.config.MAX_SPEED = 75
```

### Change Fighting Threshold

```python
surveillance.fighting_detector.lstm_threshold = 0.60  # More sensitive
surveillance.fighting_detector.lstm_threshold = 0.80  # More conservative
```

### Proximity Detection

```python
surveillance.fighting_detector.close_proximity_ratio = 0.10  # Stricter
surveillance.fighting_detector.close_proximity_ratio = 0.15  # Looser
```

### Frame Rate

```python
# Control processing frame skipping
for frame_idx, frame in enumerate(video):
    if frame_idx % 2 == 0:  # Process every 2nd frame
        surveillance.process_frame(frame)
```

## Troubleshooting

### Issue: Drone follows background objects
**Solution**: Adjust DEAD_ZONE smaller, increase SMOOTH_FACTOR

### Issue: Fighting not detected in actual fighting scene
**Solution**: Lower LSTM_FIGHTING_THRESHOLD (0.60-0.65)

### Issue: Too many false positives on random poses
**Solution**: Raise LSTM_FIGHTING_THRESHOLD (0.75-0.80)

### Issue: Drone tracking jittery/erratic
**Solution**: Increase SMOOTH_FACTOR (0.8-0.9)

### Issue: Models not loading
**Solution**: Check model paths:
```bash
ls runs/pose/human_pose_detector/weights/
ls fighting_temporal_model_v2/
```

## Integration with Real Drone APIs

### DJI SDK Integration Example

```python
from dji.sdk import Drone

class RealDroneSurveillance(AdvancedDroneSurveillance):
    def __init__(self, *args, drone_id='drone1', **kwargs):
        super().__init__(*args, **kwargs)
        self.drone = Drone(drone_id)
    
    def apply_drone_movement(self, movement):
        """Send calculated movement to real drone"""
        if movement is None:
            return
        
        # Normalize movement to drone commands
        x_cmd = int(movement[0] / 100)  # Scale down
        y_cmd = int(movement[1] / 100)
        
        self.drone.move_relative(x_cmd, y_cmd, 0, 0)
```

### DJI API Usage

```python
drone_system = RealDroneSurveillance(
    intrusion_zones=ZONES,
    video_source=0,  # Real-time camera
    drone_id='drone1'
)

drone_system.run_surveillance()
```

## Advanced Features

### Multi-Drone Coordination
```python
drones = [
    RealDroneSurveillance(ZONES, drone_id=f'drone{i}')
    for i in range(3)
]

for drone in drones:
    drone.run_surveillance()
```

### Cloud Integration
```python
# Send alerts to cloud
if surveillance.current_alert_level == "RED":
    send_to_cloud({
        'alert_level': 'RED',
        'location': surveillance.drone_tracker.current_position,
        'fighting_confidence': surveillance.current_fighting_confidence,
        'timestamp': datetime.now()
    })
```

### Database Logging
```python
import sqlite3

db = sqlite3.connect('surveillance.db')
cursor = db.cursor()

for alert in surveillance.fighting_alerts:
    cursor.execute('''
        INSERT INTO alerts VALUES (?, ?, ?, ?)
    ''', (datetime.now(), 'FIGHTING', 
          surveillance.drone_tracker.current_position.tolist(),
          surveillance.current_fighting_confidence))

db.commit()
```

## References

- YOLO11: https://docs.ultralytics.com/models/yolov11/
- LSTM Model: PyTorch BiLSTM with Multi-head Attention
- Pose Detection: Custom YOLO pose model
- RWF2000 Dataset: Real-world fighting videos

## Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all model files are in correct locations
3. Ensure video format is MP4 or AVI
4. Check GPU availability for faster processing
5. Review console output for specific error messages

---

**Last Updated**: February 2026
**Version**: 1.0
