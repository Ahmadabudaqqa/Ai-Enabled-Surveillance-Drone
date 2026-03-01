# DRONE SURVEILLANCE - TROUBLESHOOTING & DIAGNOSTICS GUIDE

## Quick Diagnostics

### Test 1: Verify Installation
```bash
python -c "import torch; import cv2; import numpy as np; print('✓ All imports OK')"
```

### Test 2: Check YOLO Models
```bash
ls runs/detect/yolo11n/weights/
ls runs/pose/human_pose_detector/weights/
```

### Test 3: Check Fighting Model
```bash
ls models/fighting_lstm_final.pt
```

---

## Common Issues & Solutions

### ISSUE 1: "No module named 'ultralytics'"
**Symptom**: ImportError when running system

**Cause**: YOLOv8 not installed

**Solution**:
```bash
pip install ultralytics
```

**Verification**:
```bash
yolo version
```

---

### ISSUE 2: "YOLO Model Not Found"
**Symptom**: FileNotFoundError for model weights

**Code Fix**:
```python
from drone_surveillance_advanced import AdvancedDroneSurveillance

system = AdvancedDroneSurveillance(
    intrusion_zones=ZONES,
    video_source='video.mp4',
    # Specify paths if needed:
    yolo_model_path='runs/detect/yolo11n/weights/best.pt',
    pose_model_path='runs/pose/human_pose_detector/weights/best.pt'
)
```

**Debug**:
```bash
find . -name "*.pt" -type f  # Find all model files
```

---

### ISSUE 3: "CUDA out of memory"
**Symptom**: RuntimeError when running inference

**Cause**: GPU memory insufficient

**Solutions** (in order of speed impact):

1. **Reduce video resolution**:
```python
import cv2
cap = cv2.VideoCapture('video.mp4')
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)   # Instead of 1920
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)  # Instead of 1080
```

2. **Force CPU mode**:
```python
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Disable GPU
```

3. **Skip pose detection**:
```python
system = AdvancedDroneSurveillance(ZONES, 'video.mp4')
# Don't extract poses for every detection, only for tracked persons
```

4. **Process every Nth frame**:
```python
class SkipFramesSurveillance(AdvancedDroneSurveillance):
    def __init__(self, *args, skip_frames=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_frames = skip_frames
        self.frame_skip_count = 0
    
    def process_frame(self, frame):
        self.frame_skip_count += 1
        if self.frame_skip_count % self.skip_frames != 0:
            return frame, {'skipped': True}
        return super().process_frame(frame)
```

---

### ISSUE 4: "Video codec not supported"
**Symptom**: Cannot read or write video file

**Cause**: Missing codec or incompatible format

**Solution - Try different codec**:
```python
import cv2

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Instead of 'H264'
out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (1920, 1080))
```

**Solution - Use FFmpeg directly**:
```python
import subprocess
subprocess.run([
    'ffmpeg', '-i', 'input.mp4',
    '-vcodec', 'libx264', '-crf', '23',
    'output.mp4'
])
```

---

### ISSUE 5: "Drone not tracking (locked to one position)"
**Symptom**: Yellow crosshair doesn't follow detected persons

**Cause #1**: Zone not defined correctly
```python
# Check if zones are valid
import json

with open('zones.json', 'r') as f:
    zones = json.load(f)
    for name, zone in zones.items():
        print(f"{name}: {zone['points']}")
```

**Cause #2**: Person detection failing
```python
class DebugSurveillance(AdvancedDroneSurveillance):
    def detect_persons(self, frame):
        results = self.model.predict(frame, verbose=False)
        persons = results[0].boxes
        
        # Debug output
        if self.frame_count % 30 == 0:
            print(f"Frame {self.frame_count}: {len(persons)} persons detected")
        
        return persons
```

**Cause #3**: Intrusion check failing
```python
def check_intrusion(self, bbox, zone):
    # Debug point-in-polygon test
    x1, y1, x2, y2 = bbox
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    
    # Add debug visualization
    print(f"Person center: {center}")
    print(f"Zone: {zone['points']}")
    
    # Verify polygon intersection
    from shapely.geometry import Point, Polygon
    poly = Polygon(zone['points'])
    point = Point(center)
    return point.within(poly) or point.touches(poly)
```

---

### ISSUE 6: "Fighting detection always false"
**Symptom**: Never detects fighting even when obvious

**Solution 1**: Lower threshold
```python
system = AdvancedDroneSurveillance(ZONES, 'video.mp4')
system.fighting_detector.lstm_threshold = 0.60  # Default: 0.70
system.run_surveillance('output.mp4')
```

**Solution 2**: Check pose extraction
```python
class DebugFightingDetection(AdvancedDroneSurveillance):
    def detect_fighting(self, frame, persons):
        poses = self.fighting_detector.extract_poses(frame)
        
        if self.frame_count % 30 == 0:
            print(f"Extracted {len(poses)} poses from {len(persons)} persons")
            for i, pose in enumerate(poses):
                print(f"  Person {i}: {len(pose[0])} keypoints")
        
        return super().detect_fighting(frame, persons)
```

**Solution 3**: Verify training data compatibility
```python
# Check if model expects same pose format
import torch
model = torch.load('models/fighting_lstm_final.pt')
print(f"Model input size: {model.input_size}")
print(f"Expected features: 68")  # Should be 68-dim for temporal features
```

---

### ISSUE 7: "Very slow processing (< 5 FPS)"
**Symptom**: Video processing takes too long

**Diagnostics**:
```python
class BenchmarkSurveillance(AdvancedDroneSurveillance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stage_times = {'detect': [], 'track': [], 'pose': [], 'fight': []}
    
    def process_frame(self, frame):
        import time
        
        # Measure each stage
        t0 = time.time()
        persons = self.detect_persons(frame)
        self.stage_times['detect'].append(time.time() - t0)
        
        t0 = time.time()
        self.drone_tracker.update_target(persons)
        self.stage_times['track'].append(time.time() - t0)
        
        # ... continue for other stages
        
        frame, stats = super().process_frame(frame)
        
        # Print average times
        if self.frame_count % 30 == 0:
            for stage, times in self.stage_times.items():
                avg_time = sum(times[-30:]) / 30
                print(f"{stage}: {avg_time*1000:.1f}ms")
        
        return frame, stats
```

**Optimization** (choose one or more):
1. Reduce resolution ↓
2. Skip frames ↓
3. Use GPU ↑
4. Use smaller YOLO model ↑
5. Disable pose extraction when not tracking ↑

---

### ISSUE 8: "Output video corrupted"
**Symptom**: Can't play output.mp4

**Check file**:
```bash
ffprobe output.mp4
```

**Rebuild without display**:
```python
system = AdvancedDroneSurveillance(ZONES, 'video.mp4')
system.run_surveillance('output.mp4', display=False)  # Faster, more stable
```

**Use FFmpeg to repair**:
```bash
ffmpeg -i output.mp4 -c:v libx264 -crf 23 output_fixed.mp4
```

---

### ISSUE 9: "Zones not visible in output"
**Symptom**: Blue zone polygons don't appear

**Check zone points**:
```python
import json

with open('zones.json', 'r') as f:
    zones = json.load(f)

# Verify points are in valid range (0-width, 0-height)
for name, zone in zones.items():
    for point in zone['points']:
        if not (0 <= point[0] <= 1920 and 0 <= point[1] <= 1080):
            print(f"❌ {name}: Invalid point {point}")
        else:
            print(f"✓ {name}: Valid point {point}")
```

**Check visualization code**:
```python
class DebugVisualization(AdvancedDroneSurveillance):
    def draw_intrusion_zones(self, frame):
        import cv2
        for zone_name, zone_data in self.intrusion_zones.items():
            points = zone_data['points']
            # Convert to numpy array
            points = np.array(points, dtype=np.int32)
            print(f"Drawing {zone_name}: {points}")
            cv2.polylines(frame, [points], True, (255, 0, 0), 2)
        return frame
```

---

### ISSUE 10: "High false positive rate"
**Symptom**: Too many false fighting alerts

**Root causes and fixes**:

```python
# 1. Increase LSTM threshold (less sensitive)
system.fighting_detector.lstm_threshold = 0.85  # More conservative

# 2. Increase proximity requirement
system.fighting_detector.proximity_ratio = 0.15  # More distant = less likely fighting

# 3. Require longer sequences
system.fighting_detector.min_frames_in_buffer = 30  # Wait longer before deciding

# 4. Add cooldown between alerts
class CooldownSurveillance(AdvancedDroneSurveillance):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_alert_frame = -100
        self.alert_cooldown = 30  # frames
    
    def detect_fighting(self, frame, persons):
        fighting, conf = super().detect_fighting(frame, persons)
        
        if fighting and self.frame_count - self.last_alert_frame > self.alert_cooldown:
            self.last_alert_frame = self.frame_count
            return True, conf
        
        return False, conf
```

---

## Performance Optimization Checklist

- [ ] Using GPU (check `torch.cuda.is_available()`)
- [ ] Processing at 960x540 or lower
- [ ] Skipping every 2nd frame if needed
- [ ] Not displaying window during batch processing
- [ ] Using SSD drive (not network storage)
- [ ] YOLO model cached in memory
- [ ] Pose model only extracted when tracking
- [ ] LSTM model loaded once, not per frame

---

## Debug Mode Setup

```python
import logging
logging.basicConfig(level=logging.DEBUG)

class DebugSurveillance(AdvancedDroneSurveillance):
    def process_frame(self, frame):
        frame, stats = super().process_frame(frame)
        
        # Log every frame
        logging.debug(
            f"Frame {self.frame_count}: "
            f"Persons={stats['persons_detected']}, "
            f"Intrusion={stats['intrusion_detected']}, "
            f"Tracking={'ON' if stats['tracking_active'] else 'OFF'}, "
            f"Fighting={stats['fighting_detected']}"
        )
        
        return frame, stats

# Usage
logging.getLogger().setLevel(logging.DEBUG)
system = DebugSurveillance(ZONES, 'video.mp4')
system.run_surveillance('output.mp4')
```

---

## Model Retraining (If accuracy poor)

```python
# If fighting detection accuracy is poor, retrain:
# 1. Collect positive/negative video samples
# 2. Extract frames and label as fighting/not-fighting
# 3. Use your existing training pipeline
# 4. Export new model to models/fighting_lstm_final.pt
# 5. Test on validation set
# 6. Update paths in code if needed

from training_pipeline import FightingLSTMTrainer

trainer = FightingLSTMTrainer(
    train_dir='data/training',
    val_dir='data/validation'
)
model = trainer.train(epochs=50, batch_size=32)
model.save('models/fighting_lstm_final.pt')
```

---

## Contact & Support

If issues persist:
1. Check logs with debug mode enabled
2. Verify all models are present and correct format
3. Test with small sample video (10 seconds)
4. Check CUDA/GPU availability
5. Simplify system (test just detection, then tracking, etc.)
6. Review QUICK_REFERENCE.py for working examples

---

## Version Information

- **System**: Drone Surveillance v1.0
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **YOLOv11**: Latest
- **OpenCV**: 4.5+
- **Last Updated**: 2024-12-18
