# 📁 Project Files Guide

## Overview

Your FYP detection system now includes comprehensive fighting detection with proper thresholds and visualization tools.

---

## Core System Files

### 🎯 **two_stage_surveillance.py** (Main Production File)
- **Purpose**: Your main surveillance system entry point
- **Status**: ⚠️ Needs update from test_smart_thresholds.py
- **What to do**: Copy the heuristic functions and 0.70 threshold to this file
- **LSTM threshold**: Currently ~0.60 (recommend changing to 0.70)

### 🔍 **test_full_2stage.py** (Working Production Test)
- **Purpose**: Fully functional 2-stage system with video output
- **Status**: ✅ Working, but has lower threshold
- **Features**: 
  - Stage 1: Intrusion detection (YOLO person)
  - Stage 2: Fighting detection (Pose + LSTM + heuristics)
  - Outputs annotated video
- **LSTM threshold**: 0.60 (good, but can raise to 0.70)

### 🧠 **test_smart_thresholds.py** (NEW - Optimized Version)
- **Purpose**: Improved 2-stage system with smart thresholds
- **Status**: ✅ NEW! Use this as reference
- **Key improvements**:
  - LSTM threshold: 0.70 (more strict)
  - Proper 68-dim feature extraction
  - Proximity + aggressive pose heuristics
  - Clear logging of detection reasons
- **Usage**: `python test_smart_thresholds.py <video> <output_video>`
- **Output**: Annotated video showing detections, LSTM%, sequence readiness

---

## Visualization Tools

### 🎬 **visualize_poses.py** (NEW - Pose Viewer)
- **Purpose**: Show all 17 COCO keypoints and skeleton on video
- **Status**: ✅ NEW! Ready to use
- **Visual indicators**:
  - 🟢 Green circles: Normal keypoints
  - 🔵 Blue lines: Skeleton connections
  - 🔴 Red circles: Aggressive pose (arms raised)
  - ⚪ White labels: Keypoint names
- **Usage**: `python visualize_poses.py <video> <output_video>`
- **Output**: Video with all pose annotations visible
- **Best for**: Understanding what the model sees, debugging pose accuracy

### 🎯 **output_smart_test.mp4** (Example Output)
- **Contents**: 103 frames with 2-stage detection results
- **What it shows**:
  - Intrusion zones highlighted
  - Person counts per frame
  - Fighting alerts with confidence scores
  - Sequence building status
  - Heuristic reasons displayed

### 🦴 **pose_visualization.mp4** (Example Output)
- **Contents**: 103 frames with pose visualization
- **What it shows**:
  - All 17 COCO keypoints per person
  - Body skeleton (green + blue)
  - Red keypoints when arms are raised
  - Frame numbers and legend

---

## Supporting Models & Resources

### 📦 **Model Files**
```
runs/pose/human_pose_detector/weights/best.pt
  - Custom YOLO pose detector
  - Extracts 17 COCO keypoints per person
  - Input: Detected person regions (from YOLO11n)
  - Output: (17, 2) keypoint coordinates

fighting_temporal_model_v2/fighting_lstm_v2.pt
  - Bidirectional LSTM with attention
  - Input: 24-frame sequence of 68-dim features
  - Output: Fighting probability (0.0-1.0)
  - Accuracy: 95.98% on training data
```

### 🎨 **YOLO Person Detection**
```
yolo11n.pt
  - Standard YOLO11 nano model
  - Used for Stage 1: Intrusion detection
  - Fast, accurate person bounding boxes
```

### 📄 **Configuration Files**
```
coco.names
  - Class names (includes "person" class)
  - Used for YOLO detection labels

intrusion_detector.py
  - Zone-based intrusion detection logic
  - Stage 1 of your 2-stage system
  - Defines protected zones/boundaries
```

---

## Detection Reference Implementations

### 🔬 **presentation_fighting_detection.py** (Reference - Strict)
- **Purpose**: Flask-based real-time fighting detection
- **Key settings**:
  - LSTM threshold: 0.90 (very strict, 90% confidence)
  - Proximity: 0.12 (12% of frame width)
  - Temporal sequence: 24 frames
  - Feature dimension: 68 (pairwise)
- **Status**: Reference implementation, shows best practices
- **Use case**: High-security areas (airports, banks)

### 🔬 **presentation_fighting_detection_noflask.py** (Reference - Sensitive)
- **Purpose**: Standalone version without Flask
- **Key settings**:
  - LSTM threshold: 0.20 (very permissive, 20% confidence)
  - Same heuristics (proximity, overlap)
  - Same 68-dim features
- **Status**: Reference, shows how heuristics help even with low threshold
- **Use case**: Shows that even 0.20 threshold works OK with proper filtering

### 🔬 **classify_fighting.py** (Training Reference)
- **Purpose**: Fighting classification model training
- **Status**: Reference only
- **Not used in**: Real-time detection

---

## Documentation (NEW!)

### 📖 **SYSTEM_SUMMARY.md** (START HERE)
- **Purpose**: Complete overview of the fixed system
- **Contents**:
  - Problem explained
  - Solutions implemented
  - Test results
  - 17 keypoints explained
  - Verification steps
  - Performance metrics
- **Best for**: Understanding the whole system

### 📖 **BEFORE_AFTER_COMPARISON.md** (Visual Comparison)
- **Purpose**: Side-by-side before/after analysis
- **Contents**:
  - Frame-by-frame comparison
  - Architecture diagrams
  - Performance charts
  - Feature extraction details
- **Best for**: Understanding what changed and why

### 📖 **FIGHTING_DETECTION_QUICKSTART.md** (Quick Reference)
- **Purpose**: Quick answers to common questions
- **Contents**:
  - Problem summary
  - Key understanding (17 keypoints)
  - Threshold levels explanation
  - Configuration guide
  - Next steps
- **Best for**: Quick lookup, answering "how does it work?"

### 📖 **FIGHTING_DETECTION_ANALYSIS.md** (Technical Deep Dive)
- **Purpose**: Detailed technical analysis
- **Contents**:
  - Feature extraction explained
  - Heuristic pre-filters
  - Code implementation reference
  - Recommendations by use case
  - Testing procedures
- **Best for**: Understanding implementation details

### 📖 **DEPLOYMENT_SUMMARY.md** (Original)
- **Purpose**: Deployment notes (from earlier session)
- **Status**: Still valid
- **Contents**: System architecture, hardware requirements

---

## Debug & Test Files

### 🧪 **test_debug_fighting.py**
- **Purpose**: Detailed frame-by-frame pose extraction debugging
- **Shows**: 
  - Pose shapes
  - Sequence building progress
  - Keypoint values per frame
  - Helps understand data flow

### 🧪 **full_test_log.json**
- **Contents**: Detailed test results from previous runs
- **Status**: Reference/archive

### 🧪 **intrusion_test_log.json**
- **Contents**: Intrusion detection test log
- **Status**: Reference/archive

---

## Input Videos (For Testing)

### ✅ **Available Test Videos**
```
test_fighting_detection.mp4
  - 103 frames of actual fighting
  - 1920×1080 @ 15.416 FPS
  - Real scenario with 2-6 people
  - Used for all recent tests

test_full_2stage_output.mp4
test_fighting_fg_field.mp4
test_fighting_fixed_output.mp4
test_fighting_final.mp4
  - Various test outputs from development
```

### ❌ **Not Available**
```
RWF2000/ (Violence dataset)
MMU field videos
  - Need to download separately
  - High-quality real-world data
```

---

## Generated Outputs (From Recent Tests)

### 📹 **New Output Videos**
```
pose_visualization.mp4 (7.1 MB)
  - Shows all 17 keypoints with skeleton
  - 103 frames of visual pose data
  - Green keypoints, red = aggressive, blue = skeleton

output_smart_test.mp4 (4.9 MB)
  - Full 2-stage detection results
  - Shows fighting alerts with reasons
  - 48 fighting detections out of 103 frames
```

---

## How to Use (Quick Start)

### Test the System
```bash
# 1. Visualize poses (understand keypoint accuracy)
python visualize_poses.py test_fighting_detection.mp4 debug_poses.mp4

# 2. Run smart detection (see improved results)
python test_smart_thresholds.py test_fighting_detection.mp4 debug_detection.mp4

# 3. Compare outputs
# - If pose_visualization shows ~17 good keypoints per person ✅
# - If output_smart_test shows fighting only during real fights ✅
# - Then system is working correctly!
```

### Integrate into Production
```
# In two_stage_surveillance.py:
1. Copy extract_temporal_features() from test_smart_thresholds.py
2. Copy check_aggressive_poses() from test_smart_thresholds.py
3. Copy distance_between_centroids() from test_smart_thresholds.py
4. Update LSTM_FIGHTING_THRESHOLD = 0.70 (from current value)
5. Update detect_fighting() to use heuristics:
   - Proximity check
   - Aggressive pose check
   - LSTM threshold check
6. Test on your videos
```

---

## File Organization

```
fyp detection/
├── Core System
│   ├── two_stage_surveillance.py      ⚠️ Update needed
│   ├── test_full_2stage.py            ✅ Works
│   ├── test_smart_thresholds.py       ✅ NEW - Use as reference
│   ├── intrusion_detector.py          ✅ Stage 1 detector
│   └── classify_fighting.py           📚 Reference
│
├── Visualization Tools (NEW!)
│   ├── visualize_poses.py             ✅ Use this!
│   ├── pose_visualization.mp4         📹 Example output
│   └── output_smart_test.mp4          📹 Example output
│
├── Documentation (NEW!)
│   ├── SYSTEM_SUMMARY.md              📖 Start here
│   ├── BEFORE_AFTER_COMPARISON.md     📖 Visual comparison
│   ├── FIGHTING_DETECTION_QUICKSTART.md  📖 Quick ref
│   ├── FIGHTING_DETECTION_ANALYSIS.md    📖 Technical
│   ├── DEPLOYMENT_SUMMARY.md          📖 Original notes
│   └── FILE_INDEX.md                  📖 Original file list
│
├── Debug/Test
│   ├── test_debug_fighting.py         🧪 Debug tool
│   ├── full_test_log.json             📊 Test results
│   └── intrusion_test_log.json        📊 Test results
│
├── Models
│   ├── runs/pose/.../best.pt          🧠 Pose model
│   ├── fighting_temporal_model_v2/    🧠 LSTM model
│   └── yolo11n.pt                     🧠 Person detector
│
└── Resources
    ├── coco.names                     📋 Class names
    ├── test_fighting_detection.mp4    🎬 Test video
    └── Other test outputs             📁 Archives
```

---

## Troubleshooting

### Issue: "Can't find video"
**Solution**: Use absolute path or check file exists
```bash
# Check if file exists
ls test_fighting_detection.mp4

# Use full path if needed
python visualize_poses.py "c:\Users\...\test_fighting_detection.mp4" output.mp4
```

### Issue: "Module not found"
**Solution**: Ensure venv is activated and dependencies installed
```bash
source .venv/Scripts/activate
pip install -r requirements.txt  # if it exists
```

### Issue: "CUDA out of memory"
**Solution**: System falls back to CPU (slower but works)
- Or reduce video resolution, process fewer frames

### Issue: "Unexpected emojis in output"
**Solution**: This is OK - it's just console logging
- View videos instead for proper annotations

---

## Next Steps

1. ✅ **Review** SYSTEM_SUMMARY.md
2. ✅ **Run** visualize_poses.py on your test video
3. ✅ **Compare** with pose_visualization.mp4
4. ✅ **Run** test_smart_thresholds.py on your test video
5. ✅ **Verify** fighting detected only during real fights
6. ⏭️ **Integrate** improvements into two_stage_surveillance.py
7. ⏭️ **Test** on field videos
8. ⏭️ **Deploy** to production

---

## Questions?

- **"How does it work?"** → Read SYSTEM_SUMMARY.md
- **"What changed?"** → Read BEFORE_AFTER_COMPARISON.md
- **"Quick answer?"** → Read FIGHTING_DETECTION_QUICKSTART.md
- **"Technical details?"** → Read FIGHTING_DETECTION_ANALYSIS.md
- **"See an example?"** → Run visualize_poses.py or view *.mp4 files

✨ **Everything is documented and ready to use!**
