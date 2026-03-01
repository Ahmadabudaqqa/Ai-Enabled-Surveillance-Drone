#!/usr/bin/env python3
"""
DRONE SURVEILLANCE - QUICK REFERENCE & EXAMPLES
Copy-paste ready code for common scenarios
"""

# ============================================================================
# EXAMPLE 1: BASIC SETUP (Simplest)
# ============================================================================

from drone_surveillance_advanced import AdvancedDroneSurveillance

# Define your protected areas
ZONES = {
    'Entrance': {'points': [[100, 100], [500, 100], [500, 400], [100, 400]]},
}

# Create system
system = AdvancedDroneSurveillance(ZONES, 'video.mp4')

# Run
system.run_surveillance('output.mp4', display=True)


# ============================================================================
# EXAMPLE 2: MULTIPLE ZONES
# ============================================================================

ZONES = {
    'Main Gate': {
        'points': [[200, 150], [700, 150], [700, 550], [200, 550]],
        'alert_distance': 50
    },
    'Parking Lot': {
        'points': [[800, 200], [1400, 200], [1400, 600], [800, 600]],
        'alert_distance': 50
    },
    'Building': {
        'points': [[1500, 300], [1850, 300], [1850, 750], [1500, 750]],
        'alert_distance': 50
    }
}

system = AdvancedDroneSurveillance(ZONES, 'video.mp4')
system.run_surveillance('output.mp4')


# ============================================================================
# EXAMPLE 3: BATCH PROCESSING
# ============================================================================

from glob import glob
from pathlib import Path

ZONES = {
    'Protected Area': {'points': [[100, 100], [500, 100], [500, 400], [100, 400]]}
}

# Process all videos in a folder
for video in glob('videos/*.mp4'):
    print(f"Processing: {video}")
    
    system = AdvancedDroneSurveillance(ZONES, video)
    output = f"output_{Path(video).stem}.mp4"
    system.run_surveillance(output, display=False)
    
    print(f"  Intrusions: {system.total_intrusions}")
    print(f"  Alerts: {len(system.fighting_alerts)}\n")


# ============================================================================
# EXAMPLE 4: REAL-TIME CAMERA
# ============================================================================

ZONES = {
    'Security Zone': {'points': [[300, 200], [900, 200], [900, 600], [300, 600]]}
}

# Use webcam (0) instead of file
system = AdvancedDroneSurveillance(ZONES, video_source=0)
system.run_surveillance('live_surveillance.mp4', display=True)


# ============================================================================
# EXAMPLE 5: CUSTOM SENSITIVITY
# ============================================================================

ZONES = {...}

system = AdvancedDroneSurveillance(ZONES, 'video.mp4')

# Adjust fighting detection sensitivity
system.fighting_detector.lstm_threshold = 0.60  # More sensitive (default: 0.70)

# Adjust drone speed
system.drone_tracker.config.MAX_SPEED = 100     # Faster (default: 50)

# Adjust tracking smoothness
system.drone_tracker.config.SMOOTH_FACTOR = 0.9  # Smoother (default: 0.7)

system.run_surveillance('output.mp4')


# ============================================================================
# EXAMPLE 6: WITH STATISTICS LOGGING
# ============================================================================

import json
from datetime import datetime

ZONES = {...}
system = AdvancedDroneSurveillance(ZONES, 'video.mp4')
system.run_surveillance('output.mp4')

# Log results
results = {
    'timestamp': str(datetime.now()),
    'total_frames': system.frame_count,
    'intrusions': system.total_intrusions,
    'fighting_alerts': len(system.fighting_alerts),
    'drone_distance': system.drone_tracker.total_distance_moved,
    'max_speed': system.drone_tracker.max_speed_reached
}

with open('results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))


# ============================================================================
# EXAMPLE 7: DATABASE LOGGING
# ============================================================================

import sqlite3
from datetime import datetime

# Create database
db = sqlite3.connect('surveillance.db')
c = db.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS alerts
             (timestamp TEXT, type TEXT, confidence REAL, position TEXT)''')

ZONES = {...}
system = AdvancedDroneSurveillance(ZONES, 'video.mp4')
system.run_surveillance('output.mp4')

# Log intrusion alerts
for alert in system.fighting_alerts:
    c.execute('INSERT INTO alerts VALUES (?, ?, ?, ?)',
              (str(alert), 'FIGHTING', system.current_fighting_confidence,
               str(system.drone_tracker.current_position.tolist())))

db.commit()
db.close()


# ============================================================================
# EXAMPLE 8: FAST TEST (No Display)
# ============================================================================

ZONES = {...}

system = AdvancedDroneSurveillance(ZONES, 'video.mp4')
system.run_surveillance(output_path='output.mp4', display=False)

# Quick summary
print(f"Processed: {system.frame_count} frames")
print(f"Intrusions: {system.total_intrusions}")
print(f"Fighting: {len(system.fighting_alerts)}")


# ============================================================================
# EXAMPLE 9: ZONE CREATOR (INTERACTIVE)
# ============================================================================

from zone_creator import ZoneCreator

creator = ZoneCreator('video.mp4', 'my_zones.json')

# Create zones by clicking on video
creator.create_zone('Zone 1')
creator.create_zone('Zone 2')

# Save and visualize
creator.save_zones()
creator.visualize_zones()


# ============================================================================
# EXAMPLE 10: INTEGRATE WITH EXISTING SYSTEM
# ============================================================================

from drone_surveillance_advanced import AdvancedDroneSurveillance

class CustomSurveillance(AdvancedDroneSurveillance):
    """Extended class for custom integration"""
    
    def process_frame(self, frame):
        # Get default processing
        frame, stats = super().process_frame(frame)
        
        # Add custom logic
        if stats['fighting_detected']:
            self.send_alert_to_security_system(stats)
            self.log_to_database(stats)
        
        return frame, stats
    
    def send_alert_to_security_system(self, stats):
        """Send alert to your security API"""
        # Custom integration
        pass
    
    def log_to_database(self, stats):
        """Log to your database"""
        # Custom logging
        pass

# Use it
ZONES = {...}
surveillance = CustomSurveillance(ZONES, 'video.mp4')
surveillance.run_surveillance('output.mp4')


# ============================================================================
# PERFORMANCE TUNING GUIDE
# ============================================================================

"""
SLOW PERFORMANCE?
- Reduce video resolution before processing
- Skip frames: process every 2nd or 3rd frame
- Use smaller YOLO model (yolo11n is already smallest)
- Enable GPU: CUDA_VISIBLE_DEVICES=0

TOO MANY FALSE POSITIVES?
- Increase LSTM threshold: 0.70 → 0.80
- Increase proximity ratio: 0.12 → 0.15
- Require longer sequences before detection

TOO MANY FALSE NEGATIVES?
- Decrease LSTM threshold: 0.70 → 0.60
- Decrease proximity ratio: 0.12 → 0.10
- Reduce arm_raised_threshold: 0.3 → 0.2

DRONE TRACKING TOO JERKY?
- Increase SMOOTH_FACTOR: 0.7 → 0.9
- Decrease MAX_SPEED: 50 → 30
- Decrease ACCELERATION: 5 → 2

DRONE TRACKING TOO SLOW?
- Decrease SMOOTH_FACTOR: 0.7 → 0.5
- Increase MAX_SPEED: 50 → 100
- Increase ACCELERATION: 5 → 10
"""


# ============================================================================
# MONITORING & DEBUGGING
# ============================================================================

import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('surveillance')

ZONES = {...}
system = AdvancedDroneSurveillance(ZONES, 'video.mp4')

# Monitor during processing
class MonitoredSurveillance(AdvancedDroneSurveillance):
    def process_frame(self, frame):
        frame, stats = super().process_frame(frame)
        
        if self.frame_count % 30 == 0:
            logger.info(f"Frame {self.frame_count}: "
                       f"Persons={stats['persons_detected']}, "
                       f"Tracking={'ON' if stats['tracking_active'] else 'OFF'}, "
                       f"Fighting={stats['fighting_detected']}")
        
        return frame, stats

system = MonitoredSurveillance(ZONES, 'video.mp4')
system.run_surveillance('output.mp4')


# ============================================================================
# COMMON MISTAKES & FIXES
# ============================================================================

"""
MISTAKE 1: Points not in correct order
❌ {'points': [[0,0], [100,0], [100,100], [0,100]]}  # May be backwards
✓ {'points': [[0,0], [100,100], [100,0], [0,100]]}  # Clockwise/CCW

MISTAKE 2: Zone too large/small
❌ {'points': [[0,0], [1920,1080]]}  # Entire frame
✓ {'points': [[300,200], [900,600]]}  # Reasonable size

MISTAKE 3: Not saving output
❌ system.run_surveillance()  # Output not saved
✓ system.run_surveillance(output_path='output.mp4')

MISTAKE 4: Display without output
❌ system.run_surveillance(display=True)  # Slow + no file
✓ system.run_surveillance('output.mp4', display=False)  # Fast

MISTAKE 5: Wrong model paths
❌ pose_model_path='wrong_path.pt'  # Will fail
✓ pose_model_path='runs/pose/human_pose_detector/weights/best.pt'
"""


if __name__ == "__main__":
    print("Drone Surveillance - Quick Reference")
    print("See examples above for copy-paste ready code")
