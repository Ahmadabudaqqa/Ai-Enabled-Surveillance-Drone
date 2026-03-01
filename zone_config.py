"""
Zone Configuration for Intrusion Detection
Football Field - Middle Area Monitoring
"""

# ============== ZONE DEFINITIONS ==============
# MMU FOOTBALL FIELD MONITORING
INTRUSION_ZONES = {
    "mmu_football_field": {
        "type": "rectangle",
        "name": "MMU Football Field - Center",
        "description": "Main playing area of MMU football field",
        # Ratios: Define as percentage of frame (0.0 to 1.0)
        # Based on drone aerial view to avoid track perimeter
        "x_min_ratio": 0.20,    # 20% from left (avoids left track)
        "y_min_ratio": 0.15,    # 15% from top
        "x_max_ratio": 0.80,    # 80% from right (avoids right track)
        "y_max_ratio": 0.85,    # 85% from bottom
        "enabled": True,
        "min_persons": 1  # Trigger if at least 1 person detected
    },
    # Optional: Add goal areas for more detailed monitoring
    "goal_area_left": {
        "type": "rectangle",
        "name": "Left Goal Area",
        "x_min_ratio": 0.05,
        "y_min_ratio": 0.35,
        "x_max_ratio": 0.25,
        "y_max_ratio": 0.65,
        "enabled": False,  # Disable if not needed
        "min_persons": 1
    },
    "goal_area_right": {
        "type": "rectangle",
        "name": "Right Goal Area",
        "x_min_ratio": 0.75,
        "y_min_ratio": 0.35,
        "x_max_ratio": 0.95,
        "y_max_ratio": 0.65,
        "enabled": False,  # Disable if not needed
        "min_persons": 1
    }
}

# ============== VISUALIZATION REFERENCE ==============
# Full frame: 160 x 120 pixels
#
#   0          40                  120         160
#   +----------+-------------------+----------+
# 0 |          |                   |          |
#   |          |  INTRUSION ZONE   |          |
# 25|          +-------------------+          |
#   |          |    (80 x 70)      |          |
# 95|          +-------------------+          |
#   |          |                   |          |
#120+----------+-------------------+----------+
#
# Middle area: 80 pixels wide x 70 pixels tall
# (Center 50% of frame width x 58% of frame height)

# ============== DETECTION PARAMETERS ==============
PERSON_CONFIDENCE_THRESHOLD = 0.4  # YOLO confidence for person detection
MIN_BOX_SIZE = 30  # Minimum bounding box size (pixels) to consider a person

# ============== ALERT SETTINGS ==============
INTRUSION_ALERT_DELAY = 0.5  # Seconds before alerting after detection
ALERT_COOLDOWN = 2.0  # Seconds between repeated alerts
LOG_DETECTIONS = True  # Log all zone entries to file
