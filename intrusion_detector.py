"""
Intrusion Detection Module
Detects if persons enter defined zones
Used as Stage 1 before fighting detection (Stage 2)
"""

import cv2
import numpy as np
from zone_config import INTRUSION_ZONES, PERSON_CONFIDENCE_THRESHOLD, MIN_BOX_SIZE
from datetime import datetime
import json

class IntrusionDetector:
    """
    Detects when persons enter intrusion zones
    Works with YOLO detection results
    """
    
    def __init__(self, log_file="intrusion_log.json"):
        self.zones = INTRUSION_ZONES
        self.log_file = log_file
        self.intrusion_history = {}  # Track which zones have active intrusions
        self.last_alert_time = {}  # Cooldown tracking
        
        # Initialize intrusion history for each zone
        for zone_name in self.zones:
            self.intrusion_history[zone_name] = {
                "active": False,
                "persons_count": 0,
                "entry_time": None,
                "duration": 0
            }
    
    def check_point_in_rectangle(self, point, x1, y1, x2, y2):
        """
        Check if a point is inside a rectangle zone
        """
        px, py = point
        return x1 <= px <= x2 and y1 <= py <= y2
    
    def check_bbox_in_zone(self, bbox, zone_config):
        """
        Check if a bounding box overlaps with a zone
        bbox format: (x1, y1, x2, y2)
        
        Returns: percentage of bbox area inside zone (0-1)
        """
        bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
        zone_x1 = zone_config["x1"]
        zone_y1 = zone_config["y1"]
        zone_x2 = zone_config["x2"]
        zone_y2 = zone_config["y2"]
        
        # Calculate intersection
        intersect_x1 = max(bbox_x1, zone_x1)
        intersect_y1 = max(bbox_y1, zone_y1)
        intersect_x2 = min(bbox_x2, zone_x2)
        intersect_y2 = min(bbox_y2, zone_y2)
        
        # If no intersection
        if intersect_x1 >= intersect_x2 or intersect_y1 >= intersect_y2:
            return 0.0
        
        # Calculate overlap percentage
        intersection_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        bbox_area = (bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1)
        
        if bbox_area == 0:
            return 0.0
        
        return intersection_area / bbox_area
    
    def detect_intrusions(self, yolo_results, frame_width, frame_height):
        """
        Check YOLO detection results for intrusions
        
        Args:
            yolo_results: Results from YOLO detection
            frame_width: Frame width in pixels
            frame_height: Frame height in pixels
        
        Returns:
            intrusions: Dict with zone names and detected persons
            alerts: List of new intrusions to alert on
        """
        intrusions = {}
        alerts = []
        
        # Extract persons from YOLO results
        persons_in_frame = []
        
        if yolo_results and hasattr(yolo_results, 'boxes'):
            for box in yolo_results.boxes:
                conf = float(box.conf[0])
                
                # Filter by confidence and box size
                if conf < PERSON_CONFIDENCE_THRESHOLD:
                    continue
                
                # Get bounding box
                bbox = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = bbox
                
                # Check minimum size
                if (x2 - x1) < MIN_BOX_SIZE or (y2 - y1) < MIN_BOX_SIZE:
                    continue
                
                persons_in_frame.append({
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        # Check each zone
        for zone_name, zone_config in self.zones.items():
            if not zone_config.get("enabled", True):
                continue
            
            # Calculate actual pixel coordinates from ratios
            if "x_min_ratio" in zone_config:
                # Ratio-based zone
                x1 = int(zone_config["x_min_ratio"] * frame_width)
                y1 = int(zone_config["y_min_ratio"] * frame_height)
                x2 = int(zone_config["x_max_ratio"] * frame_width)
                y2 = int(zone_config["y_max_ratio"] * frame_height)
                zone_config_calc = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            else:
                # Fixed pixel coordinates
                zone_config_calc = zone_config
            
            # Find persons in this zone
            persons_in_zone = []
            for person in persons_in_frame:
                overlap = self.check_bbox_in_zone(person["bbox"], zone_config_calc)
                if overlap > 0.3:  # If more than 30% of person is in zone
                    persons_in_zone.append(person)
            
            intrusions[zone_name] = {
                "persons_count": len(persons_in_zone),
                "persons": persons_in_zone,
                "zone_name": zone_config["name"],
                "timestamp": datetime.now().isoformat()
            }
            
            # Update history
            was_active = self.intrusion_history[zone_name]["active"]
            is_active = len(persons_in_zone) >= zone_config.get("min_persons", 1)
            
            self.intrusion_history[zone_name]["active"] = is_active
            self.intrusion_history[zone_name]["persons_count"] = len(persons_in_zone)
            
            # New intrusion detected
            if is_active and not was_active:
                self.intrusion_history[zone_name]["entry_time"] = datetime.now()
                alerts.append({
                    "type": "INTRUSION",
                    "zone": zone_name,
                    "zone_name": zone_config["name"],
                    "persons_count": len(persons_in_zone),
                    "timestamp": datetime.now().isoformat(),
                    "action": "TRIGGER_FIGHTING_DETECTION"
                })
            
            # Intrusion ended
            elif not is_active and was_active:
                alerts.append({
                    "type": "INTRUSION_ENDED",
                    "zone": zone_name,
                    "zone_name": zone_config["name"],
                    "timestamp": datetime.now().isoformat(),
                    "action": "STOP_MONITORING"
                })
        
        return intrusions, alerts
    
    def draw_zones(self, frame):
        """
        Draw intrusion zones on frame
        Returns: Frame with zones drawn
        """
        frame_copy = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        for zone_name, zone_config in self.zones.items():
            if not zone_config.get("enabled", True):
                continue
            
            # Calculate coordinates from ratios if needed
            if "x_min_ratio" in zone_config:
                x1 = int(zone_config["x_min_ratio"] * frame_width)
                y1 = int(zone_config["y_min_ratio"] * frame_height)
                x2 = int(zone_config["x_max_ratio"] * frame_width)
                y2 = int(zone_config["y_max_ratio"] * frame_height)
            else:
                x1, y1 = zone_config["x1"], zone_config["y1"]
                x2, y2 = zone_config["x2"], zone_config["y2"]
            
            # Color based on intrusion status
            is_intruding = self.intrusion_history[zone_name]["active"]
            color = (0, 0, 255) if is_intruding else (0, 255, 0)  # Red if intrusion, Green otherwise
            thickness = 3 if is_intruding else 2
            
            # Draw rectangle
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw zone name
            label = zone_config["name"]
            if is_intruding:
                label += f" - {self.intrusion_history[zone_name]['persons_count']} persons"
            
            cv2.putText(frame_copy, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 2)
        
        return frame_copy
    
    def log_alert(self, alert):
        """Log detection alert to file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(alert) + "\n")
        except Exception as e:
            print(f"Error logging alert: {e}")
    
    def get_intrusion_status(self):
        """Get current intrusion status for all zones"""
        return self.intrusion_history
