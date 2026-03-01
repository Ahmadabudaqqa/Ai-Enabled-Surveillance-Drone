"""
ZONE CREATOR AND VISUALIZER
Interactive tool to define and visualize intrusion zones
"""

import cv2
import numpy as np
import json
from pathlib import Path


class ZoneCreator:
    """Interactive zone creation tool"""
    
    def __init__(self, video_source, output_file='zones.json'):
        self.video_source = video_source
        self.output_file = output_file
        self.zones = {}
        self.current_zone_name = None
        self.current_points = []
        self.frame = None
        self.paused = False
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for point selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_points.append([x, y])
            print(f"Point {len(self.current_points)}: ({x}, {y})")
            
            # Draw on frame
            cv2.circle(self.frame, (x, y), 5, (0, 255, 0), -1)
            if len(self.current_points) > 1:
                cv2.line(self.frame, tuple(self.current_points[-2]), 
                        (x, y), (0, 255, 0), 2)
            
            cv2.imshow('Zone Creator', self.frame)
    
    def create_zone(self, zone_name):
        """Create a new zone by clicking points"""
        self.current_zone_name = zone_name
        self.current_points = []
        
        cap = cv2.VideoCapture(self.video_source)
        ret, self.frame = cap.read()
        cap.release()
        
        if not ret:
            print("Error: Could not read video frame")
            return
        
        cv2.namedWindow('Zone Creator')
        cv2.setMouseCallback('Zone Creator', self.mouse_callback)
        cv2.imshow('Zone Creator', self.frame)
        
        print(f"\nCreating zone: {zone_name}")
        print("Instructions:")
        print("  - Click points to define zone polygon (at least 4 points)")
        print("  - Press 'c' to confirm zone")
        print("  - Press 'r' to reset points")
        print("  - Press 'q' to quit")
        
        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('c'):
                if len(self.current_points) >= 4:
                    # Close polygon
                    self.zones[zone_name] = {
                        'points': self.current_points.copy(),
                        'alert_distance': 50
                    }
                    print(f"✓ Zone '{zone_name}' created with {len(self.current_points)} points")
                    break
                else:
                    print("✗ Need at least 4 points to create a zone")
            
            elif key == ord('r'):
                cap = cv2.VideoCapture(self.video_source)
                ret, self.frame = cap.read()
                cap.release()
                self.current_points = []
                cv2.imshow('Zone Creator', self.frame)
                print("Points reset")
            
            elif key == ord('q'):
                print("Cancelled")
                break
        
        cv2.destroyAllWindows()
    
    def save_zones(self):
        """Save zones to JSON file"""
        with open(self.output_file, 'w') as f:
            json.dump(self.zones, f, indent=2)
        print(f"✓ Zones saved to {self.output_file}")
    
    def load_zones(self):
        """Load zones from JSON file"""
        if Path(self.output_file).exists():
            with open(self.output_file, 'r') as f:
                self.zones = json.load(f)
            print(f"✓ Loaded {len(self.zones)} zones from {self.output_file}")
        else:
            print(f"✗ Zones file not found: {self.output_file}")
    
    def visualize_zones(self):
        """Visualize zones on video"""
        cap = cv2.VideoCapture(self.video_source)
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"\nVisualizing zones on: {self.video_source}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print("Controls: SPACE=pause, 'q'=quit")
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Draw zones
            for zone_name, zone_config in self.zones.items():
                points = np.array(zone_config['points'], dtype=np.int32)
                cv2.polylines(frame, [points], True, (0, 0, 255), 2)
                
                # Label
                center_x = int(np.mean(points[:, 0]))
                center_y = int(np.mean(points[:, 1]))
                cv2.putText(frame, zone_name, (center_x - 50, center_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Points
                for point in points:
                    cv2.circle(frame, tuple(point), 3, (0, 255, 0), -1)
            
            # Info
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Zones: {len(self.zones)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Zone Visualization', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)
        
        cap.release()
        cv2.destroyAllWindows()


def quick_create_zones():
    """Quick zone creation interface"""
    print("\n" + "="*60)
    print("DRONE SURVEILLANCE - ZONE CREATOR")
    print("="*60)
    
    video_source = input("\nEnter video path: ").strip()
    output_file = input("Enter output filename (default: zones.json): ").strip() or 'zones.json'
    
    creator = ZoneCreator(video_source, output_file)
    
    while True:
        print("\nOptions:")
        print("1. Create new zone")
        print("2. Visualize zones")
        print("3. Save zones")
        print("4. Load zones")
        print("5. List zones")
        print("6. Remove zone")
        print("7. Export to Python")
        print("8. Exit")
        
        choice = input("\nSelect option: ").strip()
        
        if choice == '1':
            zone_name = input("Enter zone name: ").strip()
            creator.create_zone(zone_name)
        
        elif choice == '2':
            if creator.zones:
                creator.visualize_zones()
            else:
                print("No zones created yet")
        
        elif choice == '3':
            creator.save_zones()
        
        elif choice == '4':
            creator.load_zones()
        
        elif choice == '5':
            if creator.zones:
                print("\nZones:")
                for name, config in creator.zones.items():
                    print(f"  - {name}: {len(config['points'])} points")
            else:
                print("No zones created")
        
        elif choice == '6':
            if creator.zones:
                name = input("Enter zone name to remove: ").strip()
                if name in creator.zones:
                    del creator.zones[name]
                    print(f"✓ Zone '{name}' removed")
                else:
                    print("✗ Zone not found")
            else:
                print("No zones to remove")
        
        elif choice == '7':
            if creator.zones:
                print("\nPython dict:")
                print("INTRUSION_ZONES = {")
                for name, config in creator.zones.items():
                    print(f"    '{name}': {{")
                    print(f"        'points': {config['points']},")
                    print(f"        'alert_distance': {config['alert_distance']}")
                    print(f"    }},")
                print("}")
            else:
                print("No zones to export")
        
        elif choice == '8':
            print("Exiting...")
            break
        
        else:
            print("Invalid option")


def create_default_zones(video_path):
    """Create default zones for quick testing"""
    return {
        'Zone 1': {
            'points': [[200, 150], [700, 150], [700, 500], [200, 500]],
            'alert_distance': 50
        },
        'Zone 2': {
            'points': [[900, 200], [1500, 200], [1500, 700], [900, 700]],
            'alert_distance': 50
        },
        'Zone 3': {
            'points': [[1600, 300], [1900, 300], [1900, 800], [1600, 800]],
            'alert_distance': 50
        }
    }


if __name__ == "__main__":
    quick_create_zones()
