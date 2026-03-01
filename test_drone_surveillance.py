"""
DRONE SURVEILLANCE SYSTEM - QUICK START
Test the drone surveillance on sample videos
"""

import cv2
import numpy as np
from drone_surveillance import DroneSurveillanceSystem
import sys

# Define your protected zones here
# Each zone is a polygon defined by corner points
INTRUSION_ZONES = {
    'Security Zone': {
        'points': [[300, 200], [900, 200], [900, 600], [300, 600]],
        'alert_distance': 50
    }
}

def run_drone_surveillance(video_path, output_path, display=True):
    """
    Run drone surveillance on a video
    
    Args:
        video_path: Path to input video file
        output_path: Path to save output video
        display: Whether to display video while processing
    """
    print(f"\n{'='*60}")
    print("DRONE SURVEILLANCE SYSTEM")
    print(f"{'='*60}")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Display: {display}")
    print(f"{'='*60}\n")
    
    # Initialize surveillance system
    surveillance = DroneSurveillanceSystem(
        intrusion_zones=INTRUSION_ZONES,
        video_source=video_path
    )
    
    # Run surveillance
    surveillance.run_surveillance(
        output_path=output_path,
        display=display
    )
    
    # Print summary
    print(f"\n{'='*60}")
    print("SURVEILLANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Frames Processed: {surveillance.frame_count}")
    print(f"Total Intrusions Detected: {surveillance.total_intrusions}")
    print(f"Drone Distance Traveled: {surveillance.drone_tracker.total_distance_moved:.1f} pixels")
    print(f"Max Drone Speed: {surveillance.drone_tracker.max_speed_reached:.1f} px/frame")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Test videos
    test_cases = [
        {
            'name': 'Violence Video 1',
            'input': 'rwf2000_download/Violence/V_1.mp4',
            'output': 'drone_output_v1.mp4'
        },
        {
            'name': 'Violence Video 50',
            'input': 'rwf2000_download/Violence/V_50.mp4',
            'output': 'drone_output_v50.mp4'
        },
        {
            'name': 'Field Fighting',
            'input': 'rwf2000_download/Violence/1657Pri_OutFG_C1.mp4',
            'output': 'drone_output_field.mp4'
        }
    ]
    
    # Run all test cases
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'#'*60}")
        print(f"# Test Case {i}: {test_case['name']}")
        print(f"{'#'*60}")
        
        try:
            run_drone_surveillance(
                video_path=test_case['input'],
                output_path=test_case['output'],
                display=False  # Set to True to see real-time display
            )
            print(f"✓ Completed: {test_case['output']}\n")
        except Exception as e:
            print(f"✗ Error in test case {i}: {str(e)}\n")
            continue
    
    print("\nAll tests completed!")
