"""
MMU FIELD INTRUSION DETECTION - QUICK START
Ready-to-deploy system for drone/camera monitoring
"""

import sys
import os

print("\n" + "="*70)
print("🏆 MMU FOOTBALL FIELD - INTRUSION DETECTION SYSTEM")
print("="*70)
print("\nWelcome! This system monitors your field for unauthorized entry.\n")

# Check system status
print("[✓] Checking system files...")
required_files = [
    "zone_config.py",
    "intrusion_detector.py",
    "test_intrusion_only.py",
    "yolo11n.pt"
]

missing = []
for f in required_files:
    if not os.path.exists(f):
        missing.append(f)

if missing:
    print(f"[✗] Missing files: {', '.join(missing)}")
    sys.exit(1)
else:
    print("[✓] All required files found!")

print("\n" + "="*70)
print("DEPLOYMENT OPTIONS")
print("="*70)

print("""
1. LIVE MONITORING (Drone/Camera Feed)
   python two_stage_surveillance.py
   → Opens web interface at http://localhost:5000
   → Real-time monitoring with zone overlay
   → Alerts appear on screen and in logs

2. VIDEO TESTING (Test on recorded footage first)
   python test_intrusion_only.py <video_file> <output_file>
   
   Examples:
   python test_intrusion_only.py field_drone.mp4 field_results.mp4
   python test_intrusion_only.py surveillanceVideos/0902Pri_OutPW_C1.mp4 test.mp4

3. CONFIGURATION
   - Edit zone_config.py to adjust monitoring area
   - Current zone: 25% to 75% horizontally (avoids track)

""")

print("="*70)
print("MONITORING ZONE")
print("="*70)
print("""
Your MMU field zone:
  ■ Green grass area = MONITORED (intrusion zone)
  ■ Red track = IGNORED
  ■ Buildings/Sidelines = IGNORED
  
Coordinate ranges (ratios):
  X: 0.25 to 0.75 (left to right)
  Y: 0.20 to 0.80 (top to bottom)
""")

print("="*70)
print("READY TO DEPLOY?")
print("="*70)
print("\n1. Prepare your drone/camera feed")
print("2. Choose deployment option above")
print("3. Run the command")
print("4. Monitor alerts in real-time")
print("\n" + "="*70)
print("\nFor detailed guide, see: MMU_FIELD_DEPLOYMENT.md")
print("="*70 + "\n")

# Suggest next step
print("QUICK START EXAMPLE:\n")
print("To test with a video file first:")
print('  python test_intrusion_only.py surveil lanceVideos/0902Pri_OutPW_C1.mp4 test_output.mp4\n')
print("To deploy live with camera:")
print("  python two_stage_surveillance.py\n")
