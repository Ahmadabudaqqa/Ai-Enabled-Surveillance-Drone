"""
Download UCF Crime Dataset - Fighting Videos Only
This dataset contains real surveillance footage which matches your use case better than UFC.

The UCF Crime Dataset includes:
- Fighting (what we need!)
- Arrest, Arson, Assault, Burglary, etc.
- Normal activities for comparison

We'll download only the Fighting and Normal categories.
"""

import os
import urllib.request
import zipfile
from pathlib import Path

# Output directories
OUTPUT_DIR = Path('ucf_crime_dataset')
VIOLENCE_OUTPUT = Path('rwf2000_download/Violence')
NONVIOLENCE_OUTPUT = Path('rwf2000_download/NonViolence')

# UCF Crime Dataset URLs (Fighting subset from various mirrors)
# Note: Full dataset is 13GB, we'll use a subset

def download_file(url, output_path):
    """Download file with progress"""
    print(f"Downloading: {url}")
    print(f"To: {output_path}")
    
    try:
        urllib.request.urlretrieve(url, output_path)
        print(f"✅ Downloaded: {output_path}")
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False


def main():
    print("=" * 70)
    print("📥 UCF CRIME DATASET DOWNLOADER (Fighting Videos)")
    print("=" * 70)
    print()
    print("⚠️  The full UCF Crime Dataset is ~13GB.")
    print("    For your FYP, I recommend downloading fighting videos manually:")
    print()
    print("=" * 70)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 70)
    print()
    print("1. Go to: https://www.crcv.ucf.edu/projects/real-world/")
    print("   OR: https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa?dl=0")
    print()
    print("2. Download these folders:")
    print("   - Fighting (contains surveillance fighting videos)")
    print("   - Normal_Videos_event (normal activities)")
    print()
    print("3. Extract and copy videos:")
    print(f"   - Fighting videos → {VIOLENCE_OUTPUT}")
    print(f"   - Normal videos → {NONVIOLENCE_OUTPUT}")
    print()
    print("=" * 70)
    print("🎥 ALTERNATIVE: YouTube Surveillance Fight Videos")
    print("=" * 70)
    print()
    print("Search YouTube for:")
    print("  - 'CCTV fight footage'")
    print("  - 'surveillance camera fight'")
    print("  - 'security camera violence'")
    print()
    print("Use yt-dlp to download (install with: pip install yt-dlp)")
    print("  yt-dlp -o 'rwf2000_download/Violence/%(title)s.%(ext)s' <URL>")
    print()
    print("=" * 70)
    print("📊 YOUR CURRENT DATASET STATUS:")
    print("=" * 70)
    
    # Check current dataset
    if VIOLENCE_OUTPUT.exists():
        violence_count = len(list(VIOLENCE_OUTPUT.glob('*.mp4'))) + len(list(VIOLENCE_OUTPUT.glob('*.avi')))
        print(f"  Violence videos: {violence_count}")
    else:
        print("  Violence folder: Not found")
    
    if NONVIOLENCE_OUTPUT.exists():
        nonviolence_count = len(list(NONVIOLENCE_OUTPUT.glob('*.mp4'))) + len(list(NONVIOLENCE_OUTPUT.glob('*.avi')))
        print(f"  Non-violence videos: {nonviolence_count}")
    else:
        print("  Non-violence folder: Not found")
    
    print()
    print("=" * 70)
    print("💡 RECOMMENDATION:")
    print("=" * 70)
    print()
    print("Your current RWF-2000 dataset (2000 videos) is already good!")
    print("The retraining with your surveillance videos should help.")
    print()
    print("If you still need more data, add 20-50 videos that look similar")
    print("to your test footage (same camera angle, resolution, lighting).")
    print()


if __name__ == '__main__':
    main()
