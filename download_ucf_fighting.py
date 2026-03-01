"""
Download UCF Crime Dataset - Fighting Videos
Downloads fighting and normal activity videos for training.
"""

import os
import gdown
from pathlib import Path
import zipfile
import shutil

# Output directories
OUTPUT_DIR = Path('ucf_crime_dataset')
VIOLENCE_OUTPUT = Path('rwf2000_download/Violence')
NONVIOLENCE_OUTPUT = Path('rwf2000_download/NonViolence')

# UCF Crime Dataset - Fighting videos from Google Drive
# These are shared links to UCF Crime fighting subset
UCF_FIGHTING_URLS = [
    # Fighting videos from UCF Anomaly Detection Dataset
    # Source: https://www.crcv.ucf.edu/projects/real-world/
    {
        'name': 'UCF Crime Fighting Subset',
        'url': 'https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABGm3O0NcAT3VQwlzAl9iRua/Fighting?dl=1',
        'type': 'fighting'
    }
]

def download_from_dropbox():
    """Download fighting videos from Dropbox"""
    import urllib.request
    
    OUTPUT_DIR.mkdir(exist_ok=True)
    fighting_dir = OUTPUT_DIR / 'Fighting'
    fighting_dir.mkdir(exist_ok=True)
    
    print("=" * 70)
    print("📥 DOWNLOADING UCF CRIME FIGHTING VIDEOS")
    print("=" * 70)
    print()
    print("⚠️  Note: Direct download from Dropbox requires manual steps.")
    print()
    print("=" * 70)
    print("📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("=" * 70)
    print()
    print("1. Open this link in your browser:")
    print("   https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa")
    print()
    print("2. Navigate to 'Fighting' folder")
    print()
    print("3. Click 'Download' → 'Download folder as ZIP'")
    print()
    print("4. Extract the ZIP to: ucf_crime_dataset/Fighting/")
    print()
    print("5. Run this script again to copy files to training folders")
    print()

def copy_to_training():
    """Copy downloaded UCF videos to training folders"""
    fighting_dir = OUTPUT_DIR / 'Fighting'
    normal_dir = OUTPUT_DIR / 'Normal_Videos_event'
    
    copied_fight = 0
    copied_normal = 0
    
    # Copy fighting videos
    if fighting_dir.exists():
        print(f"\n📁 Found UCF Fighting folder: {fighting_dir}")
        for video in fighting_dir.glob('*.mp4'):
            dest = VIOLENCE_OUTPUT / f"ucf_{video.name}"
            if not dest.exists():
                shutil.copy2(video, dest)
                copied_fight += 1
                print(f"  ✅ Copied: {video.name}")
        
        for video in fighting_dir.glob('*.avi'):
            dest = VIOLENCE_OUTPUT / f"ucf_{video.name}"
            if not dest.exists():
                shutil.copy2(video, dest)
                copied_fight += 1
                print(f"  ✅ Copied: {video.name}")
    else:
        print(f"⚠️  Fighting folder not found: {fighting_dir}")
        print("   Please download the Fighting videos first.")
    
    # Copy normal videos if available
    if normal_dir.exists():
        print(f"\n📁 Found UCF Normal folder: {normal_dir}")
        for video in list(normal_dir.glob('*.mp4'))[:50]:  # Limit to 50
            dest = NONVIOLENCE_OUTPUT / f"ucf_{video.name}"
            if not dest.exists():
                shutil.copy2(video, dest)
                copied_normal += 1
                print(f"  ✅ Copied: {video.name}")
    
    return copied_fight, copied_normal

def main():
    print("=" * 70)
    print("📥 UCF CRIME DATASET - FIGHTING VIDEOS DOWNLOADER")
    print("=" * 70)
    
    # Create directories
    OUTPUT_DIR.mkdir(exist_ok=True)
    VIOLENCE_OUTPUT.mkdir(parents=True, exist_ok=True)
    NONVIOLENCE_OUTPUT.mkdir(parents=True, exist_ok=True)
    
    # Check if UCF videos already downloaded
    fighting_dir = OUTPUT_DIR / 'Fighting'
    
    if fighting_dir.exists() and len(list(fighting_dir.glob('*'))) > 0:
        print(f"\n✅ UCF Fighting videos found in: {fighting_dir}")
        print(f"   Videos: {len(list(fighting_dir.glob('*.mp4'))) + len(list(fighting_dir.glob('*.avi')))}")
        
        # Copy to training folders
        copied_fight, copied_normal = copy_to_training()
        
        print("\n" + "=" * 70)
        print("📊 COPY SUMMARY:")
        print("=" * 70)
        print(f"  Fighting videos copied: {copied_fight}")
        print(f"  Normal videos copied: {copied_normal}")
        
        # Show current dataset status
        violence_count = len(list(VIOLENCE_OUTPUT.glob('*.mp4'))) + len(list(VIOLENCE_OUTPUT.glob('*.avi')))
        nonviolence_count = len(list(NONVIOLENCE_OUTPUT.glob('*.mp4'))) + len(list(NONVIOLENCE_OUTPUT.glob('*.avi')))
        
        print("\n" + "=" * 70)
        print("📊 UPDATED DATASET STATUS:")
        print("=" * 70)
        print(f"  Violence videos: {violence_count}")
        print(f"  Non-violence videos: {nonviolence_count}")
        print(f"  Total: {violence_count + nonviolence_count}")
        
        print("\n✅ Done! Now retrain with: python train_temporal_fighting.py")
        
    else:
        # Show download instructions
        download_from_dropbox()
        
        print("=" * 70)
        print("🔗 ALTERNATIVE: Direct links to download")
        print("=" * 70)
        print()
        print("UCF Crime Dataset Fighting videos:")
        print("  https://www.crcv.ucf.edu/data/UCF_Crimes.zip (13GB - Full dataset)")
        print()
        print("Or download individual categories from:")
        print("  https://www.dropbox.com/sh/75v5ehq4cdg5g5g/AABvnJSwZI7zXb8_myBA0CLHa")
        print()
        print("After downloading, extract Fighting folder to:")
        print(f"  {OUTPUT_DIR.absolute()}/Fighting/")
        print()
        print("Then run this script again to copy to training folders.")

if __name__ == '__main__':
    main()
