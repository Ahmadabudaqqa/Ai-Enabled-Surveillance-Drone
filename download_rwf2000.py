"""
Download and process RWF-2000 Fighting Dataset
Extracts frames for fighting_group and walking (non-fighting) classes
"""
import os
import cv2
import zipfile
import shutil
from pathlib import Path

# Paths
DATASET_DIR = Path("activity_dataset/train")
FIGHTING_DIR = DATASET_DIR / "fighting_group"
WALKING_DIR = DATASET_DIR / "walking"

# RWF-2000 download info
RWF_INSTRUCTIONS = """
===============================================
RWF-2000 DATASET DOWNLOAD INSTRUCTIONS
===============================================

The RWF-2000 dataset needs to be downloaded manually from Google Drive.

STEP 1: Download from one of these links:
-----------------------------------------
Option A (Recommended): 
  https://drive.google.com/drive/folders/1F3JTY_NxKE2qv7vkqX6M0HFl-7VDWGEw

Option B (GitHub page):
  https://github.com/mcheng89/RWF-2000
  
STEP 2: Download these folders:
-------------------------------
- train/Fight/ (fighting videos)
- train/NonFight/ (normal walking videos)

STEP 3: Extract to this folder:
-------------------------------
  {download_folder}

STEP 4: Run this script again:
------------------------------
  python download_rwf2000.py --process

===============================================
"""

def extract_frames_from_video(video_path, output_dir, max_frames=30, skip_frames=3):
    """Extract frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    
    count = 0
    frame_idx = 0
    video_name = video_path.stem
    
    while count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % skip_frames == 0:
            # Resize to consistent size
            frame = cv2.resize(frame, (224, 224))
            output_path = output_dir / f"{video_name}_frame{count:03d}.jpg"
            cv2.imwrite(str(output_path), frame)
            count += 1
        
        frame_idx += 1
    
    cap.release()
    return count


def process_rwf_dataset(rwf_folder):
    """Process downloaded RWF-2000 videos into frames."""
    rwf_path = Path(rwf_folder)
    
    # Check for Fight and NonFight folders
    fight_folder = None
    nonfight_folder = None
    
    # Search for folders - handle both RWF-2000 and Kaggle naming
    for folder in rwf_path.rglob("*"):
        if folder.is_dir():
            name_lower = folder.name.lower()
            # Fighting/Violence folder
            if ("fight" in name_lower or "violence" in name_lower) and "non" not in name_lower:
                if any(f.suffix.lower() in ['.avi', '.mp4', '.mov'] for f in folder.iterdir() if f.is_file()):
                    fight_folder = folder
            # Non-fighting/NonViolence folder
            elif "nonfight" in name_lower or "non_fight" in name_lower or "nonviolence" in name_lower:
                if any(f.suffix.lower() in ['.avi', '.mp4', '.mov'] for f in folder.iterdir() if f.is_file()):
                    nonfight_folder = folder
    
    if not fight_folder:
        print("ERROR: Could not find Fight folder with videos!")
        print(f"Searched in: {rwf_path}")
        return False
    
    print(f"Found Fight folder: {fight_folder}")
    print(f"Found NonFight folder: {nonfight_folder}")
    
    # Create output directories
    FIGHTING_DIR.mkdir(parents=True, exist_ok=True)
    WALKING_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process Fight videos
    print("\n" + "="*50)
    print("Processing FIGHTING videos...")
    print("="*50)
    
    video_exts = ['.avi', '.mp4', '.mov', '.mkv']
    fight_videos = [f for f in fight_folder.iterdir() if f.suffix.lower() in video_exts]
    
    total_frames = 0
    for i, video in enumerate(fight_videos[:200]):  # Limit to 200 videos
        frames = extract_frames_from_video(video, FIGHTING_DIR, max_frames=20, skip_frames=2)
        total_frames += frames
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(fight_videos[:200])} videos, {total_frames} frames extracted")
    
    print(f"  DONE: Extracted {total_frames} fighting frames")
    
    # Process NonFight videos (for walking/normal class)
    if nonfight_folder:
        print("\n" + "="*50)
        print("Processing NON-FIGHTING (walking) videos...")
        print("="*50)
        
        nonfight_videos = [f for f in nonfight_folder.iterdir() if f.suffix.lower() in video_exts]
        
        walking_frames = 0
        for i, video in enumerate(nonfight_videos[:100]):  # Limit to 100 videos
            frames = extract_frames_from_video(video, WALKING_DIR, max_frames=15, skip_frames=3)
            walking_frames += frames
            if (i + 1) % 20 == 0:
                print(f"  Processed {i+1}/{len(nonfight_videos[:100])} videos, {walking_frames} frames extracted")
        
        print(f"  DONE: Extracted {walking_frames} walking frames")
    
    print("\n" + "="*50)
    print("DATASET PROCESSING COMPLETE!")
    print("="*50)
    print(f"Fighting frames: {len(list(FIGHTING_DIR.glob('*.jpg')))}")
    print(f"Walking frames: {len(list(WALKING_DIR.glob('*.jpg')))}")
    print("\nNext step: Retrain the model with:")
    print("  python train_7class.py")
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and process RWF-2000 dataset")
    parser.add_argument("--process", type=str, help="Path to downloaded RWF-2000 folder")
    parser.add_argument("--check", action="store_true", help="Check current dataset status")
    args = parser.parse_args()
    
    if args.check:
        print("\nCurrent dataset status:")
        print("-" * 40)
        for class_dir in DATASET_DIR.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.*")))
                status = "✓" if count > 0 else "✗ EMPTY"
                print(f"  {class_dir.name}: {count} images {status}")
        return
    
    if args.process:
        process_rwf_dataset(args.process)
    else:
        download_folder = Path("rwf2000_download")
        download_folder.mkdir(exist_ok=True)
        print(RWF_INSTRUCTIONS.format(download_folder=download_folder.absolute()))
        print("\nAfter downloading, run:")
        print(f'  python download_rwf2000.py --process "{download_folder}"')


if __name__ == "__main__":
    main()
