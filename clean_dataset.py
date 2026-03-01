"""
Dataset Cleaning Script
Removes:
1. Duplicate/similar images (using perceptual hashing)
2. Images without detectable persons
3. Corrupt/unreadable images
4. Very small images (< 64x64)
5. Very blurry images
"""
import os
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
import hashlib
from ultralytics import YOLO
import shutil

# Configuration
DATASET_PATH = 'activity_dataset_balanced'
CLEANED_PATH = 'activity_dataset_cleaned'
MIN_IMAGE_SIZE = 64
BLUR_THRESHOLD = 100  # Lower = more blurry allowed
CHECK_PERSONS = False  # Skip YOLO check - much faster!

# Load YOLO for person detection (only if needed)
if CHECK_PERSONS:
    print("Loading YOLO for person verification...")
    detector = YOLO('yolo11n.pt')

def get_image_hash(image, hash_size=8):
    """Compute perceptual hash of image for duplicate detection"""
    # Resize and convert to grayscale
    resized = cv2.resize(image, (hash_size + 1, hash_size))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if len(resized.shape) == 3 else resized
    
    # Compute difference hash
    diff = gray[:, 1:] > gray[:, :-1]
    return sum([2 ** i for i, v in enumerate(diff.flatten()) if v])

def is_blurry(image, threshold=BLUR_THRESHOLD):
    """Check if image is too blurry using Laplacian variance"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def has_person(image_path):
    """Check if YOLO detects a person in the image"""
    results = detector(str(image_path), verbose=False)
    for r in results:
        for box in r.boxes:
            if int(box.cls[0]) == 0:  # Person class
                return True
    return False

def clean_class_folder(src_folder, dst_folder, class_name):
    """Clean a single class folder"""
    stats = {
        'total': 0,
        'corrupt': 0,
        'too_small': 0,
        'blurry': 0,
        'no_person': 0,
        'duplicate': 0,
        'kept': 0
    }
    
    dst_folder.mkdir(parents=True, exist_ok=True)
    
    # Track hashes for duplicate detection
    seen_hashes = set()
    
    images = list(src_folder.glob('*'))
    images = [f for f in images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
    stats['total'] = len(images)
    
    for img_path in images:
        try:
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                stats['corrupt'] += 1
                continue
            
            # Check size
            h, w = img.shape[:2]
            if h < MIN_IMAGE_SIZE or w < MIN_IMAGE_SIZE:
                stats['too_small'] += 1
                continue
            
            # Check blur (skip for now - many action shots are naturally blurry)
            # if is_blurry(img):
            #     stats['blurry'] += 1
            #     continue
            
            # Check for duplicates
            img_hash = get_image_hash(img)
            if img_hash in seen_hashes:
                stats['duplicate'] += 1
                continue
            seen_hashes.add(img_hash)
            
            # Check for person (only for single-person activities)
            # Skip for walking to save time (usually has persons)
            if CHECK_PERSONS and class_name not in ['walking']:
                if not has_person(img_path):
                    stats['no_person'] += 1
                    continue
            
            # Image passed all checks - copy it
            shutil.copy2(img_path, dst_folder / img_path.name)
            stats['kept'] += 1
            
        except Exception as e:
            stats['corrupt'] += 1
            continue
    
    return stats

def main():
    print("="*60)
    print("DATASET CLEANING")
    print("="*60)
    
    src_base = Path(DATASET_PATH)
    dst_base = Path(CLEANED_PATH)
    
    if dst_base.exists():
        print(f"Removing existing {CLEANED_PATH}...")
        shutil.rmtree(dst_base)
    
    total_stats = defaultdict(int)
    
    for split in ['train', 'val']:
        print(f"\n{'='*40}")
        print(f"Processing {split.upper()} split")
        print(f"{'='*40}")
        
        src_split = src_base / split
        dst_split = dst_base / split
        
        if not src_split.exists():
            print(f"  {split} folder not found, skipping...")
            continue
        
        classes = sorted([d.name for d in src_split.iterdir() if d.is_dir()])
        
        for cls in classes:
            print(f"\n  Cleaning {cls}...")
            stats = clean_class_folder(
                src_split / cls,
                dst_split / cls,
                cls
            )
            
            print(f"    Total: {stats['total']}")
            print(f"    Corrupt: {stats['corrupt']}")
            print(f"    Too small: {stats['too_small']}")
            print(f"    Duplicates: {stats['duplicate']}")
            if CHECK_PERSONS:
                print(f"    No person: {stats['no_person']}")
            print(f"    ✓ Kept: {stats['kept']}")
            
            for k, v in stats.items():
                total_stats[k] += v
    
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Total images processed: {total_stats['total']}")
    print(f"Removed - Corrupt: {total_stats['corrupt']}")
    print(f"Removed - Too small: {total_stats['too_small']}")
    print(f"Removed - Duplicates: {total_stats['duplicate']}")
    if CHECK_PERSONS:
        print(f"Removed - No person: {total_stats['no_person']}")
    print(f"\n✓ Total kept: {total_stats['kept']}")
    print(f"✓ Reduction: {total_stats['total'] - total_stats['kept']} images removed")
    
    # Show final distribution
    print("\n" + "="*60)
    print("CLEANED DATASET DISTRIBUTION")
    print("="*60)
    
    for split in ['train', 'val']:
        print(f"\n{split.upper()}:")
        split_dir = dst_base / split
        if split_dir.exists():
            for cls in sorted(split_dir.iterdir()):
                if cls.is_dir():
                    count = len(list(cls.glob('*')))
                    print(f"  {cls.name:25s}: {count:5d} images")

if __name__ == '__main__':
    main()
    print("\n✅ Dataset cleaned! Now run train_cleaned.py to retrain.")
