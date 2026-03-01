#!/usr/bin/env python3
"""
Balance the activity dataset:
1. Remove empty classes (fighting_group, prowling)
2. Reduce aggressive_activity to match other classes (~1350 train, ~1000 val)
3. Keep 7 balanced classes
"""

import os
import shutil
import random

random.seed(42)

DATASET_DIR = "activity_dataset"
BACKUP_DIR = "activity_dataset_backup"

# Target counts (approximate average of other classes)
TARGET_TRAIN = 1353
TARGET_VAL = 1000

# Classes to remove (empty)
EMPTY_CLASSES = ["fighting_group", "prowling"]

# Class to downsample
OVERSIZED_CLASS = "aggressive_activity"


def count_images(folder):
    if not os.path.exists(folder):
        return 0
    return len([f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))])


def downsample_folder(folder, target_count):
    """Randomly remove images to reach target count"""
    images = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    current_count = len(images)
    
    if current_count <= target_count:
        print(f"  Already at or below target ({current_count} <= {target_count})")
        return 0
    
    # Randomly select images to remove
    to_remove = random.sample(images, current_count - target_count)
    
    for img in to_remove:
        os.remove(os.path.join(folder, img))
    
    print(f"  Removed {len(to_remove)} images ({current_count} -> {target_count})")
    return len(to_remove)


def main():
    # Skip backup to save time (dataset is large)
    print("Skipping full backup (too slow). Proceeding with balancing...\n")
    
    # Remove empty classes
    print("=" * 50)
    print("STEP 1: Removing empty classes")
    print("=" * 50)
    for cls in EMPTY_CLASSES:
        for split in ["train", "val"]:
            folder = os.path.join(DATASET_DIR, split, cls)
            if os.path.exists(folder):
                count = count_images(folder)
                try:
                    shutil.rmtree(folder)
                    print(f"  Removed {split}/{cls} ({count} images)")
                except PermissionError:
                    print(f"  Skipped {split}/{cls} (permission denied, but has {count} images)")
    
    # Downsample oversized class
    print("\n" + "=" * 50)
    print(f"STEP 2: Downsampling {OVERSIZED_CLASS}")
    print("=" * 50)
    
    train_folder = os.path.join(DATASET_DIR, "train", OVERSIZED_CLASS)
    val_folder = os.path.join(DATASET_DIR, "val", OVERSIZED_CLASS)
    
    print(f"\nTrain folder:")
    downsample_folder(train_folder, TARGET_TRAIN)
    
    print(f"\nVal folder:")
    downsample_folder(val_folder, TARGET_VAL)
    
    # Show final distribution
    print("\n" + "=" * 50)
    print("FINAL DATASET DISTRIBUTION")
    print("=" * 50)
    
    for split in ["train", "val"]:
        print(f"\n{split.upper()}:")
        split_dir = os.path.join(DATASET_DIR, split)
        classes = sorted(os.listdir(split_dir))
        total = 0
        for cls in classes:
            count = count_images(os.path.join(split_dir, cls))
            total += count
            print(f"  {cls:25s}: {count:5d} images")
        print(f"  {'TOTAL':25s}: {total:5d} images")
    
    print("\n✅ Dataset balanced successfully!")
    print(f"Backup saved at: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
