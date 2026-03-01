#!/usr/bin/env python3
"""
Merge prowling and fighting_group into a single class
Creates a new dataset with combined aggressive activity class
"""

import os
import shutil
from pathlib import Path

def merge_classes():
    """Merge prowling and fighting_group into aggressive_activity"""
    
    dataset_dir = Path("activity_dataset")
    
    for split in ["train", "val"]:
        split_dir = dataset_dir / split
        
        # Create new combined folder
        combined_dir = split_dir / "aggressive_activity"
        combined_dir.mkdir(parents=True, exist_ok=True)
        
        # Move prowling images
        prowling_dir = split_dir / "prowling"
        if prowling_dir.exists():
            for img in prowling_dir.glob("*.jpg"):
                shutil.move(str(img), str(combined_dir / img.name))
            try:
                shutil.rmtree(str(prowling_dir))
            except:
                pass
            print(f"✓ Merged prowling → {split}/aggressive_activity")
        
        # Move fighting_group images
        fighting_dir = split_dir / "fighting_group"
        if fighting_dir.exists():
            for img in fighting_dir.glob("*.jpg"):
                shutil.move(str(img), str(combined_dir / img.name))
            try:
                shutil.rmtree(str(fighting_dir))
            except:
                pass
            print(f"✓ Merged fighting_group → {split}/aggressive_activity")
        
        # Count images
        count = len(list(combined_dir.glob("*.jpg")))
        print(f"  Total in aggressive_activity: {count} images\n")

if __name__ == "__main__":
    print("="*60)
    print("MERGING PROWLING + FIGHTING_GROUP")
    print("="*60)
    print()
    merge_classes()
    print("="*60)
    print("✅ Merge complete!")
    print("New class: aggressive_activity (combining prowling + fighting)")
    print("="*60)
