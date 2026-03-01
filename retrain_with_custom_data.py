"""
Retrain Model with Custom Data
Combines your original dataset with custom data collected from reCamera
This fine-tunes the model for YOUR specific environment
"""
import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO

# Paths
ORIGINAL_DATASET = 'datasets/activity_balanced'  # Your existing balanced dataset
CUSTOM_DATA = 'custom_training_data'              # New data from reCamera
COMBINED_DATASET = 'datasets/activity_finetuned'  # Combined output

# Training settings
MODEL_PATH = 'runs/train/activity_cls_improved/weights/best.pt'  # Current best model
EPOCHS = 30  # Fewer epochs for fine-tuning
BATCH_SIZE = 32

def count_images(path):
    """Count images in a directory"""
    if not os.path.exists(path):
        return 0
    return len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

def create_combined_dataset():
    """Merge original dataset with custom data"""
    print("\n" + "=" * 60)
    print("📦 CREATING COMBINED DATASET")
    print("=" * 60)
    
    # Check if custom data exists
    if not os.path.exists(CUSTOM_DATA):
        print(f"❌ No custom data found at {CUSTOM_DATA}/")
        print("   Run collect_training_data.py first!")
        return False
    
    # Get activity classes from custom data
    custom_classes = [d for d in os.listdir(CUSTOM_DATA) 
                     if os.path.isdir(os.path.join(CUSTOM_DATA, d))]
    
    if not custom_classes:
        print("❌ No activity folders found in custom data!")
        return False
    
    # Check counts
    print("\n📊 Custom data collected:")
    total_custom = 0
    for cls in custom_classes:
        count = count_images(os.path.join(CUSTOM_DATA, cls))
        total_custom += count
        status = "✅" if count >= 30 else "⚠️ Low"
        print(f"   {cls:20s}: {count:4d} images {status}")
    
    if total_custom < 100:
        print(f"\n⚠️ Only {total_custom} custom images collected.")
        print("   Recommend at least 200+ for good fine-tuning.")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Create combined dataset structure
    print(f"\n📁 Creating combined dataset at {COMBINED_DATASET}/")
    
    for split in ['train', 'val']:
        for cls in custom_classes:
            os.makedirs(os.path.join(COMBINED_DATASET, split, cls), exist_ok=True)
    
    # Copy original data
    print("\n📥 Copying original dataset...")
    if os.path.exists(ORIGINAL_DATASET):
        for split in ['train', 'val']:
            split_path = os.path.join(ORIGINAL_DATASET, split)
            if os.path.exists(split_path):
                for cls in os.listdir(split_path):
                    src = os.path.join(split_path, cls)
                    dst = os.path.join(COMBINED_DATASET, split, cls)
                    if os.path.isdir(src):
                        os.makedirs(dst, exist_ok=True)
                        for img in os.listdir(src):
                            if img.lower().endswith(('.jpg', '.png', '.jpeg')):
                                shutil.copy2(os.path.join(src, img), 
                                           os.path.join(dst, img))
                        print(f"   Copied {cls}/{split}: {count_images(dst)} images")
    
    # Add custom data (80% train, 20% val)
    print("\n📥 Adding custom data...")
    for cls in custom_classes:
        src = os.path.join(CUSTOM_DATA, cls)
        images = [f for f in os.listdir(src) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        if not images:
            continue
        
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy to train
        train_dst = os.path.join(COMBINED_DATASET, 'train', cls)
        os.makedirs(train_dst, exist_ok=True)
        for img in train_images:
            shutil.copy2(os.path.join(src, img), 
                        os.path.join(train_dst, f"custom_{img}"))
        
        # Copy to val
        val_dst = os.path.join(COMBINED_DATASET, 'val', cls)
        os.makedirs(val_dst, exist_ok=True)
        for img in val_images:
            shutil.copy2(os.path.join(src, img),
                        os.path.join(val_dst, f"custom_{img}"))
        
        print(f"   Added {cls}: {len(train_images)} train, {len(val_images)} val")
    
    # Final counts
    print("\n📊 Combined dataset summary:")
    for cls in custom_classes:
        train_count = count_images(os.path.join(COMBINED_DATASET, 'train', cls))
        val_count = count_images(os.path.join(COMBINED_DATASET, 'val', cls))
        print(f"   {cls:20s}: {train_count:4d} train, {val_count:4d} val")
    
    return True

def fine_tune_model():
    """Fine-tune the existing model with combined data"""
    print("\n" + "=" * 60)
    print("🎯 FINE-TUNING MODEL")
    print("=" * 60)
    
    # Load existing model
    if os.path.exists(MODEL_PATH):
        print(f"📥 Loading existing model: {MODEL_PATH}")
        model = YOLO(MODEL_PATH)
    else:
        print(f"⚠️ Model not found at {MODEL_PATH}")
        print("   Using fresh yolov8s-cls instead")
        model = YOLO('yolov8s-cls.pt')
    
    # Train with combined dataset
    print(f"\n🚀 Starting fine-tuning for {EPOCHS} epochs...")
    print("   This will take ~30-60 minutes\n")
    
    results = model.train(
        data=COMBINED_DATASET,
        epochs=EPOCHS,
        imgsz=224,
        batch=BATCH_SIZE,
        patience=10,
        project='runs/train',
        name='activity_finetuned',
        exist_ok=True,
        
        # Fine-tuning specific settings
        lr0=0.001,      # Lower learning rate for fine-tuning
        lrf=0.01,
        warmup_epochs=2,
        
        # Augmentation
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=15,
        translate=0.1,
        scale=0.3,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.5,
        
        # Regularization
        dropout=0.15,
        
        verbose=True,
    )
    
    return results

def update_demo_script():
    """Update the demo script to use the fine-tuned model"""
    demo_path = 'realtime_7class_demo.py'
    new_model_path = 'runs/train/activity_finetuned/weights/best.pt'
    
    if os.path.exists(new_model_path):
        print(f"\n📝 New model saved at: {new_model_path}")
        print(f"   To use it, update ACTIVITY_MODEL_PATH in {demo_path}")
        print(f"   ACTIVITY_MODEL_PATH = '{new_model_path}'")

def main():
    print("\n" + "=" * 60)
    print("🔧 RETRAIN WITH CUSTOM DATA")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Combine your original dataset with custom reCamera data")
    print("  2. Fine-tune the model for your specific environment")
    print("  3. Save an improved model")
    
    # Step 1: Create combined dataset
    if not create_combined_dataset():
        print("\n❌ Dataset creation failed. Exiting.")
        return
    
    # Step 2: Fine-tune
    print("\n" + "-" * 60)
    response = input("Ready to start fine-tuning? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    results = fine_tune_model()
    
    # Step 3: Summary
    print("\n" + "=" * 60)
    print("✅ FINE-TUNING COMPLETE!")
    print("=" * 60)
    
    update_demo_script()
    
    print("\n💡 Next steps:")
    print("  1. Update ACTIVITY_MODEL_PATH in realtime_7class_demo.py")
    print("  2. Run: python realtime_7class_demo.py")
    print("  3. Test the improved model!")

if __name__ == '__main__':
    main()
