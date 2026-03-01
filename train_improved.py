"""
Improved Training Script with Balanced Dataset + Heavy Augmentation
Target: 75%+ accuracy
"""
import os
import shutil
import random
from pathlib import Path
from ultralytics import YOLO
import yaml

# Configuration
DATASET_PATH = 'activity_dataset'
BALANCED_DATASET_PATH = 'activity_dataset_balanced'
MODEL_SIZE = 'yolov8s-cls.pt'  # Small model (better than nano)
SAMPLES_PER_CLASS = 1300  # Balance to smallest class

def count_images(folder):
    """Count images in a folder"""
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    return len([f for f in Path(folder).iterdir() if f.suffix.lower() in extensions])

def balance_dataset():
    """Create a balanced dataset with equal samples per class"""
    print("\n" + "="*60)
    print("STEP 1: BALANCING DATASET")
    print("="*60)
    
    src_train = Path(DATASET_PATH) / 'train'
    src_val = Path(DATASET_PATH) / 'val'
    
    dst_train = Path(BALANCED_DATASET_PATH) / 'train'
    dst_val = Path(BALANCED_DATASET_PATH) / 'val'
    
    # Clean old balanced dataset
    if Path(BALANCED_DATASET_PATH).exists():
        shutil.rmtree(BALANCED_DATASET_PATH)
    
    classes = [d.name for d in src_train.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes: {classes}")
    
    # Find minimum class size
    min_size = float('inf')
    class_sizes = {}
    for cls in classes:
        size = count_images(src_train / cls)
        class_sizes[cls] = size
        min_size = min(min_size, size)
    
    print(f"\nOriginal class sizes:")
    for cls, size in sorted(class_sizes.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls}: {size} images")
    
    target_samples = min(SAMPLES_PER_CLASS, min_size)
    print(f"\nBalancing to {target_samples} samples per class")
    
    # Balance training set
    for cls in classes:
        src_cls = src_train / cls
        dst_cls = dst_train / cls
        dst_cls.mkdir(parents=True, exist_ok=True)
        
        images = list(src_cls.glob('*'))
        images = [f for f in images if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}]
        
        # Random sample
        selected = random.sample(images, min(target_samples, len(images)))
        
        for img in selected:
            shutil.copy2(img, dst_cls / img.name)
        
        print(f"  {cls}: {len(selected)} images copied")
    
    # Copy validation set (use all)
    print("\nCopying validation set...")
    for cls in classes:
        src_cls = src_val / cls
        dst_cls = dst_val / cls
        
        if src_cls.exists():
            dst_cls.mkdir(parents=True, exist_ok=True)
            for img in src_cls.glob('*'):
                if img.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
                    shutil.copy2(img, dst_cls / img.name)
    
    # Create data.yaml
    data_yaml = {
        'path': str(Path(BALANCED_DATASET_PATH).absolute()),
        'train': 'train',
        'val': 'val',
        'names': {i: cls for i, cls in enumerate(sorted(classes))}
    }
    
    yaml_path = Path(BALANCED_DATASET_PATH) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✅ Balanced dataset created at: {BALANCED_DATASET_PATH}")
    print(f"   Total training: {target_samples * len(classes)} images")
    
    # Return the directory path (not YAML) for classification
    return str(Path(BALANCED_DATASET_PATH).absolute())

def train_with_augmentation():
    """Train with heavy augmentation for better generalization"""
    print("\n" + "="*60)
    print("STEP 2: TRAINING WITH HEAVY AUGMENTATION")
    print("="*60)
    
    # Balance dataset first
    data_yaml = balance_dataset()
    
    print(f"\nLoading model: {MODEL_SIZE}")
    model = YOLO(MODEL_SIZE)
    
    print("\n🚀 Starting training with heavy augmentation...")
    print("   This will take some time but should give better results!\n")
    
    results = model.train(
        data=data_yaml,
        epochs=100,
        patience=15,           # Early stopping patience
        imgsz=224,             # Standard size for classification
        batch=32,              # Adjust based on GPU memory
        device=0,              # GPU
        workers=4,
        project='runs/train',
        name='activity_cls_improved',
        exist_ok=True,
        
        # ===== HEAVY AUGMENTATION =====
        # Geometric transforms
        degrees=15.0,          # Rotation range
        translate=0.1,         # Translation
        scale=0.3,             # Scale variation (0.7x to 1.3x)
        shear=5.0,             # Shear angle
        perspective=0.0005,    # Perspective distortion
        flipud=0.1,            # Vertical flip (10% chance)
        fliplr=0.5,            # Horizontal flip (50% chance)
        
        # Color transforms
        hsv_h=0.02,            # Hue variation
        hsv_s=0.7,             # Saturation variation
        hsv_v=0.4,             # Brightness variation
        
        # Advanced augmentation
        mosaic=0.0,            # Disable mosaic for classification
        mixup=0.1,             # Mix two images (10% chance)
        copy_paste=0.0,        # Disable for classification
        erasing=0.2,           # Random erasing (20% chance)
        
        # Regularization
        dropout=0.2,           # Dropout for classification head
        
        # Optimizer settings
        lr0=0.001,             # Initial learning rate
        lrf=0.01,              # Final learning rate (lr0 * lrf)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        
        # Other
        verbose=True,
        plots=True,
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    
    # Get best model path
    best_model = Path('runs/train/activity_cls_improved/weights/best.pt')
    print(f"\n✅ Best model saved at: {best_model}")
    
    # Show final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
        print(f"\n📊 Final Metrics:")
        print(f"   Top-1 Accuracy: {metrics.get('metrics/accuracy_top1', 'N/A'):.2%}")
        print(f"   Top-5 Accuracy: {metrics.get('metrics/accuracy_top5', 'N/A'):.2%}")
    
    return best_model

def update_demo_script(model_path):
    """Update the demo script to use the new model"""
    demo_file = Path('realtime_7class_demo.py')
    
    if demo_file.exists():
        content = demo_file.read_text()
        
        # Find and replace model path
        import re
        new_content = re.sub(
            r"ACTIVITY_MODEL_PATH = '.*?'",
            f"ACTIVITY_MODEL_PATH = '{model_path}'",
            content
        )
        
        demo_file.write_text(new_content)
        print(f"\n✅ Updated realtime_7class_demo.py with new model path")

if __name__ == '__main__':
    print("="*60)
    print("IMPROVED ACTIVITY CLASSIFICATION TRAINING")
    print("="*60)
    print(f"\n📦 Model: {MODEL_SIZE} (better than nano)")
    print(f"📊 Target: Balanced dataset with {SAMPLES_PER_CLASS} samples/class")
    print(f"🎨 Heavy augmentation enabled")
    print(f"⏱️  Early stopping with patience=15")
    
    try:
        best_model = train_with_augmentation()
        update_demo_script(str(best_model))
        
        print("\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Check confusion matrix at: runs/train/activity_cls_improved/confusion_matrix.png")
        print("2. Run the demo: python realtime_7class_demo.py")
        print("3. If accuracy is still low, try:")
        print("   - More training data")
        print("   - Larger model (yolov8m-cls.pt)")
        print("   - Different augmentation settings")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise
