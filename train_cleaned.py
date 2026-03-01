"""
Train on Cleaned Dataset
Run after clean_dataset.py
"""
from ultralytics import YOLO
from pathlib import Path

# Configuration  
DATASET_PATH = 'activity_dataset_cleaned'
MODEL_OUTPUT = 'runs/train/activity_cls_cleaned'

def main():
    print("="*60)
    print("TRAINING ON CLEANED DATASET")
    print("="*60)
    
    # Check dataset exists
    if not Path(DATASET_PATH).exists():
        print(f"❌ Error: {DATASET_PATH} not found!")
        print("   Run clean_dataset.py first")
        return
    
    # Show dataset info
    train_dir = Path(DATASET_PATH) / 'train'
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    print(f"\nClasses ({len(classes)}): {classes}")
    
    total_train = 0
    for cls in classes:
        count = len(list((train_dir / cls).glob('*')))
        total_train += count
        print(f"  {cls}: {count} images")
    print(f"Total training images: {total_train}")
    
    # Load model
    print("\nLoading YOLOv8s-cls model...")
    model = YOLO('yolov8s-cls.pt')
    
    # Train with good settings
    print("\nStarting training...")
    results = model.train(
        data=DATASET_PATH,
        epochs=100,
        patience=15,         # Early stopping
        imgsz=224,
        batch=32,
        
        # Strong augmentation to prevent overfitting
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15,
        translate=0.1,
        scale=0.5,
        shear=5,
        perspective=0.0005,
        flipud=0.1,
        fliplr=0.5,
        mosaic=0.0,         # Disable mosaic for classification
        mixup=0.1,
        
        # Regularization
        dropout=0.3,        # Higher dropout
        
        # Output
        project='runs/train',
        name='activity_cls_cleaned',
        exist_ok=True,
        
        # Optimization
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        
        verbose=True
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model: {MODEL_OUTPUT}/weights/best.pt")
    
    # Test the model
    print("\nTesting on validation set...")
    model = YOLO(f'{MODEL_OUTPUT}/weights/best.pt')
    metrics = model.val(data=DATASET_PATH)
    
    print(f"\n✅ Top-1 Accuracy: {metrics.top1:.1%}")
    print(f"✅ Top-5 Accuracy: {metrics.top5:.1%}")

if __name__ == '__main__':
    main()
