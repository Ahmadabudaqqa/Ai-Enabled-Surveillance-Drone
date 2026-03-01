"""
Test the model on sample images from each class to see predictions
"""
from ultralytics import YOLO
from pathlib import Path
import random

# Load model
model = YOLO('runs/train/activity_cls_improved/weights/best.pt')
print(f"Model classes: {model.names}")
print("=" * 60)

# Test on validation set
val_path = Path('activity_dataset/val')

for cls_folder in sorted(val_path.iterdir()):
    if not cls_folder.is_dir():
        continue
    
    cls_name = cls_folder.name
    images = list(cls_folder.glob('*.jpg')) + list(cls_folder.glob('*.png'))
    
    if len(images) == 0:
        print(f"{cls_name}: No images found")
        continue
    
    # Test on 5 random images from each class
    sample_images = random.sample(images, min(5, len(images)))
    
    correct = 0
    predictions = []
    
    for img_path in sample_images:
        result = model.predict(source=str(img_path), imgsz=224, verbose=False)
        pred_idx = int(result[0].probs.top5[0])
        pred_conf = float(result[0].probs.top5conf[0])
        pred_name = model.names[pred_idx]
        
        predictions.append((pred_name, pred_conf))
        
        if pred_name == cls_name:
            correct += 1
    
    accuracy = correct / len(sample_images) * 100
    pred_str = ", ".join([f"{p[0]}({p[1]:.0%})" for p in predictions])
    
    status = "✅" if accuracy >= 60 else "⚠️" if accuracy >= 40 else "❌"
    print(f"{status} {cls_name}: {accuracy:.0f}% accuracy")
    print(f"   Predictions: {pred_str}")
    print()

print("=" * 60)
print("Legend: ✅ Good (60%+) | ⚠️ Medium (40-60%) | ❌ Poor (<40%)")
