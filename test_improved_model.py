#!/usr/bin/env python3
"""Test improved model inference"""

import os
import random
from ultralytics import YOLO

random.seed(42)

model_path = "runs/train/activity_cls_small2/weights/best.pt"
val_dir = "activity_dataset/val"

print(f"Loading improved model: {model_path}")
model = YOLO(model_path)

classes = ['prowling', 'leaving_package', 'passing_out', 'person_pushing', 
           'person_running', 'fighting_group', 'robbery_knife', 'walking']

print("\n" + "="*80)
print("IMPROVED MODEL - TESTING INFERENCE ON VALIDATION SAMPLES")
print("="*80)

class_predictions = {cls: [] for cls in classes}

for class_name in classes:
    class_dir = os.path.join(val_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
    
    if not images:
        continue
    
    sample_images = random.sample(images, min(5, len(images)))
    
    print(f"\n[{class_name}] Testing {len(sample_images)} images:")
    
    for img_file in sample_images:
        img_path = os.path.join(class_dir, img_file)
        results = model.predict(source=img_path, verbose=False)
        result = results[0]
        
        pred_class_idx = result.probs.top1
        pred_class_name = classes[pred_class_idx]
        pred_confidence = result.probs.top1conf.item()
        
        class_predictions[class_name].append(pred_class_name)
        
        correct = "✓" if pred_class_name == class_name else "✗"
        print(f"  {correct} {pred_class_name:18s} (conf: {pred_confidence:.3f})")

# Summary
print("\n" + "="*80)
print("ACCURACY SUMMARY")
print("="*80)

total_correct = 0
total_tested = 0

for class_name in classes:
    predictions = class_predictions[class_name]
    if not predictions:
        continue
    
    correct = sum(1 for p in predictions if p == class_name)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0
    
    total_correct += correct
    total_tested += total
    
    print(f"{class_name:18s}: {correct}/{total} ({accuracy*100:.1f}%)")

if total_tested > 0:
    overall_accuracy = total_correct / total_tested
    print(f"\n{'Overall':18s}: {total_correct}/{total_tested} ({overall_accuracy*100:.1f}%)")
