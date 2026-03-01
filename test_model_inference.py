#!/usr/bin/env python3
"""Test model inference to diagnose prediction patterns"""

import os
import random
from ultralytics import YOLO
from pathlib import Path

# Set seed for reproducibility
random.seed(42)

# Load best model from final2 run
model_path = "runs/train/activity_cls_final2/weights/best.pt"
val_dir = "activity_dataset/val"

if not os.path.exists(model_path):
    print(f"Error: Model not found at {model_path}")
    exit(1)

print(f"Loading model: {model_path}")
model = YOLO(model_path)

# Get class names from data.yaml
classes = ['prowling', 'leaving_package', 'passing_out', 'person_pushing', 
           'person_running', 'fighting_group', 'robbery_knife', 'walking']

# Sample 5 random images from each class
print("\n" + "="*80)
print("TESTING INFERENCE ON RANDOM VALIDATION SAMPLES")
print("="*80)

class_predictions = {cls: [] for cls in classes}

for class_name in classes:
    class_dir = os.path.join(val_dir, class_name)
    images = [f for f in os.listdir(class_dir) if f.endswith(('.jpg', '.png'))]
    
    if not images:
        print(f"\n[{class_name}] No images found!")
        continue
    
    # Sample up to 3 images
    sample_images = random.sample(images, min(3, len(images)))
    
    print(f"\n[{class_name}] Testing {len(sample_images)} images:")
    
    for img_file in sample_images:
        img_path = os.path.join(class_dir, img_file)
        
        # Run inference
        results = model.predict(source=img_path, verbose=False)
        result = results[0]
        
        # Get prediction
        pred_class_idx = result.probs.top1
        pred_class_name = classes[pred_class_idx]
        pred_confidence = result.probs.top1conf.item()
        
        class_predictions[class_name].append(pred_class_name)
        
        correct = "✓" if pred_class_name == class_name else "✗"
        print(f"  {correct} {img_file[:30]:30s} -> Predicted: {pred_class_name:18s} (conf: {pred_confidence:.3f})")

# Summary statistics
print("\n" + "="*80)
print("PREDICTION DISTRIBUTION SUMMARY")
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
    
    print(f"{class_name:18s}: {correct}/{total} correct ({accuracy*100:.1f}%)")

if total_tested > 0:
    overall_accuracy = total_correct / total_tested
    print(f"\n{'Overall Accuracy':18s}: {total_correct}/{total_tested} ({overall_accuracy*100:.1f}%)")

# Check if model is predicting single class
all_predictions = []
for predictions in class_predictions.values():
    all_predictions.extend(predictions)

unique_predictions = set(all_predictions)
print(f"\nUnique predictions made: {unique_predictions}")
if len(unique_predictions) == 1:
    print(f"⚠️  WARNING: Model is predicting ONLY '{list(unique_predictions)[0]}' for ALL samples!")
    print("   This suggests a critical data/label issue or model collapse.")
else:
    print(f"✓ Model is making diverse predictions across {len(unique_predictions)} classes")
