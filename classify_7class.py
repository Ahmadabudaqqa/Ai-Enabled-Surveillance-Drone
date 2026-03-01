#!/usr/bin/env python3
"""
Classify activities with 7-class model (prowling + fighting merged)
"""

import argparse
from pathlib import Path
from ultralytics import YOLO

def classify_image(model_path, image_path):
    """Classify activity in image"""
    
    model = YOLO(model_path)
    results = model.predict(source=str(image_path), verbose=False)
    result = results[0]
    
    # Get class names from the trained model
    class_names = model.names
    
    class_idx = int(result.probs.top1)
    class_name = class_names[class_idx]
    confidence = float(result.probs.top1conf)
    
    top5_indices = result.probs.top5
    top5_confidences = result.probs.top5conf
    
    return {
        'class': class_name,
        'confidence': confidence,
        'top5_classes': [class_names[int(idx)] for idx in top5_indices],
        'top5_confidences': [float(c) for c in top5_confidences],
    }

def main():
    parser = argparse.ArgumentParser(description='Activity Classification (7 Classes)')
    parser.add_argument('image', type=str, help='Image file')
    parser.add_argument('--model', type=str, 
                       default='runs/train/activity_cls_7class/weights/best.pt',
                       help='Model path')
    parser.add_argument('--show', action='store_true', help='Display image')
    
    args = parser.parse_args()
    
    if not Path(args.image).exists():
        print(f"Error: Image not found: {args.image}")
        return
    
    if not Path(args.model).exists():
        print(f"Error: Model not found: {args.model}")
        return
    
    print(f"Classifying: {args.image}")
    result = classify_image(args.model, args.image)
    
    print("\n" + "="*60)
    print("CLASSIFICATION RESULT (7-CLASS MODEL)")
    print("="*60)
    print(f"Predicted Activity: {result['class'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    print(f"\nTop-5 Predictions:")
    for i, (cls, conf) in enumerate(zip(result['top5_classes'], result['top5_confidences']), 1):
        print(f"  {i}. {cls:25s} {conf:.1%}")
    print("="*60)

if __name__ == "__main__":
    main()
