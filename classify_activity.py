#!/usr/bin/env python3
"""
Activity Classifier Inference Utility
Simple script to classify activities from images using the trained model
"""

import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2

# Class mapping
CLASSES = [
    'prowling',
    'leaving_package',
    'passing_out',
    'person_pushing',
    'person_running',
    'fighting_group',
    'robbery_knife',
    'walking'
]

def classify_image(model_path, image_path, confidence_threshold=0.5):
    """Classify activity in an image"""
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference
    results = model.predict(source=str(image_path), verbose=False)
    result = results[0]
    
    # Extract prediction
    class_idx = int(result.probs.top1)
    class_name = CLASSES[class_idx]
    confidence = float(result.probs.top1conf)
    
    # Get top-5 predictions
    top5_indices = result.probs.top5
    top5_confidences = result.probs.top5conf
    
    return {
        'class': class_name,
        'confidence': confidence,
        'class_index': class_idx,
        'top5_classes': [CLASSES[int(idx)] for idx in top5_indices],
        'top5_confidences': [float(c) for c in top5_confidences],
        'passed_threshold': confidence >= confidence_threshold
    }

def main():
    parser = argparse.ArgumentParser(description='Activity Classification Inference')
    parser.add_argument('image', type=str, help='Path to image file')
    parser.add_argument('--model', type=str, 
                       default='runs/train/activity_cls_small2/weights/best.pt',
                       help='Path to model weights')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Confidence threshold')
    parser.add_argument('--show', action='store_true',
                       help='Display image with prediction')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.image).exists():
        print(f"Error: Image file not found: {args.image}")
        return
    
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Classify
    print(f"Classifying: {args.image}")
    result = classify_image(args.model, args.image, args.threshold)
    
    # Display results
    print("\n" + "="*60)
    print("CLASSIFICATION RESULT")
    print("="*60)
    print(f"Predicted Activity: {result['class'].upper()}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Meets Threshold ({args.threshold}): {'✓ Yes' if result['passed_threshold'] else '✗ No'}")
    
    print(f"\nTop-5 Predictions:")
    for i, (cls, conf) in enumerate(zip(result['top5_classes'], result['top5_confidences']), 1):
        print(f"  {i}. {cls:18s} {conf:.1%}")
    
    # Show image if requested
    if args.show:
        img = cv2.imread(args.image)
        if img is not None:
            # Add text overlay
            h, w = img.shape[:2]
            text = f"{result['class']} ({result['confidence']:.1%})"
            cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                       1.5, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('Activity Classification', img)
            print("\nPress any key to close image window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
