"""
Train YOLO on COCO8-pose dataset - Person Pose Detection
This script:
1. Downloads COCO8-pose dataset (if not present)  
2. Trains a YOLO pose model for human detection with keypoints

NOTE: COCO8-pose only contains the 'person' class with 17 keypoints.
      Bags (backpack, handbag, suitcase) are NOT in this dataset.
      Use coco8.yaml for object detection including bags.
"""

from ultralytics import YOLO
import os
from pathlib import Path


# Configuration
MODEL_BASE = 'yolo11n-pose.pt'  # Pose estimation model
OUTPUT_NAME = 'human_pose_detector'
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 8

# COCO8-pose only has person class with 17 keypoints:
# nose, left_eye, right_eye, left_ear, right_ear,
# left_shoulder, right_shoulder, left_elbow, right_elbow,
# left_wrist, right_wrist, left_hip, right_hip,
# left_knee, right_knee, left_ankle, right_ankle
TARGET_CLASSES = {
    0: 'person'
}


def train_human_pose_detector(epochs=50, batch_size=8):
    """Train YOLO pose model to detect humans with keypoints"""
    
    print("\n" + "="*60)
    print("🚀 TRAINING HUMAN POSE DETECTOR (COCO8-pose)")
    print("="*60)
    print(f"Base Model: {MODEL_BASE}")
    print(f"Dataset: coco8-pose.yaml (person class with 17 keypoints)")
    print(f"Epochs: {epochs}")
    print(f"Image Size: {IMG_SIZE}")
    print("="*60 + "\n")
    
    # Load pose estimation model
    model = YOLO(MODEL_BASE)
    
    # Train the model on coco8-pose
    print("\n🏋️ Starting pose estimation training...")
    
    results = model.train(
        data='coco8-pose.yaml',  # Uses built-in coco8-pose dataset
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch_size,
        device=0,  # Use GPU
        workers=4,
        amp=True,  # Automatic Mixed Precision for faster training
        project='runs/pose',
        name=OUTPUT_NAME,
        patience=10,  # Early stopping
        save=True,
        plots=True,
        # Augmentation settings
        augment=True,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
    )
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    print(f"Best model saved to: runs/pose/{OUTPUT_NAME}/weights/best.pt")
    print(f"Results saved to: runs/pose/{OUTPUT_NAME}/")
    
    return results


def test_pose_model():
    """Test the trained pose model"""
    
    model_path = f'runs/pose/{OUTPUT_NAME}/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        print("Please train the model first!")
        return
    
    print(f"\n🔍 Testing pose model: {model_path}")
    
    model = YOLO(model_path)
    
    # Test on a sample image or webcam
    results = model.predict(
        source=0,  # Webcam
        show=True,
        conf=0.25,
    )
    
    return results


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Human Pose Detector on COCO8-pose')
    parser.add_argument('--test', action='store_true', help='Test the trained model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    
    args = parser.parse_args()
    
    if args.test:
        test_pose_model()
    else:
        train_human_pose_detector(epochs=args.epochs, batch_size=args.batch)


if __name__ == '__main__':
    main()
