#!/usr/bin/env python3
"""
YOLOv8 Classifier Training - 7 Classes (Merged Prowling + Fighting)
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Classifier (7 Classes)")
    parser.add_argument("--model", type=str, default="yolov8s-cls.pt", help="Model: yolov8n-cls, yolov8s-cls, yolov8m-cls")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    parser.add_argument("--name", type=str, default="activity_cls_7class", help="Run name")
    parser.add_argument("--data", type=str, default="activity_dataset", help="Dataset directory")
    
    args = parser.parse_args()
    
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(args.device)}")
    
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    
    print(f"\n{'='*60}")
    print("TRAINING: 7-CLASS ACTIVITY CLASSIFIER")
    print(f"{'='*60}")
    print(f"Classes: walking, leaving_package, passing_out,")
    print(f"         person_pushing, person_running, robbery_knife,")
    print(f"         aggressive_activity (prowling + fighting combined)")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | Image Size: {args.imgsz}")
    print(f"Learning Rate: {args.lr0} | Optimizer: {args.optimizer}")
    print(f"Early Stopping Patience: {args.patience}")
    print(f"{'='*60}\n")
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        optimizer=args.optimizer,
        augment=True,
        project=args.project,
        name=args.name,
        verbose=True,
        seed=42
    )
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best model: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    if results:
        print(f"Best Top-1 Accuracy: {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
        print(f"Best Top-5 Accuracy: {results.results_dict.get('metrics/accuracy_top5', 'N/A')}")
    print("="*60)

if __name__ == "__main__":
    main()
