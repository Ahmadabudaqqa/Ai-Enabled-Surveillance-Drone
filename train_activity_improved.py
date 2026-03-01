#!/usr/bin/env python3
"""
Improved YOLOv8 Classifier Training with Built-in Validation
Uses YOLO's native validation instead of custom metric parsing
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Classifier with Early Stopping")
    parser.add_argument("--model", type=str, default="yolov8s-cls.pt", help="Model size: yolov8n-cls, yolov8s-cls, yolov8m-cls, etc.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--lr0", type=float, default=0.001, help="Initial learning rate")
    parser.add_argument("--optimizer", type=str, default="SGD", help="Optimizer (SGD, Adam, AdamW, etc.)")
    parser.add_argument("--mixup", type=float, default=0.0, help="Mixup augmentation")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation ratio")
    parser.add_argument("--project", type=str, default="runs/train", help="Project directory")
    parser.add_argument("--name", type=str, default="activity_cls", help="Run name")
    parser.add_argument("--data", type=str, default="activity_dataset", help="Dataset directory path")
    
    args = parser.parse_args()
    
    # Verify CUDA availability
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(args.device)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(args.device).total_memory / 1e9:.2f} GB")
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = YOLO(args.model)
    print(f"Model parameters: {sum(p.numel() for p in model.model.parameters()):,}")
    
    # Train with YOLO's native validation
    print(f"\nStarting training...")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Learning rate: {args.lr0}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Early stopping patience: {args.patience}")
    print(f"  Run name: {args.name}")
    print()
    
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        lr0=args.lr0,
        optimizer=args.optimizer,
        mixup=args.mixup,
        mosaic=args.mosaic,
        augment=True,
        project=args.project,
        name=args.name,
        verbose=True,
        seed=42
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Best model saved to: {Path(args.project) / args.name / 'weights' / 'best.pt'}")
    if results:
        print(f"\nFinal Results:")
        print(f"  Best Top-1 Accuracy: {results.results_dict.get('metrics/accuracy_top1', 'N/A')}")
        print(f"  Best Top-5 Accuracy: {results.results_dict.get('metrics/accuracy_top5', 'N/A')}")

if __name__ == "__main__":
    main()
