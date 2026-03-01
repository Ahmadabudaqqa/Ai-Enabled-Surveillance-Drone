import argparse
import glob
import json
import csv
import os
import time
from pathlib import Path

from ultralytics import YOLO


def find_run_dir(project: Path, name: str) -> Path | None:
    base = project / name
    if base.exists():
        return base
    # fallback: find directories starting with name
    for p in sorted(project.glob(f"{name}*"), key=os.path.getmtime, reverse=True):
        if p.is_dir():
            return p
    return None


def parse_val_loss(run_dir: Path):
    # Try common metric files saved by Ultralytics
    # 1) metrics.json
    mjson = run_dir / "metrics.json"
    if mjson.exists():
        try:
            data = json.loads(mjson.read_text())
            # try common keys
            for key in ("val_loss", "val/loss", "val/loss0", "loss_val"):
                if key in data:
                    return float(data[key])
        except Exception:
            pass

    # 2) metrics.csv or results.csv
    for name in ("metrics.csv", "results.csv", "metrics.txt"):
        p = run_dir / name
        if p.exists():
            try:
                text = p.read_text().strip().splitlines()
                if not text:
                    continue
                header = [h.strip() for h in text[0].split(',')]
                last = [c.strip() for c in text[-1].split(',')]
                # find a header matching val loss
                for hidx, h in enumerate(header):
                    if any(tok in h.lower() for tok in ("val_loss", "val loss", "val/loss", "val", "loss")):
                        try:
                            return float(last[hidx])
                        except Exception:
                            continue
            except Exception:
                continue

    # 3) try searching for a JSON file with numbers
    for p in run_dir.glob("*.json"):
        try:
            data = json.loads(p.read_text())
            # search numeric values
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, (int, float)):
                        return float(v)
        except Exception:
            continue

    return None


def parse_val_top1(run_dir: Path):
    # look for metrics.csv or results.csv with a header containing top1_acc
    for name in ("metrics.csv", "results.csv", "results.csv", "metrics.txt"):
        p = run_dir / name
        if p.exists():
            try:
                lines = p.read_text().strip().splitlines()
                if not lines:
                    continue
                header = [h.strip() for h in lines[0].split(',')]
                last = [c.strip() for c in lines[-1].split(',')]
                for hidx, h in enumerate(header):
                    if 'top1' in h.lower() or 'top1_acc' in h.lower():
                        try:
                            return float(last[hidx])
                        except Exception:
                            continue
            except Exception:
                continue
    # fallback: try to read results.txt and parse a line like 'all      0.126      0.626'
    p = run_dir / 'results.txt'
    if p.exists():
        try:
            for line in p.read_text().splitlines():
                parts = line.strip().split()
                # expected format: 'all <loss> <top1_acc>' or similar
                if parts and parts[0].lower() == 'all' and len(parts) >= 3:
                    try:
                        # third column is typically top1 accuracy
                        return float(parts[2])
                    except Exception:
                        try:
                            return float(parts[1])
                        except Exception:
                            continue
        except Exception:
            pass
    return None


def main():
    parser = argparse.ArgumentParser(description="Monitored YOLOv8 classifier training with early stopping")
    parser.add_argument("--data", default="activity_dataset", help="dataset folder")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="base model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="activity_cls")
    parser.add_argument("--augment", action='store_true', help="enable augmentation during training")
    parser.add_argument("--mixup", type=float, default=0.1, help="mixup ratio for training")
    parser.add_argument("--lr0", type=float, default=0.001, help="initial learning rate")
    parser.add_argument("--optimizer", default='auto', help="optimizer name or 'auto'")
    args = parser.parse_args()

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)

    # choose monitoring direction: minimize for loss, maximize for accuracy
    monitor = 'val_loss'
    # if user provided environment variable or future arg, respect it; keep default val_loss
    best = float("inf")
    maximize = False
    wait = 0
    started = False

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        start = time.time()
        # on first epoch do not attempt to resume from the pretrained .pt; after that reload
        # the last checkpoint and continue training from it.
        if started:
            run_dir = find_run_dir(project, args.name)
            if run_dir is not None:
                candidate = run_dir / 'weights' / 'last.pt'
                if not candidate.exists():
                    candidate = run_dir / 'weights' / 'best.pt'
                if candidate.exists():
                    print(f"Resuming from checkpoint: {candidate}")
                    model = YOLO(str(candidate))

        train_kwargs = dict(
            data=args.data,
            epochs=1,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project),
            name=args.name,
            exist_ok=True,
            resume=False,
            lr0=args.lr0,
            optimizer=args.optimizer,
            mixup=args.mixup,
        )
        if args.augment:
            train_kwargs['augment'] = True

        model.train(**train_kwargs)
        started = True
        dur = time.time() - start
        print(f"Epoch finished in {dur:.1f}s — parsing metrics...")

        run_dir = find_run_dir(project, args.name)
        if run_dir is None:
            print("Could not find run folder; skipping early-stopping check.")
            continue

        # Try to parse validation top1 accuracy first (more reliable for classification),
        # fall back to val_loss if requested
        val_score = parse_val_top1(run_dir)
        if val_score is not None:
            if best == float('inf'):
                # initialize best for accuracy
                best = float('-inf')
                maximize = True
            print(f"Validation top1_acc: {val_score:.6f} (best: {best:.6f})")
            improved = val_score > best + args.min_delta if maximize else val_score < best - args.min_delta
            if improved:
                best = val_score
                wait = 0
                print("New best validation score — resetting patience.")
            else:
                wait += 1
                print(f"No improvement (patience {wait}/{args.patience}).")
                if wait >= args.patience:
                    print(f"Early stopping triggered after {wait} epochs with no improvement.")
                    break
        else:
            # fallback to val_loss
            val_loss = parse_val_loss(run_dir)
            if val_loss is None:
                print("Could not parse validation metrics from run artifacts; continuing without early stopping.")
                continue
            print(f"Validation loss: {val_loss:.6f} (best: {best:.6f})")
            if val_loss < best - args.min_delta:
                best = val_loss
                wait = 0
                print("New best validation loss — resetting patience.")
            else:
                wait += 1
                print(f"No improvement (patience {wait}/{args.patience}).")
                if wait >= args.patience:
                    print(f"Early stopping triggered after {wait} epochs with no improvement.")
                    break

    print(f"Training complete. Best validation loss: {best:.6f}")


if __name__ == "__main__":
    main()
