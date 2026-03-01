from pathlib import Path
import random
import shutil
import argparse


def balance_train_folder(train_dir: Path, target_count: int | None = None, seed: int = 42):
    random.seed(seed)
    classes = [p for p in train_dir.iterdir() if p.is_dir()]
    counts = {c.name: len(list((train_dir / c.name).glob('*.*'))) for c in classes}
    max_count = max(counts.values()) if counts else 0
    target = target_count or max_count

    print("Class counts before balancing:")
    for k, v in counts.items():
        print(f" - {k}: {v}")

    for c in classes:
        imgs = list((train_dir / c.name).glob('*.*'))
        n = len(imgs)
        if n >= target:
            continue
        need = target - n
        print(f"Oversampling {c.name}: need {need} images")
        for i in range(need):
            src = random.choice(imgs)
            dst = train_dir / c.name / f"aug_copy_{i}_{src.name}"
            shutil.copy2(src, dst)

    counts_after = {c.name: len(list((train_dir / c.name).glob('*.*'))) for c in classes}
    print("Class counts after balancing:")
    for k, v in counts_after.items():
        print(f" - {k}: {v}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='activity_dataset', help='dataset root')
    parser.add_argument('--target', type=int, default=None, help='target per-class count (default=max)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.dataset)
    train = root / 'train'
    if not train.exists():
        raise SystemExit(f"Train folder not found: {train}")

    balance_train_folder(train, target_count=args.target, seed=args.seed)


if __name__ == '__main__':
    main()
