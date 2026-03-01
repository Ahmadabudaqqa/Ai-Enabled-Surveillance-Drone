import cv2
import os
import random
import argparse
from pathlib import Path


def random_augment(img):
    # Random horizontal flip
    if random.random() < 0.5:
        img = cv2.flip(img, 1)

    h, w = img.shape[:2]

    # Random rotation
    if random.random() < 0.5:
        angle = random.uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Random brightness
    if random.random() < 0.6:
        factor = random.uniform(0.7, 1.3)
        img = cv2.convertScaleAbs(img, alpha=factor, beta=0)

    # Random blur
    if random.random() < 0.25:
        k = random.choice([1, 3, 5])
        if k > 1:
            img = cv2.GaussianBlur(img, (k, k), 0)

    # Random noise
    if random.random() < 0.15:
        noise = (np.random.randn(*img.shape) * 10).astype('uint8')
        img = cv2.add(img, noise)

    return img


def ensure_ext(fname, ext='.jpg'):
    return str(Path(fname).with_suffix(ext))


def augment_folder(folder_path, n_aug=2, exts=('.jpg', '.jpeg', '.png')):
    folder = Path(folder_path)
    if not folder.exists():
        return 0

    files = [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]
    count = 0
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        for i in range(1, n_aug + 1):
            aug = random_augment(img.copy())
            out_name = f"{p.stem}_aug{i}{p.suffix}"
            out_path = folder / out_name
            cv2.imwrite(str(out_path), aug)
            count += 1
    return count


def main(dataset_dir, modes, n_aug, classes=None):
    dataset = Path(dataset_dir)
    total = 0
    for mode in modes:
        mode_folder = dataset / mode
        if not mode_folder.exists():
            print(f"Skipping missing folder: {mode_folder}")
            continue

        class_folders = [d for d in mode_folder.iterdir() if d.is_dir()]
        if classes:
            class_folders = [d for d in class_folders if d.name in classes]

        for cls in class_folders:
            added = augment_folder(cls, n_aug=n_aug)
            print(f"{mode}/{cls.name}: added {added} augmented images")
            total += added

    print(f"Total augmented images added: {total}")


if __name__ == '__main__':
    import numpy as np

    parser = argparse.ArgumentParser(description='Augment images in dataset train/val folders')
    parser.add_argument('--dataset', default='activity_dataset', help='path to activity_dataset')
    parser.add_argument('--modes', nargs='+', default=['train', 'val'], help='which subfolders to augment')
    parser.add_argument('--n', type=int, default=2, help='number of augmentations per original image')
    parser.add_argument('--classes', nargs='*', help='optional list of class names to process')
    args = parser.parse_args()

    main(args.dataset, args.modes, args.n, classes=args.classes)
