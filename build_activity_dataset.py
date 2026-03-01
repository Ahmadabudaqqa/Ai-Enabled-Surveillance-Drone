import random
import shutil
from pathlib import Path

# ================= SETTINGS =================
SEED = 42
TRAIN_RATIO = 0.8

FRAMES_DIR = Path("extracted_frames")
OUT_DIR = Path("activity_dataset")

CODE_TO_CLASS = {
    "WL": "walking",
    "LPP": "leaving_package",
    "PO": "passing_out",
    "PPP": "pushing",
    "PR": "running",
    "FG": "fighting_group",
    "RK": "robbery_knife",
    "PW": "prowl",
}
# ===========================================

def infer_code(name: str):
    name = name.upper()
    for code in CODE_TO_CLASS:
        if code in name:
            return CODE_TO_CLASS[code]
    return None

random.seed(SEED)

if not FRAMES_DIR.exists():
    raise SystemExit(f"❌ Frames folder not found: {FRAMES_DIR.resolve()}")

frames = []
for ext in ("*.jpg", "*.png", "*.jpeg"):
    frames += list(FRAMES_DIR.rglob(ext))

if not frames:
    raise SystemExit("❌ No frames found")

by_class = {v: [] for v in CODE_TO_CLASS.values()}
unknown = []

for f in frames:
    cls = infer_code(f.name)
    if cls is None:
        unknown.append(f)
    else:
        by_class[cls].append(f)

print("===== FRAME COUNTS =====")
for cls, items in by_class.items():
    print(f"{cls:18s}: {len(items)}")
print(f"UNKNOWN: {len(unknown)}")
print("========================")

for split in ("train", "val"):
    for cls in by_class:
        (OUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)

for cls, items in by_class.items():
    random.shuffle(items)
    k = int(len(items) * TRAIN_RATIO)
    train_items = items[:k]
    val_items = items[k:]

    for src in train_items:
        shutil.copy2(src, OUT_DIR / "train" / cls / src.name)
    for src in val_items:
        shutil.copy2(src, OUT_DIR / "val" / cls / src.name)

print(f"\n✅ Dataset created at: {OUT_DIR.resolve()}")
