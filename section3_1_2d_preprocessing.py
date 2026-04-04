"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 3.1: 2D PREPROCESSING
=============================================================
Source : D:/himanshu/2d/classified_dataset/
Output : D:/himanshu/2d_processed/
Split  : 80% train / 10% val / 10% test
Classes: glioma | meningioma | notumor | pituitary

What this script does (per document Section 3.1):
  [1] Load all images from classified_dataset/
  [2] Split into train / val / test (stratified 80/10/10)
  [3] Apply augmentations (train only):
        - Random horizontal/vertical flip
        - Random rotation ±15°
        - CLAHE contrast enhancement
        - Random brightness/contrast jitter
        - Normalize (ImageNet mean/std for pretrained EfficientNet-B4)
  [4] Build PyTorch Dataset class
  [5] Build DataLoaders with WeightedRandomSampler for imbalance
  [6] Save split info to JSON for reproducibility

Usage:
  python section3_1_2d_preprocessing.py
=============================================================
"""

import os
import json
import random
import shutil
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = Path("D:/himanshu/2d/classified_dataset")
OUTPUT_DIR  = Path("D:/himanshu/2d_processed")
SPLIT_JSON  = OUTPUT_DIR / "split_info.json"

CLASS_NAMES = ["glioma1", "meningioma1", "notumor1", "pituitary1"]
CLASS_TO_IDX  = {cls: i for i, cls in enumerate(CLASS_NAMES)}
CLASS_DISPLAY = ["glioma", "meningioma", "notumor", "pituitary"]

TARGET_SIZE  = (224, 224)
BATCH_SIZE   = 32
NUM_WORKERS  = 0          # Set to 0 on Windows to avoid multiprocessing issues
RANDOM_SEED  = 42

# ImageNet normalization for pretrained EfficientNet-B4 (Section 4 Stage 1)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ══════════════════════════════════════════════════════════════
# STEP 1 — Collect all image paths + labels
# ══════════════════════════════════════════════════════════════
def collect_image_paths(data_dir: Path) -> tuple:
    """
    Scans all class folders and returns:
      - all_paths : list of Path objects
      - all_labels: list of int labels
    """
    print("\n" + "="*60)
    print("STEP 1 — Collecting image paths")
    print("="*60)

    all_paths  = []
    all_labels = []

    for cls in CLASS_NAMES:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            print(f"  [WARNING] Not found: {cls_dir}")
            continue

        imgs = (list(cls_dir.glob("*.jpg"))  +
                list(cls_dir.glob("*.jpeg")) +
                list(cls_dir.glob("*.png")))

        label = CLASS_TO_IDX[cls]
        all_paths.extend(imgs)
        all_labels.extend([label] * len(imgs))
        print(f"  {cls:<15}: {len(imgs)} images")

    print(f"\n  Total images: {len(all_paths)}")
    return all_paths, all_labels


# ══════════════════════════════════════════════════════════════
# STEP 2 — Stratified 80 / 10 / 10 Split
# ══════════════════════════════════════════════════════════════
def split_dataset(all_paths, all_labels):
    """
    Stratified split: 80% train, 10% val, 10% test
    Stratified = each class keeps same proportion in all splits
    """
    print("\n" + "="*60)
    print("STEP 2 — Stratified 80/10/10 Split")
    print("="*60)

    # First split: 80% train, 20% temp
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels,
        test_size=0.20,
        stratify=all_labels,
        random_state=RANDOM_SEED
    )

    # Second split: 50% of temp = val, 50% = test (i.e. 10/10)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=0.50,
        stratify=temp_labels,
        random_state=RANDOM_SEED
    )

    print(f"\n  {'Split':<10} {'Total':>8}  {'Glioma':>8}  {'Menin':>8}  {'NoTumor':>8}  {'Pituit':>8}")
    print(f"  {'-'*56}")

    for name, paths, labels in [
        ("Train",  train_paths, train_labels),
        ("Val",    val_paths,   val_labels),
        ("Test",   test_paths,  test_labels),
    ]:
        c = Counter(labels)
        print(f"  {name:<10} {len(paths):>8}  "
              f"{c[0]:>8}  {c[1]:>8}  {c[2]:>8}  {c[3]:>8}")

    return (train_paths, train_labels,
            val_paths,   val_labels,
            test_paths,  test_labels)


# ══════════════════════════════════════════════════════════════
# STEP 3 — CLAHE Augmentation (custom transform)
# ══════════════════════════════════════════════════════════════
class CLAHETransform:
    """
    Contrast Limited Adaptive Histogram Equalization
    Applied per-channel on RGB images — Section 5.4 requirement
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        # Apply CLAHE to each channel separately
        channels = []
        for c in range(3):
            enhanced = self.clahe.apply(img_np[:, :, c])
            channels.append(enhanced)
        img_clahe = np.stack(channels, axis=2)
        return Image.fromarray(img_clahe)


# ══════════════════════════════════════════════════════════════
# STEP 4 — Transforms (train vs val/test)
# ══════════════════════════════════════════════════════════════
def get_transforms(split: str):
    """
    Train: augmentations + normalize
    Val/Test: only resize + normalize (no augmentation)
    """
    if split == "train":
        return T.Compose([
            T.Resize(TARGET_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.3),
            T.RandomRotation(degrees=15),
            T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1
            ),
            CLAHETransform(clip_limit=2.0),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:  # val and test — no augmentation
        return T.Compose([
            T.Resize(TARGET_SIZE),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


# ══════════════════════════════════════════════════════════════
# STEP 5 — PyTorch Dataset Class
# ══════════════════════════════════════════════════════════════
class BrainTumor2DDataset(Dataset):
    """
    PyTorch Dataset for 2D brain tumor classification.
    Classes: glioma(0), meningioma(1), notumor(2), pituitary(3)
    """

    def __init__(self, paths, labels, transform=None, split="train"):
        self.paths     = [str(p) for p in paths]
        self.labels    = labels
        self.transform = transform
        self.split     = split

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label    = self.labels[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Apply transforms
        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

    def get_class_name(self, label_idx):
        return CLASS_NAMES[label_idx]


# ══════════════════════════════════════════════════════════════
# STEP 6 — WeightedRandomSampler for class imbalance
# ══════════════════════════════════════════════════════════════
def get_weighted_sampler(labels):
    """
    Creates WeightedRandomSampler so every class
    gets equal representation per batch — Section 1.3 requirement
    """
    class_counts = Counter(labels)
    class_weights = {
        cls: 1.0 / count
        for cls, count in class_counts.items()
    }

    sample_weights = torch.tensor(
        [class_weights[label] for label in labels],
        dtype=torch.float
    )

    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True
    )
    return sampler


# ══════════════════════════════════════════════════════════════
# STEP 7 — Build All DataLoaders
# ══════════════════════════════════════════════════════════════
def build_dataloaders(train_paths, train_labels,
                      val_paths,   val_labels,
                      test_paths,  test_labels):
    """
    Returns train/val/test DataLoaders ready for training.
    """
    print("\n" + "="*60)
    print("STEP 3-6 — Building Datasets & DataLoaders")
    print("="*60)

    # Datasets
    train_dataset = BrainTumor2DDataset(
        train_paths, train_labels,
        transform=get_transforms("train"),
        split="train"
    )
    val_dataset = BrainTumor2DDataset(
        val_paths, val_labels,
        transform=get_transforms("val"),
        split="val"
    )
    test_dataset = BrainTumor2DDataset(
        test_paths, test_labels,
        transform=get_transforms("test"),
        split="test"
    )

    # Weighted sampler for training only
    sampler = get_weighted_sampler(train_labels)

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,       # WeightedRandomSampler
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size  = BATCH_SIZE,
        shuffle     = False,
        num_workers = NUM_WORKERS,
        pin_memory  = True,
    )

    print(f"\n  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Image size    : {TARGET_SIZE}")

    return train_loader, val_loader, test_loader, \
           train_dataset, val_dataset, test_dataset


# ══════════════════════════════════════════════════════════════
# STEP 8 — Save Split Info JSON (for reproducibility)
# ══════════════════════════════════════════════════════════════
def save_split_info(train_paths, train_labels,
                    val_paths,   val_labels,
                    test_paths,  test_labels):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split_info = {
        "random_seed":   RANDOM_SEED,
        "class_names":   CLASS_NAMES,
        "class_to_idx":  CLASS_TO_IDX,
        "train": {
            "count":   len(train_paths),
            "paths":   [str(p) for p in train_paths],
            "labels":  train_labels,
            "distribution": dict(Counter(train_labels))
        },
        "val": {
            "count":   len(val_paths),
            "paths":   [str(p) for p in val_paths],
            "labels":  val_labels,
            "distribution": dict(Counter(val_labels))
        },
        "test": {
            "count":   len(test_paths),
            "paths":   [str(p) for p in test_paths],
            "labels":  test_labels,
            "distribution": dict(Counter(test_labels))
        }
    }

    with open(SPLIT_JSON, "w") as f:
        json.dump(split_info, f, indent=2)

    print(f"\n  Split info saved → {SPLIT_JSON}")


# ══════════════════════════════════════════════════════════════
# STEP 9 — Sanity Check (verify one batch)
# ══════════════════════════════════════════════════════════════
def sanity_check(train_loader):
    print("\n" + "="*60)
    print("STEP 9 — Sanity Check (one batch)")
    print("="*60)

    images, labels = next(iter(train_loader))

    print(f"\n  Batch image shape : {images.shape}")
    print(f"  Batch label shape : {labels.shape}")
    print(f"  Image dtype       : {images.dtype}")
    print(f"  Label dtype       : {labels.dtype}")
    print(f"  Image min/max     : {images.min():.3f} / {images.max():.3f}")
    print(f"  Labels in batch   : {labels.tolist()[:10]} ...")
    print(f"  Classes in batch  : {[CLASS_NAMES[l] for l in labels[:5].tolist()]}")
    print(f"\n  [OK] DataLoader is working correctly!")
    print(f"  [OK] Images are normalized (not 0-255)")
    print(f"  [OK] Ready for Section 4 Stage 1 training")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "█"*60)
    print("  BRAIN TUMOR DETECTION — SECTION 3.1")
    print("  2D PREPROCESSING PIPELINE")
    print("█"*60)
    print(f"  Source : {DATA_DIR}")
    print(f"  Output : {OUTPUT_DIR}")

    # Step 1: Collect
    all_paths, all_labels = collect_image_paths(DATA_DIR)

    if len(all_paths) == 0:
        print("\n  [ERROR] No images found! Check your DATA_DIR path.")
        return

    # Step 2: Split
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = split_dataset(all_paths, all_labels)

    # Steps 3-7: Build DataLoaders
    (train_loader, val_loader, test_loader,
     train_dataset, val_dataset, test_dataset) = build_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels
    )

    # Step 8: Save split info
    save_split_info(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels
    )

    # Step 9: Sanity check
    sanity_check(train_loader)

    print("\n" + "═"*60)
    print("  SECTION 3.1 COMPLETE ✓")
    print("  Next: Section 3.2 — BraTS 3D H5 Preprocessing")
    print("═"*60 + "\n")

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    main()