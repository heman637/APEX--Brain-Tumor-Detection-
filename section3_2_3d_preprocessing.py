"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 3.2: 3D BRATS PREPROCESSING v2
=============================================================
UPGRADES over v1:
  [1] Elastic deformation augmentation
  [2] Random gamma correction
  [3] Random Gaussian noise
  [4] Random modality dropout
  [5] Random intensity shift per modality
  [6] Random crop + resize
  [7] Stronger spatial augmentations

Source : D:/himanshu/3d/BraTS/BraTS2020_training_data/content/data/
Output : D:/himanshu/3d_processed/
=============================================================
"""

import random
import json
import numpy as np
from pathlib import Path

import h5py
import cv2
from scipy.ndimage import map_coordinates, gaussian_filter
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# ─────────────────────────────────────────────
# PATHS & CONFIG
# ─────────────────────────────────────────────
DATA_DIR   = Path("D:/himanshu/3d/BraTS/BraTS2020_training_data/content/data")
OUTPUT_DIR = Path("D:/himanshu/3d_processed")
SPLIT_JSON = OUTPUT_DIR / "brats_split_info.json"

IMAGE_KEY       = "image"
MASK_KEY        = "mask"
MODALITY_NAMES  = ["T1", "T1ce", "T2", "FLAIR"]
TARGET_SIZE     = (224, 224)
BATCH_SIZE      = 16
NUM_WORKERS     = 0
RANDOM_SEED     = 42
MIN_TUMOR_RATIO = 0.01


# ══════════════════════════════════════════════════════════════
# STEP 1 — Scan H5 files
# ══════════════════════════════════════════════════════════════
def scan_h5_files(data_dir: Path) -> list:
    print("\n" + "="*60)
    print("STEP 1 — Scanning H5 slice files")
    print("="*60)
    h5_files = sorted([f for f in data_dir.glob("*.h5")
                       if "volume" in f.name.lower()])
    print(f"  Total H5 slice files found : {len(h5_files)}")
    if h5_files:
        print(f"  First file : {h5_files[0].name}")
        print(f"  Last file  : {h5_files[-1].name}")
    return h5_files


# ══════════════════════════════════════════════════════════════
# STEP 2 — Z-score normalization
# ══════════════════════════════════════════════════════════════
def normalize_modality(img_2d: np.ndarray) -> np.ndarray:
    mean = img_2d.mean()
    std  = img_2d.std()
    if std < 1e-8:
        return np.zeros_like(img_2d, dtype=np.float32)
    normalized = (img_2d - mean) / std
    normalized = np.clip(normalized, -5.0, 5.0)
    normalized = (normalized - normalized.min()) / \
                 (normalized.max() - normalized.min() + 1e-8)
    return normalized.astype(np.float32)


# ══════════════════════════════════════════════════════════════
# STEP 3 — CLAHE
# ══════════════════════════════════════════════════════════════
def apply_clahe(img_2d: np.ndarray) -> np.ndarray:
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_uint8 = (img_2d * 255).astype(np.uint8)
    return (clahe.apply(img_uint8) / 255.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# STEP 4 — Process one H5 slice
# ══════════════════════════════════════════════════════════════
def process_h5_slice(h5_path: Path) -> dict:
    with h5py.File(str(h5_path), "r") as f:
        image = f[IMAGE_KEY][:]
        mask  = f[MASK_KEY][:]

    processed = []
    for m_idx in range(4):
        mod = image[:, :, m_idx].astype(np.float32)
        mod = normalize_modality(mod)
        mod = apply_clahe(mod)
        mod = cv2.resize(mod, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        processed.append(mod)

    image_tensor = torch.tensor(np.stack(processed, axis=0), dtype=torch.float32)

    processed_masks = []
    for m_idx in range(3):
        mc = mask[:, :, m_idx].astype(np.float32)
        mc = cv2.resize(mc, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
        processed_masks.append((mc > 0.5).astype(np.float32))

    mask_tensor  = torch.tensor(np.stack(processed_masks, axis=0), dtype=torch.float32)
    tumor_pixels = (mask_tensor[0] > 0).sum().item()
    tumor_ratio  = tumor_pixels / (TARGET_SIZE[0] * TARGET_SIZE[1])

    return {
        "image":       image_tensor,
        "mask":        mask_tensor,
        "has_tumor":   tumor_ratio >= MIN_TUMOR_RATIO,
        "tumor_ratio": tumor_ratio,
        "path":        str(h5_path),
    }


# ══════════════════════════════════════════════════════════════
# STEP 5 — Elastic Deformation
# ══════════════════════════════════════════════════════════════
def elastic_deform(image_np, mask_np, alpha=720, sigma=24):
    """
    Apply elastic deformation to image and mask simultaneously.
    image_np: (4, H, W)
    mask_np:  (3, H, W)
    """
    H, W = image_np.shape[1], image_np.shape[2]

    # Generate random displacement fields
    dx = gaussian_filter(
        (np.random.rand(H, W) * 2 - 1), sigma) * alpha
    dy = gaussian_filter(
        (np.random.rand(H, W) * 2 - 1), sigma) * alpha

    # Create meshgrid and add displacement
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    indices = (
        np.clip(y + dy, 0, H - 1).flatten(),
        np.clip(x + dx, 0, W - 1).flatten()
    )

    # Apply to each modality
    deformed_image = np.zeros_like(image_np)
    for i in range(image_np.shape[0]):
        deformed_image[i] = map_coordinates(
            image_np[i], indices, order=1
        ).reshape(H, W)

    # Apply to each mask channel (nearest neighbour)
    deformed_mask = np.zeros_like(mask_np)
    for i in range(mask_np.shape[0]):
        deformed_mask[i] = map_coordinates(
            mask_np[i], indices, order=0
        ).reshape(H, W)

    return deformed_image, deformed_mask


# ══════════════════════════════════════════════════════════════
# STEP 6 — BraTS Dataset v2 with Full Augmentations
# ══════════════════════════════════════════════════════════════
class BraTSDataset(Dataset):
    """
    BraTS Dataset with full augmentation pipeline.

    Augmentations (train only):
      Spatial  : flip, rotate, elastic deformation, random crop
      Intensity: gamma, noise, blur, intensity shift
      MRI      : random modality dropout
    """

    def __init__(self, h5_files: list, split: str = "train",
                 augment: bool = False):
        self.h5_files = [str(f) for f in h5_files]
        self.split    = split
        self.augment  = augment and (split == "train")

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        result = process_h5_slice(Path(self.h5_files[idx]))
        image  = result["image"]
        mask   = result["mask"]

        if self.augment:
            image, mask = self._augment(image, mask)

        return image, mask

    def _augment(self, image, mask):
        """Full augmentation pipeline."""
        # Convert to numpy for elastic deformation
        image_np = image.numpy()   # (4, H, W)
        mask_np  = mask.numpy()    # (3, H, W)

        # ── 1. Elastic deformation (p=0.3) ──
        if random.random() < 0.3:
            image_np, mask_np = elastic_deform(
                image_np, mask_np, alpha=720, sigma=24
            )

        # ── 2. Random crop + resize (p=0.4) ──
        if random.random() < 0.4:
            H, W    = image_np.shape[1], image_np.shape[2]
            crop    = random.uniform(0.75, 0.95)
            ch, cw  = int(H * crop), int(W * crop)
            y0      = random.randint(0, H - ch)
            x0      = random.randint(0, W - cw)
            image_np= np.stack([
                cv2.resize(image_np[i, y0:y0+ch, x0:x0+cw],
                           (W, H), interpolation=cv2.INTER_LINEAR)
                for i in range(image_np.shape[0])
            ])
            mask_np = np.stack([
                cv2.resize(mask_np[i, y0:y0+ch, x0:x0+cw],
                           (W, H), interpolation=cv2.INTER_NEAREST)
                for i in range(mask_np.shape[0])
            ])

        # Back to tensor for spatial transforms
        image = torch.from_numpy(image_np)
        mask  = torch.from_numpy(mask_np)

        # ── 3. Random horizontal flip ──
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)

        # ── 4. Random vertical flip ──
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)

        # ── 5. Random rotation ±20° ──
        if random.random() > 0.4:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

        # ── Intensity augmentations (image only, not mask) ──
        image_np = image.numpy()

        # ── 6. Random gamma correction per modality (p=0.5) ──
        if random.random() < 0.5:
            for i in range(image_np.shape[0]):
                gamma       = random.uniform(0.7, 1.5)
                image_np[i] = np.clip(image_np[i] ** gamma, 0, 1)

        # ── 7. Random intensity shift per modality (p=0.5) ──
        if random.random() < 0.5:
            for i in range(image_np.shape[0]):
                shift       = random.uniform(-0.1, 0.1)
                scale       = random.uniform(0.9, 1.1)
                image_np[i] = np.clip(image_np[i] * scale + shift, 0, 1)

        # ── 8. Random Gaussian noise (p=0.3) ──
        if random.random() < 0.3:
            noise       = np.random.normal(0, 0.02, image_np.shape)
            image_np    = np.clip(image_np + noise, 0, 1).astype(np.float32)

        # ── 9. Random Gaussian blur (p=0.2) ──
        if random.random() < 0.2:
            for i in range(image_np.shape[0]):
                sigma       = random.uniform(0.5, 1.5)
                image_np[i] = gaussian_filter(image_np[i], sigma=sigma)

        # ── 10. Random modality dropout (p=0.15) ──
        # Zero out one random modality — forces robustness
        if random.random() < 0.15:
            drop_idx            = random.randint(0, 3)
            image_np[drop_idx]  = 0.0

        image = torch.from_numpy(image_np.astype(np.float32))
        return image, mask


# ══════════════════════════════════════════════════════════════
# STEP 7 — Filter + Split
# ══════════════════════════════════════════════════════════════
def filter_and_split_files(h5_files: list) -> tuple:
    print("\n" + "="*60)
    print("STEP 6 — Filtering empty slices + splitting")
    print("="*60)

    tumor_files    = []
    no_tumor_files = []

    print(f"\n  Scanning {len(h5_files)} slices for tumor presence...")
    for h5_path in tqdm(h5_files, desc="  Filtering"):
        with h5py.File(str(h5_path), "r") as f:
            mask = f[MASK_KEY][:]
        ratio = np.sum(mask[:, :, 0] > 0) / (mask.shape[0] * mask.shape[1])
        if ratio >= MIN_TUMOR_RATIO:
            tumor_files.append(h5_path)
        else:
            no_tumor_files.append(h5_path)

    print(f"\n  Slices with tumor    : {len(tumor_files)}")
    print(f"  Slices without tumor : {len(no_tumor_files)}")

    all_files  = tumor_files + no_tumor_files
    all_labels = [1]*len(tumor_files) + [0]*len(no_tumor_files)

    train_files, temp_files, train_labels, temp_labels = train_test_split(
        all_files, all_labels, test_size=0.20,
        stratify=all_labels, random_state=RANDOM_SEED
    )
    val_files, test_files, val_labels, test_labels = train_test_split(
        temp_files, temp_labels, test_size=0.50,
        stratify=temp_labels, random_state=RANDOM_SEED
    )

    print(f"\n  {'Split':<10} {'Total':>8}  {'Tumor':>8}  {'NoTumor':>8}")
    print(f"  {'-'*38}")
    for name, files, labels in [
        ("Train", train_files, train_labels),
        ("Val",   val_files,   val_labels),
        ("Test",  test_files,  test_labels),
    ]:
        c = Counter(labels)
        print(f"  {name:<10} {len(files):>8}  {c[1]:>8}  {c[0]:>8}")

    return train_files, val_files, test_files


# ══════════════════════════════════════════════════════════════
# STEP 8 — Build DataLoaders
# ══════════════════════════════════════════════════════════════
def build_brats_dataloaders(train_files, val_files, test_files):
    print("\n" + "="*60)
    print("STEP 7-8 — Building BraTS Datasets & DataLoaders")
    print("="*60)

    train_dataset = BraTSDataset(train_files, split="train", augment=True)
    val_dataset   = BraTSDataset(val_files,   split="val",   augment=False)
    test_dataset  = BraTSDataset(test_files,  split="test",  augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"\n  Train slices  : {len(train_dataset)}")
    print(f"  Val slices    : {len(val_dataset)}")
    print(f"  Test slices   : {len(test_dataset)}")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Image shape   : (4, 224, 224) — 4 modalities")
    print(f"  Mask shape    : (3, 224, 224) — WT, TC, ET")
    print(f"  Augmentations : elastic, gamma, noise, blur, dropout (train only)")

    return (train_loader, val_loader, test_loader,
            train_dataset, val_dataset, test_dataset)


# ══════════════════════════════════════════════════════════════
# STEP 9 — Save Split Info
# ══════════════════════════════════════════════════════════════
def save_brats_split_info(train_files, val_files, test_files):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    split_info = {
        "random_seed":     RANDOM_SEED,
        "target_size":     TARGET_SIZE,
        "modalities":      MODALITY_NAMES,
        "mask_regions":    ["WholeTumor", "TumorCore", "EnhancingTumor"],
        "min_tumor_ratio": MIN_TUMOR_RATIO,
        "train": {"count": len(train_files),
                  "files": [str(f) for f in train_files]},
        "val":   {"count": len(val_files),
                  "files": [str(f) for f in val_files]},
        "test":  {"count": len(test_files),
                  "files": [str(f) for f in test_files]},
    }
    with open(SPLIT_JSON, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"\n  BraTS split info saved → {SPLIT_JSON}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SECTION 3.2 v2 — BRATS PREPROCESSING SELF TEST")
    print("="*60)

    # Quick augmentation test on one file
    import torch
    split_path = Path("D:/himanshu/3d_processed/brats_split_info.json")
    if split_path.exists():
        with open(split_path) as f:
            info = json.load(f)
        test_file = Path(info["train"]["files"][0])
        print(f"\n  Testing augmentation on: {test_file.name}")

        result = process_h5_slice(test_file)
        image  = result["image"]
        mask   = result["mask"]

        dataset = BraTSDataset([test_file], split="train", augment=True)
        aug_img, aug_mask = dataset[0]

        print(f"  Original image range : {image.min():.3f} - {image.max():.3f}")
        print(f"  Augmented image range: {aug_img.min():.3f} - {aug_img.max():.3f}")
        print(f"  Image shape  : {aug_img.shape}")
        print(f"  Mask shape   : {aug_mask.shape}")
        print(f"  Mask unique  : {aug_mask.unique().tolist()}")
        print(f"\n  [OK] Augmentation pipeline working!")
        print(f"  Augmentations: elastic, gamma, noise, blur, modality dropout")
    else:
        print("  Run section3_2_3d_preprocessing.py first to generate split JSON")

    print("\n  SECTION 3.2 v2 COMPLETE — Ready for Stage 2 retraining")
    print("="*60 + "\n")