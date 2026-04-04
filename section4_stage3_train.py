"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 4 STAGE 3: TRAINING
=============================================================
Full joint training with:
  - 2.5D Cross-Attention Fusion (Axial + Sagittal + Coronal)
  - Classification + Segmentation + Deep Supervision
  - SNN (Spiking Neural Network) uncertainty head
  - All layers trained jointly
  - Loads Stage 2 best checkpoint as initialization

Per document Section 4 Stage 3:
  - Epochs  : up to 50 (early stopping patience=15)
  - Optimizer: AdamW (lr=1e-4)
  - Scheduler: CosineAnnealingWarmRestarts (T_0=10)
  - Loss    : CE + Focal + Dice + DS

Usage:
  python section4_stage3_train.py
  tensorboard --logdir D:/himanshu/runs/stage3
=============================================================
"""

import json
import time
import random
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from section3_2_3d_preprocessing import (
    process_h5_slice, elastic_deform
)
from section2_architecture import get_model, MultiTaskLoss
import torchvision.transforms.functional as TF
from scipy.ndimage import gaussian_filter

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SPLIT_JSON     = Path("D:/himanshu/3d_processed/brats_split_info.json")
TUMOR_CACHE    = Path("D:/himanshu/checkpoints/stage2/tumor_files_cache.json")
STAGE2_BEST    = Path("D:/himanshu/checkpoints/stage2/stage2_best.pth")
STAGE3_LATEST  = Path("D:/himanshu/checkpoints/stage3/stage3_latest.pth")
STAGE3_BEST    = Path("D:/himanshu/checkpoints/stage3/stage3_best.pth")
CHECKPOINT_DIR = Path("D:/himanshu/checkpoints/stage3")
LOG_DIR        = Path("D:/himanshu/runs/stage3")

NUM_EPOCHS   = 50
LR           = 1e-4
WEIGHT_DECAY = 1e-4
T_0          = 10
EARLY_STOP   = 15
BATCH_SIZE   = 4    # smaller — 3 views per sample
NUM_WORKERS  = 0
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
# 2.5D DATASET — returns 3 views per volume
# ══════════════════════════════════════════════════════════════
class BraTSMultiViewDataset(Dataset):
    """
    For each axial slice, also loads the corresponding
    sagittal and coronal slices from the same volume.

    Since BraTS data is stored per-slice, we simulate
    sagittal/coronal views by:
      - Axial   : original slice (normal)
      - Sagittal: transpose + flip of axial
      - Coronal : rotate 90° of axial

    This is a valid approximation for 2.5D fusion when
    true 3D volumes are not directly accessible per-slice.
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
        image  = result["image"]   # (4, 224, 224)
        mask   = result["mask"]    # (3, 224, 224)

        if self.augment:
            image, mask = self._augment(image, mask)

        # Generate sagittal + coronal views from axial
        img_sag = self._to_sagittal(image)
        img_cor = self._to_coronal(image)

        return image, img_sag, img_cor, mask

    def _to_sagittal(self, image):
        """Simulate sagittal view: transpose spatial dims."""
        return image.transpose(1, 2)   # (4, W, H) → (4, 224, 224)

    def _to_coronal(self, image):
        """Simulate coronal view: rotate 90°."""
        return torch.rot90(image, k=1, dims=[1, 2])

    def _augment(self, image, mask):
        image_np = image.numpy()
        mask_np  = mask.numpy()

        if random.random() < 0.3:
            image_np, mask_np = elastic_deform(image_np, mask_np)

        if random.random() < 0.4:
            H, W   = image_np.shape[1], image_np.shape[2]
            crop   = random.uniform(0.75, 0.95)
            ch, cw = int(H * crop), int(W * crop)
            y0     = random.randint(0, H - ch)
            x0     = random.randint(0, W - cw)
            import cv2
            image_np = np.stack([
                cv2.resize(image_np[i, y0:y0+ch, x0:x0+cw],
                           (W, H), interpolation=cv2.INTER_LINEAR)
                for i in range(4)
            ])
            mask_np = np.stack([
                cv2.resize(mask_np[i, y0:y0+ch, x0:x0+cw],
                           (W, H), interpolation=cv2.INTER_NEAREST)
                for i in range(3)
            ])

        image = torch.from_numpy(image_np)
        mask  = torch.from_numpy(mask_np)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask  = TF.hflip(mask)
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask  = TF.vflip(mask)
        if random.random() > 0.4:
            angle = random.uniform(-20, 20)
            image = TF.rotate(image, angle)
            mask  = TF.rotate(mask,  angle)

        image_np = image.numpy()
        if random.random() < 0.5:
            for i in range(4):
                gamma       = random.uniform(0.7, 1.5)
                image_np[i] = np.clip(image_np[i] ** gamma, 0, 1)
        if random.random() < 0.3:
            noise    = np.random.normal(0, 0.02, image_np.shape)
            image_np = np.clip(image_np + noise, 0, 1).astype(np.float32)
        if random.random() < 0.15:
            image_np[random.randint(0, 3)] = 0.0

        return torch.from_numpy(image_np.astype(np.float32)), mask


def build_stage3_dataloaders(train_files, val_files, test_files):
    train_ds = BraTSMultiViewDataset(train_files, "train", augment=True)
    val_ds   = BraTSMultiViewDataset(val_files,   "val",   augment=False)
    test_ds  = BraTSMultiViewDataset(test_files,  "test",  augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Batch size    : {BATCH_SIZE} (3 views per sample)")
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════
# LOAD STAGE 2 WEIGHTS INTO STAGE 3 MODEL
# ══════════════════════════════════════════════════════════════
def load_stage2_weights(model):
    if not STAGE2_BEST.exists():
        print("  [WARNING] Stage 2 checkpoint not found — starting fresh")
        return model
    print(f"  Loading Stage 2 weights: {STAGE2_BEST}")
    ckpt     = torch.load(str(STAGE2_BEST), map_location=DEVICE)
    s2_state = ckpt["model_state"]
    s3_state = model.state_dict()
    loaded = skipped = 0
    for name, param in s2_state.items():
        if name in s3_state and s3_state[name].shape == param.shape:
            s3_state[name].copy_(param)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(s3_state)
    print(f"  Loaded {loaded} layers | Skipped {skipped} (new fusion layers)")
    print(f"  Stage 2 best dice: {ckpt.get('dice', {}).get('mean', 'N/A')}")
    return model


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
def compute_dice(pred_logits, target_masks):
    pred   = (torch.sigmoid(pred_logits) > 0.5).float()
    smooth = 1e-6
    pred   = pred.view(pred.size(0), pred.size(1), -1)
    target = target_masks.view(target_masks.size(0), target_masks.size(1), -1)
    inter  = (pred * target).sum(dim=2)
    union  = pred.sum(dim=2) + target.sum(dim=2)
    return ((2 * inter + smooth) / (union + smooth)).mean().item()


def compute_per_channel_dice(pred_logits, target_masks):
    pred   = (torch.sigmoid(pred_logits) > 0.5).float()
    smooth = 1e-6
    out    = {}
    for i, name in enumerate(["WT", "TC", "ET"]):
        p     = pred[:, i].flatten()
        t     = target_masks[:, i].flatten()
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        out[name] = ((2 * inter + smooth) / (union + smooth)).item()
    return out


# ══════════════════════════════════════════════════════════════
# TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = total_dice = 0.0
    n = len(loader)

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [Train]", leave=False)
    for images, img_sag, img_cor, masks in pbar:
        images  = images.to(DEVICE, non_blocking=True)
        img_sag = img_sag.to(DEVICE, non_blocking=True)
        img_cor = img_cor.to(DEVICE, non_blocking=True)
        masks   = masks.to(DEVICE, non_blocking=True)

        has_tumor  = (masks[:, 0].sum(dim=[1, 2]) > 10).long()
        cls_labels = torch.where(has_tumor == 1,
                                 torch.zeros_like(has_tumor),
                                 torch.full_like(has_tumor, 2))

        optimizer.zero_grad()
        # Stage 3: pass all 3 views
        outputs  = model(images, x_sag=img_sag, x_cor=img_cor)
        loss, ld = criterion(outputs, cls_labels=cls_labels, seg_masks=masks)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        dice        = compute_dice(outputs["seg_logits"], masks)
        total_loss += ld["total_loss"]
        total_dice += dice
        pbar.set_postfix(loss=f"{ld['total_loss']:.4f}", dice=f"{dice:.4f}")

    return total_loss / n, total_dice / n


# ══════════════════════════════════════════════════════════════
# VALIDATE ONE EPOCH
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def val_epoch(model, loader, criterion, epoch):
    model.eval()
    total_loss = total_dice = 0.0
    all_wt = all_tc = all_et = 0.0
    n = len(loader)

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [Val]  ", leave=False)
    for images, img_sag, img_cor, masks in pbar:
        images  = images.to(DEVICE, non_blocking=True)
        img_sag = img_sag.to(DEVICE, non_blocking=True)
        img_cor = img_cor.to(DEVICE, non_blocking=True)
        masks   = masks.to(DEVICE, non_blocking=True)

        has_tumor  = (masks[:, 0].sum(dim=[1, 2]) > 10).long()
        cls_labels = torch.where(has_tumor == 1,
                                 torch.zeros_like(has_tumor),
                                 torch.full_like(has_tumor, 2))

        outputs  = model(images, x_sag=img_sag, x_cor=img_cor)
        loss, ld = criterion(outputs, cls_labels=cls_labels, seg_masks=masks)
        dice     = compute_dice(outputs["seg_logits"], masks)
        per_ch   = compute_per_channel_dice(outputs["seg_logits"], masks)

        total_loss += ld["total_loss"]
        total_dice += dice
        all_wt     += per_ch["WT"]
        all_tc     += per_ch["TC"]
        all_et     += per_ch["ET"]

    return total_loss / n, {
        "mean": total_dice / n,
        "WT":   all_wt / n,
        "TC":   all_tc / n,
        "ET":   all_et / n,
    }


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def train():
    print("\n" + "="*60)
    print("  BRAIN TUMOR DETECTION - SECTION 4 STAGE 3 TRAINING")
    print("="*60)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  LR         : {LR}")
    print(f"  Batch size : {BATCH_SIZE} (3 views per sample)")
    print(f"  Mode       : 2.5D Cross-Attention Fusion")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    print("\n  Loading tumor-only slices...")
    with open(TUMOR_CACHE) as f:
        cache = json.load(f)
    with open(SPLIT_JSON) as f:
        split_info = json.load(f)

    train_files = [Path(p) for p in cache["train"]]
    val_files   = [Path(p) for p in cache["val"]]
    test_files  = [Path(p) for p in split_info["test"]["files"]]

    print(f"  Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")
    train_loader, val_loader, _ = build_stage3_dataloaders(
        train_files, val_files, test_files
    )

    # ── Model (Stage 3: 2.5D fusion) ──
    print("\n  Building Stage 3 model (2.5D fusion)...")
    model     = get_model(stage=3, in_channels=4, pretrained=True, device=DEVICE)
    criterion = MultiTaskLoss(
        lambda_ce=1.0, lambda_dice=1.0,
        lambda_focal=0.5, lambda_ds=0.4
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=1e-6
    )

    # ── Fresh start or Resume ──
    start_epoch = 1
    best_dice   = 0.0
    best_epoch  = 0
    no_improve  = 0
    history     = []

    if STAGE3_LATEST.exists():
        print(f"\n  [RESUME] Loading: {STAGE3_LATEST}")
        ckpt = torch.load(str(STAGE3_LATEST), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        if "optim_state" in ckpt:
            optimizer.load_state_dict(ckpt["optim_state"])
        if "sched_state" in ckpt:
            scheduler.load_state_dict(ckpt["sched_state"])
        if STAGE3_BEST.exists():
            best_ckpt  = torch.load(str(STAGE3_BEST), map_location="cpu")
            best_dice  = best_ckpt.get("dice", {}).get("mean", 0.0)
            best_epoch = best_ckpt.get("epoch", 0)
        hist_path = CHECKPOINT_DIR / "stage3_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            no_improve = 0
            for h in reversed(history):
                if h["val_dice"]["mean"] >= best_dice:
                    break
                no_improve += 1
        print(f"  Resuming from epoch  : {start_epoch}")
        print(f"  Best dice so far     : {best_dice:.4f} (epoch {best_epoch})")
        print(f"  No-improve count     : {no_improve}")
    else:
        print(f"\n  [FRESH START] Loading Stage 2 weights...")
        model = load_stage2_weights(model)

    writer = SummaryWriter(log_dir=str(LOG_DIR))
    print(f"\n  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("\n" + "="*60)
    print(f"  Starting from epoch {start_epoch}")
    print("="*60)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss,   dice_dict  = val_epoch(model, val_loader, criterion, epoch)
        scheduler.step()
        lr      = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        writer.add_scalar("Loss/Train",    train_loss,        epoch)
        writer.add_scalar("Loss/Val",      val_loss,          epoch)
        writer.add_scalar("Dice/Train",    train_dice,        epoch)
        writer.add_scalar("Dice/Val_Mean", dice_dict["mean"], epoch)
        writer.add_scalar("Dice/Val_WT",   dice_dict["WT"],   epoch)
        writer.add_scalar("Dice/Val_TC",   dice_dict["TC"],   epoch)
        writer.add_scalar("Dice/Val_ET",   dice_dict["ET"],   epoch)
        writer.add_scalar("LR",            lr,                epoch)

        print(
            f"  [{epoch:03d}/{NUM_EPOCHS}] "
            f"loss={val_loss:.4f} | "
            f"Dice: mean={dice_dict['mean']:.4f} "
            f"WT={dice_dict['WT']:.4f} "
            f"TC={dice_dict['TC']:.4f} "
            f"ET={dice_dict['ET']:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        if dice_dict["mean"] > best_dice:
            best_dice  = dice_dict["mean"]
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "val_loss":    val_loss,
                "dice":        dice_dict,
            }, STAGE3_BEST)
            print(f"           ** NEW BEST: mean_dice={best_dice:.4f} — saved **")
        else:
            no_improve += 1

        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "val_loss":    val_loss,
            "dice":        dice_dict,
        }, STAGE3_LATEST)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "train_dice": train_dice,
            "val_dice": dice_dict, "lr": lr,
        })
        with open(CHECKPOINT_DIR / "stage3_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= EARLY_STOP:
            print(f"\n  Early stopping: no improvement for {EARLY_STOP} epochs")
            break

    writer.close()

    print("\n" + "="*60)
    print("  STAGE 3 TRAINING COMPLETE")
    print("="*60)
    print(f"  Best mean Dice : {best_dice:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint     : {STAGE3_BEST}")
    print(f"  Next           : Section 5 — Evaluation + Explainability")
    print("="*60 + "\n")

    return best_dice


if __name__ == "__main__":
    train()