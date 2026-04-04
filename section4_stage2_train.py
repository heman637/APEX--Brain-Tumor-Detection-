"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 4 STAGE 2: TRAINING v3
=============================================================
UPGRADES:
  - New decoder: skip connections + attention gates
  - Deep supervision loss (3 scales)
  - Full augmentation pipeline (elastic, gamma, noise etc.)
  - LR = 3e-4, uniform after unfreeze
  - CosineAnnealingWarmRestarts T_0=10
  - FREEZE_EPOCHS = 3
  - EARLY_STOP = 15

Usage:
  python section4_stage2_train.py
  tensorboard --logdir D:/himanshu/runs/stage2
=============================================================
"""

import json
import time
from pathlib import Path

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from section3_2_3d_preprocessing import build_brats_dataloaders
from section2_architecture import get_model, MultiTaskLoss

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SPLIT_JSON     = Path("D:/himanshu/3d_processed/brats_split_info.json")
TUMOR_CACHE    = Path("D:/himanshu/checkpoints/stage2/tumor_files_cache.json")
STAGE1_CKPT    = Path("D:/himanshu/checkpoints/stage1/stage1_best.pth")
STAGE2_LATEST  = Path("D:/himanshu/checkpoints/stage2/stage2_latest.pth")
STAGE2_BEST    = Path("D:/himanshu/checkpoints/stage2/stage2_best.pth")
CHECKPOINT_DIR = Path("D:/himanshu/checkpoints/stage2")
LOG_DIR        = Path("D:/himanshu/runs/stage2")

NUM_EPOCHS    = 60
LR            = 3e-4
WEIGHT_DECAY  = 1e-4
T_0           = 10
EARLY_STOP    = 15
FREEZE_EPOCHS = 3
BATCH_SIZE    = 8
NUM_WORKERS   = 0
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
def get_tumor_files():
    with open(SPLIT_JSON) as f:
        split_info = json.load(f)
    test_files = [Path(p) for p in split_info["test"]["files"]]

    if TUMOR_CACHE.exists():
        print("  Loading tumor file cache...")
        with open(TUMOR_CACHE) as f:
            cache = json.load(f)
        train_files = [Path(p) for p in cache["train"]]
        val_files   = [Path(p) for p in cache["val"]]
    else:
        print("  Filtering tumor slices (one-time scan)...")
        all_train = [Path(p) for p in split_info["train"]["files"]]
        all_val   = [Path(p) for p in split_info["val"]["files"]]

        def is_tumor(p):
            with h5py.File(str(p), "r") as f:
                return f["mask"][:, :, 0].sum() > 0

        train_files = [p for p in tqdm(all_train, desc="  Scan train") if is_tumor(p)]
        val_files   = [p for p in tqdm(all_val,   desc="  Scan val  ") if is_tumor(p)]

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        with open(TUMOR_CACHE, "w") as f:
            json.dump({
                "train": [str(p) for p in train_files],
                "val":   [str(p) for p in val_files],
            }, f, indent=2)
        print(f"  Cache saved → {TUMOR_CACHE}")

    print(f"  Train tumor slices : {len(train_files)}")
    print(f"  Val tumor slices   : {len(val_files)}")
    print(f"  Test slices        : {len(test_files)}")
    return train_files, val_files, test_files


# ══════════════════════════════════════════════════════════════
# LOAD STAGE 1 WEIGHTS
# ══════════════════════════════════════════════════════════════
def load_stage1_weights(model):
    if not STAGE1_CKPT.exists():
        print("  [WARNING] Stage 1 checkpoint not found")
        return model
    print(f"  Loading Stage 1 weights: {STAGE1_CKPT}")
    ckpt     = torch.load(str(STAGE1_CKPT), map_location=DEVICE)
    s1_state = ckpt["model_state"]
    s2_state = model.state_dict()
    loaded = skipped = 0
    for name, param in s1_state.items():
        if name in s2_state and s2_state[name].shape == param.shape:
            s2_state[name].copy_(param)
            loaded += 1
        else:
            skipped += 1
    model.load_state_dict(s2_state)
    print(f"  Loaded {loaded} layers | Skipped {skipped}")
    print(f"  Stage 1 val_acc: {ckpt.get('val_acc', 'N/A')}")
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
    for images, masks in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        has_tumor  = (masks[:, 0].sum(dim=[1, 2]) > 10).long()
        cls_labels = torch.where(has_tumor == 1,
                                 torch.zeros_like(has_tumor),
                                 torch.full_like(has_tumor, 2))

        optimizer.zero_grad()
        outputs  = model(images)
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
    for images, masks in pbar:
        images = images.to(DEVICE, non_blocking=True)
        masks  = masks.to(DEVICE, non_blocking=True)

        has_tumor  = (masks[:, 0].sum(dim=[1, 2]) > 10).long()
        cls_labels = torch.where(has_tumor == 1,
                                 torch.zeros_like(has_tumor),
                                 torch.full_like(has_tumor, 2))

        outputs  = model(images)
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
    print("  BRAIN TUMOR DETECTION - SECTION 4 STAGE 2 TRAINING v3")
    print("="*60)
    print(f"  Device        : {DEVICE}")
    print(f"  Epochs        : {NUM_EPOCHS}")
    print(f"  LR            : {LR}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Freeze epochs : {FREEZE_EPOCHS}")
    print(f"  Scheduler     : CosineAnnealingWarmRestarts (T_0={T_0})")
    print(f"  New features  : Skip connections + Attention + Deep supervision")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    print("\n  Loading BraTS tumor-only slices...")
    train_files, val_files, test_files = get_tumor_files()
    train_loader, val_loader, _, _, _, _ = build_brats_dataloaders(
        train_files, val_files, test_files
    )

    # ── Model ──
    print("\n  Building Stage 2 model (v2 architecture)...")
    model     = get_model(stage=2, in_channels=4, pretrained=True, device=DEVICE)
    criterion = MultiTaskLoss(
        lambda_ce=1.0, lambda_dice=1.0,
        lambda_focal=0.5, lambda_ds=0.4
    )
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T_0, T_mult=1, eta_min=1e-6
    )

    # ── Fresh start or Resume ──
    start_epoch      = 1
    best_dice        = 0.0
    best_epoch       = 0
    no_improve       = 0
    history          = []
    encoder_unfrozen = False

    if STAGE2_LATEST.exists():
        print(f"\n  [RESUME] Loading: {STAGE2_LATEST}")
        ckpt = torch.load(str(STAGE2_LATEST), map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt["epoch"] + 1
        if "optim_state" in ckpt:
            optimizer.load_state_dict(ckpt["optim_state"])
        if "sched_state" in ckpt:
            scheduler.load_state_dict(ckpt["sched_state"])
        if STAGE2_BEST.exists():
            best_ckpt  = torch.load(str(STAGE2_BEST), map_location="cpu")
            best_dice  = best_ckpt.get("dice", {}).get("mean", 0.0)
            best_epoch = best_ckpt.get("epoch", 0)
        hist_path = CHECKPOINT_DIR / "stage2_history.json"
        if hist_path.exists():
            with open(hist_path) as f:
                history = json.load(f)
            no_improve = 0
            for h in reversed(history):
                if h["val_dice"]["mean"] >= best_dice:
                    break
                no_improve += 1
        if start_epoch > FREEZE_EPOCHS + 1:
            encoder_unfrozen = True
        print(f"  Resuming from epoch  : {start_epoch}")
        print(f"  Best dice so far     : {best_dice:.4f} (epoch {best_epoch})")
        print(f"  No-improve count     : {no_improve}")
    else:
        print(f"\n  [FRESH START] Loading Stage 1 weights...")
        model = load_stage1_weights(model)
        model.freeze_encoder(num_layers=3)

    writer = SummaryWriter(log_dir=str(LOG_DIR))
    print(f"\n  TensorBoard: tensorboard --logdir {LOG_DIR}")
    print("\n" + "="*60)
    print(f"  Starting from epoch {start_epoch}")
    print("="*60)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        # Unfreeze all with uniform LR after FREEZE_EPOCHS
        if epoch == FREEZE_EPOCHS + 1 and not encoder_unfrozen:
            model.unfreeze_all()
            encoder_unfrozen = True
            optimizer = optim.AdamW(
                model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
            )
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=T_0, T_mult=1, eta_min=1e-6
            )
            print(f"\n  [Epoch {epoch}] Encoder unfrozen — uniform LR={LR}")

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
            }, STAGE2_BEST)
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
        }, STAGE2_LATEST)

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "train_dice": train_dice,
            "val_dice": dice_dict, "lr": lr,
        })
        with open(CHECKPOINT_DIR / "stage2_history.json", "w") as f:
            json.dump(history, f, indent=2)

        if no_improve >= EARLY_STOP:
            print(f"\n  Early stopping: no improvement for {EARLY_STOP} epochs")
            break

    writer.close()

    print("\n" + "="*60)
    print("  STAGE 2 TRAINING COMPLETE")
    print("="*60)
    print(f"  Best mean Dice : {best_dice:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint     : {STAGE2_BEST}")
    print(f"  Next           : Stage 3 — Full joint training + SNN")
    print("="*60 + "\n")

    return best_dice


if __name__ == "__main__":
    train()