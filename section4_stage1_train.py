"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 4 STAGE 1: TRAINING
=============================================================
Trains EfficientNet-B4 classifier on 2D brain tumor images.

Per document Section 4 Stage 1:
  - Dataset   : D:/himanshu/2d/classified_dataset/
  - Classes   : glioma, meningioma, notumor, pituitary
  - Epochs    : up to 100 (early stopping patience=10)
  - Optimizer : AdamW (lr=1e-4, weight_decay=1e-4)
  - Scheduler : CosineAnnealingLR (T_max=50)
  - Loss      : CrossEntropy + Focal
  - Logging   : TensorBoard
  - Checkpoint: Best val accuracy saved

Usage:
  python section4_stage1_train.py
  tensorboard --logdir D:/himanshu/runs/stage1
=============================================================
"""

import os
import time
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from section3_1_2d_preprocessing import (
    collect_image_paths,
    split_dataset,
    build_dataloaders,
    CLASS_NAMES,
)
from section2_architecture import get_model, MultiTaskLoss

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR       = Path("D:/himanshu/2d/classified_dataset")
CHECKPOINT_DIR = Path("D:/himanshu/checkpoints/stage1")
LOG_DIR        = Path("D:/himanshu/runs/stage1")

NUM_EPOCHS  = 100
LR          = 1e-4
WEIGHT_DECAY= 1e-4
T_MAX       = 50
EARLY_STOP  = 10
BATCH_SIZE  = 32
NUM_WORKERS = 0
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"


# ══════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════
def compute_accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item()


def compute_per_class_accuracy(logits, labels, num_classes=4):
    preds = logits.argmax(dim=1)
    out = {}
    for c in range(num_classes):
        mask = (labels == c)
        if mask.sum() == 0:
            out[CLASS_NAMES[c]] = 0.0
        else:
            out[CLASS_NAMES[c]] = (preds[mask] == labels[mask]).float().mean().item()
    return out


# ══════════════════════════════════════════════════════════════
# TRAIN ONE EPOCH
# ══════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = total_acc = 0.0

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [Train]", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs  = model(images)
        loss, ld = criterion(outputs, cls_labels=labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        acc         = compute_accuracy(outputs["cls_logits"], labels)
        total_loss += ld["total_loss"]
        total_acc  += acc
        pbar.set_postfix(loss=f"{ld['total_loss']:.4f}", acc=f"{acc:.4f}")

    n = len(loader)
    return total_loss / n, total_acc / n


# ══════════════════════════════════════════════════════════════
# VALIDATE ONE EPOCH
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def val_epoch(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = total_acc = 0.0
    all_logits, all_labels = [], []

    pbar = tqdm(loader, desc=f"  Epoch {epoch:03d} [Val]  ", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs  = model(images)
        loss, ld = criterion(outputs, cls_labels=labels)
        acc         = compute_accuracy(outputs["cls_logits"], labels)
        total_loss += ld["total_loss"]
        total_acc  += acc
        all_logits.append(outputs["cls_logits"].cpu())
        all_labels.append(labels.cpu())

    n          = len(loader)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    per_cls    = compute_per_class_accuracy(all_logits, all_labels)
    return total_loss / n, total_acc / n, per_cls


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def train():
    print("\n" + "="*60)
    print("  BRAIN TUMOR DETECTION - SECTION 4 STAGE 1 TRAINING")
    print("="*60)
    print(f"  Device     : {DEVICE}")
    print(f"  Epochs     : {NUM_EPOCHS}")
    print(f"  LR         : {LR}")
    print(f"  Batch size : {BATCH_SIZE}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ── Data ──
    print("\n  Loading data...")
    all_paths, all_labels = collect_image_paths(DATA_DIR)
    (train_paths, train_labels,
     val_paths,   val_labels,
     test_paths,  test_labels) = split_dataset(all_paths, all_labels)

    train_loader, val_loader, _, _, _, _ = build_dataloaders(
        train_paths, train_labels,
        val_paths,   val_labels,
        test_paths,  test_labels
    )

    # ── Model ──
    print("\n  Building model...")
    model = get_model(stage=1, in_channels=3,
                      pretrained=True, device=DEVICE)

    # ── Loss, Optimizer, Scheduler ──
    criterion = MultiTaskLoss(lambda_ce=1.0, lambda_dice=0.0, lambda_focal=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=T_MAX, eta_min=1e-6
    )

    # ── TensorBoard ──
    writer = SummaryWriter(log_dir=str(LOG_DIR))
    print(f"  TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    best_val_acc = 0.0
    best_epoch   = 0
    no_improve   = 0
    history      = []

    print("="*60)
    print("  Starting Training")
    print("="*60)

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc          = train_epoch(model, train_loader, criterion, optimizer, DEVICE, epoch)
        val_loss,   val_acc,   per_cls = val_epoch(model, val_loader, criterion, DEVICE, epoch)

        scheduler.step()
        lr = scheduler.get_last_lr()[0]

        # TensorBoard
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val",   val_loss,   epoch)
        writer.add_scalar("Acc/Train",  train_acc,  epoch)
        writer.add_scalar("Acc/Val",    val_acc,    epoch)
        writer.add_scalar("LR",         lr,         epoch)
        for k, v in per_cls.items():
            writer.add_scalar(f"PerClass/{k}", v, epoch)

        elapsed = time.time() - t0
        print(
            f"  [{epoch:03d}/{NUM_EPOCHS}] "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val loss={val_loss:.4f} acc={val_acc:.4f} | "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )
        cls_str = " ".join(f"{k.replace('1','')}={v:.3f}" for k, v in per_cls.items())
        print(f"           per-class: {cls_str}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(),
                "val_acc":     val_acc,
                "val_loss":    val_loss,
                "per_cls_acc": per_cls,
            }, CHECKPOINT_DIR / "stage1_best.pth")
            print(f"           ** NEW BEST: val_acc={val_acc:.4f} — checkpoint saved **")
        else:
            no_improve += 1

        # Save latest
        torch.save({
            "epoch":       epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
            "sched_state": scheduler.state_dict(),
            "val_acc":     val_acc,
        }, CHECKPOINT_DIR / "stage1_latest.pth")

        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "train_acc": train_acc, "val_loss": val_loss,
            "val_acc": val_acc, "lr": lr,
        })

        # Early stopping
        if no_improve >= EARLY_STOP:
            print(f"\n  Early stopping: no improvement for {EARLY_STOP} epochs")
            break

    with open(CHECKPOINT_DIR / "stage1_history.json", "w") as f:
        json.dump(history, f, indent=2)

    writer.close()

    print("\n" + "="*60)
    print("  STAGE 1 TRAINING COMPLETE")
    print("="*60)
    print(f"  Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Checkpoint        : {CHECKPOINT_DIR}/stage1_best.pth")
    print(f"  Next: Stage 2 — BraTS fine-tuning + segmentation")
    print("="*60 + "\n")

    return best_val_acc


if __name__ == "__main__":
    train()