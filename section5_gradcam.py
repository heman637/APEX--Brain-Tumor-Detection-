"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 5b: GRAD-CAM++ EXPLAINABILITY
=============================================================
Generates Grad-CAM++ heatmaps showing what the model focuses on.

Outputs:
  D:/himanshu/results/gradcam/
    ├── cls_gradcam_glioma_N.png
    ├── cls_gradcam_meningioma_N.png
    ├── cls_gradcam_notumor_N.png
    ├── cls_gradcam_pituitary_N.png
    └── seg_gradcam_N.png

Usage:
  python section5_gradcam.py
=============================================================
"""

import json
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

warnings.filterwarnings("ignore")

from section3_1_2d_preprocessing import (
    collect_image_paths, split_dataset,
    build_dataloaders, CLASS_NAMES,
    get_transforms
)
from section3_2_3d_preprocessing import process_h5_slice
from section2_architecture import get_model

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR_2D  = Path("D:/himanshu/2d/classified_dataset")
SPLIT_JSON   = Path("D:/himanshu/3d_processed/brats_split_info.json")
TUMOR_CACHE  = Path("D:/himanshu/checkpoints/stage2/tumor_files_cache.json")
STAGE1_CKPT  = Path("D:/himanshu/checkpoints/stage1/stage1_best.pth")
STAGE2_CKPT  = Path("D:/himanshu/checkpoints/stage2/stage2_best.pth")
OUTPUT_DIR   = Path("D:/himanshu/results/gradcam")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

# ImageNet normalization values (used during training)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ══════════════════════════════════════════════════════════════
# GRAD-CAM++ IMPLEMENTATION
# ══════════════════════════════════════════════════════════════
class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation.
    Produces better localization than standard Grad-CAM.

    Works by:
    1. Forward pass through model
    2. Compute gradients of target class w.r.t. feature maps
    3. Weight feature maps by second-order gradient importance
    4. Generate heatmap from weighted sum
    """

    def __init__(self, model, target_layer):
        self.model        = model
        self.target_layer = target_layer
        self.gradients    = None
        self.activations  = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """
        Generate Grad-CAM++ heatmap.

        Args:
          input_tensor : (1, C, H, W)
          target_class : class index to explain (None = predicted class)

        Returns:
          heatmap : (H, W) numpy array, values in [0, 1]
          pred_class : predicted class index
        """
        self.model.eval()
        self.model.zero_grad()

        output = self.model(input_tensor)
        cls_logits = output["cls_logits"]   # (1, num_classes)

        if target_class is None:
            target_class = cls_logits.argmax(dim=1).item()

        # Backward pass for target class
        score = cls_logits[0, target_class]
        score.backward()

        # Grad-CAM++ weighting
        gradients   = self.gradients[0]    # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Second order gradient weights (Grad-CAM++ formula)
        grad_sq   = gradients ** 2
        grad_cube = gradients ** 3
        sum_act   = activations.sum(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        alpha_denom = 2 * grad_sq + sum_act * grad_cube
        alpha_denom = torch.where(
            alpha_denom != 0,
            alpha_denom,
            torch.ones_like(alpha_denom)
        )
        alpha    = grad_sq / alpha_denom

        # Weights = sum of alpha * ReLU(gradients)
        weights  = (alpha * F.relu(gradients)).sum(dim=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = (weights[:, None, None] * activations).sum(dim=0)  # (H, W)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class


# ══════════════════════════════════════════════════════════════
# VISUALIZATION HELPERS
# ══════════════════════════════════════════════════════════════
def denormalize_image(tensor):
    """Convert normalized tensor back to displayable image."""
    img = tensor.squeeze().cpu().numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)  # CHW → HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def overlay_heatmap(image_rgb, heatmap, alpha=0.4):
    """
    Overlay Grad-CAM heatmap on original image.

    image_rgb : (H, W, 3) uint8
    heatmap   : (h, w) float [0,1]

    Returns:
      overlay : (H, W, 3) uint8
    """
    H, W = image_rgb.shape[:2]

    # Resize heatmap to image size
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Apply colormap (jet: blue=low, red=high attention)
    heatmap_uint8  = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color  = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color  = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = (1 - alpha) * image_rgb.astype(float) + \
              alpha * heatmap_color.astype(float)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def save_gradcam_figure(image_rgb, heatmap, pred_class, true_class,
                        confidence, save_path, title=""):
    """Save a 3-panel figure: original | heatmap | overlay."""
    H, W = image_rgb.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (W, H))
    heatmap_uint8   = (heatmap_resized * 255).astype(np.uint8)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = overlay_heatmap(image_rgb, heatmap)

    # Create 3-panel figure
    panel = np.zeros((H, W * 3 + 20, 3), dtype=np.uint8)
    panel[:, :W]           = image_rgb
    panel[:, W+10:W*2+10]  = heatmap_color
    panel[:, W*2+20:]      = overlay

    # Add text labels
    panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
    font      = cv2.FONT_HERSHEY_SIMPLEX

    # Panel labels
    cv2.putText(panel_bgr, "Original",  (10, 20),   font, 0.5, (255,255,255), 1)
    cv2.putText(panel_bgr, "Grad-CAM++", (W+15, 20), font, 0.5, (255,255,255), 1)
    cv2.putText(panel_bgr, "Overlay",   (W*2+25, 20),font, 0.5, (255,255,255), 1)

    # Prediction info
    pred_name = CLASS_NAMES[pred_class].replace("1", "")
    true_name = CLASS_NAMES[true_class].replace("1", "")
    color     = (0, 255, 0) if pred_class == true_class else (0, 0, 255)
    cv2.putText(panel_bgr,
                f"Pred: {pred_name} ({confidence:.1%})",
                (10, H - 30), font, 0.5, color, 1)
    cv2.putText(panel_bgr,
                f"True: {true_name}",
                (10, H - 10), font, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(save_path), panel_bgr)


# ══════════════════════════════════════════════════════════════
# GENERATE CLASSIFICATION GRAD-CAM++
# ══════════════════════════════════════════════════════════════
def generate_classification_gradcam(n_samples_per_class=3):
    print("\n" + "="*60)
    print("  GRAD-CAM++ — CLASSIFICATION (Stage 1)")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = get_model(stage=1, in_channels=3,
                      pretrained=False, device=DEVICE)
    ckpt  = torch.load(str(STAGE1_CKPT), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    # Target layer: last encoder stage before pooling
    target_layer = model.encoder.stage8

    gradcam = GradCAMPlusPlus(model, target_layer)

    # Get test images
    all_paths, all_labels = collect_image_paths(DATA_DIR_2D)
    (_, _, _, _, test_paths, test_labels) = split_dataset(all_paths, all_labels)

    transform = get_transforms("val")
    saved     = {i: 0 for i in range(4)}
    total_saved = 0

    print(f"\n  Generating {n_samples_per_class} samples per class...")

    for path, true_label in zip(test_paths, test_labels):
        if saved[true_label] >= n_samples_per_class:
            continue

        # Load and preprocess image
        img_pil    = Image.open(str(path)).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

        # Generate Grad-CAM++
        cam, pred_class = gradcam.generate(img_tensor, target_class=true_label)

        # Get confidence
        with torch.no_grad():
            out        = model(img_tensor)
            probs      = F.softmax(out["cls_logits"], dim=1)
            confidence = probs[0, pred_class].item()

        # Denormalize original image
        img_rgb = denormalize_image(img_tensor)

        # Save figure
        cls_name  = CLASS_NAMES[true_label].replace("1", "")
        save_path = OUTPUT_DIR / f"cls_{cls_name}_{saved[true_label]+1}.png"
        save_gradcam_figure(
            img_rgb, cam, pred_class, true_label,
            confidence, save_path
        )

        saved[true_label] += 1
        total_saved += 1

        if all(v >= n_samples_per_class for v in saved.values()):
            break

    print(f"  Saved {total_saved} classification Grad-CAM++ images")
    print(f"  Output: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# GENERATE SEGMENTATION GRAD-CAM++
# ══════════════════════════════════════════════════════════════
def generate_segmentation_gradcam(n_samples=5):
    print("\n" + "="*60)
    print("  GRAD-CAM++ — SEGMENTATION (Stage 2)")
    print("="*60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = get_model(stage=2, in_channels=4,
                      pretrained=False, device=DEVICE)
    ckpt  = torch.load(str(STAGE2_CKPT), map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Target layer: last encoder stage
    target_layer = model.encoder.stage8

    # Load tumor test files
    with open(TUMOR_CACHE) as f:
        cache = json.load(f)
    val_files = [Path(p) for p in cache["val"]][:n_samples]

    print(f"\n  Generating Grad-CAM++ for {len(val_files)} BraTS slices...")

    for i, h5_path in enumerate(val_files):
        result     = process_h5_slice(h5_path)
        img_tensor = result["image"].unsqueeze(0).to(DEVICE)
        mask       = result["mask"].numpy()

        # Forward pass for segmentation
        model.zero_grad()
        activations_list = []
        gradients_list   = []

        def fwd_hook(m, inp, out):
            activations_list.append(out.detach())

        def bwd_hook(m, gin, gout):
            gradients_list.append(gout[0].detach())

        fwd_h = target_layer.register_forward_hook(fwd_hook)
        bwd_h = target_layer.register_full_backward_hook(bwd_hook)

        out      = model(img_tensor)
        seg_pred = out["seg_logits"]

        # Backprop through Whole Tumor channel (most important)
        wt_score = torch.sigmoid(seg_pred[:, 0]).mean()
        wt_score.backward()

        fwd_h.remove()
        bwd_h.remove()

        if not activations_list or not gradients_list:
            continue

        acts = activations_list[0][0]   # (C, H, W)
        grads = gradients_list[0][0]    # (C, H, W)

        # Standard Grad-CAM (simpler for segmentation)
        weights = grads.mean(dim=(1, 2))             # (C,)
        cam     = (weights[:, None, None] * acts).sum(dim=0)
        cam     = F.relu(cam).cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()

        # Use FLAIR modality (index 3) for visualization — best contrast
        flair_img = img_tensor[0, 3].cpu().numpy()  # (224, 224)
        flair_rgb = np.stack([flair_img] * 3, axis=-1)
        flair_rgb = (flair_rgb * 255).astype(np.uint8)

        # Create visualization
        H, W = flair_rgb.shape[:2]
        cam_resized     = cv2.resize(cam, (W, H))
        cam_uint8       = (cam_resized * 255).astype(np.uint8)
        cam_color       = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
        cam_color_rgb   = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
        overlay         = overlay_heatmap(flair_rgb, cam)

        # Ground truth mask overlay (Whole Tumor = red)
        gt_overlay      = flair_rgb.copy()
        wt_mask         = cv2.resize(mask[0], (W, H),
                                     interpolation=cv2.INTER_NEAREST)
        gt_overlay[wt_mask > 0.5, 0] = 255
        gt_overlay[wt_mask > 0.5, 1] = 0

        # Predicted mask overlay (green)
        with torch.no_grad():
            pred_mask = (torch.sigmoid(
                model(img_tensor)["seg_logits"]
            ) > 0.5).cpu().numpy()[0]

        pred_overlay    = flair_rgb.copy()
        pred_wt         = cv2.resize(pred_mask[0].astype(float), (W, H),
                                     interpolation=cv2.INTER_NEAREST)
        pred_overlay[pred_wt > 0.5, 1] = 255
        pred_overlay[pred_wt > 0.5, 0] = 0

        # 5-panel figure
        gap    = 5
        panel  = np.zeros((H, W * 5 + gap * 4, 3), dtype=np.uint8)
        for j, img in enumerate([flair_rgb, cam_color_rgb,
                                   overlay, gt_overlay, pred_overlay]):
            panel[:, j*(W+gap):j*(W+gap)+W] = img

        panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
        font      = cv2.FONT_HERSHEY_SIMPLEX
        labels    = ["FLAIR", "Grad-CAM++", "Overlay", "GT (red)", "Pred (green)"]
        for j, lbl in enumerate(labels):
            cv2.putText(panel_bgr, lbl,
                        (j*(W+gap)+5, 20), font, 0.4,
                        (255, 255, 255), 1)

        dice_wt = float(
            (2 * (pred_wt > 0.5).sum() * (wt_mask > 0.5).sum() + 1e-6) /
            ((pred_wt > 0.5).sum() + (wt_mask > 0.5).sum() + 1e-6)
        )
        cv2.putText(panel_bgr,
                    f"WT Dice: {dice_wt:.3f}",
                    (5, H - 10), font, 0.4, (0, 255, 0), 1)

        save_path = OUTPUT_DIR / f"seg_gradcam_{i+1}.png"
        cv2.imwrite(str(save_path), panel_bgr)

    print(f"  Saved {len(val_files)} segmentation Grad-CAM++ images")
    print(f"  Output: {OUTPUT_DIR}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "="*60)
    print("  BRAIN TUMOR DETECTION — SECTION 5b: GRAD-CAM++")
    print("="*60)
    print(f"  Device     : {DEVICE}")
    print(f"  Output dir : {OUTPUT_DIR}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Classification Grad-CAM++ — 3 samples per class
    generate_classification_gradcam(n_samples_per_class=3)

    # Segmentation Grad-CAM++ — 5 BraTS slices
    generate_segmentation_gradcam(n_samples=5)

    print("\n" + "="*60)
    print("  GRAD-CAM++ COMPLETE")
    print("="*60)
    print(f"  All images saved to: {OUTPUT_DIR}")
    print(f"  Classification: 12 images (3 per class)")
    print(f"  Segmentation  :  5 images (FLAIR + CAM + overlay + GT + pred)")
    print("\n  Next: Section 6 — Clinical Report Generator")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()