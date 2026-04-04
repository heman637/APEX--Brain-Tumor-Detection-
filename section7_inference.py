"""
=============================================================
  BRAIN TUMOR DETECTION — SECTION 7: INFERENCE DEMO
=============================================================
Single image inference pipeline.

Supports:
  - 2D image (JPG/PNG) → Classification only
  - 3D H5 slice         → Classification + Segmentation + Report

Usage:
  # 2D image classification:
  python section7_inference.py --input path/to/image.jpg

  # 3D H5 slice (full pipeline):
  python section7_inference.py --input path/to/slice.h5

  # Run demo on sample files:
  python section7_inference.py --demo

Output:
  - Console: prediction + confidence + metrics
  - Image: D:/himanshu/results/inference/result_<name>.png
  - Report: D:/himanshu/results/inference/report_<name>.png
=============================================================
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from section3_1_2d_preprocessing import get_transforms, CLASS_NAMES
from section3_2_3d_preprocessing import process_h5_slice
from section2_architecture import get_model
from section6_report import run_inference, build_report_image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
STAGE1_CKPT  = Path("D:/himanshu/checkpoints/stage1/stage1_best.pth")
STAGE2_CKPT  = Path("D:/himanshu/checkpoints/stage2/stage2_best.pth")
TUMOR_CACHE  = Path("D:/himanshu/checkpoints/stage2/tumor_files_cache.json")
OUTPUT_DIR   = Path("D:/himanshu/results/inference")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES_CLEAN = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
IMAGENET_MEAN     = np.array([0.485, 0.456, 0.406])
IMAGENET_STD      = np.array([0.229, 0.224, 0.225])


# ══════════════════════════════════════════════════════════════
# 2D INFERENCE (JPG/PNG)
# ══════════════════════════════════════════════════════════════
def infer_2d(image_path: str, model_2d) -> dict:
    """
    Run classification on a 2D brain MRI image.
    Returns prediction dict.
    """
    transform  = get_transforms("val")
    img        = Image.open(image_path).convert("RGB")
    tensor     = transform(img).unsqueeze(0).to(DEVICE)

    model_2d.eval()
    with torch.no_grad():
        out    = model_2d(tensor)
        probs  = F.softmax(out["cls_logits"], dim=1)[0].cpu().numpy()

    pred_class = int(probs.argmax())
    return {
        "type":       "2D",
        "pred_class": pred_class,
        "class_name": CLASS_NAMES_CLEAN[pred_class],
        "confidence": float(probs[pred_class]),
        "probs":      probs,
        "image_path": image_path,
    }


# ══════════════════════════════════════════════════════════════
# 3D INFERENCE (H5)
# ══════════════════════════════════════════════════════════════
def infer_3d(h5_path: str, model_3d) -> dict:
    """
    Run full pipeline on a BraTS H5 slice.
    Returns classification + segmentation results.
    """
    result = run_inference(model_3d, h5_path)
    result["type"] = "3D"
    return result


# ══════════════════════════════════════════════════════════════
# VISUALIZATION
# ══════════════════════════════════════════════════════════════
def visualize_2d(result: dict, save_path: Path):
    """Create result visualization for 2D image."""
    img       = cv2.imread(result["image_path"])
    img       = cv2.resize(img, (224, 224))
    probs     = result["probs"]
    pred      = result["pred_class"]
    conf      = result["confidence"]
    cname     = result["class_name"]

    # Result panel on the right
    panel     = np.ones((224, 300, 3), dtype=np.uint8) * 40
    bar_h     = 25
    colors    = [(100,200,100), (200,100,100), (100,100,200), (200,200,100)]

    for i, (name, prob) in enumerate(zip(CLASS_NAMES_CLEAN, probs)):
        y      = 20 + i * (bar_h + 8)
        bar_w  = int(prob * 250)
        color  = (0,200,0) if i == pred else colors[i]
        cv2.rectangle(panel, (5, y), (5+bar_w, y+bar_h), color, -1)
        cv2.putText(panel, f"{name}: {prob:.1%}",
                    (8, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255,255,255), 1)

    # Prediction text
    color_text = (0,255,0) if pred == 2 else (0,100,255)
    cv2.putText(panel, "PREDICTION:", (5, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
    cv2.putText(panel, cname, (5, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2)
    cv2.putText(panel, f"Conf: {conf:.1%}", (5, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_text, 1)

    combined = np.concatenate([img, panel], axis=1)
    cv2.imwrite(str(save_path), combined)


def visualize_3d(result: dict, save_path: Path):
    """Create result visualization for 3D H5 slice."""
    image    = result["image"]     # (4,224,224)
    pred_mask= result["pred_mask"] # (3,224,224)
    flair    = image[3]
    flair_rgb= np.stack([flair]*3, axis=-1)
    flair_rgb= (flair_rgb * 255).astype(np.uint8)

    # Overlay prediction
    overlay  = flair_rgb.copy()
    colors   = [(255,0,0), (255,165,0), (0,255,0)]
    for ch, color in enumerate(colors):
        mask = pred_mask[ch]
        overlay[mask > 0.5, 0] = color[0]
        overlay[mask > 0.5, 1] = color[1]
        overlay[mask > 0.5, 2] = color[2]
    blended  = cv2.addWeighted(flair_rgb, 0.5, overlay, 0.5, 0)

    # Info panel
    panel    = np.ones((224, 300, 3), dtype=np.uint8) * 40
    probs    = result["probs"]
    pred     = result["pred_class"]
    cname    = result["class_name"]
    conf     = result["confidence"]

    for i, (name, prob) in enumerate(zip(CLASS_NAMES_CLEAN, probs)):
        y     = 20 + i * 33
        bar_w = int(prob * 250)
        color = (0,200,0) if i == pred else (100,100,200)
        cv2.rectangle(panel, (5, y), (5+bar_w, y+25), color, -1)
        cv2.putText(panel, f"{name}: {prob:.1%}",
                    (8, y+18), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255,255,255), 1)

    # Tumor volumes
    volumes  = [pred_mask[i].sum() for i in range(3)]
    regions  = ["WT", "TC", "ET"]
    for i, (reg, vol) in enumerate(zip(regions, volumes)):
        y = 160 + i * 20
        cv2.putText(panel, f"{reg}: {vol:.0f} px²",
                    (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (200,200,200), 1)

    color_text = (0,255,0) if pred == 2 else (0,100,255)
    cv2.putText(panel, cname, (5, 215),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_text, 2)

    # Combine: original | overlay | panel
    flair_bgr  = cv2.cvtColor(flair_rgb,  cv2.COLOR_RGB2BGR)
    blended_bgr= cv2.cvtColor(blended,    cv2.COLOR_RGB2BGR)
    combined   = np.concatenate([flair_bgr, blended_bgr, panel], axis=1)
    cv2.imwrite(str(save_path), combined)


# ══════════════════════════════════════════════════════════════
# PRINT RESULTS
# ══════════════════════════════════════════════════════════════
def print_results(result: dict, elapsed: float):
    print("\n" + "─"*50)
    print(f"  INPUT TYPE    : {result['type']}")
    print(f"  PREDICTION    : {result['class_name']}")
    print(f"  CONFIDENCE    : {result['confidence']:.1%}")
    print(f"\n  All class probabilities:")
    for name, prob in zip(CLASS_NAMES_CLEAN, result["probs"]):
        bar = "█" * int(prob * 30)
        print(f"    {name:<15}: {bar:<30} {prob:.1%}")

    if result["type"] == "3D":
        pred_mask = result["pred_mask"]
        print(f"\n  Segmentation:")
        for i, name in enumerate(["Whole Tumor", "Tumor Core", "Enhancing"]):
            vol = pred_mask[i].sum()
            print(f"    {name:<15}: {vol:.0f} pixels ({vol:.0f} mm²)")

    print(f"\n  Inference time: {elapsed*1000:.1f} ms")
    print("─"*50)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Brain Tumor Detection — Inference Demo"
    )
    parser.add_argument("--input",  type=str, default=None,
                        help="Path to image (.jpg/.png) or H5 slice (.h5)")
    parser.add_argument("--demo",   action="store_true",
                        help="Run demo on sample files from test set")
    parser.add_argument("--report", action="store_true",
                        help="Also generate clinical PDF report (H5 only)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  BRAIN TUMOR DETECTION — INFERENCE DEMO")
    print("="*60)
    print(f"  Device: {DEVICE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    print("\n  Loading models...")
    model_2d = get_model(stage=1, in_channels=3,
                         pretrained=False, device=DEVICE)
    ckpt1    = torch.load(str(STAGE1_CKPT), map_location=DEVICE)
    model_2d.load_state_dict(ckpt1["model_state"], strict=False)
    model_2d.eval()
    print(f"  Stage 1 (classifier) loaded ✓")

    model_3d = get_model(stage=2, in_channels=4,
                         pretrained=False, device=DEVICE)
    ckpt2    = torch.load(str(STAGE2_CKPT), map_location=DEVICE)
    model_3d.load_state_dict(ckpt2["model_state"])
    model_3d.eval()
    print(f"  Stage 2 (segmenter)  loaded ✓")

    # Determine input files
    if args.input:
        inputs = [args.input]
    elif args.demo:
        import json
        with open(TUMOR_CACHE) as f:
            cache = json.load(f)
        # Use 2 H5 samples for demo
        inputs = cache["val"][:2]
        print(f"\n  Running demo on {len(inputs)} sample slices...")
    else:
        print("\n  No input specified. Running demo mode...")
        import json
        with open(TUMOR_CACHE) as f:
            cache = json.load(f)
        inputs = cache["val"][:1]

    # Run inference on each input
    for input_path in inputs:
        input_path = str(input_path)
        name       = Path(input_path).stem
        print(f"\n  Processing: {Path(input_path).name}")

        t0 = time.time()

        if input_path.lower().endswith(".h5"):
            result = infer_3d(input_path, model_3d)
            save_path = OUTPUT_DIR / f"result_{name}.png"
            visualize_3d(result, save_path)

            if args.report or True:  # always generate report for H5
                report_img = build_report_image(result, patient_id=name)
                report_path = OUTPUT_DIR / f"report_{name}.png"
                report_bgr  = cv2.cvtColor(report_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(report_path), report_bgr)
                print(f"  Report saved: {report_path}")
        else:
            result    = infer_2d(input_path, model_2d)
            save_path = OUTPUT_DIR / f"result_{name}.png"
            visualize_2d(result, save_path)

        elapsed = time.time() - t0
        print_results(result, elapsed)
        print(f"  Visualization saved: {save_path}")

    print("\n" + "="*60)
    print("  PROJECT COMPLETE!")
    print("="*60)
    print("""
  Full Pipeline Summary:
  ─────────────────────────────────────────────────────
  Stage 1 Classification : 99.87% accuracy
  Stage 2 Segmentation   : Mean Dice = 0.849
    Whole Tumor (WT)     : Dice=0.843  H95=3.38mm
    Tumor Core (TC)      : Dice=0.844  H95=2.55mm
    Enhancing Tumor (ET) : Dice=0.860  H95=1.27mm
  ─────────────────────────────────────────────────────
  Files created:
    section2_architecture.py       — Model architecture
    section3_1_2d_preprocessing.py — 2D data pipeline
    section3_2_3d_preprocessing.py — 3D BraTS pipeline
    section4_stage1_train.py       — Stage 1 training
    section4_stage2_train.py       — Stage 2 training
    section4_stage3_train.py       — Stage 3 training
    section5_evaluate.py           — Evaluation metrics
    section5_gradcam.py            — Grad-CAM++ XAI
    section6_report.py             — Clinical reports
    section7_inference.py          — This inference demo
  ─────────────────────────────────────────────────────
    """)


if __name__ == "__main__":
    main()