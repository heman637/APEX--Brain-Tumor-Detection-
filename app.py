"""
=============================================================
  BRAIN TUMOR DETECTION — FLASK API BACKEND
=============================================================
Run:
  python app.py

Endpoints:
  GET  /              → Dashboard UI
  POST /predict/2d    → 2D image classification
  POST /predict/3d    → 3D H5 slice full pipeline
  GET  /results/<id>  → Get saved result
  GET  /report/<id>   → Download clinical report
=============================================================
"""

import os
import uuid
import json
import base64
import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify, render_template, send_file
from PIL import Image
import io

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent))

from section3_1_2d_preprocessing import get_transforms
from section3_2_3d_preprocessing import process_h5_slice
from section2_architecture import get_model
from section6_report import run_inference, build_report_image

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
STAGE1_CKPT  = Path("D:/himanshu/checkpoints/stage1/stage1_best.pth")
STAGE2_CKPT  = Path("D:/himanshu/checkpoints/stage2/stage2_best.pth")
RESULTS_DIR  = Path("D:/himanshu/results/web")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
REGION_NAMES = ["Whole Tumor", "Tumor Core", "Enhancing Tumor"]

# Model performance metrics (from evaluation)
MODEL_METRICS = {
    "classification_accuracy": 99.87,
    "segmentation": {
        "WT": {"dice": 0.843, "h95": 3.38, "sensitivity": 0.858},
        "TC": {"dice": 0.844, "h95": 2.55, "sensitivity": 0.847},
        "ET": {"dice": 0.860, "h95": 1.27, "sensitivity": 0.886},
    }
}

app = Flask(__name__)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load models at startup ──
print("Loading models...")
model_2d = get_model(stage=1, in_channels=3, pretrained=False, device=DEVICE)
ckpt1    = torch.load(str(STAGE1_CKPT), map_location=DEVICE)
model_2d.load_state_dict(ckpt1["model_state"], strict=False)
model_2d.eval()
print(f"  Stage 1 classifier loaded ({DEVICE})")

model_3d = get_model(stage=2, in_channels=4, pretrained=False, device=DEVICE)
ckpt2    = torch.load(str(STAGE2_CKPT), map_location=DEVICE)
model_3d.load_state_dict(ckpt2["model_state"])
model_3d.eval()
print(f"  Stage 2 segmenter loaded ({DEVICE})")
print("Models ready!")


# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def numpy_to_b64(img_np):
    """Convert numpy image to base64 string for JSON response."""
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).astype(np.uint8)
    _, buffer = cv2.imencode(".png", img_np)
    return base64.b64encode(buffer).decode("utf-8")


def make_segmentation_overlay(image_np, pred_mask):
    """Create FLAIR + segmentation overlay image."""
    flair     = image_np[3]  # FLAIR modality
    flair_rgb = np.stack([flair]*3, axis=-1)
    flair_rgb = (flair_rgb * 255).astype(np.uint8)

    overlay   = flair_rgb.copy()
    colors    = [(255, 50, 50), (255, 165, 0), (50, 255, 50)]  # WT, TC, ET
    for ch, color in enumerate(colors):
        mask = pred_mask[ch]
        overlay[mask > 0.5, 0] = color[0]
        overlay[mask > 0.5, 1] = color[1]
        overlay[mask > 0.5, 2] = color[2]

    blended = cv2.addWeighted(flair_rgb, 0.5, overlay, 0.5, 0)
    return flair_rgb, blended


def make_modality_grid(image_np, pred_mask):
    """Create 2x2 grid of all modalities."""
    names  = ["T1", "T1ce", "T2", "FLAIR"]
    panels = []

    for i in range(4):
        mod  = image_np[i]
        rgb  = np.stack([mod]*3, axis=-1)
        rgb  = (rgb * 255).astype(np.uint8)

        if i == 3:  # FLAIR — add overlay
            overlay = rgb.copy()
            for ch, color in enumerate([(255,50,50),(255,165,0),(50,255,50)]):
                overlay[pred_mask[ch] > 0.5, 0] = color[0]
                overlay[pred_mask[ch] > 0.5, 1] = color[1]
                overlay[pred_mask[ch] > 0.5, 2] = color[2]
            rgb = cv2.addWeighted(rgb, 0.55, overlay, 0.45, 0)

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.putText(rgb_bgr, names[i], (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        panels.append(rgb_bgr)

    row1 = np.concatenate([panels[0], panels[1]], axis=1)
    row2 = np.concatenate([panels[2], panels[3]], axis=1)
    grid = np.concatenate([row1, row2], axis=0)
    return grid


# ══════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html",
                           device=DEVICE,
                           metrics=MODEL_METRICS)


@app.route("/predict/2d", methods=["POST"])
def predict_2d():
    """2D image classification endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file    = request.files["file"]
    if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
        return jsonify({"error": "Invalid file type. Use JPG or PNG"}), 400

    try:
        # Load image
        img_bytes = file.read()
        img_pil   = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        transform = get_transforms("val")
        tensor    = transform(img_pil).unsqueeze(0).to(DEVICE)

        # Inference
        import time
        t0        = time.time()
        with torch.no_grad():
            out   = model_2d(tensor)
            probs = F.softmax(out["cls_logits"], dim=1)[0].cpu().numpy()
        elapsed   = (time.time() - t0) * 1000

        pred_class = int(probs.argmax())
        confidence = float(probs[pred_class])

        # Original image as base64
        img_np  = np.array(img_pil.resize((224, 224)))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_b64 = numpy_to_b64(img_bgr)

        # Save result
        result_id = str(uuid.uuid4())[:8]
        result    = {
            "id":          result_id,
            "type":        "2D",
            "timestamp":   datetime.datetime.now().isoformat(),
            "prediction":  CLASS_NAMES[pred_class],
            "confidence":  confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i])
                for i in range(4)
            },
            "inference_ms": elapsed,
            "device":       DEVICE,
        }

        with open(RESULTS_DIR / f"{result_id}.json", "w") as f:
            json.dump(result, f, indent=2)

        return jsonify({
            **result,
            "image_b64": img_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/3d", methods=["POST"])
def predict_3d():
    """3D H5 slice full pipeline endpoint."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if not file.filename.lower().endswith(".h5"):
        return jsonify({"error": "Invalid file type. Use .h5"}), 400

    try:
        # Save H5 temporarily
        tmp_path = RESULTS_DIR / f"tmp_{uuid.uuid4().hex}.h5"
        file.save(str(tmp_path))

        import time
        t0     = time.time()

        # Run inference
        result = run_inference(model_3d, str(tmp_path))
        elapsed = (time.time() - t0) * 1000

        # Cleanup temp file
        tmp_path.unlink()

        image_np  = result["image"]
        pred_mask = result["pred_mask"]
        true_mask = result["true_mask"]

        # Generate visualizations
        flair_rgb, seg_overlay = make_segmentation_overlay(image_np, pred_mask)
        modality_grid          = make_modality_grid(image_np, pred_mask)

        # Per-region masks as b64
        mask_images = {}
        for i, region in enumerate(["wt", "tc", "et"]):
            flair_bgr  = cv2.cvtColor(flair_rgb, cv2.COLOR_RGB2BGR)
            mask_vis   = flair_bgr.copy()
            colors_bgr = [(0,50,255), (0,165,255), (0,255,50)]
            mask_vis[pred_mask[i] > 0.5] = colors_bgr[i]
            blended = cv2.addWeighted(flair_bgr, 0.5, mask_vis, 0.5, 0)
            mask_images[region] = numpy_to_b64(blended)

        # Dice scores (vs true mask)
        dice_scores = {}
        smooth = 1e-6
        for i, region in enumerate(["WT", "TC", "ET"]):
            p = pred_mask[i].flatten()
            t = true_mask[i].flatten()
            dice = float((2*(p*t).sum() + smooth) / (p.sum() + t.sum() + smooth))
            dice_scores[region] = round(dice, 4)

        # Tumor volumes
        volumes = {
            REGION_NAMES[i]: int(pred_mask[i].sum())
            for i in range(3)
        }

        # Generate report
        result_id   = str(uuid.uuid4())[:8]
        report_img  = build_report_image(result, patient_id=f"WEB_{result_id}")
        report_bgr  = cv2.cvtColor(report_img, cv2.COLOR_RGB2BGR)
        report_path = RESULTS_DIR / f"report_{result_id}.png"
        cv2.imwrite(str(report_path), report_bgr)

        response = {
            "id":           result_id,
            "type":         "3D",
            "timestamp":    datetime.datetime.now().isoformat(),
            "prediction":   result["class_name"],
            "confidence":   result["confidence"],
            "probabilities": {
                CLASS_NAMES[i]: float(result["probs"][i])
                for i in range(4)
            },
            "segmentation": {
                "dice_scores":    dice_scores,
                "tumor_volumes":  volumes,
            },
            "inference_ms":  elapsed,
            "device":        DEVICE,
            "images": {
                "flair":         numpy_to_b64(cv2.cvtColor(flair_rgb, cv2.COLOR_RGB2BGR)),
                "segmentation":  numpy_to_b64(cv2.cvtColor(seg_overlay, cv2.COLOR_RGB2BGR)),
                "modality_grid": numpy_to_b64(modality_grid),
                "masks":         mask_images,
            },
            "report_url": f"/report/{result_id}",
        }

        # Save result JSON
        result_save = {k: v for k, v in response.items() if k != "images"}
        with open(RESULTS_DIR / f"{result_id}.json", "w") as f:
            json.dump(result_save, f, indent=2)

        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/report/<result_id>")
def get_report(result_id):
    """Download clinical report PNG."""
    report_path = RESULTS_DIR / f"report_{result_id}.png"
    if not report_path.exists():
        return jsonify({"error": "Report not found"}), 404
    return send_file(str(report_path), mimetype="image/png",
                     as_attachment=True,
                     download_name=f"brain_tumor_report_{result_id}.png")


@app.route("/metrics")
def get_metrics():
    """Return model performance metrics."""
    return jsonify(MODEL_METRICS)


@app.route("/health")
def health():
    return jsonify({
        "status":  "healthy",
        "device":  DEVICE,
        "models":  {"stage1": True, "stage2": True}
    })


if __name__ == "__main__":
    print("\n" + "="*60)
    print("  BRAIN TUMOR DETECTION — WEB SERVER")
    print("="*60)
    print(f"  Device  : {DEVICE}")
    print(f"  URL     : http://localhost:5000")
    print(f"  Results : {RESULTS_DIR}")
    print("="*60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)