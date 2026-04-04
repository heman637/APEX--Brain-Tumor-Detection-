<div align="center">

# 🧠 APEX — Brain Tumor Detection System

### *Advanced Predictive EfficientNet-X for Brain Tumor Analysis*

[![Python](https://img.shields.io/badge/Python-3.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-red?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.9-green?style=for-the-badge&logo=nvidia)](https://developer.nvidia.com/cuda-toolkit)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**Classification Accuracy: 99.87% | Mean Dice Score: 0.849 | Hausdorff95: < 3.5mm**

[Features](#features) • [Architecture](#architecture) • [Results](#results) • [Installation](#installation) • [Usage](#usage) • [Web App](#web-app)

</div>

---

## 📌 Overview

APEX is a state-of-the-art deep learning system for **automatic brain tumor detection, classification, and segmentation** from multi-modal MRI scans. The system achieves clinical-grade performance and includes explainable AI, uncertainty estimation via Spiking Neural Networks, and a real-time clinical web dashboard.

The system is trained on two datasets:
- **22,151 brain MRI images** (Figshare + Kaggle) for 4-class tumor classification
- **BraTS 2020** (57,195 slices from 369 cases) for 3-region tumor segmentation

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎯 **4-Class Classification** | Glioma, Meningioma, No Tumor, Pituitary — 99.87% accuracy |
| 🧩 **3-Region Segmentation** | Whole Tumor, Tumor Core, Enhancing Tumor — Dice 0.849 |
| ⚡ **SNN Uncertainty** | Spiking Neural Network measures prediction uncertainty via LIF neuron spike variance |
| 🔗 **Attention Gates** | Focus decoder on tumor-relevant regions via attention-gated skip connections |
| 📐 **Deep Supervision** | Multi-scale loss at 3 decoder levels for improved gradient flow |
| 🌐 **2.5D Fusion** | Cross-attention fusion of axial, sagittal, coronal views |
| 🔬 **Grad-CAM++** | Visual explanation of model decisions |
| 📋 **Clinical Reports** | Automated clinical report generation with tumor volumes |
| 🌍 **Web Dashboard** | Professional Flask web app for real-time inference |
| 📦 **Volume Inference** | Process 200 MRI slices per volume in one batch |

---

## 🏗️ Architecture

```
Input MRI (224×224)
       │
       ▼
┌─────────────────────────────────────────────────────────┐
│              EfficientNet-B4 Encoder                    │
│   Pretrained ImageNet | 5-scale feature extraction      │
│   stage0(48ch) → stage2(32ch) → stage3(56ch)           │
│   → stage5(160ch) → stage8(1792ch)                     │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
┌─────────────┐  ┌─────────────┐  ┌──────────────┐
│Classification│  │  Attention  │  │     SNN      │
│    Head     │  │ U-Net Decoder│  │ Uncertainty  │
│  (4 classes)│  │(Skip+Attn)  │  │  Estimator   │
│   ANN-based │  │ 3 seg masks │  │ LIF neurons  │
└─────────────┘  └─────────────┘  └──────────────┘
        │                │                │
        ▼                ▼                ▼
  Tumor Type      Segmentation      Uncertainty
  + Confidence    (WT, TC, ET)      Score [0-1]
```

### Key Components

**EfficientNet-B4 Encoder**
- Pretrained on ImageNet-1K (18.5M params)
- Modified for 4-channel MRI input (T1, T1ce, T2, FLAIR)
- Extracts skip connections at 5 spatial scales

**Attention-Gated U-Net Decoder**
- Skip connections at all 5 encoder scales
- Attention gates suppress background activations
- Deep supervision at 28×28 and 56×56 decoder outputs

**SNN Uncertainty Estimator**
- Leaky Integrate-and-Fire (LIF) neurons via spikingjelly
- Runs T=8 time steps on extracted features
- Spike rate variance → uncertainty score
- High variance = uncertain prediction, Low variance = confident

**2.5D Cross-Attention Fusion**
- Projects axial, sagittal, coronal features to 512-dim
- 8-head multi-head attention across views
- Captures inter-planar anatomical context

---

## 📊 Results

### Classification (Stage 1)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Glioma | 0.998 | 0.997 | 0.998 |
| Meningioma | 0.997 | 0.994 | 0.996 |
| No Tumor | 0.999 | 1.000 | 0.999 |
| Pituitary | 0.999 | 1.000 | 0.999 |
| **Overall** | — | — | **99.87%** |

### Segmentation — BraTS 2020 (Stage 2)

| Region | Dice ↑ | Hausdorff95 ↓ | Sensitivity ↑ | Specificity ↑ |
|---|---|---|---|---|
| Whole Tumor (WT) | 0.843 | 3.38 mm | 0.858 | 0.999 |
| Tumor Core (TC) | 0.844 | 2.55 mm | 0.847 | 0.997 |
| Enhancing Tumor (ET) | 0.860 | 1.27 mm | 0.886 | 0.999 |
| **Mean** | **0.849** | **2.40 mm** | **0.864** | **0.998** |

### Ablation Study

| Configuration | WT | TC | ET | Mean |
|---|---|---|---|---|
| Baseline (no skip connections) | 0.609 | 0.617 | 0.514 | 0.580 |
| + Skip connections | 0.739 | 0.736 | 0.726 | 0.734 |
| + Attention gates | 0.780 | 0.775 | 0.760 | 0.772 |
| + Deep supervision | 0.810 | 0.808 | 0.832 | 0.817 |
| + Full augmentation | 0.822 | 0.836 | 0.858 | 0.839 |
| **+ 2.5D Fusion (Full)** | **0.843** | **0.844** | **0.860** | **0.849** |

---

## 🗂️ Project Structure

```
APEX--Brain-Tumor-Detection/
│
├── section2_architecture.py          # Model architecture (EfficientNet + U-Net + SNN)
├── section3_1_2d_preprocessing.py    # 2D MRI preprocessing + augmentation
├── section3_2_3d_preprocessing.py    # BraTS H5 preprocessing + augmentation
├── section3_2_volume_preprocessing.py# Volume-level processing (200 slices/batch)
├── section4_stage1_train.py          # Stage 1: 2D classification training
├── section4_stage2_train.py          # Stage 2: BraTS segmentation training
├── section4_stage3_train.py          # Stage 3: 2.5D fusion training
├── section5_evaluate.py              # Full evaluation (Dice, H95, sensitivity)
├── section5_gradcam.py               # Grad-CAM++ explainability
├── section6_report.py                # Clinical report generator
├── section7_inference.py             # Single image inference demo
├── app.py                            # Flask web application
├── templates/
│   └── index.html                    # Clinical dashboard UI
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.13
- CUDA 12.x (GPU recommended)
- 8GB+ GPU VRAM

### Setup

```bash
# Clone the repository
git clone https://github.com/heman637/APEX--Brain-Tumor-Detection-.git
cd APEX--Brain-Tumor-Detection-

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install all dependencies
pip install monai h5py nibabel SimpleITK grad-cam spikingjelly opencv-python Pillow scipy scikit-image scikit-learn pandas tqdm tensorboard efficientnet-pytorch flask flask-cors
```

---

## 📂 Dataset Setup

### Classification Dataset
Download from [Kaggle Brain MRI](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) and place at:
```
D:/himanshu/2d/classified_dataset/
├── glioma1/
├── meningioma1/
├── notumor1/
└── pituitary1/
```

### BraTS 2020 Dataset
Download from [BraTS 2020](https://www.med.upenn.edu/cbica/brats2020/) and convert to H5 slices:
```
D:/himanshu/3d/BraTS/BraTS2020_training_data/content/data/
├── volume_1_slice_0.h5
├── volume_1_slice_1.h5
└── ...
```

---

## 🚀 Usage

### Training Pipeline

```bash
# Stage 1: Train 2D classifier (99.87% accuracy)
python section4_stage1_train.py

# Stage 2: Train segmentation decoder (Dice 0.849)
python section4_stage2_train.py

# Stage 3: Train 2.5D fusion
python section4_stage3_train.py
```

### Evaluation

```bash
# Full evaluation with all metrics
python section5_evaluate.py

# Generate Grad-CAM++ visualizations
python section5_gradcam.py
```

### Inference

```bash
# 2D image classification
python section7_inference.py --input path/to/image.jpg

# 3D H5 slice full pipeline
python section7_inference.py --input path/to/slice.h5

# Demo mode
python section7_inference.py --demo
```

---

## 🌍 Web App

```bash
python app.py
```

Open `http://localhost:5000`

**Features:**
- Drag & drop MRI upload (2D: JPG/PNG, 3D: H5)
- Real-time classification with confidence bars
- Segmentation overlay with WT, TC, ET masks
- SNN uncertainty score (VERY LOW → VERY HIGH)
- Downloadable clinical report
- Volume-level inference (200 slices)

---

## 🧪 Key Technical Details

| Component | Detail |
|---|---|
| Backbone | EfficientNet-B4 (ImageNet pretrained) |
| Decoder | Attention-Gated U-Net (5-scale skip connections) |
| Uncertainty | SNN with LIF neurons (T=8 time steps) |
| Loss | CE + Focal (γ=2) + Dice + Deep supervision |
| Optimizer | AdamW (lr=3e-4, weight decay=1e-4) |
| Scheduler | CosineAnnealingWarmRestarts (T₀=10) |
| Augmentation | Elastic deform, gamma, noise, modality dropout |
| Training data | 22,151 (2D) + 12,164 tumor slices (BraTS) |
| Batch size | 32 (Stage 1), 8 (Stage 2), 4 (Stage 3) |

---

## 📖 Citation

```bibtex
@article{himanshu2026apex,
  title     = {APEX: Brain Tumor Detection and Segmentation Using 2.5D
               Multi-View Cross-Attention Fusion with EfficientNet-B4},
  author    = {Himanshu},
  year      = {2026},
  journal   = {Preprint},
  note      = {Classification: 99.87\%, Segmentation Mean Dice: 0.849}
}
```

---

## ⚠️ Disclaimer

This system is intended for **research and clinical decision support only**. It does not replace the judgment of qualified radiologists or treating physicians. All AI predictions must be verified by medical professionals before clinical use.

---

<div align="center">

Built with ❤️ for the IIITGwalior  Hacksagon 2026

**[⭐ Star this repo](https://github.com/heman637/APEX--Brain-Tumor-Detection-)** if you find it useful!

</div>
