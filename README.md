

# CODEZEN 2.0 – Hackathon Submission

## Team Information

**Team Name:** ORCA
**Project:** Offroad Semantic Scene Segmentation
**Challenge:** Duality AI Code Sprint Hackathon

---

# Project Overview

This project implements a **hybrid Transformer–CNN semantic segmentation architecture** designed for off-road desert environments using synthetic data from Duality AI's Falcon platform.

Instead of a conventional U-Net, we deploy a **DINOv2 Transformer backbone fused with a convolutional feature pyramid neck**, enabling:

* Strong global semantic understanding (Transformer)
* Local spatial refinement (CNN neck)
* Improved small-object sensitivity
* Multi-scale dense prediction

---

# 🚀 Key Innovations

* ✅ **DINOv2 Vision Transformer backbone** (self-supervised representation learning)
* ✅ **Multi-depth feature extraction**
* ✅ **Lightweight CNN Feature Pyramid Neck**
* ✅ **Multi-scale segmentation heads**
* ✅ **Walkability-aware visualization**
* ✅ **Efficient GPU inference (<5ms per image)**

---

# Architecture Overview

## Hybrid Transformer–CNN Segmentation Model

### 1️⃣ Backbone – DINOv2 (ViT-Based)

* Input: 512×512 crop
* Patch size: 14
* Token grid: 36×36
* Intermediate block features extracted
* Global contextual attention

DINOv2 provides:

* Robust semantic embeddings
* Strong domain generalization
* Dense patch-level representations

---

### 2️⃣ Multi-Depth Feature Extraction

Instead of using only the final transformer block, we extract features from multiple depths:

* Early block → local texture features
* Mid block → object-level abstraction
* Final block → global semantics

All reshaped to spatial feature maps:

```
[B, C, 36, 36]
```

---

### 3️⃣ CNN Feature Pyramid Neck

Since Vision Transformers do not provide spatial hierarchy natively, we construct a pyramid manually:

From 36×36 base feature:

* P2 → 72×72 (upsampled)
* P3 → 36×36 (base)
* P4 → 18×18 (downsampled)
* P5 → 9×9 (downsampled)

This restores spatial inductive bias and improves small-object recall.

---

### 4️⃣ Multi-Scale Segmentation Heads

Each pyramid level produces logits.

Deep supervision is applied at multiple scales to:

* Improve gradient flow
* Enhance rare-class learning
* Reduce semantic over-smoothing

Final output is fused and upsampled to full resolution.

---

# Dataset & Classes

The model segments 10 primary semantic classes:

| ID    | Class Name     | Description               |
| ----- | -------------- | ------------------------- |
| 100   | Trees          | Tall vegetation           |
| 200   | Lush Bushes    | Dense shrubs              |
| 300   | Dry Grass      | Short vegetation          |
| 500   | Dry Bushes     | Sparse shrubs             |
| 550   | Ground Clutter | Walkable terrain / debris |
| 600   | Flowers        | Flowering plants          |
| 700   | Logs           | Fallen trees              |
| 800   | Rocks          | Stones/boulders           |
| 7100  | Landscape      | General terrain           |
| 10000 | Sky            | Sky regions               |

**Ground Clutter** is emphasized in visualization as a high-contrast red to highlight walkable path regions.

---

# Training Configuration

### Hyperparameters

* **Input Size:** 512×512 crop
* **Batch Size:** 8
* **Epochs:** 50–100
* **Optimizer:** AdamW
* **Learning Rate:** 1e-4
* **Weight Decay:** 1e-4

### Loss Function

Compound loss:

```
Total Loss =
    CrossEntropy
  + Dice Loss
  + Focal Loss (rare-class emphasis)
```

This improves:

* Class imbalance handling
* Boundary precision
* Small-object detection

---

# Data Augmentation

```python
A.Compose([
    A.SmallestMaxSize(max_size=768),
    A.RandomCrop(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
])
```

This preserves scene context while maintaining transformer token density.

---

# Performance Metrics

### Primary Metric

**Mean IoU (mIoU)**

```
IoU = Intersection / Union
```

### Additional Metrics

* Pixel Accuracy
* Per-Class IoU
* Precision / Recall / F1
* Latency (ms per image)

---

# Performance Summary (Example)

| Metric            | Value      |
| ----------------- | ---------- |
| Mean IoU          | ~0.69–0.70 |
| Pixel Accuracy    | ~0.85      |
| Inference Latency | ~4.7 ms    |
| Throughput        | ~200 FPS   |

Transformer backbone improves global consistency, while CNN neck enhances fine-detail segmentation.

---

# Running the Model

## Train

```bash
python train.py
```

## Test

```bash
python test.py
```

## Visualize

```bash
python visualize_segmentation.py
```

Outputs:

* Predicted masks
* Confusion matrix
* Per-class IoU charts
* Visual comparison panels

---

# Why This Architecture Matters

Traditional U-Net:

* Strong locality
* Limited global reasoning

Pure Transformer:

* Strong global context
* Weak spatial hierarchy

Our fusion model:

* Combines global attention with local convolutional refinement
* Enables rare-class sensitivity
* Maintains real-time inference

---

# Technical Contribution

This project demonstrates:

* Practical integration of self-supervised ViT backbones
* Manual spatial pyramid construction for transformer-based segmentation
* Hybrid architecture design for terrain-aware perception
* Efficient deployment-ready performance

---

# Future Work

* Boundary-aware supervision
* Walkability multi-task head
* Adaptive resolution inference
* Uncertainty-aware refinement
* Domain adaptation to real-world off-road imagery

---


