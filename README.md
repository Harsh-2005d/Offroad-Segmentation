
---

# Offroad Semantic Scene Segmentation 🚙💨

> **Codezen2.0 Submission 2026**
> *Off-road Autonomy via Synthetic Data and Robust Semantic Segmentation*

![Project Banner](Visual_Report/Visual_Report_V3_0.png)

## 🚧 Problem Statement

Autonomous systems perform well in structured road environments due to clear cues like lane markings and traffic signs. In off-road settings, these cues are absent. Unmanned Ground Vehicles (UGVs) must instead rely on dense scene understanding to reason about terrain safety.

The goal of this project is to train a **semantic scene segmentation model** that classifies every pixel in an off-road environment into meaningful terrain categories. This enables UGVs to distinguish **drivable regions** such as dirt and short grass from **non-drivable obstacles** including trees, rocks, logs, and water, which is critical for safe navigation and path planning.

## 💻 Tech Stack

* **Core Framework**: PyTorch
* **Backbone Architecture**: DINOv2 (Vision Transformer)
* **Data Augmentation**: Albumentations
* **Data Processing**: NumPy, Pillow
* **Visualization & Demo**: Tkinter, Matplotlib

## 📌 Overview

This project is developed as part of the **Duality AI Offroad Autonomy Segmentation Challenge**, which focuses on training robust perception models using **synthetic data generated from digital twin environments**.

The dataset consists of annotated off-road scenes simulated using Duality AI’s digital twin platform. These environments are designed to mimic real-world desert and off-road conditions while enabling controlled variation in terrain, lighting, and environmental factors.

The objective is not only to achieve high segmentation accuracy on known scenes, but also to **evaluate generalization** on novel yet similar environments, reflecting real-world deployment challenges in off-road autonomy.

## 🧠 Model Architecture: Transformer-Based Semantic Segmentation

Rather than training a model from scratch, we adopt a **transfer learning strategy** to leverage pretrained visual representations and improve robustness under domain shifts.

### 1. Backbone: DINOv2 (Vision Transformer)

We use **DINOv2**, a self-supervised Vision Transformer pretrained on large-scale diverse image data.

* The backbone captures high-level semantic structure such as vegetation, terrain texture, and depth cues.
* Feature extraction is performed using a frozen backbone, producing dense embeddings at a coarse spatial resolution.

This approach significantly reduces training requirements while improving performance on unseen environments.

### 2. Decoder: Progressive Semantic Decoder (PSD-Net)

To convert transformer features into dense pixel-level predictions, we design a **Progressive Semantic Decoder**.

* Features are upsampled gradually through multiple stages rather than a single large interpolation step.
* Each stage includes lightweight convolutional refinement blocks to recover spatial details.
* This progressive decoding improves boundary sharpness for small or thin objects such as rocks, logs, and branches.

## 🔬 Methodology & Training Strategy

### 1. Addressing Class Imbalance

Analysis of the dataset revealed severe class imbalance. Dominant classes such as sky and background account for the majority of pixels, while critical obstacle classes appear very infrequently.

To address this:

* We apply a **class-weighted loss function** with **square-root dampening**.
* This reduces over-penalization of rare classes while still encouraging correct obstacle detection.
* The approach stabilizes training and improves overall generalization.

### 2. Progressive Decoder Design

The decoder architecture follows a multi-stage upsampling strategy:

* Spatial resolution is increased incrementally.
* Feature refinement occurs at each stage, improving semantic consistency and edge quality.
* This design is especially effective in cluttered off-road scenes with complex terrain boundaries.

### 3. Training Protocol

* **Backbone**: DINOv2 (frozen)
* **Optimizer**: AdamW
* **Learning Rate Scheduling**: ReduceLROnPlateau
* **Augmentation**: Random cropping, flipping, and color jitter to improve robustness
* **Training Duration**: 15 epochs

The model converges efficiently due to strong pretrained representations.

## 📊 Performance & Evaluation

Model evaluation is performed on a held-out set of **unseen off-road scenes**, simulating deployment in novel environments.

### Key Metrics

* **Mean Pixel Accuracy**: **80.44%**
* **Mean IoU**: ~0.72
* **Inference Time**: ~20 ms per image (real-time capable)

### Training Dynamics

The pretrained transformer backbone enables rapid convergence and stable optimization.

|                 Loss Curve                 |                   IoU Curve                  |
| :----------------------------------------: | :------------------------------------------: |
| ![Loss](Visual_Report/training_curves.png) |     ![IoU](Visual_Report/iou_curves.png)     |
|  *Validation loss stabilizes consistently* | *Steady improvement in segmentation quality* |

### Quantitative Analysis

Performance remains consistent across diverse terrain configurations, with notable improvements in rare obstacle classes.

|                 Class-Wise Accuracy (IoU)                 |                  Test Set Reliability                 |
| :-------------------------------------------------------: | :---------------------------------------------------: |
| ![Class Chart](Visual_Report/class_accuracy_chart_v3.png) | ![Histogram](Visual_Report/accuracy_histogram_v3.png) |
|           *Improved detection of rocks and logs*          |      *Stable accuracy across 1000+ test samples*      |

### Qualitative Results

Below is a qualitative comparison showing the model’s ability to identify terrain categories and obstacles under varying conditions.

![Performance Summary](Visual_Report/performance_bar_chart.png)

## 🛠️ Installation & Usage

### Prerequisites

* Python 3.10+
* CUDA-enabled GPU recommended

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Demo

A lightweight GUI is provided for visual inspection of model predictions.

```bash
python demo.py
```

* Press **O** to load an image
* View segmentation output and metrics in real time

### 3. Reproduce Evaluation Results

To evaluate on the full test set:

```bash
python evaluate_test_set.py
```

This generates a detailed report with per-image and aggregate metrics.

### 4. Segmentation Color Legend

* 🟢 Trees / Vegetation
* 🔵 Sky
* 🟤 Logs / Trunks
* ⚫ Background
* 🪨 Rocks
* 🌫️ Distant Terrain

## 📂 Project Structure

* `src/model.py`: DINOv2 + Progressive Decoder definition
* `src/train.py`: Training and validation pipeline
* `src/dataset.py`: Dataset loader for synthetic off-road scenes
* `evaluate_test_set.py`: Evaluation script
* `demo.py`: Interactive visualization tool

---

*Built for off-road autonomy research using synthetic digital twin data.*

---
