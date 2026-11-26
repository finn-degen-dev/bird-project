<div align="center">

# Piepmatz Polizei

**Autonomous acoustic monitoring for European bird species**  
*Jugend forscht 2026 — Finn Degen & Jonathan Ledwon*

[![Paper](https://img.shields.io/badge/Research%20Paper-PDF-B31B1B?style=for-the-badge&logo=adobeacrobatreader&logoColor=white)](docs/main.pdf)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/BoredVoidAter/piepmatz-polizei)
[![PyTorch](https://img.shields.io/badge/PyTorch-ONNX%20Runtime-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](#ml-pipeline--results)

<br>

<img src="bird_project_cover.png" width="420" alt="Piepmatz Polizei Logo">

</div>

**Awards (Jugend forscht 2026):**
* **Regionalwettbewerb Erlangen:** 1. Preis (Mathematik/Informatik, Dieter Schwarz Stiftung) & Sonderpreis Umwelttechnik (DBU)
* **Landeswettbewerb Bayern:** Sonderpreis „Biodiversität“ (Hermann Gutmann Stiftung) & Sonderpreis Naturgucker (NABU)

---

## Overview

Piepmatz Polizei combines a solar-powered **ESP32** audio node, a **Raspberry Pi 5** inference server, and **CNN classifiers** trained on Xeno-Canto European recordings to detect bird species (and song vs. call vocalizations) continuously in the field.

The full methodology, experiments, and hardware documentation are available in **[docs/main.pdf](docs/main.pdf)**.

---

## System Architecture & Hardware

To maximize battery life on the sensor node while running full-precision inference, recording and classification are split across two devices:

1. **Solar Edge Node (`hardware/esp32_mic/`):** An **ESP32-WROOM-32U** reads 32 kHz audio via I2S from an **INMP441** MEMS microphone and streams 16-bit mono PCM over TCP (`port 5000`). Powered off-grid by two 6V 1W solar panels, a **CN3791** MPPT charger, two **18650** Li-Ion cells, and an MT3608 step-up converter inside a 3D-printed Onshape enclosure.
2. **Base Station (`hardware/pi_server/`):** A **Raspberry Pi 5** acting as a local Wi-Fi hotspot runs a **FastAPI + WebSocket** dashboard (`web_dashboard.py`) with bandpass filtering (1–12 kHz), **Wake-on-Sound** RMS gating, **ONNX Runtime** inference, audio replay logging, CSV export, and Open-Meteo weather correlation.

| Field Enclosure | CAD Structure (Onshape) | Wiring Schematic (KiCad) |
| :---: | :---: | :---: |
| <img src="visuals/cad/RealView.jpg" alt="Assembled Sensor" width="260"/> | <img src="visuals/cad/OnshapeModelView_OnlyCasing.png" alt="CAD Casing" width="260"/> | <img src="visuals/electronics/schematic.png" alt="KiCad Schematic" width="260"/> |

---

## ML Pipeline & Results

Audio is segmented into 5-second windows, transformed into 128-band Mel-spectrograms (`224×224`), and trained with **SpecAugment**, **MixUp**, and class-balanced sampling (`WeightedRandomSampler`). Non-bird distractor classes (insects, amphibians, mammals) are filtered during inference.

### Model Benchmarks (European Species Dataset)

| Architecture | Test Accuracy | Notes |
| :--- | :---: | :--- |
| **EfficientNet-B0** (3-ch, FP32) | **96.7%** | Best accuracy; deployed as `efficientnet_fp32.onnx` |
| **ResNet-18** (3-ch) | **94.6%** | Strong baseline |
| **Custom CNN** (5 layers) | **61.5%** | Trained from scratch (`2` layers: 40.8%, `3` layers + Attn: 60.6%) |
| **Metric Learning** (ArcFace) | **60.0%** | Embedding-based nearest-centroid / k-NN evaluation |

> **Edge Optimization Findings:** Post-training INT8 quantization shrank the ONNX model from `19.1 MB` to `5.2 MB` but degraded accuracy due to EfficientNet Squeeze-and-Excitation scaling, while shrinking spectrogram resolution below `128 px` blurred Mel frequency bands (`94.0%` → `75.6%` on MobileNetV3). Keeping `224×224` FP32 with **Wake-on-Sound** RMS gating cuts daily compute load by up to ~65% without losing accuracy.

### Signal Processing & Explainability

* **Temporal Smoothing & Context Suppression:** 1D sliding-window convolution (`[0.1, 0.2, 0.4, 0.2, 0.1]`) and root-scaled confidence suppression ($P(t, c) \cdot \sqrt{P_{\max}(c)}$) filter transient wind/branch noise while preserving quiet distant calls.
* **Grad-CAM Verification:** Saliency overlays confirm the models attend to species-specific vocalization frequencies rather than background noise.

| Grad-CAM Sliding Window | False-Positive Suppression |
| :---: | :---: |
| <img src="visuals/experiments/europe/gradcam_sliding_window.png" alt="Grad-CAM Visualization"/> | <img src="visuals/experiments/false_positive_suppression.png" alt="False Positive Suppression"/> |

---

## Quick Start

### 1. Local Inference & Live Microphone Radar

```bash
git clone https://github.com/finn-degen-dev/bird-project.git
cd bird-project

# Using NixOS flake devShell (Python 3.12 + CUDA + LaTeX):
nix develop

# Or standard venv:
pip install -r requirements.txt

# Download pre-trained weights from Hugging Face
python download_models.py

# Run live microphone radar (or pass --file path/to/audio.mp3)
python src/showcase.py --model efficientnet --live
```

### 2. Pi Base Station Dashboard

```bash
pip install -r hardware/pi_server/requirements.txt
python hardware/pi_server/web_dashboard.py
# Dashboard available at http://localhost:8000 (ESP32 TCP stream on port 5000)
```

---

## Pre-trained Models

All checkpoints and class mappings are hosted on Hugging Face ([`BoredVoidAter/piepmatz-polizei`](https://huggingface.co/BoredVoidAter/piepmatz-polizei)):

* `--model efficientnet` — **EfficientNet-B0** (highest accuracy for PC / Pi 5)
* `--model mobilenet` — **MobileNetV3-Small** (lowest latency for constrained CPUs)
* `--model resnet` — **ResNet-18** (baseline benchmark)
* `--model efficientnet-type` — **Song vs. Call** vocalization classifier

```bash
python download_models.py --list             # Show all available checkpoints
python download_models.py --model mobilenet  # Download a single model
```

