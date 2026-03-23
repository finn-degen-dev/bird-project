# Piepmatz Polizei
> Jugend forscht 2026 — Finn Degen, Jonathan Ledwon

<p align="center">
  <img src="bird_project_cover.png" width="450" alt="Piepmatz Polizei Project Logo">
</p>

This project develops an autonomous, AI-powered system for the acoustic monitoring of European bird species. It uses efficient deep learning models deployed on low-power hardware (Raspberry Pi) to classify bird calls in real-time from audio recordings.

## Project Paper
The full research paper detailing our methodology, experiments, and results is available in the `docs` folder.

**[Read the full paper (PDF)](docs/main.pdf)**

## Quick Start

Follow these steps to get the main model running on your machine.

```bash
# 1. Clone the repository
git clone https://github.com/BoredVoidAter/bird-project
cd bird-project

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the pre-trained models
python download_models.py

# 4. Run the live showcase with the best model!
python src/showcase.py --model efficientnet --live
```

### Available Models

You can download all models at once or select a specific one. The most important models are:

*   `--model efficientnet`
    *   **Description**: The best all-around model for **accuracy**. Uses an EfficientNet-B0 architecture with transfer learning. This is the recommended model for analysis on a PC or powerful hardware.
    *   **Use Case**: High-precision classification, data analysis, generating figures.

*   `--model mobilenet`
    *   **Description**: The best model for **performance on edge devices**. Uses a MobileNetV3-Small architecture, which is highly optimized for CPUs with limited power, like the Raspberry Pi Zero.
    *   **Use Case**: Real-time deployment on the Raspberry Pi, live monitoring where speed is critical.

*   `--model resnet`
    *   **Description**: A classic ResNet18 architecture that serves as a strong and reliable baseline for comparison.
    *   **Use Case**: Benchmarking, comparison with other architectures.

**To download all models (default):**
```bash
python download_models.py
```

**To download only a specific model:**
```bash
python download_models.py --model mobilenet
```

## Pre-trained Models
All model weights are hosted on the Hugging Face Hub.

[![View on Hugging Face](https://img.shields.io/badge/Hugging%20Face-Models-yellow)](https://huggingface.co/BoredVoidAter/piepmatz-polizei)
