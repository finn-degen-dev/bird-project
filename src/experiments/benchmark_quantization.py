import os
import sys
import time
import torch
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from data import SmartBirdDataset
from models import get_model

def get_dir_size(path):
    if not os.path.exists(path):
        return 0
    return os.path.getsize(path) / (1024 * 1024)

def main():
    print("Starte Quantisierungs-Benchmark (Größe, Latenz, Genauigkeit)...")

    model_name = "efficientnet"
    exp_name = f"{model_name}_task-species_attn-False_layers-3"

    onnx_fp32_path = os.path.join(MODELS_DIR, f"{model_name}_fp32.onnx")
    onnx_int8_minmax_path = os.path.join(MODELS_DIR, f"{model_name}_quantized_int8_minmax.onnx")
    onnx_int8_entropy_path = os.path.join(MODELS_DIR, f"{model_name}_quantized_int8_entropy.onnx")

    if not os.path.exists(onnx_int8_entropy_path):
        print("ONNX Modelle fehlen. Bitte zuerst ausführen: python src/quantize.py --model efficientnet")
        return

    # Größen messen
    size_fp32 = get_dir_size(onnx_fp32_path)
    size_int8_minmax = get_dir_size(onnx_int8_minmax_path)
    size_int8_entropy = get_dir_size(onnx_int8_entropy_path)

    dummy_input = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    # Sessions laden
    sess_fp32 = ort.InferenceSession(onnx_fp32_path, providers=['CPUExecutionProvider'])
    sess_int8_minmax = ort.InferenceSession(onnx_int8_minmax_path, providers=['CPUExecutionProvider'])
    sess_int8_entropy = ort.InferenceSession(onnx_int8_entropy_path, providers=['CPUExecutionProvider'])

    def measure_latency(session, input_data, runs=100):
        input_name = session.get_inputs()[0].name
        # Warmup
        for _ in range(10): session.run(None, {input_name: input_data})
        # Benchmark
        start = time.time()
        for _ in range(runs): session.run(None, {input_name: input_data})
        return ((time.time() - start) / runs) * 1000

    print("Messe Inferenzzeiten...")
    lat_fp32 = measure_latency(sess_fp32, dummy_input)
    lat_int8_minmax = measure_latency(sess_int8_minmax, dummy_input)
    lat_int8_entropy = measure_latency(sess_int8_entropy, dummy_input)

    print("Evaluiere Accuracy auf 500 Test-Samples (das kann einen Moment dauern)...")
    dataset = SmartBirdDataset(mode='test', task='species')
    subset = Subset(dataset, np.random.choice(len(dataset), min(500, len(dataset)), replace=False))
    loader = DataLoader(subset, batch_size=1, shuffle=False)

    correct_fp32, correct_int8_minmax, correct_int8_entropy = 0, 0, 0
    input_name_fp32 = sess_fp32.get_inputs()[0].name
    input_name_int8_minmax = sess_int8_minmax.get_inputs()[0].name
    input_name_int8_entropy = sess_int8_entropy.get_inputs()[0].name

    for img, label in loader:
        img_np = img.numpy()
        label_np = label.numpy()[0]

        out_fp32 = sess_fp32.run(None, {input_name_fp32: img_np})[0]
        out_int8_minmax = sess_int8_minmax.run(None, {input_name_int8_minmax: img_np})[0]
        out_int8_entropy = sess_int8_entropy.run(None, {input_name_int8_entropy: img_np})[0]

        if np.argmax(out_fp32) == label_np: correct_fp32 += 1
        if np.argmax(out_int8_minmax) == label_np: correct_int8_minmax += 1
        if np.argmax(out_int8_entropy) == label_np: correct_int8_entropy += 1

    acc_fp32 = (correct_fp32 / len(loader)) * 100
    acc_int8_minmax = (correct_int8_minmax / len(loader)) * 100
    acc_int8_entropy = (correct_int8_entropy / len(loader)) * 100

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    labels = ['32-Bit (FP32)', '8-Bit (MinMax)', '8-Bit (Entropie)']
    # Farben: Blau für Basis, Rot für den Accuracy-Zusammenbruch, Grün für die Rettung
    colors = ['#1f77b4', '#d62728', '#2ca02c']

    # 1. Dateigröße
    sns.barplot(x=labels, y=[size_fp32, size_int8_minmax, size_int8_entropy], ax=axes[0], palette=colors)
    axes[0].set_title("Modellgröße (MB) ", fontweight='bold')
    axes[0].set_ylabel("Megabyte (MB)")
    for i, v in enumerate([size_fp32, size_int8_minmax, size_int8_entropy]): 
        axes[0].text(i, v+0.2, f"{v:.1f} MB", ha='center', fontweight='bold')

    # 2. Latenz
    sns.barplot(x=labels, y=[lat_fp32, lat_int8_minmax, lat_int8_entropy], ax=axes[1], palette=colors)
    axes[1].set_title("Inferenzzeit pro Audio-Clip (ms) ", fontweight='bold')
    axes[1].set_ylabel("Millisekunden (ms)")
    for i, v in enumerate([lat_fp32, lat_int8_minmax, lat_int8_entropy]): 
        axes[1].text(i, v+0.2, f"{v:.1f} ms", ha='center', fontweight='bold')

    # 3. Accuracy
    sns.barplot(x=labels, y=[acc_fp32, acc_int8_minmax, acc_int8_entropy], ax=axes[2], palette=colors)
    axes[2].set_title("Genauigkeit (Accuracy) ", fontweight='bold')
    axes[2].set_ylabel("Accuracy (%)")
    axes[2].set_ylim(0, 110)
    for i, v in enumerate([acc_fp32, acc_int8_minmax, acc_int8_entropy]): 
        axes[2].text(i, v+2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.suptitle("Auswirkungen der Quantisierungs-Methodik (MinMax vs. Entropie)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()

    out_path = os.path.join(VISUALS_DIR, "comparison_quantization.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n Erfolgreich gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()
