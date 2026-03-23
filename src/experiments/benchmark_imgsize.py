import os
import sys
import time
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models import get_model
from data import SmartBirdDataset

def measure_latency(model, dummy_input, runs=100):
    model.eval()

    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy_input)

    start_time = time.time()
    with torch.no_grad():
        for _ in range(runs):
            _ = model(dummy_input)

    total_time = time.time() - start_time
    return (total_time / runs) * 1000

def main():
    print("Starte IMG_SIZE Benchmark (MobileNetV3 bei verschiedenen Auflösungen)...")

    sizes = [224, 160, 128, 96]
    results = []

    np.random.seed(42)

    for size in sizes:
        print(f"\n[Auflösung {size}x{size}] Analysiere Modell...")

        exp_name = f"mobilenet_task-species_ch-3_size-{size}_attn-False_layers-3"
        json_path = os.path.join(MODELS_DIR, f"{exp_name}_classes.json")
        pth_path = os.path.join(MODELS_DIR, f"{exp_name}_best.pth")

        if not os.path.exists(pth_path) or not os.path.exists(json_path):
            print(f" -> FEHLER: Modell {exp_name} nicht gefunden. Überspringe...")
            continue

        with open(json_path, "r") as f:
            classes = json.load(f)['classes']
        num_classes = len(classes)

        model = get_model("mobilenet", num_classes, in_channels=3).to('cpu')
        model.load_state_dict(torch.load(pth_path, map_location='cpu'))
        model.eval()

        dummy_input = torch.randn(1, 3, size, size).to('cpu')
        latency_ms = measure_latency(model, dummy_input, runs=100)
        print(f" -> CPU Latenz: {latency_ms:.2f} ms")

        dataset = SmartBirdDataset(mode='test', task='species', in_channels=3, img_size=size)
        subset_indices = np.random.choice(len(dataset), min(500, len(dataset)), replace=False)
        subset = Subset(dataset, subset_indices)

        loader = DataLoader(subset, batch_size=32, shuffle=False)

        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to('cpu')
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = (correct / total) * 100
        print(f" -> Accuracy (auf {total} Samples): {accuracy:.1f}%")

        results.append({
            "Auflösung": f"{size}x{size}",
            "Latenz (ms)": latency_ms,
            "Accuracy (%)": accuracy
        })

    if not results:
        print("Es wurden keine Modelle gefunden. Bitte führe das Trainingsskript vorher aus.")
        return

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    names = [r["Auflösung"] for r in results]
    latencies = [r["Latenz (ms)"] for r in results]
    accuracies = [r["Accuracy (%)"] for r in results]

    colors = sns.color_palette("flare", len(names))

    sns.barplot(x=names, y=latencies, ax=axes[0], palette=colors)
    axes[0].set_title("Inferenzzeit auf der CPU pro Spektrogramm", fontweight='bold')
    axes[0].set_ylabel("Latenz (Millisekunden)")
    for i, v in enumerate(latencies):
        axes[0].text(i, v + (max(latencies)*0.02), f"{v:.1f} ms", ha='center', fontweight='bold')

    sns.barplot(x=names, y=accuracies, ax=axes[1], palette=colors)
    axes[1].set_title("Genauigkeit (Accuracy auf 500 Test-Aufnahmen)", fontweight='bold')
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 110)
    for i, v in enumerate(accuracies):
        axes[1].text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')

    plt.suptitle("Optimierung für Edge-Hardware: Auswirkung der Spektrogramm-Auflösung (MobileNetV3)", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()

    out_path = os.path.join(VISUALS_DIR, "imgsize_speed_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n Benchmark Plot erfolgreich gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()
