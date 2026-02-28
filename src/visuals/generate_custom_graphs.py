import os
import glob
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import *
from data import SmartBirdDataset
from models import get_model

def get_sliding_window_chunks(spec):
    """Generates 5-second sliding window chunks from a full spectrogram."""
    chunks =[]
    current_width = spec.shape[1]
    stride = TARGET_WIDTH // 2

    if current_width < TARGET_WIDTH:
        pad_amount = TARGET_WIDTH - current_width
        chunk = torch.nn.functional.pad(spec, (0, pad_amount))
        chunks.append(chunk)
    else:

        for start in range(0, current_width - TARGET_WIDTH + 1, stride):
            chunk = spec[:, start : start + TARGET_WIDTH]
            chunks.append(chunk)

        if current_width > TARGET_WIDTH and (current_width - TARGET_WIDTH) % stride != 0:
            chunks.append(spec[:, -TARGET_WIDTH:])

    formatted_chunks =[]
    for c in chunks:
        tensor = c.unsqueeze(0).repeat(3, 1, 1)
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
        ).squeeze(0)
        formatted_chunks.append(tensor)

    return torch.stack(formatted_chunks)

def plot_sliding_window_all_models():
    print("\n--- Generating Graph 1: Sliding Window Comparison (All Models) ---")
    dataset = SmartBirdDataset(mode='test', task='species')
    model_paths = glob.glob(os.path.join(MODELS_DIR, "*_best.pth"))

    results =[]

    for model_path in model_paths:
        exp_name = os.path.basename(model_path).replace("_best.pth", "")

        if "task-type" in exp_name or "embedding" in exp_name:
            continue

        model_type = exp_name.split("_")[0]
        use_attention = "attn-True" in exp_name
        cnn_layers = 3
        if "layers-" in exp_name:
            cnn_layers = int(exp_name.split("layers-")[1].split("_")[0])

        with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
            classes = json.load(f)['classes']

        model = get_model(model_type, len(classes), use_attention, cnn_layers).to(DEVICE)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()

        correct = 0
        total = len(dataset)

        print(f"Evaluating: {exp_name}")
        with torch.no_grad():
            for idx in tqdm(range(total), desc="Sliding Window Inference", leave=False):
                row = dataset.df.iloc[idx]
                pt_path = os.path.join(PROCESSED_DIR, row['filename'].replace('.mp3', '.pt'))

                spec = torch.load(pt_path).float()
                chunks = get_sliding_window_chunks(spec).to(DEVICE)

                if len(chunks) == 0:
                    continue

                outputs = model(chunks)
                avg_output = outputs.mean(dim=0)
                pred = avg_output.argmax().item()
                label = dataset._get_label(row)

                if pred == label:
                    correct += 1

        acc = (correct / total) * 100
        results.append({"Model": exp_name, "Accuracy": acc})
        print(f"-> Accuracy: {acc:.2f}%\n")

    df = pd.DataFrame(results)
    df['Display Name'] = df['Model'].str.replace('task-species_attn-', '')                                    .str.replace('False', 'NoAttn')                                    .str.replace('True', 'Attn')                                    .str.replace('layers-', 'L')                                    .str.replace('_', ' ').str.title()
    df = df.sort_values(by='Accuracy', ascending=False)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")
    sns.barplot(data=df, x='Display Name', y='Accuracy', palette="mako")

    plt.title("Model Comparison: Sliding Window Accuracy", fontsize=16, fontweight='bold')
    plt.xlabel("Model Architecture", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")

    for i, val in enumerate(df['Accuracy']):
        plt.text(i, val + 1.5, f"{val:.1f}%", ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(VISUALS_DIR, "sliding_window_comparison.png")
    plt.savefig(out_path, dpi=300)
    print(f" Saved Graph 1 to {out_path}")

def plot_quality_top_model():
    print("\n--- Generating Graph 2: Top Model Performance by Audio Quality ---")
    exp_name = "efficientnet_task-species_attn-False_layers-3"
    model_path = os.path.join(MODELS_DIR, f"{exp_name}_best.pth")

    if not os.path.exists(model_path):
        print(f"Top model ({exp_name}) not found. Please train it first.")
        return

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
        classes = json.load(f)['classes']

    dataset = SmartBirdDataset(mode='test', task='species')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_model("efficientnet", len(classes), False, 3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    qualities =[]
    correctness =[]

    print("Evaluating EfficientNet for quality breakdown...")
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader, desc="Eval Quality")):
            inputs = inputs.to(DEVICE)
            labels = labels.numpy()

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()

            start_idx = i * 32
            end_idx = start_idx + len(labels)
            batch_qualities = dataset.df.iloc[start_idx:end_idx]['quality'].values

            for p, l, q in zip(preds, labels, batch_qualities):
                correctness.append(1 if p == l else 0)
                qualities.append(q)

    results_df = pd.DataFrame({'Quality': qualities, 'Correct': correctness})
    results_df = results_df[results_df['Quality'].isin(['A', 'B', 'C', 'D'])]

    acc_by_quality = results_df.groupby('Quality')['Correct'].mean() * 100
    acc_by_quality = acc_by_quality.reindex(['A', 'B', 'C', 'D'])

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=acc_by_quality.index, y=acc_by_quality.values, palette="viridis")

    plt.title("Performance by Audio Quality (EfficientNet)", fontsize=16, fontweight='bold')
    plt.xlabel("Xeno-Canto Audio Quality Rating", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)

    for i, v in enumerate(acc_by_quality.values):
        plt.text(i, v + 1.5, f"{v:.1f}%", ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(VISUALS_DIR, "performance_by_quality.png")
    plt.savefig(out_path, dpi=300)
    print(f" Saved Graph 2 to {out_path}")

if __name__ == "__main__":
    plot_sliding_window_all_models()
    plot_quality_top_model()
