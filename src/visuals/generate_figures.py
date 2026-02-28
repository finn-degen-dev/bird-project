import os
import glob
import json
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from data import SmartBirdDataset
from models import get_model

LOGS_DIR = os.path.join(VISUALS_DIR, "logs")
CACHE_FILE = os.path.join(LOGS_DIR, "evaluation_cache.json")

def plot_training_curves():
    print("\n--- Generating Figure: Training & Loss Curves ---")
    log_files = glob.glob(os.path.join(LOGS_DIR, "*_history.json"))

    if not log_files:
        print("No JSON logs found. Train models first!")
        return

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    for file in log_files:
        exp_name = os.path.basename(file).replace("_history.json", "")
        if "embedding" in exp_name or "task-type" in exp_name:
            continue

        with open(file, "r") as f:
            data = json.load(f)

        df = pd.DataFrame(data)
        label_name = exp_name.split("_")[0].title()
        if "custom" in exp_name:
            label_name += " (Custom CNN)"

        max_acc = df['val_acc'].max()
        plt.plot(df['epoch'], df['val_acc'], label=f"{label_name} (Max: {max_acc:.1f}%)", linewidth=2.5, marker='o', markersize=4)

    plt.title("Training Progress: Validation Accuracy", fontsize=16, fontweight='bold')
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Validation Accuracy (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.legend(title="Architectures")
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "comparison_model_performance.png"), dpi=300)

    resnet_log = os.path.join(LOGS_DIR, "resnet_task-species_attn-False_layers-3_history.json")
    if os.path.exists(resnet_log):
        with open(resnet_log, "r") as f:
            df = pd.DataFrame(json.load(f))

        plt.figure(figsize=(10, 6))
        plt.plot(df['epoch'], df['train_loss'], label="Train Loss", color='#1f77b4', linewidth=2.5)
        plt.plot(df['epoch'], df['val_loss'], label="Validation Loss", color='#ff7f0e', linewidth=2.5)
        plt.title("ResNet18: Training vs Validation Loss (Impact of SpecAugment)", fontsize=16)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(VISUALS_DIR, "loss_curve_15epoch_resnet18pre.png"), dpi=300)

    print(" Training curves saved.")

def evaluate_and_plot_bar_chart():
    print("\n--- Generating Figure: Sliding Window Model Comparison ---")

    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            cache = json.load(f)

    model_paths = glob.glob(os.path.join(MODELS_DIR, "*_best.pth"))
    results =[]

    for path in model_paths:
        exp_name = os.path.basename(path).replace("_best.pth", "")
        if "task-type" in exp_name: continue

        if exp_name in cache:
            results.append({"Model": exp_name, "Accuracy": cache[exp_name]})
        else:
            print(f" Model {exp_name} not in cache. You must evaluate it first via eval scripts! Defaulting to 0%.")
            results.append({"Model": exp_name, "Accuracy": 0.0})

    if not results: return

    df = pd.DataFrame(results)
    df['Display Name'] = df['Model'].str.replace('task-species_attn-', '').str.replace('False', 'NoAttn').str.replace('True', 'Attn').str.replace('layers-', 'L').str.replace('_', ' ').str.title()
    df = df.sort_values(by='Accuracy', ascending=False)

    plt.figure(figsize=(13, 7))
    sns.set_theme(style="whitegrid")

    colors =['#ff7f0e' if 'Embedding' in name else '#1f77b4' for name in df['Display Name']]
    sns.barplot(data=df, x='Display Name', y='Accuracy', hue='Display Name', palette=colors, legend=False)

    plt.title("Model Comparison: Accuracy (All Classes)", fontsize=16, fontweight='bold')
    plt.xlabel("Model Architecture", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 110)
    plt.xticks(rotation=35, ha="right")

    for i, val in enumerate(df['Accuracy']):
        plt.text(i, val + 1.5, f"{val:.1f}%", ha='center', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "sliding_window_comparison_updated.png"), dpi=300)
    print(" Bar chart saved.")

def plot_quality_metrics():
    print("\n--- Generating Figure: Audio Quality vs Accuracy ---")
    exp_name = "efficientnet_task-species_attn-False_layers-3"
    model_path = os.path.join(MODELS_DIR, f"{exp_name}_best.pth")

    if not os.path.exists(model_path):
        print("EfficientNet not found. Train it first!")
        return

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
        classes = json.load(f)['classes']

    dataset = SmartBirdDataset(mode='test', task='species')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_model("efficientnet", len(classes), False, 3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    qualities, correctness = [],[]

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(loader, desc="Scanning Dataset Qualities")):
            outputs = model(inputs.to(DEVICE))
            _, preds = torch.max(outputs, 1)

            start_idx = i * 32
            batch_qualities = dataset.df.iloc[start_idx:start_idx + len(labels)]['quality'].values

            for p, l, q in zip(preds.cpu().numpy(), labels.numpy(), batch_qualities):
                correctness.append(1 if p == l else 0)
                qualities.append(q)

    results_df = pd.DataFrame({'Quality': qualities, 'Correct': correctness})
    results_df = results_df[results_df['Quality'].isin(['A', 'B', 'C', 'D'])]

    acc_by_quality = results_df.groupby('Quality')['Correct'].mean() * 100
    acc_by_quality = acc_by_quality.reindex(['A', 'B', 'C', 'D'])
    counts = results_df['Quality'].value_counts().reindex(['A', 'B', 'C', 'D'])

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    sns.barplot(x=acc_by_quality.index, y=acc_by_quality.values, hue=acc_by_quality.index, palette="viridis", legend=False)

    plt.title("Performance by Audio Quality (EfficientNet)", fontsize=16, fontweight='bold')
    plt.xlabel("Xeno-Canto Audio Quality Rating", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)
    plt.ylim(0, 110)

    for i, (acc, count) in enumerate(zip(acc_by_quality.values, counts.values)):
        plt.text(i, acc + 2, f"{acc:.1f}%\n(n={count})", ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(VISUALS_DIR, "performance_by_quality.png"), dpi=300)
    print(" Quality graph saved.")

if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)
    plot_training_curves()
    evaluate_and_plot_bar_chart()
    plot_quality_metrics()
