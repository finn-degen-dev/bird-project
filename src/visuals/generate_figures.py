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
            attn_str = "Attn" if "attn-True" in exp_name else "NoAttn"
            layer_str = "3"
            for part in exp_name.split("_"):
                if part.startswith("layers-"):
                    layer_str = part.split("-")[1]
            label_name = f"Custom CNN ({layer_str} Layers, {attn_str})"

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

def plot_data_vs_accuracy_bubble():
    print("\n--- Generating Figure: Data Volume vs Accuracy (Bubble Chart) ---")
    exp_name = "efficientnet_task-species_attn-False_layers-3"
    model_path = os.path.join(MODELS_DIR, f"{exp_name}_best.pth")

    if not os.path.exists(model_path):
        print("  -> EfficientNet not found. Train it first! Skipping Bubble Chart...")
        return

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
        classes = json.load(f)['classes']

    dataset = SmartBirdDataset(mode='test', task='species')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_model("efficientnet", len(classes), False, 3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    class_correct = {c: 0 for c in classes}
    class_total = {c: 0 for c in classes}

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluating Accuracy per Species"):
            outputs = model(inputs.to(DEVICE))
            _, preds = torch.max(outputs, 1)

            for p, l in zip(preds.cpu().numpy(), labels.numpy()):
                class_name = classes[l]
                class_total[class_name] += 1
                if p == l:
                    class_correct[class_name] += 1

    df_meta = pd.read_csv(METADATA_FILE)
    all_counts = df_meta['species'].value_counts().to_dict()

    data = []
    for c in classes:
        acc = class_correct[c] / class_total[c] if class_total[c] > 0 else 0
        total_recs = all_counts.get(c, 0)

        if total_recs == 0:
            continue

        if total_recs < 50:
            cat = "Rare"
        elif total_recs < 100:
            cat = "Medium"
        else:
            cat = "Common"

        data.append({
            "Species": c,
            "Validation Accuracy (0.0 - 1.0)": acc,
            "Total Recordings in Database (Log Scale)": total_recs,
            "category": cat,
            "total_data_count": total_recs
        })

    if not data:
        print("  -> No data to plot.")
        return

    df_plot = pd.DataFrame(data)

    plt.figure(figsize=(14, 9))
    sns.set_theme(style="whitegrid", rc={"grid.linestyle": "--", "axes.grid": True, "grid.alpha": 0.5})

    palette = {"Rare": "#ff4d4d", "Medium": "#ffbc47", "Common": "#4caf50"}

    sns.regplot(
        data=df_plot,
        x="Total Recordings in Database (Log Scale)",
        y="Validation Accuracy (0.0 - 1.0)",
        scatter=False,
        logx=True,
        color="#444444",
        line_kws={"linestyle": "--", "linewidth": 2.5, "alpha": 0.8}
    )

    ax = sns.scatterplot(
        data=df_plot,
        x="Total Recordings in Database (Log Scale)",
        y="Validation Accuracy (0.0 - 1.0)",
        size="total_data_count",
        sizes=(50, 600),
        hue="category",
        hue_order=["Rare", "Medium", "Common"],
        palette=palette,
        alpha=0.8,
        edgecolor="white"
    )

    plt.xscale('log')
    plt.title("Impact of Training Data Volume on Species Accuracy", fontsize=18, fontweight="bold", pad=15)
    plt.ylim(-0.05, 1.05)

    top_species = df_plot.loc[df_plot["Total Recordings in Database (Log Scale)"].idxmax()]
    plt.annotate(
        top_species["Species"],
        xy=(top_species["Total Recordings in Database (Log Scale)"], top_species["Validation Accuracy (0.0 - 1.0)"]),
        xytext=(15, 0),
        textcoords="offset points",
        fontweight="bold",
        va="center"
    )

    plt.legend(loc="lower left", frameon=True)
    plt.tight_layout()

    out_path = os.path.join(VISUALS_DIR, "scatter_data_vs_acc_bubble.png")
    plt.savefig(out_path, dpi=300)
    print(f" Bubble chart saved to: {out_path}")

if __name__ == "__main__":
    os.makedirs(LOGS_DIR, exist_ok=True)
    plot_training_curves()
    evaluate_and_plot_bar_chart()
    plot_quality_metrics()
    plot_data_vs_accuracy_bubble()
