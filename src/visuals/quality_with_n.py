import os
import json
import torch
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

def plot_quality_top_model_with_counts():
    print("\n--- Generating Graph 2: Top Model Performance by Audio Quality ---")
    exp_name = "efficientnet_task-species_attn-False_layers-3"
    model_path = os.path.join(MODELS_DIR, f"{exp_name}_best.pth")

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
        classes = json.load(f)['classes']

    dataset = SmartBirdDataset(mode='test', task='species')
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    model = get_model("efficientnet", len(classes), False, 3).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    qualities = []
    correctness =[]

    print("Evaluating EfficientNet for quality breakdown (takes ~30 seconds)...")
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

    counts_by_quality = results_df['Quality'].value_counts().reindex(['A', 'B', 'C', 'D'])

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        x=acc_by_quality.index,
        y=acc_by_quality.values,
        hue=acc_by_quality.index,
        palette="viridis",
        legend=False
    )

    plt.title("Performance by Audio Quality (EfficientNet)", fontsize=16, fontweight='bold')
    plt.xlabel("Xeno-Canto Audio Quality Rating", fontsize=14)
    plt.ylabel("Accuracy (%)", fontsize=14)

    plt.ylim(0, 110)

    for i, (acc, count) in enumerate(zip(acc_by_quality.values, counts_by_quality.values)):
        text = f"{acc:.1f}%\n(n={count})"
        plt.text(i, acc + 2, text, ha='center', fontweight='bold', fontsize=12)

    plt.tight_layout()
    out_path = os.path.join(VISUALS_DIR, "performance_by_quality.png")
    plt.savefig(out_path, dpi=300)
    print(f" Saved updated Graph 2 to {out_path}")

if __name__ == "__main__":
    plot_quality_top_model_with_counts()
