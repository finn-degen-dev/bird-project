import sys
import os
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- CONFIG ---
MODEL_PATH = "data/models/german_bird_classifier_pro.pth"
JSON_PATH = "data/models/german_bird_classifier_pro_classes.json"
BATCH_SIZE = 32
WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("📊 ANALYZING PERFORMANCE: Common vs. Rare Species")
    
    # 1. Load Data & Counts
    full_ds = SmartBirdDataset(mode='train', task='species')
    df = full_ds.df
    species_counts = df['species'].value_counts()
    
    # Thresholds
    threshold_high = species_counts.quantile(0.66)
    threshold_low = species_counts.quantile(0.33)
    
    def get_category(species_name):
        count = species_counts.get(species_name, 0)
        if count > threshold_high: return "Common"
        if count < threshold_low: return "Rare"
        return "Medium"

    # 2. Validation Split (Fixed Seed)
    generator = torch.Generator().manual_seed(42) 
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    _, val_ds = random_split(full_ds, [train_size, val_size], generator=generator)
    
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    
    # 3. Model
    with open(JSON_PATH, 'r') as f:
        classes = json.load(f)['classes']
    
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 4. Inference
    print(f"   Evaluating on {len(val_ds)} samples...")
    results = [] 

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                species_name = classes[labels[i].item()]
                is_correct = (labels[i].item() == preds[i].item())
                
                results.append({
                    "species": species_name,
                    "correct": is_correct,
                    "total_data_count": species_counts[species_name],
                    "category": get_category(species_name)
                })

    res_df = pd.DataFrame(results)
    
    # --- AGGREGATE STATS ---
    species_perf = res_df.groupby("species").agg(
        accuracy=('correct', 'mean'),
        val_samples=('correct', 'count'), 
        total_data_count=('total_data_count', 'first'), # FIXED: Naming consistency
        category=('category', 'first')
    ).reset_index()

    # Filter out species that didn't appear in Validation set at all
    species_perf = species_perf[species_perf['val_samples'] > 0]

    # --- PLOT 1: The Bubble Chart ---
    plt.figure(figsize=(12, 8))
    
    # Size of bubble = Number of training samples (Log scale for better visual)
    sns.scatterplot(
        data=species_perf, 
        x="total_data_count", 
        y="accuracy", 
        hue="category",
        size="total_data_count",
        sizes=(20, 400), # Min/Max bubble size
        alpha=0.6,
        palette={"Rare": "red", "Medium": "orange", "Common": "green"}
    )
    
    # Trend Line
    sns.regplot(data=species_perf, x="total_data_count", y="accuracy", scatter=False, color="#444", logx=True, line_kws={"linestyle": "--"})
    
    # Annotate Interesting Points
    # 1. Identify the 0% birds with the most data (The "Disappointments")
    zeros = species_perf[species_perf['accuracy'] == 0]
    if not zeros.empty:
        worst = zeros.sort_values('total_data_count', ascending=False).iloc[0]
        plt.text(worst.total_data_count, worst.accuracy, f"  {worst.species} (0%)", fontsize=9, va='center')

    # 2. Identify the most common bird
    best_common = species_perf.sort_values('total_data_count', ascending=False).iloc[0]
    plt.text(best_common.total_data_count, best_common.accuracy, f"  {best_common.species}", fontsize=9, va='center', fontweight='bold')

    plt.title("Impact of Training Data Volume on Species Accuracy", fontsize=16)
    plt.xlabel("Total Recordings in Database (Log Scale)", fontsize=12)
    plt.ylabel("Validation Accuracy (0.0 - 1.0)", fontsize=12)
    plt.xscale('log')
    plt.ylim(-0.05, 1.05)
    plt.grid(True, which="both", ls="--", alpha=0.2)
    
    out_path = "data/visuals/scatter_data_vs_acc_bubble.png"
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ Bubble Chart saved to {out_path}")
    print("   (Bigger bubbles = More Data. Note how small bubbles vary wildly, but big bubbles stay high.)")

if __name__ == "__main__":
    main()
