import sys
import os

# --- PATH FIX ---
# Add the parent directory ('src') to Python's path so we can import dataset.py
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) # Go up one level from visuals/ to src/
sys.path.append(src_dir)
# ----------------

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Now this import will work
from dataset import BirdRawDataset

# --- Configuration ---
TEST_DIR = "data/dataset/test"
MODEL_PATH = "data/models/bird_classifier_pro.pth"
BATCH_SIZE = 1 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_quality_from_filename(filepath):
    """
    Extracts quality tag from filenames like 'XC12345_qA.mp3'.
    Returns 'A', 'B', 'C', etc., or 'Unknown' if not found.
    """
    filename = os.path.basename(filepath)
    if '_q' in filename:
        try:
            # Split by '_q' and take the first character after it
            parts = filename.split('_q')
            if len(parts) > 1:
                quality_char = parts[1][0] # "A" from "A.mp3" or "A_..."
                return quality_char.upper()
        except:
            pass
    return "Unknown"

def main():
    if not os.path.exists(TEST_DIR):
        print(f"Error: {TEST_DIR} not found.")
        return

    print("Initializing Dataset...")
    # Mode='test' ensures no augmentation
    try:
        dataset = BirdRawDataset(TEST_DIR, mode='test')
    except FileNotFoundError:
        print("Could not load dataset. Check if data/dataset/test exists.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Loading Model...")
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset.classes))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    results = []

    print(f"Evaluating {len(dataset)} files by Quality...")
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Inference
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            is_correct = (preds == labels).item()
            
            # Get Filename to determine Quality
            file_path = dataset.files[i][0]
            quality = get_quality_from_filename(file_path)
            
            results.append({
                "Quality": quality,
                "Correct": is_correct
            })

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        print("No results generated.")
        return

    # Calculate Accuracy per Quality
    summary = df.groupby("Quality")["Correct"].mean().reset_index()
    summary["Accuracy"] = summary["Correct"] * 100
    
    # Add counts to labels
    counts = df["Quality"].value_counts().to_dict()
    summary["Label"] = summary["Quality"].apply(lambda x: f"{x}\n(n={counts[x]})")

    # Filter to standard qualities
    summary = summary[summary["Quality"].isin(['A', 'B', 'C', 'D', 'E'])]

    print("\n--- Results Summary ---")
    print(summary[["Quality", "Accuracy"]])

    # --- Plotting ---
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    
    # Create bar plot
    if not summary.empty:
        palette = sns.color_palette("Blues_d", n_colors=len(summary))
        ax = sns.barplot(x="Label", y="Accuracy", data=summary, palette=palette)
        
        # Add numbers on top
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{p.get_height():.1f}%', 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points',
                           fontweight='bold')
    else:
        print("Warning: No A-E quality tags found. Did you download new data with the quality tags?")

    plt.title("Model Performance by Audio Quality", fontsize=16)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.xlabel("Xeno-Canto Quality Rating", fontsize=12)
    plt.ylim(0, 110)
    
    # Ensure output directory exists
    os.makedirs("data/visuals", exist_ok=True)
    output_path = "data/visuals/performance_by_quality.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nChart saved to {output_path}")

if __name__ == "__main__":
    main()
