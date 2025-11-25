import sys
import os
import json
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import models
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- CONFIG ---
MODEL_PATH = "data/models/german_bird_classifier_pro.pth"
JSON_PATH = "data/models/german_bird_classifier_pro_classes.json"
BATCH_SIZE = 32
WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_classes():
    # Load Class Names
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        classes = data['classes']
    
    # Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    return model, classes

def analyze():
    model, classes = load_model_and_classes()
    
    # Load Validation Data (Same split as training)
    # Note: To be perfectly scientific, we should use a fixed seed or a separate test set.
    # For now, we recreate the split to get the validation portion.
    full_ds = SmartBirdDataset(mode='train', task='species')
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    _, val_ds = random_split(full_ds, [train_size, val_size])
    
    loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)
    
    print(f"📊 Analyzing {len(val_ds)} validation samples...")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    confusion_list = [] # Stores (True Label, Pred Label)

    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            
            # --- Top-1 Accuracy ---
            _, pred_1 = torch.max(outputs, 1)
            correct_top1 += torch.sum(pred_1 == labels).item()
            
            # --- Top-5 Accuracy ---
            # Get top 5 probabilities
            _, pred_5 = outputs.topk(5, 1, largest=True, sorted=True)
            labels_expanded = labels.view(-1, 1).expand_as(pred_5)
            correct_top5 += (pred_5 == labels_expanded).sum().item()
            
            # --- Collect Errors for Confusion Report ---
            for t, p in zip(labels.cpu().numpy(), pred_1.cpu().numpy()):
                if t != p:
                    confusion_list.append((classes[t], classes[p]))
            
            total += labels.size(0)

    acc1 = correct_top1 / total
    acc5 = correct_top5 / total
    
    print("\n" + "="*40)
    print(f"🏆 FINAL RESULTS")
    print(f"Top-1 Accuracy: {acc1:.2%} (Exact match)")
    print(f"Top-5 Accuracy: {acc5:.2%} (Correct bird in top 5 guesses)")
    print("="*40)
    
    # --- Generate Confusion Report ---
    if confusion_list:
        print("\n🔍 Most Common Confusions:")
        df_conf = pd.DataFrame(confusion_list, columns=['Actual', 'Predicted'])
        # Group by pair and count
        report = df_conf.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
        report = report.sort_values('Count', ascending=False).head(10)
        
        print(report.to_string(index=False))
        
        # Plot Bar Chart of Top Confusions
        plt.figure(figsize=(10, 6))
        sns.barplot(data=report, x='Count', y='Actual', hue='Predicted', dodge=False)
        plt.title("Top 10 Most Confusing Bird Pairs")
        plt.xlabel("Number of Errors")
        plt.tight_layout()
        
        os.makedirs("data/visuals", exist_ok=True)
        plt.savefig("data/visuals/top_errors.png")
        print("\n✅ Error chart saved to data/visuals/top_errors.png")

if __name__ == "__main__":
    analyze()
