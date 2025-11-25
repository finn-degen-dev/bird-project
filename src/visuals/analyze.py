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
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- CONFIG ---
MODEL_PATH = "data/models/german_bird_classifier_pro.pth"
JSON_PATH = "data/models/german_bird_classifier_pro_classes.json"
WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_classes():
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
        classes = data['classes']
    
    print(f"Loading model from {MODEL_PATH}...")
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model, classes

def sliding_window_inference(model, full_spectrogram):
    """
    Scans the entire audio file with a sliding window.
    Returns the averaged probability vector across all windows.
    """
    # full_spectrogram shape: (1, 3, 128, Time)
    spec = full_spectrogram.squeeze(0) # (3, 128, Time)
    time_steps = spec.shape[2]
    
    window_size = 313 # ~5 seconds
    stride = 157      # 50% overlap
    
    windows = []
    
    # 1. Slide across time
    for start in range(0, time_steps - window_size + 1, stride):
        end = start + window_size
        crop = spec[:, :, start:end]
        
        # Resize to 224x224 (EfficientNet input)
        # Note: We do this manually here because we are in 'full' mode
        crop = torch.nn.functional.interpolate(
            crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        )
        windows.append(crop)
    
    if not windows:
        # File shorter than 5s? Padding handled in dataset, but just in case
        return None

    # 2. Batch predictions
    batch = torch.cat(windows, dim=0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(batch)
        probs = F.softmax(logits, dim=1)
    
    # 3. Aggregation Strategy: MAX Pooling
    # "If the bird sings in ANY window, it's a match."
    # Winning solutions often use MAX or Average of Top-K.
    max_probs, _ = torch.max(probs, dim=0)
    
    return max_probs

def analyze():
    model, classes = load_model_and_classes()
    
    # Load Validation Data using 'full' mode (No random cropping)
    full_ds = SmartBirdDataset(mode='full', task='species')
    
    # Recreate split (Needs to match train split seed ideally)
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    _, val_ds = random_split(full_ds, [train_size, val_size])
    
    # Batch size MUST be 1 because tensors vary in length
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=WORKERS)
    
    print(f"📊 Analyzing {len(val_ds)} validation samples using SLIDING WINDOW inference...")
    
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    confusion_list = [] 

    for inputs, labels in tqdm(loader):
        # inputs is (1, 3, 128, Variable_Time)
        # labels is (1)
        
        label = labels.item()
        
        # Run Sliding Window
        pred_probs = sliding_window_inference(model, inputs)
        
        if pred_probs is None:
            continue
            
        # --- Top-1 ---
        pred_1 = torch.argmax(pred_probs).item()
        if pred_1 == label:
            correct_top1 += 1
            
        # --- Top-5 ---
        _, top5_indices = torch.topk(pred_probs, 5)
        if label in top5_indices.tolist():
            correct_top5 += 1
            
        # --- Errors ---
        if pred_1 != label:
            confusion_list.append((classes[label], classes[pred_1]))
            
        total += 1

    acc1 = correct_top1 / total
    acc5 = correct_top5 / total
    
    print("\n" + "="*40)
    print(f"🏆 FINAL SLIDING WINDOW RESULTS")
    print(f"Top-1 Accuracy: {acc1:.2%}")
    print(f"Top-5 Accuracy: {acc5:.2%}")
    print("="*40)
    
    if confusion_list:
        print("\n🔍 Most Common Confusions:")
        df_conf = pd.DataFrame(confusion_list, columns=['Actual', 'Predicted'])
        report = df_conf.groupby(['Actual', 'Predicted']).size().reset_index(name='Count')
        report = report.sort_values('Count', ascending=False).head(10)
        print(report.to_string(index=False))
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=report, x='Count', y='Actual', hue='Predicted', dodge=False)
        plt.title("Top 10 Errors (Sliding Window)")
        plt.tight_layout()
        os.makedirs("data/visuals", exist_ok=True)
        plt.savefig("data/visuals/top_errors.png")

if __name__ == "__main__":
    analyze()
