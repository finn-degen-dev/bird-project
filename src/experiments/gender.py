import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
import time
from tqdm import tqdm  # Added progress bar

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- FAST CONFIG ---
EPOCHS = 10
BATCH_SIZE = 32        # Increased for speed
LR = 0.001
WORKERS = 0            # Validated by your stress test

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # tqdm provides a progress bar per epoch
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels.data)
        total += labels.size(0)
        
        # Update progress bar with current accuracy
        pbar.set_postfix({'loss': loss.item(), 'acc': (correct.double()/total).item()})
        
    return running_loss / total, correct.double() / total

def run_experiment(name, filter_sex=None):
    print(f"\n🚀 STARTING EXPERIMENT: {name} [Sex Filter: {filter_sex}]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Using Device: {device}")
    
    # 1. Dataset
    try:
        ds = SmartBirdDataset(mode='train', filter_sex=filter_sex, task='species')
    except ValueError:
        print("   ❌ Not enough data for this filter.")
        return []

    # 2. Optimized DataLoader
    loader = DataLoader(
        ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=WORKERS, 
        pin_memory=True,      # Fast transfer to GPU
        persistent_workers=False # Keeps workers alive between epochs
    )
    print(f"   Data: {len(ds)} samples | {len(ds.classes)} classes")

    # 3. Model
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(ds.classes))
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Train Loop
    acc_history = []
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(model, loader, criterion, optimizer, device)
        acc_history.append(acc.item())
        print(f"   Epoch {epoch+1}/{EPOCHS} | Acc: {acc:.2%}")
        
    duration = time.time() - start_time
    print(f"   ⏱️ Total Time: {int(duration)}s ({duration/EPOCHS:.1f}s per epoch)")
    return acc_history

def main():
    # 1. Baseline (All Data)
    history_all = run_experiment("Baseline (All)", filter_sex=None)
    
    # 2. Experiment (Male Only)
    history_male = run_experiment("Male Only", filter_sex="male")
    
    # 3. Plotting
    if history_all and history_male:
        plt.figure(figsize=(10, 6))
        plt.plot(history_all, label='Baseline (All Data)', marker='o', linewidth=2)
        plt.plot(history_male, label='Male Only', marker='s', linewidth=2)
        
        plt.title("Does Filtering for Male Birds Improve Learning?")
        plt.xlabel("Epoch")
        plt.ylabel("Training Accuracy")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        os.makedirs("data/visuals", exist_ok=True)
        out_path = "data/visuals/experiment_gender_result.png"
        plt.savefig(out_path)
        print(f"\n✅ Graph saved to {out_path}")

if __name__ == "__main__":
    main()
