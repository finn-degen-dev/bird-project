import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- PRO CONFIGURATION ---
EPOCHS = 40                  # Increased: MixUp requires longer training
BATCH_SIZE = 32              # Fits in 6GB VRAM
EFFECTIVE_BATCH_SIZE = 64    # Gradient Accumulation steps = Effective / Batch
LR = 0.001
WEIGHT_DECAY = 1e-4          # Regularization to prevent overfitting
MIXUP_ALPHA = 0.4            # Strength of MixUp (0.0 = Off, 1.0 = Strong)
LABEL_SMOOTHING = 0.1        # Helps with noisy labels
WORKERS = 0                  # Safe mode for Docker

MODEL_NAME = "german_bird_classifier_pro"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MIXUP HELPER ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    print(f"🚀 STARTING ADVANCED TRAINING: {MODEL_NAME}")
    print(f"   Configs: MixUp={MIXUP_ALPHA}, Smoothing={LABEL_SMOOTHING}, AMP=True")
    
    # 1. Dataset
    full_ds = SmartBirdDataset(mode='train', task='species')
    
    # Save Class Mapping
    mapping = {'classes': full_ds.classes}
    os.makedirs("data/models", exist_ok=True)
    with open(f"data/models/{MODEL_NAME}_classes.json", "w") as f:
        json.dump(mapping, f)
    
    # 90/10 Split
    train_size = int(0.9 * len(full_ds))
    val_size = len(full_ds) - train_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=WORKERS, pin_memory=True)

    print(f"   Training on {len(train_ds)} files. Validating on {len(val_ds)}.")

    # 2. Model (EfficientNet B0)
    # Note: If you had more VRAM, we'd use EfficientNet B2 or B3 here
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(full_ds.classes))
    model = model.to(device)

    # 3. Optimizer & Scheduler
    # AdamW is better than Adam for generalizing
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Cosine Annealing: Drops LR smoothly, helps find better minima
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Loss with Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    
    # Mixed Precision Scaler
    scaler = GradScaler()
    
    # Gradient Accumulation
    accum_steps = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
    
    best_acc = 0.0

    # --- TRAINING LOOP ---
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total_train = 0
        
        optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Train]", leave=False)
        
        for i, (inputs, labels) in enumerate(pbar):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Automatic Mixed Precision context
            with autocast():
                # Apply MixUp
                if np.random.rand() < 0.5: # 50% chance to mix
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA, device)
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                # Normalize loss for accumulation
                loss = loss / accum_steps

            # Scale loss and backward
            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # Metrics (Approximate for MixUp)
            train_loss += loss.item() * inputs.size(0) * accum_steps
            _, preds = torch.max(outputs, 1)
            # For accuracy logic with MixUp, we compare to the "dominant" label (a)
            if 'targets_a' in locals():
                train_correct += torch.sum(preds == targets_a.data)
                del targets_a # Cleanup
            else:
                train_correct += torch.sum(preds == labels.data)
            
            total_train += labels.size(0)
            pbar.set_postfix({'loss': loss.item() * accum_steps})

        # Step Scheduler
        scheduler.step()
        
        train_acc = train_correct.double() / total_train
        avg_train_loss = train_loss / total_train

        # --- VALIDATION (No MixUp, No AMP needed for simple val) ---
        model.eval()
        val_correct = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Ep {epoch+1}/{EPOCHS} [Val]", leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data)
                total_val += labels.size(0)
        
        val_acc = val_correct.double() / total_val
        
        # Print Stats
        print(f"   Ep {epoch+1}: TrAcc: {train_acc:.2%} | ValAcc: {val_acc:.2%} | Best: {best_acc:.2%}")
        
        # Save Best
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"data/models/{MODEL_NAME}.pth")

    duration = int(time.time() - start_time)
    print(f"\n🏆 Training Finished in {duration//60}m {duration%60}s.")
    print(f"   Final Best Validation Accuracy: {best_acc:.2%}")
    print(f"   Model saved to data/models/{MODEL_NAME}.pth")

if __name__ == "__main__":
    main()
