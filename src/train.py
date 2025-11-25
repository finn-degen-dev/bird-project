import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights
import torchaudio.transforms as T
from torch.cuda.amp import autocast, GradScaler
from torch.hub import load_state_dict_from_url  # Needed for manual loading
from tqdm import tqdm
from collections import Counter
import ssl

# Fix for some Docker environments having strict SSL
ssl._create_default_https_context = ssl._create_unverified_context

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- PRO CONFIGURATION ---
EPOCHS = 25                  
BATCH_SIZE = 32              
EFFECTIVE_BATCH_SIZE = 64    
LR = 0.0005                  
WEIGHT_DECAY = 1e-4          
MIXUP_ALPHA = 0.4            
LABEL_SMOOTHING = 0.1        
WORKERS = 0                  

MODEL_NAME = "german_bird_classifier_pro"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM AUGMENTATIONS ---
class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.05):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, tensor):
        return tensor + torch.randn(tensor.size()).to(tensor.device) * self.std + self.mean

# Augmentation Pipeline (Applied on GPU)
augmentations = nn.Sequential(
    T.FrequencyMasking(freq_mask_param=15),
    T.TimeMasking(time_mask_param=35),
    AddGaussianNoise(std=0.03)
).to(device)

# --- MIXUP HELPER ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
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
    print(f"🚀 STARTING SOTA TRAINING: {MODEL_NAME}")
    print("   Techniques: Transfer Learning, Weighted Sampling, SpecAugment, Noise Injection")
    
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
    
    # Create Balanced Sampler
    train_indices = train_ds.indices
    all_labels = np.array(full_ds.get_all_labels())
    train_labels = all_labels[train_indices]
    
    class_counts = Counter(train_labels)
    weight_per_class = {c: 1.0 / np.sqrt(count) for c, count in class_counts.items()}
    samples_weight = torch.tensor([weight_per_class[l] for l in train_labels])
    
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # shuffle MUST be False when using sampler
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, 
                              num_workers=WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, 
                            num_workers=WORKERS, pin_memory=True)

    print(f"   Training on {len(train_ds)} files. Validating on {len(val_ds)}.")

    # 2. Model (Transfer Learning)
    print("⬇️  Loading Pretrained ImageNet Weights (Ignoring Hash Check)...")
    
    # Initialize empty model structure
    model = models.efficientnet_b0(weights=None)
    
    # Manually load weights ignoring hash mismatch
    weights_url = EfficientNet_B0_Weights.IMAGENET1K_V1.url
    try:
        state_dict = load_state_dict_from_url(weights_url, progress=True, check_hash=False)
        model.load_state_dict(state_dict)
        print("✅  Weights loaded successfully (Hash check bypassed).")
    except Exception as e:
        print(f"⚠️  Warning: Could not load pretrained weights: {e}")
        print("   Continuing with random initialization...")

    # Replace head
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(full_ds.classes))
    model = model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scaler = GradScaler()
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
            
            with autocast():
                # --- APPLY AUGMENTATIONS (GPU) ---
                if epoch < EPOCHS - 5: 
                    inputs = augmentations(inputs)

                # --- MIXUP ---
                if np.random.rand() < 0.5: 
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA, device)
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                loss = loss / accum_steps

            scaler.scale(loss).backward()
            
            if (i + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * inputs.size(0) * accum_steps
            _, preds = torch.max(outputs, 1)
            
            if 'targets_a' in locals():
                train_correct += torch.sum(preds == targets_a.data)
                del targets_a
            else:
                train_correct += torch.sum(preds == labels.data)
            
            total_train += labels.size(0)
            pbar.set_postfix({'loss': loss.item() * accum_steps})

        scheduler.step()
        train_acc = train_correct.double() / total_train
        
        # --- VALIDATION ---
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
        print(f"   Ep {epoch+1}: TrAcc: {train_acc:.2%} | ValAcc: {val_acc:.2%} | Best: {best_acc:.2%}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"data/models/{MODEL_NAME}.pth")

    duration = int(time.time() - start_time)
    print(f"\n🏆 Finished. Best Validation Accuracy: {best_acc:.2%}")

if __name__ == "__main__":
    main()
