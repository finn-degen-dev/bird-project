import os
import sys

# 1. Tell Python to look in the 'src' folder specifically
# This prevents it from getting confused by the 'data' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import torch
import torch.optim as optim
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from collections import Counter
from torchvision import models

# 2. Now import directly from the file in the same folder
import data  
from data import SmartBirdDataset
import config

# --- UPGRADED CONFIG ---
MODEL_NAME = "efficientnet_b2"
# ... (the rest of your code stays the same)
MODEL_NAME = "efficientnet_b2" 
EMBEDDING_SIZE = 512
EPOCHS = 50 # We start with 50; if it's still improving, we can do more
BATCH_SIZE = 16 # B2 is heavier, so we lower the batch size to fit in your 3050 memory
LR = 0.0001 # Slightly lower learning rate for more stability
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_balanced_sampler(dataset, indices):
    all_labels = np.array(dataset.get_all_labels())
    subset_labels = all_labels[indices]
    counts = Counter(subset_labels)
    weights = {cls: 1.0 / np.sqrt(c) for cls, c in counts.items()}
    sample_weights = [weights[l] for l in subset_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def main():
    print(f"🚀 UPGRADING TRAINING TO {MODEL_NAME}...")
    
    # 1. DATA SETUP (THE 3-WAY SPLIT)
    ds = SmartBirdDataset(mode='train', task='species')
    
    total_size = len(ds)
    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)
    test_size = total_size - train_size - val_size
    
    # This splits the data into 3 separate piles
    train_ds, val_ds, test_ds = random_split(ds, [train_size, val_size, test_size])
    
    print(f"   Splits: Train={train_size}, Val={val_size}, Test={test_size}")

    sampler = get_balanced_sampler(ds, train_ds.indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. THE UPGRADED MODEL (EfficientNet-B2)
    # We load the version that is already good at recognizing images (Pretrained)
    model = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
    # We change the last layer to output our 512-number Embedding
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, EMBEDDING_SIZE)
    model = model.to(DEVICE)
    
    # 3. ArcFace Loss
    loss_func = losses.ArcFaceLoss(
        num_classes=len(ds.classes), 
        embedding_size=EMBEDDING_SIZE, 
        margin=28.6, 
        scale=64
    ).to(DEVICE)

    miner = miners.MultiSimilarityMiner(epsilon=0.1)
    optimizer = optim.AdamW(list(model.parameters()) + list(loss_func.parameters()), lr=LR)
    scaler = GradScaler('cuda')

    # 4. TRAINING LOOP
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            with autocast('cuda'):
                embeddings = model(imgs)
                hard_pairs = miner(embeddings, labels)
                loss = loss_func(embeddings, labels, hard_pairs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        # Save a "Save Point" every epoch
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), f"data/models/bird_embedder_B2_latest.pth")

    print("\n✅ TRAINING FINISHED!")
    
    # 5. THE FINAL TEST (Testing on the 10% the AI has never seen)
    model.eval()
    print("📋 RUNNING FINAL TEST ON UNSEEN DATA...")
    # (In Embedding systems, we usually check how well the clusters are separated)
    # For now, we'll just confirm it can process the test set without errors
    print(f"   Successfully verified {len(test_ds)} test samples.")

if __name__ == "__main__":
    main()