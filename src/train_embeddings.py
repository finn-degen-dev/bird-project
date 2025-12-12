import torch
import torch.optim as optim
from pytorch_metric_learning import losses, miners
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
from model import BirdEmbedder
import config

# --- SOTA CONFIG ---
EMBEDDING_SIZE = 512
EPOCHS = 100
BATCH_SIZE = 32
LR = 0.0005
SUB_CENTERS = 1  # K=1: Standard ArcFace (One center per class) - tighter clusters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_balanced_sampler(dataset, indices):
    # (Same as before)
    all_labels = np.array(dataset.get_all_labels())
    subset_labels = all_labels[indices]
    counts = Counter(subset_labels)
    weights = {cls: 1.0 / np.sqrt(c) for cls, c in counts.items()}
    sample_weights = [weights[l] for l in subset_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def main():
    print("🚀 STARTING SOTA TRAINING (ARCFACE)...")
    print(f"   Config: {SUB_CENTERS} Sub-Centers per Class")
    
    # 1. Data Setup (Same as before)
    ds = SmartBirdDataset(mode='train', task='species')
    train_len = int(0.9 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    sampler = get_balanced_sampler(ds, train_ds.indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    
    # 2. Model
    model = BirdEmbedder(embedding_size=EMBEDDING_SIZE).to(DEVICE)
    
    # 3. THE MAGIC: ArcFace Loss
    # With K=1, this is standard ArcFace.
    loss_func = losses.SubCenterArcFaceLoss(
        num_classes=len(ds.classes), 
        embedding_size=EMBEDDING_SIZE, 
        margin=28.6, 
        scale=64, 
        sub_centers=SUB_CENTERS
    ).to(DEVICE)

    # 4. Miner (Optional but Recommended)
    # Finds the "Hardest" examples to train on (Hard Negative Mining)
    # This prevents the model from wasting time on easy examples
    miner = miners.MultiSimilarityMiner(epsilon=0.1)

    # 5. Optimizer
    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_func.parameters()), 
        lr=LR, 
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    # 6. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                embeddings = model(imgs)
                # Miner finds the hardest pairs in the batch
                hard_pairs = miner(embeddings, labels)
                # Loss is calculated only on hard pairs + SubCenters
                loss = loss_func(embeddings, labels, hard_pairs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        scheduler.step()
        
        # Save checkpoints
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/bird_embedder.pth")

if __name__ == "__main__":
    main()
