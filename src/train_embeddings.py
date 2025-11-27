import torch
import torch.optim as optim
from pytorch_metric_learning import losses
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler, Subset
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
from model import BirdEmbedder
import config

# --- CONFIG ---
EMBEDDING_SIZE = 512
EPOCHS = 250
BATCH_SIZE = 32
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SNAPSHOT_INTERVAL = 5  # Save a visual snapshot every N epochs

def get_balanced_sampler(dataset, indices):
    all_labels = np.array(dataset.get_all_labels())
    subset_labels = all_labels[indices]
    counts = Counter(subset_labels)
    weights = {cls: 1.0 / np.sqrt(c) for cls, c in counts.items()}
    sample_weights = [weights[l] for l in subset_labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def save_loss_curve(losses_list):
    plt.figure(figsize=(10, 5))
    plt.plot(losses_list, label='ArcFace Loss', color='#2c3e50', linewidth=2)
    plt.title('Training Dynamics: ArcFace Loss Reduction', fontsize=14)
    plt.xlabel('Batch Iterations')
    plt.ylabel('Angular Margin Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("data/visuals/training_loss_curve.png")
    plt.close()

def save_latent_snapshot(model, dataset, epoch):
    """Uses PCA (Fast) to show clustering progress"""
    # Use a small subset for speed
    indices = np.random.choice(len(dataset), 1000, replace=False)
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=32, shuffle=False)
    
    model.eval()
    vecs, labels = [], []
    
    with torch.no_grad():
        for imgs, y in loader:
            imgs = imgs.to(DEVICE)
            embeddings = model(imgs)
            vecs.append(embeddings.cpu().float().numpy())
            labels.extend(y.numpy())
    
    X = np.vstack(vecs)
    
    # PCA for speed (UMAP is too slow for training loop)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10, alpha=0.6)
    plt.title(f'Latent Space Evolution - Epoch {epoch}', fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f"data/visuals/evolution_epoch_{epoch:02d}.png")
    plt.close()
    model.train()

def main():
    print("🧠 STARTING METRIC LEARNING (ARCFACE)...")
    print(f"   Visuals will be saved to data/visuals/ every {SNAPSHOT_INTERVAL} epochs.")
    
    os.makedirs("data/visuals", exist_ok=True)
    
    # 1. Data
    ds = SmartBirdDataset(mode='train', task='species')
    train_len = int(0.9 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    
    sampler = get_balanced_sampler(ds, train_ds.indices)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)
    
    # 2. Model & Loss
    model = BirdEmbedder(embedding_size=EMBEDDING_SIZE).to(DEVICE)
    
    # ArcFace: margin=28.6 degrees ~ 0.5 radians
    loss_func = losses.ArcFaceLoss(
        num_classes=len(ds.classes), 
        embedding_size=EMBEDDING_SIZE, 
        margin=28.6, 
        scale=64
    ).to(DEVICE)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(loss_func.parameters()), 
        lr=LR, 
        weight_decay=1e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()

    history_loss = []

    # 3. Initial Snapshot (Chaos)
    print("📸  Taking baseline snapshot...")
    save_latent_snapshot(model, val_ds, 0)

    # 4. Training Loop
    for epoch in range(1, EPOCHS + 1):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            with autocast():
                embeddings = model(imgs)
                loss = loss_func(embeddings, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            val = loss.item()
            history_loss.append(val)
            epoch_loss += val
            batch_count += 1
            
            pbar.set_postfix({'loss': f"{val:.4f}"})
            
        scheduler.step()
        
        # Save Model
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), "data/models/bird_embedder.pth")
        
        # Snapshots
        if epoch % SNAPSHOT_INTERVAL == 0 or epoch == EPOCHS:
            print(f"📸  Saving visuals for Epoch {epoch}...")
            save_latent_snapshot(model, val_ds, epoch)
            save_loss_curve(history_loss)

    print("\n✅ Training Complete. Check data/visuals/ for the evolution GIF!")

if __name__ == "__main__":
    main()
