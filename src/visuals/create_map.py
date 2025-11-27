import torch
import numpy as np
import matplotlib.pyplot as plt
import umap
from torch.utils.data import DataLoader, Subset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
from model import BirdEmbedder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("🗺️  MAPPING BIODIVERSITY (UMAP)...")
    
    # 1. Load Data (Subset for speed/clarity)
    ds = SmartBirdDataset(mode='train', task='species')
    indices = np.random.choice(len(ds), 2000, replace=False) # Plot 2000 points
    loader = DataLoader(Subset(ds, indices), batch_size=32, shuffle=False)
    
    # 2. Load Model
    model = BirdEmbedder(embedding_size=512, pretrained=False)
    model.load_state_dict(torch.load("data/models/bird_embedder_sota.pth"))
    model.to(DEVICE)
    model.eval()
    
    # 3. Get Vectors
    vecs = []
    labels = []
    with torch.no_grad():
        for imgs, y in loader:
            vecs.append(model(imgs.to(DEVICE)).cpu().numpy())
            labels.extend(y.numpy())
    
    X = np.vstack(vecs)
    
    # 4. UMAP Projection
    print("   Computing 2D Projection...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine')
    embedding = reducer.fit_transform(X)
    
    # 5. Plot
    plt.figure(figsize=(12, 10))
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='Spectral', s=5, alpha=0.7)
    plt.title("The Acoustic Landscape of German Birds (UMAP)", fontsize=16)
    plt.colorbar(label='Species ID')
    plt.axis('off')
    
    os.makedirs("data/visuals", exist_ok=True)
    plt.savefig("data/visuals/biodiversity_map.png", dpi=300)
    print("✅ Map saved to data/visuals/biodiversity_map.png")

if __name__ == "__main__":
    main()
