import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
from model import BirdEmbedder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WINDOW_SIZE = 313  # ~5 seconds
STRIDE = 157       # 50% overlap

def get_windows(full_spec):
    """Slices spectrogram into windows."""
    spec = full_spec.squeeze(0) 
    windows = []
    for start in range(0, spec.shape[2] - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        crop = spec[:, :, start:end]
        # Resize to 224x224
        crop = F.interpolate(crop.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        windows.append(crop)
    if not windows: return None
    return torch.cat(windows, dim=0)

def build_memory(model, loader):
    vecs, lbls = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="🧠 Memorizing"):
            vecs.append(model(imgs.to(DEVICE)).cpu().numpy())
            lbls.append(labels.numpy())
    return np.vstack(vecs), np.concatenate(lbls)

def main():
    print("🔬 VALIDATING: SLIDING WINDOW + EMBEDDINGS")
    
    # 1. Load Model
    model = BirdEmbedder(embedding_size=512, pretrained=False)
    model.load_state_dict(torch.load("data/models/bird_embedder_sota.pth"))
    model.to(DEVICE).eval()

    # 2. Build Memory (Training Data)
    train_ds = SmartBirdDataset(mode='train', task='species')
    train_len = int(0.9 * len(train_ds))
    # Use the same split seed/logic as training to avoid leakage in a real paper, 
    # but for this demo, we just need a big reference database.
    mem_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
    
    # Limit memory size for speed (optional, remove [:10000] for full accuracy)
    # Using full memory is slower but more accurate
    print("1️⃣  Building Reference Database...")
    X_mem, y_mem = build_memory(model, mem_loader)
    
    print("2️⃣  Training Search Engine (KNN)...")
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=-1)
    knn.fit(X_mem, y_mem)

    # 3. Validate (Test Data - Full Length)
    val_ds = SmartBirdDataset(mode='full', task='species')
    val_len = int(0.1 * len(val_ds))
    _, val_split = random_split(val_ds, [len(val_ds)-val_len, val_len])
    val_loader = DataLoader(val_split, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"3️⃣  Scanning {len(val_split)} files (Sliding Window)...")
    correct = 0
    total = 0
    
    for full_spec, label in tqdm(val_loader):
        windows = get_windows(full_spec)
        if windows is None: continue
        
        # Get embedding for every window in the file
        with torch.no_grad():
            window_vecs = model(windows.to(DEVICE)).cpu().numpy()
        
        # Find the closest match for EVERY window
        dists, indices = knn.kneighbors(window_vecs)
        
        # The Logic: "The minimum distance found in the entire file represents the bird"
        best_idx = np.argmin(dists) 
        pred_label = y_mem[indices[best_idx][0]]
        
        if pred_label == label.item():
            correct += 1
        total += 1

    print(f"\n🏆 TRUE ACCURACY: {correct/total:.2%}")

if __name__ == "__main__":
    main()
