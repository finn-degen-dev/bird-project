import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
from model import BirdEmbedder
import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def extract_embeddings(model, loader):
    """Passes data through model to get vectors."""
    vecs = []
    lbls = []
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Extracting"):
            imgs = imgs.to(DEVICE)
            embeddings = model(imgs)
            vecs.append(embeddings.cpu().numpy())
            lbls.append(labels.numpy())
    return np.vstack(vecs), np.concatenate(lbls)

def main():
    print("🔬 VALIDATING EMBEDDING SPACE...")
    
    # 1. Load Data
    ds = SmartBirdDataset(mode='train', task='species')
    train_len = int(0.9 * len(ds))
    val_len = len(ds) - train_len
    train_ds, val_ds = random_split(ds, [train_len, val_len])
    
    # We need the training data to build the "Memory" (Reference Database)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # 2. Load Model
    model = BirdEmbedder(embedding_size=512, pretrained=False)
    model.load_state_dict(torch.load("data/models/bird_embedder.pth"))
    model.to(DEVICE)
    model.eval()
    
    # 3. Extract Vectors
    print("1️⃣  Building Reference Database (Memory)...")
    X_train, y_train = extract_embeddings(model, train_loader)
    
    print("2️⃣  Encoding Validation Data...")
    X_val, y_val = extract_embeddings(model, val_loader)
    
    # 4. K-Nearest Neighbor Classification
    # This simulates: "Find the closest vector in memory"
    print("3️⃣  Running KNN Classifier...")
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine')
    knn.fit(X_train, y_train)
    
    preds = knn.predict(X_val)
    acc = accuracy_score(y_val, preds)
    
    print("\n" + "="*40)
    print(f"🧬 EMBEDDING SYSTEM RESULTS")
    print(f"Metric: 1-Nearest Neighbor Accuracy (Cosine)")
    print(f"Score:  {acc:.2%}")
    print("="*40)
    print("Interpretation: If this is high (>90%), your model has successfully")
    print("learned a 'Map of Biodiversity' rather than just memorizing classes.")

if __name__ == "__main__":
    main()
