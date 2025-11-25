import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import models
from sklearn.manifold import TSNE
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- CONFIG ---
EPOCHS = 8             # Fast but effective
BATCH_SIZE = 32
LR = 0.001
WORKERS = 0            # Safe mode
SAMPLES_TO_PLOT = 1000 # Number of dots in t-SNE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_model():
    # Use weights=None to avoid the "Hash Error" / Download issues
    model = models.efficientnet_b0(weights=None)
    # Output: 2 Classes (Song vs Call)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    return model.to(device)

def train_and_save():
    print(f"\n🎵 1. TRAINING (Song vs. Call)...")
    
    # Dataset
    full_ds = SmartBirdDataset(mode='train', task='type')
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    print(f"   Training on {len(train_ds)} samples...")

    model = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    for epoch in range(EPOCHS):
        model.train()
        correct = 0
        total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            pbar.set_postfix({'acc': (correct.double()/total).item()})
            
    # Save Model
    os.makedirs("data/models", exist_ok=True)
    save_path = "data/models/song_call_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved to {save_path}")
    return model, full_ds.classes

def generate_tsne(model, classes):
    print(f"\n🎨 2. GENERATING T-SNE VISUALIZATION...")
    
    # Load Test Data
    ds = SmartBirdDataset(mode='test', task='type')
    
    # Pick random samples
    indices = np.random.choice(len(ds), SAMPLES_TO_PLOT, replace=False)
    subset = Subset(ds, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Hook for Embeddings
    features = []
    def hook(module, input, output):
        features.append(input[0].cpu().numpy())
    
    # Attach hook to the layer BEFORE the final classifier
    handle = model.classifier[1].register_forward_hook(hook)
    
    model.eval()
    labels_list = []
    
    print("   Extracting AI features...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader):
            inputs = inputs.to(device)
            _ = model(inputs)
            labels_list.extend(labels.cpu().numpy())
            
    handle.remove()
    
    # Process Embeddings
    embeddings = np.concatenate(features)[:len(labels_list)]
    
    # Run t-SNE
    print("   Computing t-SNE (this works like a map projection)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    features_2d = tsne.fit_transform(embeddings)
    
    # Plot
    print("   Plotting...")
    plt.figure(figsize=(10, 8))
    label_names = [classes[i] for i in labels_list]
    
    sns.scatterplot(
        x=features_2d[:, 0], 
        y=features_2d[:, 1], 
        hue=label_names, 
        palette={'song': '#1f77b4', 'call': '#ff7f0e'},
        alpha=0.7,
        s=60
    )
    
    plt.title("AI Internal Representation: Song vs. Call", fontsize=16)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    os.makedirs("data/visuals", exist_ok=True)
    out_path = "data/visuals/tsne_song_call.png"
    plt.savefig(out_path, dpi=300)
    print(f"✅ Visualization saved to {out_path}")

if __name__ == "__main__":
    # 1. Train & Save
    trained_model, class_names = train_and_save()
    
    # 2. Visualize
    generate_tsne(trained_model, class_names)
