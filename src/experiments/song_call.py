import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import time
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset
import config

# --- CONFIG ---
EPOCHS = 10
BATCH_SIZE = 32
LR = 0.001
WORKERS = 0  # Safe mode

def train_model():
    print(f"\n🎵 STARTING EXPERIMENT: Song vs. Call Classification")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Initialize Dataset with task='type'
    # This automatically sets labels to 0 (Call) and 1 (Song)
    full_ds = SmartBirdDataset(mode='train', task='type')
    
    # 2. Split into Train (80%) and Test (20%)
    # We need a test set to draw the confusion matrix later
    train_size = int(0.8 * len(full_ds))
    test_size = len(full_ds) - train_size
    train_ds, test_ds = random_split(full_ds, [train_size, test_size])
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=True)
    
    print(f"   Data: {len(train_ds)} Training samples | {len(test_ds)} Test samples")
    print(f"   Classes: {full_ds.classes} (0=Call, 1=Song)")

    # 3. Model (Binary Classification)
    model = models.efficientnet_b0(weights=None)
    # Output is 2 classes (Song vs Call)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 4. Training Loop
    history = []
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
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
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels.data)
            total += labels.size(0)
            
            pbar.set_postfix({'acc': (correct.double()/total).item()})
            
        epoch_acc = correct.double() / total
        print(f"   Epoch {epoch+1}: Train Acc: {epoch_acc:.2%}")
        history.append(epoch_acc.item())

    # 5. Final Evaluation & Confusion Matrix
    print("\n📊 Evaluating on Test Set...")
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Test Accuracy
    correct_test = sum(np.array(all_preds) == np.array(all_labels))
    test_acc = correct_test / len(all_labels)
    print(f"   🏆 Final Test Accuracy: {test_acc:.2%}")

    # Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=full_ds.classes, yticklabels=full_ds.classes)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Song vs. Call Classification (Acc: {test_acc:.1%})')
    
    os.makedirs("data/visuals", exist_ok=True)
    plt.savefig("data/visuals/confusion_matrix_type.png")
    print("✅ Confusion Matrix saved to data/visuals/confusion_matrix_type.png")

if __name__ == "__main__":
    train_model()
