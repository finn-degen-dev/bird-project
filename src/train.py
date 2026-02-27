import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data import BirdRawDataset # Eure vereinte Dataset Klasse
from models import get_model
from config import *
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "resnet", "efficientnet", "embedding"], required=True)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    print(f"Starte Training für: {args.model}")
    
    # 1. Daten laden
    train_ds = BirdRawDataset(os.path.join(DATA_DIR, "dataset/train"), mode='train')
    val_ds = BirdRawDataset(os.path.join(DATA_DIR, "dataset/test"), mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    # 2. Modell laden
    model = get_model(args.model, len(train_ds.classes)).to(DEVICE)
    
    # Hier ArcFace für Embeddings oder CrossEntropy für Klassifikatoren einbauen
    if args.model == "embedding":
        from pytorch_metric_learning import losses
        criterion = losses.SubCenterArcFaceLoss(num_classes=len(train_ds.classes), embedding_size=512).to(DEVICE)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Trainingsschleife (stark vereinfacht dargestellt, nutzt eure bestehende MixUp/SpecAugment Logik hier)
    for epoch in range(args.epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            
            if args.model == "embedding":
                embeddings = model(inputs)
                loss = criterion(embeddings, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} abgeschlossen. Loss: {loss.item():.4f}")
        
    # Speichern
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{args.model}_best.pth"))
    print("Modell gespeichert!")

if __name__ == "__main__":
    main()
