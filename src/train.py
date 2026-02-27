import argparse
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from config import *
from data import SmartBirdDataset, get_balanced_sampler, get_spec_augmenter, mixup_data, mixup_criterion
from models import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "resnet", "efficientnet", "embedding"], required=True)
    parser.add_argument("--task", choices=["species", "type"], default="species", help="'type' für Song vs Call Experiment")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-attention", action="store_true", help="Aktiviert Attention im Custom CNN")
    parser.add_argument("--cnn-layers", type=int, default=3, help="Anzahl Layers (nur Custom CNN)")
    args = parser.parse_args()

    exp_name = f"{args.model}_task-{args.task}_attn-{args.use_attention}_layers-{args.cnn_layers}"
    print(f" Starte Experiment: {exp_name}")

    full_ds = SmartBirdDataset(mode='train', task=args.task)

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "w") as f:
        json.dump({'classes': full_ds.classes}, f)

    train_size = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])

    from collections import Counter
    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    print(" Berechne Klassen-Gewichte...")
    all_labels = full_ds.get_all_labels()  # Nur EINMAL aufrufen!
    train_labels = [all_labels[i] for i in train_ds.indices]

    counts = Counter(train_labels)
    weights = {cls: 1.0 / np.sqrt(c) for cls, c in counts.items()}
    sample_weights = torch.tensor([weights[l] for l in train_labels])

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = get_model(args.model, len(full_ds.classes), args.use_attention, args.cnn_layers).to(DEVICE)

    if args.model == "embedding":
        from pytorch_metric_learning import losses
        criterion = losses.SubCenterArcFaceLoss(num_classes=len(full_ds.classes), embedding_size=512).to(DEVICE)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()
    spec_augmenter = get_spec_augmenter()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast():

                inputs = spec_augmenter(inputs)

                if args.model != "embedding" and torch.rand(1).item() > 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if args.model != "embedding":
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{correct/total:.2%}"})

        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        if args.model != "embedding":
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            print(f"   => Val Acc: {val_acc:.2%}")

            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{exp_name}_best.pth"))
        else:

            torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{exp_name}_best.pth"))

if __name__ == "__main__":
    main()
