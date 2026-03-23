import argparse
import time
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from config import *
from data import SmartBirdDataset, get_balanced_sampler, get_spec_augmenter, mixup_data, mixup_criterion
from models import get_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["custom", "resnet", "efficientnet", "mobilenet", "embedding"], required=True)
    parser.add_argument("--task", choices=["species", "type"], default="species", help="'type' für Song vs Call Experiment")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, choices=[1, 3], default=3, help="1 für Graustufen, 3 für RGB")
    parser.add_argument("--img-size", type=int, default=224, help="Auflösung des Spektrogramms (für Latenz-Experimente)")
    parser.add_argument("--use-attention", action="store_true", help="Aktiviert Attention im Custom CNN")
    parser.add_argument("--cnn-layers", type=int, default=3, help="Anzahl Layers (nur Custom CNN)")
    args = parser.parse_args()

    exp_name = f"{args.model}_task-{args.task}_ch-{args.in_channels}_size-{args.img_size}_attn-{args.use_attention}_layers-{args.cnn_layers}"
    print(f" Starte Experiment: {exp_name}")

    full_ds = SmartBirdDataset(mode='train', task=args.task, in_channels=args.in_channels, img_size=args.img_size)

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "w") as f:
        json.dump({'classes': full_ds.classes}, f)

    train_size = int(0.9 * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])

    from collections import Counter
    import numpy as np
    from torch.utils.data import WeightedRandomSampler

    print(" Berechne Klassen-Gewichte...")
    all_labels = full_ds.get_all_labels()
    train_labels = [all_labels[i] for i in train_ds.indices]

    counts = Counter(train_labels)
    weights = {cls: 1.0 / np.sqrt(c) for cls, c in counts.items()}
    sample_weights = torch.tensor([weights[l] for l in train_labels])

    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    model = get_model(args.model, len(full_ds.classes), args.use_attention, args.cnn_layers, args.in_channels).to(DEVICE)

    with torch.no_grad():
        dummy_input = torch.randn(2, args.in_channels, args.img_size, args.img_size).to(DEVICE)
        model(dummy_input)

    if args.model == "embedding":
        from pytorch_metric_learning import losses
        criterion = losses.ArcFaceLoss(num_classes=len(full_ds.classes), embedding_size=512).to(DEVICE)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = optim.AdamW(list(model.parameters()) + list(criterion.parameters()), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
    spec_augmenter = get_spec_augmenter()

    best_metric = 0.0 if args.model != "embedding" else float('inf')

    LOGS_DIR = os.path.join(VISUALS_DIR, "logs")
    os.makedirs(LOGS_DIR, exist_ok=True)
    history_log = []

    for epoch in range(args.epochs):
        model.train()
        train_loss, correct, total = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                inputs = spec_augmenter(inputs)

                if args.model not in ["embedding", "custom"] and torch.rand(1).item() > 0.5:
                    inputs, targets_a, targets_b, lam = mixup_data(inputs, labels)
                    outputs = model(inputs)
                    loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                else:
                    outputs = model(inputs)
                    if args.model != "embedding":
                        loss = criterion(outputs, labels)

            if args.model == "embedding":
                loss = criterion(outputs.float(), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            if args.model != "embedding":
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                pbar.set_postfix({'loss': f"{loss.item():.3f}", 'acc': f"{correct/total:.2%}"})
            else:
                pbar.set_postfix({'loss': f"{loss.item():.3f}"})

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        epoch_stats = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss
        }

        if args.model != "embedding":
            val_correct, val_total, val_loss = 0, 0, 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = torch.max(outputs, 1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            val_acc = val_correct / val_total
            avg_val_loss = val_loss / len(val_loader)
            print(f"   => Val Acc: {val_acc:.2%} | Val Loss: {avg_val_loss:.4f}")

            epoch_stats["val_acc"] = val_acc * 100
            epoch_stats["val_loss"] = avg_val_loss

            if val_acc > best_metric:
                best_metric = val_acc
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{exp_name}_best.pth"))
        else:
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs.float(), labels)
                    val_loss += loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"   => Val ArcFace Loss: {avg_val_loss:.4f}")

            epoch_stats["val_loss"] = avg_val_loss

            if avg_val_loss < best_metric:
                best_metric = avg_val_loss
                torch.save(model.state_dict(), os.path.join(MODELS_DIR, f"{exp_name}_best.pth"))
                torch.save(criterion.state_dict(), os.path.join(MODELS_DIR, f"{exp_name}_arcface.pth"))

        history_log.append(epoch_stats)
        with open(os.path.join(LOGS_DIR, f"{exp_name}_history.json"), "w") as f:
            json.dump(history_log, f, indent=4)

if __name__ == "__main__":
    main()
