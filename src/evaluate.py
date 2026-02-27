import argparse
import json
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from config import *
from data import SmartBirdDataset
from models import get_model

def evaluate_model(exp_name, model_type, use_attention, cnn_layers):
    print(f"Evaluatiere {exp_name}...")

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
        classes = json.load(f)['classes']

    val_ds = SmartBirdDataset(mode='test', task='species')
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = get_model(model_type, len(classes), use_attention, cnn_layers).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{exp_name}_best.pth"), map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [],[]
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs.to(DEVICE))
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))

    if len(classes) <= 50:
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    else:
        sns.heatmap(cm, annot=False, cmap='Blues')

    plt.title(f"Confusion Matrix ({exp_name}) - Acc: {acc:.1%}")
    plt.savefig(os.path.join(VISUALS_DIR, f"{exp_name}_confusion.png"), dpi=300)
    plt.close()

    return acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["custom", "resnet", "efficientnet"], required=True)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--cnn-layers", type=int, default=3)
    args = parser.parse_args()

    exp_name = f"{args.model}_task-species_attn-{args.use_attention}_layers-{args.cnn_layers}"
    acc = evaluate_model(exp_name, args.model, args.use_attention, args.cnn_layers)
    print(f"\n {exp_name} Accuracy: {acc:.2%}")

if __name__ == "__main__":
    main()
