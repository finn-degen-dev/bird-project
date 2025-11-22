import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

# Import your custom dataset
from dataset import BirdRawDataset

# --- Configuration ---
RAW_DATA_DIR = "data/dataset" 
MODEL_PATH = "data/models/bird_classifier_pro.pth"
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=['train', 'test'], default='test', 
                        help="Choose 'test' for validation data or 'train' for training data.")
    args = parser.parse_args()

    # 1. Setup Data
    # We always use mode='test' here to disable random augmentation during evaluation
    # This gives us a stable "clean" accuracy score.
    target_dir = os.path.join(RAW_DATA_DIR, args.split)
    
    if not os.path.exists(target_dir):
        print(f"Error: {target_dir} not found. Did you run split_data.py?")
        return

    print(f"Initializing {args.split.upper()} Dataset...")
    dataset = BirdRawDataset(target_dir, mode='test')
    
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = dataset.classes
    print(f"Classes: {class_names}")

    # 2. Load Model
    print("Loading Pro Model...")
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except:
        model = models.efficientnet_b0(weights=None)
    
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    # Load Weights
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model.to(device)
    model.eval()

    # 3. Evaluation Loop
    all_preds = []
    all_labels = []

    print(f"Evaluating on {len(dataset)} {args.split} files...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics & Report
    print(f"\n--- Classification Report ({args.split.upper()} Data) ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(f'Confusion Matrix ({args.split.upper()})')
    plt.tight_layout()
    
    save_path = f"data/models/confusion_matrix_pro_{args.split}.png"
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")

if __name__ == "__main__":
    main()
