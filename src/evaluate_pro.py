import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import your custom dataset to ensure data processing matches training exactly
from dataset import BirdRawDataset

# --- Configuration ---
# Point this to where your raw split data is
RAW_DATA_DIR = "data/dataset" 
MODEL_PATH = "data/models/bird_classifier_pro.pth"
BATCH_SIZE = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Setup Data (Use 'test' mode to disable augmentation)
    test_dir = os.path.join(RAW_DATA_DIR, "test")
    if not os.path.exists(test_dir):
        print(f"Error: {test_dir} not found. Did you run split_data.py?")
        return

    print("Initializing Test Dataset...")
    test_dataset = BirdRawDataset(test_dir, mode='test')
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")

    # 2. Load the EfficientNet Model
    print("Loading Pro Model...")
    try:
        # Try robust load
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except:
        # Fallback
        model = models.efficientnet_b0(weights=None)
    
    # Re-attach the correct classifier head
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    # Load your trained weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Evaluation Loop
    all_preds = []
    all_labels = []

    print(f"Evaluating on {len(test_dataset)} audio files...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Metrics & Report
    print("\n--- Classification Report (Pro Model) ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (Pro Model)')
    plt.tight_layout()
    
    save_path = "data/models/confusion_matrix_pro.png"
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to {save_path}")

if __name__ == "__main__":
    main()
