import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Config ---
DATA_DIR = "data/processed/test"
MODEL_PATH = "data/models/bird_classifier_efficientnet_advanced.pth"
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # 1. Setup Data
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    test_dataset = datasets.ImageFolder(DATA_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    class_names = test_dataset.classes
    print(f"Classes: {class_names}")

    # 2. Load the EfficientNet Architecture
    print("Loading Model...")
    # robust load pattern
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except:
        model = models.efficientnet_b0(weights=None)
    
    # Recreate the classifier head exactly as in training
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    
    # Load the trained weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # 3. Run Predictions
    all_preds = []
    all_labels = []

    print("Running Evaluation...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 4. Report
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix (EfficientNet-B0)')
    plt.tight_layout()
    plt.savefig("data/models/confusion_matrix_advanced.png")
    print("Saved confusion matrix to data/models/confusion_matrix_advanced.png")

if __name__ == "__main__":
    main()
