import os
import time
import copy
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --- Configuration ---
DEFAULT_DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "data/models"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LEARNING_RATE = 0.001
IMG_SIZE = 224  # ResNet standard input size

# Detect Device (GPU if available)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_data_transforms():
    """
    Define transformations for the images.
    ResNet expects 224x224.
    """
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            # Simple augmentation: Randomly crop to make model robust to slight shifts
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            # Normalize with ImageNet mean/std (since we use transfer learning)
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    return data_transforms

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_loss_history = []
    val_loss_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                # Track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            else:
                val_loss_history.append(epoch_loss)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best one so far
            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history

def evaluate_model(model, dataloaders, class_names):
    """
    Generate detailed metrics and confusion matrix.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Classification Report
    print("\n--- Classification Report ---")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'confusion_matrix.png'))
    print(f"Confusion Matrix saved to {MODEL_SAVE_PATH}/confusion_matrix.png")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_DIR, help="Path to processed data (containing train/test folders)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    args = parser.parse_args()

    # Create model output directory
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    # 1. Load Data
    data_transforms = get_data_transforms()
    
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, x), data_transforms[x]) 
                      for x in ['train', 'test']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                                                  shuffle=True, num_workers=4)
                   for x in ['train', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"Classes found: {class_names}")
    print(f"Training on {dataset_sizes['train']} images, Validating on {dataset_sizes['test']} images.")
    print(f"Using device: {device}")

    # 2. Setup Model (Transfer Learning: ResNet18)
    # We use ResNet18 because it's lightweight and efficient
    print("Loading ResNet18 pre-trained model...")
    weights = models.ResNet18_Weights.DEFAULT
    model_ft = models.resnet18(weights=weights)

    # Modify the last layer to match our number of bird species
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)

    # 3. Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # We optimize all parameters
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE)

    # 4. Train
    model_ft, t_hist, v_hist = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=args.epochs)

    # 5. Save Model
    save_path = os.path.join(MODEL_SAVE_PATH, "bird_classifier_resnet18.pth")
    torch.save(model_ft.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 6. Evaluation
    evaluate_model(model_ft, dataloaders, class_names)

    # 7. Plot Loss Curves
    plt.figure()
    plt.plot(t_hist, label='Train Loss')
    plt.plot(v_hist, label='Val Loss')
    plt.title('Training Metrics')
    plt.legend()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'loss_curve.png'))

if __name__ == "__main__":
    main()
