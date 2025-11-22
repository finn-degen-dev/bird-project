import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Configuration ---
DEFAULT_DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "data/models"
BATCH_SIZE = 32 
NUM_EPOCHS = 25       # Increased for Cosine Annealing
LEARNING_RATE = 1e-3
IMG_SIZE = 224
MIXUP_ALPHA = 0.4     # Strength of MixUp (0.2 - 0.4 is standard)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- 1. SpecAugment (The "Right" Augmentation for Audio) ---
class SpecAugment:
    """
    Randomly masks out time or frequency stripes.
    Replaces standard image cropping which distorts audio features.
    """
    def __init__(self, freq_mask_param=20, time_mask_param=30, num_masks=2):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def __call__(self, img_tensor):
        # img_tensor is (C, H, W)
        _, h, w = img_tensor.shape
        
        # Frequency Masking (Horizontal bars)
        for _ in range(self.num_masks):
            f = np.random.randint(0, self.freq_mask_param)
            f0 = np.random.randint(0, h - f)
            img_tensor[:, f0:f0+f, :] = 0.0

        # Time Masking (Vertical bars)
        for _ in range(self.num_masks):
            t = np.random.randint(0, self.time_mask_param)
            t0 = np.random.randint(0, w - t)
            img_tensor[:, :, t0:t0+t] = 0.0
            
        return img_tensor

def get_transforms():
    # Train: Resize -> Tensor -> Norm -> SpecAugment
    train_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        SpecAugment() 
    ])
    
    # Test: No Augmentation
    test_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return {'train': train_transforms, 'test': test_transforms}

# --- 2. MixUp Logic ---
def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# --- 3. Training Loop with MixUp & Cosine Scheduler ---
def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    # Apply MixUp ONLY in training
                    if phase == 'train' and np.random.random() > 0.5: # Apply mixup 50% of the time
                        inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
                        inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
                        
                        outputs = model(inputs)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                        
                        # For accuracy calc in mixup, we just take the max prediction
                        _, preds = torch.max(outputs, 1)
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                
                # Accuracy calculation (approximation for MixUp)
                if phase == 'train':
                    # During training with mixup, accuracy is fuzzy, so we just trust loss
                    running_corrects += torch.sum(preds == labels.data)
                else:
                    running_corrects += torch.sum(preds == labels.data)

            # Scheduler Step (Cosine Annealing updates every epoch)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                history['train_loss'].append(epoch_loss)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
                
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model, history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # Data Setup
    transforms_dict = get_transforms()
    image_datasets = {x: datasets.ImageFolder(os.path.join(args.data, x), transforms_dict[x]) for x in ['train', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=4) for x in ['train', 'test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)

    print(f"Training EfficientNet-B0 on {num_classes} classes: {class_names}")

    # --- 4. Model: EfficientNet B0 ---
    # EfficientNet is generally better for spectrograms than ResNet
    # EfficientNet is generally better for spectrograms than ResNet
    print("Loading EfficientNet-B0...")
    try:
        # Try standard load
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except RuntimeError as e:
        print(f"Standard load failed ({e}). Attempting to load with check_hash=False...")
        # Fallback: Load architecture first
        model = models.efficientnet_b0(weights=None)
        # Manually load weights ignoring the hash check
        url = models.EfficientNet_B0_Weights.DEFAULT.url
        state_dict = torch.hub.load_state_dict_from_url(url, check_hash=False)
        model.load_state_dict(state_dict)
    
    # Adjust Classifier Head
    # EfficientNet's classifier is usually a Sequential block, we access the last linear layer
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Train
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=args.epochs)

    # Save
    save_path = os.path.join(MODEL_SAVE_PATH, "bird_classifier_efficientnet_advanced.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Advanced Model saved to {save_path}")

    # Simple Plot
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Advanced Training Loss (MixUp + Cosine)')
    plt.legend()
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'loss_curve_advanced.png'))

if __name__ == "__main__":
    main()
