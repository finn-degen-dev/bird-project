import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt

# Import our new Custom Dataset
from dataset import BirdRawDataset

# --- Configuration ---
RAW_DATA_DIR = "data/dataset" # Contains 'train' and 'test' folders with MP3s
MODEL_SAVE_PATH = "data/models"
BATCH_SIZE = 32 # Might need to lower to 16 if "Out of Memory" occurs
NUM_EPOCHS = 25
LEARNING_RATE = 1e-3
MIXUP_ALPHA = 0.4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- SpecAugment (Still useful even with Audio Augmentation) ---
class SpecAugment(nn.Module):
    def __init__(self, freq_mask_param=20, time_mask_param=30, num_masks=2):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_masks = num_masks

    def forward(self, img_tensor):
        # img_tensor is (Batch, C, H, W)
        B, C, H, W = img_tensor.shape
        
        # Create a copy to avoid in-place modification errors
        out = img_tensor.clone()
        
        for i in range(B):
            # Frequency Masking
            for _ in range(self.num_masks):
                f = np.random.randint(0, self.freq_mask_param)
                f0 = np.random.randint(0, H - f)
                out[i, :, f0:f0+f, :] = 0.0

            # Time Masking
            for _ in range(self.num_masks):
                t = np.random.randint(0, self.time_mask_param)
                t0 = np.random.randint(0, W - t)
                out[i, :, :, t0:t0+t] = 0.0
        return out

# --- MixUp Logic ---
def mixup_data(x, y, alpha=1.0):
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

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, spec_augmenter, num_epochs=25):
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

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    if phase == 'train':
                        # 1. Apply SpecAugment (on GPU)
                        inputs = spec_augmenter(inputs)
                        
                        # 2. Apply MixUp (50% chance)
                        if np.random.random() > 0.5:
                            inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, MIXUP_ALPHA)
                            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
                            outputs = model(inputs)
                            loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                            _, preds = torch.max(outputs, 1)
                        else:
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                            _, preds = torch.max(outputs, 1)
                        
                        loss.backward()
                        optimizer.step()
                        scheduler.step() # Step every batch for OneCycle/Cosine
                    else:
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

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
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    
    # 1. Datasets (Now using Raw Audio MP3s)
    train_dir = os.path.join(RAW_DATA_DIR, "train")
    test_dir = os.path.join(RAW_DATA_DIR, "test")
    
    if not os.path.exists(train_dir):
        print(f"Error: Could not find {train_dir}. Did you run 'src/split_data.py'?")
        return

    train_dataset = BirdRawDataset(train_dir, mode='train')
    test_dataset = BirdRawDataset(test_dir, mode='test')
    
    # 2. DataLoaders
    # num_workers is crucial here because we process audio on the CPU
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True),
        'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)
    }
    
    dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
    class_names = train_dataset.classes
    print(f"Classes: {class_names}")
    print(f"Training on {len(train_dataset)} audio files. Validation on {len(test_dataset)} files.")

    # 3. Model Setup (Robust Load)
    print("Loading EfficientNet-B0...")
    try:
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
    except:
        model = models.efficientnet_b0(weights=None)
        
    num_ftrs = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    # 4. Training Components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # OneCycleLR is often better for "Super Convergence" with heavy augmentation
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LEARNING_RATE, 
                                              steps_per_epoch=len(dataloaders['train']), 
                                              epochs=args.epochs)
    
    spec_augmenter = SpecAugment().to(device)

    # 5. Run Training
    model, history = train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, spec_augmenter, num_epochs=args.epochs)

    # 6. Save
    save_path = os.path.join(MODEL_SAVE_PATH, "bird_classifier_pro.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Pro Model saved to {save_path}")

    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Pro Training Loss (Audio Augment + MixUp)')
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'loss_curve_pro.png'))

if __name__ == "__main__":
    main()
