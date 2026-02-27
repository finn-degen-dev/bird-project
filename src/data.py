import os
import torch
import pandas as pd
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter
from config import *

class SmartBirdDataset(Dataset):
    def __init__(self, mode='train', task='species', filter_sex=None, min_samples=10):
        self.mode = mode
        self.task = task
        
        df = pd.read_csv(METADATA_FILE)
        
        # Blacklist & Filter
        df = df[~df['species'].isin(["Identity unknown", "Soundscape", "Mystery bird", "?"])]
        if filter_sex:
            df = df[df['sex'].astype(str).str.lower() == filter_sex.lower()]
            
        # Seltene Klassen filtern (für Stabilität bei 600+ Klassen)
        if task == 'species':
            counts = df['species'].value_counts()
            valid_species = counts[counts >= min_samples].index
            df = df[df['species'].isin(valid_species)]

        # Nur Zeilen behalten, zu denen der Tensor wirklich existiert
        df['pt_exists'] = df['filename'].apply(
            lambda x: os.path.exists(os.path.join(PROCESSED_DIR, x.replace('.mp3', '.pt')))
        )
        self.df = df[df['pt_exists']].reset_index(drop=True)

        # Labels definieren
        if task == 'species':
            self.classes = sorted(self.df['species'].unique().tolist())
        elif task == 'type':
            self.classes = ['call', 'song']
            
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"[{mode.upper()}] Dataset geladen: {len(self.df)} Samples, {len(self.classes)} Klassen.")

    def __len__(self):
        return len(self.df)

    def _get_label(self, row):
        if self.task == 'species':
            return self.class_to_idx[row['species']]
        elif self.task == 'type':
            desc = str(row['type']).lower()
            return self.class_to_idx['song' if 'song' in desc else 'call']

    def get_all_labels(self):
        return[self._get_label(row) for _, row in self.df.iterrows()]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pt_path = os.path.join(PROCESSED_DIR, row['filename'].replace('.mp3', '.pt'))
        
        spec = torch.load(pt_path).float()
        target_width = 313 # ca. 5 Sekunden

        # Random Crop (Train) oder Center Crop (Test)
        current_width = spec.shape[1]
        if current_width > target_width:
            start = np.random.randint(0, current_width - target_width) if self.mode == 'train' else (current_width - target_width) // 2
            spec = spec[:, start : start + target_width]
        elif current_width < target_width:
            spec = torch.nn.functional.pad(spec, (0, target_width - current_width))

        # Für EfficientNet/ResNet auf 3 Kanäle und 224x224 bringen
        tensor = spec.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
        tensor = torch.nn.functional.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
        
        return tensor, self._get_label(row)

# GPU-basiertes SpecAugment (extrem schnell)
def get_spec_augmenter():
    return torch.nn.Sequential(
        T.FrequencyMasking(freq_mask_param=15),
        T.TimeMasking(time_mask_param=35)
    ).to(DEVICE)

# MixUp Logik
def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_balanced_sampler(dataset):
    labels = np.array(dataset.get_all_labels())
    counts = Counter(labels)
    weights = {cls: 1.0 / np.sqrt(c) for cls, c in counts.items()}
    sample_weights = torch.tensor([weights[l] for l in labels])
    return WeightedRandomSampler(sample_weights, len(sample_weights))
