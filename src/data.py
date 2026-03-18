import os
import torch
import pandas as pd
import numpy as np
import torchaudio.transforms as T
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import Counter
from config import *

class SmartBirdDataset(Dataset):
    def __init__(self, mode='train', task='species', filter_sex=None, min_samples=10, in_channels=3):
        self.mode = mode
        self.task = task
        self.in_channels = in_channels

        if not os.path.exists(METADATA_FILE):
            raise FileNotFoundError(f"Die Datei {METADATA_FILE} fehlt. Bitte 'python src/data_prep.py' ausführen.")

        df = pd.read_csv(METADATA_FILE)

        df = df[~df['species'].isin(["Identity unknown", "Soundscape", "Mystery bird", "?"])]

        if filter_sex:
            df = df[df['sex'].astype(str).str.lower() == filter_sex.lower()]

        if task == 'species':
            counts = df['species'].value_counts()
            valid_species = counts[counts >= min_samples].index
            df = df[df['species'].isin(valid_species)]

        df['pt_exists'] = df['filename'].apply(
            lambda x: os.path.exists(os.path.join(PROCESSED_DIR, x.replace('.mp3', '.pt')))
        )
        self.df = df[df['pt_exists']].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("Keine gültigen Tensoren gefunden! Stimmen die Pfade?")

        if task == 'species':
            self.classes = sorted(self.df['species'].unique().tolist())
        elif task == 'type':
            self.classes = ['call', 'song']

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        print(f"[{mode.upper()}] Dataset bereit: {len(self.df)} Samples, {len(self.classes)} Klassen (Kanäle: {self.in_channels}).")

    def __len__(self):
        return len(self.df)

    def _get_label(self, row):
        if self.task == 'species':
            return self.class_to_idx[row['species']]
        elif self.task == 'type':
            desc = str(row['type']).lower()
            return self.class_to_idx['song' if 'song' in desc else 'call']

    def get_all_labels(self):
        return [self._get_label(row) for _, row in self.df.iterrows()]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pt_path = os.path.join(PROCESSED_DIR, row['filename'].replace('.mp3', '.pt'))

        spec = torch.load(pt_path).float()
        current_width = spec.shape[1]

        if current_width > TARGET_WIDTH:
            if self.mode == 'train':
                start = np.random.randint(0, current_width - TARGET_WIDTH)
            else:
                start = (current_width - TARGET_WIDTH) // 2
            spec = spec[:, start : start + TARGET_WIDTH]

        elif current_width < TARGET_WIDTH:
            pad_amount = TARGET_WIDTH - current_width
            spec = torch.nn.functional.pad(spec, (0, pad_amount))

        tensor = spec.unsqueeze(0)

        # Modulares Kanal-Handling
        if self.in_channels == 3:
            tensor = tensor.repeat(3, 1, 1)

        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False
        ).squeeze(0)

        return tensor, self._get_label(row)

def get_spec_augmenter():
    """Erstellt den GPU-beschleunigten SpecAugment Filter"""
    return torch.nn.Sequential(
        T.FrequencyMasking(freq_mask_param=15),
        T.TimeMasking(time_mask_param=35)
    ).to(DEVICE)

def mixup_data(x, y, alpha=0.4):
    """Mischt zwei Spektrogramme und ihre Labels ineinander"""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    return mixed_x, y, y[index], lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Berechnet den Loss für gemischte (MixUp) Daten"""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def get_balanced_sampler(dataset):
    """Sorgt dafür, dass seltene und häufige Vogelarten gleich oft trainiert werden"""
    labels = np.array(dataset.get_all_labels())
    counts = Counter(labels)

    weights = {cls: 1.0 / np.sqrt(c) for cls, c in counts.items()}
    sample_weights = torch.tensor([weights[l] for l in labels])
    return WeightedRandomSampler(sample_weights, len(sample_weights))
