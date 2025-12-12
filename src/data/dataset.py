import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import sys

# Ensure we can find config.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Path to tensors
PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed_tensors")

class SmartBirdDataset(Dataset):
    def __init__(self, mode='train', filter_sex=None, filter_type=None, task='species', min_samples=10):
        """
        Args:
            mode (str): 'train' (Random Crop), 'test' (Center Crop), or 'full' (No Crop, return all).
            filter_sex (str): 'male', 'female', or None.
            filter_type (str): 'song', 'call', or None.
            task (str): 'species' (Label = Bird Name) or 'type' (Label = Song/Call).
            min_samples (int): Drop classes with fewer than N samples.
        """
        self.mode = mode
        self.task = task
        
        # 1. Load Metadata
        if not os.path.exists(config.METADATA_FILE):
            raise FileNotFoundError(f"Metadata not found at {config.METADATA_FILE}. Run download logic first.")
            
        df = pd.read_csv(config.METADATA_FILE)
        
        # 2. BLACKLIST: Remove generic/unknown labels
        blacklist = ["Identity unknown", "Soundscape", "Mystery bird", "?"]
        df = df[~df['species'].isin(blacklist)]
        
        # 3. Apply Filters
        if filter_sex:
            df = df[df['sex'].astype(str).str.lower() == filter_sex.lower()]
        
        if filter_type:
            df = df[df['type'].astype(str).str.lower().str.contains(filter_type.lower())]

        # 4. Filter Rare Classes (Crucial for stability)
        if task == 'species':
            counts = df['species'].value_counts()
            valid_species = counts[counts >= min_samples].index
            df = df[df['species'].isin(valid_species)]

        # 5. IMPORTANT: Only keep rows where the .pt file actually exists
        df['pt_exists'] = df['filename'].apply(
            lambda x: os.path.exists(os.path.join(PROCESSED_DIR, x.replace('.mp3', '.pt')))
        )
        self.df = df[df['pt_exists']].reset_index(drop=True)
        
        if len(self.df) == 0:
            raise ValueError(f"No valid data found! Filters: Sex={filter_sex}, Type={filter_type}")

        # 6. Define Classes/Labels
        if task == 'species':
            self.classes = sorted(self.df['species'].unique().tolist())
        elif task == 'type':
            self.classes = ['call', 'song']
            
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        print(f"[{mode.upper()}] Dataset ready. {len(self.df)} tensors. {len(self.classes)} classes.")

    def __len__(self):
        return len(self.df)
    
    def get_all_labels(self):
        """Returns a list of all labels for creating WeightedRandomSampler"""
        labels = []
        for _, row in self.df.iterrows():
            labels.append(self._get_label(row))
        return labels

    def _get_label(self, row):
        if self.task == 'species':
            return self.class_to_idx[row['species']]
        elif self.task == 'type':
            desc = str(row['type']).lower()
            label_str = 'song' if 'song' in desc else 'call'
            return self.class_to_idx[label_str]

    def __getitem__(self, idx):
        attempt = 0
        while attempt < 3:
            try:
                row = self.df.iloc[idx]
                pt_filename = row['filename'].replace('.mp3', '.pt')
                pt_path = os.path.join(PROCESSED_DIR, pt_filename)
                
                # 1. Load Tensor
                spec = torch.load(pt_path).float() 

                # 2. Crop Logic
                if self.mode == 'full':
                    # Return the WHOLE spectrogram (variable length)
                    # Note: Cannot use batch_size > 1 with this mode without custom collate
                    tensor = spec.unsqueeze(0).repeat(3, 1, 1) # (3, 128, Time)
                    return tensor, self._get_label(row)

                # Standard Fixed Width Logic
                target_width = 313 # approx 5 seconds
                current_width = spec.shape[1]

                if current_width > target_width:
                    if self.mode == 'train':
                        start = np.random.randint(0, current_width - target_width)
                    else:
                        # Center Crop
                        start = (current_width - target_width) // 2
                    spec = spec[:, start : start + target_width]
                elif current_width < target_width:
                    pad_amount = target_width - current_width
                    spec = torch.nn.functional.pad(spec, (0, pad_amount))

                # --- AUGMENTATION (Train Only) ---
                if self.mode == 'train':
                    # 1. Frequency Masking
                    # Mask 1-2 bands of up to 30 bins
                    num_masks = np.random.randint(1, 3)
                    for _ in range(num_masks):
                        f_width = np.random.randint(1, 30)
                        f_start = np.random.randint(0, spec.shape[0] - f_width)
                        spec[f_start : f_start + f_width, :] = 0

                    # 2. Time Masking
                    # Mask 1-2 blocks of up to 40 frames
                    num_masks = np.random.randint(1, 3)
                    for _ in range(num_masks):
                        t_width = np.random.randint(1, 40)
                        t_start = np.random.randint(0, spec.shape[1] - t_width)
                        spec[:, t_start : t_start + t_width] = 0
                    
                    # 3. Gaussian Noise
                    if np.random.rand() < 0.5:
                        noise = torch.randn_like(spec) * 0.05 * spec.mean()
                        spec = spec + noise

                # 3. Prepare for EfficientNet
                tensor = spec.unsqueeze(0).repeat(3, 1, 1) 
                
                # Resize to 224x224 (Standard ImageNet size)
                tensor = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                ).squeeze(0)

                return tensor, self._get_label(row)

            except Exception as e:
                print(f"⚠️ Warning: Corrupt file {pt_filename}. Skipping. Error: {e}")
                idx = np.random.randint(0, len(self.df))
                attempt += 1
        
        return torch.zeros((3, 224, 224)), 0
