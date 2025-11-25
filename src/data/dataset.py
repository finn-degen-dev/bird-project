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
            mode (str): 'train' (Random Crop) or 'test' (Center Crop).
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
        # This automatically filters out files that failed preprocessing
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

    def _get_label(self, row):
        if self.task == 'species':
            return self.class_to_idx[row['species']]
        elif self.task == 'type':
            desc = str(row['type']).lower()
            label_str = 'song' if 'song' in desc else 'call'
            return self.class_to_idx[label_str]

    def __getitem__(self, idx):
        # --- ROBUST LOADING LOOP ---
        # If a file is corrupt, we try up to 3 times to pick a replacement
        # so training doesn't crash.
        attempt = 0
        while attempt < 3:
            try:
                row = self.df.iloc[idx]
                pt_filename = row['filename'].replace('.mp3', '.pt')
                pt_path = os.path.join(PROCESSED_DIR, pt_filename)
                
                # 1. Load Tensor (Super Fast)
                # Shape is typically (128, Time)
                # We convert to float32 immediately for PyTorch operations
                spec = torch.load(pt_path).float() 

                # 2. Random Crop (in Time dimension)
                # We want exactly 5 seconds.
                # 5s * 32000sr / 512 hop_length ≈ 313 frames
                target_width = 313 
                current_width = spec.shape[1]

                if current_width > target_width:
                    if self.mode == 'train':
                        start = np.random.randint(0, current_width - target_width)
                    else:
                        start = (current_width - target_width) // 2
                    spec = spec[:, start : start + target_width]
                elif current_width < target_width:
                    # Padding if too short
                    pad_amount = target_width - current_width
                    spec = torch.nn.functional.pad(spec, (0, pad_amount))

                # 3. Prepare for EfficientNet (3 Channels, Resize)
                # (1, 128, 313) -> (3, 128, 313)
                tensor = spec.unsqueeze(0).repeat(3, 1, 1) 
                
                # Resize to 224x224
                tensor = torch.nn.functional.interpolate(
                    tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
                ).squeeze(0)

                return tensor, self._get_label(row)

            except Exception as e:
                # If we are here, the file is likely corrupt
                print(f"⚠️ Warning: Corrupt file {pt_filename}. Skipping. Error: {e}")
                # Pick a random new index and try again
                idx = np.random.randint(0, len(self.df))
                attempt += 1
        
        # Fallback if 3 attempts fail (avoids crash, returns empty batch)
        # EfficientNet expects (3, 224, 224)
        print("❌ Critical: Failed to load 3 consecutive files. Returning zeros.")
        return torch.zeros((3, 224, 224)), 0
