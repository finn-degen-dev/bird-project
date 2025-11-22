import os
import glob
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain

# --- Configuration (Must match your Process.py logic) ---
SAMPLE_RATE = 32000
CHUNK_DURATION = 5  # Seconds
N_MELS = 128
FMAX = 16000

class BirdRawDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        Args:
            root_dir (str): Path to 'data/dataset/train' or 'data/dataset/test'
            mode (str): 'train' or 'test'. 
                        'train' enables Audio Augmentation and random cropping.
                        'test' disables augmentation and takes the center crop.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Collect all files
        self.files = []
        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            cls_files = glob.glob(os.path.join(cls_folder, "*.mp3"))
            cls_files += glob.glob(os.path.join(cls_folder, "*.wav"))
            for f in cls_files:
                self.files.append((f, self.class_to_idx[cls_name]))

        # --- Audio Augmentation Pipeline ---
        # Only applied if mode == 'train'
        self.augment = Compose([
            # Add static noise (wind/mic hiss)
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
            # Make the bird sound slightly different (age/size variation)
            PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
            # Make the bird sing faster or slower
            TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
            # Simulates distance
            Gain(min_gain_db=-12, max_gain_db=6, p=0.5),
            # Shift the audio in time (so the chirp isn't always in the center)
            Shift(min_shift=-0.5, max_shift=0.5, shift_unit="fraction", p=0.5),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path, label = self.files[idx]
        
        # 1. Load Audio
        # We load up to chunk_duration * 2 to give us room to shift/crop
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=15)
        except Exception as e:
            # Fallback for corrupted files: return silence
            print(f"Error loading {file_path}: {e}")
            y = np.zeros(int(SAMPLE_RATE * CHUNK_DURATION))
            sr = SAMPLE_RATE

        # 2. Pad if too short
        target_len = int(SAMPLE_RATE * CHUNK_DURATION)
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')

        # 3. Crop / Select Segment
        if len(y) > target_len:
            if self.mode == 'train':
                # Random Crop
                start = np.random.randint(0, len(y) - target_len)
            else:
                # Center Crop for validation
                start = (len(y) - target_len) // 2
            y = y[start : start + target_len]

        # 4. Apply Augmentation (Raw Waveform)
        if self.mode == 'train':
            try:
                y = self.augment(samples=y, sample_rate=sr)
            except Exception as e:
                pass # Skip augmentation if it fails (rare size mismatch)

        # 5. Convert to Mel Spectrogram (Matches process.py math)
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, fmax=FMAX
        )
        log_melspec = librosa.power_to_db(melspec, ref=np.max)

        # 6. Normalize (Min-Max to 0.0 - 1.0)
        min_val = log_melspec.min()
        max_val = log_melspec.max()
        spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
        
        # Flip (Low freq at bottom) AND COPY to fix negative stride error
        spec_norm = np.flipud(spec_norm).copy()

        # 7. Convert to Tensor
        # Shape becomes (1, 128, time_steps)
        # We repeat to 3 channels because EfficientNet expects RGB
        tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
        tensor = tensor.repeat(3, 1, 1)

        # Resize to 224x224 (Model Input Size)
        # We use torch.nn.functional.interpolate for this
        tensor = torch.nn.functional.interpolate(tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
        tensor = tensor.squeeze(0)

        return tensor, label
