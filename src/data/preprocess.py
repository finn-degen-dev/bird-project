import os
import sys
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings

# Suppress the annoying warnings during preprocessing
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Folder to save fast tensors
PROCESSED_DIR = os.path.join(config.DATA_DIR, "processed_tensors")

def process_audio():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    df = pd.read_csv(config.METADATA_FILE)
    print(f"🚀 Pre-processing {len(df)} files to Tensors...")
    print(f"   Source: {config.RAW_AUDIO_DIR}")
    print(f"   Dest:   {PROCESSED_DIR}")

    success = 0
    errors = 0

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        mp3_path = os.path.join(config.RAW_AUDIO_DIR, row['filename'])
        pt_filename = row['filename'].replace('.mp3', '.pt')
        pt_path = os.path.join(PROCESSED_DIR, pt_filename)

        # Skip if already done
        if os.path.exists(pt_path):
            success += 1
            continue

        try:
            # 1. Load & Resample (The slow part)
            # duration=15 limits huge files
            y, sr = librosa.load(mp3_path, sr=config.SAMPLE_RATE, duration=15)
            
            # 2. Padding (Ensure it's at least 5 seconds)
            target_len = config.SAMPLE_RATE * config.DURATION
            if len(y) < target_len:
                y = np.pad(y, (0, target_len - len(y)), mode='constant')
            
            # Note: We do NOT crop here. We save the full 15s so we can 
            # random-crop during training for variety.

            # 3. Mel Spectrogram (Math)
            melspec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=128, fmax=16000
            )
            log_melspec = librosa.power_to_db(melspec, ref=np.max)

            # 4. Normalize (0 to 1)
            min_val, max_val = log_melspec.min(), log_melspec.max()
            spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
            spec_norm = np.flipud(spec_norm).copy()

            # 5. Save as Float16 Tensor (Saves 50% disk space, equal accuracy)
            tensor = torch.tensor(spec_norm, dtype=torch.float16)
            torch.save(tensor, pt_path)
            success += 1

        except Exception as e:
            errors += 1
            # Optional: Print error only if it's not a known "bad header" issue
            pass

    print(f"\n✅ Done! {success} tensors created. {errors} corrupt files failed.")

if __name__ == "__main__":
    process_audio()
