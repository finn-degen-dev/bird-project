import os
import random
import numpy as np
import soundfile as sf
import librosa
from glob import glob

# --- Config ---
DATA_DIR = "data/dataset/test"
OUTPUT_FILE = "data/mock_forest_recording.wav"
DURATION_TARGET = 120  # 2 Minutes of audio
SAMPLE_RATE = 32000

def main():
    print("Generating mock forest recording...")
    
    # Find all files
    all_files = glob(os.path.join(DATA_DIR, "*", "*.mp3"))
    if not all_files:
        print("No files found!")
        return

    full_audio = []
    current_len = 0
    target_samples = DURATION_TARGET * SAMPLE_RATE
    
    while current_len < target_samples:
        # Pick random file
        f = random.choice(all_files)
        y, sr = librosa.load(f, sr=SAMPLE_RATE)
        
        # Occasionally insert silence (to simulate gaps)
        if random.random() > 0.7:
            silence_dur = random.randint(1, 5) # 1-5 seconds silence
            full_audio.extend(np.zeros(silence_dur * SAMPLE_RATE))
        
        full_audio.extend(y)
        current_len += len(y) + (5 * SAMPLE_RATE) # Rough estimate including silence
        print(f"Added {os.path.basename(f)}")

    # Save
    full_audio = np.array(full_audio)
    sf.write(OUTPUT_FILE, full_audio, SAMPLE_RATE)
    print(f"\nCreated {OUTPUT_FILE} ({len(full_audio)/SAMPLE_RATE:.1f} seconds)")

if __name__ == "__main__":
    main()
