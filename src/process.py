import os
import glob
import argparse
import numpy as np
import librosa
from PIL import Image
from tqdm import tqdm

# --- Configuration ---
SAMPLE_RATE = 32000    # Standard for BirdCLEF models
CHUNK_DURATION = 5     # 5 second chunks
OVERLAP_DURATION = 2.5 # 2.5 second overlap
N_MELS = 128           # Height of the image
FMIN = 20
FMAX = 16000
N_FFT = 2048
HOP_LENGTH_SPEC = 512

def normalize_to_image(spec):
    """
    Normalizes a dB spectrogram to 0-255 uint8 for PNG saving.
    """
    eps = 1e-6
    min_val = spec.min()
    max_val = spec.max()
    
    # Normalize to 0.0 - 1.0
    norm = (spec - min_val) / (max_val - min_val + eps)
    
    # Flip Y-axis so low freq is at the bottom
    norm = np.flipud(norm)
    
    # Convert to 0-255
    img_data = (norm * 255).astype(np.uint8)
    return Image.fromarray(img_data)

def process_file(file_path, output_root, input_root):
    try:
        # 1. Load Audio
        # librosa handles mp3, wav, ogg automatically
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"\nError loading {file_path}: {e}")
        return

    # 2. Segmentation Math
    frame_length = int(CHUNK_DURATION * sr)
    hop_length_segment = int((CHUNK_DURATION - OVERLAP_DURATION) * sr)

    # 3. Padding Logic (Handle short files or uneven ends)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)), mode='constant')
    elif (len(y) - frame_length) % hop_length_segment != 0:
        n_steps = np.ceil((len(y) - frame_length) / hop_length_segment)
        target_len = int(n_steps * hop_length_segment + frame_length)
        padding = target_len - len(y)
        y = np.pad(y, (0, padding), mode='constant')

    # 4. Fast Segmentation (The Librosa Shortcut)
    chunks = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length_segment).T

    # 5. Determine Output Path
    # mirrors the 'Amsel', 'Buchfink' structure automatically
    rel_path = os.path.relpath(os.path.dirname(file_path), input_root)
    save_dir = os.path.join(output_root, rel_path)
    os.makedirs(save_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(file_path))[0]

    # 6. Loop through chunks -> Spectrogram -> Image
    for i, chunk in enumerate(chunks):
        chunk_contiguous = np.ascontiguousarray(chunk)
        
        # Mel Spectrogram
        melspec = librosa.feature.melspectrogram(
            y=chunk_contiguous, sr=sr, n_mels=N_MELS,
            fmin=FMIN, fmax=FMAX, n_fft=N_FFT, hop_length=HOP_LENGTH_SPEC
        )
        
        # Logarithmic scale (dB)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Save Image
        img = normalize_to_image(log_melspec)
        save_path = os.path.join(save_dir, f"{base_name}_{i}.png")
        img.save(save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input raw data folder")
    parser.add_argument("--output", required=True, help="Output folder for spectrograms")
    args = parser.parse_args()

    # Find all audio files (mp3, ogg, wav)
    extensions = ['*.mp3', '*.ogg', '*.wav']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(args.input, '**', ext), recursive=True))
    
    if not files:
        print(f"No audio files found in {args.input}")
        return

    print(f"Found {len(files)} audio files.")
    print(f"Processing from '{args.input}' to '{args.output}'...")

    for file_path in tqdm(files):
        process_file(file_path, args.output, args.input)

if __name__ == "__main__":
    main()
