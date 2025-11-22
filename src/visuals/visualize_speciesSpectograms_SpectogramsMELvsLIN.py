import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import glob

# --- Configuration ---
DATA_DIR = "data/raw"
SPECIES_LIST = ["Amsel", "Buchfink", "Haussperling", "Kohlmeise", "Rotkehlchen"]
SAMPLE_RATE = 32000
DURATION = 20  # Seconds to load

def get_first_file(species_name):
    """Finds the first mp3 file for a specific species."""
    search_path = os.path.join(DATA_DIR, species_name, "*.mp3")
    files = glob.glob(search_path)
    if files:
        return files[0]
    else:
        print(f"Warning: No files found for {species_name}")
        return None

def plot_five_birds():
    """
    Plot 1: Creates a 5-row figure comparing the 5 different birds.
    """
    print("Generating Plot 1: 5-Bird Comparison...")
    
    fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(20, 4), sharey=True)
    fig.suptitle('Spectrogram Comparison: 5 Different Species', fontsize=16)

    for i, species in enumerate(SPECIES_LIST):
        file_path = get_first_file(species)
        if not file_path:
            continue

        # Load Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Compute Mel Spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=16000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        # Plot
        img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', 
                                     sr=sr, fmax=16000, ax=axes[i], cmap='magma')
        axes[i].set_title(species)
        axes[i].set_xlabel('Time')
        
        # Only show Y label for the first plot to clean up look
        if i > 0:
            axes[i].set_ylabel('')

    plt.tight_layout()
    plt.savefig("src/visuals/comparison_species.png", dpi=300)
    print("Saved to src/visuals/comparison_species.png")
    plt.show()

def plot_linear_vs_mel():
    """
    Plot 2: Compares Linear Spectrogram (STFT) vs Mel Spectrogram.
    Uses 'Amsel' as the example.
    """
    print("\nGenerating Plot 2: Linear vs Mel Comparison...")
    
    file_path = get_first_file("Amsel") # Using Blackbird as example
    if not file_path:
        return

    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 8))
    
    # --- 1. Linear Frequency Spectrogram (STFT) ---
    D = librosa.stft(y)  # Short-time Fourier transform
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    
    librosa.display.specshow(S_db, y_axis='log', x_axis='time', sr=sr, ax=ax[0], cmap='magma')
    ax[0].set_title('Linear Spectrogram (Standard STFT)')
    ax[0].set_ylabel('Frequency (Hz)')
    ax[0].label_outer()

    # --- 2. Mel Spectrogram ---
    M = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=16000)
    M_db = librosa.power_to_db(M, ref=np.max)
    
    librosa.display.specshow(M_db, y_axis='mel', x_axis='time', sr=sr, fmax=16000, ax=ax[1], cmap='magma')
    ax[1].set_title('Mel Spectrogram (Biologically Inspired)')
    ax[1].set_ylabel('Frequency (Hz)')

    plt.tight_layout()
    plt.savefig("src/visuals/comparison_linear_vs_mel.png", dpi=300)
    print("Saved to src/visuals/comparison_linear_vs_mel.png")
    plt.show()

if __name__ == "__main__":
    # Ensure we are in the project root or can find the data
    if not os.path.exists(DATA_DIR):
        print(f"Error: Could not find {DATA_DIR}. Run this from the project root.")
    else:
        plot_five_birds()
        plot_linear_vs_mel()
