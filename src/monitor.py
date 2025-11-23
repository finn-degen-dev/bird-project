import argparse
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import models
import torch.nn.functional as F
import os
import pandas as pd

# --- Config ---
MODEL_PATH = "data/models/bird_classifier_pro.pth"
CLASSES = ['Amsel', 'Buchfink', 'Haussperling', 'Kohlmeise', 'Rotkehlchen']
SAMPLE_RATE = 32000
CHUNK_DURATION = 5  # Seconds
STEP_SIZE = 2.5     # 50% Overlap for better detection
N_MELS = 128
FMAX = 16000
CONFIDENCE_THRESHOLD = 0.60  # Only count if model is 60% sure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    print("Loading Model...")
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

def preprocess_chunk(y, sr):
    """Converts a single 5s audio array into a tensor"""
    # 1. Mel Spec
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    
    # 2. Normalize
    min_val, max_val = log_melspec.min(), log_melspec.max()
    spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
    
    # 3. Flip & Copy
    spec_norm = np.flipud(spec_norm).copy()
    
    # 4. Tensor
    tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
    tensor = tensor.repeat(3, 1, 1) # RGB
    
    # 5. Resize
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    return tensor

def analyze_long_audio(audio_path, model):
    print(f"Loading audio: {audio_path} ... (this might take a moment)")
    # Load full audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_duration = len(y) / sr
    print(f"Audio Duration: {total_duration:.1f} seconds")

    # Sliding Window
    chunk_samples = int(CHUNK_DURATION * sr)
    step_samples = int(STEP_SIZE * sr)
    
    results = []
    
    print("Scanning...")
    # Use a sliding window
    for start_idx in range(0, len(y) - chunk_samples, step_samples):
        end_idx = start_idx + chunk_samples
        chunk = y[start_idx:end_idx]
        
        # Preprocess
        tensor = preprocess_chunk(chunk, sr).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probs = F.softmax(outputs, dim=1)
        
        # Get top prediction
        conf, idx = torch.max(probs, 1)
        conf = conf.item()
        idx = idx.item()
        
        time_sec = start_idx / sr
        
        # Store result if confident enough
        if conf >= CONFIDENCE_THRESHOLD:
            results.append({
                "time": time_sec,
                "species": CLASSES[idx],
                "confidence": conf
            })
            
    return pd.DataFrame(results)

def plot_timeline(df, output_path="data/models/monitoring_timeline.png"):
    if df.empty:
        print("No birds detected above threshold.")
        return

    plt.figure(figsize=(15, 6))
    
    # Scatter plot: X=Time, Y=Species, Color=Confidence
    sns.scatterplot(data=df, x="time", y="species", hue="confidence", 
                    palette="viridis", s=100, alpha=0.7)
    
    plt.title("Bird Activity Timeline")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Species")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Timeline saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to long audio file")
    args = parser.parse_args()
    
    model = load_model()
    df = analyze_long_audio(args.file, model)
    
    if not df.empty:
        print("\n--- Detections ---")
        # Group by species to count occurrences (rough estimate)
        print(df['species'].value_counts())
        
        # Save detailed CSV
        csv_path = "data/models/monitoring_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to {csv_path}")
        
        plot_timeline(df)
    else:
        print("No birds detected.")

if __name__ == "__main__":
    main()
