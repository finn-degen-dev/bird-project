import argparse
import torch
import torch.nn as nn
import numpy as np
import librosa
from torchvision import models
import torch.nn.functional as F
import os

# --- Config ---
MODEL_PATH = "data/models/bird_classifier_pro.pth"
CLASSES = ['Amsel', 'Buchfink', 'Haussperling', 'Kohlmeise', 'Rotkehlchen']
SAMPLE_RATE = 32000
CHUNK_DURATION = 5
N_MELS = 128
FMAX = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_audio(audio_path):
    """
    Loads audio and converts to spectrogram tensor matching BirdRawDataset logic.
    Returns a batch of tensors (one for each 5s chunk).
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Pad to ensure at least one chunk exists
    target_len = int(SAMPLE_RATE * CHUNK_DURATION)
    if len(y) < target_len:
        y = np.pad(y, (0, target_len - len(y)), mode='constant')
        
    # Split into 5s chunks (no overlap for simple prediction)
    chunks = librosa.util.frame(y, frame_length=target_len, hop_length=target_len).T
    
    tensors = []
    
    for chunk in chunks:
        chunk = np.ascontiguousarray(chunk)
        
        # Mel Spectrogram
        melspec = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_mels=N_MELS, fmax=FMAX
        )
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Normalize 0.0 to 1.0
        min_val = log_melspec.min()
        max_val = log_melspec.max()
        spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
        
        # Flip and Copy (Important!)
        spec_norm = np.flipud(spec_norm).copy()
        
        # Convert to Tensor (1, 128, time)
        tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0)
        
        # Repeat channels (3, 128, time)
        tensor = tensor.repeat(3, 1, 1)
        
        # Resize to 224x224
        tensor = torch.nn.functional.interpolate(
            tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze(0)
        
        tensors.append(tensor)
        
    return torch.stack(tensors)

def predict_file(audio_path):
    print(f"Analyzing {os.path.basename(audio_path)}...")
    
    # Setup Model
    try:
        model = models.efficientnet_b0(weights=None)
    except:
        print("Error loading architecture.")
        return

    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"Model not found at {MODEL_PATH}")
        return

    model.to(device)
    model.eval()
    
    # Prepare Data
    try:
        batch = preprocess_audio(audio_path).to(device)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # Inference
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        
    # Average probability across all chunks (if file > 5s)
    avg_probs = torch.mean(probs, dim=0).cpu().numpy()
    best_idx = np.argmax(avg_probs)
    
    print(f"\nPrediction: {CLASSES[best_idx].upper()}")
    print(f"Confidence: {avg_probs[best_idx]:.2%}")
    print("-" * 20)
    for i, cls in enumerate(CLASSES):
        print(f"{cls}: {avg_probs[i]:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to audio file")
    args = parser.parse_args()
    predict_file(args.file)
