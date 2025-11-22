import argparse
import torch
import torch.nn as nn
import numpy as np
import librosa
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os

# --- Configuration ---
MODEL_PATH = "data/models/bird_classifier_efficientnet_advanced.pth"
CLASSES = ['Amsel', 'Buchfink', 'Haussperling', 'Kohlmeise', 'Rotkehlchen'] # Alphabetical!
SAMPLE_RATE = 32000
CHUNK_DURATION = 5
N_MELS = 128
FMAX = 16000
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def audio_to_images(audio_path):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    frame_length = int(CHUNK_DURATION * sr)
    hop_length = int(frame_length) 
    
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)), mode='constant')

    chunks = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    
    images = []
    for chunk in chunks:
        chunk = np.ascontiguousarray(chunk)
        melspec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=N_MELS, fmax=FMAX)
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        min_val, max_val = log_melspec.min(), log_melspec.max()
        norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
        img_data = (np.flipud(norm) * 255).astype(np.uint8)
        
        img = Image.fromarray(img_data).convert("RGB")
        images.append(img)
        
    return images

def predict_file(audio_path):
    print(f"Analyzing {os.path.basename(audio_path)}...")
    
    # Load EfficientNet Architecture
    try:
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    except:
        model = models.efficientnet_b0(weights=None)
        
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    try:
        pil_images = audio_to_images(audio_path)
    except Exception as e:
        print(f"Error: {e}")
        return

    transform = get_transforms()
    batch = torch.stack([transform(img) for img in pil_images]).to(device)
    
    with torch.no_grad():
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        
    avg_probs = torch.mean(probs, dim=0).cpu().numpy()
    best_idx = np.argmax(avg_probs)
    
    print(f"\nPrediction: {CLASSES[best_idx].upper()}")
    print(f"Confidence: {avg_probs[best_idx]:.2%}")
    print("-" * 20)
    for i, cls in enumerate(CLASSES):
        print(f"{cls}: {avg_probs[i]:.2%}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True)
    args = parser.parse_args()
    predict_file(args.file)
