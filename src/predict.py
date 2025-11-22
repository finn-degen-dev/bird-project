import argparse
import os
import torch
import numpy as np
import librosa
from torchvision import transforms, models
from PIL import Image
import torch.nn.functional as F

# --- Config matches Training ---
MODEL_PATH = "data/models/bird_classifier_resnet18.pth"
# ALPHABETICAL ORDER IS CRITICAL (Must match ImageFolder order)
CLASSES = ['Amsel', 'Buchfink', 'Haussperling', 'Kohlmeise', 'Rotkehlchen']
SAMPLE_RATE = 32000
CHUNK_DURATION = 5
N_MELS = 128
FMAX = 16000

# Detect Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_transforms():
    """Must match the 'test' transforms from train.py exactly."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def audio_to_images(audio_path):
    """
    Loads audio, splits into chunks, converts to Spectrogram PIL Images.
    (This replicates logic from process.py but in-memory)
    """
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    
    # Calculate dimensions
    frame_length = int(CHUNK_DURATION * sr)
    hop_length = int(frame_length) # Non-overlapping for inference speed
    
    # Pad if too short
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)), mode='constant')

    # Split into chunks
    chunks = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length).T
    
    images = []
    for chunk in chunks:
        chunk = np.ascontiguousarray(chunk)
        
        # Generate Mel Spec
        melspec = librosa.feature.melspectrogram(
            y=chunk, sr=sr, n_mels=N_MELS, fmax=FMAX
        )
        log_melspec = librosa.power_to_db(melspec, ref=np.max)
        
        # Normalize to 0-255 (Same math as process.py)
        min_val = log_melspec.min()
        max_val = log_melspec.max()
        norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
        norm = np.flipud(norm) # Flip Y
        img_data = (norm * 255).astype(np.uint8)
        
        # Convert to PIL
        img = Image.fromarray(img_data)
        # Convert to RGB (ResNet expects 3 channels)
        img = img.convert("RGB")
        images.append(img)
        
    return images

def predict_file(audio_path):
    print(f"Processing: {os.path.basename(audio_path)}...")
    
    # 1. Prepare Model
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASSES))
    
    # Load weights
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    
    # 2. Prepare Data
    try:
        pil_images = audio_to_images(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    transform = get_transforms()
    # Stack images into a batch tensor
    batch = torch.stack([transform(img) for img in pil_images]).to(device)
    
    # 3. Inference
    with torch.no_grad():
        outputs = model(batch)
        # Convert logits to probabilities (0.0 to 1.0)
        probs = F.softmax(outputs, dim=1)
        
    # 4. Aggregation (Average probability across all chunks)
    avg_probs = torch.mean(probs, dim=0).cpu().numpy()
    
    # 5. Result
    best_idx = np.argmax(avg_probs)
    confidence = avg_probs[best_idx]
    predicted_class = CLASSES[best_idx]
    
    print("\n--- Results ---")
    print(f"Prediction: {predicted_class.upper()}")
    print(f"Confidence: {confidence:.2%}")
    print("-" * 20)
    print("Details:")
    for i, cls in enumerate(CLASSES):
        print(f"{cls}: {avg_probs[i]:.2%}")
    print("-" * 20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to audio file")
    args = parser.parse_args()
    
    predict_file(args.file)
