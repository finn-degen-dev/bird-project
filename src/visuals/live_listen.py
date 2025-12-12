import sys
import os
import time
import json
import torch
import numpy as np
import librosa
import sounddevice as sd
import torch.nn.functional as F
from torchvision import models

# Add parent directory to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- SETTINGS ---
MODEL_PATH = "data/models/german_bird_classifier_pro.pth"
JSON_PATH = "data/models/german_bird_classifier_pro_classes.json"
DEVICE = torch.device("cpu") # Use CPU for inference to avoid CUDA overhead/lag on laptops
UPDATE_INTERVAL = 0.5        # How often to predict (seconds)
CONFIDENCE_THRESHOLD = 0.20  # Only show birds above this %

# Global Buffer
# We need to hold exactly DURATION seconds of audio
BUFFER_SIZE = config.SAMPLE_RATE * config.DURATION
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

def load_system():
    """Loads the trained model and class names"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(JSON_PATH):
        print(f" Model files not found at {MODEL_PATH}")
        sys.exit(1)

    print("⏳ Loading Model...")
    with open(JSON_PATH, 'r') as f:
        classes = json.load(f)['classes']

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    
    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    print(" Model Loaded!")
    return model, classes

def audio_callback(indata, frames, time, status):
    """
    Called by sounddevice whenever there is new audio.
    We shift the buffer and append new data.
    """
    global audio_buffer
    if status:
        print(status, file=sys.stderr)
    
    # Flatten input (indata is [frames, channels])
    new_data = indata[:, 0] 
    
    # Shift buffer left
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    
    # Insert new data at the end
    audio_buffer[-len(new_data):] = new_data

def preprocess_live_audio(audio_chunk):
    """
    Mirrors the logic in src/data/preprocess.py exactly.
    """
    # 1. Mel Spectrogram
    melspec = librosa.feature.melspectrogram(
        y=audio_chunk, 
        sr=config.SAMPLE_RATE, 
        n_mels=128, 
        fmax=16000
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    # 2. Normalize (0 to 1) - Dynamic per chunk like training
    min_val, max_val = log_melspec.min(), log_melspec.max()
    if max_val - min_val == 0:
        return None # Silence/Error
        
    spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
    spec_norm = np.flipud(spec_norm).copy()

    # 3. Convert to Tensor
    tensor = torch.tensor(spec_norm, dtype=torch.float32)

    # 4. Resize to 224x224 (EfficientNet requirement)
    # Shape: (1, 128, Time) -> (1, 1, 128, Time) for interpolate
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    
    # 5. Expand to 3 Channels (ImageNet expectation)
    # Shape: (1, 3, 224, 224)
    tensor = tensor.squeeze(0).repeat(3, 1, 1).unsqueeze(0)
    
    return tensor.to(DEVICE)

def draw_ui(top_predictions):
    """Clears screen and draws a cool UI"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*50)
    print(f"  LIVE BIRD LISTENER (Sample Rate: {config.SAMPLE_RATE}Hz)")
    print("="*50)
    print("Listening... (Press Ctrl+C to stop)\n")

    found_anything = False
    
    for name, score in top_predictions:
        if score < CONFIDENCE_THRESHOLD:
            continue
            
        found_anything = True
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        
        # Color coding (Green for high confidence)
        if score > 0.7:
            color = "\033[92m" # Green
        elif score > 0.4:
            color = "\033[93m" # Yellow
        else:
            color = "\033[0m"  # White
            
        print(f"{color}{name:<30} {bar} {score:.1%}\033[0m")

    if not found_anything:
        print("\033[90m(No clear bird song detected...)\033[0m")
        
    print("\n" + "-"*50)

def main():
    model, classes = load_system()
    
    # Start InputStream
    # We request the specific sample rate to match the model
    try:
        stream = sd.InputStream(
            channels=1,
            samplerate=config.SAMPLE_RATE,
            callback=audio_callback,
            blocksize=int(config.SAMPLE_RATE * UPDATE_INTERVAL) 
        )
    except Exception as e:
        print(f"\n Error opening microphone: {e}")
        print("Try installing PortAudio: 'brew install portaudio' (Mac) or 'sudo apt install libportaudio2' (Linux)")
        return

    print("\n Microphone initialized. Starting stream...")
    
    with stream:
        while True:
            try:
                time.sleep(UPDATE_INTERVAL)
                
                # Make a copy of the buffer to avoid threading issues
                current_chunk = audio_buffer.copy()
                
                # Preprocess
                input_tensor = preprocess_live_audio(current_chunk)
                
                if input_tensor is None:
                    continue

                # Inference
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1)
                
                # Get Top 5
                top_probs, top_indices = torch.topk(probs, 5)
                
                results = []
                for i in range(5):
                    idx = top_indices[0][i].item()
                    score = top_probs[0][i].item()
                    results.append((classes[idx], score))
                
                draw_ui(results)

            except KeyboardInterrupt:
                print("\n Stopping...")
                break

if __name__ == "__main__":
    main()
