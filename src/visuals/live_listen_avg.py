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
from collections import deque

# Add parent directory to path to find config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# --- SETTINGS ---
MODEL_PATH = "data/models/german_bird_classifier_pro.pth"
JSON_PATH = "data/models/german_bird_classifier_pro_classes.json"
DEVICE = torch.device("cpu") # CPU is stable for real-time inference
UPDATE_INTERVAL = 0.5        # How often to run inference (seconds)
CONFIDENCE_THRESHOLD = 0.50  # Show birds with >15% probability
HISTORY_LEN = 5              # Average the last 5 predictions (2.5s smoothing)

# --- GLOBAL BUFFERS ---
# 1. AUDIO BUFFER: Holds exactly 5 seconds of raw samples
BUFFER_SIZE = config.SAMPLE_RATE * config.DURATION
audio_buffer = np.zeros(BUFFER_SIZE, dtype=np.float32)

def load_system():
    """Loads the trained model and class names"""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(JSON_PATH):
        print(f"❌ Model files not found at {MODEL_PATH}")
        print("   Did you run the training script inside Docker first?")
        sys.exit(1)

    print("⏳ Loading Model...")
    with open(JSON_PATH, 'r') as f:
        classes = json.load(f)['classes']

    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(classes))
    
    # Load weights
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f" Error loading weights: {e}")
        sys.exit(1)
        
    model.to(DEVICE)
    model.eval()
    
    print(" Model Loaded!")
    return model, classes

def audio_callback(indata, frames, time, status):
    """
    Called by sounddevice whenever there is new audio.
    We shift the buffer and append new data (Ring Buffer).
    """
    global audio_buffer
    if status:
        print(status, file=sys.stderr)
    
    # Flatten input (indata is [frames, channels])
    new_data = indata[:, 0] 
    
    # Shift buffer left by the size of new data
    audio_buffer = np.roll(audio_buffer, -len(new_data))
    
    # Insert new data at the end
    audio_buffer[-len(new_data):] = new_data

def preprocess_live_audio(audio_chunk):
    """
    Converts raw audio chunk to a normalized Mel Spectrogram Tensor.
    Mirrors src/data/preprocess.py
    """
    # 1. Mel Spectrogram
    melspec = librosa.feature.melspectrogram(
        y=audio_chunk, 
        sr=config.SAMPLE_RATE, 
        n_mels=128, 
        fmax=16000
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    # 2. Normalize (0 to 1) - Dynamic per chunk
    min_val, max_val = log_melspec.min(), log_melspec.max()
    
    # Avoid division by zero on silence
    if max_val - min_val == 0:
        return None 
        
    spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
    spec_norm = np.flipud(spec_norm).copy()

    # 3. Convert to Tensor
    tensor = torch.tensor(spec_norm, dtype=torch.float32)

    # 4. Resize to 224x224 (EfficientNet requirement)
    # Shape: (128, Time) -> (1, 1, 128, Time)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(224, 224), mode='bilinear', align_corners=False)
    
    # 5. Expand to 3 Channels (ImageNet expectation) -> (1, 3, 224, 224)
    tensor = tensor.squeeze(0).repeat(3, 1, 1).unsqueeze(0)
    
    return tensor.to(DEVICE)

def draw_ui(top_predictions):
    """Clears screen and draws the bars"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("="*60)
    print(f"  LIVE BIRD LISTENER")
    print(f"   Sample Rate: {config.SAMPLE_RATE}Hz | Smoothing: {HISTORY_LEN} frames")
    print("="*60)
    print("Listening... (Press Ctrl+C to stop)\n")

    found_anything = False
    
    for name, score in top_predictions:
        if score < CONFIDENCE_THRESHOLD:
            continue
            
        found_anything = True
        
        # Visual Bar Logic
        bar_len = int(score * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        
        # Color coding
        if score > 0.7:
            color = "\033[92m" # Green
        elif score > 0.4:
            color = "\033[93m" # Yellow
        else:
            color = "\033[0m"  # White
            
        print(f"{color}{name:<30} {bar} {score:.1%}\033[0m")

    if not found_anything:
        print("\033[90m(No distinct bird song detected...)\033[0m")
        
    print("\n" + "-"*60)

def main():
    model, classes = load_system()
    
    # 2. PREDICTION BUFFER: Stores the last N probability vectors
    prediction_history = deque(maxlen=HISTORY_LEN)

    # Initialize Stream
    try:
        stream = sd.InputStream(
            channels=1,
            samplerate=config.SAMPLE_RATE,
            callback=audio_callback,
            blocksize=int(config.SAMPLE_RATE * UPDATE_INTERVAL) 
        )
    except Exception as e:
        print(f"\n Error opening microphone: {e}")
        return

    print("\n Microphone initialized. Starting stream...")
    
    with stream:
        while True:
            try:
                time.sleep(UPDATE_INTERVAL)
                
                # Make a copy of the ring buffer to work on
                current_chunk = audio_buffer.copy()
                
                # Preprocess
                input_tensor = preprocess_live_audio(current_chunk)
                
                if input_tensor is None:
                    continue

                # Inference
                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = F.softmax(logits, dim=1) # Shape: [1, NumClasses]
                
                # --- AVERAGING LOGIC ---
                prediction_history.append(probs)
                
                # Calculate mean across the history stack
                if len(prediction_history) > 0:
                    avg_probs = torch.stack(list(prediction_history)).mean(dim=0)
                else:
                    avg_probs = probs
                
                # Get Top 5 from the AVERAGED probabilities
                top_probs, top_indices = torch.topk(avg_probs, 5)
                
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
