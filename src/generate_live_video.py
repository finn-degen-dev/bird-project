import sys
import os
import subprocess
import argparse
import numpy as np
import librosa
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

# --- Config ---
MODEL_PATH = "data/models/bird_classifier_pro.pth"
CLASSES = ['Amsel', 'Buchfink', 'Haussperling', 'Kohlmeise', 'Rotkehlchen']
SAMPLE_RATE = 32000
CHUNK_DURATION = 5
N_MELS = 128
FMAX = 16000
VIDEO_FPS = 30
HEIGHT = 720
WIDTH = 1280
SCROLL_SPEED = 200 

# --- GATES ---
# 1. RMS (Volume) Threshold: Needs to be reasonably loud
RMS_THRESHOLD = 0.005 

# 2. Dynamic Range Threshold (dB): 
# If the difference between min and max dB in the spectrogram is less than this,
# it means the image is "flat" (just noise). Don't normalize it!
DB_DELTA_THRESHOLD = 20.0 

# 3. AI Confidence
CONFIDENCE_THRESHOLD = 0.50

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
        print(f"Error: {e}")
        sys.exit(1)

def get_prediction(model, audio_chunk, sr):
    """Returns probability array for the chunk, with Strict Silence filtering."""
    
    # --- GATE 1: RAW AUDIO VOLUME ---
    rms = np.sqrt(np.mean(audio_chunk**2))
    if rms < RMS_THRESHOLD:
        return np.zeros(len(CLASSES))

    # Pad if too short
    if len(audio_chunk) < int(CHUNK_DURATION * sr):
        pad = int(CHUNK_DURATION * sr) - len(audio_chunk)
        audio_chunk = np.pad(audio_chunk, (0, pad), mode='constant')
    
    # Generate Spectrogram
    melspec = librosa.feature.melspectrogram(y=audio_chunk, sr=sr, n_mels=N_MELS, fmax=FMAX)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    
    # --- GATE 2: SPECTROGRAM CONTRAST (The Fix) ---
    min_val, max_val = log_melspec.min(), log_melspec.max()
    delta = max_val - min_val
    
    # If the "loudest" pixel and "quietest" pixel are close (e.g., < 20dB difference),
    # it means the audio is just flat hiss/static. 
    # Normalizing this would create hallucinations.
    if delta < DB_DELTA_THRESHOLD:
        return np.zeros(len(CLASSES))

    # Safe Normalization
    spec_norm = (log_melspec - min_val) / (delta + 1e-6)
    spec_norm = np.flipud(spec_norm).copy()
    
    tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
    tensor = torch.nn.functional.interpolate(
        tensor.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    with torch.no_grad():
        outputs = model(tensor.unsqueeze(0).to(device))
        probs = F.softmax(outputs, dim=1)
    
    probs_np = probs.cpu().numpy()[0]

    # --- GATE 3: AI CONFIDENCE ---
    if np.max(probs_np) < CONFIDENCE_THRESHOLD:
        return np.zeros(len(CLASSES))

    return probs_np

def create_spectrogram_image(y, sr):
    """Generates a long panoramic spectrogram image"""
    print("Generating full spectrogram texture...")
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    S_dB = librosa.power_to_db(S, ref=np.max)
    
    S_min, S_max = S_dB.min(), S_dB.max()
    S_norm = 255 * (S_dB - S_min) / (S_max - S_min)
    S_norm = np.flipud(S_norm).astype(np.uint8)
    
    colormap = plt.get_cmap('magma')
    S_color = (colormap(S_norm / 255.0) * 255).astype(np.uint8)[:, :, :3]
    S_color = cv2.cvtColor(S_color, cv2.COLOR_RGB2BGR)
    
    target_width = int(len(y) / sr * SCROLL_SPEED)
    target_height = int(HEIGHT * 0.6)
    
    img = cv2.resize(S_color, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", required=True, help="Path to audio file")
    parser.add_argument("--output", default="data/visuals/live_detection.mp4")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    print(f"Loading audio: {args.file}")
    y, sr = librosa.load(args.file, sr=SAMPLE_RATE)
    duration = len(y) / sr
    
    model = load_model()
    
    print("Pre-calculating AI predictions...")
    predictions = []
    step = 0.2
    for t in tqdm(np.arange(0, duration, step)):
        start_sample = int(t * sr)
        end_sample = start_sample + int(CHUNK_DURATION * sr)
        if end_sample > len(y): break
        
        chunk = y[start_sample:end_sample]
        probs = get_prediction(model, chunk, sr)
        predictions.append(probs)
    
    predictions = np.array(predictions)

    spec_img = create_spectrogram_image(y, sr)
    
    temp_video = "temp_video_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video, fourcc, VIDEO_FPS, (WIDTH, HEIGHT))
    
    print(f"Rendering video ({int(duration * VIDEO_FPS)} frames)...")
    
    for frame_idx in tqdm(range(int(duration * VIDEO_FPS))):
        current_time = frame_idx / VIDEO_FPS
        
        # Background
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30) 
        
        # Spectrogram
        spec_x = int(current_time * SCROLL_SPEED)
        screen_center = WIDTH // 2
        source_start = spec_x - screen_center
        source_end = spec_x + screen_center
        
        dest_start = 0
        dest_end = WIDTH
        
        if source_start < 0:
            dest_start = -source_start
            source_start = 0
        if source_end > spec_img.shape[1]:
            diff = source_end - spec_img.shape[1]
            dest_end = WIDTH - diff
            source_end = spec_img.shape[1]
            
        if dest_end > dest_start:
            spec_slice = spec_img[:, source_start:source_end]
            frame[0:spec_slice.shape[0], dest_start:dest_end] = spec_slice
            
        cv2.line(frame, (WIDTH//2, 0), (WIDTH//2, HEIGHT), (0, 255, 255), 2)
        
        # Predictions
        pred_idx = int(current_time / step)
        if pred_idx < len(predictions):
            probs = predictions[pred_idx]
        else:
            probs = np.zeros(len(CLASSES))
            
        bar_width = (WIDTH - 100) // len(CLASSES)
        
        for i, (cls_name, prob) in enumerate(zip(CLASSES, probs)):
            x = 50 + i * bar_width
            h = int(prob * 200)
            
            if prob > 0.6:
                color = (0, 255, 0) 
            else:
                color = (0, int(255 * prob * 1.5), 255)
            
            if h > 0:
                cv2.rectangle(frame, (x + 10, HEIGHT - 50 - h), (x + bar_width - 10, HEIGHT - 50), color, -1)
                cv2.putText(frame, f"{int(prob*100)}%", (x + 10, HEIGHT - 55 - h), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.putText(frame, cls_name, (x + 10, HEIGHT - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        out.write(frame)
        
    out.release()
    
    print("Merging audio...")
    cmd = [
        "ffmpeg", "-y",
        "-i", temp_video,
        "-i", args.file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        args.output
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    os.remove(temp_video)
    print(f"\n🎉 Video saved to {args.output}")

if __name__ == "__main__":
    main()
