import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import librosa
import librosa.display
import cv2
import os
import glob

# --- Configuration ---
MODEL_PATH = "data/models/bird_classifier_resnet18.pth"
DATA_DIR = "data/raw"
SPECIES_LIST = ["Amsel", "Buchfink", "Haussperling", "Kohlmeise", "Rotkehlchen"]
CLASSES = sorted(SPECIES_LIST)
SAMPLE_RATE = 32000
CHUNK_DURATION = 5
TOTAL_DURATION = 20
STRIDE = 0.01  # Slide the window every 1 second (High resolution!)
N_MELS = 128
FMAX = 16000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Grad-CAM Class ---
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_cam(self, input_tensor, target_class_idx):
        output = self.model(input_tensor)
        self.model.zero_grad()
        target = output[:, target_class_idx]
        target.backward()

        weights = torch.mean(self.gradients, dim=(2, 3))[0]
        cam = torch.zeros(self.activations.shape[2:], dtype=torch.float32).to(device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0, i, :, :]
            
        cam = F.relu(cam)
        cam = cam.cpu().detach().numpy()
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam) + 1e-8)
        return cam

def main():
    print("Loading Model...")
    model = models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    grad_cam = GradCAM(model, model.layer4[-1])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(15, 12))
    fig.suptitle(f'Continuous AI Attention (Sliding Window Stride: {STRIDE}s)', fontsize=16)

    for i, species in enumerate(SPECIES_LIST):
        print(f"Processing {species}...")
        search_path = os.path.join(DATA_DIR, species, "*.mp3")
        files = glob.glob(search_path)
        if not files: continue
        file_path = files[0]

        # 1. Load full 20s Audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=TOTAL_DURATION)
        target_len = TOTAL_DURATION * SAMPLE_RATE
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]

        # 2. Compute Full Spectrogram (Background)
        full_melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
        full_melspec_db = librosa.power_to_db(full_melspec, ref=np.max)
        
        # Normalize Full Spectrogram for display (0-1)
        spec_min, spec_max = full_melspec_db.min(), full_melspec_db.max()
        display_spec = (full_melspec_db - spec_min) / (spec_max - spec_min + 1e-6)

        # 3. Create Accumulator Buffers
        # shape is (n_mels, time_steps)
        heatmap_sum = np.zeros_like(display_spec)
        heatmap_count = np.zeros_like(display_spec)

        # 4. Sliding Window Logic
        samples_per_chunk = int(CHUNK_DURATION * sr)
        samples_stride = int(STRIDE * sr)
        
        # Calculate how many pixels correspond to one chunk in the spectrogram
        # librosa stft behavior: time_steps = samples / hop_length
        # We need to map time -> pixels accurately
        pixels_per_sec = display_spec.shape[1] / TOTAL_DURATION

        current_sample = 0
        while current_sample + samples_per_chunk <= len(y):
            # Extract Audio Chunk
            y_chunk = y[current_sample : current_sample + samples_per_chunk]
            
            # Create Mel Spec for Model (Match training preprocessing exactly)
            chunk_melspec = librosa.feature.melspectrogram(y=y_chunk, sr=sr, n_mels=N_MELS, fmax=FMAX)
            chunk_db = librosa.power_to_db(chunk_melspec, ref=np.max)
            
            # Normalize 0-255 uint8 for PIL
            c_min, c_max = chunk_db.min(), chunk_db.max()
            chunk_norm = (chunk_db - c_min) / (c_max - c_min + 1e-6)
            chunk_img_data = (np.flipud(chunk_norm) * 255).astype(np.uint8)
            pil_img = Image.fromarray(chunk_img_data).convert("RGB") # FIX APPLIED HERE

            # Get GradCAM
            input_tensor = transform(pil_img).unsqueeze(0).to(device)
            target_idx = CLASSES.index(species)
            cam = grad_cam.generate_cam(input_tensor, target_idx) # returns 224x224 typically

            # Mapping back to the global spectrogram canvas
            # Start and End time of this window
            t_start = current_sample / sr
            t_end = (current_sample + samples_per_chunk) / sr
            
            # Convert time to spectrogram pixel indices
            px_start = int(t_start * pixels_per_sec)
            px_end = int(t_end * pixels_per_sec)
            
            # Safety check for dimensions
            if px_end > display_spec.shape[1]: px_end = display_spec.shape[1]
            width = px_end - px_start
            if width <= 0: break

            # Resize CAM to fit this specific slice of the spectrogram
            # We resize to (width, 128)
            cam_resized = cv2.resize(cam, (width, N_MELS))
            cam_resized = np.flipud(cam_resized) # Flip back to match librosa orientation

            # Add to accumulators
            heatmap_sum[:, px_start:px_end] += cam_resized
            heatmap_count[:, px_start:px_end] += 1.0

            # Slide
            current_sample += samples_stride

        # 5. Average and Normalize Heatmap
        # Avoid division by zero
        heatmap_count[heatmap_count == 0] = 1.0
        final_heatmap = heatmap_sum / heatmap_count
        
        # Normalize final 0-1
        if final_heatmap.max() > 0:
            final_heatmap = final_heatmap / final_heatmap.max()

        # 6. Plotting
        ax = axes[i]
        
        # Background: Spectrogram (Grayscale)
        ax.imshow(display_spec, aspect='auto', origin='lower', cmap='gray', extent=[0, 20, 0, FMAX])
        
        # Foreground: Heatmap (Jet, Alpha blended)
        # We need to map 0-1 values to RGB color map manually for overlay
        colored_map = plt.cm.jet(final_heatmap) # Returns RGBA
        # Set alpha channel based on intensity (so low attention is transparent)
        colored_map[:, :, 3] = final_heatmap * 0.6 
        
        ax.imshow(colored_map, aspect='auto', origin='lower', extent=[0, 20, 0, FMAX])

        ax.set_title(f"{species}")
        ax.set_ylabel("Freq (Hz)")
        if i < 4:
            ax.set_xticks([])
        else:
            ax.set_xlabel("Time (seconds)")

    plt.tight_layout()
    save_path = "src/visuals/sliding_window_attention.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nSaved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
