import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from config import *

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx):
        output = self.model(x)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)

        cam -= cam.min()
        cam /= cam.max()
        return cam.cpu().detach().numpy()

def main():

    model = get_model("resnet", len(CLASSES)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "resnet_best.pth"), map_location=DEVICE))
    model.eval()

    cam_extractor = GradCAM(model, model.layer4[-1])

    sample_path = os.path.join(RAW_AUDIO_DIR, "Amsel/beispiel.mp3")
    y, sr = librosa.load(sample_path, sr=SAMPLE_RATE, duration=DURATION)

    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, fmax=FMAX)
    log_melspec = librosa.power_to_db(melspec, ref=np.max)
    log_melspec = (log_melspec - log_melspec.min()) / (log_melspec.max() - log_melspec.min())

    tensor = torch.tensor(log_melspec, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(DEVICE)
    tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

    heatmap = cam_extractor(tensor, class_idx=0)

    plt.figure(figsize=(10, 5))
    plt.imshow(log_melspec, origin='lower', aspect='auto', cmap='gray')

    import cv2
    heatmap_resized = cv2.resize(heatmap, (log_melspec.shape[1], log_melspec.shape[0]))
    plt.imshow(heatmap_resized, origin='lower', aspect='auto', cmap='jet', alpha=0.5)

    plt.title("Grad-CAM: Worauf achtet die KI?")
    plt.ylabel("Frequenz")
    plt.xlabel("Zeit")
    plt.savefig("docs/poster/gradcam_amsel.png", dpi=300)
    print("Heatmap gespeichert unter docs/poster/gradcam_amsel.png")

if __name__ == "__main__":
    main()
