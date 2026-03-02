import sys
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import get_model
from config import *
from data import SmartBirdDataset

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

def main():
    print("Lade Datensatz und Modell für hochauflösendes Grad-CAM...")

    exp_name = "resnet_task-species_attn-False_layers-3"
    classes_path = os.path.join(MODELS_DIR, f"{exp_name}_classes.json")

    with open(classes_path, "r") as f:
        classes = json.load(f)['classes']

    model = get_model("resnet", len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{exp_name}_best.pth"), map_location=DEVICE))
    model.eval()

    cam_extractor = GradCAM(model, model.layer3[-1])

    dataset = SmartBirdDataset(mode='test', task='species')

    all_indices = list(range(len(dataset)))
    random.shuffle(all_indices)

    selected_indices =[]
    seen_classes = set()

    non_birds =['cricket', 'frog', 'toad', 'cicada', 'bat', 'grasshopper', 'katydid', 'roesel', 'mouse']

    for idx in all_indices:
        row = dataset.df.iloc[idx]
        pt_path = os.path.join(PROCESSED_DIR, row['filename'].replace('.mp3', '.pt'))

        species_name = row['species'].lower()
        is_bird = not any(ignore_word in species_name for ignore_word in non_birds)

        if os.path.exists(pt_path) and is_bird:
            true_label = dataset.class_to_idx[row['species']]
            if true_label not in seen_classes:
                seen_classes.add(true_label)
                selected_indices.append((idx, pt_path, true_label))
        if len(selected_indices) >= 5:
            break

    fig, axes = plt.subplots(5, 1, figsize=(14, 22))

    target_width = TARGET_WIDTH
    stride = target_width // 2

    for i, (idx, pt_path, true_label) in enumerate(selected_indices):
        true_name = classes[true_label]

        spec = torch.load(pt_path).float()
        current_width = spec.shape[1]

        global_cam = np.zeros((128, current_width))
        counts = np.zeros((128, current_width))

        windows =[]
        if current_width < target_width:
            pad_amount = target_width - current_width
            spec_padded = torch.nn.functional.pad(spec, (0, pad_amount))
            windows.append((0, current_width, spec_padded))
        else:
            for start in range(0, current_width - target_width + 1, stride):
                windows.append((start, start + target_width, spec[:, start:start+target_width]))
            if (current_width - target_width) % stride != 0:
                start = current_width - target_width
                windows.append((start, current_width, spec[:, start:current_width]))

        max_window_conf = 0.0

        for start, end, window_spec in windows:
            tensor = window_spec.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
            tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
            tensor = tensor.to(DEVICE)
            tensor.requires_grad_(True)

            model.zero_grad()
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)

            conf = probs[0, true_label].item()
            if conf > max_window_conf:
                max_window_conf = conf

            logits[0, true_label].backward()

            if cam_extractor.gradients is not None and cam_extractor.activations is not None:
                weights = torch.mean(cam_extractor.gradients, dim=(2, 3), keepdim=True)
                cam = torch.sum(weights * cam_extractor.activations, dim=1).squeeze()
                cam = F.relu(cam).cpu().detach().numpy()

                cam_min, cam_max = cam.min(), cam.max()
                if cam_max > cam_min:
                    cam = (cam - cam_min) / (cam_max - cam_min)
                elif cam_max > 0:
                    cam = cam / cam_max

                cam = cam * conf
                cam_resized = cv2.resize(cam, (target_width, 128))
                valid_len = end - start
                global_cam[:, start:end] += cam_resized[:, :valid_len]
                counts[:, start:end] += 1

        global_cam = global_cam / np.maximum(counts, 1)

        g_min, g_max = global_cam.min(), global_cam.max()
        if g_max > g_min:
            global_cam = (global_cam - g_min) / (g_max - g_min)
        elif g_max > 0:
            global_cam = global_cam / g_max

        spectrogram = spec.cpu().numpy()
        spectrogram = np.flipud(spectrogram)
        spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)

        global_cam_flipped = np.flipud(global_cam)

        audio_seconds = current_width * (HOP_LENGTH / SAMPLE_RATE)
        extent =[0, audio_seconds, 0, FMAX/1000]

        cmap = plt.get_cmap('magma')
        heatmap_rgba = cmap(global_cam_flipped)

        heatmap_rgba[..., 3] = (global_cam_flipped ** 1.5) * 0.9

        ax = axes[i]
        ax.imshow(spectrogram, origin='lower', aspect='auto', cmap='gray', extent=extent)
        ax.imshow(heatmap_rgba, origin='lower', aspect='auto', extent=extent)

        title = f"Art: {true_name} | Max. Konfidenz: {max_window_conf:.1%}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel("Frequenz (kHz)", fontsize=12)

    axes[-1].set_xlabel("Zeitachse (Sekunden)", fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(VISUALS_DIR, "gradcam_sliding_window.png")
    plt.savefig(out_path, dpi=300)
    print(f"\n Hochauflösende Sliding-Window Heatmaps erfolgreich generiert unter: {out_path}")

if __name__ == "__main__":
    main()
