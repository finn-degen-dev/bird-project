import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from models import get_model
from data import SmartBirdDataset

def main():
    print(" T-SNE Visualisierung: Rufe vs. Gesänge...")

    exp_name = "efficientnet_task-type_attn-False_layers-3"
    model_path = os.path.join(MODELS_DIR, f"{exp_name}_best.pth")

    if not os.path.exists(model_path):
        print(f" Bitte zuerst das Modell trainieren: python src/train.py --model efficientnet --task type")
        return

    ds = SmartBirdDataset(mode='test', task='type')
    indices = np.random.choice(len(ds), min(1000, len(ds)), replace=False)
    loader = DataLoader(Subset(ds, indices), batch_size=32, shuffle=False)

    model = get_model("efficientnet", 2).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    features =[]
    def hook(module, input, output):
        features.append(input[0].cpu().numpy())
    handle = model.classifier[1].register_forward_hook(hook)

    labels_list =[]
    with torch.no_grad():
        for inputs, labels in loader:
            _ = model(inputs.to(DEVICE))
            labels_list.extend(labels.numpy())

    handle.remove()
    embeddings = np.concatenate(features)

    print("Berechne T-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    class_names = [ds.classes[i] for i in labels_list]
    sns.scatterplot(x=features_2d[:, 0], y=features_2d[:, 1], hue=class_names, palette={'song': '#1f77b4', 'call': '#ff7f0e'}, s=60)
    plt.title("Internes KI-Verständnis: Ruf vs. Gesang")

    out_path = os.path.join(VISUALS_DIR, "tsne_song_vs_call.png")
    plt.savefig(out_path, dpi=300)
    print(f" Gespeichert unter {out_path}")

if __name__ == "__main__":
    main()
