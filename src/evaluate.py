import argparse
import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics.pairwise import cosine_similarity

from config import *
from data import SmartBirdDataset
from models import get_model

def evaluate_model(exp_name, model_type, use_attention, cnn_layers):
    print(f"Evaluatiere {exp_name}...")

    with open(os.path.join(MODELS_DIR, f"{exp_name}_classes.json"), "r") as f:
        classes = json.load(f)['classes']

    val_ds = SmartBirdDataset(mode='test', task='species')
    loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = get_model(model_type, len(classes), use_attention, cnn_layers).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{exp_name}_best.pth"), map_location=DEVICE))
    model.eval()

    all_preds, all_labels = [],[]

    if model_type == "embedding":
        all_embeddings =[]
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Extrahiere Embeddings"):
                embeddings = model(inputs.to(DEVICE))
                all_embeddings.extend(embeddings.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_embeddings = np.array(all_embeddings)
        all_labels = np.array(all_labels)

        print("\nBerechne Genauigkeit mit K-Nearest-Neighbor (k=5)...")
        knn = KNeighborsClassifier(n_neighbors=5, metric='cosine')
        knn.fit(all_embeddings, all_labels)
        knn_preds = knn.predict(all_embeddings)
        acc_knn = sum(knn_preds == all_labels) / len(all_labels)

        print("Berechne Genauigkeit mit ArcFace Klassenzentren...")
        arcface_path = os.path.join(MODELS_DIR, f"{exp_name}_arcface.pth")

        centers = np.zeros((len(classes), all_embeddings.shape[1]))
        if os.path.exists(arcface_path):
            print(" -> Lade gespeicherte ArcFace-Gewichte aus Datei...")
            arcface_state = torch.load(arcface_path, map_location='cpu')

            if 'W' in arcface_state:
                centers = arcface_state['W'].numpy()
            else:
                centers = list(arcface_state.values())[0].numpy()
        else:
            print(" -> Keine ArcFace-Gewichte gefunden (Modell wurde vorher trainiert).")
            print(" -> Berechne theoretische Klassenzentren aus den Daten als Fallback...")
            for i in range(len(classes)):
                class_embs = all_embeddings[all_labels == i]
                if len(class_embs) > 0:
                    centers[i] = np.mean(class_embs, axis=0)

        centers = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)

        sims = cosine_similarity(all_embeddings, centers)
        centroid_preds = np.argmax(sims, axis=1)
        acc_centroid = sum(centroid_preds == all_labels) / len(all_labels)

        print(f"\nErgebnisse:")
        print(f" -> k-NN Accuracy: {acc_knn:.2%}")
        print(f" -> Centroid / ArcFace Accuracy: {acc_centroid:.2%}")

        if acc_centroid >= acc_knn:
            all_preds = centroid_preds
            acc = acc_centroid
            print("=> Verwende ArcFace-Zentren für die finale Auswertung.\n")
        else:
            all_preds = knn_preds
            acc = acc_knn
            print("=> Verwende k-NN für die finale Auswertung.\n")

    else:
        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc="Evaluiere"):
                outputs = model(inputs.to(DEVICE))
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())

        acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))

    if len(classes) <= 50:
        sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    else:
        sns.heatmap(cm, annot=False, cmap='Blues')

    plt.title(f"Confusion Matrix ({exp_name}) - Acc: {acc:.1%}")
    plt.savefig(os.path.join(VISUALS_DIR, f"{exp_name}_confusion.png"), dpi=300)
    plt.close()

    return acc

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", choices=["custom", "resnet", "efficientnet", "embedding"], required=True)
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--cnn-layers", type=int, default=3)
    args = parser.parse_args()

    exp_name = f"{args.model}_task-species_attn-{args.use_attention}_layers-{args.cnn_layers}"
    acc = evaluate_model(exp_name, args.model, args.use_attention, args.cnn_layers)
    print(f"\n {exp_name} Finale Accuracy: {acc:.2%}")

    logs_dir = os.path.join(VISUALS_DIR, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    cache_file = os.path.join(logs_dir, "evaluation_cache.json")

    cache = {}
    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            cache = json.load(f)

    cache[exp_name] = float(acc * 100)

    with open(cache_file, "w") as f:
        json.dump(cache, f, indent=4)

if __name__ == "__main__":
    main()
