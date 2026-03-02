import sys
import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from collections import Counter
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from data import SmartBirdDataset
from models import get_model

def get_distractor_indices(classes):
    non_bird_keywords =[
        'cricket', 'frog', 'toad', 'cicada', 'bat', 'grasshopper', 
        'katydid', 'roesel', 'mouse', 'mole', 'shrew', 'squirrel', 
        'deer', 'fox', 'wolf', 'marmot', 'locust', 'insect', 'bush-cricket'
    ]
    bird_idx, distractor_idx = [],[]
    for i, c in enumerate(classes):
        if any(kw in c.lower() for kw in non_bird_keywords):
            distractor_idx.append(i)
        else:
            bird_idx.append(i)
    return set(bird_idx), distractor_idx

def main():
    print("Lade Datensatz und Modell für Fehleranalyse...")
    
    exp_name = "efficientnet_task-species_attn-False_layers-3"
    classes_path = os.path.join(MODELS_DIR, f"{exp_name}_classes.json")
    
    with open(classes_path, "r") as f:
        classes = json.load(f)['classes']
        
    bird_idx_set, distractor_idx = get_distractor_indices(classes)

    model = get_model("efficientnet", len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{exp_name}_best.pth"), map_location=DEVICE))
    model.eval()

    # Nutze das Test-Set
    dataset = SmartBirdDataset(mode='test', task='species')
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    error_counter = Counter()

    print("\nScanne Testdaten und sammle Verwechslungen...")
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Evaluiere"):
            inputs = inputs.to(DEVICE)
            logits = model(inputs)
            
            # WICHTIG: Setze die Wahrscheinlichkeit für Insekten/Amphibien auf minus unendlich!
            # Dadurch WIRD das Modell gezwungen, einen Vogel zu raten.
            logits[:, distractor_idx] = -float('inf')
            
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            
            for true_idx, pred_idx in zip(labels, preds):
                # Wir werten nur Aufnahmen aus, auf denen wirklich ein Vogel zu hören ist
                if true_idx in bird_idx_set:
                    if true_idx != pred_idx:
                        true_name = classes[true_idx]
                        pred_name = classes[pred_idx]
                        # Erstelle einen lesbaren String für das Diagramm
                        pair_str = f"Wahr: {true_name}\nKI rät: {pred_name}"
                        error_counter[pair_str] += 1

    # Hole die Top 15 Verwechslungen
    top_errors = error_counter.most_common(15)
    
    if not top_errors:
        print("Keine Fehler gefunden?! Dein Modell ist zu gut.")
        return
        
    # Daten für Plotting vorbereiten
    error_labels = [item[0] for item in top_errors]
    error_counts = [item[1] for item in top_errors]
    
    # --- PLOTTING ---
    plt.figure(figsize=(12, 10))
    sns.set_theme(style="whitegrid")
    
    # Farbverlauf von Rot (häufigster Fehler) zu Orange
    palette = sns.color_palette("flare", len(error_labels))[::-1]
    
    ax = sns.barplot(x=error_counts, y=error_labels, palette=palette)
    
    plt.title("Die 15 häufigsten KI-Verwechslungen (Nur Vögel)", fontsize=16, fontweight='bold', pad=20)
    plt.xlabel("Anzahl der Fehlklassifikationen im Test-Set", fontsize=14)
    plt.ylabel("", fontsize=12)
    
    # Zahlenwerte an die Balken schreiben
    for i, v in enumerate(error_counts):
        ax.text(v + 0.5, i + 0.1, str(v), color='black', fontweight='bold', fontsize=12)

    # Etwas Platz rechts lassen, damit die Zahlen nicht abgeschnitten werden
    plt.xlim(0, max(error_counts) * 1.15)
    
    plt.tight_layout()
    out_path = os.path.join(VISUALS_DIR, "top_errors.png")
    plt.savefig(out_path, dpi=300)
    print(f"\n✅ Erfolgreich gespeichert! Schau dir: {out_path} an.")
    print("Tipp: Du kannst das Bild direkt in dein LaTeX Dokument (main.tex) einbinden!")

if __name__ == "__main__":
    main()
