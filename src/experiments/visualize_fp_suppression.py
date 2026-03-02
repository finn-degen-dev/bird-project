import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISUALS_DIR

def main():
    print("Generiere Visualisierung für Root-Scaled Context Suppression...")

    time_windows = np.arange(0, 60, 5)

    # 1. Echter, aber etwas entfernterer Vogel (True Positive)
    true_bird_raw = np.array([0.1, 0.1, 0.55, 0.70, 0.65, 0.40, 0.1, 0.1, 0.50, 0.1, 0.1, 0.1])
    
    # 2. Fehlalarm: Knackender Ast klingt kurz wie ein Specht (False Positive)
    false_bird_raw = np.array([0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.55, 0.05, 0.05, 0.05, 0.05])

    # --- OUR TWIST: ROOT-SCALED CONTEXT SUPPRESSION ---
    # Formel: P_neu = P_alt * sqrt(P_max_im_file)
    true_bird_max = np.max(true_bird_raw)
    false_bird_max = np.max(false_bird_raw)

    true_bird_scaled = true_bird_raw * np.sqrt(true_bird_max)
    false_bird_scaled = false_bird_raw * np.sqrt(false_bird_max)

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

    threshold = 0.45 

    # Plot 1: Rohdaten
    axes[0].plot(time_windows, true_bird_raw, label="Echter Vogel (Distanziert)", color='#1f77b4', linewidth=3, marker='o')
    axes[0].plot(time_windows, false_bird_raw, label="Isoliertes Störgeräusch", color='#d62728', linewidth=3, linestyle='--', marker='X', markersize=8)
    axes[0].axhline(threshold, color='black', linestyle=':', linewidth=2, label=f"Schwellenwert ({threshold*100:.0f}%)")
    axes[0].set_title("Vorher: Rohe KI-Konfidenz", fontsize=15, fontweight='bold')
    axes[0].set_xlabel("Zeitachse (Sekunden)", fontsize=13)
    axes[0].set_ylabel("Konfidenz (0.0 - 1.0)", fontsize=13)
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right")

    # Plot 2: Nach unserem Root-Scaling
    axes[1].plot(time_windows, true_bird_scaled, label="Echter Vogel (Erhalten)", color='#1f77b4', linewidth=3, marker='o')
    axes[1].plot(time_windows, false_bird_scaled, label="Störgeräusch (Unterdrückt)", color='#d62728', linewidth=3, linestyle='--', marker='X', markersize=8)
    axes[1].axhline(threshold, color='black', linestyle=':', linewidth=2, label=f"Schwellenwert ({threshold*100:.0f}%)")
    axes[1].set_title("Nachher: Root-Scaled Context Suppression", fontsize=15, fontweight='bold')
    axes[1].set_xlabel("Zeitachse (Sekunden)", fontsize=13)
    
    axes[0].annotate('Fehlalarm!', xy=(35, 0.57), xytext=(20, 0.8),
                 arrowprops=dict(facecolor='red', shrink=0.05), fontsize=12, fontweight='bold', color='red')
    axes[1].annotate('Erfolgreich\nunterdrückt', xy=(35, 0.40), xytext=(20, 0.6),
                 arrowprops=dict(facecolor='green', shrink=0.05), fontsize=12, fontweight='bold', color='green')

    plt.suptitle("Algorithmische Fehlalarm-Unterdrückung durch globale Kontext-Skalierung", fontsize=16, fontweight='bold', y=1.05)
    plt.tight_layout()

    out_path = os.path.join(VISUALS_DIR, "false_positive_suppression.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualisierung gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()
