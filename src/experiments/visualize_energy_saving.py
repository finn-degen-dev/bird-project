import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VISUALS_DIR

def main():
    print("Generiere Visualisierung für Wake-on-Sound Energieeinsparung...")

    # Simuliere 24 Stunden (in 5-Minuten-Intervallen -> 288 Datenpunkte)
    time_hours = np.linspace(0, 24, 288)
    
    # Simuliere Wald-Aktivität (RMS Energie)
    # Nachts: ruhig (Basisrauschen)
    # 04:30 - 09:00: Morgenchor (sehr aktiv)
    # 12:00 - 15:00: Mittagshitze (eher ruhig)
    # 17:00 - 20:00: Abendchor (aktiv)
    
    energy = np.random.normal(0.02, 0.005, 288) # Grundrauschen (Wind, Blätter)
    
    # Morgenchor (Gauß-Glocke)
    energy += 0.15 * np.exp(-((time_hours - 6.5) ** 2) / (2 * 1.5 ** 2))
    # Abendchor
    energy += 0.08 * np.exp(-((time_hours - 18.5) ** 2) / (2 * 1.0 ** 2))
    
    # Füge ein paar zufällige Spitzen hinzu (z.B. tagsüber mal ein lauter Ruf)
    np.random.seed(42)
    spikes = np.random.choice(288, 15, replace=False)
    energy[spikes] += np.random.uniform(0.05, 0.12, 15)
    
    energy = np.clip(energy, 0, 1) # Begrenze auf 0 bis 1
    
    # Wake-on-Sound Schwellenwert (darunter schläft die KI)
    rms_threshold = 0.04 

    # Berechne Ersparnis
    active_inferences = np.sum(energy > rms_threshold)
    total_inferences = len(energy)
    saved_percent = 100 * (1 - (active_inferences / total_inferences))

    # --- PLOTTING ---
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))
    
    # Plot der Energie-Kurve
    plt.plot(time_hours, energy, color='black', linewidth=1.5, alpha=0.7, label="Akustische Signalenergie (RMS)")
    
    # Fülle Bereiche
    plt.fill_between(time_hours, energy, rms_threshold, where=(energy > rms_threshold), 
                     interpolate=True, color='#d62728', alpha=0.6, label="KI Aktiv (Inferenz wird berechnet)")
    plt.fill_between(time_hours, 0, rms_threshold, 
                     color='#1f77b4', alpha=0.3, label="Deep Sleep (KI pausiert, CPU spart Strom)")
    
    plt.axhline(rms_threshold, color='black', linestyle='--', linewidth=2, label="Wake-on-Sound Schwellenwert")
    
    plt.title("Energieeffizienz durch heuristische Vorfilterung (24h Simulation)", fontsize=16, fontweight='bold')
    plt.xlabel("Uhrzeit (Stunden)", fontsize=13)
    plt.ylabel("Signalamplitude (RMS Energie)", fontsize=13)
    plt.xlim(0, 24)
    plt.ylim(0, max(energy) + 0.05)
    plt.xticks(np.arange(0, 25, 2),[f"{i:02d}:00" for i in range(0, 25, 2)])
    
    # Text-Annotation für die Ersparnis
    plt.text(12, max(energy)*0.85, f"Eingesparte Rechenzeit/Akkuladung: {saved_percent:.1f}%", 
             fontsize=14, fontweight='bold', color='green', 
             bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.5'))
    
    plt.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(VISUALS_DIR, exist_ok=True)
    out_path = os.path.join(VISUALS_DIR, "wake_on_sound_energy.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Energie-Plot erfolgreich gespeichert unter: {out_path}")

if __name__ == "__main__":
    main()
