import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

CSV_DIR = "data/visuals/csv_data"
VISUALS_DIR = "visuals/experiments"

os.makedirs(VISUALS_DIR, exist_ok=True)

def plot_graph(csv_files, config, title, output_filename):
    """Universelle Funktion, um basierend auf einer Konfiguration Graphen zu zeichnen."""
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid")

    found_any = False

    for file in csv_files:
        filename = os.path.basename(file)

        matched_label = None
        matched_color = None

        for keywords, label, color in config:

            if all(k in filename for k in keywords):
                matched_label = label
                matched_color = color
                break

        if matched_label is None:
            continue

        df = pd.read_csv(file)
        if len(df) == 0: continue
        found_any = True

        max_acc = df['val_acc'].max()
        plt.plot(df['epoch'], df['val_acc'],
                 label=f"{matched_label} (Max: {max_acc:.1f}%)",
                 color=matched_color,
                 linewidth=2.5, marker='o', markersize=5)

    if not found_any:
        print(f" Warnung: Keine passenden Daten für '{title}' gefunden.")
        plt.close()
        return

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel("Trainingsepoche", fontsize=14)
    plt.ylabel("Validierungs-Genauigkeit (%)", fontsize=14)
    plt.ylim(0, 100)
    plt.xlim(1, None)

    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(title="Architekturen", fontsize=12, title_fontsize=13, loc="lower right")

    out_path = os.path.join(VISUALS_DIR, output_filename)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f" Graph gespeichert: {out_path}")
    plt.close()

def main():
    csv_files = glob.glob(os.path.join(CSV_DIR, "*.csv"))
    if not csv_files:
        print(f" Keine CSV-Dateien in {CSV_DIR} gefunden. Führe zuerst parse_console.py aus!")
        return

    print("Erstelle Graphen aus den CSV-Daten...")

    custom_cnn_config = [
        (["custom", "task_species", "attn_true", "layers_3"], "Custom CNN (3-Layer + Attention)", "#2ca02c"),
        (["custom", "task_species", "attn_false", "layers_5"], "Custom CNN (5-Layer)", "#d62728"),
        (["custom", "task_species", "attn_false", "layers_3"], "Custom CNN (3-Layer Standard)", "#1f77b4"),
        (["custom", "task_species", "attn_false", "layers_2"], "Custom CNN (2-Layer)", "#9467bd")
    ]

    plot_graph(
        csv_files=csv_files,
        config=custom_cnn_config,
        title="Training custom cnn",
        output_filename="graph1_custom_cnn_comparison.png"
    )

    best_models_config = [
        (["efficientnet", "task_species"], "EfficientNet", "#2ca02c"),
        (["resnet", "task_species"], "ResNet18", "#1f77b4"),
        (["custom", "task_species", "attn_false", "layers_3"], "Bestes Custom CNN (3-Layer)", "#ff7f0e")
    ]

    plot_graph(
        csv_files=csv_files,
        config=best_models_config,
        title="Training custom cnn vs. pretrained cnn",
        output_filename="graph2_best_models_comparison.png"
    )

    sliding_window_csv = os.path.join(CSV_DIR, "sliding_window_comparison.csv")

    if os.path.exists(sliding_window_csv):
        print("Erstelle Graph 3: Frame vs. Sliding Window Accuracy...")
        df_sliding = pd.read_csv(sliding_window_csv)

        frame_accs = {}
        for file in csv_files:
            if "sliding" in file: continue
            m_name = os.path.basename(file).replace('.csv', '')
            df_frame = pd.read_csv(file)
            frame_accs[m_name] = df_frame['val_acc'].max()

        plot_data =[]
        for _, row in df_sliding.iterrows():
            m_name = row['model_name']
            s_acc = row['sliding_window_acc']
            f_acc = frame_accs.get(m_name, 0.0)

            display_name = m_name.replace('task_species_attn_', '')                                 .replace('false', 'NoAttn')                                 .replace('true', 'Attn')                                 .replace('_', ' ').title()

            plot_data.append({"Modell": display_name, "Genauigkeit": f_acc, "Methode": "Einzelner 5s Frame"})
            plot_data.append({"Modell": display_name, "Genauigkeit": s_acc, "Methode": "Ganze Datei (Sliding Window)"})

        df_plot = pd.DataFrame(plot_data)

        plt.figure(figsize=(14, 7))
        sns.set_theme(style="whitegrid")
        ax = sns.barplot(data=df_plot, x="Modell", y="Genauigkeit", hue="Methode", palette=["#1f77b4", "#ff7f0e"])

        plt.title("Auswirkung von Sliding-Window (Zeitliche Aggregation) auf die KI-Leistung", fontsize=16, fontweight='bold')
        plt.xlabel("KI-Architektur", fontsize=14)
        plt.ylabel("Genauigkeit (%)", fontsize=14)
        plt.ylim(0, 100)
        plt.xticks(rotation=45, ha="right")

        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=10)

        plt.legend(title="Evaluierungs-Methode", loc="lower right")
        plt.tight_layout()

        out_path = os.path.join(VISUALS_DIR, "graph3_sliding_window_impact.png")
        plt.savefig(out_path, dpi=300)
        print(f"Graph 3 gespeichert: {out_path}")

if __name__ == "__main__":
    main()
