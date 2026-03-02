import argparse
import time
import torch
import torch.nn.functional as F
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from tqdm import tqdm
import sounddevice as sd

from config import *
from models import get_model

RMS_THRESHOLD = 0.005

def preprocess_chunk(y_chunk, sr):
    """Wandelt einen Audio-Ausschnitt exakt wie im Training in einen Tensor um."""

    rms = np.sqrt(np.mean(y_chunk**2))
    if rms < RMS_THRESHOLD:
        return None

    target_len = int(sr * DURATION)
    if len(y_chunk) < target_len:
        y_chunk = np.pad(y_chunk, (0, target_len - len(y_chunk)))
    else:
        y_chunk = y_chunk[:target_len]

    melspec = librosa.feature.melspectrogram(
        y=y_chunk, sr=sr, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH
    )
    log_melspec = librosa.power_to_db(melspec, ref=np.max)

    min_val, max_val = log_melspec.min(), log_melspec.max()
    if max_val - min_val < 1e-6:
        return None

    spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
    spec_norm = np.flipud(spec_norm).copy()

    tensor = torch.tensor(spec_norm, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)

    return tensor

def live_showcase(model, classes, threshold, model_type):
    print("\n LIVE MIKROFON MODUS GESTARTET! (Beenden mit Strg+C)")
    print("Spiele Vogelstimmen über dein Handy ab...")

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.canvas.manager.set_window_title("Piepmatz Polizei - Live Radar")

    buffer_size = int(SAMPLE_RATE * DURATION)
    audio_buffer = np.zeros(buffer_size, dtype=np.float32)

    results =[]
    start_time = time.time()

    def audio_callback(indata, frames, time_info, status):
        """Wird kontinuierlich vom Mikrofon aufgerufen"""
        nonlocal audio_buffer
        if status:
            pass
        new_data = indata[:, 0]

        audio_buffer = np.roll(audio_buffer, -len(new_data))
        audio_buffer[-len(new_data):] = new_data

    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
            while True:
                time.sleep(1.0)

                chunk = audio_buffer.copy()
                tensor = preprocess_chunk(chunk, SAMPLE_RATE)

                if tensor is None:
                    continue

                tensor = tensor.to(DEVICE)
                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1)

                conf, idx = torch.max(probs, 1)
                conf = conf.item()
                idx = idx.item()

                if conf >= threshold:
                    current_time = time.time() - start_time
                    bird_name = classes[idx]

                    non_birds =['cricket', 'frog', 'toad', 'cicada', 'bat', 'grasshopper']
                    if any(x in bird_name.lower() for x in non_birds):
                        print(f"[{current_time:04.1f}s]  Ignoriere Insekt/Amphibie: {bird_name}")
                        continue

                    results.append({"Zeit (s)": current_time, "Vogelart": bird_name, "Konfidenz": conf})
                    print(f"[{current_time:04.1f}s]  {bird_name:<20} | Sicherheit: {conf:.1%}")

                ax.clear()
                if results:
                    df = pd.DataFrame(results)
                    sns.scatterplot(
                        data=df, x="Zeit (s)", y="Vogelart", size="Konfidenz",
                        sizes=(100, 400), hue="Vogelart", palette="Set1", ax=ax, legend=False
                    )

                    max_time = current_time
                    min_time = max(0, max_time - 30)
                    ax.set_xlim(min_time, max_time + 2)

                ax.set_title(f"Live-Radar: Was hört die KI gerade? | {model_type.capitalize()}", fontsize=14)
                ax.grid(True, linestyle='--', alpha=0.4)
                plt.pause(0.01)

    except KeyboardInterrupt:
        print("\n Live-Modus beendet.")
        plt.ioff()
        plt.show()

def file_showcase(audio_path, model, classes, threshold, model_type):
    print(f" Analysiere Datei: {audio_path}")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    print(f"Länge der Aufnahme: {len(y)/sr:.1f} Sekunden")

    window_samples = int(DURATION * sr)
    stride_samples = int((DURATION / 2.0) * sr)

    results =[]
    print(" Starte Sliding-Window Scan...")
    for start in tqdm(range(0, len(y) - window_samples + 1, stride_samples)):
        chunk = y[start : start + window_samples]
        tensor = preprocess_chunk(chunk, sr)
        if tensor is None: continue

        with torch.no_grad():
            logits = model(tensor.to(DEVICE))
            probs = F.softmax(logits, dim=1)

        conf, idx = torch.max(probs, 1)
        if conf.item() >= threshold:
            results.append({"Zeit (s)": start / sr, "Vogelart": classes[idx.item()], "Konfidenz": conf.item()})

    if not results:
        print(f" Keine Vögel gefunden (> {threshold*100:.0f}% Konfidenz).")
        return

    df = pd.DataFrame(results)
    print("\n Gefundene Vögel:")
    print(df['Vogelart'].value_counts().to_string())

    plt.figure(figsize=(14, 7))
    sns.scatterplot(data=df, x="Zeit (s)", y="Vogelart", size="Konfidenz", sizes=(50, 300), hue="Konfidenz", palette="viridis")
    plt.title(f"Aktivitäts-Timeline | Datei: {os.path.basename(audio_path)}")
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    out_path = os.path.join(VISUALS_DIR, f"showcase_{model_type}.png")
    plt.savefig(out_path, dpi=300)
    print(f" Grafik gespeichert unter: {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Live-Radar oder Datei-Analyse für Vogelstimmen")

    parser.add_argument("--live", action="store_true", help="Startet das Live-Mikrofon Radar")
    parser.add_argument("--file", type=str, help="Analysiert eine Audio-Datei (.mp3/.wav)")

    parser.add_argument("--model", choices=["custom", "resnet", "efficientnet"], default="efficientnet")
    parser.add_argument("--use-attention", action="store_true")
    parser.add_argument("--cnn-layers", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.60, help="Mindest-Sicherheit der KI")

    args = parser.parse_args()

    if not args.live and not args.file:
        print(" Du musst entweder --live ODER --file [PFAD] angeben!")
        return

    exp_name = f"{args.model}_task-species_attn-{args.use_attention}_layers-{args.cnn_layers}"
    json_path = os.path.join(MODELS_DIR, f"{exp_name}_classes.json")

    if not os.path.exists(json_path):
        print(f" Modell {exp_name} nicht gefunden. Hast du es trainiert?")
        return

    with open(json_path, "r") as f:
        classes = json.load(f)['classes']

    model = get_model(args.model, len(classes), args.use_attention, args.cnn_layers).to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"{exp_name}_best.pth"), map_location=DEVICE))
    model.eval()

    if args.live:
        live_showcase(model, classes, args.threshold, args.model)
    else:
        file_showcase(args.file, model, classes, args.threshold, args.model)

if __name__ == "__main__":
    main()
