import os
import glob
import torch
import pandas as pd
from tqdm import tqdm
import re

try:
    from src.models import get_model
except ImportError:
    from models import get_model

MODELS_DIR = "data/models"
VAL_TENSORS_DIR = "data/tensors/val"
CSV_OUT_DIR = "data/visuals/csv_data"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 55

os.makedirs(CSV_OUT_DIR, exist_ok=True)

def parse_model_params_from_name(filename):
    """Liest die Modell-Architektur aus dem Dateinamen ab."""
    name = filename.lower()

    if "efficientnet" in name: model_type = "efficientnet"
    elif "resnet" in name: model_type = "resnet"
    else: model_type = "custom"

    use_attention = "attn_true" in name

    cnn_layers = 3
    layer_match = re.search(r'layers_(\d+)', name)
    if layer_match:
        cnn_layers = int(layer_match.group(1))

    return model_type, use_attention, cnn_layers

def group_chunks_to_files(val_dir):
    """
    Sucht alle 5s-Chunks (.pt) und gruppiert sie nach der Original-Audiodatei.
    Erwartetes Format: data/tensors/val/<Klasse>/<Datei_ID>_<ChunkNummer>.pt
    """
    file_groups = {}
    class_to_idx = {}

    classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    for idx, cls_name in enumerate(classes):
        class_to_idx[cls_name] = idx

        cls_dir = os.path.join(val_dir, cls_name)
        chunks = glob.glob(os.path.join(cls_dir, "*.pt"))

        for chunk_path in chunks:
            filename = os.path.basename(chunk_path)

            base_file_id = "_".join(filename.split("_")[:-1])

            group_key = f"{cls_name}/{base_file_id}"

            if group_key not in file_groups:
                file_groups[group_key] = {
                    "class_idx": idx,
                    "chunks":[]
                }
            file_groups[group_key]["chunks"].append(chunk_path)

    return file_groups, classes

def main():
    print(f" Suche nach Modellen in {MODELS_DIR}...")
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))

    if not model_files:
        print(" Keine trainierten Modelle (.pth) gefunden!")
        return

    print(" Gruppiere Tensor-Chunks zu Original-Audiodateien (Sliding Window)...")
    file_groups, classes = group_chunks_to_files(VAL_TENSORS_DIR)
    print(f"   => {len(file_groups)} komplette Audiodateien aus {len(classes)} Klassen gefunden.")

    results =[]

    for model_path in model_files:
        model_filename = os.path.basename(model_path)

        if "best" not in model_filename and "task_species" not in model_filename:
            continue

        print(f"\n Evaluiere Modell: {model_filename}")

        m_type, use_attn, m_layers = parse_model_params_from_name(model_filename)

        try:
            model = get_model(m_type, len(classes), use_attn, m_layers).to(DEVICE)
            model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            model.eval()
        except Exception as e:
            print(f" Konnte {model_filename} nicht laden. Fehler: {e}")
            continue

        correct_files = 0
        total_files = len(file_groups)

        with torch.no_grad():
            for group_key, data in tqdm(file_groups.items(), desc="Audio-Dateien"):
                true_class = data["class_idx"]
                chunk_paths = data["chunks"]

                batch =[]
                for p in chunk_paths:
                    tensor = torch.load(p)

                    if tensor.ndim == 3:
                        batch.append(tensor)

                if not batch: continue

                batch = torch.stack(batch).to(DEVICE)
                outputs = model(batch)

                avg_output = outputs.mean(dim=0)
                predicted_class = avg_output.argmax().item()

                if predicted_class == true_class:
                    correct_files += 1

        file_accuracy = (correct_files / total_files) * 100 if total_files > 0 else 0
        print(f" Sliding Window Acc: {file_accuracy:.2f}%")

        clean_name = model_filename.replace('.pth', '').replace('best_', '')

        results.append({
            "model_name": clean_name,
            "sliding_window_acc": file_accuracy
        })

    df = pd.DataFrame(results)
    out_csv = os.path.join(CSV_OUT_DIR, "sliding_window_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\n Alle Sliding-Window Ergebnisse gespeichert in: {out_csv}")

if __name__ == "__main__":
    main()
