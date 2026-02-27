import os
import torch

# Absolute Pfade für garantierte Stabilität, egal von wo das Skript aufgerufen wird
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_tensors")
MODELS_DIR = os.path.join(DATA_DIR, "models")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals", "experiments")

# Audio & Spektrogramm Konstanten
SAMPLE_RATE = 32000
DURATION = 5  # Sekunden
N_MELS = 128
FMAX = 16000
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sicherstellen, dass die Ausgabe-Ordner existieren
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
