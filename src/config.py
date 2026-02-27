import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_tensors")
MODELS_DIR = os.path.join(DATA_DIR, "models")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals", "experiments")

# --- AUDIO & BILD PARAMETER ---
SAMPLE_RATE = 32000
DURATION = 5         # <--- Ändere das, und der ganze Code passt sich an!
N_MELS = 128
FMAX = 16000
IMG_SIZE = 224
HOP_LENGTH = 512     # Librosa Standard. Wichtig für die Zeit-Berechnung!

# Errechnet automatisch die nötige Breite des Spektrogramms (Pixel)
TARGET_WIDTH = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

# --- XENO-CANTO FILTER PARAMETER ---
# Welche Arten wollen wir? (Gib hier einen gültigen Xeno-Canto Ort-Tag ein)
# Beispiele: "cnt:germany" (Deutschland), "area:europe" (Ganz Europa)
XC_SPECIES_LOCATION = "cnt:germany"  

# Woher dürfen die Audio-Aufnahmen dieser Arten stammen?
# Beispiele: "area:europe" (Europa), "cnt:germany" (Nur Deutschland)
XC_RECORDING_LOCATION = "area:europe"    

# Welche Qualitäten lassen wir zu? (A=Studio, E=Viel Rauschen)
XC_QUALITIES =["A", "B", "C", "D"] 

XC_MIN_LENGTH = 5  # in Sekunden
XC_MAX_LENGTH = 60 # in Sekunden

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
