import os
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_AUDIO_DIR = os.path.join(DATA_DIR, "raw_audio")
MODELS_DIR = os.path.join(DATA_DIR, "models")

SAMPLE_RATE = 32000
DURATION = 5  # Sekunden pro Chunk
N_MELS = 128
FMAX = 16000
IMG_SIZE = 224

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES =['Amsel', 'Buchfink', 'Haussperling', 'Kohlmeise', 'Rotkehlchen'] # Oder laden aus metadata
