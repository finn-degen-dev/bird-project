import os
import torch

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
# We will use "processed" as the standard name
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
RAW_AUDIO_DIR = os.path.join(DATA_DIR, "raw_audio")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
MODELS_DIR = os.path.join(DATA_DIR, "models")

# Audio Settings
SAMPLE_RATE = 32000
DURATION = 5  
N_MELS = 128
FMAX = 16000
HOP_LENGTH = 512
# This calculates how wide the spectrogram picture should be
TARGET_WIDTH = (SAMPLE_RATE * DURATION) // HOP_LENGTH

# AI Settings
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Xeno-Canto Settings
API_KEY = os.getenv("XENOCANTO_API_KEY", "") 
USER_AGENT = "BirdCLEF-Student-Project/1.0"