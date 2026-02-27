import os
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")
RAW_AUDIO_DIR = os.path.join(DATA_DIR, "raw_audio")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed_tensors")
MODELS_DIR = os.path.join(DATA_DIR, "models")
VISUALS_DIR = os.path.join(BASE_DIR, "visuals", "experiments")

SAMPLE_RATE = 32000
DURATION = 5
N_MELS = 128
FMAX = 16000
IMG_SIZE = 224
HOP_LENGTH = 512

TARGET_WIDTH = int((SAMPLE_RATE * DURATION) / HOP_LENGTH) + 1

XC_SPECIES_LOCATION = "cnt:germany"

XC_RECORDING_LOCATION = "area:europe"

XC_QUALITIES =["A", "B", "C", "D"]

XC_MIN_LENGTH = 5
XC_MAX_LENGTH = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(VISUALS_DIR, exist_ok=True)
