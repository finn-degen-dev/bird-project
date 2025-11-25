import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_AUDIO_DIR = os.path.join(DATA_DIR, "raw_audio")
METADATA_FILE = os.path.join(DATA_DIR, "metadata.csv")

# Audio Settings
SAMPLE_RATE = 32000
DURATION = 5  # Seconds (chunk size)

# Xeno-Canto Settings
API_KEY = os.getenv("XENOCANTO_API_KEY", "") # Falls back to empty if not set
USER_AGENT = "BirdCLEF-Student-Project/1.0"
