import os
import shutil
import random
import argparse
from glob import glob
from tqdm import tqdm

# --- Configuration ---
DEFAULT_SOURCE = "data/raw"
DEFAULT_DEST = "data/dataset"
SPLIT_RATIO = 0.8  # 80% Train, 20% Test
SEED = 42          # Ensures the split is the same every time we run it

def split_dataset(source_dir, dest_dir, split_ratio):
    # Set seed for reproducibility
    random.seed(SEED)
    
    train_dir = os.path.join(dest_dir, "train")
    test_dir = os.path.join(dest_dir, "test")

    # Clean up previous split if it exists to avoid duplicates
    if os.path.exists(dest_dir):
        print(f"Warning: '{dest_dir}' already exists. Merging/Overwriting...")
    
    # Get list of species (subdirectories in data/raw)
    species_list = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"Found {len(species_list)} species categories.")
    print(f"Splitting data into Training ({split_ratio*100:.0f}%) and Testing ({(1-split_ratio)*100:.0f}%)...")

    for species in species_list:
        species_source_path = os.path.join(source_dir, species)
        
        # Find all audio files for this species
        # (Checking for common audio formats)
        extensions = ['*.mp3', '*.wav', '*.ogg']
        files = []
        for ext in extensions:
            files.extend(glob(os.path.join(species_source_path, ext)))
            
        # Shuffle files randomly
        random.shuffle(files)
        
        # Calculate split index
        split_idx = int(len(files) * split_ratio)
        
        train_files = files[:split_idx]
        test_files = files[split_idx:]
        
        # Create species directories in train/test
        os.makedirs(os.path.join(train_dir, species), exist_ok=True)
        os.makedirs(os.path.join(test_dir, species), exist_ok=True)
        
        # Copy files (Use copy instead of move to keep raw data safe)
        print(f"Processing {species}: {len(train_files)} Train, {len(test_files)} Test")
        
        for f in train_files:
            shutil.copy2(f, os.path.join(train_dir, species, os.path.basename(f)))
            
        for f in test_files:
            shutil.copy2(f, os.path.join(test_dir, species, os.path.basename(f)))

    print("\n✅ Data split successfully!")
    print(f"Train data: {train_dir}")
    print(f"Test data:  {test_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split raw audio data into train and test sets.")
    parser.add_argument("--input", default=DEFAULT_SOURCE, help="Input raw data folder")
    parser.add_argument("--output", default=DEFAULT_DEST, help="Output folder for split data")
    parser.add_argument("--ratio", type=float, default=SPLIT_RATIO, help="Ratio for training data (default 0.8)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' not found.")
    else:
        split_dataset(args.input, args.output, args.ratio)
