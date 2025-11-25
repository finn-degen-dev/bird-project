import sys
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path so we can import dataset and config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset import SmartBirdDataset
import config

def test_loading(filter_sex=None, filter_type=None, task='species'):
    print(f"\n--- Testing Dataset [Sex={filter_sex}, Type={filter_type}, Task={task}] ---")
    
    try:
        # Initialize Dataset
        ds = SmartBirdDataset(mode='train', filter_sex=filter_sex, filter_type=filter_type, task=task)
        
        # 1. Check Basics
        print(f"✅ Loaded {len(ds)} samples.")
        print(f"✅ Found {len(ds.classes)} classes.")
        print(f"   Sample classes: {ds.classes[:5]}")
        
        # 2. Check a random item
        idx = np.random.randint(0, len(ds))
        tensor, label = ds[idx]
        
        # 3. Check Shape & Type
        print(f"✅ Item {idx}:")
        print(f"   - Tensor Shape: {tensor.shape} (Should be [3, 224, 224])")
        print(f"   - Label Index: {label}")
        print(f"   - Label Name: {ds.classes[label]}")
        print(f"   - Min/Max Values: {tensor.min():.2f} / {tensor.max():.2f} (Should be approx 0.0 - 1.0)")
        
        return ds, tensor

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None

def plot_sample(tensor, title="Spectrogram"):
    """Visualizes the tensor to make sure it looks like a spectrogram"""
    # Tensor is (3, 224, 224). We take the first channel for plotting.
    img = tensor[0].numpy()
    
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='magma', origin='lower')
    plt.title(title)
    plt.axis('off')
    
    # Save to a temp file instead of showing (better for Docker/Headless)
    out_path = "data/test_spectrogram_preview.png"
    plt.savefig(out_path)
    print(f"🖼️  Saved preview image to {out_path}")
    plt.close()

if __name__ == "__main__":
    # Test 1: Standard load (All data)
    ds, tensor = test_loading()
    
    if ds and tensor is not None:
        plot_sample(tensor, title="Standard Load")

    # Test 2: Filter for Males only
    test_loading(filter_sex='male')

    # Test 3: Filter for Songs only
    test_loading(filter_type='song')
    
    # Test 4: Task = Type (Classify Song vs Call)
    test_loading(task='type')
