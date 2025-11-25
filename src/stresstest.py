import sys
import os
import time
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import SmartBirdDataset

def main():
    print("🔥 SHM STRESS TEST STARTED")
    
    # 1. Initialize Dataset
    try:
        ds = SmartBirdDataset(mode='train', task='species')
    except Exception as e:
        print(f"Dataset error: {e}")
        return

    # 2. Initialize DataLoader with MULTIPLE WORKERS
    # We use 8 workers to really push the memory limits
    workers = 8
    batch_size = 64
    print(f"Testing with num_workers={workers}, batch_size={batch_size}...")
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=workers)
    
    # 3. Fast Iteration Loop
    start = time.time()
    count = 0
    
    try:
        # We iterate through the whole dataset but do nothing with the data
        # This isolates the data loading / shared memory aspect
        for _ in tqdm(loader, desc="Stressing Shared Memory"):
            count += 1
            
        duration = time.time() - start
        print(f"\n✅ SUCCESS! Iterated {count} batches in {duration:.2f}s.")
        print("Your 16GB SHM is working perfectly. You can use num_workers=4 safely.")
        
    except RuntimeError as e:
        print(f"\n❌ CRASHED: {e}")
        print("This usually means Shared Memory is still too low.")

if __name__ == "__main__":
    main()
