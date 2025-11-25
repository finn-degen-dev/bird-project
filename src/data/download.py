import pandas as pd
import requests
import os
import sys
import concurrent.futures
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def download_file(url, filename):
    filepath = os.path.join(config.RAW_AUDIO_DIR, filename)
    
    # Skip if exists
    if os.path.exists(filepath):
        return 0 # Skipped
        
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(r.content)
            return 1 # Downloaded
    except:
        pass
    return -1 # Error

def main():
    if not os.path.exists(config.METADATA_FILE):
        print("❌ No metadata.csv found.")
        return

    print(f"📂 Reading {config.METADATA_FILE}...")
    df = pd.read_csv(config.METADATA_FILE)
    
    tasks = []
    print(f"🔎 Checking {len(df)} files...")
    
    # Prepare list of missing files
    for _, row in df.iterrows():
        filepath = os.path.join(config.RAW_AUDIO_DIR, row['filename'])
        if not os.path.exists(filepath):
            tasks.append((row['url'], row['filename']))

    if not tasks:
        print("✅ All files are already downloaded!")
        return

    print(f"⬇️ Downloading {len(tasks)} missing files...")
    
    # Parallel Download
    downloaded = 0
    errors = 0
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(download_file, url, fname) for url, fname in tasks]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(tasks)):
            res = future.result()
            if res == 1: downloaded += 1
            elif res == -1: errors += 1

    print(f"\n✨ Sync Complete. Downloaded: {downloaded}. Errors: {errors}.")

if __name__ == "__main__":
    main()
