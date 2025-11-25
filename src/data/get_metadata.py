import pandas as pd
import requests
import os
import sys
import concurrent.futures
from tqdm import tqdm

# Import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def fetch_page(url, params, page):
    """Helper to fetch one API page"""
    params['page'] = page
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get('recordings', [])
    except:
        pass
    return []

def get_german_species_list():
    """Step 1: Get the list of ALL birds found in Germany"""
    print("🇩🇪  Step 1: Mapping German Biodiversity...")
    url = "https://xeno-canto.org/api/3/recordings"
    # q:A ensures we only count species that have at least some good data
    params = {'query': 'cnt:germany', 'key': config.API_KEY, 'per_page': 500}
    
    # Get page count
    try:
        r = requests.get(url, params=params)
        num_pages = r.json()['numPages']
    except Exception as e:
        print(f"❌ API Error: {e}")
        return set()

    species_set = set()
    
    # Parallel Fetch
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_page, url, params.copy(), p) for p in range(1, num_pages + 1)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_pages, desc="   Scanning Germany"):
            for rec in future.result():
                species_set.add(rec['en'])

    print(f"✅ Found {len(species_set)} relevant species in Germany.")
    return species_set

def get_european_recordings(relevant_species):
    """Step 2: Get ALL European recordings, but ONLY for the relevant species"""
    print("\n🇪🇺  Step 2: Fetching European Data for these species...")
    url = "https://xeno-canto.org/api/3/recordings"
    # q:A is essential for training data
    query = 'area:europe grp:birds q:A len:5-60' 
    params = {'query': query, 'key': config.API_KEY, 'per_page': 500}
    
    try:
        r = requests.get(url, params=params)
        data = r.json()
        num_pages = data['numPages']
        total_found = data['numRecordings']
        print(f"   Xeno-Canto has {total_found} European recordings ({num_pages} pages).")
    except:
        return []

    valid_recordings = []
    
    # Parallel Fetch
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_page, url, params.copy(), p) for p in range(1, num_pages + 1)]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_pages, desc="   Fetching Metadata"):
            recordings = future.result()
            for rec in recordings:
                # --- THE FILTER ---
                if rec['en'] in relevant_species:
                    valid_recordings.append({
                        'id': rec['id'],
                        'species': rec['en'],
                        'scientific': f"{rec['gen']} {rec['sp']}",
                        'sex': rec['sex'],
                        'type': rec['type'],
                        'quality': rec['q'],
                        'url': rec['file'],
                        'length': rec['length'],
                        'filename': f"XC{rec['id']}.mp3"
                    })

    return valid_recordings

def main():
    # 1. Get the Whitelist
    german_species = get_german_species_list()
    if not german_species:
        return

    # 2. Get the Data
    recordings = get_european_recordings(german_species)
    
    # 3. Save
    print(f"\n💾 Saving {len(recordings)} recordings to {config.METADATA_FILE}...")
    df = pd.DataFrame(recordings)
    
    # Optional: Basic balancing (remove species with < 5 samples)
    counts = df['species'].value_counts()
    valid_counts = counts[counts >= 5].index
    df = df[df['species'].isin(valid_counts)]
    
    df.to_csv(config.METADATA_FILE, index=False)
    print(f"✅ Done! Final Dataset: {len(df)} recordings from {df['species'].nunique()} species.")

if __name__ == "__main__":
    main()
