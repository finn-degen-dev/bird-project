import pandas as pd
import requests
import os
import concurrent.futures
from tqdm import tqdm
import torch
import librosa
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from config import *

API_KEY = os.getenv("XENOCANTO_API_KEY", "")

def fetch_page(url, params, page):
    params['page'] = page
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return r.json().get('recordings',[])
    except:
        pass
    return[]

def step1_get_metadata():
    print(f" Schritt 1/3: Erfasse Metadaten (Vögel aus {XC_SPECIES_COUNTRY}, Audio aus {XC_RECORDING_AREA})...")

    if os.path.exists(METADATA_FILE):
        print(" Metadaten existieren bereits. (Lösche die metadata.csv, falls du Filter geändert hast!)")
        return pd.read_csv(METADATA_FILE)

    url = "https://xeno-canto.org/api/3/recordings"

    params_species = {'query': XC_SPECIES_LOCATION, 'key': API_KEY, 'per_page': 500}
    try:
        num_pages = requests.get(url, params=params_species).json()['numPages']
    except Exception as e:
        print(f" API Error bei Xeno-Canto (Schritt 1): {e}")
        return None

    target_species = set()
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures =[executor.submit(fetch_page, url, params_species.copy(), p) for p in range(1, num_pages + 1)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_pages, desc="Scanne Arten"):
            for rec in future.result():
                target_species.add(rec['en'])

    query_str = f'{XC_RECORDING_LOCATION} grp:birds len:{XC_MIN_LENGTH}-{XC_MAX_LENGTH}'
    params_rec = {'query': query_str, 'key': API_KEY, 'per_page': 500}

    try:
        num_pages_rec = requests.get(url, params=params_rec).json()['numPages']
    except Exception as e:
        print(f" API Error bei Xeno-Canto (Schritt 2): {e}")
        return None

    valid_recordings =[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures =[executor.submit(fetch_page, url, params_rec.copy(), p) for p in range(1, num_pages_rec + 1)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_pages_rec, desc="Lade Datensätze"):
            for rec in future.result():

                if rec['en'] in target_species and rec.get('q') in XC_QUALITIES:
                    valid_recordings.append({
                        'id': rec['id'], 'species': rec['en'], 'scientific': f"{rec['gen']} {rec['sp']}",
                        'sex': rec['sex'], 'type': rec['type'], 'quality': rec['q'],
                        'url': rec['file'], 'length': rec['length'], 'filename': f"XC{rec['id']}.mp3"
                    })

    df = pd.DataFrame(valid_recordings)

    if len(df) == 0:
        print(" Keine Aufnahmen gefunden, die auf die Filter passen!")
        return None

    counts = df['species'].value_counts()
    valid_counts = counts[counts >= 10].index
    df = df[df['species'].isin(valid_counts)]

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(METADATA_FILE, index=False)
    print(f" {len(df)} Aufnahmen in metadata.csv gespeichert (Qualitäten: {XC_QUALITIES}).")
    return df

def step2_download_audio(df):
    print("\n Schritt 2/3: Lade fehlende Audio-Dateien herunter...")
    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)

    tasks =[]
    for _, row in df.iterrows():
        filepath = os.path.join(RAW_AUDIO_DIR, row['filename'])
        if not os.path.exists(filepath):
            tasks.append((row['url'], filepath))

    if not tasks:
        print(" Alle Audio-Dateien sind bereits heruntergeladen.")
        return

    def download_file(task):
        url, filepath = task
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                with open(filepath, 'wb') as f:
                    f.write(r.content)
        except:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(download_file, tasks), total=len(tasks), desc="Downloading"))

def step3_preprocess_tensors(df):
    print(f"\n Schritt 3/3: Generiere Tensoren ({DURATION}s Fenster, Hop-Length {HOP_LENGTH})...")
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    success = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Verarbeite"):
        mp3_path = os.path.join(RAW_AUDIO_DIR, row['filename'])
        pt_path = os.path.join(PROCESSED_DIR, row['filename'].replace('.mp3', '.pt'))

        if os.path.exists(pt_path) or not os.path.exists(mp3_path):
            continue

        try:

            load_duration = max(15, DURATION * 2)
            y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE, duration=load_duration)

            target_len_samples = SAMPLE_RATE * DURATION
            if len(y) < target_len_samples:
                y = np.pad(y, (0, target_len_samples - len(y)), mode='constant')

            melspec = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=N_MELS, fmax=FMAX, hop_length=HOP_LENGTH
            )
            log_melspec = librosa.power_to_db(melspec, ref=np.max)

            min_val, max_val = log_melspec.min(), log_melspec.max()
            spec_norm = (log_melspec - min_val) / (max_val - min_val + 1e-6)
            spec_norm = np.flipud(spec_norm).copy()

            tensor = torch.tensor(spec_norm, dtype=torch.float16)
            torch.save(tensor, pt_path)
            success += 1
        except Exception as e:
            pass

    print(f" Fertig. {success} neue Tensoren generiert.")

def main():
    df = step1_get_metadata()
    if df is not None:
        step2_download_audio(df)
        step3_preprocess_tensors(df)
    print("\n Datenpipeline abgeschlossen. Du kannst jetzt das Training starten!")

if __name__ == "__main__":
    main()
