import pandas as pd
import requests
import os
import time
import concurrent.futures
from tqdm import tqdm
import torch
import librosa
import numpy as np
import warnings
import threading

warnings.filterwarnings("ignore")

from config import *

API_KEY = os.getenv("XENOCANTO_API_KEY", "")

def fetch_page(url, params, page):
    params['page'] = page
    for attempt in range(3):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json().get('recordings',[])
            elif r.status_code in [429, 503]:
                time.sleep(5)
            else:
                break
        except:
            time.sleep(2)
    return[]

def step1_get_metadata():
    print(f" Schritt 1: Erfasse Metadaten (Vögel aus {XC_SPECIES_LOCATION}, Audio aus {XC_RECORDING_LOCATION})...")

    if os.path.exists(METADATA_FILE):
        print(" Metadaten existieren bereits. (Lösche die metadata.csv, falls du Filter geändert hast!)")
        return pd.read_csv(METADATA_FILE)

    url = "https://xeno-canto.org/api/3/recordings"

    params_species = {'query': XC_SPECIES_LOCATION, 'key': API_KEY, 'per_page': 500}
    try:
        r = requests.get(url, params=params_species, timeout=15)
        r.raise_for_status()
        num_pages = r.json()['numPages']
    except Exception as e:
        print(f" API Error bei Xeno-Canto (Schritt 1): {e}")
        return None

    target_species = set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures =[executor.submit(fetch_page, url, params_species.copy(), p) for p in range(1, num_pages + 1)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=num_pages, desc="Scanne Arten"):
            for rec in future.result():
                if 'en' in rec and rec['en']:
                    target_species.add(rec['en'])

    print(f" Gefundene Ziel-Arten: {len(target_species)}")

    valid_recordings =[]

    def fetch_species_recordings(species):
        query_str = f'en:"{species}" {XC_RECORDING_LOCATION} len:{XC_MIN_LENGTH}-{XC_MAX_LENGTH}'
        params_rec = {'query': query_str, 'key': API_KEY, 'per_page': 500}

        recs =[]
        for attempt in range(3):
            try:
                r = requests.get(url, params=params_rec, timeout=15)
                if r.status_code == 200:
                    data = r.json()
                    pages = data.get('numPages', 0)
                    recs = data.get('recordings',[])

                    if pages > 1:
                        for p in range(2, min(pages + 1, 4)):
                            time.sleep(1)
                            r_page = requests.get(url, params={**params_rec, 'page': p}, timeout=15)
                            if r_page.status_code == 200:
                                recs.extend(r_page.json().get('recordings',[]))
                    break
                elif r.status_code in [429, 503]:
                    time.sleep(5)
            except:
                time.sleep(2)
        return recs

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(fetch_species_recordings, sp): sp for sp in target_species}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(target_species), desc="Lade Datensätze (Art für Art)"):
            time.sleep(0.5)
            for rec in future.result():
                if rec.get('q') in XC_QUALITIES:
                    valid_recordings.append({
                        'id': rec['id'], 'species': rec['en'], 'scientific': f"{rec['gen']} {rec['sp']}",
                        'sex': rec['sex'], 'type': rec['type'], 'quality': rec['q'],
                        'url': rec['file'], 'length': rec['length'], 'filename': f"XC{rec['id']}.mp3"
                    })

    df = pd.DataFrame(valid_recordings)

    if len(df) == 0:
        print(" Keine Aufnahmen gefunden, die auf die Filter passen! (Wahrscheinlich API Rate-Limit)")
        return None

    counts = df['species'].value_counts()
    valid_counts = counts[counts >= 10].index
    df = df[df['species'].isin(valid_counts)]

    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(METADATA_FILE, index=False)
    print(f" {len(df)} Aufnahmen in metadata.csv gespeichert (Qualitäten: {XC_QUALITIES}).")
    return df

def process_audio_pipeline(df):
    print(f"\n Schritt 2: Intelligente Download & Umwandlungs-Pipeline (Fenster {DURATION}s)...")
    os.makedirs(RAW_AUDIO_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    tasks =[]
    for _, row in df.iterrows():
        mp3_path = os.path.join(RAW_AUDIO_DIR, row['filename'])
        pt_path = os.path.join(PROCESSED_DIR, row['filename'].replace('.mp3', '.pt'))
        tasks.append((row['url'], mp3_path, pt_path))

    thread_local = threading.local()

    def get_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = requests.Session()
        return thread_local.session

    def pipeline_worker(task):
        url, mp3_path, pt_path = task

        if os.path.exists(pt_path):

            if os.path.exists(mp3_path):
                try: os.remove(mp3_path)
                except: pass
            return

        if not os.path.exists(mp3_path):
            session = get_session()
            try:
                with session.get(url, timeout=10, stream=True) as r:
                    if r.status_code == 200:
                        with open(mp3_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                f.write(chunk)
                    else:
                        return
            except:
                return

        if os.path.exists(mp3_path):
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
            except Exception as e:

                pass
            finally:

                if os.path.exists(mp3_path):
                    try: os.remove(mp3_path)
                    except: pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        list(tqdm(executor.map(pipeline_worker, tasks), total=len(tasks), desc="Verarbeite Audios"))

def main():
    df = step1_get_metadata()
    if df is not None:
        process_audio_pipeline(df)
    print("\n Datenpipeline abgeschlossen. Rohe Daten wurden automatisch aufgeräumt!")
    print(" Du kannst jetzt das Training starten!")

if __name__ == "__main__":
    main()
