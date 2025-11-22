import os
import asyncio
import aiohttp
import shutil

# --- KONFIGURATION ---
DATA_DIR = "/app/data/raw"
MAX_FILES = 200  # Tripled the amount of data!

API_KEY = os.getenv("XENOCANTO_API_KEY")
if not API_KEY:
    print("❌ FEHLER: Kein API-Key gefunden!")
    exit(1)

API_VERSION = "3"

birds = {
    "Turdus merula": "Amsel",
    "Parus major": "Kohlmeise",
    "Erithacus rubecula": "Rotkehlchen",
    "Fringilla coelebs": "Buchfink",
    "Passer domesticus": "Haussperling"
}

async def download_file(session, url, filepath):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = await response.content.read(4096)
                        if not chunk: break
                        f.write(chunk)
                return True
            return False
    except Exception as e:
        print(f"⚠️ Fehler: {e}")
        return False

async def process_bird(session, latin_name, german_name):
    print(f"\n--- Suche nach: {german_name} ({latin_name}) ---")
    
    target_folder = os.path.join(DATA_DIR, german_name)
    os.makedirs(target_folder, exist_ok=True)
    
    # Check existing files
    existing = len([f for f in os.listdir(target_folder) if f.endswith('.mp3')])
    if existing >= MAX_FILES:
        print(f"✅ Genug Daten vorhanden ({existing}).")
        return

    # --- STRATEGIE ÄNDERUNG ---
    # Wir entfernen "q:A" (Quality A).
    # Das gibt uns alle Qualitäten (A, B, C, D, E).
    # Die API sortiert standardmäßig nicht nach Qualität, das müssen wir beachten.
    query = f'sp:"{latin_name}" cnt:germany len:10-60 type:song'
    
    api_url = f"https://xeno-canto.org/api/{API_VERSION}/recordings"
    params = {'query': query, 'key': API_KEY}

    try:
        async with session.get(api_url, params=params) as response:
            data = await response.json()
            recordings = data.get('recordings', [])
            print(f"-> {len(recordings)} Aufnahmen gefunden.")
    except Exception as e:
        print(f"❌ API Fehler: {e}")
        return

    # --- SORTIERUNG ---
    # Wir sortieren die Liste manuell: 'q' ist die Qualität.
    # A > B > C > D > E.
    # Wir wollen erst die guten, dann die schlechten, bis MAX_FILES voll ist.
    quality_order = {'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'no score': 6}
    
    recordings.sort(key=lambda x: quality_order.get(x.get('q', 'no score'), 6))

    count = existing
    for rec in recordings:
        if count >= MAX_FILES:
            break
            
        file_url = rec.get('file')
        # Wir speichern die Qualität im Dateinamen! 
        # z.B. "XC12345_qA.mp3". Das hilft uns später beim Filtern, falls nötig.
        quality = rec.get('q', 'N')
        file_name = f"{rec.get('id')}_q{quality}.mp3"
        file_path = os.path.join(target_folder, file_name)

        if not os.path.exists(file_path):
            if await download_file(session, file_url, file_path):
                count += 1
                if count % 10 == 0:
                    print(f"   ... {count}/{MAX_FILES} (Qualität: {quality})")
        else:
            count += 1
    
    print(f"✅ {german_name}: {count} Dateien fertig.")

async def main():
    async with aiohttp.ClientSession() as session:
        for latin, german in birds.items():
            await process_bird(session, latin, german)

if __name__ == "__main__":
    asyncio.run(main())
