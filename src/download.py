import os
import asyncio
import aiohttp
import shutil

# --- KONFIGURATION ---
DATA_DIR = "/app/data/raw"
MAX_FILES = 60

# Wir holen den Key aus der Umgebungsvariable
API_KEY = os.getenv("XENOCANTO_API_KEY")

# Prüfung: Ist der Key da?
if not API_KEY:
    print("❌ FEHLER: Kein API-Key gefunden!")
    print("   Bitte setze 'XENOCANTO_API_KEY' in deiner .env Datei")
    print("   und starte den Container neu.")
    exit(1)

API_VERSION = "3"  # Jetzt nutzen wir explizit v3

# Unsere Vögel
birds = {
    "Turdus merula": "Amsel",
    "Parus major": "Kohlmeise",
    "Erithacus rubecula": "Rotkehlchen",
    "Fringilla coelebs": "Buchfink",
    "Passer domesticus": "Haussperling"
}

async def download_file(session, url, filepath):
    # ... (Code bleibt gleich wie vorher) ...
    try:
        async with session.get(url) as response:
            if response.status == 200:
                with open(filepath, 'wb') as f:
                    while True:
                        chunk = await response.content.read(4096)
                        if not chunk:
                            break
                        f.write(chunk)
                return True
            else:
                print(f"❌ Download fehlgeschlagen (Status {response.status}): {url}")
                return False
    except Exception as e:
        print(f"⚠️ Fehler beim Download von {url}: {e}")
        return False

async def process_bird(session, latin_name, german_name):
    print(f"\n--- Suche nach: {german_name} ({latin_name}) ---")
    
    target_folder = os.path.join(DATA_DIR, german_name)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        
    existing_files = len([f for f in os.listdir(target_folder) if f.endswith('.mp3')])
    if existing_files >= MAX_FILES:
        print(f"✅ Bereits {existing_files} Dateien vorhanden. Überspringe...")
        return

    # --- Query Aufbauen ---
    query = f'sp:"{latin_name}" cnt:germany q:A len:10-60 type:song'
    
    # API v3 URL
    api_url = f"https://xeno-canto.org/api/{API_VERSION}/recordings"
    
    # Parameter für v3 (Query + Key sind jetzt separate Felder, nicht alles im Query-String)
    params = {
        'query': query,
        'key': API_KEY  # Hier wird der Key sicher übergeben
    }

    try:
        async with session.get(api_url, params=params) as response:
            if response.status != 200:
                print(f"❌ API Fehler für {german_name}: Status {response.status}")
                error_text = await response.text()
                print(f"   Server Antwort: {error_text}")
                return
            
            data = await response.json()
            recordings = data.get('recordings', [])
            print(f"-> {len(recordings)} Aufnahmen gefunden. Starte Download...")

            if not recordings:
                print("   ⚠️ Keine Aufnahmen gefunden. Überprüfe die Suchparameter.")
                return

    except Exception as e:
        print(f"❌ Verbindungsfehler zur API: {e}")
        return

    # --- Downloads ---
    count = existing_files
    for rec in recordings:
        if count >= MAX_FILES:
            break
            
        file_url = rec.get('file')
        file_name = f"{rec.get('id')}.mp3"
        file_path = os.path.join(target_folder, file_name)

        if not os.path.exists(file_path):
            success = await download_file(session, file_url, file_path)
            if success:
                count += 1
                if count % 5 == 0:
                    print(f"   ... {count}/{MAX_FILES} geladen")
        else:
            count += 1
    
    print(f"✅ {german_name}: {count} Dateien bereit.")

async def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    async with aiohttp.ClientSession() as session:
        for latin, german in birds.items():
            await process_bird(session, latin, german)

    print("\n\n🎉 Alle Downloads abgeschlossen!")
    print(f"Daten liegen in: {DATA_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
