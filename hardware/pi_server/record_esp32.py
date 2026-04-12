import socket
import wave
import numpy as np
import scipy.signal

HOST = '0.0.0.0'
PORT = 5000
SAMPLE_RATE = 32000
DURATION = 10 
CHANNELS = 1
GAIN = 30.0  

# Bandpass Filter: Lässt nur Töne zwischen 1kHz und 12kHz durch (Vögel!)
# Schneidet Wind (unter 1kHz) und WLAN-Fiepen (über 12kHz) ab.
b, a = scipy.signal.butter(4, [1000, 12000], btype='bandpass', fs=SAMPLE_RATE)

def record_audio():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    
    print(f"[*] Warte auf ESP32 auf Port {PORT}...")
    conn, addr = server.accept()
    print(f"[+] ESP32 verbunden! Nehme {DURATION} Sekunden auf...")

    # Mono: 32000 Samples * 1 Kanal * 2 Bytes = 64.000 Bytes pro Sekunde
    BYTES_PER_SECOND = SAMPLE_RATE * 1 * 2 
    bytes_to_read = BYTES_PER_SECOND * DURATION  
    audio_data = bytearray()

    while len(audio_data) < bytes_to_read:
        chunk = conn.recv(4096)
        if not chunk: break
        audio_data.extend(chunk)

    conn.close()
    server.close()
    print("[*] Verarbeite Audiodaten (Filter & Gain)...")
    
    # 1. Bytes einlesen (es ist jetzt sauberes Mono-Audio vom ESP32)
    mono_audio = np.frombuffer(bytes(audio_data[:bytes_to_read]), dtype=np.int16)
    
    # 2. In Float umwandeln und verstärken
    audio_float = (mono_audio.astype(np.float32) / 32768.0) * GAIN
    
    # 3. Rauschen wegfiltern (Bandpass)
    audio_float = scipy.signal.filtfilt(b, a, audio_float)
    
    # 4. Clipping verhindern und zurück in 16-Bit wandeln
    audio_float = np.clip(audio_float, -1.0, 1.0)
    audio_final = (audio_float * 32767.0).astype(np.int16)

    with wave.open("test_recording.wav", "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_final.tobytes())
        
    print("[+] Fertig! Hör dir test_recording.wav an. Pitch und Rauschen sollten behoben sein!")

if __name__ == "__main__":
    record_audio()
