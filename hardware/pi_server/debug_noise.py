import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Load the raw audio file
# Make sure you've run record_esp32.py first to generate test_recording.wav
audio_path = "test_recording.wav"
print(f"Loading {audio_path}...")

# sr=None preserves the original sample rate (32000)
y, sr = librosa.load(audio_path, sr=None)

# Create the plot
plt.figure(figsize=(12, 6))

# Calculate the spectrogram
D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

# Display it with a linear frequency axis (best for hardware debugging)
librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='linear', cmap='magma')

plt.colorbar(format='%+2.0f dB')
plt.title('Raw ESP32/INMP441 Spectrogram (No Bandpass)')
plt.tight_layout()

# Save it as an image
plt.savefig('noise_debug.png', dpi=300)
print("Done! Please upload 'noise_debug.png' to the chat.")
