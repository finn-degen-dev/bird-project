# Wir starten mit einem schlanken Python 3.10
#FROM python:3.10-slim (cpu)
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

ARG DEBIAN_FRONTEND=noninteractive

# 1. System-Updates und Audio-Bibliotheken installieren
# (Ohne libsndfile1 stürzt librosa sofort ab!)
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Arbeitsverzeichnis im Container setzen
WORKDIR /app

# 3. Requirements kopieren und installieren
# (Wir machen das VOR dem Code-Kopieren, damit Docker cachen kann)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Standard-Befehl (hält den Container am Leben)
CMD ["bash"]
