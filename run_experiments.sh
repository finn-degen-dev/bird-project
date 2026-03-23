#!/usr/bin/env bash
# Stoppt das Skript sofort, falls ein Fehler auftritt
set -e

echo "============================================================"
echo " 🐦 Image Size Experiment (MobileNetV3 Speed vs. Accuracy)"
echo "============================================================"
echo "Wir testen, wie klein wir die Spektrogramme skalieren können,"
echo "bevor die KI die Vögel nicht mehr erkennt."
echo "============================================================"

# 1. Standard Auflösung (224x224)
echo -e "\n>>> [1/4] Trainiere MobileNetV3 (Auflösung: 224x224) ..."
python src/train.py --model mobilenet --in-channels 3 --img-size 224

# 2. Mittlere Auflösung (160x160)
echo -e "\n>>> [2/4] Trainiere MobileNetV3 (Auflösung: 160x160) ..."
python src/train.py --model mobilenet --in-channels 3 --img-size 160

# 3. Kleine Auflösung (128x128)
echo -e "\n>>> [3/4] Trainiere MobileNetV3 (Auflösung: 128x128) ..."
python src/train.py --model mobilenet --in-channels 3 --img-size 128

# 4. Winzige Auflösung (96x96)
echo -e "\n>>> [4/4] Trainiere MobileNetV3 (Auflösung: 96x96) ..."
python src/train.py --model mobilenet --in-channels 3 --img-size 96

echo -e "\n============================================================"
echo " 📊 Starte Benchmark (Latenz und Accuracy messen)..."
echo "============================================================"

python src/experiments/benchmark_imgsize.py

echo -e "\n🎉 Alles fertig! Schau dir den fertigen Graphen unter 'visuals/experiments/imgsize_speed_comparison.png' an!"
