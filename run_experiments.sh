#!/usr/bin/env bash
# Stoppt das Skript sofort, falls ein Fehler auftritt
set -e 

echo "============================================================"
echo " 🐦 Piepmatz Polizei - The Big European Overnight Training"
echo "============================================================"

# --- 1. TRAINING ALLER MODELLE ---
echo -e "\n>>> 1/8: Training EfficientNet (Hauptmodell) ..."
python src/train.py --model efficientnet --epochs 80

echo -e "\n>>> 2/8: Training ResNet (Vergleichsmodell) ..."
python src/train.py --model resnet --epochs 50

echo -e "\n>>> 3/8: Training Custom CNN (2 Layers) ..."
python src/train.py --model custom --cnn-layers 2 --epochs 50

echo -e "\n>>> 4/8: Training Custom CNN (3 Layers) ..."
python src/train.py --model custom --cnn-layers 3 --epochs 50

echo -e "\n>>> 5/8: Training Custom CNN (5 Layers) ..."
python src/train.py --model custom --cnn-layers 5 --epochs 50

echo -e "\n>>> 6/8: Training Custom CNN (3 Layers + Attention) ..."
python src/train.py --model custom --cnn-layers 3 --use-attention --epochs 50

echo -e "\n>>> 7/8: Training Song vs. Call Classifier ..."
python src/train.py --model efficientnet --task type --epochs 40

echo -e "\n>>> 8/8: Training Metric Learning (ArcFace Embedding) ..."
python src/train.py --model embedding --epochs 60


# --- 2. EVALUATION & CONFUSION MATRIZEN ---
echo -e "\n============================================================"
echo " 📊 Starte Evaluation (Test-Set) & Cache Generierung..."
echo "============================================================"

python src/evaluate.py --model efficientnet
python src/evaluate.py --model resnet
python src/evaluate.py --model custom --cnn-layers 2
python src/evaluate.py --model custom --cnn-layers 3
python src/evaluate.py --model custom --cnn-layers 5
python src/evaluate.py --model custom --cnn-layers 3 --use-attention


# --- 3. GRAFIKEN UND EXPERIMENTE NEU GENERIEREN ---
echo -e "\n============================================================"
echo " 🎨 Generiere finale Grafiken für das JuFo Poster..."
echo "============================================================"

# Erstellt Performance-, Accuracy- und Quality-Graphen
python src/visuals/generate_figures.py

# Generiert den t-SNE Plot (Song vs Call)
python src/experiments/song_vs_call.py

# Optional: GradCAM (Achtung: Prüfe ob "Amsel/beispiel.mp3" in src/experiments/gradcam.py existiert!)
if[ -f "data/raw_audio/Amsel/beispiel.mp3" ]; then
    python src/experiments/gradcam.py
else
    echo "Überspringe GradCAM (Beispiel-MP3 fehlt in data/raw_audio/Amsel/beispiel.mp3)"
fi

echo -e "\n🎉 ALLES FERTIG! Gute Nacht und viel Erfolg mit den neuen Resultaten! 🚀"
