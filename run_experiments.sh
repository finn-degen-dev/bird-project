# 1. Hauptmodell (Sehr gründlich trainiert)
python src/train.py --model efficientnet --epochs 80

# 2. ResNet Vergleich
python src/train.py --model resnet --epochs 50

# 3. Custom CNN (Vergleich für die Theorie-Sektion)
python src/train.py --model custom --cnn-layers 5 --epochs 50

# 4. Song vs Call (Für die t-SNE Grafiken)
python src/train.py --model efficientnet --task type --epochs 40

# 5. Embedding / ArcFace (Dein "Profi-Feature")
python src/train.py --model embedding --epochs 60
