#!/usr/bin/env bash
# =============================================================================
# run_architecture_experiment.sh
#
# Trainiert alle neuen Modell-Varianten für das Architektur-Experiment und
# führt anschließend den Benchmark (Inferenzzeit & Parametergröße) aus.
#
# Neue Modelle (bereits vorhanden, werden übersprungen):
#   - efficientnet  3-Kanal  (bereits trainiert als Baseline)
#
# Neue Modelle (werden hier trainiert):
#   - efficientnet  1-Kanal
#   - mobilenet     3-Kanal
#   - mobilenet     1-Kanal
#
# Verwendung:
#   chmod +x run_architecture_experiment.sh
#   ./run_architecture_experiment.sh
#
# Optionale Umgebungsvariablen:
#   EPOCHS      Anzahl Trainings-Epochen (Standard: 30)
#   BATCH_SIZE  Batch-Größe              (Standard: 32)
# =============================================================================

set -euo pipefail

# --- Konfiguration -----------------------------------------------------------
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)/src"
MODELS_DIR="$(cd "$(dirname "$0")" && pwd)/data/models"

# Farben für schönere Ausgabe
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_section() {
  echo ""
  echo -e "${CYAN}============================================================${NC}"
  echo -e "${CYAN}  $1${NC}"
  echo -e "${CYAN}============================================================${NC}"
}

log_skip() {
  echo -e "${YELLOW}  [SKIP]${NC} $1 – bereits trainiert, überspringe."
}

log_train() {
  echo -e "${GREEN}  [TRAIN]${NC} Starte Training: $1"
}

log_done() {
  echo -e "${GREEN}  [OK]${NC} $1"
}

log_error() {
  echo -e "${RED}  [FEHLER]${NC} $1"
}

# --- Hilfsfunktion: Prüft ob ein Modell bereits existiert -------------------
model_exists() {
  local exp_name="$1"
  [ -f "${MODELS_DIR}/${exp_name}_best.pth" ] &&
    [ -f "${MODELS_DIR}/${exp_name}_classes.json" ]
}

# --- Prüfe ob src/ und train.py vorhanden ------------------------------------
if [ ! -f "${SRC_DIR}/train.py" ]; then
  log_error "src/train.py nicht gefunden! Bitte das Skript aus dem Projekt-Root ausführen."
  exit 1
fi

# =============================================================================
# SCHRITT 1: Training der neuen Modell-Varianten
# =============================================================================
log_section "SCHRITT 1/2 – Training der Architektur-Varianten"
echo "  Epochen:    ${EPOCHS}"
echo "  Batch-Size: ${BATCH_SIZE}"

# --- 1a) EfficientNet 3-Kanal (Baseline – bereits vorhanden) ----------------
EXP_EFF_3CH="efficientnet_task-species_ch-3_attn-False_layers-3"
if model_exists "${EXP_EFF_3CH}"; then
  log_skip "EfficientNet 3-Kanal (${EXP_EFF_3CH})"
else
  log_train "EfficientNet 3-Kanal"
  python "${SRC_DIR}/train.py" \
    --model efficientnet \
    --in-channels 3 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}"
  log_done "EfficientNet 3-Kanal fertig."
fi

# --- 1b) EfficientNet 1-Kanal ------------------------------------------------
EXP_EFF_1CH="efficientnet_task-species_ch-1_attn-False_layers-3"
if model_exists "${EXP_EFF_1CH}"; then
  log_skip "EfficientNet 1-Kanal (${EXP_EFF_1CH})"
else
  log_train "EfficientNet 1-Kanal (kondensierte ImageNet-Gewichte)"
  python "${SRC_DIR}/train.py" \
    --model efficientnet \
    --in-channels 1 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}"
  log_done "EfficientNet 1-Kanal fertig."
fi

# --- 1c) MobileNetV3-Small 3-Kanal ------------------------------------------
EXP_MOB_3CH="mobilenet_task-species_ch-3_attn-False_layers-3"
if model_exists "${EXP_MOB_3CH}"; then
  log_skip "MobileNetV3 3-Kanal (${EXP_MOB_3CH})"
else
  log_train "MobileNetV3-Small 3-Kanal"
  python "${SRC_DIR}/train.py" \
    --model mobilenet \
    --in-channels 3 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}"
  log_done "MobileNetV3 3-Kanal fertig."
fi

# --- 1d) MobileNetV3-Small 1-Kanal ------------------------------------------
EXP_MOB_1CH="mobilenet_task-species_ch-1_attn-False_layers-3"
if model_exists "${EXP_MOB_1CH}"; then
  log_skip "MobileNetV3 1-Kanal (${EXP_MOB_1CH})"
else
  log_train "MobileNetV3-Small 1-Kanal (kondensierte ImageNet-Gewichte)"
  python "${SRC_DIR}/train.py" \
    --model mobilenet \
    --in-channels 1 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}"
  log_done "MobileNetV3 1-Kanal fertig."
fi

# =============================================================================
# SCHRITT 2: Architektur-Benchmark (Inferenzzeit & Parametergröße)
# =============================================================================
log_section "SCHRITT 2/2 – Architektur-Benchmark"
echo "  Misst CPU-Inferenzzeit und Modellgröße für alle 4 Varianten."
echo "  Plot wird unter data/visuals/architecture_speed_comparison.png gespeichert."
echo ""

python "${SRC_DIR}/experiments/benchmark_architecture.py"

log_done "Benchmark abgeschlossen."
echo ""
echo -e "${GREEN}Experiment komplett! Ergebnis-Plot:${NC}"
echo "  data/visuals/architecture_speed_comparison.png"
echo ""
echo "Nächster Schritt: IMG_SIZE-Experiment starten."
echo "  Empfehlung: Gewinner-Architektur mit IMG_SIZE 64 / 96 / 128 / 160 / 224 vergleichen."
