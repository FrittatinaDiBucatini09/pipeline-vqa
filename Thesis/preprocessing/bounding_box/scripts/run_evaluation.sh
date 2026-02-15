#!/bin/bash
# ==============================================================================
# GEMeX Evaluation Pipeline: Docker Runtime Wrapper
# ==============================================================================
# Description:
#   Configures the Docker environment and launches the IoU evaluation script.
#   CPU-only — no GPU allocation required.
# ==============================================================================

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================

IMAGE_NAME="bbox_evaluation:cpu"
PHYS_DIR=$(pwd)

# --- Path Definitions ---
# Predictions from the generation step
PREDICTIONS_FILE="/workspace/predictions/predictions.jsonl"

# Dataset root for resolving image paths (read-only)
INPUT_DIR="/datasets/MIMIC-CXR"

# Output directory inside container
OUTPUT_DIR_CONTAINER="/workspace/results"

# Host paths
# Host paths (Can be overridden by env vars)
PREDICTIONS_HOST="${PREDICTIONS_DIR:-$PHYS_DIR/results}"
OUTPUT_HOST="${OUTPUT_METRICS_DIR:-$PHYS_DIR/results}"

# --- Evaluation Parameters ---
REF_DIM="224"

# ==============================================================================
# 2. CONFIGURATION INJECTION (OVERRIDE LOGIC)
# ==============================================================================

CLI_CONFIG="${1:-}"

if [ -n "$CLI_CONFIG" ] && [ -f "$CLI_CONFIG" ]; then
    echo "🔵 [CONFIG] Loading config from: $CLI_CONFIG"
    source "$CLI_CONFIG"
fi

# ==============================================================================
# 3. SAFETY CHECKS
# ==============================================================================

echo "----------------------------------------------------------------"
echo "[SAFETY CHECK] Verifying Evaluation Prerequisites..."

# Check 1: Predictions file
if [ ! -f "$PREDICTIONS_HOST/predictions.jsonl" ]; then
    echo "❌ [CRITICAL ERROR] Predictions file not found!"
    echo "   Expected: $PREDICTIONS_HOST/predictions.jsonl"
    echo "   Run the generation pipeline first."
    exit 1
fi

# Check 2: Dataset availability
if [ ! -d "/datasets/MIMIC-CXR" ]; then
    echo "❌ [CRITICAL ERROR] Dataset directory not found!"
    echo "   Path '/datasets/MIMIC-CXR' does not exist on this node ($HOSTNAME)."
    echo "   Ensure you are running on faretra or the mount is active."
    exit 1
fi

echo "✅ [OK] All prerequisites met."
echo "----------------------------------------------------------------"

# ==============================================================================
# 4. ENVIRONMENT SETUP
# ==============================================================================

mkdir -p "$OUTPUT_HOST"
chmod -R 777 "$OUTPUT_HOST" 2>/dev/null || true

# ==============================================================================
# 5. COMMAND ARGUMENT ASSEMBLY
# ==============================================================================

CMD_ARGS=(
    "--predictions_file" "$PREDICTIONS_FILE"
    "--input_dir" "$INPUT_DIR"
    "--output_dir" "$OUTPUT_DIR_CONTAINER"
    "--ref_dim" "$REF_DIM"
)

# ==============================================================================
# 6. DOCKER EXECUTION (CPU-ONLY)
# ==============================================================================

ENV_FILE="$PHYS_DIR/.env"
DOCKER_ENV_ARGS=""
if [ -f "$ENV_FILE" ]; then DOCKER_ENV_ARGS="--env-file $ENV_FILE"; fi

echo "[INFO] Launching Evaluation Container (CPU-only)..."

docker run --rm \
    --memory="8g" \
    --user "$(id -u):$(id -g)" \
    $DOCKER_ENV_ARGS \
    -v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro" \
    -v "$PREDICTIONS_HOST":/workspace/predictions:ro \
    -v "$OUTPUT_HOST":/workspace/results \
    "$IMAGE_NAME" \
    python3 evaluate_bbox.py "${CMD_ARGS[@]}"

echo "[INFO] Evaluation wrapper completed."
