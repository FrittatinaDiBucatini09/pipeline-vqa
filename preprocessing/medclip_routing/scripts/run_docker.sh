#!/bin/bash
# ==============================================================================
# MedCLIP Agentic Routing: Docker Execution Wrapper (NLP Middleware)
# ==============================================================================
# Description:
#   Configures the Docker environment, mounts necessary volumes (read-only where
#   appropriate), and launches the NLP query expansion pipeline.
# ==============================================================================
set -e


# ==============================================================================
# 1. DEFAULT CONFIGURATION (HARDCODED FALLBACKS)
# ==============================================================================

# --- Core Setup ---
IMAGE_NAME="medclip_routing:3090"
PHYS_DIR=$(pwd)

# --- GPU & CUDA Settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Smart GPU Selection:
# - If CUDA_VISIBLE_DEVICES already set -> respect it
# - If running under Slurm (SLURM_JOB_ID exists) -> use Slurm's GPU allocation
# - If running interactively -> default to GPU 0
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    echo "[INFO] Using pre-configured GPU $CUDA_VISIBLE_DEVICES"
elif [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "[INFO] Running in interactive mode -> defaulting to GPU 0"
else
    echo "[INFO] Running under Slurm -> using GPU $CUDA_VISIBLE_DEVICES"
fi

# --- Host Path Definitions ---
OUTPUT_DIR="$PHYS_DIR/results"
METADATA_DIR="$PHYS_DIR"
HF_CACHE_DIR="$PHYS_DIR/hf_cache"

# --- CSV Column Mapping ---
METADATA_FILENAME="gemex_VQA_mimic_mapped.csv"
PATH_COLUMN="image_path"
TEXT_COLUMN="question"

# --- Debugging & Limits ---
WANDB_MODE="disabled"
STOP_AFTER=""

# --- Routing Thresholds ---
ENTITY_THRESHOLD=2
WORD_THRESHOLD=5

# --- Gemma Query Expansion ---
GEMMA_MAX_NEW_TOKENS=128


# ==============================================================================
# 2. CONFIGURATION INJECTION (OVERRIDE LOGIC)
# ==============================================================================

CLI_ARG="${1:-}"
DEFAULT_CONFIG_REF="configs/default.conf"

if [ -n "$CLI_ARG" ]; then
    CONFIG_FILE="$CLI_ARG"
elif [ -f "$DEFAULT_CONFIG_REF" ]; then
    CONFIG_FILE="$DEFAULT_CONFIG_REF"
else
    CONFIG_FILE=""
fi

if [ -n "$CONFIG_FILE" ]; then
    if [ -f "$CONFIG_FILE" ]; then
        echo "[CONFIG] Loading experiment config from: $CONFIG_FILE"
        echo "------------------------------------------------"
        source "$CONFIG_FILE"
        echo "   -> ENTITY_THRESHOLD: $ENTITY_THRESHOLD"
        echo "   -> WORD_THRESHOLD: $WORD_THRESHOLD"
        echo "   -> GEMMA_MAX_NEW_TOKENS: $GEMMA_MAX_NEW_TOKENS"
        echo "------------------------------------------------"
    else
        echo "[ERROR] Config file not found: $CONFIG_FILE"
        exit 1
    fi
else
    echo "[CONFIG] No config file provided. Using script DEFAULTS."
fi

# OVERRIDE: Allow Orchestrator to force a dataset
if [ -n "$DATA_FILE_OVERRIDE" ]; then
    echo "[OVERRIDE] Forcing Dataset: $DATA_FILE_OVERRIDE"
    METADATA_FILENAME="$DATA_FILE_OVERRIDE"
fi

# Resolve OUTPUT_DIR to absolute path (required for Docker mount)
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PHYS_DIR/$OUTPUT_DIR"
fi


# ==============================================================================
# 3. CONFIGURATION SAFETY CHECKS
# ==============================================================================

echo "----------------------------------------------------------------"
echo "[SAFETY CHECK] Verifying Configuration Constraints..."

# Check: Metadata File Existence
if [ ! -f "$METADATA_DIR/$METADATA_FILENAME" ]; then
    echo "[CRITICAL ERROR] Metadata file not found!"
    echo "   Looking for: $METADATA_DIR/$METADATA_FILENAME"
    exit 1
fi

echo "[OK] Configuration valid. Mode: NLP Query Expansion Middleware."
echo "----------------------------------------------------------------"


# ==============================================================================
# 4. ENVIRONMENT SETUP
# ==============================================================================

mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_CACHE_DIR"
chmod -R 777 "$HF_CACHE_DIR" 2>/dev/null || true
chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true


# ==============================================================================
# 5. COMMAND ARGUMENT ASSEMBLY
# ==============================================================================

CMD_ARGS=(
    "--output_dir" "/workspace/data/output"
    "--metadata_file" "/workspace/metadata/$METADATA_FILENAME"
    "--path_col" "$PATH_COLUMN"
    "--text_col" "$TEXT_COLUMN"
    "--entity_threshold" "$ENTITY_THRESHOLD"
    "--word_threshold" "$WORD_THRESHOLD"
    "--gemma_max_new_tokens" "$GEMMA_MAX_NEW_TOKENS"
    "--wandb_mode" "$WANDB_MODE"
)

# Optional: Debug limit
if [ -n "$STOP_AFTER" ]; then
    CMD_ARGS+=("--stop_after" "$STOP_AFTER")
fi


# ==============================================================================
# 6. DOCKER EXECUTION
# ==============================================================================

ENV_FILE="$PHYS_DIR/.env"
DOCKER_ENV_ARGS=""
if [ -f "$ENV_FILE" ]; then DOCKER_ENV_ARGS="--env-file $ENV_FILE"; fi

echo "[INFO] Launching Docker Container..."

docker run --rm \
    --memory="30g" \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --shm-size=16g \
    --user "$(id -u):$(id -g)" \
    -e HOME="/tmp" \
    -e WANDB_MODE="$WANDB_MODE" \
    $DOCKER_ENV_ARGS \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -v "$OUTPUT_DIR":/workspace/data/output \
    -v "$METADATA_DIR":/workspace/metadata \
    -v "$HF_CACHE_DIR":/workspace/hf_cache \
    -v "/llms:/llms:ro" \
    -e HF_HOME="/llms" \
    "$IMAGE_NAME" \
    python3 src/main_routing.py "${CMD_ARGS[@]}"
