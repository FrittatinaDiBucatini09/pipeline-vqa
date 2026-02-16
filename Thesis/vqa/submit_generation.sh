#!/bin/bash
# ============================================================
# SLURM/DOCKER ORCHESTRATOR FOR VQA PIPELINE
# ============================================================
# This script prepares the environment, handles permissions,
# and launches the Docker container with the correct mounts.
#
# Usage:
#   ./submit_generation.sh [optional_config_file]
#
# Example:
#   ./submit_generation.sh configs/generation/medgemma_cot.conf
# ============================================================

#SBATCH --job-name=vqa_gen
#SBATCH --output=slurm_vqa_%j.out
#SBATCH --error=slurm_vqa_%j.err
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --time=15:00:00
#SBATCH -w faretra

# ----------------------------------------------------
# 1. ARGUMENT PARSING & VALIDATION
# ----------------------------------------------------

CONFIG_FILE="" # Configuration file path (optional argument)

# Check if an argument is provided
if [ -n "$1" ]; then
    CONFIG_FILE="$1"
fi

PHYS_DIR=$(pwd)

# Fail-fast: If a config file is specified but doesn't exist on host, stop immediately.
if [ -n "$CONFIG_FILE" ]; then
    if [ ! -f "$PHYS_DIR/$CONFIG_FILE" ]; then
        echo "❌ [ERROR] Config file not found on host: $PHYS_DIR/$CONFIG_FILE"
        exit 1
    fi
    echo "📋 Configuration file detected: $CONFIG_FILE"
else
    echo "⚠️  No config file provided. Using default inside container."
fi

# ----------------------------------------------------
# 2. ENVIRONMENT SETUP
# ----------------------------------------------------

# Docker Image Name
IMAGE_NAME="med_vqa_project:3090"

# GPU Settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Smart GPU selection:
# - If running via sbatch (SLURM_JOB_ID exists): Let SLURM manage GPU allocation
# - If running directly (./script.sh): Use GPU 0 for local testing
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 Running in LOCAL mode - Using GPU 0"
else
    echo "🔧 Running in SLURM mode - GPU managed by scheduler"
fi 

# Cache Directories (Shared Storage)
LLM_CACHE_DIR="/llms"

# ----------------------------------------------------
# 3. TOKEN MANAGEMENT
# ----------------------------------------------------

# Load .env if HF_TOKEN is missing
if [ -z "$HF_TOKEN" ] && [ -f "$PHYS_DIR/.env" ]; then
    echo "🔑 Loading secrets from .env..."
    set -a
    source "$PHYS_DIR/.env"
    set +a
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ [CRITICAL] HF_TOKEN not found! Please set it in .env or environment."
    exit 1
fi

# ----------------------------------------------------
# 4. PREPROCESSING BRIDGE DETECTION
# ----------------------------------------------------
# When VQA_IMAGE_PATH is set externally (by the orchestrator bridge),
# it points to the bbox preprocessing output directory on the host.
# We mount it as a separate read-only volume inside the container.

EXTRA_MOUNTS=""
DOCKER_VQA_IMAGE_PATH="/workspace"

if [ -n "$VQA_IMAGE_PATH" ] && [ -d "$VQA_IMAGE_PATH" ] && [ "$VQA_IMAGE_PATH" != "$PHYS_DIR" ]; then
    echo "🔗 [BRIDGE MODE] Preprocessing output detected at: $VQA_IMAGE_PATH"
    EXTRA_MOUNTS="-v $VQA_IMAGE_PATH:/preprocessed_images:ro"
    DOCKER_VQA_IMAGE_PATH="/preprocessed_images"

    # Remap DATA_FILE_OVERRIDE to the container path if it points into VQA_IMAGE_PATH
    if [ -n "$DATA_FILE_OVERRIDE" ]; then
        OVERRIDE_BASENAME=$(basename "$DATA_FILE_OVERRIDE")
        DATA_FILE_OVERRIDE="/preprocessed_images/$OVERRIDE_BASENAME"
        echo "🔗 [BRIDGE MODE] DATA_FILE_OVERRIDE remapped to: $DATA_FILE_OVERRIDE"
    fi
fi

# ----------------------------------------------------
# 5. CONTAINER EXECUTION
# ----------------------------------------------------

echo "🚀 Starting Container..."
echo "   -> Node: $SLURMD_NODENAME"
echo "   -> GPU:  $CUDA_VISIBLE_DEVICES"
echo "   -> Image: $IMAGE_NAME"
echo "   -> VQA_IMAGE_PATH: $DOCKER_VQA_IMAGE_PATH"

# Note on the Docker Command:
# We pass "$CONFIG_FILE" at the end. This argument travels through Docker
# and is received by scripts/run_generation.sh inside the container.

docker run \
    --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --memory="30g" \
    --shm-size=16g \
    -v "$PHYS_DIR":/workspace \
    -v "$LLM_CACHE_DIR":/llms \
    -v "/datasets:/datasets:ro" \
    $EXTRA_MOUNTS \
    -e HF_HOME="/llms" \
    -e VQA_IMAGE_PATH="$DOCKER_VQA_IMAGE_PATH" \
    -e HF_TOKEN="$HF_TOKEN" \
    -e DATA_FILE_OVERRIDE="${DATA_FILE_OVERRIDE:-}" \
    "$IMAGE_NAME" \
    /bin/bash /workspace/scripts/run_generation.sh "$CONFIG_FILE"