#!/bin/bash
# run_docker.sh
#  Standard wrapper for Docker inside Slurm

# --- GPU & CUDA Settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Allow PyTorch to expand memory segments as needed
export CUDA_VISIBLE_DEVICES=0 # Use the first GPU device; adjust as necessary for multi-GPU setups

# 1. Capture Arguments from Slurm
MODE="${1}"
CONF_STEP1="${2}"
CONF_STEP2="${3}"
LIMIT="${4}"

# 2. Define Physical Paths
# Get the absolute path of the directory where this script is located (scripts/)
# and go up one level to get the Project Root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHYS_DIR="$(dirname "$SCRIPT_DIR")" 


# Load .env variables
if [ -f "$SCRIPT_DIR/.env" ]; then
  export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
fi

# Auto-detect Docker image from Step 2 config
# If the config references a medsam3 model, use the SAM3 image (CUDA 12.6+)
# Otherwise, use the default 3090 image (CUDA 12.2)
_CONF2_HOST="$PHYS_DIR/configs/${CONF_STEP2#/workspace/configs/}"
if [ -f "$_CONF2_HOST" ] && grep -q "medsam3" "$_CONF2_HOST" 2>/dev/null; then
    IMAGE_NAME="pipeline_segmentation:sam3"
    echo "[INFO] Detected MedSAM3 config → using Docker image: $IMAGE_NAME"
else
    IMAGE_NAME="pipeline_segmentation:3090"
fi

# Create cache and output directories on host to prevent permission issues
# 1. Cache
mkdir -p "$PHYS_DIR/results/cache/matplotlib"
mkdir -p "$PHYS_DIR/results/cache/wandb"
mkdir -p "$PHYS_DIR/results/cache/huggingface/hub"

# 2. Outputs (Ensure they exist and are writable)
mkdir -p "$PHYS_DIR/results/step1_bboxes"
mkdir -p "$PHYS_DIR/results/step2_masks"

# Broaden permissions for all results (including cache and outputs)
# Broaden permissions for all results (including cache and outputs)
# HOST user cannot chmod root-owned files from previous runs.
# Use a lightweight container to fix permissions as ROOT.
docker run --rm \
    -v "$PHYS_DIR/results":/workspace/data/results \
    "$IMAGE_NAME" \
    chmod -R 777 /workspace/data/results

# 3. Launch Container
# Explicitly pass GPU device ID provided by Slurm
# Mount shared /llms folder
docker run --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --memory="30g" \
    --shm-size=16g \
    --user "$(id -u):$(id -g)" \
    -e USER="$(whoami)" \
    -e HOME="/workspace" \
    -e HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-$HF_TOKEN}" \
    -e HF_TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}" \
    -e WANDB_API_KEY="$WANDB_API_KEY" \
    -e MPLCONFIGDIR="/workspace/data/results/cache/matplotlib" \
    -e WANDB_CACHE_DIR="/workspace/data/results/cache/wandb" \
    -e XDG_CACHE_HOME="/workspace/data/results/cache" \
    -e HF_HOME="/workspace/data/results/cache/huggingface" \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -e TORCHINDUCTOR_CACHE_DIR="/workspace/data/results/cache/torch_inductor" \
    -e TRITON_CACHE_DIR="/workspace/data/results/cache/triton" \
    -e TORCH_HOME="/workspace/data/results/cache/torch" \
    \
    -v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro" \
    -v "/llms:/llms:ro" \
    \
    -v "$PHYS_DIR/results":/workspace/data/results \
    -v "$PHYS_DIR/checkpoints":/workspace/checkpoints \
    -v "$PHYS_DIR/metadata":/workspace/metadata \
    -v "$PHYS_DIR/hf_cache":/workspace/hf_cache \
    \
    -v "$PHYS_DIR/src":/workspace/src \
    -v "$PHYS_DIR/configs":/workspace/configs \
    -v "$PHYS_DIR/scripts":/workspace/scripts \
    \
    "$IMAGE_NAME" \
    /workspace/scripts/run_pipeline.sh "$MODE" "$CONF_STEP1" "$CONF_STEP2" "$LIMIT"