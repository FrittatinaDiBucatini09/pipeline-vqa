#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT: Attention Map Heatmap Generation
# ==============================================================================
# Description:
#   Orchestrates the submission of the heatmap generation job to the HPC cluster.
#   Performs node-local Docker image verification (auto-build if missing)
#   and delegates the runtime execution to the wrapper script.
# ==============================================================================

# --- Job Identification & Logging ---
#SBATCH --job-name=heatmap_gen
#SBATCH --output=slurm_heatmap_gen_%j.out
#SBATCH --error=slurm_heatmap_gen_%j.err

# --- Resource Allocation ---
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --time=5:00:00

# --- Node Constraint ---
# CRITICAL: Forces execution on 'faretra' as the dataset is stored locally.
#SBATCH -w faretra

# ==============================================================================
# 1. JOB CONFIGURATION
# ==============================================================================

# Target configuration
TARGET_CONFIG="${1:-configs/gemex/heatmap_default.conf}"

# Docker image tag
IMAGE_NAME="heatmap_gen:3090"

# Capture the submission directory
PHYS_DIR=$(pwd)

# Smart GPU selection:
# - If running via sbatch (SLURM_JOB_ID exists): Let SLURM manage GPU allocation
# - If running directly (./script.sh): Use GPU 0 for local testing
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 Running in LOCAL mode - Using GPU 0"
else
    echo "🔧 Running in SLURM mode - GPU managed by scheduler"
fi

# ==============================================================================
# 2. DEBUGGING INFORMATION
# ==============================================================================

echo "----------------------------------------------------------------"
echo "Job execution started."
echo "Node Name         : $SLURMD_NODENAME"
echo "Assigned GPU(s)   : $CUDA_VISIBLE_DEVICES"
echo "Working Directory : $PHYS_DIR"
echo "Target config.    : $TARGET_CONFIG"
echo "----------------------------------------------------------------"

# ==============================================================================
# 3. DOCKER IMAGE VERIFICATION & AUTO-BUILD
# ==============================================================================

if [[ -z "$(docker images -q "$IMAGE_NAME" 2> /dev/null)" ]]; then
    echo "[WARNING] Docker image '$IMAGE_NAME' not found on node $SLURMD_NODENAME."
    echo "[INFO] Initiating auto-build sequence..."

    if [ -f "$PHYS_DIR/docker/Dockerfile.3090" ]; then
        echo "[INFO] Building from source: docker/Dockerfile.3090"
        docker build -f "$PHYS_DIR/docker/Dockerfile.3090" -t "$IMAGE_NAME" "$PHYS_DIR"

    elif [ -f "$PHYS_DIR/Dockerfile.3090" ]; then
        echo "[INFO] Building from source: ./Dockerfile.3090"
        docker build -f "$PHYS_DIR/Dockerfile.3090" -t "$IMAGE_NAME" "$PHYS_DIR"
    else
        echo "[CRITICAL] Dockerfile not found. Auto-build failed."
        echo "Expected locations: $PHYS_DIR/docker/Dockerfile.3090 or $PHYS_DIR/Dockerfile.3090"
        exit 1
    fi

    if [ $? -ne 0 ]; then
        echo "[ERROR] Docker build command failed. Exiting job."
        exit 1
    fi
    echo "[SUCCESS] Image built successfully."
else
    echo "[INFO] Docker image '$IMAGE_NAME' found locally. Skipping build."
fi

# ==============================================================================
# 4. PIPELINE DELEGATION
# ==============================================================================

chmod +x scripts/run_heatmap_gen.sh

echo "[INFO] Launching runtime wrapper script..."
./scripts/run_heatmap_gen.sh "$TARGET_CONFIG"

echo "[INFO] Job submission script completed."
