#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT: Bounding Box Preprocessing Pipeline
# ==============================================================================
# Description:
#   Orchestrates the submission of the preprocessing job to the HPC cluster.
#   It performs node-local Docker image verification (auto-build if missing)
#   and delegates the runtime execution to the wrapper script.
# ==============================================================================

# --- Job Identification & Logging ---
#SBATCH --job-name=bbox_preprocessing
#SBATCH --output=slurm_bbox_preprocessing_%j.out
#SBATCH --error=slurm_bbox_preprocessing_%j.err

# --- Resource Allocation ---
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --time=5:00:00

# --- Node Constraint ---
# CRITICAL: Forces execution on 'faretra' as the dataset is stored locally 
# on this node. The cluster lacks a distributed file system for large datasets.
#SBATCH -w faretra

# ==============================================================================
# 1. JOB CONFIGURATION
# ==============================================================================

# Target configuration
# Target configuration (Accepts argument or defaults to hardcoded)
if [ -n "$1" ]; then
    TARGET_CONFIG="$1"
else
    TARGET_CONFIG="configs/gemex/hard_coded.conf"
fi

# Docker image tag (must match the tag used in the build process)
IMAGE_NAME="bbox_preprocessing:3090"

# Capture the submission directory (Physical path on the host)
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
# Ensures the Docker image exists on the assigned node before execution.
# Since Slurm nodes do not share the Docker registry, we must verify/build locally.

if [[ -z "$(docker images -q "$IMAGE_NAME" 2> /dev/null)" ]]; then
    echo "[WARNING] Docker image '$IMAGE_NAME' not found on node $SLURMD_NODENAME."
    echo "[INFO] Initiating auto-build sequence..."

    # Build Strategy: 
    # 1. Priority: Check 'scripts/' directory (standard project structure).
    # 2. Fallback: Check root directory.
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

    # Validate build exit code
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

# Grant execution permissions to the wrapper script
chmod +x scripts/run_bbox_preprocessing.sh

# Delegate to the wrapper script.
# The wrapper handles Docker volume mounting and passes environment arguments.
echo "[INFO] Launching runtime wrapper script..."
./scripts/run_bbox_preprocessing.sh "$TARGET_CONFIG"

echo "[INFO] Job submission script completed."