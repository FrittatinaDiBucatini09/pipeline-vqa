#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT: Bounding Box Evaluation (IoU Metrics)
# ==============================================================================
# Description:
#   Submits the CPU-only evaluation job to the HPC cluster.
#   Auto-builds the Docker image if not present on the assigned node.
# ==============================================================================

# --- Job Identification & Logging ---
#SBATCH --job-name=bbox_evaluation
#SBATCH --output=slurm_bbox_evaluation_%j.out
#SBATCH --error=slurm_bbox_evaluation_%j.err

# --- Resource Allocation (CPU-Only) ---
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=0:30:00

# --- Node Constraint ---
# CRITICAL: Dataset is stored locally on 'faretra'.
#SBATCH -w faretra

# ==============================================================================
# 1. JOB CONFIGURATION
# ==============================================================================

IMAGE_NAME="bbox_evaluation:cpu"
PHYS_DIR=$(pwd)

# Smart GPU selection:
# - If running via sbatch (SLURM_JOB_ID exists): Let SLURM manage GPU allocation
# - If running directly (./script.sh): Use GPU 0 for local testing
# NOTE: This is a CPU-only job, but we keep this for consistency
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
echo "Evaluation job started."
echo "Node Name         : $SLURMD_NODENAME"
echo "Working Directory : $PHYS_DIR"
echo "----------------------------------------------------------------"

# ==============================================================================
# 3. DOCKER IMAGE VERIFICATION & AUTO-BUILD
# ==============================================================================

if [[ -z "$(docker images -q "$IMAGE_NAME" 2> /dev/null)" ]]; then
    echo "[WARNING] Docker image '$IMAGE_NAME' not found on node $SLURMD_NODENAME."
    echo "[INFO] Initiating auto-build sequence..."

    if [ -f "$PHYS_DIR/docker/Dockerfile.eval" ]; then
        echo "[INFO] Building from source: docker/Dockerfile.eval"
        docker build -f "$PHYS_DIR/docker/Dockerfile.eval" -t "$IMAGE_NAME" "$PHYS_DIR"
    else
        echo "[CRITICAL] Dockerfile.eval not found in $PHYS_DIR/docker/. Auto-build failed."
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

# Parse arguments (Input Directory containing predictions.jsonl, Output Directory)
PRED_DIR="${1:-}"
OUT_DIR="${2:-}"

# Export for the wrapper to pick up
if [ -n "$PRED_DIR" ]; then export PREDICTIONS_DIR="$PRED_DIR"; fi
if [ -n "$OUT_DIR" ]; then export OUTPUT_METRICS_DIR="$OUT_DIR"; fi

echo "----------------------------------------------------------------"
echo "Target Pairs:"
echo "Predictions : ${PREDICTIONS_DIR:-DEFAULT}"
echo "Output      : ${OUTPUT_METRICS_DIR:-DEFAULT}"
echo "----------------------------------------------------------------"

chmod +x scripts/run_evaluation.sh

echo "[INFO] Launching evaluation wrapper..."
./scripts/run_evaluation.sh

echo "[INFO] Evaluation submission script completed."
