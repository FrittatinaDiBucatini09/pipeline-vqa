#!/bin/bash
# ==============================================================================
# SLURM SUBMISSION SCRIPT: Full Benchmark Suite (Sequential)
# ==============================================================================
# Description:
#   Runs all grid search configurations sequentially in a SINGLE job.
#   - Iterates through configs in generated_configs/benchmark_v2/
#   - Executes BBox Generation (GPU) using the engine at preprocessing/bounding_box/
#   - Executes IoU Evaluation (CPU) using the evaluation wrapper
#   - Logs progress to a single output file.
# ==============================================================================

# --- Job Identification & Logging ---
#SBATCH --job-name=benchmark_suite
#SBATCH --output=slurm_benchmark_%j.out
#SBATCH --error=slurm_benchmark_%j.err

# --- Resource Allocation ---
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

# --- Node Constraint ---
#SBATCH -w faretra

# ==============================================================================
# 1. SETUP & CHECKS
# ==============================================================================

# Smart GPU selection:
# - If running via sbatch (SLURM_JOB_ID exists): Let SLURM manage GPU allocation
# - If running directly (./script.sh): Use GPU 0 for local testing
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 Running in LOCAL mode - Using GPU 0"
else
    echo "🔧 Running in SLURM mode - GPU managed by scheduler"
fi

echo "----------------------------------------------------------------"
echo "Benchmark Suite Started"
echo "Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "----------------------------------------------------------------"

# Directories
EXPERIMENT_DIR=$(pwd)
BBOX_ENGINE_DIR="$EXPERIMENT_DIR/../../preprocessing/bounding_box"
CONFIGS_DIR="$EXPERIMENT_DIR/generated_configs/benchmark_v2"

# Verify Configs
if [ ! -d "$CONFIGS_DIR" ]; then
    echo "❌ [ERROR] Config directory not found: $CONFIGS_DIR"
    exit 1
fi

# Verify Engine directory
if [ ! -d "$BBOX_ENGINE_DIR/scripts" ]; then
    echo "❌ [ERROR] BBox engine not found: $BBOX_ENGINE_DIR"
    exit 1
fi

# Ensure Docker Image Exists (Auto-Build)
IMAGE_NAME="bbox_preprocessing:3090"
if [[ -z "$(docker images -q "$IMAGE_NAME" 2> /dev/null)" ]]; then
    echo "[INFO] Building Docker image..."
    docker build -f "$BBOX_ENGINE_DIR/docker/Dockerfile.3090" -t "$IMAGE_NAME" "$BBOX_ENGINE_DIR"
fi

# Prepare Evaluation Image
EVAL_IMAGE="bbox_evaluation:cpu"
if [[ -z "$(docker images -q "$EVAL_IMAGE" 2> /dev/null)" ]]; then
    echo "[INFO] Building Evaluation Docker image..."
    docker build -f "$BBOX_ENGINE_DIR/docker/Dockerfile.eval" -t "$EVAL_IMAGE" "$BBOX_ENGINE_DIR/docker"
fi

# ==============================================================================
# 2. EXECUTION LOOP
# ==============================================================================

COUNT=0
TOTAL=$(ls "$CONFIGS_DIR"/*.conf | wc -l)

for config in "$CONFIGS_DIR"/*.conf; do
    FILENAME=$(basename "$config")
    CONFIG_NAME="${FILENAME%.*}"
    ((COUNT++))

    echo "================================================================"
    echo "[$COUNT/$TOTAL] Processing: $CONFIG_NAME"
    echo "================================================================"

    # --- A. GENERATION STEP ---
    echo "▶️  Starting Generation..."

    # Run from the bounding_box engine directory so relative paths resolve
    cd "$BBOX_ENGINE_DIR"

    # Run wrapper directly (NO sbatch)
    ./scripts/run_bbox_preprocessing.sh "$config"

    GEN_EXIT_CODE=$?

    if [ $GEN_EXIT_CODE -ne 0 ]; then
        echo "❌ [ERROR] Generation failed for $CONFIG_NAME (Exit Code: $GEN_EXIT_CODE)"
    else
        echo "✅ Generation Completed."
    fi

    # --- CRITICAL: GPU CLEANUP ---
    echo "🧹 [CLEANUP] Releasing GPU resources..."
    docker ps -q --filter "ancestor=bbox_preprocessing:3090" | xargs -r docker kill 2>/dev/null || true
    sleep 3
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || true
    echo "✅ [CLEANUP] Completed."

    # --- B. EVALUATION STEP ---
    echo "▶️  Starting Evaluation..."

    # Define paths based on convention (results now live under bounding_box/)
    EVAL_INPUT_DIR="$BBOX_ENGINE_DIR/results/benchmark_v2/$CONFIG_NAME"
    EVAL_OUTPUT_DIR="$BBOX_ENGINE_DIR/results/evaluation/benchmark_v2/$CONFIG_NAME"

    # VERIFY OUTPUT EXISTENCE (Fail-Safe)
    if [ ! -f "$EVAL_INPUT_DIR/predictions.jsonl" ]; then
        echo "❌ [ERROR] Generation failed to produce predictions.jsonl for $CONFIG_NAME"
        echo "   Skipping Evaluation."
        cd "$EXPERIMENT_DIR"
        continue
    fi

    # Configure env vars for run_evaluation.sh
    export PREDICTIONS_DIR="$EVAL_INPUT_DIR"
    export OUTPUT_METRICS_DIR="$EVAL_OUTPUT_DIR"

    cd "$BBOX_ENGINE_DIR"
    ./scripts/run_evaluation.sh

    EVAL_EXIT_CODE=$?

    if [ $EVAL_EXIT_CODE -ne 0 ]; then
        echo "❌ [ERROR] Evaluation failed for $CONFIG_NAME"
    else
        echo "✅ Evaluation Completed."
    fi

    # Return to experiment dir for next iteration
    cd "$EXPERIMENT_DIR"

    echo "----------------------------------------------------------------"
done

echo "================================================================"
echo "Benchmark Suite Finished at $(date)"
echo "================================================================"
