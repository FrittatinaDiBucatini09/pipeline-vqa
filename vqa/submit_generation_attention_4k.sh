#!/bin/bash
# ============================================================
# SLURM SUBMISSION SCRIPT: ATTENTION 4K VQA GENERATION
# ============================================================
# This script sets up a custom environment to run VQA generation
# using the output from the ATTENTION_4K experiment.
#
# It performs the following:
# 1. Hardcodes the path to the Attention Map results.
# 2. Mounts the results directory to /preprocessed_images (Read-Only).
# 3. Mounts the experiment output directory to /output (Read-Write).
# 4. Sets environment variables to redirect data loading.
#
# Usage:
#   sbatch submit_generation_attention_4k.sh
# ============================================================

#SBATCH --job-name=vqa_attn4k
#SBATCH --output=slurm_vqa_attn4k_%j.out
#SBATCH --error=slurm_vqa_attn4k_%j.err
#SBATCH -N 1
#SBATCH --gpus=nvidia_geforce_rtx_3090:1
#SBATCH --time=15:00:00
#SBATCH -w faretra

# ----------------------------------------------------
# 1. PATH DEFINITIONS
# ----------------------------------------------------

# Current directory (Thesis/vqa)
PHYS_DIR=$(pwd)
echo "📍 Working Directory: $PHYS_DIR"

# Project Root (Thesis)
PROJECT_ROOT=$(dirname "$PHYS_DIR")
echo "📂 Project Root: $PROJECT_ROOT"

# Input: Attention Map Results
# Points to: Thesis/orchestrator_runs/ATTENTION_4K/step_01_attn_map/results
INPUT_DIR="$PROJECT_ROOT/orchestrator_runs/ATTENTION_4K/step_01_attn_map/results"

# Validation
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ [ERROR] Input directory not found: $INPUT_DIR"
    exit 1
fi
echo "📥 Input Directory: $INPUT_DIR"

# Output: Experiment Results
# Points to: Thesis/experiments/02_vqa_attention_4k/results
OUTPUT_DIR="$PROJECT_ROOT/experiments/02_vqa_attention_4k/results"

# Ensure output dir exists
mkdir -p "$OUTPUT_DIR"
echo "📤 Output Directory: $OUTPUT_DIR"

# Configuration File
CONFIG_FILE="configs/generation/attention_4k.conf"

# ----------------------------------------------------
# 2. ENVIRONMENT SETUP
# ----------------------------------------------------

# Docker Image Name
IMAGE_NAME="med_vqa_project:3090"

# GPU Settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Helper for local vs SLURM execution
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 Running in LOCAL mode - Using GPU 0"
else
    echo "🔧 Running in SLURM mode - GPU managed by scheduler"
fi

# Token Management
if [ -z "$HF_TOKEN" ] && [ -f "$PHYS_DIR/.env" ]; then
    echo "🔑 Loading secrets from .env..."
    set -a
    source "$PHYS_DIR/.env"
    set +a
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ [CRITICAL] HF_TOKEN not found!"
    exit 1
fi

# ----------------------------------------------------
# 3. VQA GENERATION
# ----------------------------------------------------
# Generation runs with fail-fast: if it crashes, we stop immediately.

echo "🚀 [STAGE 1/2] Starting VQA Generation..."

docker run \
    --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --memory="30g" \
    --shm-size=16g \
    -v "$PHYS_DIR":/workspace \
    -v "/llms:/llms" \
    -v "/datasets:/datasets:ro" \
    -v "$INPUT_DIR":/preprocessed_images:ro \
    -v "$OUTPUT_DIR":/output \
    -e HF_HOME="/llms" \
    -e VQA_IMAGE_PATH="/preprocessed_images" \
    -e DATA_FILE_OVERRIDE="/preprocessed_images/vqa_manifest.csv" \
    -e HF_TOKEN="$HF_TOKEN" \
    "$IMAGE_NAME" \
    /bin/bash /workspace/scripts/run_generation.sh "$CONFIG_FILE"

echo "✅ [STAGE 1/2] VQA Generation completed. Results saved to: $OUTPUT_DIR"

# ----------------------------------------------------
# 4. VQA EVALUATION (CRASH-ISOLATED)
# ----------------------------------------------------
# The evaluation runs in a SEPARATE container.
# We disable fail-fast (set +e) so that if the judge crashes,
# the generation results (already saved) are NOT affected.

echo ""
echo "⚖️  [STAGE 2/2] Starting VQA Evaluation (LLM Judge)..."
echo "   ⚠️  This step is crash-isolated: generation results are safe."

set +e  # Disable fail-fast for evaluation

docker run \
    --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --memory="30g" \
    --shm-size=16g \
    -v "$PHYS_DIR":/workspace \
    -v "/llms:/llms" \
    -v "/datasets:/datasets:ro" \
    -v "$OUTPUT_DIR":/output \
    -e HF_HOME="/llms" \
    -e HF_TOKEN="$HF_TOKEN" \
    "$IMAGE_NAME" \
    /bin/bash -c '
        # Point the judge at the generation output directory
        export OUTPUT_DIR="/output"

        # Find the latest generation file
        LATEST_GEN=$(ls -t /output/*_generations_*.json 2>/dev/null | head -n 1)
        if [ -z "$LATEST_GEN" ]; then
            echo "❌ [JUDGE ERROR] No generation files found in /output/"
            exit 1
        fi
        echo "📄 Judge input: $LATEST_GEN"

        # Run the judge
        python3 src/llm_judge.py \
            --generations_file "$LATEST_GEN" \
            --judge_model "Qwen/Qwen2.5-7B-Instruct" \
            --output_dir "/output/judge_results" \
            --batch_size 32 \
            --max_tokens 512 \
            --temperature 0.0 \
            --top_k -1 \
            --top_p 1.0 \
            --top_p 1.0 \
            --min_p 0.0 \
            --gpu_memory_utilization 0.9
    '

JUDGE_EXIT=$?
set -e  # Re-enable fail-fast

if [ $JUDGE_EXIT -ne 0 ]; then
    echo ""
    echo "⚠️  [STAGE 2/2] Evaluation FAILED (exit code: $JUDGE_EXIT)"
    echo "   Generation results are still safe in: $OUTPUT_DIR"
    echo "   You can re-run the judge manually later."
else
    echo "✅ [STAGE 2/2] Evaluation completed. Results in: $OUTPUT_DIR/judge_results"
fi

echo ""
echo "🏁 All stages finished."

