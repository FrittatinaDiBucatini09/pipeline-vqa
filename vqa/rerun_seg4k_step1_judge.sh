#!/bin/bash
# ============================================================
# RE-RUN: Step 1 (Bounding Boxes) - Judge ONLY
# ============================================================
# Generation already succeeded. This re-runs just the LLM Judge
# with reduced batch_size to avoid OOM (previous run: exit 137).
#
# Usage: ./rerun_seg4k_step1_judge.sh
# (Self-submits to SLURM when run from head node)
# ============================================================

if [ -z "$SLURM_JOB_ID" ]; then
    echo "📤 Submitting Step 1 Judge re-run to SLURM..."
    sbatch \
        --job-name=judge_s1_rerun \
        --output=slurm_judge_s1_rerun_%j.out \
        --error=slurm_judge_s1_rerun_%j.err \
        -N 1 \
        --gpus=nvidia_geforce_rtx_3090:1 \
        --time=04:00:00 \
        -w faretra \
        "$0"
    echo "✅ Job submitted."
    exit 0
fi

# ============================================================
# WORKER MODE
# ============================================================
set -e

PHYS_DIR=$(pwd)
PROJECT_ROOT=$(dirname "$PHYS_DIR")
OUTPUT_DIR="$PROJECT_ROOT/experiments/03_vqa_segmentation_4k/step1_bboxes"
IMAGE_NAME="med_vqa_project:3090"

echo "============================================================"
echo "  RE-RUN: Step 1 (BBox) Judge Only"
echo "  Output Dir: $OUTPUT_DIR"
echo "============================================================"

# Token Management
if [ -z "$HF_TOKEN" ] && [ -f "$PHYS_DIR/.env" ]; then
    set -a
    source "$PHYS_DIR/.env"
    set +a
fi

# Find the generation file
GEN_FILE=$(ls -t "$OUTPUT_DIR"/*_generations_*.json 2>/dev/null | head -n 1)
if [ -z "$GEN_FILE" ]; then
    echo "❌ No generation files found in $OUTPUT_DIR"
    exit 1
fi
echo "📄 Generation file: $(basename "$GEN_FILE")"

docker run --rm \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --memory="50g" \
    --shm-size=16g \
    -v "$PHYS_DIR":/workspace \
    -v "/llms:/llms" \
    -v "$OUTPUT_DIR":/output \
    -e HF_HOME="/llms" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    "$IMAGE_NAME" \
    /bin/bash -c '
        LATEST_GEN=$(ls -t /output/*_generations_*.json 2>/dev/null | head -n 1)
        echo "📄 Judge input: $LATEST_GEN"

        python3 src/llm_judge.py \
            --generations_file "$LATEST_GEN" \
            --judge_model "Qwen/Qwen2.5-7B-Instruct" \
            --output_dir "/output/judge_results" \
            --batch_size 16 \
            --max_tokens 512 \
            --temperature 0.0 \
            --top_k -1 \
            --top_p 1.0 \
            --min_p 0.0 \
            --gpu_memory_utilization 0.85
    '

echo "✅ Judge completed. Results in: $OUTPUT_DIR/judge_results"
