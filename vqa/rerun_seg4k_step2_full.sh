#!/bin/bash
# ============================================================
# RE-RUN: Step 2 (Segmentation Masks) - Generation + Judge
# ============================================================
# Previous generation crashed on a corrupted overlay PNG.
# The updated generate_vqa.py now detects and skips truncated
# images during the verification phase.
#
# Usage: ./rerun_seg4k_step2_full.sh
# (Self-submits to SLURM when run from head node)
# ============================================================

if [ -z "$SLURM_JOB_ID" ]; then
    echo "📤 Submitting Step 2 full re-run to SLURM..."
    sbatch \
        --job-name=vqa_seg_s2_rerun \
        --output=slurm_seg_s2_rerun_%j.out \
        --error=slurm_seg_s2_rerun_%j.err \
        -N 1 \
        --gpus=nvidia_geforce_rtx_3090:1 \
        --time=15:00:00 \
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
INPUT_DIR="$PROJECT_ROOT/orchestrator_runs/SEGMENTATION_4K/step_01_segmentation/results/step2_masks"
OUTPUT_DIR="$PROJECT_ROOT/experiments/03_vqa_segmentation_4k/step2_masks"
CONFIG_FILE="configs/generation/attention_4k.conf"
IMAGE_NAME="med_vqa_project:3090"

echo "============================================================"
echo "  RE-RUN: Step 2 (Segmentation Masks) - Full Pipeline"
echo "  Input Dir:  $INPUT_DIR"
echo "  Output Dir: $OUTPUT_DIR"
echo "============================================================"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Token Management
if [ -z "$HF_TOKEN" ] && [ -f "$PHYS_DIR/.env" ]; then
    set -a
    source "$PHYS_DIR/.env"
    set +a
fi

if [ -z "$HF_TOKEN" ]; then
    echo "❌ HF_TOKEN not found!"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ----------------------------------------------------
# STAGE 1: VQA Generation
# ----------------------------------------------------
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

echo "✅ [STAGE 1/2] VQA Generation completed."

# ----------------------------------------------------
# STAGE 2: VQA Evaluation (Judge) - crash-isolated
# ----------------------------------------------------
echo ""
echo "⚖️  [STAGE 2/2] Starting VQA Evaluation (LLM Judge)..."

set +e

docker run \
    --rm \
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
        if [ -z "$LATEST_GEN" ]; then
            echo "❌ [JUDGE ERROR] No generation files found in /output/"
            exit 1
        fi
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

JUDGE_EXIT=$?
set -e

if [ $JUDGE_EXIT -ne 0 ]; then
    echo "⚠️  [STAGE 2/2] Evaluation FAILED (exit code: $JUDGE_EXIT)"
    echo "   Generation results are still safe in: $OUTPUT_DIR"
else
    echo "✅ [STAGE 2/2] Evaluation completed. Results in: $OUTPUT_DIR/judge_results"
fi

echo ""
echo "🏁 All stages finished."
