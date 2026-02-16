#!/bin/bash
#SBATCH -N 1
#SBATCH --gpus nvidia_geforce_rtx_3090:1
#SBATCH -w faretra
#SBATCH --job-name=vqa_integration_test
#SBATCH --output=slurm-integration-test-%j.out
#SBATCH --time=01:00:00

# ==============================================================================
# INTEGRATION TEST: VQA Pipeline (Generation → Judge)
# ==============================================================================
# Purpose: Validate the full pipeline with real models on cluster
# Scope: 50 samples for quick validation
# ==============================================================================

echo "=========================================="
echo "Integration Test: VQA Pipeline"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "=========================================="

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/llms

# Smart GPU selection:
# - If running via sbatch (SLURM_JOB_ID exists): Let SLURM manage GPU allocation
# - If running directly (./script.sh): Use GPU 0 for local testing
if [ -z "$SLURM_JOB_ID" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "🔧 Running in LOCAL mode - Using GPU 0"
else
    echo "🔧 Running in SLURM mode - GPU managed by scheduler"
fi

# Navigate to project directory
cd /home/rbalzani/medical-vqa/Thesis/vqa || exit 1

# ==============================================================================
# PHASE 1: VQA GENERATION
# ==============================================================================
echo ""
echo "=========================================="
echo "PHASE 1: VQA GENERATION"
echo "=========================================="

docker run --rm --gpus '"device=0"' \
    -v "$PWD":/workspace \
    -v /llms:/llms \
    -v /datasets:/datasets:ro \
    -e HF_HOME=/llms \
    med_vqa_project:3090 \
    python3 src/generate_vqa.py \
        --model_name "google/medgemma-2b-it" \
        --data_file gemex_VQA_mimic_mapped.csv \
        --image_column original_hf_path \
        --question_column question \
        --answer_column answer \
        --batch_size 8 \
        --max_new_tokens 100 \
        --max_samples 50 \
        --output_dir results/cluster_integration_test \
        --use_images \
        --use_few_shot \
        --num_few_shot 3

GENERATION_EXIT_CODE=$?

if [ $GENERATION_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Phase 1 (Generation) failed with exit code $GENERATION_EXIT_CODE"
    echo "Check the output above for errors."
    exit 1
fi

echo ""
echo "✅ Phase 1 (Generation) completed successfully"

# Find the generated file
GENERATIONS_FILE=$(ls -t results/cluster_integration_test/*_generations_*.json 2>/dev/null | head -n 1)

if [ -z "$GENERATIONS_FILE" ]; then
    echo "❌ No generation file found in results/cluster_integration_test/"
    exit 1
fi

echo "Generated file: $GENERATIONS_FILE"

# Validate generation file
NUM_GENERATIONS=$(python3 -c "import json; print(len(json.load(open('$GENERATIONS_FILE'))))" 2>/dev/null)
if [ -z "$NUM_GENERATIONS" ]; then
    echo "❌ Failed to read generation file (invalid JSON?)"
    exit 1
fi

echo "Number of generations: $NUM_GENERATIONS"

# ==============================================================================
# PHASE 2: JUDGE EVALUATION
# ==============================================================================
echo ""
echo "=========================================="
echo "PHASE 2: JUDGE EVALUATION"
echo "=========================================="

docker run --rm --gpus '"device=0"' \
    -v "$PWD":/workspace \
    -v /llms:/llms \
    -e HF_HOME=/llms \
    med_vqa_project:3090 \
    python3 src/llm_judge.py \
        --generations_file "$GENERATIONS_FILE" \
        --judge_model "Qwen/Qwen2.5-7B-Instruct" \
        --output_dir results/cluster_integration_test/judge \
        --batch_size 16 \
        --max_tokens 200 \
        --verbose

JUDGE_EXIT_CODE=$?

if [ $JUDGE_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ Phase 2 (Judge) failed with exit code $JUDGE_EXIT_CODE"
    echo "Check the output above for errors."
    exit 1
fi

echo ""
echo "✅ Phase 2 (Judge) completed successfully"

# ==============================================================================
# SUMMARY
# ==============================================================================
echo ""
echo "=========================================="
echo "✅ INTEGRATION TEST COMPLETE"
echo "=========================================="
echo "End: $(date)"
echo "Duration: $SECONDS seconds"
echo ""

# Find judge results
JUDGE_FILE=$(ls -t results/cluster_integration_test/judge/judge_results_*.json 2>/dev/null | head -n 1)

echo "Output Files:"
echo "  - Generations: $GENERATIONS_FILE"
echo "  - Judge Results: $JUDGE_FILE"
echo ""

# Print key metrics
if [ -n "$JUDGE_FILE" ] && [ -f "$JUDGE_FILE" ]; then
    echo "Key Metrics:"
    python3 -c "
import json
with open('$JUDGE_FILE', 'r') as f:
    data = json.load(f)
    metrics = data['metrics']
    print(f\"  Total Samples: {metrics['total_samples']}\")
    print(f\"  Judge Accuracy: {metrics['judge_accuracy']:.2%}\")
    print(f\"  Exact Match Accuracy: {metrics['exact_match_accuracy']:.2%}\")
    print(f\"  Agreement Rate: {metrics['agreement_rate']:.2%}\")
    print(f\"  Judge Correct (Exact Wrong): {metrics['judge_correct_exact_wrong']} (paraphrases)\")
" 2>/dev/null || echo "  (Could not parse metrics)"
fi

echo ""
echo "=========================================="
echo "Integration test completed successfully!"
echo "Review the output files above for detailed results."
echo "=========================================="

exit 0
