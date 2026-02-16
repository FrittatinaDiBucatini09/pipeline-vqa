#!/bin/bash

# ==============================================================================
# 1. DEFAULT CONFIGURATION (HARDCODED FALLBACKS)
# ==============================================================================

# --- GPU & CUDA Settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Smart GPU selection (already set by submit script, but provide fallback)
# - If CUDA_VISIBLE_DEVICES is not set, default to GPU 0
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# ==============================================================================
# JUDGE MODEL CONFIGURATION
# ==============================================================================
JUDGE_MODEL="Qwen/Qwen2.5-7B-Instruct"  # Model used as the evaluator
JUDGE_USE_COT=false                     # Enable Chain-of-Thought for the judge

# ==============================================================================
# INPUT/OUTPUT CONFIGURATION
# ==============================================================================
# Input file strategy:
# If GENERATIONS_FILE is empty, the script will try to find the latest file automatically.
GENERATIONS_FILE=""                     
OUTPUT_DIR="results/judge_results"      # Directory to save evaluation results

# ==============================================================================
# INFERENCE PARAMETERS
# ==============================================================================
BATCH_SIZE=32                           # Batch size for judge inference
MAX_TOKENS=512                          # Max tokens for judge response (higher for CoT)
TEMPERATURE=0.0                         # Greedy decoding for consistent evaluation
TOP_K=-1
TOP_P=1.0
MIN_P=0.0
MAX_SAMPLES=""                          # Empty = evaluate all samples

# ==============================================================================
# SECURITY
# ==============================================================================
TRUST_REMOTE_CODE=false                 # Only enable for authorized models (e.g., MedGemma)

# ==============================================================================
# DEBUG
# ==============================================================================
VERBOSE=false


# ==============================================================================
# 2. CONFIGURATION INJECTION (HIERARCHICAL LOAD)
# ==============================================================================

DEFAULT_CONFIG_FILE="configs/judge/hard_coded_judge.conf"
CLI_CONFIG_FILE="${1:-}" # Optional command-line argument

# Determine which config to load
if [ -n "$CLI_CONFIG_FILE" ]; then
    TARGET_CONFIG="$CLI_CONFIG_FILE"
    echo "🔵 [MODE] User Override: Loading config from argument."
elif [ -f "$DEFAULT_CONFIG_FILE" ]; then
    TARGET_CONFIG="$DEFAULT_CONFIG_FILE"
    echo "🟡 [MODE] Default: Loading standard config file."
else
    TARGET_CONFIG=""
    echo "🟠 [MODE] Fallback: No config file found. Using script HARDCODED defaults."
fi

# Load configuration
if [ -n "$TARGET_CONFIG" ]; then
    if [ -f "$TARGET_CONFIG" ]; then
        echo "------------------------------------------------"
        echo "📂 Reading Config: $TARGET_CONFIG"
        set -a
        source "$TARGET_CONFIG"
        set +a
        echo "✅ Configuration loaded active settings:"
        echo "   -> Judge Model: $JUDGE_MODEL"
        echo "------------------------------------------------"
    else
        echo "❌ [ERROR] The specified config file does not exist: $TARGET_CONFIG"
        exit 1
    fi
fi


# ==============================================================================
# 3. INPUT FILE RESOLUTION (AUTO-DISCOVERY)
# ==============================================================================

# If GENERATIONS_FILE is not set in config, try to find the latest one
if [ -z "$GENERATIONS_FILE" ]; then
    echo "🔍 No input file specified. Searching for latest generation..."
    # Finds the most recently modified json file matching the pattern
    LATEST_FILE=$(ls -t results/vqa_results/*_generations_*.json 2>/dev/null | head -n 1)
    
    if [ -z "$LATEST_FILE" ]; then
        echo "❌ [ERROR] No VQA generation files found in results/vqa_results/"
        echo "   Possible causes:"
        echo "   1. VQA inference has not been executed yet."
        echo "   2. Output directory is different (check configs)."
        exit 1
    fi
    GENERATIONS_FILE="$LATEST_FILE"
    echo "   -> Auto-selected: $GENERATIONS_FILE"
else
    # Verify the manually specified file exists
    if [ ! -f "$GENERATIONS_FILE" ]; then
        echo "❌ [ERROR] Specified input file not found:"
        echo "   -> $GENERATIONS_FILE"
        exit 1
    fi
    echo "   -> Using specified file: $GENERATIONS_FILE"
fi


# ==============================================================================
# 4. COMMAND ARGUMENT ASSEMBLY
# ==============================================================================

CMD="python3 src/llm_judge.py"

# Required Arguments
CMD="$CMD --generations_file $GENERATIONS_FILE"
CMD="$CMD --judge_model $JUDGE_MODEL"
CMD="$CMD --output_dir $OUTPUT_DIR"

# Inference Arguments
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_tokens $MAX_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_k $TOP_K"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --min_p $MIN_P"

# Optional Arguments
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

if [ "$JUDGE_USE_COT" = true ]; then
    CMD="$CMD --judge_use_cot"
fi

if [ "$VERBOSE" = true ]; then
    CMD="$CMD --verbose"
fi

if [ "$TRUST_REMOTE_CODE" = true ]; then
    CMD="$CMD --trust_remote_code"
fi


# ==============================================================================
# 5. PRINT CONFIGURATION SUMMARY
# ==============================================================================

echo "=================================="
echo "⚖️  LLM Judge Configuration"
echo "=================================="
echo "Input Source:"
echo "  📄 File: $GENERATIONS_FILE"
echo ""
echo "Judge Model:"
echo "  🤖 Model: $JUDGE_MODEL 🤖"
echo "  🧠 Chain-of-Thought: $JUDGE_USE_COT"
echo ""
echo "Output:"
echo "  📂 Directory: $OUTPUT_DIR"
echo ""
echo "Inference Params:"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Max Tokens: $MAX_TOKENS"
echo "  - Temperature: $TEMPERATURE"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  - Max Samples: $MAX_SAMPLES"
fi
echo "=================================="
echo ""
echo "Running command:"
echo "$CMD"
echo ""

# ==============================================================================
# 6. COMMAND EXECUTION
# ==============================================================================

$CMD