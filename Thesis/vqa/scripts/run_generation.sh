#!/bin/bash

# ==============================================================================
# 1. DEFAULT CONFIGURATION (HARDCODED FALLBACKS)
# ==============================================================================

# --- GPU & CUDA Settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Allow PyTorch to expand memory segments as needed

# Smart GPU selection (already set by submit script, but provide fallback)
# - If CUDA_VISIBLE_DEVICES is not set, default to GPU 0
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================
MODEL_NAME="google/medgemma-4b-it"  # Model to use for VQA inference

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
# Option 1: HuggingFace Dataset
DATASET_NAME=""  # HuggingFace dataset name (leave empty to use local file)
SPLIT="train"                             # Dataset split (train/test/validation)

# Option 2: Local File
DATA_FILE="gemex_VQA_mimic_mapped.csv"  # Path to local dataset file (JSON or CSV)

# Dataset Column Names
IMAGE_COLUMN="image_path"                     # Name of image column in dataset
QUESTION_COLUMN="question"               # Name of question column in dataset
ANSWER_COLUMN="answer"                   # Name of answer column in dataset

# Output Configuration
OUTPUT_DIR="results/vqa_results"         # Directory to save results
IMAGES_DIR="images"                      # Directory to save images (for HuggingFace datasets)
MAX_SAMPLES=""                          # Maximum samples to evaluate (empty = all)

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
BATCH_SIZE=16                            # Batch size for inference
MAX_TOKENS=100                           # Maximum tokens to generate
TEMPERATURE=0.0                          # Sampling temperature (0.0 = greedy)
TOP_K=-1                                 # Top-k sampling (-1 = disabled)
TOP_P=1.0                                # Top-p sampling (1.0 = disabled)
MIN_P=0.0                                # Min-p sampling (0.0 = disabled)
GPU_MEMORY_UTILIZATION=0.6               # Fraction of GPU memory for vLLM (0.0-1.0)

# ============================================================================
# PROMPTING STRATEGY
# ============================================================================
USE_COT=false                            # Enable Chain-of-Thought prompting
USE_IMAGES=true                          # Enable multimodal input with images
ENABLE_THINKING=false                    # Enable thinking mode

# ============================================================================
# FEW-SHOT LEARNING CONFIGURATION
# ============================================================================
USE_FEW_SHOT=true                       # Enable few-shot prompting
NUM_FEW_SHOT=5                           # Number of few-shot examples
FEW_SHOT_SEED=42                         # Random seed for few-shot sampling

# ============================================================================
# OUTPUT CONTROL
# ============================================================================
SAVE_GENERATIONS=true                   # Save model generations to file
TRUST_REMOTE_CODE=false                 # Trust remote code (for custom models)



# ==============================================================================
# 2. CONFIGURATION INJECTION (OVERRIDE LOGIC)
# ==============================================================================

DEFAULT_CONFIG_FILE="configs/generation/hard_coded_gen.conf"
CLI_CONFIG_FILE="${1:-}" # Optional command-line argument for config file path

# Configuration loading logic
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

# Load configuration from the target config file if it exists
if [ -n "$TARGET_CONFIG" ]; then
    if [ -f "$TARGET_CONFIG" ]; then
        echo "------------------------------------------------"
        echo "📂 Reading Config: $TARGET_CONFIG"
        set -a
        source "$TARGET_CONFIG"
        set +a
        
        echo "✅ Configuration loaded active settings:"
        echo "   -> 🤖 Model: $MODEL_NAME"
        
        # Display data source information based on config
        if [ -n "$DATASET_NAME" ]; then
            echo "   -> Data Source: ☁️  HuggingFace Hub"
            echo "   -> Dataset ID:  $DATASET_NAME (Split: $SPLIT)"
        else
            echo "   -> Data Source: 📂 Local File"
            echo "   -> File Path:   $DATA_FILE"
        fi
        
        echo "------------------------------------------------"
    else
        echo "❌ [ERROR] The specified config file does not exist: $TARGET_CONFIG"
        exit 1
    fi
fi

# OVERRIDE: Allow Orchestrator to force a dataset
if [ -n "$DATA_FILE_OVERRIDE" ]; then
    echo "🔵 [OVERRIDE] Forcing Dataset: $DATA_FILE_OVERRIDE"
    DATA_FILE="$DATA_FILE_OVERRIDE"
    DATASET_NAME="" # Disable HF dataset if local file is forced
fi

# ==============================================================================
# 3. COMMAND ARGUMENT ASSEMBLY
# ==============================================================================

# Build the command
CMD="python3 src/generate_vqa.py"
CMD="$CMD --model_name $MODEL_NAME"

# Add data source (HuggingFace or local file)
if [ -n "$DATASET_NAME" ]; then
    CMD="$CMD --dataset_name $DATASET_NAME"
    CMD="$CMD --split $SPLIT"
else
    CMD="$CMD --data_file $DATA_FILE"
fi

# Add dataset column names
CMD="$CMD --image_column $IMAGE_COLUMN"
CMD="$CMD --question_column $QUESTION_COLUMN"
CMD="$CMD --answer_column $ANSWER_COLUMN"

# Add output directories
CMD="$CMD --output_dir $OUTPUT_DIR"
CMD="$CMD --images_dir $IMAGES_DIR"

# Add inference parameters
CMD="$CMD --batch_size $BATCH_SIZE"
CMD="$CMD --max_tokens $MAX_TOKENS"
CMD="$CMD --temperature $TEMPERATURE"
CMD="$CMD --top_k $TOP_K"
CMD="$CMD --top_p $TOP_P"
CMD="$CMD --min_p $MIN_P"
CMD="$CMD --gpu_memory_utilization $GPU_MEMORY_UTILIZATION"

# Add optional parameters
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
fi

# Add boolean flags
if [ "$USE_COT" = true ]; then
    CMD="$CMD --use_cot"
fi

if [ "$USE_IMAGES" = true ]; then
    CMD="$CMD --use_images"
fi

if [ "$ENABLE_THINKING" = true ]; then
    CMD="$CMD --enable_thinking"
fi

if [ "$USE_FEW_SHOT" = true ]; then
    CMD="$CMD --use_few_shot"
    CMD="$CMD --num_few_shot $NUM_FEW_SHOT"
    CMD="$CMD --few_shot_seed $FEW_SHOT_SEED"

    if [ -n "$FEW_SHOT_SOURCE" ]; then
        CMD="$CMD --few_shot_source $FEW_SHOT_SOURCE"
    fi
fi

if [ "$SAVE_GENERATIONS" = true ]; then
    CMD="$CMD --save_generations"
fi

# Add trust remote code flag
if [ "$TRUST_REMOTE_CODE" = true ]; then
    CMD="$CMD --trust_remote_code"
fi

# ==============================================================================
# 4. PRINT CONFIGURATION SUMMARY
# ==============================================================================

# Print configuration
echo "=================================="
echo "VQA Generation Configuration"
echo "=================================="
echo "🤖 Model: $MODEL_NAME 🤖"
echo ""
echo "🗂️ Data Configuration: 🗂️"

if [ -n "$DATASET_NAME" ]; then
    # HuggingFace dataset case
    echo "  🔹 SOURCE TYPE:  HuggingFace Hub"
    echo "  🔹 DATASET NAME: $DATASET_NAME"
    echo "  🔹 SPLIT:        $SPLIT"
    echo "  🔹 IMAGE DIR:    $IMAGES_DIR (Download target)"
else
    # Local file case
    echo "  🔸 SOURCE TYPE:  Local Storage"
    echo "  🔸 FILE PATH:    $DATA_FILE"
    echo "  🔸 IMAGE DIR:    $IMAGES_DIR (Source folder)"
fi

echo ""
echo "Column Mapping:"
echo "  - Image:    $IMAGE_COLUMN"
echo "  - Question: $QUESTION_COLUMN"
echo "  - Answer:   $ANSWER_COLUMN"
echo ""
echo "Output:"
echo "  - Results: $OUTPUT_DIR"
echo "  - Images: $IMAGES_DIR"
echo ""
echo "Inference:"
echo "  - Batch Size: $BATCH_SIZE"
echo "  - Max Tokens: $MAX_TOKENS"
echo "  - Temperature: $TEMPERATURE"
echo "  - Top-K: $TOP_K"
echo "  - Top-P: $TOP_P"
echo "  - Min-P: $MIN_P"
echo "  - GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
if [ -n "$MAX_SAMPLES" ]; then
    echo "  - Max Samples: $MAX_SAMPLES"
fi
echo ""
echo "Prompting Strategy:"
echo "  - Chain-of-Thought: $USE_COT"
echo "  - Use Images: $USE_IMAGES"
echo "  - Enable Thinking: $ENABLE_THINKING"
echo "  - Few-Shot Learning: $USE_FEW_SHOT"
if [ "$USE_FEW_SHOT" = true ]; then
    echo "    - Num Examples: $NUM_FEW_SHOT"
    echo "    - Random Seed: $FEW_SHOT_SEED"
    echo "    - Source: ${FEW_SHOT_SOURCE:-'Same as test data'}"
fi
echo ""
echo "Output Control:"
echo "  - Save Generations: $SAVE_GENERATIONS"
echo "=================================="
echo ""
echo "Running command:"
echo "$CMD"
echo ""

# ==============================================================================
# 5. COMMAND EXECUTION
# ==============================================================================

# Run the command
$CMD
