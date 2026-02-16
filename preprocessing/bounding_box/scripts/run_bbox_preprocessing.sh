#!/bin/bash
# ==============================================================================
# GEMeX Preprocessing Pipeline: Question-Driven Multi-Box Mode
# ==============================================================================
# Description:
#   Configures the Docker environment, mounts necessary volumes (read-only where
#   appropriate), and launches the Python pipeline with the specified hyperparameters.
# ==============================================================================


# ==============================================================================
# 1. DEFAULT CONFIGURATION (HARDCODED FALLBACKS)
# ==============================================================================

# --- Core Setup ---
IMAGE_NAME="bbox_preprocessing:3090"
PHYS_DIR=$(pwd)

# --- GPU & CUDA Settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True # Allow PyTorch to expand memory segments as needed

# Smart GPU Selection:
# - If CUDA_VISIBLE_DEVICES already set → respect it (from test scripts or manual override)
# - If running under Slurm (SLURM_JOB_ID exists) → use Slurm's GPU allocation
# - If running interactively (./run_bbox_preprocessing.sh) → default to GPU 0
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
    # CUDA_VISIBLE_DEVICES already set — respect it
    echo "[INFO] Using pre-configured GPU $CUDA_VISIBLE_DEVICES"
elif [ -z "$SLURM_JOB_ID" ]; then
    # Interactive mode with no GPU specified — default to GPU 0
    export CUDA_VISIBLE_DEVICES=0
    echo "[INFO] Running in interactive mode → defaulting to GPU 0"
else
    # Slurm mode — respect the allocation (CUDA_VISIBLE_DEVICES set by Slurm)
    echo "[INFO] Running under Slurm → using GPU $CUDA_VISIBLE_DEVICES"
fi

# --- Host Path Definitions ---
# Note: MIMIC_ROOT is mounted Read-Only to ensure dataset integrity
OUTPUT_DIR="$PHYS_DIR/results"
METADATA_DIR="$PHYS_DIR"
HF_CACHE_DIR="$PHYS_DIR/hf_cache"

# --- CSV Column Mapping ---
METADATA_FILENAME="gemex_VQA_mimic_mapped.csv"
PATH_COLUMN="image_path"
BOX_COLUMN="visual_locations"
TEXT_COLUMN="question"
VIS_REGIONS_COL="visual_regions"
FALLBACK_PROMPT="medical abnormality"

# --- Hardware & Performance ---
# Larger batch sizes may introduce training instability.
BATCH_SIZE=8               # Reduced from 16 to fix OOM error
NUM_WORKERS=4               # Number of parallel data loading workers (CPU threads)

# --- Debugging & Limits ---
WANDB_MODE="disabled"       # Disable WandB logging for speed
# Set to an integer (e.g., "100") to limit processing to the first N images.
# Leave empty ("") to process the entire dataset.
STOP_AFTER=""

# --- Inference Mode ---
MODE="inference"            # Options: 'inference' or 'gold'
OUTPUT_FORMAT="image"       # 'image' for visual debug, 'jsonl' for production

# --- CRITICAL: Question-Driven Multi-Box Configuration ---
USE_DYNAMIC_PROMPTS="true"  # USE 'question' column for prompts
USE_VISUAL_REGIONS="false"  # IGNORE anatomical region lists (heart, lungs...)
COMPOSITE_REGIONS="false"   # Set to "true" to create a single box encompassing all visual regions in the list
EXPLODE_REGIONS="false"     # Set to "true" to create separate boxes for each visual region in the list
INCLUDE_CONTEXT_IN_INFERENCE="false"    # If "true", appends the prompt (Question or Fallback) to the region list during inference.

# --- Adaptive Padding Strategy ---
# Dynamic expansion of bounding boxes to include context.
# Small boxes receive PAD_MAX, large boxes receive PAD_MIN.
PAD_MAX=0.25  # +25% for very small boxes (e.g., nodules)
PAD_MIN=0.02  # +2% for very large boxes (e.g., cardiomegaly)
ENABLE_SMART_PADDING="true"  # Set to "false" to allow boxes on black background

# --- Multi-Box Strategy  ---
ENABLE_MULTI_BOXES="true"   # Set to "true" to detect multiple regions, "false" for single largest.
MIN_BOX_AREA="0.005"        # 0.5% of image size. Ignores tiny speckles/noise.

# --- Thresholds & Tuning ---
# IMPORTANT: 0.20 is too low for generic questions (creates blob).
# 0.45 forces model to select only high-confidence regions.
CAM_THRESHOLD=0.45          
CAM_VERSION="gScoreCAM"     # Optimal for precise localization

# --- Additional Parameters ---
SKIP_CRF="true"             # True = Faster, "raw" but separate boxes. False = More precise.
ENABLE_BODY_MASK="true"     # Always enabled
ENABLE_ANATOMICAL_CHECK="true"
DRAW_LABELS="true"
BOX_COLOR="255,0,255"         # BGR format for OpenCV (magenta)

# --- Model Specifications ---
MODEL_NAME="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLIP_VERSION="ViT-B/16"
TOPK=300

# --- Dense CRF Hyperparameters (RESTRICTIVE MODE) ---
CRF_M=6
CRF_TAU=0.80
CRF_GAUSSIAN_SXY=15
CRF_GAUSSIAN_COMPAT=2
CRF_BILATERAL_SXY=15
CRF_BILATERAL_SRGB=5
CRF_BILATERAL_COMPAT=3

# --- Advanced Refinement for Dense CRF ---
# Set a value between 0.0 and 1.0 to enable pre-CRF Hard Cut.
# Leave empty ("") to disable it (standard behavior).
# Sweet spot: 0.4 - 0.5
# Recommended value to restrict boxes: 0.6 or 0.7
# If boxes are too small or missing, try to decrease this value.
CRF_CUTOFF="0.4"


# ==============================================================================
# 2. CONFIGURATION INJECTION (OVERRIDE LOGIC)
# ==============================================================================

CLI_ARG="${1:-}"
DEFAULT_CONFIG_REF="configs/mimic_ext/hard_coded.conf" # Hardcoded fallback for quick runs

if [ -n "$CLI_ARG" ]; then
    CONFIG_FILE="$CLI_ARG"
elif [ -f "$DEFAULT_CONFIG_REF" ]; then
    CONFIG_FILE="$DEFAULT_CONFIG_REF"
else
    CONFIG_FILE="configs/mimic_ext/exp_01_vqa_ext.conf"
fi

if [ -n "$CONFIG_FILE" ]; then
    if [ -f "$CONFIG_FILE" ]; then
        echo "🔵 [CONFIG] Loading experiment config from: $CONFIG_FILE"
        echo "------------------------------------------------"
        source "$CONFIG_FILE"
        echo "   -> MODE loaded: $MODE"
        echo "   -> SMART PADDING: $ENABLE_SMART_PADDING"
        echo "------------------------------------------------"
    else
        echo "❌ [ERROR] Config file not found: $CONFIG_FILE"
        exit 1
    fi
else
    echo "🟡 [CONFIG] No config file provided. Using script DEFAULTS."
fi

# OVERRIDE: Allow Orchestrator to force a dataset
if [ -n "$DATA_FILE_OVERRIDE" ]; then
    echo "🔵 [OVERRIDE] Forcing Dataset: $DATA_FILE_OVERRIDE"
    METADATA_FILENAME="$DATA_FILE_OVERRIDE"
fi

# Resolve OUTPUT_DIR to absolute path (required for Docker mount)
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$PHYS_DIR/$OUTPUT_DIR"
fi


# ==============================================================================
# 3. CONFIGURATION SAFETY CHECKS
# ==============================================================================

echo "----------------------------------------------------------------"
echo "[SAFETY CHECK] Verifying Configuration Constraints..."

# Check 1: Composite Mode without Visual Regions = Giant Box
if [ "$USE_VISUAL_REGIONS" = "false" ] && [ "$COMPOSITE_REGIONS" = "true" ]; then
    echo "❌ [CRITICAL ERROR] UNSAFE CONFIGURATION DETECTED!"
    echo "   COMPOSITE_REGIONS enabled but USE_VISUAL_REGIONS disabled."
    echo "   This causes creation of a Giant BBox covering the entire image."
    echo "   -> Fix: Set COMPOSITE_REGIONS=\"false\"."
    exit 1
fi

# Check 2: Threshold Too Low for Question-Based Inference
if [ "$USE_VISUAL_REGIONS" = "false" ] && (( $(echo "$CAM_THRESHOLD < 0.3" | bc -l) )); then
    echo "⚠️ [WARNING] CAM_THRESHOLD ($CAM_THRESHOLD) appears too low for question-based inference."
    echo "   Risk: Single blob artifact. Recommended: > 0.40"
    # Non-blocking warning with pause for visibility
    sleep 2
fi

# Check 3: Dataset Availability
if [ ! -d "/datasets/MIMIC-CXR" ]; then
    echo "❌ [CRITICAL ERROR] Dataset directory not found!"
    echo "   Path '/datasets/MIMIC-CXR' does not exist on this node ($HOSTNAME)."
    echo "   Ensure you are running on the correct node (faretra) or the mount is active."
    exit 1
fi

# Check 4: Metadata File Existence
if [ ! -f "$METADATA_DIR/$METADATA_FILENAME" ]; then
    echo "❌ [CRITICAL ERROR] Metadata file not found!"
    echo "   Looking for: $METADATA_DIR/$METADATA_FILENAME"
    exit 1
fi

# Check 5: Multi-Box Noise Filter
if [ "$ENABLE_MULTI_BOXES" = "true" ] && (( $(echo "$MIN_BOX_AREA <= 0.0" | bc -l) )); then
    echo "⚠️ [WARNING] Multi-Box is enabled but MIN_BOX_AREA is 0."
    echo "   Risk: Generating thousands of tiny noise boxes."
    echo "   Action: Setting safe default MIN_BOX_AREA=0.005"
    MIN_BOX_AREA="0.005"
fi

echo "✅ [OK] Configuration valid. Mode: Question-Driven Multi-Box."
echo "----------------------------------------------------------------"

# ==============================================================================
# 4. ENVIRONMENT SETUP
# ==============================================================================
mkdir -p "$OUTPUT_DIR"
mkdir -p "$HF_CACHE_DIR"
chmod -R 777 "$HF_CACHE_DIR" 2>/dev/null || true
chmod -R 777 "$OUTPUT_DIR" 2>/dev/null || true


# ==============================================================================
# 5. COMMAND ARGUMENT ASSEMBLY
# ==============================================================================

# Construct the array of arguments to be passed to the Python script
CMD_ARGS=(
   "--input_dir" "/datasets/MIMIC-CXR"
   "--output_dir" "/workspace/data/output"
   "--metadata_file" "/workspace/metadata/$METADATA_FILENAME"
   "--batch_size" "$BATCH_SIZE"
   "--num_workers" "$NUM_WORKERS"
   "--mode" "$MODE"
   "--output_format" "$OUTPUT_FORMAT"
   "--path_col" "$PATH_COLUMN"
   "--box_col" "$BOX_COLUMN"
   "--text_col" "$TEXT_COLUMN"
   "--pad_max" "$PAD_MAX"
   "--pad_min" "$PAD_MIN"
   "--prompt" "$FALLBACK_PROMPT"
   "--model_name" "$MODEL_NAME"
   "--box_color" "$BOX_COLOR"
   "--clip_version" "$CLIP_VERSION"
   "--cam_version" "$CAM_VERSION"
   "--cam_threshold" "$CAM_THRESHOLD"
   "--topk" "$TOPK"
   "--crf_M" "$CRF_M"
   "--crf_tau" "$CRF_TAU"
   "--crf_gaussian_sxy" "$CRF_GAUSSIAN_SXY"
   "--crf_gaussian_compat" "$CRF_GAUSSIAN_COMPAT"
   "--crf_bilateral_sxy" "$CRF_BILATERAL_SXY"
   "--crf_bilateral_srgb" "$CRF_BILATERAL_SRGB"
   "--crf_bilateral_compat" "$CRF_BILATERAL_COMPAT"
)

# Optional argument conditional addition
if [ ! -z "$STOP_AFTER" ]; then 
    CMD_ARGS+=( "--stop_after" "$STOP_AFTER" ); 
fi

if [ "$USE_DYNAMIC_PROMPTS" = "true" ]; then 
    CMD_ARGS+=( "--use_dynamic_prompts" );
fi

if [ "$ENABLE_BODY_MASK" = "true" ]; then 
    CMD_ARGS+=( "--enable_body_mask" ); 
fi

if [ "$ENABLE_ANATOMICAL_CHECK" = "true" ]; then 
    CMD_ARGS+=( "--enable_anatomical_check" ); 
fi

if [ "$DRAW_LABELS" = "true" ]; then 
    CMD_ARGS+=( "--draw_labels" ); 
fi

# Visual regions configuration
if [ "$USE_VISUAL_REGIONS" = "true" ]; then
    CMD_ARGS+=( "--use_visual_regions" )
    CMD_ARGS+=( "--visual_regions_col" "$VIS_REGIONS_COL" )

    if [ "$COMPOSITE_REGIONS" = "true" ]; then 
        CMD_ARGS+=( "--composite_regions" ); 
    fi
    if [ "$EXPLODE_REGIONS" = "true" ]; then 
        CMD_ARGS+=( "--explode_regions" ); 
    fi
    if [ "$INCLUDE_CONTEXT_IN_INFERENCE" = "true" ]; then 
        CMD_ARGS+=( "--include_context_in_inference" ); 
    fi
fi

# Multi-box and processing flags
if [ "$ENABLE_MULTI_BOXES" = "true" ]; then 
    CMD_ARGS+=( "--multi_label" ); 
fi

if [ "$SKIP_CRF" = "true" ]; then 
    CMD_ARGS+=( "--skip_crf" ); 
fi

if [ "$ENABLE_SMART_PADDING" = "true" ]; then 
    CMD_ARGS+=( "--enable_smart_padding" )
fi


# ==============================================================================
# 6. DOCKER EXECUTION
# ==============================================================================

ENV_FILE="$PHYS_DIR/.env"
DOCKER_ENV_ARGS=""
if [ -f "$ENV_FILE" ]; then DOCKER_ENV_ARGS="--env-file $ENV_FILE"; fi

echo "[INFO] Launching Docker Container..."

docker run --rm \
    --memory="30g" \
    --gpus "device=$CUDA_VISIBLE_DEVICES" \
    --shm-size=16g \
    --user "$(id -u):$(id -g)" \
    -e HOME="/tmp" \
    -e MPLCONFIGDIR="/tmp/matplotlib_cache" \
    -e WANDB_MODE="$WANDB_MODE" \
    $DOCKER_ENV_ARGS \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro" \
    -v "$OUTPUT_DIR":/workspace/data/output \
    -v "$METADATA_DIR":/workspace/metadata \
    -v "$HF_CACHE_DIR":/workspace/hf_cache \
    -v "/llms:/llms:ro" \
    -e HF_HOME="/workspace/hf_cache" \
    "$IMAGE_NAME" \
    python3 src/bbox_preprocessing.py "${CMD_ARGS[@]}"