#!/bin/bash
# ==============================================================================
# Attention Map Heatmap Generation: Docker Wrapper Script
# ==============================================================================
# Description:
#   Configures the Docker environment, mounts necessary volumes (read-only where
#   appropriate), and launches the heatmap generation pipeline.
# ==============================================================================


# ==============================================================================
# 1. DEFAULT CONFIGURATION (HARDCODED FALLBACKS)
# ==============================================================================

# --- Core Setup ---
IMAGE_NAME="heatmap_gen:3090"
PHYS_DIR=$(pwd)

# --- GPU & CUDA Settings ---
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

# --- Host Path Definitions ---
OUTPUT_DIR="$PHYS_DIR/results"
METADATA_DIR="$PHYS_DIR"
HF_CACHE_DIR="$PHYS_DIR/hf_cache"

# --- CSV Column Mapping ---
METADATA_FILENAME="gemex_VQA_mimic_mapped.csv"
PATH_COLUMN="image_path"
TEXT_COLUMN="question"
VIS_REGIONS_COL="visual_regions"
FALLBACK_PROMPT="medical abnormality"

# --- Hardware & Performance ---
BATCH_SIZE=16
NUM_WORKERS=4

# --- Debugging & Limits ---
STOP_AFTER="5"

# --- Heatmap-Specific Parameters ---
ALPHA=0.5                   # Blending factor: 0.0 = original, 1.0 = heatmap only
COLORMAP="jet"              # Options: 'jet', 'turbo', 'inferno', 'hot'
SAVE_RAW_CAM="false"        # If true, saves raw grayscale CAM alongside overlay

# --- CRF Parameters ---
SKIP_CRF="true"
CRF_CUTOFF=0.0
CRF_M=6
CRF_TAU=0.80
CRF_GAUSSIAN_SXY=15
CRF_GAUSSIAN_COMPAT=2
CRF_BILATERAL_SXY=15
CRF_BILATERAL_SRGB=5
CRF_BILATERAL_COMPAT=3

# --- Inference Configuration ---
USE_DYNAMIC_PROMPTS="true"
USE_VISUAL_REGIONS="false"
COMPOSITE_REGIONS="false"
EXPLODE_REGIONS="false"
INCLUDE_CONTEXT_IN_INFERENCE="false"

# --- Artifact Removal ---
ENABLE_BODY_MASK="true"

# --- Display ---
DRAW_LABELS="true"

# --- Model Specifications ---
MODEL_NAME="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
CLIP_VERSION="ViT-B/16"
CAM_VERSION="gScoreCAM"
TOPK=300


# ==============================================================================
# 2. CONFIGURATION INJECTION (OVERRIDE LOGIC)
# ==============================================================================

CLI_ARG="${1:-}"
DEFAULT_CONFIG_REF="configs/gemex/heatmap_default.conf"

if [ -n "$CLI_ARG" ]; then
    CONFIG_FILE="$CLI_ARG"
elif [ -f "$DEFAULT_CONFIG_REF" ]; then
    CONFIG_FILE="$DEFAULT_CONFIG_REF"
else
    CONFIG_FILE=""
fi

if [ -n "$CONFIG_FILE" ]; then
    if [ -f "$CONFIG_FILE" ]; then
        echo "🔵 [CONFIG] Loading experiment config from: $CONFIG_FILE"
        echo "------------------------------------------------"
        source "$CONFIG_FILE"
        echo "   -> ALPHA: $ALPHA"
        echo "   -> COLORMAP: $COLORMAP"
        echo "   -> BODY MASK: $ENABLE_BODY_MASK"
        echo "------------------------------------------------"
    else
        echo "❌ [ERROR] Config file not found: $CONFIG_FILE"
        exit 1
    fi
else
    echo "🟡 [CONFIG] No config file provided. Using script DEFAULTS."
fi

# CRITICAL: Enforce the correct Docker Image Name
# (Prevents config files from overriding with the old bbox image name)
IMAGE_NAME="heatmap_gen:3090"


# ==============================================================================
# 3. CONFIGURATION SAFETY CHECKS
# ==============================================================================

echo "----------------------------------------------------------------"
echo "[SAFETY CHECK] Verifying Configuration Constraints..."

# Check 1: Composite Mode without Visual Regions
if [ "$USE_VISUAL_REGIONS" = "false" ] && [ "$COMPOSITE_REGIONS" = "true" ]; then
    echo "❌ [CRITICAL ERROR] UNSAFE CONFIGURATION DETECTED!"
    echo "   COMPOSITE_REGIONS enabled but USE_VISUAL_REGIONS disabled."
    echo "   -> Fix: Set COMPOSITE_REGIONS=\"false\"."
    exit 1
fi

# Check 2: Alpha validation
if (( $(echo "$ALPHA < 0.0 || $ALPHA > 1.0" | bc -l) )); then
    echo "❌ [CRITICAL ERROR] ALPHA must be between 0.0 and 1.0. Got: $ALPHA"
    exit 1
fi

# Check 3: Dataset Availability
if [ ! -d "/datasets/MIMIC-CXR" ]; then
    echo "❌ [CRITICAL ERROR] Dataset directory not found!"
    echo "   Path '/datasets/MIMIC-CXR' does not exist on this node ($HOSTNAME)."
    exit 1
fi

# Check 4: Metadata File Existence
if [ ! -f "$METADATA_DIR/$METADATA_FILENAME" ]; then
    echo "❌ [CRITICAL ERROR] Metadata file not found!"
    echo "   Looking for: $METADATA_DIR/$METADATA_FILENAME"
    exit 1
fi

echo "✅ [OK] Configuration valid."
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

CMD_ARGS=(
   "--input_dir" "/datasets/MIMIC-CXR"
   "--output_dir" "/workspace/data/output"
   "--metadata_file" "/workspace/metadata/$METADATA_FILENAME"
   "--batch_size" "$BATCH_SIZE"
   "--num_workers" "$NUM_WORKERS"
   "--alpha" "$ALPHA"
   "--colormap" "$COLORMAP"
   "--path_col" "$PATH_COLUMN"
   "--text_col" "$TEXT_COLUMN"
   "--prompt" "$FALLBACK_PROMPT"
   "--model_name" "$MODEL_NAME"
   "--clip_version" "$CLIP_VERSION"
   "--cam_version" "$CAM_VERSION"
   "--topk" "$TOPK"
)

# Optional arguments
if [ ! -z "$STOP_AFTER" ]; then 
    CMD_ARGS+=( "--stop_after" "$STOP_AFTER" ); 
fi

if [ "$USE_DYNAMIC_PROMPTS" = "true" ]; then 
    CMD_ARGS+=( "--use_dynamic_prompts" );
fi

if [ "$ENABLE_BODY_MASK" = "true" ]; then 
    CMD_ARGS+=( "--enable_body_mask" ); 
fi

if [ "$DRAW_LABELS" = "true" ]; then 
    CMD_ARGS+=( "--draw_labels" ); 
fi

if [ "$SAVE_RAW_CAM" = "true" ]; then 
    CMD_ARGS+=( "--save_raw_cam" ); 
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

# CRF Configuration
if [ "$SKIP_CRF" = "true" ]; then
    CMD_ARGS+=( "--skip_crf" )
else
    CMD_ARGS+=( 
        "--crf_M" "$CRF_M"
        "--crf_tau" "$CRF_TAU"
        "--crf_gaussian_sxy" "$CRF_GAUSSIAN_SXY"
        "--crf_gaussian_compat" "$CRF_GAUSSIAN_COMPAT"
        "--crf_bilateral_sxy" "$CRF_BILATERAL_SXY"
        "--crf_bilateral_srgb" "$CRF_BILATERAL_SRGB"
        "--crf_bilateral_compat" "$CRF_BILATERAL_COMPAT"
        "--crf_cutoff" "$CRF_CUTOFF"
    )
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
    -e HOME="/workspace" \
    -e MPLCONFIGDIR="/tmp/matplotlib_cache" \
    $DOCKER_ENV_ARGS \
    -e PYTORCH_CUDA_ALLOC_CONF="$PYTORCH_CUDA_ALLOC_CONF" \
    -v "/datasets/MIMIC-CXR:/datasets/MIMIC-CXR:ro" \
    -v "$OUTPUT_DIR":/workspace/data/output \
    -v "$METADATA_DIR":/workspace/metadata \
    -v "$HF_CACHE_DIR":/workspace/hf_cache \
    -v "/llms:/llms:ro" \
    -e HF_HOME="/workspace/hf_cache" \
    "$IMAGE_NAME" \
    python3 src/generate_heatmaps.py "${CMD_ARGS[@]}"
