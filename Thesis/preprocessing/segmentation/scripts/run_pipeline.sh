#!/bin/bash
# ==============================================================================
# MEDICAL VISION PIPELINE ORCHESTRATOR (Localization → Segmentation)
# ==============================================================================
# Description:
#   Master controller for end-to-end medical image analysis within Docker environment.
#   Handles configuration loading and sequential execution of vision-language
#   localization followed by segmentation refinement.
#
# Usage:
#   ./run_pipeline.sh <MODE> <CONFIG_STEP1> <CONFIG_STEP2>
#
# Operational Modes:
#   1   : Execute Step 1 only (Vision-language localization / BBox generation)
#   2   : Execute Step 2 only (MedSAM segmentation refinement)
#   all : Execute full pipeline sequentially (Step 1 → Step 2)
#
# Dependencies:
#   - Step 1 config: Defines CAM thresholds, prompting strategies, VLM parameters
#   - Step 2 config: Defines MedSAM hyperparameters, augmentation policies
#   - Docker volume mounts: /datasets, /workspace/checkpoints, /workspace/metadata
# ==============================================================================

# Exit immediately on command failure
set -e

# ------------------------------------------------------------------------------
# 1. PARAMETER INITIALIZATION & VALIDATION
# ------------------------------------------------------------------------------
MODE="${1:-all}"
CONF_S1="${2:-/workspace/configs/step1/gemex/exp_01_vqa.conf}"
CONF_S2="${3:-/workspace/configs/step2/sam_exp01.conf}"
LIMIT="${4:-}"

echo "=========================================================="
echo "🔧 PIPELINE CONFIGURATION"
echo "   Execution Mode:      $MODE"
echo "   Step 1 Config:       $CONF_S1"
echo "   Step 2 Config:       $CONF_S2"
echo "=========================================================="

# ------------------------------------------------------------------------------
# 2. PATH DEFINITIONS (Container Internal Paths)
# ------------------------------------------------------------------------------
DATASET_DIR="/datasets/MIMIC-CXR"
METADATA_FILE="/workspace/metadata/gemex_VQA_mimic_mapped.csv"
if [ -n "$DATA_FILE_OVERRIDE" ]; then
    METADATA_FILE="/workspace/metadata/$DATA_FILE_OVERRIDE"
    echo "🔵 [OVERRIDE] Forcing Dataset: $METADATA_FILE"
fi
BASE_OUT="/workspace/data/results"

# Step-specific output directories
STEP1_OUT="$BASE_OUT/step1_bboxes"
STEP2_OUT="$BASE_OUT/step2_masks"

# Intermediate data exchange file (Step 1 → Step 2)
JSONL_INTERMEDIATE="$STEP1_OUT/predictions.jsonl"

# MedSAM checkpoint (Default to shared /llms if not locally overridden)
MEDSAM_CKPT="/llms/sam_vit_b_01ec64.pth"

# Ensure output directory hierarchy exists
mkdir -p "$STEP1_OUT"
mkdir -p "$STEP2_OUT"

# ==============================================================================
# FUNCTION: STEP 1 - VISION-LANGUAGE LOCALIZATION
# ==============================================================================
run_step_1() {
    echo ""
    echo "-----------------------------------------------------------"
    echo "🩻 STEP 1: Weakly-Supervised Vision-Language Localization 🩻"
    echo "-----------------------------------------------------------"
    
    # 1. CONFIGURATION LOADING
    if [ -f "$CONF_S1" ]; then
        echo "   → Loading configuration: $CONF_S1"
        source "$CONF_S1"
    else
        echo "❌ [ERROR] Step 1 configuration file not found at: $CONF_S1"
        exit 1
    fi

    # 2. DYNAMIC ARGUMENT CONSTRUCTION
    # Note: Uses default values (:-default) if config variable is undefined
    CMD_ARGS=(
        "--input_dir" "$DATASET_DIR"
        "--output_dir" "$STEP1_OUT"
        "--metadata_file" "$METADATA_FILE"
        "--mode" "inference"
        "--output_format" "${OUTPUT_FORMAT:-jsonl}"
        "--model_name" "${MODEL_NAME:-hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224}"
        "--cam_threshold" "${CAM_THRESHOLD:-0.45}"
        "--batch_size" "${BATCH_SIZE:-16}"
    )

    if [ -n "$LIMIT" ]; then
        CMD_ARGS+=("--stop_after" "$LIMIT")
    fi

    # 3. CONDITIONAL FLAG INJECTION
    # Boolean flags are added only when configuration enables them
    if [ "${ENABLE_BODY_MASK:-true}" = "true" ]; then CMD_ARGS+=("--enable_body_mask"); fi
    if [ "${ENABLE_ANATOMICAL_CHECK:-true}" = "true" ]; then CMD_ARGS+=("--enable_anatomical_check"); fi
    if [ "${USE_DYNAMIC_PROMPTS:-true}" = "true" ]; then CMD_ARGS+=("--use_dynamic_prompts"); fi
    if [ "${ENABLE_SMART_PADDING:-true}" = "true" ]; then CMD_ARGS+=("--enable_smart_padding"); fi

    # 4. PIPELINE EXECUTION
    # Assumes run_localization.py is located in src/step1_localization/
    
    # Disable immediate exit to handle potential cleanup crashes
    set +e
    python3 src/step1_localization/run_localization.py "${CMD_ARGS[@]}"
    EXIT_CODE=$?
    set -e

    # Check if critical output exists
    if [ -f "$JSONL_INTERMEDIATE" ] && [ -s "$JSONL_INTERMEDIATE" ]; then
        if [ $EXIT_CODE -eq 0 ]; then
            echo "✅ Step 1 completed successfully."
        else
            echo "⚠️ [WARNING] Step 1 finished with non-zero exit code ($EXIT_CODE), but output file exists."
            echo "   Proceeding to Step 2..."
        fi
        echo "   Output file: $JSONL_INTERMEDIATE"
    else
        echo "❌ [ERROR] Step 1 failed (Exit Code: $EXIT_CODE) and output file is missing/empty."
        echo "   Expected: $JSONL_INTERMEDIATE"
        exit $EXIT_CODE
    fi
}

# ==============================================================================
# FUNCTION: STEP 2 - MEDSAM SEGMENTATION REFINEMENT
# ==============================================================================
run_step_2() {
    echo ""
    echo "-----------------------------------------------------------"
    echo "🔬 STEP 2: MedSAM Instance Segmentation Refinement 🔬"
    echo "-----------------------------------------------------------"

    # 1. PREREQUISITE VALIDATION (INPUT)
    if [ ! -f "$JSONL_INTERMEDIATE" ]; then
        echo "❌ [ERROR] Input JSONL not found: $JSONL_INTERMEDIATE"
        echo "   Possible causes:"
        echo "   - Step 1 has not been executed (run mode 'all' or mode '1')"
        echo "   - Step 1 failed to generate predictions"
        echo "   - File path mismatch between pipeline steps"
        exit 1
    fi

    # 2. CONFIGURATION LOADING
    if [ -f "$CONF_S2" ]; then
        echo "   → Loading configuration: $CONF_S2"
        source "$CONF_S2"
    else
        echo "❌ [ERROR] Step 2 configuration file not found at: $CONF_S2"
        exit 1
    fi

    # 3. CHECKPOINT VALIDATION
    # Use value from Config if set, otherwise default
    MEDSAM_CKPT="${MEDSAM_CHECKPOINT:-$MEDSAM_CKPT}"
    
    # If it is a file path (starts with / or ./), verify existence
    if [[ "$MEDSAM_CKPT" == /* ]] || [[ "$MEDSAM_CKPT" == ./* ]]; then
        if [ ! -f "$MEDSAM_CKPT" ]; then
            echo "❌ [ERROR] MedSAM checkpoint file not found: $MEDSAM_CKPT"
            exit 1
        fi
    else
        echo "   → Model selection: 🧠 '$MEDSAM_CKPT' 🧠 (Auto-download enabled)"
    fi

    # 4. ARGUMENT BUILDER
    CMD_ARGS=(
        "--input_file" "$JSONL_INTERMEDIATE"
        "--input_root" "$DATASET_DIR"
        "--output_dir" "$STEP2_OUT"
        "--model_checkpoint" "$MEDSAM_CKPT"
        "--scenario" "${SCENARIO:-A}"
    )

    # MedSAM2 config YAML (only relevant for SAM2-based models)
    if [ -n "${MEDSAM2_MODEL_CFG:-}" ]; then
        CMD_ARGS+=("--model_cfg" "$MEDSAM2_MODEL_CFG")
    fi

    # MedSAM3 / SAM3 config and text prompt options
    if [ -n "${MEDSAM3_CFG:-}" ]; then
        CMD_ARGS+=("--sam3_cfg" "$MEDSAM3_CFG")
    fi
    if [ -n "${TEXT_PROMPT_MODE:-}" ]; then
        CMD_ARGS+=("--text_prompt_mode" "$TEXT_PROMPT_MODE")
    fi
    if [ "${SAM3_USE_BBOX:-false}" = "true" ]; then
        CMD_ARGS+=("--sam3_use_bbox")
    fi

    if [ -n "$LIMIT" ]; then
        CMD_ARGS+=("--limit" "$LIMIT")
    fi
    
    if [ "${SAVE_VISUALS:-true}" = "true" ]; then
        CMD_ARGS+=("--save_overlays")
    fi

    # 5. PIPELINE EXECUTION
    # SCENARIO is loaded from config (A, B, or C)
    python3 src/step2_segmentation/run_segmentation.py "${CMD_ARGS[@]}"

    echo "✅ Step 2 completed successfully."
    echo "   Results directory: $STEP2_OUT"
}

# ==============================================================================
# 3. PIPELINE EXECUTION CONTROLLER
# ==============================================================================

case "$MODE" in
    1)
        echo "🔧 EXECUTION MODE: Step 1 Only (Localization)"
        run_step_1
        ;;
    2)
        echo "🔧 EXECUTION MODE: Step 2 Only (Segmentation)"
        run_step_2
        ;;
    all)
        echo "🔧 EXECUTION MODE: Full Pipeline (Step 1 → Step 2)"
        run_step_1
        run_step_2
        ;;
    *)
        echo "❌ [ERROR] Invalid execution mode: $MODE"
        echo "   Valid options: '1', '2', or 'all'"
        echo "   Usage: ./run_pipeline.sh [1|2|all] [config_step1] [config_step2]"
        exit 1
        ;;
esac

echo ""
echo "=========================================================="
echo "🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY"
echo "=========================================================="